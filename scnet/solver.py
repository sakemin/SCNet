import torch
from pathlib import Path
from .utils import copy_state, EMA, new_sdr
from .apply import apply_model
from .ema import ModelEMA
from . import augment
from .loss import spec_rmse_loss
from tqdm import tqdm
from .log import logger
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import soundfile as sf
import torchaudio
import tempfile
from pydub import AudioSegment
import torch.distributed as dist
from .wav import SOURCE_VARIATIONS  # Variants of source filenames

def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())

def _convert_to_mp3(wav_path, bitrate="192k"):
    """Convert a wav file to mp3 and return the temporary mp3 path."""
    temp_dir = tempfile.gettempdir()
    mp3_filename = os.path.basename(wav_path).replace('.wav', '.mp3')
    mp3_path = os.path.join(temp_dir, mp3_filename)
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate=bitrate)
    return mp3_path

class Solver(object):
    def __init__(self, loaders, model, optimizer, config, args):
        self.config = config
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.device = next(iter(self.model.parameters())).device
        self.accelerator = Accelerator()
        self.scaler = GradScaler()

        # Initialize global step before potential checkpoint loading
        self.global_step = 0

        self.stft_config = {
            'n_fft': config.model.nfft,
            'hop_length': config.model.hop_size,
            'win_length': config.model.win_size,
            'center': True,
            'normalized': config.model.normalized
        }
        # Exponential moving average of the model
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(config.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # data augment
        augments = [augment.Shift(shift=int(config.data.samplerate * config.data.shift),
                                  same=config.augment.shift_same)]
        if config.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(config.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        if self.accelerator.is_main_process:
            
            # Convert ConfigDict to a plain Python dict for cleaner logging on WandB
            wandb_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
            wandb_init_kwargs = dict(
                entity='sakemin',
                project='scnet' if not config.data.multi_root else 'scnet-10insts',
                config=wandb_config
            )

            # If resuming from an existing W&B run, extract run id and enable resume mode
            if hasattr(args, 'wandb_path') and args.wandb_path:
                run_path = Path(args.wandb_path)
                # Expected directory name like 'run-YYYYMMDD_HHMMSS-<id>' → take the last dash-separated token as id
                run_id = run_path.name.split('-')[-1]
                # W&B always creates a 'wandb' subfolder inside the directory provided via `dir`.
                # If we pass run_path.parent (which is already '<root>/wandb'), we end up with
                # '<root>/wandb/wandb'. Instead, pass the *root* directory that contains the first
                # 'wandb' folder so paths align with the existing run directory.
                base_dir = run_path.parent
                if base_dir.name == 'wandb':
                    base_dir = base_dir.parent  # strip the extra 'wandb'
                wandb_init_kwargs.update({
                    'id': run_id,
                    'resume': 'must',
                    'dir': str(base_dir)
                })

            self.run = wandb.init(**wandb_init_kwargs)
            # wandb.config.update(config)

        # Broadcast the checkpoint directory path from the main process to all others
        if self.accelerator.is_main_process:
            run_dir = self.run.dir
        else:
            run_dir = None

        # Broadcast run_dir from rank 0 to all other ranks using torch.distributed
        if self.accelerator.state.distributed_type != DistributedType.NO and dist.is_initialized():
            obj_list = [run_dir]
            dist.broadcast_object_list(obj_list, src=0)
            run_dir = obj_list[0]

        self.checkpoint_file = Path(run_dir) / 'checkpoints' / 'checkpoint.th'

        # Determine checkpoint to resume from if user provided a WandB run path
        self.resume_checkpoint_file = None
        if hasattr(args, 'wandb_path') and args.wandb_path:
            if self.accelerator.is_main_process:
                candidate_dir = Path(args.wandb_path)
                # Prefer files/checkpoints → checkpoints → provided directory
                if (candidate_dir / 'files' / 'checkpoints').exists():
                    search_dir = candidate_dir / 'files' / 'checkpoints'
                elif (candidate_dir / 'checkpoints').exists():
                    search_dir = candidate_dir / 'checkpoints'
                else:
                    search_dir = candidate_dir
                self.resume_checkpoint_file = self._find_latest_checkpoint(search_dir)
            # Sync the selected checkpoint path across all processes
            if self.accelerator.state.distributed_type != DistributedType.NO and dist.is_initialized():
                obj_list = [str(self.resume_checkpoint_file) if self.resume_checkpoint_file else ""]
                dist.broadcast_object_list(obj_list, src=0)
                self.resume_checkpoint_file = Path(obj_list[0]) if obj_list[0] else None

        # Checkpoint states
        self.best_state = None
        self.best_nsdr = 0
        self.epoch = -1

        if self.accelerator.is_main_process:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Call _reset on all processes so attributes like global_step exist everywhere
        self._reset()
        self.mp3_temp_files = []  # keep track of temporary mp3 files to delete later
        # Audio logging configuration (optional)
        self.audio_log = False
        if hasattr(config, 'audio_log') and getattr(config.audio_log, 'enable', False):
            self.audio_log = True
            self.audio_log_samples = getattr(config.audio_log, 'samples', [])  # list[str] of track folder names
            self.audio_log_root = getattr(config.audio_log, 'root', getattr(config.data, 'wav', None) if hasattr(config, 'data') else None)
            self.audio_log_segment = getattr(config.audio_log, 'segment', 11.0)  # seconds
            self.audio_log_start = getattr(config.audio_log, 'start', 0.0)  # seconds
            self.audio_log_use_mp3 = getattr(config.audio_log, 'use_mp3', True)
            self.audio_log_mp3_bitrate = getattr(config.audio_log, 'mp3_bitrate', '192k')

    def _serialize(self, epoch, steps=0):
        package = {}
        package['state'] = self.model.state_dict()
        package['best_nsdr'] = self.best_nsdr
        package['best_state'] = self.best_state
        package['optimizer'] = self.optimizer.state_dict()
        package['epoch'] = epoch
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        if steps: 
            checkpoint_with_steps = Path(self.checkpoint_file).with_name(f'checkpoint_{epoch+1}_{steps}.th')
            self.accelerator.save(package, checkpoint_with_steps)
        else:
            self.accelerator.save(package, self.checkpoint_file)

    def _find_latest_checkpoint(self, ckpt_dir):
        """Return the most recently modified checkpoint file (.th) inside `ckpt_dir`, or None."""
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            return None
        candidates = sorted(ckpt_dir.glob('checkpoint*.th'), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _parse_epoch_step_from_filename(self, path):
        """Extract (epoch, step) from filenames like 'checkpoint_E_S.th'. Returns (epoch_zero_based, step_zero_based) or (None, None)."""
        name = Path(path).stem  # Strip directory and extension
        parts = name.split('_')
        if len(parts) >= 3 and parts[0] == 'checkpoint':
            try:
                epoch_1b = int(parts[1])  # 1-based epoch stored in filename
                step_1b = int(parts[2])   # 1-based global step stored in filename
                return epoch_1b - 1, step_1b - 1  # convert to 0-based
            except ValueError:
                pass
        return None, None

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        ckpt_path = getattr(self, 'resume_checkpoint_file', None) or self.checkpoint_file
        if ckpt_path.exists():
            logger.info(f'Loading checkpoint model: {ckpt_path}')
            package = torch.load(ckpt_path, map_location=self.accelerator.device)
            self.model.load_state_dict(package['state'])
            self.best_nsdr = package['best_nsdr']
            self.best_state = package['best_state']
            self.optimizer.load_state_dict(package['optimizer'])
            self.epoch = package['epoch'] - 1
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])
            # Derive global step from filename, fallback to epoch-based estimate
            ep, st = self._parse_epoch_step_from_filename(ckpt_path)
            if st is not None:
                self.global_step = st + 1  # resume *after* the saved step
            else:
                # Assume completed whole epochs up to self.epoch
                self.global_step = (self.epoch + 1) * len(self.loaders['train'])
        else:
            # Fresh start
            self.epoch = -1
            self.global_step = 0
        # Ensure global_step is defined even if _reset didn't set it (e.g., fresh run)
        if not hasattr(self, 'global_step'):
            self.global_step = 0

    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.config.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        # Optimizing the model
        for epoch in range(self.epoch + 1, self.config.epochs):
            #Adjust learning rate
            for param_group in self.optimizer.param_groups:
              param_group['lr'] = self.config.optim.lr * (self.config.optim.decay_rate**((epoch)//self.config.optim.decay_step))
              logger.info(f"Learning rate adjusted to {self.optimizer.param_groups[0]['lr']}")
            
            # Log learning rate to wandb
            if self.accelerator.is_main_process:
                wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']}, step=epoch)

            # Train one epoch
            self.model.train()
            metrics = {}
            logger.info('-' * 70)
            logger.info(f'Training Epoch {epoch + 1} ...')


            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}')


            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid['nsdr']
                        b = bvalid['nsdr']
                        if a > b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname



            formatted = self._format_train(metrics['valid'])
            logger.info(
                f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}')
                
            # Log validation metrics (numeric only) averaged across GPUs
            numeric_valid = {k: v for k, v in metrics['valid'].items() if isinstance(v, (int, float, torch.Tensor))}
            self._log_wandb(numeric_valid, prefix='valid/', step=self.global_step)

            # Also log the epoch number at the same step
            self._log_wandb({"epoch": epoch + 1})

            valid_nsdr = metrics['valid']['nsdr']

            # Save the best model
            if valid_nsdr > self.best_nsdr:
              logger.info('New best valid nsdr %.4f', valid_nsdr)
              self.best_state = copy_state(state)
              self.best_nsdr = valid_nsdr

            if self.accelerator.is_main_process:
                self._serialize(epoch)
            if epoch == self.config.epochs - 1:
                break


    def _run_one_epoch(self, epoch, train=True):
        config = self.config
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        data_loader.sampler.epoch = epoch

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)

        averager = EMA()

        if self.accelerator.is_main_process:
            data_loader = tqdm(data_loader)

        for idx, sources in enumerate(data_loader):
            global_step = self.global_step
            sources = sources.to(self.device)
            if train:
                sources = self.augment(sources)
                mix = sources.sum(dim=1)
            else:
                mix = sources[:, 0]
                sources = sources[:, 1:]

            if not train:
                estimate = apply_model(self.model, mix, split=True, overlap=0)
            else:
                with autocast():
                   estimate = self.model(mix)

            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)

            loss = spec_rmse_loss(estimate, sources, self.stft_config)

            losses = {}

            losses['loss'] = loss
            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                nsdrs = self.accelerator.reduce(nsdrs, reduction="mean")
                total = 0
                for source, nsdr in zip(self.config.model.sources, nsdrs):
                    losses[f'nsdr_{source}'] = nsdr
                    total += nsdr
                losses['nsdr'] = total / len(self.config.model.sources)

            # optimize model in training mode
            if train:
                scaled_loss = self.scaler.scale(loss)
                self.accelerator.backward(scaled_loss)

                # Unscale the gradients and apply gradient clipping
                self.scaler.unscale_(self.optimizer)  # required before clipping when using mixed precision
                max_norm = getattr(self.config.optim, 'max_grad_norm', 1.0)
                if max_norm is not None and max_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm)

                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                for ema in self.emas['batch']:
                    ema.update()
                if self.config.save_every and (global_step+1) % self.config.save_every == 0:
                    self._serialize(epoch, steps = global_step+1)
            
            if train and (global_step+1) % self.config.log_every == 0:
                self._log_wandb(losses, step=global_step, prefix="train_nonavg/")

            losses = averager(losses)

            if train and (global_step+1) % self.config.log_every == 0:
                self._log_wandb(losses, step=global_step, prefix="train/")
                formatted = self._format_train(losses)
                logger.info(
                    f'Train Summary | Epoch {epoch + 1} | Step {idx+1} | Global Step {global_step+1} | {_summary(formatted)}')

            del loss, estimate

            if (global_step+1) % self.config.val_every == 0 and train:
                self.model.eval()  # Turn off Batchnorm & Dropout
                metrics = {}
                with torch.no_grad():
                    valid = self._run_one_epoch(epoch, train=False)
                    bvalid = valid
                    bname = 'main'
                    state = copy_state(self.model.state_dict())
                    metrics['valid'] = {}
                    metrics['valid']['main'] = valid
                    for kind, emas in self.emas.items():
                        for k, ema in enumerate(emas):
                            with ema.swap():
                                valid = self._run_one_epoch(epoch, train=False)
                            name = f'ema_{kind}_{k}'
                            metrics['valid'][name] = valid
                            a = valid['nsdr']
                            b = bvalid['nsdr']
                            if a > b:
                                bvalid = valid
                                state = ema.state
                                bname = name
                        metrics['valid'].update(bvalid)
                        metrics['valid']['bname'] = bname

                formatted = self._format_train(metrics['valid'])
                logger.info(
                    f'Valid Summary | Epoch {epoch + 1} | Step {idx+1} | Global Step {global_step+1} | {_summary(formatted)}')
                self._log_wandb(metrics['valid']['main'], step=global_step, prefix="valid/")
                
                if self.audio_log:
                    # Log fixed audio samples
                    self._log_audio_samples(global_step)
                self.model.train()
 
            # Advance global step counter after each training batch
            if train:
                self.global_step += 1
        if train:
            for ema in self.emas['epoch']:
                ema.update()
        return losses

    def _log_audio_samples(self, step):
        """Run separation on fixed audio segments and log to WandB."""
        if not self.accelerator.is_main_process:
            return
        if not self.audio_log_samples or self.audio_log_root is None:
            return
        for sample_name in self.audio_log_samples:
            track_dir = Path(self.audio_log_root) / sample_name
            mixture_path = track_dir / 'mixture.wav'
            if not mixture_path.exists():
                continue

            # Load mixture
            mix, sr = torchaudio.load(str(mixture_path))  # shape (C, T)
            
            abs_max = mix.abs().max()
            # Normalize mixture
            mix = mix / (abs_max + 1e-8)

            start_sample = int(self.audio_log_start * sr)
            end_sample = start_sample + int(self.audio_log_segment * sr)
            mix_seg = mix[:, start_sample:end_sample]

            mix_tensor = mix_seg.unsqueeze(0).to(self.device)
            with torch.no_grad():
                est = apply_model(self.model, mix_tensor, split=False, overlap=0)
            est = est[0].cpu()  # (sources, C, T)

            if step + 1 == self.config.log_every:
                # Save mixture to temporary wav before optional mp3 conversion
                tmp_mix_wav = tempfile.mktemp(suffix='.wav')
                sf.write(tmp_mix_wav, mix_seg.T.numpy(), sr)
                if self.audio_log_use_mp3:
                    mix_path = _convert_to_mp3(tmp_mix_wav, self.audio_log_mp3_bitrate)
                    self.mp3_temp_files.append(mix_path)
                    os.remove(tmp_mix_wav)
                else:
                    mix_path = tmp_mix_wav
                wandb.log({f"audio_eval/{sample_name}/mixture": wandb.Audio(mix_path, caption="Mixture", sample_rate=sr)}, step=step)
                if not self.audio_log_use_mp3:
                    self.mp3_temp_files.append(mix_path)
            # Iterate over sources
            for src_idx, source in enumerate(self.config.model.sources):
                # Resolve ground-truth file, considering filename variations
                gt_path = None
                for fname in [f"{source}.wav"] + SOURCE_VARIATIONS.get(source, []):
                    candidate = track_dir / fname
                    if candidate.exists():
                        gt_path = candidate
                        break
                gt_audio = None
                if gt_path is not None:
                    gt, _ = torchaudio.load(str(gt_path))
                    gt = gt / (abs_max + 1e-8) # normalize ground truth with mixture abs_max
                    gt_audio = gt[:, start_sample:end_sample].numpy()
                pred_audio = est[src_idx].numpy()
                # Prepare WandB Audio objects (optionally convert to mp3)
                entries = []
                for label, audio_arr in [("Ground Truth", gt_audio), ("Prediction", pred_audio)]:
                    if audio_arr is None:
                        entries.append(None)
                        continue
                    tmp_wav = tempfile.mktemp(suffix='.wav')
                    sf.write(tmp_wav, audio_arr.T, sr)
                    if self.audio_log_use_mp3:
                        path = _convert_to_mp3(tmp_wav, self.audio_log_mp3_bitrate)
                        os.remove(tmp_wav)
                        self.mp3_temp_files.append(path)
                    else:
                        path = tmp_wav
                        self.mp3_temp_files.append(path)
                    entries.append(wandb.Audio(path, caption=f"{label} {source}", sample_rate=sr))
                if any(a is not None for a in entries):
                    table = wandb.Table(columns=["Ground Truth", "Prediction"])
                    table.add_data(*entries)
                    wandb.log({f"audio_eval/{sample_name}/{source}": table}, step=step)

    def _log_wandb(self, metrics: dict, *, step=None, prefix=""):
        """Average every tensor/float in `metrics` across GPUs and log once."""
        reduced = {}
        for key, val in metrics.items():
          # Accept tensors directly or convert numeric types; skip non-numeric entries (e.g., strings)
          if torch.is_tensor(val):
            # Ensure tensor is in floating dtype for safe division during reduction
            val_tensor = val.float() if not val.is_floating_point() else val
          elif isinstance(val, (int, float)):
            val_tensor = torch.tensor(float(val), device=self.device)
          else:
            # Ignore values that cannot be converted to tensors (such as strings)
            continue
          val_mean = self.accelerator.reduce(val_tensor.detach(), reduction="mean").item()
          reduced[f"{prefix}{key}"] = val_mean
        # Only log if there is at least one numeric metric to report
        if self.accelerator.is_main_process and reduced:
          wandb.log(reduced, step=step)

    def __del__(self):
        """Clean up temporary mp3 files created during logging."""
        for mp3_path in self.mp3_temp_files:
            try:
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {mp3_path}: {e}")
