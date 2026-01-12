#From HT demucs https://github.com/facebookresearch/demucs/tree/release_v4?tab=readme-ov-file

from collections import OrderedDict
import hashlib
import math
import json
import os
from pathlib import Path
import tqdm

import random
import julius
import torch as th
import torchaudio as ta
from torch.nn import functional as F

from .utils import convert_audio_channels
from accelerate import Accelerator

accelerator = Accelerator()

MIXTURE = "mixture"
EXT = ".wav"

# Define possible variations for each source (for in-house dataset)
SOURCE_VARIATIONS = {
              'mixture': ['Mixed.wav', 'mixed.wav', 'mixture.wav'],
              'high': ['high.wav', 'HIGH.wav'],
              'mid': ['mid.wav', 'MID.wav'],
              'low': ['low.wav', 'LOW.wav'],
              'rhythm': ['rhythm.wav', 'Rhythm.wav', 'rhy.wav'],
              'melody': ['melody.wav', 'Melody.wav'],
              'fx': ['fx.wav', 'FX.wav'],
              'percussion': ['Percussion_&_Drums.wav', 'Percussion.wav'],
              'string': ['Strings_Instruments.wav'],
              'fretted': ['Fretted_Instruments.wav', 'Fretted.wav'],
              'vocal': ['Vocal_Elements.wav', "Vocal.wav"],
              'wind': ['Wind_Instruments.wav'],
              'brass': ['Brass_Instruments.wav'],
              'keyboard': ['Keyboard_Instruments.wav'],
              'electronic': ['Electronic_&_Synthesized_Instruments.wav'],
              'misc': ['Miscellaneous_Instruments_&_Effects.wav'],
              'synth_idiophone': ['Synth_&_Idiophone.wav'],
              'string_brass_wind': ['String_&_Brass_&_Wind.wav'],
              'bass': ['Bass.wav']
            }

def _track_metadata(track, sources, normalize=True, ext=EXT, path_name=None):
    track_length = None
    track_samplerate = None
    mean = 0
    std = 1
    source_length = {}
    source_filename = {}
    for source in [MIXTURE] + sources:
        if path_name in ["beatpulse_audio", "beatpulse_audio_1992", "beatpulse_audio_904", "pointune_audio", "pointune_10insts", "beatpulse_10insts", "mixaudio_10insts", "beatpulse_6insts", "pointune_6insts", "mixaudio_6insts", "beatpulse_tracks", "seoul_pointune_839_tracks", "mixaudio_package_data_stems"]:
            # Find matching file for the source
            found_file = None
            if source in SOURCE_VARIATIONS:
              for variant in SOURCE_VARIATIONS[source]:
                test_file = track / variant
                if os.path.exists(test_file):
                  file = test_file
                  found_file = True
                  break

            if not found_file:
              # Default to original source name if no variant found
              file = track / f"{source}{ext}"
        else:
            file = track / f"{source}{ext}"
        if os.path.exists(file):
            try:
                info = ta.info(str(file))
            except RuntimeError:
                print(file)
                raise
            length = info.num_frames
            source_length[source] = length
            source_filename[source] = file.name
            if track_length is None:
                track_length = length
                track_samplerate = info.sample_rate
            elif track_length != length:
                if length > track_length:
                    track_length = length
            elif info.sample_rate != track_samplerate:
                raise ValueError(f"Sample rate mismatch for {file}")
            if source == MIXTURE and normalize:
                try:
                    wav, _ = ta.load(str(file))
                except RuntimeError:
                    print(file)
                    raise
                wav = wav.mean(0)
                mean = wav.mean().item()
                std = wav.std().item()

    if normalize and std == 1:
        try:
            combined = None
            for src, fname in source_filename.items():
                if src == MIXTURE: continue
                w, _ = ta.load(str(track / fname))
                if combined is None: combined = w
                else:
                    l = min(combined.shape[-1], w.shape[-1])
                    combined = combined[..., :l] + w[..., :l]

            if combined is not None:
                ref = combined.mean(0)
                mean = ref.mean().item()
                std = ref.std().item()
        except: pass

    return {"length": track_length, "mean": mean, "std": std, "samplerate": track_samplerate, "source_length": source_length, "source_filename": source_filename}


def build_metadata(path, sources, normalize=True, ext=EXT):
    """
    Build the metadata for `Wavset`.

    Args:
        path (str or Path): path to dataset.
        sources (list[str]): list of sources to look for.
        normalize (bool): if True, loads full track and store normalization
            values based on the mixture file.
        ext (str): extension of audio files (default is .wav).
    """

    meta = {}
    path = Path(path)
    pendings = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as pool:
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith('.') or folders or root == path:
                continue
            name = str(root.relative_to(path))
            pendings.append((name, pool.submit(_track_metadata, root, sources, normalize, ext, path.name)))
            # meta[name] = _track_metadata(root, sources, normalize, ext)
        for name, pending in tqdm.tqdm(pendings, ncols=120):
            if pending.result()['length'] is None: # If the track is not found, skip it
                continue
            meta[name] = pending.result()
    return meta


class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT, toothless='replace', noise_inject=False, noise_inject_prob=1.0, replace_silence=False, replace_silence_prob=1.0):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.

        Args:
            root (Path or str): root folder for the dataset.
            metadata (dict): output from `build_metadata`.
            sources (list[str]): list of source names.
            segment (None or float): segment length in seconds. If `None`, returns entire tracks.
            shift (None or float): stride in seconds bewteen samples.
            normalize (bool): normalizes input audio, **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).

        samplerate and channels are converted on the fly.
        """
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.toothless = toothless
        self.noise_inject = noise_inject
        self.noise_inject_prob = noise_inject_prob
        self.replace_silence = replace_silence
        self.replace_silence_prob = replace_silence_prob
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                if not os.path.exists(file):
                    if self.toothless == 'zero':
                        wav = th.zeros(self.channels, num_frames)
                    elif self.toothless == 'replace':
                        # Try to find a random existing file for the source
                        wav = None
                        while 1:
                            # Pick a random track name from metadata
                            random_name = random.choice(list(self.metadata.keys()))
                            random_file = self.get_file(random_name, source)

                            if os.path.exists(random_file):
                                random_meta = self.metadata[random_name]
                                # Calculate random offset if segment is defined
                                if self.segment is not None:
                                    max_offset = int(random_meta['length'] - num_frames)
                                    random_offset = random.randint(0, max(0, max_offset))
                                else:
                                    random_offset = 0

                                # Load audio from random file
                                wav, _ = ta.load(str(random_file), frame_offset=random_offset, num_frames=num_frames)
                                if random_meta['samplerate'] != meta['samplerate']: # resample to the same sample rate as the target track
                                    wav = julius.resample_frac(wav, random_meta['samplerate'], meta['samplerate'])
                                wav = convert_audio_channels(wav, self.channels)

                                break

                        if wav is None:
                            # If no valid file found after max attempts, use zeros
                            print(f"No valid file found for {source} in {name}, using zeros")
                            wav = th.zeros(self.channels, num_frames)
                    else:
                        raise ValueError(f"Invalid toothless value: {self.toothless}")
                else:
                    wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                    wav = convert_audio_channels(wav, self.channels)

                if self.replace_silence:
                    if (wav.abs() > 10**(-60/20)).float().mean() < 0.3:
                        while 1:
                            random_name = random.choice(list(self.metadata.keys()))
                            random_file = self.get_file(random_name, source)
                            if os.path.exists(random_file):
                                random_meta = self.metadata[random_name]
                                if self.segment is not None:
                                    max_offset = int(random_meta['length'] - num_frames)
                                    random_offset = random.randint(0, max(0, max_offset))
                                else:
                                    random_offset = 0
                                wav, _ = ta.load(str(random_file), frame_offset=random_offset, num_frames=num_frames)
                                if random_meta['samplerate'] != meta['samplerate']: # resample to the same sample rate as the target track
                                    wav = julius.resample_frac(wav, random_meta['samplerate'], meta['samplerate'])
                                wav = convert_audio_channels(wav, self.channels)

                                if wav.count_nonzero() > 0.3 * wav.numel():
                                    break

                wavs.append(wav)

            # Determine the minimum length across all loaded sources and trim
            min_length = min(wav.shape[-1] for wav in wavs)
            wavs = [wav[..., :min_length] for wav in wavs]

            # Convert list -> tensor of shape (nb_sources, channels, time)
            example = th.stack(wavs)  # will be further processed below

            # Optionally add noise to silent regions in batch
            if self.noise_inject:
                # Compute per-source non-zero counts
                elems_per_src = example.shape[1] * example.shape[2]
                nonzero_counts = example.ne(0).sum(dim=(1, 2))
                silence_mask = nonzero_counts < 0.3 * elems_per_src  # boolean mask per source

                if silence_mask.any() and random.random() < self.noise_inject_prob:
                    # Random std ∈ [3e-5, 9e-5] for each source
                    stds = th.empty(example.size(0), dtype=example.dtype, device=example.device).uniform_(0.00003, 0.00009)
                    noise = th.randn_like(example) * stds[:, None, None]
                    example = example + noise * silence_mask[:, None, None]

            # "example" now contains the stacked, processed audio for all sources

            # julius expects (nb_sources, channels, time)

            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)

            # Normalization and padding remain unchanged below
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example


def get_wav_datasets(args):
    """Extract the wav datasets from the XP arguments."""
    if args.multi_root: # no train/valid split here, just use the entire dataset -> Need to split manually after acquiring the metadata
        if isinstance(args.wav, str):
            args.wav = [args.wav]
        trains = {}
        valids = {}
        for wav in args.wav:
            sig = hashlib.sha1(str(wav).encode()).hexdigest()[:8]
            print(f"Dataset: {wav}")
            print(f"Sig: {sig}")
            metadata_file = Path(args.metadata) / ('wav_' + sig + ".json")
            if not metadata_file.is_file() and accelerator.is_main_process:
                metadata_file.parent.mkdir(exist_ok=True, parents=True)
                data = build_metadata(Path(wav), args.sources)
                json.dump(data, open(metadata_file, "w"))
            accelerator.wait_for_everyone()

            train, valid = json.load(open(metadata_file))
            trains[wav] = train
            valids[wav] = valid
        kw_cv = {}
        train_set = MultiRootWavset(args.wav, trains, args.sources,
                        segment=args.segment, shift=args.shift,
                        samplerate=args.samplerate, channels=args.channels,
                        normalize=args.normalize, random_mix=True)
        valid_set = MultiRootWavset(args.wav, valids, [MIXTURE] + list(args.sources),
                        segment=args.segment, shift=args.shift, samplerate=args.samplerate, channels=args.channels,
                        normalize=args.normalize, random_mix=False, random_mix_deterministic=True, **kw_cv)

    else:
        sig = hashlib.sha1(str(args.wav).encode()).hexdigest()[:8]
        metadata_file = Path(args.metadata) / ('wav_' + sig + ".json")
        train_path = Path(args.wav) / "train"
        valid_path = Path(args.wav) / "valid"
        if not metadata_file.is_file() and accelerator.is_main_process:
            metadata_file.parent.mkdir(exist_ok=True, parents=True)
            train = build_metadata(train_path, args.sources)
            valid = build_metadata(valid_path, args.sources)
            json.dump([train, valid], open(metadata_file, "w"))
        accelerator.wait_for_everyone()

        train, valid = json.load(open(metadata_file))
        kw_cv = {}

        train_set = Wavset(train_path, train, args.sources,
                        segment=args.segment, shift=args.shift,
                        samplerate=args.samplerate, channels=args.channels,
                        normalize=args.normalize, toothless=args.toothless, noise_inject=args.noise_inject, noise_inject_prob=args.noise_inject_prob, replace_silence=args.replace_silence, replace_silence_prob=args.replace_silence_prob)
        valid_set = Wavset(valid_path, valid, [MIXTURE] + list(args.sources),
                        samplerate=args.samplerate, channels=args.channels,
                        normalize=args.normalize, toothless="zero", noise_inject=False, replace_silence=False, **kw_cv)
    return train_set, valid_set


class MultiRootWavset:
    def __init__(
            self,
            roots, metadatas, sources,
            segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT,
            toothless='replace', noise_inject=False, noise_inject_prob=1.0,
            replace_silence=False, replace_silence_prob=1.0,
            silence_file_name="non_silent_segments.json",
            random_mix=False,
            random_mix_deterministic=False):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.

        Args:
            roots (list[Path or str]): root folders for the dataset.
            metadatas (dict): outputs from `build_metadata`.
            sources (list[str]): list of source names.
            segment (None or float): segment length in seconds. If `None`, returns entire tracks.
            shift (None or float): stride in seconds bewteen samples.
            normalize (bool): normalizes input audio, **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).

        samplerate and channels are converted on the fly.
        """
        self.roots = [Path(root) for root in roots]
        self.metadatas = {wav: OrderedDict(metadata) for wav, metadata in metadatas.items()}
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        # Behaviour flags (mirrors Wavset)
        self.toothless = toothless
        self.noise_inject = noise_inject
        self.noise_inject_prob = noise_inject_prob
        self.replace_silence = replace_silence
        self.replace_silence_prob = replace_silence_prob

        # Random mixing control
        self.random_mix = random_mix
        self.random_mix_deterministic = random_mix_deterministic
        # Placeholder for backward compatibility (filled later; not used with on-the-fly logic)

        # Track path → root string (first occurrence). Used to resolve root later.
        self.track_to_root = {}

        # Will be populated by loading precomputed silence index files.
        self.silence_file_name = silence_file_name

        self.num_examples = {}
        self.num_examples_total = 0
        self.dataset_start_idx = {}
        self.num_examples_per_dataset = {}
        for root in self.roots:
            if accelerator.is_main_process:
                print("Scanning dataset: ", root)
            r = str(root)
            metadata = self.metadatas[r]
            self.num_examples[r] = []
            self.dataset_start_idx[r] = self.num_examples_total
            for name, meta in tqdm.tqdm(metadata.items(), desc=f"Processing {root}", ncols=120, disable=not accelerator.is_main_process):
                try:
                    track_duration = meta['length'] / meta['samplerate']
                except:
                    print(root)
                    print(name)
                    print(meta)
                    raise
                if self.segment is None or track_duration < segment:
                    examples = 1
                else:
                    examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
                self.num_examples[r].append(examples)
                self.num_examples_total += examples

                # Map track to root for quick lookup later
                if name not in self.track_to_root:
                    self.track_to_root[name] = r

            self.num_examples_per_dataset[r] = sum(self.num_examples[r])

        # -------- Precomputed silence indices not used anymore --------
        # We previously loaded non-silent segment indices from JSON files and
        # pre-populated ``self.valid_replacements``.  This logic has been
        # replaced by on-the-fly silence detection (see ``_sample_non_silent_segment``),
        # so we skip the expensive file IO.
        self.non_silent_segments_by_source = {s: {} for s in self.sources}  # keep attribute for compatibility
        self.valid_replacements = {s: [] for s in self.sources}  # no-op placeholder

        # Curriculum: upper bound for active sources in random mixing
        self.max_active_sources = len(self.sources)
        self.min_active_sources = 2
        
        # --- On-the-fly silence detection parameters ---
        # We treat any sample quieter than -60 dBFS as silence.
        self._silence_amp_threshold = 10 ** (-60.0 / 20)  # linear amplitude for –60 dB
        # A segment is considered "non-silent" if at least 30% of its samples exceed the threshold.
        self._min_non_silent_ratio = 0.3
        
        # ------------------------------------------------------------------

    def __len__(self):
        return self.num_examples_total

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _sample_non_silent_segment(self, rng, source, max_attempts: int = 30):
        """Randomly sample a (root, track, segment) triple for the given *source*
        such that the selected audio segment is considered *non-silent*.

        A segment is accepted if at least ``self._min_non_silent_ratio`` fraction
        of its samples have an absolute amplitude greater than
        ``self._silence_amp_threshold``. Returns a tuple ``(wav, meta)`` on
        success, otherwise ``(None, None)`` if no suitable segment was found
        within *max_attempts* tries.
        """

        for _ in range(max_attempts):
            rand_root = rng.choice(self.roots)
            rand_root_str = str(rand_root)
            metadata_dict = self.metadatas[rand_root_str]

            if not metadata_dict:
                continue

            rand_name = rng.choice(list(metadata_dict.keys()))
            rand_meta = metadata_dict[rand_name]

            # Skip tracks without the requested source
            if source not in rand_meta.get('source_filename', {}):
                continue

            # Build full path to the audio file
            rand_file = Path(rand_root) / rand_name / rand_meta['source_filename'][source]
            if not rand_file.is_file():
                continue

            # Determine segment parameters
            if self.segment is not None:
                # Total number of possible segments for this track
                track_dur = rand_meta['length'] / rand_meta['samplerate']
                total_segs = max(
                    1,
                    int(math.ceil((track_dur - self.segment) / self.shift) + 1),
                )
                rand_seg = rng.randint(0, total_segs - 1)
                offset = int(rand_meta['samplerate'] * self.shift * rand_seg)
                num_frames = int(math.ceil(rand_meta['samplerate'] * self.segment))
            else:
                offset = 0
                num_frames = -1

            try:
                wav, _ = ta.load(str(rand_file), frame_offset=offset, num_frames=num_frames)
            except Exception:
                continue  # corrupted file / read error – try another

            wav = convert_audio_channels(wav, self.channels)

            # Quick silence check
            if wav.numel() == 0:
                continue
            if (wav.abs() > self._silence_amp_threshold).float().mean().item() < self._min_non_silent_ratio:
                continue  # Too silent – search again

            return wav, rand_meta  # Success

        # Fallback: nothing found
        return None, None

    def get_file(self, root, name, source):
        try:
            return Path(root) / name / self.metadatas[str(root)][name]['source_filename'][source]
        except: # if there's no source filename(no source for this track), return None
            return None

    def __getitem__(self, index):
        """Return one training example, mirroring the logic of ``Wavset.__getitem__``."""

        # ---------------- Random cross-track mixing ----------------
        # Instead of taking stems from the same track, we randomly sample
        # individual segments for each source from the global non-silent pool.
        # This behaviour creates mixtures containing 2-10 active instruments.

        if self.random_mix:
            if self.random_mix_deterministic:
                rng = random.Random(index)
            else:
                rng = random
            # Decide which sources are active between min_active_sources and max_active_sources
            # upper = max(self.min_active_sources, min(self.max_active_sources, len(self.sources)))
            # n_active = rng.randint(self.min_active_sources, upper)
            # active_sources = rng.sample(self.sources, n_active)
            active_sources = self.sources
            num_frames = None
            wavs = []

            for source in self.sources:
                if source in active_sources:
                    wav, rand_meta = self._sample_non_silent_segment(rng, source)

                    # If we failed to find a suitable segment, fallback to zeros
                    if wav is None:
                        if num_frames is None:
                            if self.segment is not None:
                                num_frames = int(math.ceil(self.samplerate * self.segment))
                            else:
                                num_frames = 1
                        wavs.append(th.zeros(self.channels, num_frames))
                        continue

                    # Update num_frames lazily (first successful load)
                    if num_frames is None:
                        num_frames = wav.shape[-1]

                    # Resample if needed
                    if rand_meta['samplerate'] != self.samplerate:
                        wav = julius.resample_frac(wav, rand_meta['samplerate'], self.samplerate)

                    wav = convert_audio_channels(wav, self.channels)

                    # Normalise for better source balance
                    if self.normalize:
                        src_mean = rand_meta.get('mean', 0.0)
                        src_std = rand_meta.get('std', 1.0)
                        denom = src_std if src_std > 1e-6 else 1.0
                        wav = (wav - src_mean) / denom

                    wavs.append(wav)
                else:
                    # Inactive source → zeros (masked in loss)
                    if num_frames is None:
                        if self.segment is not None:
                            num_frames = int(math.ceil(self.samplerate * self.segment))
                        else:
                            num_frames = 1
                    wavs.append(th.zeros(self.channels, num_frames))

            # Ensure common length
            min_length = min(w.shape[-1] for w in wavs)
            wavs = [w[..., :min_length] for w in wavs]

            example = th.stack(wavs)

            # Final padding if shorter than desired segment length
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = F.pad(example, (0, length - example.shape[-1]))

            return example

        # ---------------- Original same-track logic (fallback) ----------------
        # Locate which dataset/root the global index falls into.
        target_root = None
        for root in self.roots:
            r = str(root)
            if self.dataset_start_idx[r] <= index < self.dataset_start_idx[r] + self.num_examples_per_dataset[r]:
                target_root = r
                relative_index = index - self.dataset_start_idx[r]
                break

        metadata = self.metadatas[target_root]

        # Identify the exact track within the chosen dataset.
        for name, examples in zip(metadata, self.num_examples[target_root]):
            if relative_index >= examples:
                relative_index -= examples
                continue

            meta = metadata[name]

            # Compute offset/length for the requested segment (if any)
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * relative_index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))

            wavs = []
            for source in self.sources:
                file = self.get_file(target_root, name, source)

                # -------------------- Handle missing stems -------------------
                if file is None or not os.path.exists(file):
                    if self.toothless == 'zero':
                        wav = th.zeros(self.channels, num_frames)
                    elif self.toothless == 'replace':
                        wav_rep, rep_meta = self._sample_non_silent_segment(random, source)
                        if wav_rep is not None:
                            # Match stats of replacement to target track
                            if rep_meta['samplerate'] != meta['samplerate']:
                                wav_rep = julius.resample_frac(wav_rep, rep_meta['samplerate'], meta['samplerate'])
                            wav_rep = convert_audio_channels(wav_rep, self.channels)

                            if self.normalize:
                                src_mean = rep_meta.get('mean', 0.0)
                                src_std = rep_meta.get('std', 1.0)
                                target_mean, target_std = meta['mean'], meta['std']
                                denom = src_std if src_std > 1e-6 else 1.0
                                wav_rep = (wav_rep - src_mean) * (target_std / denom) + target_mean
                            wav = wav_rep
                        else:  # still nothing found → zeros
                            wav = th.zeros(self.channels, num_frames)
                    else:
                        raise ValueError(f"Invalid toothless value: {self.toothless}")
                else:
                    wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                    wav = convert_audio_channels(wav, self.channels)

                # ---------------- Optional silence replacement ---------------
                if self.replace_silence:
                    if (wav.abs() > 10**(-60/20)).float().mean() < 0.3 and random.random() < self.replace_silence_prob:
                        for _ in range(20):
                            wav_rep, rep_meta = self._sample_non_silent_segment(random, source)
                            if wav_rep is None:
                                continue
                            # Resample to match target track samplerate
                            if rep_meta['samplerate'] != meta['samplerate']:
                                wav_rep = julius.resample_frac(wav_rep, rep_meta['samplerate'], meta['samplerate'])
                            wav_rep = convert_audio_channels(wav_rep, self.channels)

                            # Volume/statistics matching
                            if self.normalize:
                                target_mean, target_std = meta['mean'], meta['std']
                                src_mean, src_std = rep_meta.get('mean', 0.0), rep_meta.get('std', 1.0)
                                if src_std > 0:
                                    wav_rep = (wav_rep - src_mean) * (target_std / src_std) + target_mean

                            if (wav_rep.abs() > 10**(-60/20)).float().mean() > 0.3:
                                wav = wav_rep
                                break

                wavs.append(wav)

            # Ensure all stems share the same length (min across stems).
            if not wavs:
                min_length = 0
            else:
                min_length = min(wav.shape[-1] for wav in wavs)

            if min_length == 0:
                # If loading failed or empty files, return zeros of target length
                fallback_frames = int(self.samplerate * (self.segment or 1.0))
                return th.zeros(len(self.sources), self.channels, fallback_frames)

            wavs = [wav[..., :min_length] for wav in wavs]

            example = th.stack(wavs)  # (nb_sources, channels, time)

            # --------------------- Noise injection -------------------------
            if self.noise_inject:
                elems_per_src = example.shape[1] * example.shape[2]
                nonzero_counts = example.ne(0).sum(dim=(1, 2))
                silence_mask = nonzero_counts < 0.3 * elems_per_src
                if silence_mask.any() and random.random() < self.noise_inject_prob:
                    stds = th.empty(example.size(0), dtype=example.dtype, device=example.device).uniform_(0.00003, 0.00009)
                    noise = th.randn_like(example) * stds[:, None, None]
                    example = example + noise * silence_mask[:, None, None]

            # ---------------- Resampling / Normalisation -------------------
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)

            if self.normalize:
                denom = meta['std'] if meta.get('std',1.0) > 1e-6 else 1.0
                example = (example - meta['mean']) / denom
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example
        