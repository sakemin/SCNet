import os
import torch
import argparse
import yaml
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Dataset
import datetime

from .SCNet import SCNet
from .utils import load_model, new_sdr, convert_audio
from .apply import apply_model
# from .wav import WavDataset
from .log import logger

class TestDataset(Dataset):
    """Dataset for test data with mixture and sources"""
    def __init__(self, test_dir, sources, samplerate=44100, channels=2):
        self.test_dir = Path(test_dir)
        self.sources = sources
        self.samplerate = samplerate
        self.channels = channels

        # Find all test samples (folders containing mixture.wav and source files)
        self.samples = []
        for item in self.test_dir.iterdir():
            if item.is_dir() and (item / "mixture.wav").exists():
                # Check if all source files exist
                has_all_sources = True
                for source in sources:
                    if not (item / f"{source}.wav").exists():
                        has_all_sources = False
                        break

                if has_all_sources:
                    self.samples.append(item)

        print(f"Found {len(self.samples)} test samples with all required sources")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load mixture
        mix_path = sample_dir / "mixture.wav"
        mix, sr = sf.read(mix_path, dtype='float32')
        if mix.ndim == 1:
            mix = mix.reshape(-1, 1)

        # Convert to target sample rate and channels if needed
        if sr != self.samplerate or mix.shape[1] != self.channels:
            mix = convert_audio(
                torch.from_numpy(mix.T),
                sr,
                self.samplerate,
                self.channels
            ).numpy().T

        # Load sources
        sources = []
        for source_name in self.sources:
            source_path = sample_dir / f"{source_name}.wav"
            source, sr = sf.read(source_path, dtype='float32')
            if source.ndim == 1:
                source = source.reshape(-1, 1)

            # Convert to target sample rate and channels if needed
            if sr != self.samplerate or source.shape[1] != self.channels:
                source = convert_audio(
                    torch.from_numpy(source.T),
                    sr,
                    self.samplerate,
                    self.channels
                ).numpy().T

            sources.append(source)

        # Stack sources
        sources = np.stack(sources, axis=0)

        # Create tensor with mixture first, then sources
        data = np.concatenate([mix[None], sources], axis=0)
        return torch.from_numpy(data.transpose(0, 2, 1))  # [1+S, C, T]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SCNet model on test set")
    parser.add_argument('--config_path', type=str, default='./conf/config_large.yaml', help='Path to configuration file')
    # parser.add_argument('--checkpoint_path', type=str, default='./result/checkpoint_large_batch_size_3_epoch_36_mixaudio.th', help='Path to model checkpoint file')
    parser.add_argument('--checkpoint_path', type=str, default='./result/checkpoint.th', help='Path to model checkpoint file')
    parser.add_argument('--test_dir', type=str, default='/opt/dlami/nvme/kof008/datasets/separation_dataset/test', help='Path to test dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--gpu', type=str, default=None, help="Specific GPUs to use (e.g., '0,1' or '1,2')")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    return parser.parse_args()

def evaluate_model(model, test_loader, device, config, output_dir=None, sample_names=None):
    """Evaluate model on test set and compute SDR metrics"""
    model.eval()

    # Initialize metrics
    all_sdrs = []
    source_sdrs = {source: [] for source in config.model.sources}

    with torch.no_grad():
        for idx, sources in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data to device
            sources = sources.to(device)

            # Extract mixture and reference sources
            mix = sources[:, 0]
            references = sources[:, 1:]

            # Apply model with overlap for better results
            estimates = apply_model(model, mix, split=True, overlap=0.5)

            # Calculate SDR
            sdr_values = new_sdr(references, estimates)  # [B, S]

            # Store metrics
            all_sdrs.append(sdr_values)

            # Save separated sources if output directory is provided
            if output_dir:
                for b in range(estimates.shape[0]):
                    # Get sample name if available, otherwise use index
                    sample_name = sample_names[idx * test_loader.batch_size + b] if sample_names else f"sample_{idx}_{b}"

                    # Create directory for this sample
                    sample_dir = os.path.join(output_dir, sample_name)
                    os.makedirs(sample_dir, exist_ok=True)

                    # Save mixture
                    mix_audio = mix[b].cpu().numpy().T
                    sf.write(os.path.join(sample_dir, "mixture.wav"), mix_audio, config.data.samplerate)

                    # Save each source
                    for s_idx, source in enumerate(config.model.sources):
                        # Save reference
                        ref_audio = references[b, s_idx].cpu().numpy().T
                        sf.write(os.path.join(sample_dir, f"{source}_reference.wav"), ref_audio, config.data.samplerate)

                        # Save estimate
                        est_audio = estimates[b, s_idx].cpu().numpy().T
                        sf.write(os.path.join(sample_dir, f"{source}_estimate.wav"), est_audio, config.data.samplerate)

    # Concatenate all metrics
    all_sdrs = torch.cat(all_sdrs, dim=0)  # [N, S]

    # Calculate average SDR for each source
    for s_idx, source in enumerate(config.model.sources):
        source_sdrs[source] = all_sdrs[:, s_idx].mean().item()

    # Calculate overall SDR
    overall_sdr = all_sdrs.mean().item()

    return overall_sdr, source_sdrs

def main():
    args = parse_args()

    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPUs: {args.gpu}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    # Load configuration
    with open(args.config_path, 'r') as file:
        config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

    # Extract checkpoint name for folder naming
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_name = checkpoint_path.stem

    # Create output directory with checkpoint name
    eval_dir = os.path.join(args.output_dir, checkpoint_name)
    os.makedirs(eval_dir, exist_ok=True)

    # Load model
    model = SCNet(**config.model)
    model = load_model(model, args.checkpoint_path)
    model.to(device)
    model.eval()

    # Create test dataset
    test_set = TestDataset(
        args.test_dir,
        config.model.sources,
        samplerate=config.data.samplerate,
        channels=config.data.channels
    )

    # Get sample names (folder names)
    sample_names = [sample_dir.name for sample_dir in test_set.samples]

    # Create data loader
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.misc.num_workers
    )

    # Evaluate model
    overall_sdr, source_sdrs = evaluate_model(
        model,
        test_loader,
        device,
        config,
        eval_dir, 
        sample_names
    )

    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results for {checkpoint_name}:")
    print(f"Overall SDR: {overall_sdr:.3f} dB")
    print("-"*50)
    print("Source-specific SDR:")
    for source, sdr in source_sdrs.items():
        print(f"  {source}: {sdr:.3f} dB")
    print("="*50)

    # Save results to file
    results_file = os.path.join(eval_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {checkpoint_name}:\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Test directory: {args.test_dir}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Overall SDR: {overall_sdr:.3f} dB\n")
        f.write("-"*50 + "\n")
        f.write("Source-specific SDR:\n")
        for source, sdr in source_sdrs.items():
            f.write(f"  {source}: {sdr:.3f} dB\n")

    # Save configuration used for evaluation
    config_file = os.path.join(eval_dir, "config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config.to_dict(), f)

    print(f"Results saved to {eval_dir}")

    # NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m scnet.evaluate --config_path ./conf/config_large.yaml --checkpoint_path ./result/checkpoint_large_batch_size_3_epoch_36_mixaudio.th

if __name__ == "__main__":
    main()