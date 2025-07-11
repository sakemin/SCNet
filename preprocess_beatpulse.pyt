#!/usr/bin/env python
"""
Preprocess BeatPulse stems into 10 instrument-category stems.

For every JSON metadata file inside
/data1/dropbox/neutune's shared workspace/beatpulse_tracks/tags/tagged
this script does the following:
 1. Locate the corresponding directory with stems under
    /data1/dropbox/neutune's shared workspace/beatpulse_tracks/tracks/<json_name>/Stems
 2. For every stem entry whose `isRejected` is False, pick the audio file
    (prefer .wav, else .mp3) and group by the first instrument's `category`.
 3. For each of the 10 predefined instrument categories, mix all grouped
    stems together (simple sample-wise sum, then average to avoid clipping).
 4. Write each mixed stem into
    /data4/sake/beatpulse_10insts/<json_name>/__10_inst_files__/<category>.wav

Only categories with at least one source stem are produced.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from torchaudio.transforms import Resample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocess_beatpulse.log", mode="a", encoding="utf-8")
    ]
)
# Ensure file handler logs only ERROR and above
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.ERROR)

# Add a separate handler that records only WARNING messages
warnings_handler = logging.FileHandler("preprocess_beatpulse_warnings.log", mode="a", encoding="utf-8")
warnings_handler.setLevel(logging.WARNING)

# Filter to log *only* warnings (exclude debug/info/error/critical)
class _OnlyWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return record.levelno == logging.WARNING

warnings_handler.addFilter(_OnlyWarningFilter())
warnings_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(warnings_handler)

# Constants ---------------------------------------------------------------
JSON_DIR = Path("/data1/dropbox/neutune's shared workspace/beatpulse_tracks/tags/tagged")
TRACKS_DIR = Path("/data1/dropbox/neutune's shared workspace/beatpulse_tracks/tracks")
OUTPUT_ROOT = Path("/data4/sake/beatpulse_10insts")

# 10 canonical instrument categories
INSTRUMENT_CATEGORIES = {
    "Percussion & Drums",
    "Strings Instruments",
    "Fretted Instruments",
    "Wind Instruments",
    "Brass Instruments",
    "Keyboard Instruments",
    "Electronic & Synthesized Instruments",
    "Vocal Elements",
    "FX",
    "Miscellaneous Instruments & Effects",
}

TARGET_SAMPLE_RATE = 44_100  # Hz

def load_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file with torchaudio and return a (channels, samples) tensor.

    If the file's sample rate differs from `target_sr`, it will be resampled.
    If the file is mono, it will be kept mono; multi-channel audio will be kept as-is.
    """
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        resampler = Resample(sr, target_sr, dtype=waveform.dtype)
        waveform = resampler(waveform)
    # Ensure stereo (2-channel) output for consistent mixing
    if waveform.shape[0] == 1:
        # Duplicate mono channel
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        # Collapse to mono by averaging, then duplicate to stereo
        waveform = waveform.mean(dim=0, keepdim=True).repeat(2, 1)
    return waveform

def mix_waveforms(waveforms: List[torch.Tensor], json_name: str, target_length: int | None = None) -> torch.Tensor:
    """Mix a list of tensors into one by padding/averaging.

    If ``target_length`` is provided, all stems are padded to that length;
    otherwise it defaults to the maximum length found in ``waveforms``.
    """
    assert waveforms, "No waveforms provided for mixing"
    lengths = [w.shape[1] for w in waveforms]
    max_len_local = max(lengths)
    if target_length is None:
        target_length = max_len_local
    else:
        # Ensure supplied target length is at least as large as any local length
        if max_len_local > target_length:
            target_length = max_len_local
    if not all(l == target_length for l in lengths):
        logging.warning("  · Found stems with different lengths in track: %s", json_name)
        for i, length in enumerate(lengths):
            if length != target_length:
                logging.warning("    - Stem %d: %d samples (target length is %d)", i, length, target_length)

    dtype = waveforms[0].dtype
    device = waveforms[0].device
    mix = torch.zeros((waveforms[0].shape[0], target_length), dtype=dtype, device=device)

    for w in waveforms:
        if w.shape[1] < target_length:
            pad = torch.zeros((w.shape[0], target_length - w.shape[1]), dtype=dtype, device=device)
            w = torch.cat([w, pad], dim=1)
        mix += w

    mix /= len(waveforms)
    return mix

def safe_stem_path(stems_dir: Path, trackname: str) -> Optional[Path]:
    """Return existing Path to the stem audio, preferring .wav over .mp3."""
    # Original trackname as stored in JSON (likely .mp3)
    original = stems_dir / trackname
    base_without_ext = original.with_suffix("").name  # remove extension only, keep other dots

    # Candidate paths
    wav_path = stems_dir / f"{base_without_ext}.wav"
    mp3_path = stems_dir / f"{base_without_ext}.mp3"

    if wav_path.exists():
        return wav_path
    if mp3_path.exists():
        return mp3_path
    if original.exists():
        return original

    # File not found
    logging.warning("    × File not found for stem '%s' in %s", trackname, stems_dir)
    return None

def process_json(json_path: Path):
    json_name = json_path.stem
    logging.info("Processing %s", json_name)

    # Skip if output already exists (indicates this track was processed before)
    dest_dir_ready = OUTPUT_ROOT / json_name / "__10_inst_files__"
    if (dest_dir_ready / "mixture.wav").exists():
        logging.info("  · Skipping %s (already processed)", json_name)
        return

    # Locate stems directory for this track
    stems_dir = TRACKS_DIR / json_name / "Stems"
    if not stems_dir.is_dir():
        logging.warning("  × Stems directory not found: %s", stems_dir)
        return

    # Load metadata
    try:
        with json_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except Exception as e:
        logging.error("  × Failed to load JSON %s: %s", json_path, e)
        return

    stems = meta.get("stems", [])
    if not stems:
        logging.warning("  × No stems found in metadata: %s", json_path)
        return

    # Group stems by instrument category
    grouped: Dict[str, List[Path]] = {cat: [] for cat in INSTRUMENT_CATEGORIES}
    all_paths: List[Path] = []
    for stem in stems:
        if stem.get("isRejected", False):
            continue  # skip rejected stems
        instruments = stem.get("instruments", [])
        if not instruments:
            logging.debug("    · No instrument info for stem %s", stem.get("trackname"))
            continue
        category = instruments[0].get("category")
        if category not in INSTRUMENT_CATEGORIES:
            logging.debug("    · Unknown category '%s' for stem %s", category, stem.get("trackname"))
            continue
        trackname = stem.get("trackname")
        if not trackname:
            continue
        path = safe_stem_path(stems_dir, trackname)
        if path is not None:
            grouped[category].append(path)
            all_paths.append(path)

    # Determine global max length across all non-rejected stems
    global_max_len = 0
    for p in all_paths:
        try:
            info = torchaudio.info(str(p))
            global_max_len = max(global_max_len, info.num_frames)
        except Exception as e:
            logging.warning("    × Failed to inspect %s: %s", p, e)

    # Process each category
    
    for category, files in grouped.items():
        if not files:
            continue  # skip empty categories
        logging.info("  · Mixing category '%s' with %d stems", category, len(files))
        waveforms = []
        for fpath in files:
            try:
                waveforms.append(load_audio(fpath))
            except Exception as e:
                logging.warning("    × Failed to load %s: %s", fpath, e)
        if not waveforms:
            continue
        mixed = mix_waveforms(waveforms, json_name, target_length=global_max_len)

        # Build destination path
        dest_dir = OUTPUT_ROOT / json_name / "__10_inst_files__"
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe_category_name = category.replace("/", "-").replace(" ", "_")
        dest_path = dest_dir / f"{safe_category_name}.wav"
        try:
            torchaudio.save(str(dest_path), mixed, TARGET_SAMPLE_RATE)
        except Exception as e:
            logging.error("    × Failed to save mixed stem %s: %s", dest_path, e)

    # Create overall mixture of all stems
    if all_paths:
        logging.info("  · Mixing overall mixture with %d stems", len(all_paths))
        mixture_waveforms = []
        for fpath in all_paths:
            try:
                mixture_waveforms.append(load_audio(fpath))
            except Exception as e:
                logging.warning("    × Failed to load %s: %s", fpath, e)
        if mixture_waveforms:
            mixed_all = mix_waveforms(mixture_waveforms, json_name, target_length=global_max_len)
            dest_dir = OUTPUT_ROOT / json_name / "__10_inst_files__"
            dest_dir.mkdir(parents=True, exist_ok=True)
            mixture_path = dest_dir / "mixture.wav"
            try:
                torchaudio.save(str(mixture_path), mixed_all, TARGET_SAMPLE_RATE)
            except Exception as e:
                logging.error("    × Failed to save mixture stem %s: %s", mixture_path, e)


def main():
    if not JSON_DIR.is_dir():
        raise RuntimeError(f"JSON directory not found: {JSON_DIR}")

    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        logging.error("No JSON files found in %s", JSON_DIR)
        return

    for jpath in json_files:
        process_json(jpath)

    logging.info("Done.")


if __name__ == "__main__":
    main()
