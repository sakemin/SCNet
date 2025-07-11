#!/usr/bin/env python
"""
Preprocess Pointune stems into 10 instrument-category stems.

For every JSON metadata file inside
/data1/dropbox/neutune's shared workspace/pointune_tracks/tags/tagged
this script does the following:
 1. Locate the corresponding directory with stems under
    /data1/dropbox/neutune's shared workspace/pointune_tracks/tracks/seoul_pointune_839_tracks/<json_name>/Stems
 2. For every stem entry whose `isRejected` is False, pick the audio file
    (prefer .wav, else .mp3) and group by the first instrument's `category`.
 3. For each of the 10 predefined instrument categories, mix all grouped
    stems together (simple sample-wise sum, then average to avoid clipping).
 4. Additionally create an overall `mixture.wav` with all non-rejected stems.
 5. Write each mixed stem into
    /data4/sake/pointune_10insts/<json_name>/__10_inst_files__/<category or mixture>.wav

Only categories with at least one source stem are produced.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from torchaudio.transforms import Resample

# -------------------------------------------------------------------------
# Logging configuration: console INFO, file ERROR, warnings file WARNING
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocess_pointune.log", mode="a", encoding="utf-8"),
    ],
)
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.ERROR)

warnings_handler = logging.FileHandler("preprocess_pointune_warnings.log", mode="a", encoding="utf-8")
warnings_handler.setLevel(logging.WARNING)

class _OnlyWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return record.levelno == logging.WARNING

warnings_handler.addFilter(_OnlyWarningFilter())
warnings_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(warnings_handler)

# -------------------------------------------------------------------------
# Dataset-specific paths
JSON_DIR = Path("/data1/dropbox/neutune's shared workspace/pointune_tracks/tags/tagged")
TRACKS_DIR = Path("/data1/dropbox/neutune's shared workspace/pointune_tracks/tracks/seoul_pointune_839_tracks")
OUTPUT_ROOT = Path("/data4/sake/pointune_10insts")

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

# -------------------------------------------------------------------------
# Helper functions

def load_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Load audio file, resample, convert to stereo."""
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        waveform = Resample(sr, target_sr, dtype=waveform.dtype)(waveform)
    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform.mean(dim=0, keepdim=True).repeat(2, 1)
    return waveform

def mix_waveforms(waveforms: List[torch.Tensor], json_name: str, target_length: int | None = None) -> torch.Tensor:
    if not waveforms:
        raise ValueError("No waveforms to mix")
    lengths = [w.shape[1] for w in waveforms]
    local_max = max(lengths)
    if target_length is None or local_max > target_length:
        target_length = local_max
    if any(l != target_length for l in lengths):
        logging.warning("  · Found stems with different lengths in track: %s", json_name)
        for i, l in enumerate(lengths):
            if l != target_length:
                logging.warning("    - Stem %d: %d samples (target %d)", i, l, target_length)
    mix = torch.zeros((2, target_length), dtype=waveforms[0].dtype, device=waveforms[0].device)
    for w in waveforms:
        if w.shape[1] < target_length:
            w = torch.cat([w, torch.zeros((2, target_length - w.shape[1]), dtype=w.dtype, device=w.device)], dim=1)
        mix += w
    mix /= len(waveforms)
    return mix

def safe_stem_path(stems_dir: Path, trackname: str) -> Optional[Path]:
    """Find stem file preferring .wav over .mp3."""
    original = stems_dir / trackname
    base = original.with_suffix("").name
    wav_path = stems_dir / f"{base}.wav"
    mp3_path = stems_dir / f"{base}.mp3"
    if wav_path.exists():
        return wav_path
    if mp3_path.exists():
        return mp3_path
    if original.exists():
        return original
    logging.warning("    × File not found for stem '%s' in %s", trackname, stems_dir)
    return None

# -------------------------------------------------------------------------
# Core processing

def process_json(json_path: Path):
    json_name = json_path.stem
    logging.info("Processing %s", json_name)

    dest_dir_ready = OUTPUT_ROOT / json_name / "__10_inst_files__"
    if (dest_dir_ready / "mixture.wav").exists():
        logging.info("  · Skipping %s (already processed)", json_name)
        return

    stems_dir = TRACKS_DIR / json_name / "Stems"
    if not stems_dir.is_dir():
        logging.warning("  × Stems directory not found: %s", stems_dir)
        return

    try:
        with json_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except Exception as e:
        logging.error("  × Failed to load JSON %s: %s", json_path, e)
        return

    stems = meta.get("stems", [])
    if not stems:
        logging.warning("  × No stems listed in metadata: %s", json_path)
        return

    grouped: Dict[str, List[Path]] = {cat: [] for cat in INSTRUMENT_CATEGORIES}
    all_paths: List[Path] = []

    for stem in stems:
        if stem.get("isRejected", False):
            continue
        instruments = stem.get("instruments", [])
        if not instruments:
            continue
        category = instruments[0].get("category")
        if category not in INSTRUMENT_CATEGORIES:
            continue
        trackname = stem.get("trackname")
        if not trackname:
            continue
        path = safe_stem_path(stems_dir, trackname)
        if path:
            grouped[category].append(path)
            all_paths.append(path)

    # Compute global max length
    global_max = 0
    for p in all_paths:
        try:
            info = torchaudio.info(str(p))
            global_max = max(global_max, info.num_frames)
        except Exception as e:
            logging.warning("    × Failed to inspect %s: %s", p, e)

    # Mix per category
    for category, files in grouped.items():
        if not files:
            continue
        logging.info("  · Mixing category '%s' with %d stems", category, len(files))
        waveforms = []
        for f in files:
            try:
                waveforms.append(load_audio(f))
            except Exception as e:
                logging.warning("    × Failed to load %s: %s", f, e)
        if not waveforms:
            continue
        mixed = mix_waveforms(waveforms, json_name, target_length=global_max)
        dest_dir = OUTPUT_ROOT / json_name / "__10_inst_files__"
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe_name = category.replace("/", "-").replace(" ", "_")
        torchaudio.save(str(dest_dir / f"{safe_name}.wav"), mixed, TARGET_SAMPLE_RATE)

    # Overall mixture
    if all_paths:
        logging.info("  · Creating overall mixture with %d stems", len(all_paths))
        mixture_waveforms = []
        for p in all_paths:
            try:
                mixture_waveforms.append(load_audio(p))
            except Exception as e:
                logging.warning("    × Failed to load %s: %s", p, e)
        if mixture_waveforms:
            mixed_all = mix_waveforms(mixture_waveforms, json_name, target_length=global_max)
            dest_dir_ready.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(dest_dir_ready / "mixture.wav"), mixed_all, TARGET_SAMPLE_RATE)


def main():
    if not JSON_DIR.is_dir():
        raise RuntimeError(f"JSON directory not found: {JSON_DIR}")
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        logging.error("No JSON files found in %s", JSON_DIR)
        return
    for jp in json_files:
        process_json(jp)
    logging.info("Done.")


if __name__ == "__main__":
    main() 