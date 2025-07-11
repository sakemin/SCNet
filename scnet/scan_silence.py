import argparse
import json
import os
from pathlib import Path
# Optional parallelism
from concurrent.futures import ThreadPoolExecutor
import math
import tqdm
import torchaudio as ta
from .wav import SOURCE_VARIATIONS, convert_audio_channels

def find_stem_file(track: Path, source: str, ext: str):
    """Locate stem file within ``track`` (possibly nested) handling SOURCE_VARIATIONS."""
    variants = SOURCE_VARIATIONS.get(source, []) + [f"{source}{ext}"]
    # First, direct children check
    for variant in variants:
        p = track / variant
        if p.exists():
            return p
    # Recursive descent (one match enough)
    for variant in variants:
        for p in track.rglob(variant):
            if p.is_file():
                return p
    return None


def scan_track(root: Path, track: Path, sources, segment, shift, threshold, ext):
    """Return {(src, track_name, seg_idx)} for non-silent segments in one track."""
    out = []
    # Determine sample rate & length from mixture (or any available stem)
    sr = None
    mix_file = find_stem_file(track, 'mixture', ext)
    if mix_file is None:
        # find first existing file to get sr
        for src in sources:
            f = find_stem_file(track, src, ext)
            if f:
                mix_file = f
                break
    if mix_file is None:
        return out  # skip track
    try:
        info = ta.info(str(mix_file))
    except RuntimeError:
        return out
    length_frames = info.num_frames
    sr = info.sample_rate

    if segment is None:
        examples = 1
        seg_frames = None
        stride_frames = None
    else:
        effective_shift = shift or segment  # fallback
        track_duration = length_frames / sr
        examples = 1 if track_duration < segment else int(math.ceil((track_duration - segment) / effective_shift) + 1)
        seg_frames = int(sr * segment)
        stride_frames = int(sr * effective_shift)

    for src in sources:
        stem_path = find_stem_file(track, src, ext)
        if stem_path is None:
            continue
        try:
            wav_full, _ = ta.load(str(stem_path))
        except Exception:
            continue
        wav_full = convert_audio_channels(wav_full, 1)
        total_frames = wav_full.shape[-1]

        if segment is None:
            # Single whole-track check
            if wav_full.count_nonzero() / wav_full.numel() > threshold:
                out.append((src, str(track.relative_to(root)), 0))
            continue  # next source

        # Segment-based evaluation
        assert stride_frames is not None and seg_frames is not None
        for seg_idx in range(examples):
            offset = seg_idx * stride_frames
            if offset >= total_frames:
                break
            end = min(offset + seg_frames, total_frames)
            seg = wav_full[..., offset:end]
            if seg.numel() == 0:
                continue
            if seg.count_nonzero() / seg.numel() > threshold:
                out.append((src, str(track.relative_to(root)), seg_idx))
    return out


def scan_root(root: Path, sources, segment, shift, threshold=0.8, ext='.wav', num_workers=16, output_file='non_silent_segments.json'):
    """Scan entire dataset root and write JSON index."""
    root = Path(root)
    tasks = [p for p in root.glob('**/') if p.is_dir() and not p.name.startswith('.')]

    # Temporarily collect per-source dict track->list(indices)
    result_tmp = {src: {} for src in sources}

    if num_workers and num_workers > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(scan_track, root, track, sources, segment, shift, threshold, ext) for track in tasks]
            for fut in tqdm.tqdm(futures, desc=f"Scanning {root}", ncols=120):
                for src, tname, sidx in fut.result():
                    if tname not in result_tmp[src]:
                        result_tmp[src][tname] = []
                    result_tmp[src][tname].append(sidx)
    else:
        # Sequential execution (deterministic and avoids libsndfile thread issues)
        for track in tqdm.tqdm(tasks, desc=f"Scanning {root}", ncols=120):
            for src, tname, sidx in scan_track(root, track, sources, segment, shift, threshold, ext):
                if tname not in result_tmp[src]:
                    result_tmp[src][tname] = []
                result_tmp[src][tname].append(sidx)

    # Convert to list-of-dict format per user request
    result = {}
    for src, tracks in result_tmp.items():
        result[src] = [ {"track": t, "segments": idxs} for t, idxs in tracks.items() ]

    out_path = root / output_file
    with open(out_path, 'w') as f:
        json.dump(result, f)
    print(f"Saved index to {out_path} (per-source counts: {[len(v) for v in result.values()]})")


def main():
    p = argparse.ArgumentParser(description='Precompute non-silent segment index for SCNet datasets.')
    p.add_argument('--root', nargs='+', help='Dataset root folder(s)')
    p.add_argument('--sources', nargs='+', help='Source names to analyse')
    p.add_argument('--segment', type=float, default=11, help='Segment length (seconds)')
    p.add_argument('--shift', type=float, default=0.5, help='Shift/stride between segments (seconds)')
    p.add_argument('--threshold', type=float, default=0.8, help='Non-zero ratio threshold')
    p.add_argument('--ext', default='.wav', help='Audio file extension')
    p.add_argument('--workers', type=int, default=16, help='Max parallel workers (0 or 1 for sequential)')
    p.add_argument('--output', default='non_silent_segments.json', help='Output file name placed inside each root')
    args = p.parse_args()

    if not args.root or not args.sources:
        print("--root and --sources must be provided.")
        return

    for r in args.root:
        scan_root(Path(r), args.sources, args.segment, args.shift, threshold=args.threshold, ext=args.ext, num_workers=args.workers, output_file=args.output)

if __name__ == '__main__':
    main() 