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
              'mixture': ['Mixed.wav', 'mixed.wav'],
              'high': ['high.wav', 'HIGH.wav'],
              'mid': ['mid.wav', 'MID.wav'],
              'low': ['low.wav', 'LOW.wav'],
              'rhythm': ['rhythm.wav', 'Rhythm.wav', 'rhy.wav'],
              'melody': ['melody.wav', 'Melody.wav'],
              'fx': ['fx.wav', 'FX.wav'],
              'percussion': ['Percussion_&_Drums.wav'],
              'string': ['Strings_Instruments.wav'],
              'fretted': ['Fretted_Instruments.wav'],
              'vocal': ['Vocal_Elements.wav'],
              'wind': ['Wind_Instruments.wav'],
              'brass': ['Brass_Instruments.wav'],
              'keyboard': ['Keyboard_Instruments.wav'],
              'electronic': ['Electronic_&_Synthesized_Instruments.wav'],
              'misc': ['Miscellaneous_Instruments_&_Effects.wav']
            }

def _track_metadata(track, sources, normalize=True, ext=EXT, path_name=None):
    track_length = None
    track_samplerate = None
    mean = 0
    std = 1
    source_length = {}
    source_filename = {}
    for source in [MIXTURE] + sources:
        if path_name in ["beatpulse_audio", "beatpulse_audio_1992", "beatpulse_audio_904", "pointune_audio", "pointune_10insts", "beatpulse_10insts"]:
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
        elif path_name in ["mixaudio_audio"]:
            pass # TODO: implement mixaudio_audio
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
                    if wav.count_nonzero() < 0.8 * wav.numel():
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

                                if wav.count_nonzero() > 0.8 * wav.numel():
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
                silence_mask = nonzero_counts < 0.8 * elems_per_src  # boolean mask per source

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
                        normalize=args.normalize)
        valid_set = MultiRootWavset(args.wav, valids, [MIXTURE] + list(args.sources),
                        samplerate=args.samplerate, channels=args.channels,
                        normalize=args.normalize, **kw_cv)

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
            silence_file_name="non_silent_segments.json"):
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
        # src -> track_path -> [segment_indices]
        self.non_silent_segments_by_source = {src: {} for src in self.sources}

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

        # -------- Load precomputed silence index files --------
        if self.replace_silence or self.toothless == 'replace':
            for root in self.roots:
                if accelerator.is_main_process:
                    print(f"Loading silence index from {root}")
                silence_path = Path(root) / self.silence_file_name
                if not silence_path.is_file():
                    if accelerator.is_main_process:
                        print(f"[warn] Silence index not found for {root}: {silence_path}")
                    continue
                try:
                    data = json.load(open(silence_path, "r"))
                    for src, entries in data.items():
                        if src not in self.sources:
                            continue
                        for entry in entries:  # each entry is {"track": str, "segments": [idx, ...]}
                            track_name = entry.get("track")
                            if track_name not in self.metadatas[str(root)]:
                                continue  # ignore tracks not in current split
                            seg_list = entry.get("segments", [])
                            if not seg_list:
                                continue
                            # Store segments list
                            self.non_silent_segments_by_source[src][track_name] = seg_list
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"[warn] Failed loading silence index {silence_path}: {e}")
            # print summary counts
            if accelerator.is_main_process:
                for s, d in self.non_silent_segments_by_source.items():
                    total_segs = sum(len(v) for v in d.values())
                    print(f"Loaded {total_segs} non-silent segments for source '{s}'.")

        # Pre-compute valid replacements for silence replacement
        self.valid_replacements = {s: [] for s in self.sources}
        for src, track_dict in self.non_silent_segments_by_source.items():
            for t, segs in track_dict.items():
                meta = self.metadatas[self.track_to_root[t]][t]
                for sidx in segs:
                    self.valid_replacements[src].append(
                        (self.track_to_root[t], t, sidx, meta['mean'], meta['std'])
                    )

    def __len__(self):
        return self.num_examples_total

    def get_file(self, root, name, source):
        try:
            return Path(root) / name / self.metadatas[str(root)][name]['source_filename'][source]
        except: # if there's no source filename(no source for this track), return None
            return None

    def __getitem__(self, index):
        """Return one training example, mirroring the logic of ``Wavset.__getitem__``."""
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
                        cands = self.valid_replacements[source]
                        if cands:
                            rand_root, rand_name, rand_seg, src_mean, src_std = random.choice(cands)
                            rand_meta = self.metadatas[rand_root][rand_name]
                            rand_file = self.get_file(rand_root, rand_name, source)
                            rand_offset = int(rand_meta['samplerate'] * self.shift * rand_seg)
                            wav, _ = ta.load(str(rand_file), frame_offset=rand_offset, num_frames=num_frames)
                            wav = convert_audio_channels(wav, self.channels)
                            if src_std > 0:
                                wav = (wav - src_mean) * (meta['std'] / src_std) + meta['mean']
                        else:                                    # nothing for that source → zeros
                            wav = th.zeros(self.channels, num_frames)
                    else:
                        raise ValueError(f"Invalid toothless value: {self.toothless}")
                else:
                    wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                    wav = convert_audio_channels(wav, self.channels)

                # ---------------- Optional silence replacement ---------------
                if self.replace_silence:
                    if wav.count_nonzero() < 0.8 * wav.numel() and random.random() < self.replace_silence_prob:
                        candidates_dict = self.non_silent_segments_by_source.get(source, {})
                        for _ in range(20):
                            if not candidates_dict:
                                break
                            rand_name = random.choice(list(candidates_dict.keys()))
                            segs = candidates_dict[rand_name]
                            if not segs:
                                continue
                            rand_seg = random.choice(segs)
                            rand_root = self.track_to_root.get(rand_name)
                            if rand_root is None:
                                continue
                            rand_meta = self.metadatas[rand_root][rand_name]
                            rand_file = self.get_file(rand_root, rand_name, source)
                            if rand_file is None or not os.path.exists(rand_file):
                                continue
                            if self.segment is not None:
                                rand_offset = int(rand_meta['samplerate'] * self.shift * rand_seg)
                            else:
                                rand_offset = 0
                            wav, _ = ta.load(str(rand_file), frame_offset=rand_offset, num_frames=num_frames)
                            if rand_meta['samplerate'] != meta['samplerate']:
                                wav = julius.resample_frac(wav, rand_meta['samplerate'], meta['samplerate'])
                            wav = convert_audio_channels(wav, self.channels)
                            # Match volume statistics to target track
                            target_mean, target_std = meta['mean'], meta['std']
                            src_mean, src_std = rand_meta.get('mean', 0.0), rand_meta.get('std', 1.0)
                            if src_std > 0:
                                wav = (wav - src_mean) * (target_std / src_std) + target_mean
                            if wav.count_nonzero() > 0.8 * wav.numel():
                                break

                wavs.append(wav)

            # Ensure all stems share the same length (min across stems).
            min_length = min(wav.shape[-1] for wav in wavs)
            wavs = [wav[..., :min_length] for wav in wavs]

            example = th.stack(wavs)  # (nb_sources, channels, time)

            # --------------------- Noise injection -------------------------
            if self.noise_inject:
                elems_per_src = example.shape[1] * example.shape[2]
                nonzero_counts = example.ne(0).sum(dim=(1, 2))
                silence_mask = nonzero_counts < 0.8 * elems_per_src
                if silence_mask.any() and random.random() < self.noise_inject_prob:
                    stds = th.empty(example.size(0), dtype=example.dtype, device=example.device).uniform_(0.00003, 0.00009)
                    noise = th.randn_like(example) * stds[:, None, None]
                    example = example + noise * silence_mask[:, None, None]

            # ---------------- Resampling / Normalisation -------------------
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)

            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example
        