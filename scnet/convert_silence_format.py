import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def is_old_format(value: Any) -> bool:
    """Return True if the per-source value appears to be the old [[track, idx], ...] list."""
    if not isinstance(value, list):
        return False
    if not value:
        return False
    first = value[0]
    return isinstance(first, list) and len(first) == 2 and isinstance(first[1], int)


def convert_old_to_new(old_list: List[List[Any]]) -> List[Dict[str, Any]]:
    """Aggregate segment indices per track and output list of dicts."""
    track_map: Dict[str, List[int]] = {}
    for track, idx in old_list:
        track_map.setdefault(track, []).append(idx)
    return [{"track": t, "segments": sorted(seg_ids)} for t, seg_ids in track_map.items()]


def convert_file(path: Path, inplace: bool = True, suffix: str = ".v2") -> None:
    data = json.loads(Path(path).read_text())
    changed = False
    new_data = {}
    for src, value in data.items():
        if is_old_format(value):
            new_data[src] = convert_old_to_new(value)
            changed = True
        else:
            new_data[src] = value  # already new format or unknown structure
    if not changed:
        print(f"{path}: already up-to-date; no changes written.")
        return

    out_path = path if inplace else path.with_suffix(path.suffix + suffix)
    out_path.write_text(json.dumps(new_data))
    print(f"Converted {path} → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Convert old silence-index JSONs to new aggregated format.")
    p.add_argument("files", nargs="+", help="Path(s) to silence JSON files or directories containing them.")
    p.add_argument("--inplace", action="store_true", help="Overwrite files instead of writing *.v2.json")
    args = p.parse_args()

    for item in args.files:
        pth = Path(item)
        if pth.is_dir():
            for json_file in pth.rglob("*.json"):
                convert_file(json_file, inplace=args.inplace)
        else:
            convert_file(pth, inplace=args.inplace)


if __name__ == "__main__":
    main() 