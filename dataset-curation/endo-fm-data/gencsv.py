"""Create the headerless manifest expected by the EndoFM pretraining loader."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing prepared MP4 clips.")
    parser.add_argument("--output", type=Path, help="Manifest path (default: ROOT/train.csv).")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Video directory does not exist: {root}")

    videos = sorted(path for path in root.rglob("*.mp4") if path.is_file())
    if not videos:
        parser.error(f"No MP4 clips found under {root}")
    relative_paths = [video.relative_to(root) for video in videos]
    for relative in relative_paths:
        if any(char in str(relative) for char in (",", "\n", "\r")):
            parser.error(f"Clip paths must not contain commas or newlines: {relative}")
    output = args.output or root / "train.csv"
    if output.exists():
        parser.error(f"Manifest already exists: {output}; choose another --output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as manifest:
        writer = csv.writer(manifest)
        for relative in relative_paths:
            writer.writerow([str(relative.parent), relative.name, -1])
    print(f"Wrote {len(videos)} clips to {output}")


if __name__ == "__main__":
    main()
