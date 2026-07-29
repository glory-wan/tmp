#!/usr/bin/env python3
"""Copy source images referenced by hard_<image_id>.jpg synthetic images."""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_SYNTHETIC_DIR = Path(r"E:\Experiment\round_1\synthetic\images")
DEFAULT_SOURCE_DIR = Path(r"E:\Dataset\Detection\COCOmini\COCOmini\images\minitrain2017")
HARD_IMAGE_PATTERN = re.compile(r"^hard_(?P<image_id>\d+)\.jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract image IDs from hard_<image_id>.jpg files and copy the matching "
            "<image_id>.jpg source images to an output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR,
        help="directory containing hard_<image_id>.jpg images",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="directory containing the original <image_id>.jpg images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to which matching original images are copied",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite destination files that already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the result without creating the output directory or copying files",
    )
    return parser.parse_args()


def find_image_ids(synthetic_dir: Path) -> Tuple[List[str], List[Path]]:
    image_ids = set()
    invalid_files = []

    for image_path in sorted(synthetic_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() != ".jpg":
            continue

        match = HARD_IMAGE_PATTERN.fullmatch(image_path.name)
        if match:
            image_ids.add(match.group("image_id"))
        else:
            invalid_files.append(image_path)

    return sorted(image_ids), invalid_files


def print_examples(title: str, values: Iterable[object], limit: int = 20) -> None:
    values = list(values)
    if not values:
        return

    print(f"{title} ({len(values)}):")
    for value in values[:limit]:
        print(f"  {value}")
    if len(values) > limit:
        print(f"  ... and {len(values) - limit} more")


def copy_source_images(
    image_ids: Iterable[str],
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {
        "copied": [],
        "existing": [],
        "missing": [],
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        source_path = source_dir / f"{image_id}.jpg"
        destination_path = output_dir / source_path.name

        if not source_path.is_file():
            result["missing"].append(source_path)
            continue

        if destination_path.exists() and not overwrite:
            result["existing"].append(destination_path)
            continue

        if not dry_run:
            shutil.copy2(source_path, destination_path)
        result["copied"].append(destination_path)

    return result


def main() -> int:
    args = parse_args()
    synthetic_dir = args.synthetic_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not synthetic_dir.is_dir():
        print(f"ERROR: synthetic image directory does not exist: {synthetic_dir}")
        return 2
    if not source_dir.is_dir():
        print(f"ERROR: source image directory does not exist: {source_dir}")
        return 2

    image_ids, invalid_files = find_image_ids(synthetic_dir)
    if not image_ids:
        print(f"ERROR: no valid hard_<image_id>.jpg files found in: {synthetic_dir}")
        print_examples("Files with invalid names", invalid_files)
        return 1

    result = copy_source_images(
        image_ids=image_ids,
        source_dir=source_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    action = "Would copy" if args.dry_run else "Copied"
    print(f"Synthetic directory: {synthetic_dir}")
    print(f"Source directory:    {source_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"Valid image IDs:     {len(image_ids)}")
    print(f"{action}:              {len(result['copied'])}")
    print(f"Already existing:    {len(result['existing'])}")
    print(f"Missing source files:{len(result['missing']):>5}")
    print(f"Invalid names:       {len(invalid_files)}")

    print_examples("Missing source files", result["missing"])
    print_examples("Files with invalid names", invalid_files)
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
