from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lightweight original-image reference dataset from a generated condition manifest."
    )
    parser.add_argument("--condition-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def yolo_line(item: dict) -> str:
    x1, y1, x2, y2 = [float(value) for value in item["gligen_box"]]
    return (
        f"{int(item['class_id'])} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} "
        f"{x2 - x1:.8f} {y2 - y1:.8f}"
    )


def main() -> None:
    args = parse_args()
    condition_dir = Path(args.condition_dir)
    output_dir = Path(args.output_dir)
    if args.clean_output:
        for name in ("images", "labels"):
            path = output_dir / name
            if path.exists():
                shutil.rmtree(path)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((condition_dir / "manifest.json").read_text(encoding="utf-8"))
    reference_manifest = []
    for item in manifest:
        name = str(item["file_name"])
        source = Path(item["source_file"]).resolve()
        destination = images_dir / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source, destination)
        lines = [yolo_line(layout_item) for layout_item in item["layout_objects"]]
        (labels_dir / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        reference_item = dict(item)
        reference_item["result_file"] = str(destination.relative_to(output_dir))
        reference_item["generation_mode"] = "source_original_reference"
        reference_manifest.append(reference_item)

    (output_dir / "manifest.json").write_text(
        json.dumps(reference_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "prompt_condition.json").write_text(
        json.dumps({"phrase_mode": "source_original", "source_condition": str(condition_dir)}, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(reference_manifest)} paired source images in {output_dir}")


if __name__ == "__main__":
    main()
