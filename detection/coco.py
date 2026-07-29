from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class CocoDataset:
    def __init__(self, annotation_file: str | Path, images_dir: str | Path):
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.images = {int(x["id"]): x for x in data["images"]}
        self.categories = {int(x["id"]): x for x in data["categories"]}
        ordered = sorted(self.categories)
        self.category_to_class = {cat_id: idx for idx, cat_id in enumerate(ordered)}
        self.class_names = [self.categories[x]["name"] for x in ordered]
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            if not ann.get("iscrowd", 0) and ann.get("area", 0) > 0:
                item = dict(ann)
                item["class_id"] = self.category_to_class[int(ann["category_id"])]
                self.annotations[int(ann["image_id"])].append(item)

    def image_path(self, image_id: int) -> Path:
        return self.images_dir / self.images[int(image_id)]["file_name"]

    def iter_images(self, limit: int | None = None) -> Iterable[tuple[int, Path, list[dict]]]:
        ids = sorted(self.images)
        if limit is not None:
            ids = ids[:limit]
        for image_id in ids:
            path = self.image_path(image_id)
            if path.exists():
                yield image_id, path, self.annotations.get(image_id, [])

    def export_yolo(self, output: str | Path, image_ids: Iterable[int], *, link_images: bool = True) -> Path:
        output = Path(output)
        images_out, labels_out = output / "images", output / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        kept = []
        for image_id in image_ids:
            info = self.images[int(image_id)]
            src = self.image_path(image_id)
            if not src.exists():
                continue
            dst = images_out / info["file_name"]
            if not dst.exists():
                if link_images:
                    os.symlink(src.resolve(), dst)
                else:
                    dst.write_bytes(src.read_bytes())
            width, height = float(info["width"]), float(info["height"])
            lines = []
            for ann in self.annotations.get(int(image_id), []):
                x, y, w, h = map(float, ann["bbox"])
                lines.append(f"{ann['class_id']} {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}")
            (labels_out / (Path(info["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            # Keep the symlink path: YOLO derives labels by replacing /images/
            # with /labels/. Resolving would point back to the raw COCO folder.
            kept.append(str(dst.absolute()))
        (output / "images.txt").write_text("\n".join(kept) + "\n")
        return output / "images.txt"


def write_yolo_data_yaml(path: str | Path, train_list: Path, val_list: Path, names: list[str]) -> None:
    import yaml
    payload = {"train": str(train_list), "val": str(val_list), "nc": len(names), "names": names}
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False
