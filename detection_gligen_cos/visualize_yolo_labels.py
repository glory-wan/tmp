from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def visualize_one(image_path: Path, label_path: Path, output_path: Path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    draw = ImageDraw.Draw(image)
    font = load_font()

    if not label_path.exists():
        print(f"missing label: {label_path}")
        return

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            print(f"invalid label line: {line}")
            continue

        class_id = int(float(parts[0]))
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        class_name = (
            COCO_NAMES[class_id]
            if 0 <= class_id < len(COCO_NAMES)
            else "unknown"
        )

        label = f"{class_name} [{class_id}]"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_y = max(0, y1 - text_height - 6)

        draw.rectangle(
            [
                x1,
                text_y,
                x1 + text_width + 8,
                text_y + text_height + 6,
            ],
            fill="red",
        )

        draw.text(
            (x1 + 4, text_y + 3),
            label,
            fill="white",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)

    print(f"saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO txt labels."
    )

    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=7500,
    )

    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    # output_dir = Path(args.output_dir)


    if args.output_dir is None:
        output_dir = images_dir.parent / "label_vis"
    else:
        output_dir = Path(args.output_dir)

    image_paths = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.png"))
    )

    for image_path in image_paths[: args.max_images]:
        label_path = labels_dir / f"{image_path.stem}.txt"
        output_path = output_dir / image_path.name

        visualize_one(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()