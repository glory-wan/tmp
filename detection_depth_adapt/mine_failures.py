from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from detection.detector import YoloV7Detector

from .common import CocoMini, FailureSample, parse_class_subset, save_failures, setup_logger, subset_label


def xywh_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x, y, w, h = box
    a = np.array([x, y, x + w, y + h], dtype=np.float32)
    b = boxes.astype(np.float32).copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
    tl, br = np.maximum(a[:2], b[:, :2]), np.minimum(a[2:], b[:, 2:])
    inter = np.maximum(br - tl, 0).prod(1)
    union = w * h + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / (union + 1e-9)


def classify(image_id, annotations, predictions, cfg, forgotten):
    failures = []
    boxes = np.asarray([p["bbox"] for p in predictions], dtype=np.float32).reshape(-1, 4)
    for ann in annotations:
        ious = xywh_iou(ann["bbox"], boxes)
        best_idx = int(ious.argmax()) if len(ious) else -1
        best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
        pred = predictions[best_idx] if best_idx >= 0 else {}
        confidence = float(pred.get("confidence", 0.0))
        top2_gap = float(pred.get("top2_gap", 1.0))
        correct_class = int(pred.get("class_id", -1)) == int(ann["class_id"])
        failure_type = None
        if best_iou < cfg["match_iou"]:
            failure_type = "miss"
        elif not correct_class:
            failure_type = "classification"
        elif confidence < cfg["low_confidence"]:
            failure_type = "low_confidence"
        elif top2_gap < cfg["ambiguous_gap"]:
            failure_type = "top1_top2_close"
        key = f"{image_id}:{ann['id']}"
        state = forgotten.get(key, {"forgotten_count": 0, "was_correct": False})
        if failure_type:
            forgotten_count = int(state.get("forgotten_count", 0)) + int(state.get("was_correct", False))
            forgotten[key] = {"forgotten_count": forgotten_count, "was_correct": False}
            if forgotten_count >= cfg.get("forgotten_threshold", 1):
                failure_type = "forgotten"
            failures.append(FailureSample(int(image_id), tuple(map(float, ann["bbox"])), int(ann["class_id"]),
                                          failure_type, confidence, top2_gap, forgotten_count))
        else:
            forgotten[key] = {"forgotten_count": int(state.get("forgotten_count", 0)), "was_correct": True}
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--yolov7", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--low-confidence", type=float, default=0.25)
    parser.add_argument("--ambiguous-gap", type=float, default=0.08)
    parser.add_argument("--forgotten-threshold", type=int, default=1)
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    args = parser.parse_args()

    logger = setup_logger(Path(args.output).parent)
    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    dataset = CocoMini(args.annotations, args.images, class_subset=class_subset)
    total_annotations = sum(len(v) for v in dataset.annotations.values())
    logger.info("dataset loaded: annotations=%s images_dir=%s images=%d annotations=%d classes=%d class_subset=%s",
                dataset.annotation_file, dataset.images_dir, len(dataset.images), total_annotations, len(dataset.class_names), subset_label(class_subset))
    detector = YoloV7Detector(args.yolov7, args.weights, args.device, args.image_size)
    state_file = Path(args.state)
    forgotten = json.loads(state_file.read_text()) if state_file.exists() else {}
    cfg = vars(args)
    failures = []
    images = list(dataset.iter_images(args.max_images))
    logger.info("failure mining start: images=%d weights=%s", len(images), args.weights)
    for idx, (image_id, path, anns) in enumerate(tqdm(images, desc="failure mining"), start=1):
        preds = detector.predict(Image.open(path).convert("RGB"), conf=args.conf)
        failures.extend(classify(image_id, anns, preds, cfg, forgotten))
        if idx == 1 or idx % 250 == 0 or idx == len(images):
            logger.info("failure mining progress: %d/%d failures=%d", idx, len(images), len(failures))
    by_type = {}
    by_class = {}
    for failure in failures:
        by_type[failure.failure_type] = by_type.get(failure.failure_type, 0) + 1
        by_class[failure.class_id] = by_class.get(failure.class_id, 0) + 1
    save_failures(args.output, failures, {"source": "real_only", "weights": args.weights, "class_subset": sorted(class_subset) if class_subset is not None else None})
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(forgotten, indent=2))
    logger.info("failure mining complete: failures=%d by_type=%s top_classes=%s output=%s",
                len(failures), by_type, sorted(by_class.items(), key=lambda x: x[1], reverse=True)[:10], args.output)


if __name__ == "__main__":
    main()
