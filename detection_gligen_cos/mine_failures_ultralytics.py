from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .common import CocoMini, FailureSample, parse_class_subset, save_failures, setup_logger, subset_label
from .modeling import create_detector


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
        raw_top2_gap = pred.get("top2_gap")
        top2_gap = float(raw_top2_gap) if raw_top2_gap is not None else None
        correct_class = int(pred.get("class_id", -1)) == int(ann["class_id"])
        failure_type = None
        if best_iou < cfg["match_iou"]:
            failure_type = "miss"
        elif not correct_class:
            failure_type = "classification"
        elif confidence < cfg["low_confidence"]:
            failure_type = "low_confidence"
        elif top2_gap is not None and top2_gap < cfg["ambiguous_gap"]:
            failure_type = "top1_top2_close"
        key = f"{image_id}:{ann['id']}"
        state = forgotten.get(key, {"forgotten_count": 0, "was_correct": False})
        if failure_type:
            forgotten_count = int(state.get("forgotten_count", 0)) + int(state.get("was_correct", False))
            forgotten[key] = {"forgotten_count": forgotten_count, "was_correct": False}
            if forgotten_count >= cfg.get("forgotten_threshold", 1):
                failure_type = "forgotten"
            failures.append(FailureSample(int(image_id), tuple(map(float, ann["bbox"])), int(ann["class_id"]),
                                          failure_type, confidence, top2_gap, forgotten_count, int(ann["id"])))
        else:
            forgotten[key] = {"forgotten_count": int(state.get("forgotten_count", 0)), "was_correct": True}
    return failures


def format_failure_counts(counts: dict[int, int]) -> str:
    return json.dumps({str(class_id): counts[class_id] for class_id in sorted(counts)}, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help="Ultralytics dataset YAML whose names define model class ids and COCO category mapping.",
    )
    parser.add_argument("--model-family", required=True, choices=["yolo", "rtdetr"])
    parser.add_argument("--ultralytics-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-failures-per-class", type=int, default=None,
                        help="Keep mining until every target class reaches this many failures, or max-images/dataset end is reached.")
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
    dataset = CocoMini(
        args.annotations,
        args.images,
        class_subset=class_subset,
        dataset_yaml=args.dataset_yaml,
    )
    total_annotations = sum(len(v) for v in dataset.annotations.values())
    logger.info(
        "dataset loaded: annotations=%s images_dir=%s dataset_yaml=%s images=%d annotations=%d "
        "classes=%d class_subset=%s class_mapping=%s",
        dataset.annotation_file,
        dataset.images_dir,
        dataset.dataset_yaml,
        len(dataset.images),
        total_annotations,
        len(dataset.class_names),
        subset_label(class_subset),
        dataset.class_mapping,
    )
    detector = create_detector(
        family=args.model_family,
        weights=args.weights,
        source_root=args.ultralytics_root,
        device=args.device,
        image_size=args.image_size,
        iou=args.iou,
        max_det=args.max_det,
    )
    detector.validate_dataset_names(dataset.class_names)
    logger.info(
        "detector loaded: backend=ultralytics family=%s version=%s weights=%s device=%s image_size=%d",
        detector.family,
        detector.version,
        detector.weights,
        detector.device,
        detector.image_size,
    )
    if not detector.supports_top2_gap:
        logger.warning(
            "top1_top2_close mining is disabled because public Ultralytics detection results do not expose "
            "complete per-box class scores"
        )
    state_file = Path(args.state)
    forgotten = json.loads(state_file.read_text()) if state_file.exists() else {}
    cfg = vars(args)
    failures = []
    # images = list(dataset.iter_images(args.max_images)) #固定前max_images

    all_images = list(dataset.iter_images(None))
    class_aware = args.min_failures_per_class is not None

    if class_aware:
        search_limit = min(len(all_images), args.max_images) if args.max_images is not None else len(all_images)
        images = all_images[:search_limit]
        target_class_ids = sorted(class_subset) if class_subset is not None else sorted(dataset.class_to_category)
        failure_counts = {int(class_id): 0 for class_id in target_class_ids}
        logger.info("class-aware mining enabled: min_failures_per_class=%d target_classes=%s max_images=%s",
                    args.min_failures_per_class, target_class_ids, args.max_images)
    elif args.max_images is not None and len(all_images) > args.max_images:
        rng = random.Random(0)
        images = rng.sample(all_images, args.max_images)
        failure_counts = {}
    else:
        images = all_images
        failure_counts = {}

    logger.info("failure mining start: images=%d weights=%s", len(images), args.weights)
    stop_reason = "dataset_exhausted"
    for idx, (image_id, path, anns) in enumerate(tqdm(images, desc="failure mining"), start=1):
        preds = detector.predict(Image.open(path).convert("RGB"), conf=args.conf)
        image_failures = classify(image_id, anns, preds, cfg, forgotten)
        failures.extend(image_failures)
        if class_aware:
            for failure in image_failures:
                class_id = int(failure.class_id)
                if class_id in failure_counts:
                    failure_counts[class_id] += 1
        if idx == 1 or idx % 250 == 0 or idx == len(images):
            logger.info("failure mining progress: %d/%d failures=%d", idx, len(images), len(failures))
            if class_aware:
                logger.info("class-aware mining progress: %d/%d failures_by_class=%s",
                            idx, len(images), format_failure_counts(failure_counts))
        if class_aware and failure_counts and all(count >= args.min_failures_per_class for count in failure_counts.values()):
            stop_reason = "min_failures_per_class_reached"
            logger.info("class-aware mining early stop: reason=%s processed=%d/%d failures_by_class=%s",
                        stop_reason, idx, len(images), format_failure_counts(failure_counts))
            break
    if class_aware and stop_reason != "min_failures_per_class_reached":
        if args.max_images is not None and len(images) < len(all_images):
            stop_reason = "max_images_reached"
        logger.info("class-aware mining stop: reason=%s processed=%d/%d failures_by_class=%s",
                    stop_reason, len(images), len(images), format_failure_counts(failure_counts))
    by_type = {}
    by_class = {}
    for failure in failures:
        by_type[failure.failure_type] = by_type.get(failure.failure_type, 0) + 1
        by_class[failure.class_id] = by_class.get(failure.class_id, 0) + 1
    save_failures(
        args.output,
        failures,
        {
            "source": "real_only",
            "model_backend": "ultralytics",
            "model_family": detector.family,
            "model_task": "detect",
            "ultralytics_version": detector.version,
            "weights": str(detector.weights),
            "dataset_yaml": str(dataset.dataset_yaml) if dataset.dataset_yaml is not None else None,
            "class_names": dataset.class_names,
            "class_mapping": dataset.class_mapping,
            "class_subset": sorted(class_subset) if class_subset is not None else None,
        },
    )
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(forgotten, indent=2))
    logger.info("failure mining complete: failures=%d by_type=%s top_classes=%s output=%s",
                len(failures), by_type, sorted(by_class.items(), key=lambda x: x[1], reverse=True)[:10], args.output)


if __name__ == "__main__":
    main()

