from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .common import CocoMini, append_yolov7_device_arg, parse_class_subset, run_logged_subprocess, setup_logger, subset_label, write_yolo_yaml


def find_prediction_json(run_dir: Path, weights: str | Path) -> Path | None:
    expected = run_dir / f"{Path(weights).stem}_predictions.json"
    if expected.exists():
        return expected
    candidates = sorted(run_dir.glob("*_predictions.json"))
    return candidates[0] if candidates else None


def _mean_valid(values) -> float | None:
    import numpy as np

    valid = values[values > -1]
    if valid.size == 0:
        return None
    return float(np.mean(valid))


def compute_per_class_metrics(annotation_file: Path, prediction_file: Path, dataset: CocoMini) -> list[dict]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotation_file))
    coco_dt = coco_gt.loadRes(str(prediction_file))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]  # [iou, recall, class, area, max_det]
    recall = evaluator.eval["recall"]  # [iou, class, area, max_det]
    iou_thrs = list(evaluator.params.iouThrs)
    ap50_idx = min(range(len(iou_thrs)), key=lambda i: abs(float(iou_thrs[i]) - 0.50))
    ap75_idx = min(range(len(iou_thrs)), key=lambda i: abs(float(iou_thrs[i]) - 0.75))

    gt_counts: dict[int, int] = {}
    for ann in coco_gt.dataset.get("annotations", []):
        cat_id = int(ann["category_id"])
        gt_counts[cat_id] = gt_counts.get(cat_id, 0) + 1

    rows = []
    for k, category_id in enumerate(evaluator.params.catIds):
        category_id = int(category_id)
        class_id = dataset.category_to_class.get(category_id)
        category = dataset.categories.get(category_id, {"name": str(category_id)})
        p_all = precision[:, :, k, 0, -1]
        r_all = recall[:, k, 0, -1]
        rows.append({
            "category_id": category_id,
            "class_id": int(class_id) if class_id is not None else None,
            "class_name": str(category["name"]),
            "num_gt": int(gt_counts.get(category_id, 0)),
            "AP": _mean_valid(p_all),
            "AP50": _mean_valid(precision[ap50_idx, :, k, 0, -1]),
            "AP75": _mean_valid(precision[ap75_idx, :, k, 0, -1]),
            "AP_small": _mean_valid(precision[:, :, k, 1, -1]),
            "AP_medium": _mean_valid(precision[:, :, k, 2, -1]),
            "AP_large": _mean_valid(precision[:, :, k, 3, -1]),
            "AR": _mean_valid(r_all),
        })
    return rows


def write_per_class_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["category_id", "class_id", "class_name", "num_gt", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-list", default=None, help="Pre-exported YOLO val image list. Skips COCO->YOLO export when set.")
    parser.add_argument("--coco-annotations", default=None, help="Pre-exported COCO annotation file for YOLOv7 json evaluation.")
    parser.add_argument("--yolov7", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--run-name", default="eval_depth_adapt")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    (output / "datasets").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)
    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    val = CocoMini(args.val_annotations, args.val_images, class_subset=class_subset)
    logger.info("class subset: %s", subset_label(class_subset))
    val_ids = [x[0] for x in val.iter_images(None)]
    if args.val_list and args.coco_annotations:
        val_list = Path(args.val_list).resolve()
        ann_file = Path(args.coco_annotations).resolve()
        logger.info("using shared YOLO val dataset: val_list=%s coco_annotations=%s", val_list, ann_file)
    else:
        val_list = val.export_yolo(output / "datasets" / "val", val_ids)
        ann_file = val.export_coco_annotations(output / "datasets" / "instances_val_subset.json", val_ids)
        logger.info("exported local YOLO val dataset: val_list=%s coco_annotations=%s", val_list, ann_file)
    data_yaml = output / "datasets" / "eval.yaml"
    write_yolo_yaml(data_yaml, val_list, val_list, val.class_names)
    val_list.with_suffix(".cache").unlink(missing_ok=True)
    logger.info("evaluation start: val_images=%d val_annotations=%d weights=%s data_yaml=%s",
                len(val_ids), sum(len(v) for v in val.annotations.values()), args.weights, data_yaml)
    command = [
        sys.executable, str(Path(args.yolov7) / "test.py"),
        "--data", str(data_yaml.resolve()),
        "--img", str(args.image_size),
        "--batch", str(args.batch_size),
        "--weights", str(Path(args.weights).resolve()),
        "--task", "val",
        "--project", str((output / "runs").resolve()),
        "--name", args.run_name,
        "--exist-ok",
        "--save-json",
    ]
    append_yolov7_device_arg(command, args.device)
    import os
    env = os.environ.copy()
    env["YOLO_IS_COCO"] = "1"
    env["COCO_ANNOTATIONS"] = str(ann_file.resolve())
    log_file = output / "yolov7_eval.log"
    logger.info("COCO annotations for eval: %s", ann_file)
    run_logged_subprocess(command, Path(args.yolov7), logger, log_file, prefix="yolov7-eval", env=env)
    run_dir = output / "runs" / args.run_name
    metrics_file = run_dir / "coco_metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    pred_json = find_prediction_json(run_dir, args.weights)
    per_class_rows = []
    if pred_json is None:
        logger.warning("per-class metrics skipped: prediction json not found in %s", run_dir)
    else:
        try:
            per_class_rows = compute_per_class_metrics(ann_file, pred_json, val)
            per_class_json = output / "per_class_metrics.json"
            per_class_csv = output / "per_class_metrics.csv"
            per_class_json.write_text(json.dumps(per_class_rows, indent=2, ensure_ascii=False), encoding="utf-8")
            write_per_class_csv(per_class_csv, per_class_rows)
            metrics["per_class"] = per_class_rows
            metrics["per_class_metrics_file"] = str(per_class_json)
            metrics["per_class_metrics_csv"] = str(per_class_csv)
            logger.info("per-class metrics saved: json=%s csv=%s classes=%d", per_class_json, per_class_csv, len(per_class_rows))
        except Exception as exc:
            logger.warning("per-class metrics skipped: %s", exc)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary_metrics = {k: v for k, v in metrics.items() if k != "per_class"}
    logger.info("evaluation complete: metrics=%s metrics_file=%s", summary_metrics, output / "metrics.json")


if __name__ == "__main__":
    main()
