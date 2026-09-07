from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated layout instances with YOLOv7 and extract backbone ROI features."
    )
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--yolov7", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--eval-conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou-thres", type=float, default=0.65)
    parser.add_argument("--match-iou-thres", type=float, default=0.5)
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=50,
        help="YOLOv7 module index. Layer 50 is the deepest backbone feature in yolov7.yaml.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def read_yolo_labels(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id, cx, cy, width, height = map(float, fields[:5])
        rows.append({
            "class_id": int(class_id),
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        })
    return rows


def normalized_box_to_input(label: dict, original_width: int, original_height: int,
                            ratio: tuple[float, float], pad: tuple[float, float]) -> np.ndarray:
    cx = label["cx"] * original_width
    cy = label["cy"] * original_height
    width = label["width"] * original_width
    height = label["height"] * original_height
    x1 = (cx - width / 2) * ratio[0] + pad[0]
    y1 = (cy - height / 2) * ratio[1] + pad[1]
    x2 = (cx + width / 2) * ratio[0] + pad[0]
    y2 = (cy + height / 2) * ratio[1] + pad[1]
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def find_tensor_feature(value) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.ndim != 4:
            raise ValueError(f"Hooked feature must be 4D, got shape={tuple(value.shape)}")
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if torch.is_tensor(item) and item.ndim == 4:
                return item
    raise TypeError(f"Unable to find a 4D tensor in hooked output type={type(value).__name__}")


def roi_pool(feature: torch.Tensor, box: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    _, _, feature_height, feature_width = feature.shape
    x1 = int(np.floor(box[0] / input_width * feature_width))
    y1 = int(np.floor(box[1] / input_height * feature_height))
    x2 = int(np.ceil(box[2] / input_width * feature_width))
    y2 = int(np.ceil(box[3] / input_height * feature_height))
    x1 = min(max(x1, 0), feature_width - 1)
    y1 = min(max(y1, 0), feature_height - 1)
    x2 = min(max(x2, x1 + 1), feature_width)
    y2 = min(max(y2, y1 + 1), feature_height)
    return feature[0, :, y1:y2, x1:x2].float().mean(dim=(1, 2)).cpu().numpy()


def summarize(rows: list[dict], key: str | None = None) -> dict | dict[str, dict]:
    if key is not None:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[str(row[key])].append(row)
        return {name: summarize(items) for name, items in sorted(grouped.items())}
    if not rows:
        return {"instances": 0}
    return {
        "instances": len(rows),
        "mean_matched_confidence": float(np.mean([row["matched_confidence"] for row in rows])),
        "median_matched_confidence": float(np.median([row["matched_confidence"] for row in rows])),
        "false_negative_rate": float(np.mean([row["false_negative"] for row in rows])),
        "class_error_rate": float(np.mean([row["class_error"] for row in rows])),
        "detection_error_rate": float(np.mean([row["detection_error"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    synthetic_dir = Path(args.synthetic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yolov7_dir = Path(args.yolov7).resolve()
    sys.path.insert(0, str(yolov7_dir))

    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = attempt_load(str(Path(args.weights).resolve()), map_location=device)
    model.eval()
    use_half = device.type == "cuda"
    model.half() if use_half else model.float()
    core_model = model.module if hasattr(model, "module") else model
    if not hasattr(core_model, "model"):
        raise TypeError("Feature extraction requires a single YOLOv7 model, not an ensemble")
    if not (-len(core_model.model) <= args.feature_layer < len(core_model.model)):
        raise IndexError(f"feature-layer {args.feature_layer} outside model range 0..{len(core_model.model) - 1}")

    captured: dict[str, torch.Tensor] = {}

    def hook_feature(_module, _inputs, output):
        captured["feature"] = find_tensor_feature(output).detach()

    handle = core_model.model[args.feature_layer].register_forward_hook(hook_feature)
    condition_file = synthetic_dir / "prompt_condition.json"
    condition_meta = json.loads(condition_file.read_text(encoding="utf-8")) if condition_file.exists() else {}
    condition = str(condition_meta.get("phrase_mode", synthetic_dir.name))
    image_paths = sorted((synthetic_dir / "images").glob("*"))
    image_paths = [path for path in image_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    rows: list[dict] = []
    features: list[np.ndarray] = []
    false_positives = 0
    prediction_count = 0
    for image_index, image_path in enumerate(image_paths):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue
        original_height, original_width = bgr.shape[:2]
        resized, ratio, pad = letterbox(bgr, new_shape=args.image_size, stride=int(model.stride.max()))
        rgb = resized[:, :, ::-1].transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
        tensor = tensor.half() if use_half else tensor.float()
        tensor /= 255.0
        tensor = tensor.unsqueeze(0)
        captured.clear()
        with torch.no_grad():
            raw_prediction = model(tensor, augment=False)[0]
            prediction = non_max_suppression(
                raw_prediction,
                conf_thres=args.conf_thres,
                iou_thres=args.nms_iou_thres,
            )[0]
        if "feature" not in captured:
            raise RuntimeError(f"Feature hook at layer {args.feature_layer} did not run")
        pred_np = prediction.detach().float().cpu().numpy() if prediction is not None else np.empty((0, 6))
        pred_np = pred_np[pred_np[:, 4] >= args.eval_conf_thres]
        prediction_count += len(pred_np)

        labels = read_yolo_labels(synthetic_dir / "labels" / f"{image_path.stem}.txt")
        gt_boxes = [normalized_box_to_input(label, original_width, original_height, ratio, pad) for label in labels]
        matched_prediction_indices: set[int] = set()
        for object_index, (label, gt_box) in enumerate(zip(labels, gt_boxes)):
            if pred_np.size:
                ious = box_iou_one_to_many(gt_box, pred_np[:, :4])
                same_class = pred_np[:, 5].astype(int) == int(label["class_id"])
                valid_same = np.where(same_class & (ious >= args.match_iou_thres))[0]
                valid_any = np.where(ious >= args.match_iou_thres)[0]
            else:
                ious = np.empty((0,), dtype=np.float32)
                valid_same = np.empty((0,), dtype=int)
                valid_any = np.empty((0,), dtype=int)

            if valid_same.size:
                best_index = int(valid_same[np.argmax(pred_np[valid_same, 4])])
                matched_confidence = float(pred_np[best_index, 4])
                matched_iou = float(ious[best_index])
                matched_prediction_indices.add(best_index)
            else:
                matched_confidence = 0.0
                matched_iou = 0.0
            class_error = bool(
                valid_same.size == 0
                and valid_any.size > 0
                and np.any(pred_np[valid_any, 5].astype(int) != int(label["class_id"]))
            )
            false_negative = valid_same.size == 0
            row = {
                "condition": condition,
                "transfer_source_class_id": condition_meta.get("transfer_source_class_id"),
                "image": image_path.name,
                "image_index": image_index,
                "object_index": object_index,
                "class_id": int(label["class_id"]),
                "matched_confidence": matched_confidence,
                "matched_iou": matched_iou,
                "false_negative": int(false_negative),
                "class_error": int(class_error),
                "detection_error": int(false_negative),
                "bbox_area_ratio": float(label["width"] * label["height"]),
            }
            rows.append(row)
            features.append(roi_pool(captured["feature"], gt_box, tensor.shape[3], tensor.shape[2]))
        false_positives += max(len(pred_np) - len(matched_prediction_indices), 0)

    handle.remove()
    fieldnames = list(rows[0]) if rows else ["condition", "image", "object_index", "class_id"]
    with (output_dir / "instances.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    feature_matrix = np.stack(features) if features else np.empty((0, 0), dtype=np.float32)
    np.savez_compressed(
        output_dir / "backbone_roi_features.npz",
        features=feature_matrix,
        condition=np.asarray([row["condition"] for row in rows]),
        image=np.asarray([row["image"] for row in rows]),
        object_index=np.asarray([row["object_index"] for row in rows], dtype=np.int64),
        class_id=np.asarray([row["class_id"] for row in rows], dtype=np.int64),
        false_negative=np.asarray([row["false_negative"] for row in rows], dtype=np.int64),
    )
    metrics = {
        "condition": condition,
        "synthetic_dir": str(synthetic_dir.resolve()),
        "weights": str(Path(args.weights).resolve()),
        "feature_layer": args.feature_layer,
        "feature_dimension": int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
        "images": len(image_paths),
        "predictions": prediction_count,
        "false_positives": false_positives,
        "false_positives_per_image": false_positives / max(len(image_paths), 1),
        "overall": summarize(rows),
        "per_class": summarize(rows, "class_id"),
        "thresholds": {
            "candidate_confidence": args.conf_thres,
            "evaluation_confidence": args.eval_conf_thres,
            "match_iou": args.match_iou_thres,
            "nms_iou": args.nms_iou_thres,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
