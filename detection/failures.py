from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class FailureSample:
    image_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    failure_type: str
    confidence: float
    top2_gap: float
    best_iou: float
    annotation_id: int
    forgotten_count: int = 0
    failure_count: int = 0
    source: str = "real"


def xywh_iou(box: list[float] | tuple[float, ...], boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x, y, w, h = box
    a = np.array([x, y, x + w, y + h], dtype=np.float32)
    b = boxes.astype(np.float32).copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
    tl, br = np.maximum(a[:2], b[:, :2]), np.minimum(a[2:], b[:, 2:])
    inter = np.maximum(br - tl, 0).prod(1)
    return inter / ((w * h) + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) - inter + 1e-9)


def classify_failures(image_id: int, annotations: list[dict], predictions: list[dict], cfg: dict, forgotten: dict[str, int]) -> list[FailureSample]:
    out = []
    pred_boxes = np.asarray([p["bbox"] for p in predictions], dtype=np.float32).reshape(-1, 4)
    iou_matrix = np.stack([xywh_iou(ann["bbox"], pred_boxes) for ann in annotations]) if annotations else np.empty((0, len(predictions)))
    # One prediction may explain at most one GT instance.
    assignments = {}
    candidates = [(float(iou_matrix[g, p]), g, p) for g in range(len(annotations)) for p in range(len(predictions))]
    used_gt, used_pred = set(), set()
    for score, gt_idx, pred_idx in sorted(candidates, reverse=True):
        if score <= 0:
            break
        if gt_idx not in used_gt and pred_idx not in used_pred:
            assignments[gt_idx] = pred_idx
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
    for gt_idx, ann in enumerate(annotations):
        ious = iou_matrix[gt_idx]
        best_idx = assignments.get(gt_idx, -1)
        best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
        pred = predictions[best_idx] if best_idx >= 0 else {}
        confidence = float(pred.get("confidence", 0.0))
        gap = float(pred.get("top2_gap", 1.0))
        correct_class = int(pred.get("class_id", -1)) == int(ann["class_id"])
        failure = None
        if best_iou < cfg["match_iou"]:
            same_class = [i for i, p in enumerate(predictions) if int(p["class_id"]) == int(ann["class_id"])]
            failure = "localization" if same_class and max(ious[same_class], default=0) >= cfg["localization_iou"] else "miss"
        elif not correct_class:
            failure = "classification"
        elif gap < cfg["ambiguous_gap"]:
            failure = "ambiguous"
        elif confidence < cfg["low_confidence"]:
            failure = "low_confidence"
        key = f"{image_id}:{ann['id']}"
        previous = forgotten.get(key, {})
        if isinstance(previous, int):
            previous = {"forgotten_count": previous, "failure_count": previous, "was_correct": False}
        if failure:
            failure_count = int(previous.get("failure_count", 0)) + 1
            forgotten_count = int(previous.get("forgotten_count", 0)) + int(previous.get("was_correct", False))
            forgotten[key] = {"forgotten_count": forgotten_count, "failure_count": failure_count, "was_correct": False}
            out.append(FailureSample(image_id, tuple(ann["bbox"]), int(ann["class_id"]), failure,
                                     confidence, gap, best_iou, int(ann["id"]), forgotten_count, failure_count))
        else:
            forgotten[key] = {"forgotten_count": int(previous.get("forgotten_count", 0)),
                              "failure_count": int(previous.get("failure_count", 0)), "was_correct": True}
    return out


def save_failures(path: str | Path, failures: list[FailureSample], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "failures": [asdict(x) for x in failures]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_failures(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    failures = data["failures"]
    if any(x.get("source") != "real" for x in failures):
        raise ValueError("Failure pool contains synthetic samples; only real images are permitted")
    return failures


def stratified_limit(items: list[FailureSample], per_prompt: int) -> list[FailureSample]:
    groups: dict[tuple[int, str], list[FailureSample]] = defaultdict(list)
    for item in items:
        groups[(item.class_id, item.failure_type)].append(item)
    selected = []
    for group in groups.values():
        group.sort(key=lambda x: (x.forgotten_count, x.failure_count, 1.0 - x.confidence, 1.0 - x.best_iou), reverse=True)
        selected.extend(group[:per_prompt])
    return selected
