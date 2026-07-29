from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image


@contextmanager
def yolo_imports(repo: str | Path):
    repo = str(Path(repo).resolve())
    sys.path.insert(0, repo)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == repo:
            sys.path.pop(0)


class YoloV7Detector:
    """Small adapter around the official YOLOv7 repository.

    `predict` retains the raw candidate's two largest class scores, which is
    required for ambiguity mining and is unavailable in ordinary NMS output.
    """

    def __init__(self, repo: str | Path, weights: str | Path, device: str = "cuda:0", image_size: int = 640):
        import torch
        self.torch = torch
        self.repo, self.weights = Path(repo), Path(weights)
        self.image_size = int(image_size)
        with yolo_imports(self.repo):
            from models.experimental import attempt_load
            from utils.general import check_img_size
            from utils.torch_utils import select_device
            self.device = select_device(device)
            self.model = attempt_load(str(self.weights), map_location=self.device)
            if not hasattr(self.model, "hyp"):
                import yaml
                self.model.hyp = yaml.safe_load((self.repo / "data" / "hyp.scratch.p5.yaml").read_text())
            if not hasattr(self.model, "gr"):
                self.model.gr = 1.0
            self.stride = int(self.model.stride.max())
            self.image_size = check_img_size(self.image_size, s=self.stride)
            self.model.eval()
        for parameter in self.model.parameters():
            if not parameter.is_leaf:
                parameter.detach_()
            parameter.requires_grad_(False)

    def _preprocess(self, image: Image.Image):
        import torch
        with yolo_imports(self.repo):
            from utils.datasets import letterbox
        rgb = np.asarray(image.convert("RGB"))
        array, ratio, pad = letterbox(rgb, self.image_size, stride=self.stride, auto=False)
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).to(self.device).float() / 255.0
        return tensor.unsqueeze(0), ratio, pad

    def predict(self, image: Image.Image, conf: float = 0.001, iou: float = 0.6) -> list[dict]:
        import torch
        with torch.no_grad(), yolo_imports(self.repo):
            from utils.general import non_max_suppression, scale_coords
            tensor, ratio, pad = self._preprocess(image)
            inference, _ = self.model(tensor)
            # Associate ambiguity with the nearest raw candidate after NMS.
            class_probs = inference[0, :, 5:] * inference[0, :, 4:5]
            top2 = class_probs.topk(min(2, class_probs.shape[1]), dim=1).values
            raw_xywh = inference[0, :, :4]
            detections = non_max_suppression(inference, conf, iou, multi_label=False)[0]
            if detections is None:
                return []
            scaled = detections.clone()
            scale_coords(tensor.shape[2:], scaled[:, :4], image.size[::-1])
            output = []
            for det, original in zip(detections, scaled):
                center = (det[:2] + det[2:4]) / 2
                nearest = ((raw_xywh[:, :2] - center) ** 2).sum(1).argmin()
                gap = float(top2[nearest, 0] - top2[nearest, 1]) if top2.shape[1] > 1 else 1.0
                x1, y1, x2, y2 = original[:4].tolist()
                output.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "confidence": float(det[4]),
                               "class_id": int(det[5]), "top2_gap": gap})
            return output

    def differentiable_loss(self, images, targets):
        """Positive-anchor YOLO loss for only the active GT instance.

        Unlike the normal training loss, this deliberately omits objectness at
        every non-target grid cell, so other objects in the source image do not
        enter the prompt objective.
        """
        import torch
        import torch.nn.functional as F
        with yolo_imports(self.repo):
            from utils.loss import ComputeLoss
            from utils.general import bbox_iou
        resized = F.interpolate(images, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        _, raw = self.model(resized)
        criterion = ComputeLoss(self.model)
        tcls, tbox, indices, anchors = criterion.build_targets(raw, targets)
        lbox = torch.zeros(1, device=images.device)
        lobj = torch.zeros(1, device=images.device)
        lcls = torch.zeros(1, device=images.device)
        for layer, prediction in enumerate(raw):
            b, a, gj, gi = indices[layer]
            if not b.numel():
                continue
            selected = prediction[b, a, gj, gi]
            pxy = selected[:, :2].sigmoid() * 2 - 0.5
            pwh = (selected[:, 2:4].sigmoid() * 2) ** 2 * anchors[layer]
            iou = bbox_iou(torch.cat((pxy, pwh), 1).T, tbox[layer], x1y1x2y2=False, CIoU=True)
            lbox += (1 - iou).mean()
            object_target = ((1 - criterion.gr) + criterion.gr * iou.detach().clamp(0)).to(selected.dtype)
            lobj += criterion.BCEobj(selected[:, 4], object_target)
            if criterion.nc > 1:
                class_target = torch.full_like(selected[:, 5:], criterion.cn)
                class_target[range(len(selected)), tcls[layer]] = criterion.cp
                lcls += criterion.BCEcls(selected[:, 5:], class_target)
        return criterion.hyp["box"] * lbox + criterion.hyp["obj"] * lobj + criterion.hyp["cls"] * lcls


class MockDetector:
    """Deterministic backend for CI/smoke tests; never used for real results."""

    def __init__(self, seed: int = 0, **_):
        self.rng = np.random.default_rng(seed)

    def predict(self, image: Image.Image, **_) -> list[dict]:
        width, height = image.size
        if self.rng.random() < 0.5:
            return []
        return [{"bbox": [0.2 * width, 0.2 * height, 0.4 * width, 0.4 * height],
                 "confidence": float(self.rng.uniform(.1, .8)), "class_id": 0,
                 "top2_gap": float(self.rng.uniform(0, .2))}]


def _run_logged_subprocess(command: list[str], *, cwd: Path, env: dict | None = None,
                           logger: logging.Logger | None = None, log_file: Path | None = None) -> None:
    if logger:
        logger.info("command: %s", " ".join(command))
    handle = log_file.open("a", encoding="utf-8") if log_file else None
    try:
        if handle:
            handle.write(f"$ {' '.join(command)}\n")
            handle.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if handle:
                handle.write(line + "\n")
                handle.flush()
            if logger:
                logger.info("[yolov7] %s", line)
            else:
                print(line, flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if handle:
            handle.close()


def run_yolov7_train(repo: Path, data_yaml: Path, weights: Path, output: Path, cfg: dict,
                     logger: logging.Logger | None = None, log_file: Path | None = None) -> Path:
    weights = weights.resolve()
    run_dir = output / cfg["run_name"]
    if logger:
        logger.info("YOLOv7 train start: data=%s weights=%s epochs=%s batch=%s imgsz=%s run_dir=%s",
                    data_yaml, weights, cfg["epochs"], cfg["batch_size"], cfg["image_size"], run_dir)
    command = [sys.executable, str(repo / "train.py"), "--workers", str(cfg["workers"]), "--device", cfg["device"],
               "--batch-size", str(cfg["batch_size"]), "--data", str(data_yaml), "--img", str(cfg["image_size"]),
               "--cfg", str(repo / cfg["model_cfg"]), "--weights", str(weights), "--name", cfg["run_name"],
               "--hyp", str(repo / cfg["hyp"]), "--epochs", str(cfg["epochs"]), "--project", str(output), "--exist-ok"]
    _run_logged_subprocess(command, cwd=repo, logger=logger, log_file=log_file)
    result = run_dir / "weights" / "best.pt"
    if not result.exists():
        result = run_dir / "weights" / "last.pt"
    if logger:
        logger.info("YOLOv7 train complete: checkpoint=%s exists=%s", result, result.exists())
    return result


def run_yolov7_eval(repo: Path, data_yaml: Path, weights: Path, output: Path, cfg: dict,
                    logger: logging.Logger | None = None, log_file: Path | None = None) -> dict:
    weights = weights.resolve()
    run_dir = output / cfg["run_name"]
    if logger:
        logger.info("YOLOv7 eval start: data=%s weights=%s batch=%s imgsz=%s run_dir=%s",
                    data_yaml, weights, cfg["batch_size"], cfg["image_size"], run_dir)
    command = [sys.executable, str(repo / "test.py"), "--data", str(data_yaml), "--img", str(cfg["image_size"]),
               "--batch", str(cfg["batch_size"]), "--device", cfg["device"], "--weights", str(weights),
               "--task", "val", "--project", str(output), "--name", cfg["run_name"], "--exist-ok", "--save-json"]
    env = os.environ.copy()
    env["YOLO_IS_COCO"] = "1"
    env["COCO_ANNOTATIONS"] = cfg["annotation_file"]
    _run_logged_subprocess(command, cwd=repo, env=env, logger=logger, log_file=log_file)
    metrics_file = run_dir / "coco_metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    if logger:
        logger.info("YOLOv7 eval complete: metrics=%s", metrics)
    return {"weights": str(weights), "run_dir": str(run_dir), **metrics}
