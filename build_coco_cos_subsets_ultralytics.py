#!/usr/bin/env python3
"""Split source images by detector-head gradient cosine using Ultralytics models.

The scoring definition intentionally matches ``build_coco_cos_subsets.py``:

1. average detector-loss gradients from the validation Guide set;
2. compute the same gradients for every source image;
3. compare the gradients with global cosine similarity;
4. materialize positive-, negative-, and zero-cosine image/label subsets.

YOLO detection models use their native loss. YOLO instance-segmentation, pose,
and OBB models use the detection component shared by their prediction head,
because the scoring labels contain axis-aligned boxes only. RT-DETR models use
their native detection loss. No model parameters are updated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_ULTRALYTICS_ROOT = SCRIPT_ROOT / "ultralytics"
DEFAULT_MODEL = DEFAULT_ULTRALYTICS_ROOT / "yolo26n.pt"
DEFAULT_OUTPUT_BASE = Path("/home/suhu/data/wgr/datasets/cocomini_cos_ultralytics")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOX_GUIDANCE_TASKS = {"detect", "segment", "pose", "obb"}

torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score source images by global Guide-set-gradient cosine "
            "using a local Ultralytics YOLO or RT-DETR model."
        )
    )
    parser.add_argument("--model", "--weights", dest="model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--task",
        choices=("auto", "detect", "segment", "pose", "obb"),
        default="detect",
        help="Normally inferred from the checkpoint.",
    )
    parser.add_argument(
        "--ultralytics-root",
        type=Path,
        default=DEFAULT_ULTRALYTICS_ROOT,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Images to score and their YOLO labels. Supported layouts: "
            "ROOT/{images,labels}, ROOT/{images,labels}/train2017, or "
            "ROOT/{images,labels}/minitrain2017."
        ),
    )
    parser.add_argument(
        "--guide-root",
        type=Path,
        required=True,
        help=(
            "Guide set root. ROOT/{images,labels}/val2017 is preferred; "
            "ROOT/{images,labels} is also supported."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: cocomini_cos_ultralytics/MODEL_STEM.",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=None,
        help="Default: OUTPUT_ROOT/gradient_scores_ultralytics_val.",
    )
    parser.add_argument("--positive-name", default="train_cos_more_0")
    parser.add_argument("--negative-name", default="train_cos_less_0")
    parser.add_argument("--zero-name", default="train_cos_equal_0")
    parser.add_argument(
        "--zero-policy",
        choices=("separate",),
        default="separate",
        help="Store cosine == 0 samples in the dedicated --zero-name subset.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--scaleup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow letterbox preprocessing to enlarge small images.",
    )
    parser.add_argument("--cuda-visible-devices", default="6")
    parser.add_argument(
        "--device",
        default="0",
        help="Logical device after CUDA_VISIBLE_DEVICES is applied, or 'cpu'.",
    )
    parser.add_argument("--guide-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument(
        "--max-source-images",
        "--max-train-images",
        dest="max_source_images",
        type=int,
        default=None,
        help="Optional deterministic prefix for smoke tests.",
    )
    parser.add_argument(
        "--max-guide-images",
        type=int,
        default=None,
        help="Optional deterministic prefix for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-gradient-scoring",
        action="store_true",
        help="Reuse a complete scores.jsonl and only rebuild subsets.",
    )
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在或不是文件：{path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label}不存在或不是目录：{path}")


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_yolo_label(
    path: Path,
) -> tuple[np.ndarray, list[str], int, int]:
    rows: list[list[float]] = []
    unique_lines: list[str] = []
    seen: set[bytes] = set()
    duplicate_count = 0
    converted_segment_count = 0
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 5:
            values = [float(value) for value in fields]
            output_line = line
            converted_segment = False
        elif len(fields) >= 7 and len(fields) % 2 == 1:
            polygon = [float(value) for value in fields]
            coordinates = polygon[1:]
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            values = [
                polygon[0],
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                x_max - x_min,
                y_max - y_min,
            ]
            output_line = " ".join(f"{value:.10g}" for value in values)
            converted_segment = True
        else:
            raise ValueError(
                f"标签既不是五列 YOLO 检测框，也不是 YOLO 分割多边形："
                f"{path}:{line_number}，字段数={len(fields)}"
            )
        class_id = int(values[0])
        if values[0] != class_id or class_id < 0:
            raise ValueError(f"类别编号无效：{path}:{line_number} -> {values[0]}")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"标签包含非有限值：{path}:{line_number}")
        if any(value < 0.0 or value > 1.0 for value in values[1:]):
            raise ValueError(f"检测框未归一化到 [0,1]：{path}:{line_number}")
        row_key = struct.pack("!5f", *values)
        if row_key in seen:
            duplicate_count += 1
            continue
        seen.add(row_key)
        rows.append(values)
        unique_lines.append(output_line)
        if converted_segment:
            converted_segment_count += 1
    array = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    return array, unique_lines, duplicate_count, converted_segment_count


def validate_label_pairs(
    images: Sequence[Path],
    label_directory: Path,
    description: str,
) -> None:
    require_directory(label_directory, f"{description}标签目录")
    expected = {f"{image.stem}.txt" for image in images}
    available = {
        path.name
        for path in label_directory.iterdir()
        if path.is_file() and path.suffix == ".txt"
    }
    missing = sorted(expected - available)
    if missing:
        raise FileNotFoundError(
            f"{description}缺少 {len(missing)} 个标签，示例：{missing[:5]}"
        )
    with tqdm(
        images,
        desc=f"校验{description}标签",
        unit="file",
        dynamic_ncols=True,
    ) as progress:
        for image in progress:
            parse_yolo_label(label_directory / f"{image.stem}.txt")


def resolve_image_label_directories(
    root: Path,
    role: str,
    layouts: Sequence[tuple[str, str]],
) -> tuple[Path, Path]:
    require_directory(root, f"{role}根目录")
    tried: list[str] = []
    for image_relative, label_relative in layouts:
        image_directory = root / image_relative
        label_directory = root / label_relative
        tried.append(f"{image_relative} + {label_relative}")
        if (
            image_directory.is_dir()
            and label_directory.is_dir()
            and image_files(image_directory)
        ):
            return image_directory, label_directory
    raise FileNotFoundError(
        f"{role}根目录中没有找到匹配的图像/标签目录：{root}；"
        f"支持的结构={tried}"
    )


def prepare_compatible_source_labels(
    images: Sequence[Path],
    source_directory: Path,
    destination: Path,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{image.stem}.txt" for image in images}
    unexpected = sorted(path.name for path in destination.iterdir() if path.name not in expected_names)
    if unexpected:
        raise RuntimeError(
            f"兼容标签目录包含非本次文件，请先检查：{destination}；示例={unexpected[:5]}"
        )

    duplicate_files = 0
    duplicate_rows = 0
    converted_segment_files = 0
    converted_segment_rows = 0
    for image in tqdm(
        images,
        desc="准备兼容 source 标签",
        unit="file",
        dynamic_ncols=True,
    ):
        source = source_directory / f"{image.stem}.txt"
        _, unique_lines, removed, converted = parse_yolo_label(source)
        target = destination / source.name
        if removed or converted:
            content = "\n".join(unique_lines) + ("\n" if unique_lines else "")
            if target.is_symlink() or not target.is_file() or target.read_text(encoding="utf-8") != content:
                target.unlink(missing_ok=True)
                target.write_text(content, encoding="utf-8")
            if removed:
                duplicate_files += 1
            duplicate_rows += removed
            if converted:
                converted_segment_files += 1
            converted_segment_rows += converted
        elif not (target.is_symlink() and target.resolve() == source.resolve()):
            target.unlink(missing_ok=True)
            target.symlink_to(source)

    summary = {
        "source_directory": str(source_directory),
        "compatible_directory": str(destination),
        "label_count": len(images),
        "duplicate_file_count": duplicate_files,
        "removed_row_count": duplicate_rows,
        "converted_segment_file_count": converted_segment_files,
        "converted_segment_row_count": converted_segment_rows,
    }
    (destination.parent / "label_sanitization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    model = resolved(args.model)
    ultralytics_root = resolved(args.ultralytics_root)
    source_root = resolved(args.source_root)
    guide_root = resolved(args.guide_root)

    require_file(model, "Ultralytics 模型")
    if model.suffix.lower() not in {".pt", ".yaml", ".yml"}:
        raise ValueError("--model 必须是 Ultralytics 原生 .pt/.yaml/.yml 文件")
    require_file(ultralytics_root / "ultralytics" / "__init__.py", "本地 Ultralytics 包")
    source_image_dir, source_labels = resolve_image_label_directories(
        source_root,
        "source",
        (
            ("images", "labels"),
            ("images/train2017", "labels/train2017"),
            ("images/minitrain2017", "labels/minitrain2017"),
        ),
    )
    guide_image_dir, guide_labels = resolve_image_label_directories(
        guide_root,
        "Guide",
        (
            ("images/val2017", "labels/val2017"),
            ("images", "labels"),
        ),
    )

    source_images_full = image_files(source_image_dir)
    guide_images_full = image_files(guide_image_dir)
    if not source_images_full:
        raise RuntimeError(f"source 图像目录为空：{source_image_dir}")
    if not guide_images_full:
        raise RuntimeError(f"Guide 图像目录为空：{guide_image_dir}")
    if len({image.stem for image in source_images_full}) != len(source_images_full):
        raise RuntimeError(f"source 图像存在同名不同扩展名文件：{source_image_dir}")
    if len({image.stem for image in guide_images_full}) != len(guide_images_full):
        raise RuntimeError(f"Guide 图像存在同名不同扩展名文件：{guide_image_dir}")
    overlap = {image.resolve() for image in source_images_full} & {
        image.resolve() for image in guide_images_full
    }
    if overlap:
        raise RuntimeError(
            f"source 与 Guide 集存在 {len(overlap)} 个重复图像，示例："
            f"{str(next(iter(overlap)))}"
        )

    validate_label_pairs(source_images_full, source_labels, "source")
    validate_label_pairs(guide_images_full, guide_labels, "Guide")

    if args.max_source_images is not None:
        if args.max_source_images <= 0:
            raise ValueError("--max-source-images 必须大于 0")
        source_images = source_images_full[: args.max_source_images]
    else:
        source_images = source_images_full
    if args.max_guide_images is not None:
        if args.max_guide_images <= 0:
            raise ValueError("--max-guide-images 必须大于 0")
        guide_images = guide_images_full[: args.max_guide_images]
    else:
        guide_images = guide_images_full

    output_root = (
        resolved(args.output_root)
        if args.output_root is not None
        else DEFAULT_OUTPUT_BASE / model.stem
    )
    score_dir = (
        resolved(args.score_dir)
        if args.score_dir is not None
        else output_root / "gradient_scores_ultralytics_val"
    )
    if args.imgsz <= 0 or args.guide_batch_size <= 0 or args.copy_workers <= 0:
        raise ValueError("--imgsz、--guide-batch-size、--copy-workers 必须大于 0")
    if args.workers < 0:
        raise ValueError("--workers 不能小于 0")
    subset_names = (args.positive_name, args.negative_name, args.zero_name)
    if len(set(subset_names)) != len(subset_names):
        raise ValueError("正、负、零值子集目录名不能相同")
    if any(not value or "/" in value or "\\" in value for value in subset_names):
        raise ValueError("正、负、零值子集名称必须是非空的单层目录名")

    output_root.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    scoring_labels = prepare_compatible_source_labels(
        source_images_full,
        source_labels,
        score_dir / "compatible_source_labels",
    )
    return {
        "model": model,
        "ultralytics_root": ultralytics_root,
        "source_root": source_root,
        "source_image_dir": source_image_dir,
        "source_labels": source_labels,
        "guide_root": guide_root,
        "guide_image_dir": guide_image_dir,
        "guide_labels": guide_labels,
        "scoring_labels": scoring_labels,
        "source_images": source_images,
        "guide_images": guide_images,
        "source_count": len(source_images),
        "guide_count": len(guide_images),
        "output_root": output_root,
        "score_dir": score_dir,
    }


class YoloBoxDataset:
    def __init__(
        self,
        images: Sequence[Path],
        label_directory: Path,
        image_size: int,
        scaleup: bool,
    ) -> None:
        self.images = list(images)
        self.label_directory = label_directory
        self.image_size = int(image_size)
        self.scaleup = bool(scaleup)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch as local_torch

        image_path = self.images[index]
        labels, _, _, _ = parse_yolo_label(
            self.label_directory / f"{image_path.stem}.txt"
        )
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            original_width, original_height = source.size
            scale = min(
                self.image_size / original_height,
                self.image_size / original_width,
            )
            if not self.scaleup:
                scale = min(scale, 1.0)
            resized_width = max(1, round(original_width * scale))
            resized_height = max(1, round(original_height * scale))
            if (resized_width, resized_height) != source.size:
                source = source.resize(
                    (resized_width, resized_height),
                    Image.Resampling.BILINEAR,
                )
            resized = np.asarray(source, dtype=np.uint8)

        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height
        left = round(pad_width / 2 - 0.1)
        top = round(pad_height / 2 - 0.1)
        canvas = np.full(
            (self.image_size, self.image_size, 3),
            114,
            dtype=np.uint8,
        )
        canvas[top : top + resized_height, left : left + resized_width] = resized

        transformed = labels.copy()
        if transformed.size:
            x_center = transformed[:, 1] * original_width
            y_center = transformed[:, 2] * original_height
            box_width = transformed[:, 3] * original_width
            box_height = transformed[:, 4] * original_height
            transformed[:, 1] = (x_center * scale + left) / self.image_size
            transformed[:, 2] = (y_center * scale + top) / self.image_size
            transformed[:, 3] = box_width * scale / self.image_size
            transformed[:, 4] = box_height * scale / self.image_size
            transformed[:, 1:] = np.clip(transformed[:, 1:], 0.0, 1.0)

        image = local_torch.from_numpy(canvas.copy()).permute(2, 0, 1)
        image = image.contiguous().float().div_(255.0)
        return image, local_torch.from_numpy(transformed), str(image_path)


def collate_boxes(batch):
    import torch as local_torch

    images, label_groups, paths = zip(*batch)
    batch_indices = []
    classes = []
    boxes = []
    for image_index, labels in enumerate(label_groups):
        if not len(labels):
            continue
        batch_indices.append(
            local_torch.full((len(labels),), image_index, dtype=local_torch.float32)
        )
        classes.append(labels[:, 0:1].float())
        boxes.append(labels[:, 1:5].float())
    if boxes:
        batch_idx = local_torch.cat(batch_indices)
        cls = local_torch.cat(classes)
        bboxes = local_torch.cat(boxes)
    else:
        batch_idx = local_torch.empty((0,), dtype=local_torch.float32)
        cls = local_torch.empty((0, 1), dtype=local_torch.float32)
        bboxes = local_torch.empty((0, 4), dtype=local_torch.float32)
    return {
        "img": local_torch.stack(images),
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        "paths": list(paths),
    }


def resolve_device(value: str):
    text = str(value).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if text.isdigit():
        text = f"cuda:{text}"
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 CUDA 设备，但 CUDA 不可用：{device}")
    return device


def import_local_ultralytics(root: Path) -> dict[str, Any]:
    package_root = str(root)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    import ultralytics
    from ultralytics import RTDETR, YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss

    imported_from = Path(ultralytics.__file__).resolve()
    try:
        imported_from.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"导入了错误的 ultralytics：{imported_from}；期望位于 {root}"
        ) from error
    return {
        "package": ultralytics,
        "RTDETR": RTDETR,
        "YOLO": YOLO,
        "get_cfg": get_cfg,
        "v8DetectionLoss": v8DetectionLoss,
    }


class UltralyticsGradientModel:
    def __init__(
        self,
        model_path: Path,
        task_override: Optional[str],
        device,
        modules: dict[str, Any],
    ) -> None:
        is_rtdetr_path = "rtdetr" in model_path.stem.lower().replace("-", "")
        if is_rtdetr_path:
            facade = modules["RTDETR"](str(model_path))
        else:
            facade = modules["YOLO"](
                str(model_path),
                task=task_override,
                verbose=False,
            )
        if not isinstance(facade.model, torch.nn.Module):
            raise RuntimeError("模型没有加载为原生 torch.nn.Module")
        self.model = facade.model.float().to(device)
        self.model.args = modules["get_cfg"](
            overrides=getattr(self.model, "args", {})
        )
        head = self.model.model[-1]
        if not hasattr(self.model, "nc") and hasattr(head, "nc"):
            self.model.nc = int(head.nc)
        self.task = str(facade.task)
        if self.task not in BOX_GUIDANCE_TASKS:
            raise ValueError(
                f"任务 {self.task!r} 不能使用五列框标签做梯度匹配；"
                f"支持：{sorted(BOX_GUIDANCE_TASKS)}"
            )
        self.is_rtdetr = "RTDETR" in type(self.model).__name__ or "RTDETR" in type(
            self.model.model[-1]
        ).__name__
        self.criterion = None
        if self.task != "detect" and not self.is_rtdetr:
            self.criterion = modules["v8DetectionLoss"](self.model)
            self.loss_kind = f"{self.task}_detection_component"
        else:
            self.loss_kind = "native_detection"

        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.names = {int(index): str(name) for index, name in names.items()}
        else:
            self.names = {index: str(name) for index, name in enumerate(names)}
        self.nc = len(self.names)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        prefix = f"model.{len(self.model.model) - 1}"
        self.selected_parameters = []
        for name, parameter in head.named_parameters():
            parameter.requires_grad_(True)
            self.selected_parameters.append(
                (f"{prefix}.{name}" if name else prefix, parameter)
            )
        if not self.selected_parameters:
            raise RuntimeError("最后预测头没有可用于梯度匹配的参数")
        self.model.eval()

    @staticmethod
    def detection_predictions(output: Any) -> dict[str, Any]:
        if isinstance(output, dict):
            if {"boxes", "scores", "feats"}.issubset(output):
                return output
            for name in ("one2many", "one2one"):
                if name in output:
                    try:
                        return UltralyticsGradientModel.detection_predictions(
                            output[name]
                        )
                    except TypeError:
                        pass
        elif isinstance(output, (tuple, list)):
            for value in reversed(output):
                try:
                    return UltralyticsGradientModel.detection_predictions(value)
                except TypeError:
                    continue
        raise TypeError("模型输出中没有找到共享检测分支 boxes/scores/feats")

    def loss(self, batch: dict[str, Any]):
        if self.criterion is not None:
            predictions = self.detection_predictions(self.model(batch["img"]))
            result = self.criterion(predictions, batch)
        else:
            model_batch = batch
            if hasattr(self.model, "txt_feats") and "txt_feats" not in batch:
                model_batch = dict(batch)
                text_features = self.model.txt_feats.to(
                    device=batch["img"].device,
                    dtype=batch["img"].dtype,
                )
                if text_features.shape[0] != batch["img"].shape[0]:
                    text_features = text_features.expand(
                        batch["img"].shape[0], -1, -1
                    )
                model_batch["txt_feats"] = text_features.detach()
            result = self.model.loss(model_batch)

        if isinstance(result, tuple):
            raw_loss = result[0]
            details = result[1] if len(result) > 1 else {}
        else:
            raw_loss = result
            details = {}
        if not isinstance(raw_loss, torch.Tensor):
            raise RuntimeError(f"不支持的损失返回类型：{type(raw_loss)}")
        total = raw_loss.float().sum()
        if isinstance(details, dict):
            components = {
                str(name): float(value.detach().float().mean().item())
                for name, value in details.items()
                if isinstance(value, torch.Tensor)
            }
        elif isinstance(details, torch.Tensor):
            components = {
                f"component_{index}": float(value)
                for index, value in enumerate(
                    details.detach().float().cpu().flatten().tolist()
                )
            }
        else:
            components = {}
        return total, components

    @property
    def parameter_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
            }
            for name, parameter in self.selected_parameters
        ]


def move_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        "img": batch["img"].to(device, non_blocking=True),
        "batch_idx": batch["batch_idx"].to(device, non_blocking=True),
        "cls": batch["cls"].to(device, non_blocking=True),
        "bboxes": batch["bboxes"].to(device, non_blocking=True),
        "paths": batch["paths"],
    }


def compute_loss_gradients(
    adapter: UltralyticsGradientModel,
    batch: dict[str, Any],
    device,
) -> tuple[dict[str, float], list[Any]]:
    batch = move_batch(batch, device)
    adapter.model.zero_grad(set_to_none=True)
    loss, components = adapter.loss(batch)
    parameters = [parameter for _, parameter in adapter.selected_parameters]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    detached = [
        torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        if gradient is None
        else gradient.detach().float().cpu()
        for parameter, gradient in zip(parameters, gradients)
    ]
    batch_size = int(batch["img"].shape[0])
    stats = {
        "total": float(loss.detach().item() / max(batch_size, 1)),
        **components,
    }
    adapter.model.zero_grad(set_to_none=True)
    return stats, detached


def add_gradients(
    accumulator: Optional[list[Any]],
    gradients: Sequence[Any],
    weight: float = 1.0,
) -> list[Any]:
    if accumulator is None:
        return [gradient.clone().mul_(weight) for gradient in gradients]
    for total, gradient in zip(accumulator, gradients):
        total.add_(gradient, alpha=weight)
    return accumulator


def scale_gradients(gradients: Sequence[Any], scale: float) -> list[Any]:
    return [gradient.clone().mul_(scale) for gradient in gradients]


def gradient_statistics(
    first: Sequence[Any],
    second: Sequence[Any],
    epsilon: float = 1e-12,
) -> dict[str, float]:
    dot = torch.zeros((), dtype=torch.float64)
    first_norm_sq = torch.zeros((), dtype=torch.float64)
    second_norm_sq = torch.zeros((), dtype=torch.float64)
    for first_value, second_value in zip(first, second):
        a = first_value.detach().double().cpu()
        b = second_value.detach().double().cpu()
        dot += (a * b).sum()
        first_norm_sq += a.square().sum()
        second_norm_sq += b.square().sum()
    first_norm = float(first_norm_sq.sqrt().item())
    second_norm = float(second_norm_sq.sqrt().item())
    dot_value = float(dot.item())
    denominator = first_norm * second_norm
    cosine = dot_value / (denominator + epsilon) if denominator > 0.0 else 0.0
    return {
        "dot": dot_value,
        "cosine": float(cosine),
        "norm_a": first_norm,
        "norm_b": second_norm,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_torch_save(path: Path, payload: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def make_loader(
    dataset: YoloBoxDataset,
    batch_size: int,
    workers: int,
    device,
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(workers, batch_size, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0 and len(dataset) > batch_size,
        collate_fn=collate_boxes,
    )


def compute_guide_gradient(
    adapter: UltralyticsGradientModel,
    loader,
    device,
) -> tuple[list[Any], dict[str, float]]:
    accumulator = None
    sample_count = 0
    metric_sums: dict[str, float] = defaultdict(float)
    for batch in tqdm(
        loader,
        total=len(loader),
        desc="Guide 参考梯度",
        unit="batch",
        dynamic_ncols=True,
    ):
        stats, gradients = compute_loss_gradients(adapter, batch, device)
        batch_size = int(batch["img"].shape[0])
        accumulator = add_gradients(accumulator, gradients)
        sample_count += batch_size
        for name, value in stats.items():
            metric_sums[name] += float(value) * batch_size
    if accumulator is None or sample_count == 0:
        raise RuntimeError("Guide 集没有可用于梯度计算的图像")
    return scale_gradients(accumulator, 1.0 / sample_count), {
        name: value / sample_count for name, value in metric_sums.items()
    }


def load_or_compute_guide(
    args: argparse.Namespace,
    paths: dict[str, Any],
    adapter: UltralyticsGradientModel,
    device,
    metadata: dict[str, Any],
) -> tuple[list[Any], dict[str, float]]:
    cache_path = paths["score_dir"] / "guide_gradients.pt"
    if args.resume and cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") == metadata:
            tqdm.write(f"复用 Guide 梯度缓存：{cache_path}")
            return payload["gradient"], payload.get("losses", {})
        tqdm.write("Guide 梯度缓存配置已变化，将重新计算")

    dataset = YoloBoxDataset(
        paths["guide_images"],
        paths["guide_labels"],
        args.imgsz,
        args.scaleup,
    )
    loader = make_loader(
        dataset,
        args.guide_batch_size,
        args.workers,
        device,
    )
    gradient, losses = compute_guide_gradient(adapter, loader, device)
    atomic_torch_save(
        cache_path,
        {
            "metadata": metadata,
            "gradient": gradient,
            "losses": losses,
        },
    )
    return gradient, losses


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"JSONL 解析失败：{path}:{line_number}") from error
            records.append(record)
    return records


def write_scores_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    preferred = [
        "sample_id",
        "image_path",
        "label_path",
        "model_task",
        "loss_kind",
        "cosine",
        "dot",
        "cosine_global",
        "dot_global",
        "generated_gradient_norm",
        "guide_gradient_norm",
        "loss_total",
    ]
    extra = sorted(set().union(*(record.keys() for record in records)) - set(preferred))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred + extra)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def score_source_images(
    args: argparse.Namespace,
    paths: dict[str, Any],
    adapter: UltralyticsGradientModel,
    guide_gradient: Sequence[Any],
    device,
    run_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    scores_path = paths["score_dir"] / "scores.jsonl"
    metadata_path = paths["score_dir"] / "run_metadata.json"
    errors_path = paths["score_dir"] / "errors.jsonl"

    if not args.resume:
        scores_path.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)
    elif scores_path.exists():
        if not metadata_path.is_file():
            raise RuntimeError("scores.jsonl 缺少 run_metadata.json；请使用 --no-resume")
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != run_metadata:
            raise RuntimeError("已有评分配置与当前运行不一致；请更换输出目录或使用 --no-resume")
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    records = read_jsonl(scores_path)
    expected_names = {path.name for path in paths["source_images"]}
    seen: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen or sample_id not in expected_names:
            raise RuntimeError(f"已有评分包含无效 sample_id：{sample_id!r}")
        seen.add(sample_id)
    pending_images = [
        path for path in paths["source_images"] if path.name not in seen
    ]
    if not pending_images:
        tqdm.write(f"复用完整 source 评分：{scores_path}")
        return records

    errors_path.unlink(missing_ok=True)
    dataset = YoloBoxDataset(
        pending_images,
        paths["scoring_labels"],
        args.imgsz,
        args.scaleup,
    )
    loader = make_loader(dataset, 1, args.workers, device)
    with (
        scores_path.open("a", encoding="utf-8") as score_handle,
        errors_path.open("w", encoding="utf-8") as error_handle,
        tqdm(
            total=paths["source_count"],
            initial=len(records),
            desc="Source 梯度匹配",
            unit="img",
            dynamic_ncols=True,
        ) as progress,
    ):
        for batch in loader:
            sample_path = Path(batch["paths"][0])
            try:
                loss_stats, gradient = compute_loss_gradients(
                    adapter,
                    batch,
                    device,
                )
                stats = gradient_statistics(gradient, guide_gradient)
                record = {
                    "sample_id": sample_path.name,
                    "image_path": str(sample_path.resolve()),
                    "label_path": str(
                        (
                            paths["scoring_labels"]
                            / f"{sample_path.stem}.txt"
                        ).resolve()
                    ),
                    "model_task": adapter.task,
                    "loss_kind": adapter.loss_kind,
                    "reference_mode": "global",
                    "cosine": stats["cosine"],
                    "dot": stats["dot"],
                    "cosine_global": stats["cosine"],
                    "dot_global": stats["dot"],
                    "generated_gradient_norm": stats["norm_a"],
                    "guide_gradient_norm": stats["norm_b"],
                    "loss_total": loss_stats.pop("total"),
                    **{f"loss_{name}": value for name, value in loss_stats.items()},
                }
                score_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                score_handle.flush()
                records.append(record)
                progress.set_postfix(
                    cosine=f"{stats['cosine']:.4f}",
                    loss=f"{record['loss_total']:.4f}",
                    refresh=False,
                )
            except Exception as error:
                failure = {
                    "sample_id": sample_path.name,
                    "image_path": str(sample_path),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                error_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                error_handle.flush()
            progress.update(1)

    error_records = read_jsonl(errors_path)
    if error_records:
        raise RuntimeError(
            f"有 {len(error_records)} 张图像评分失败，请检查：{errors_path}"
        )
    if len(records) != paths["source_count"]:
        raise RuntimeError(
            f"评分数量异常：期望 {paths['source_count']}，实际 {len(records)}"
        )
    records.sort(key=lambda record: str(record["sample_id"]))
    write_scores_csv(paths["score_dir"] / "scores.csv", records)
    return records


def split_records(
    records: Sequence[dict[str, Any]],
    zero_policy: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if zero_policy != "separate":
        raise ValueError("zero_policy 必须为 separate")
    positive = []
    negative = []
    zero = []
    scores = []
    for record in records:
        cosine = float(record["cosine_global"])
        if not math.isfinite(cosine):
            raise RuntimeError(f"余弦值不是有限数：{record['sample_id']}")
        scores.append(cosine)
        if cosine > 0:
            positive.append(record)
        elif cosine < 0:
            negative.append(record)
        else:
            zero.append(record)
    return positive, negative, zero, {
        "scored_count": len(records),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "zero_count": len(zero),
        "zero_policy": zero_policy,
        "cosine_mean": float(np.mean(scores)),
        "cosine_min": float(np.min(scores)),
        "cosine_max": float(np.max(scores)),
    }


def copy_one(source: Path, destination: Path) -> str:
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return "skipped"
        raise FileExistsError(f"目标文件已存在且大小不同：{destination}")
    shutil.copy2(source, destination)
    return "copied"


def copy_records(
    records: Sequence[dict[str, Any]],
    source_key: str,
    destination: Path,
    workers: int,
    description: str,
) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    pairs = [
        (Path(str(record[source_key])), destination / Path(str(record[source_key])).name)
        for record in records
    ]
    expected = {target.name for _, target in pairs}
    unexpected = sorted(path.name for path in destination.iterdir() if path.name not in expected)
    if unexpected:
        raise RuntimeError(
            f"目标目录混有旧文件：{destination}；示例={unexpected[:5]}"
        )
    counts = {"copied": 0, "skipped": 0}
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        tqdm(total=len(pairs), desc=description, unit="file", dynamic_ncols=True) as progress,
    ):
        futures = {
            executor.submit(copy_one, source, target): target
            for source, target in pairs
        }
        for future in as_completed(futures):
            counts[future.result()] += 1
            progress.update(1)
    return counts


def materialize_subsets(
    args: argparse.Namespace,
    paths: dict[str, Any],
    positive: Sequence[dict[str, Any]],
    negative: Sequence[dict[str, Any]],
    zero: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    output_root = paths["output_root"]
    directories = {
        "positive_images": output_root / "images" / args.positive_name,
        "negative_images": output_root / "images" / args.negative_name,
        "zero_images": output_root / "images" / args.zero_name,
        "positive_labels": output_root / "labels" / args.positive_name,
        "negative_labels": output_root / "labels" / args.negative_name,
        "zero_labels": output_root / "labels" / args.zero_name,
    }
    copy_statistics = {
        "positive_images": copy_records(
            positive,
            "image_path",
            directories["positive_images"],
            args.copy_workers,
            "复制正集图像",
        ),
        "negative_images": copy_records(
            negative,
            "image_path",
            directories["negative_images"],
            args.copy_workers,
            "复制负集图像",
        ),
        "zero_images": copy_records(
            zero,
            "image_path",
            directories["zero_images"],
            args.copy_workers,
            "复制零值集图像",
        ),
        "positive_labels": copy_records(
            positive,
            "label_path",
            directories["positive_labels"],
            args.copy_workers,
            "复制正集标签",
        ),
        "negative_labels": copy_records(
            negative,
            "label_path",
            directories["negative_labels"],
            args.copy_workers,
            "复制负集标签",
        ),
        "zero_labels": copy_records(
            zero,
            "label_path",
            directories["zero_labels"],
            args.copy_workers,
            "复制零值集标签",
        ),
    }
    return {
        "directories": {name: str(path) for name, path in directories.items()},
        "copy_statistics": copy_statistics,
    }


def write_outputs(
    args: argparse.Namespace,
    paths: dict[str, Any],
    positive: Sequence[dict[str, Any]],
    negative: Sequence[dict[str, Any]],
    zero: Sequence[dict[str, Any]],
    statistics: dict[str, Any],
    materialized: dict[str, Any],
    model_metadata: dict[str, Any],
) -> None:
    output_root = paths["output_root"]
    for name, records, directory_key in (
        (args.positive_name, positive, "positive_images"),
        (args.negative_name, negative, "negative_images"),
        (args.zero_name, zero, "zero_images"),
    ):
        directory = Path(materialized["directories"][directory_key])
        (output_root / f"{name}.txt").write_text(
            "\n".join(
                str((directory / Path(str(record["image_path"])).name).resolve())
                for record in records
            )
            + ("\n" if records else ""),
            encoding="utf-8",
        )
    summary = {
        **statistics,
        "score_field": "cosine_global",
        "model": str(paths["model"]),
        "source_root": str(paths["source_root"]),
        "source_images": str(paths["source_image_dir"]),
        "source_labels": str(paths["source_labels"]),
        "guide_root": str(paths["guide_root"]),
        "guide_images": str(paths["guide_image_dir"]),
        "guide_labels": str(paths["guide_labels"]),
        "scoring_labels": str(paths["scoring_labels"]),
        "scores_jsonl": str(paths["score_dir"] / "scores.jsonl"),
        "model_metadata": model_metadata,
        **materialized,
    }
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    paths = validate_inputs(args)
    scores_path = paths["score_dir"] / "scores.jsonl"
    model_metadata: dict[str, Any] = {}

    if args.skip_gradient_scoring:
        require_file(scores_path, "已有梯度评分")
        records = read_jsonl(scores_path)
        if len(records) != paths["source_count"]:
            raise RuntimeError(
                f"评分数量异常：期望 {paths['source_count']}，实际 {len(records)}"
            )
        metadata_path = paths["score_dir"] / "run_metadata.json"
        if metadata_path.is_file():
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_metadata = saved_metadata.get("model_metadata", {})
    else:
        if str(args.device).lower() != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        matplotlib_config = paths["score_dir"] / ".matplotlib"
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
        global torch
        import torch as torch_module

        torch = torch_module
        device = resolve_device(args.device)
        modules = import_local_ultralytics(paths["ultralytics_root"])
        adapter = UltralyticsGradientModel(
            paths["model"],
            None if args.task == "auto" else args.task,
            device,
            modules,
        )
        class_ids = [
            int(value)
            for images, label_directory in (
                (paths["source_images"], paths["scoring_labels"]),
                (paths["guide_images"], paths["guide_labels"]),
            )
            for image in images
            for value in parse_yolo_label(
                label_directory / f"{image.stem}.txt"
            )[0][:, 0]
        ]
        maximum_class = max(class_ids, default=-1)
        if maximum_class >= adapter.nc:
            raise RuntimeError(
                f"标签最大类别 ID={maximum_class}，但模型只有 {adapter.nc} 类"
            )

        model_hash = file_sha256(paths["model"])
        guide_metadata = {
            "schema_version": 1,
            "model_hash": model_hash,
            "task": adapter.task,
            "loss_kind": adapter.loss_kind,
            "parameters": adapter.parameter_metadata,
            "guide_paths_hash": json_sha256([str(path) for path in paths["guide_images"]]),
            "guide_labels": str(paths["guide_labels"]),
            "image_size": args.imgsz,
            "scaleup": args.scaleup,
        }
        guide_gradient, guide_losses = load_or_compute_guide(
            args,
            paths,
            adapter,
            device,
            guide_metadata,
        )
        model_metadata = {
            "ultralytics_version": modules["package"].__version__,
            "task": adapter.task,
            "loss_kind": adapter.loss_kind,
            "class_count": adapter.nc,
            "selected_parameter_tensors": len(adapter.selected_parameters),
            "selected_parameter_count": sum(
                item["numel"] for item in adapter.parameter_metadata
            ),
            "guide_losses": guide_losses,
            "device": str(device),
            "physical_cuda_devices": str(args.cuda_visible_devices),
        }
        run_metadata = {
            **guide_metadata,
            "source_paths_hash": json_sha256(
                [str(path) for path in paths["source_images"]]
            ),
            "source_labels": str(paths["scoring_labels"]),
            "reference_mode": "global",
            "model_metadata": model_metadata,
        }
        tqdm.write(
            f"模型任务={adapter.task}，损失={adapter.loss_kind}，"
            f"物理 GPU={args.cuda_visible_devices}，逻辑设备={device}"
        )
        records = score_source_images(
            args,
            paths,
            adapter,
            guide_gradient,
            device,
            run_metadata,
        )

    positive, negative, zero, statistics = split_records(records, args.zero_policy)
    materialized = materialize_subsets(
        args,
        paths,
        positive,
        negative,
        zero,
    )
    write_outputs(
        args,
        paths,
        positive,
        negative,
        zero,
        statistics,
        materialized,
        model_metadata,
    )
    tqdm.write(
        f"完成：scored={statistics['scored_count']}，"
        f"cos>0={statistics['positive_count']}，"
        f"cos<0={statistics['negative_count']}，"
        f"cos==0={statistics['zero_count']}，"
        f"summary={paths['output_root'] / 'split_summary.json'}"
    )


if __name__ == "__main__":
    main()
