from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GUIDE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GradientCosine:
    cosine: torch.Tensor
    dot: torch.Tensor
    generated_norm: torch.Tensor
    guide_norm: torch.Tensor


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paths_content_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        path = path.resolve()
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def image_files(directory: str | Path) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Guide image directory not found: {directory}")
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"Guide image directory is empty: {directory}")
    if len({path.stem for path in images}) != len(images):
        raise RuntimeError(
            f"Guide images contain duplicate stems with different suffixes: {directory}"
        )
    return images


def parse_yolo_detection_label(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Guide label not found: {path}")
    rows: list[list[float]] = []
    seen: set[bytes] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) == 5:
            values = [float(value) for value in fields]
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
        else:
            raise ValueError(
                f"Prompt gradient alignment requires a five-column YOLO box or polygon: "
                f"{path}:{line_number} has {len(fields)} fields"
            )
        class_id = int(values[0])
        if values[0] != class_id or class_id < 0:
            raise ValueError(f"Invalid class id at {path}:{line_number}: {values[0]}")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite label value at {path}:{line_number}")
        if any(value < 0.0 or value > 1.0 for value in values[1:]):
            raise ValueError(f"Box is not normalized to [0,1] at {path}:{line_number}")
        key = struct.pack("!5f", *values)
        if key not in seen:
            rows.append(values)
            seen.add(key)
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)


def differentiable_letterbox(
    image: torch.Tensor,
    targets: torch.Tensor,
    image_size: int,
    scaleup: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Letterbox an image tensor and normalized targets using the legacy scoring rules."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image [B,3,H,W], got {tuple(image.shape)}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError(f"Expected targets [N,6], got {tuple(targets.shape)}")

    batch_size, _, original_height, original_width = image.shape
    if targets.numel():
        batch_indices = targets[:, 0]
        if (batch_indices < 0).any() or (batch_indices >= batch_size).any():
            raise ValueError("Target batch indices are outside the image batch")
        if (targets[:, 2:] < 0).any() or (targets[:, 2:] > 1).any():
            raise ValueError("Target boxes must be normalized to [0,1]")

    scale = min(image_size / original_height, image_size / original_width)
    if not scaleup:
        scale = min(scale, 1.0)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = image
    if (resized_height, resized_width) != (original_height, original_width):
        resized = F.interpolate(
            image,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

    pad_width = image_size - resized_width
    pad_height = image_size - resized_height
    left = round(pad_width / 2 - 0.1)
    top = round(pad_height / 2 - 0.1)
    right = pad_width - left
    bottom = pad_height - top
    letterboxed = F.pad(
        resized,
        (left, right, top, bottom),
        mode="constant",
        value=114.0 / 255.0,
    ).clamp(0, 1)

    transformed = targets.clone()
    if transformed.numel():
        transformed[:, 2] = (
            transformed[:, 2] * original_width * scale + left
        ) / image_size
        transformed[:, 3] = (
            transformed[:, 3] * original_height * scale + top
        ) / image_size
        transformed[:, 4] = transformed[:, 4] * original_width * scale / image_size
        transformed[:, 5] = transformed[:, 5] * original_height * scale / image_size
        transformed[:, 2:] = transformed[:, 2:].clamp(0, 1)
    return letterboxed, transformed


class GuideDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_directory: str | Path,
        label_directory: str | Path,
        image_size: int,
        scaleup: bool,
        max_images: int | None = None,
    ) -> None:
        self.image_directory = Path(image_directory).expanduser().resolve()
        self.label_directory = Path(label_directory).expanduser().resolve()
        if not self.label_directory.is_dir():
            raise FileNotFoundError(
                f"Guide label directory not found: {self.label_directory}"
            )
        self.images = image_files(self.image_directory)
        if max_images is not None:
            if max_images <= 0:
                raise ValueError("max_images must be positive when provided")
            self.images = self.images[:max_images]
        self.labels = [self.label_directory / f"{path.stem}.txt" for path in self.images]
        for label in self.labels:
            if not label.is_file():
                raise FileNotFoundError(f"Guide label not found: {label}")
        self.image_size = int(image_size)
        self.scaleup = bool(scaleup)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        labels = parse_yolo_detection_label(self.labels[index])
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
                    (resized_width, resized_height), Image.Resampling.BILINEAR
                )
            resized = np.asarray(source, dtype=np.uint8)

        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height
        left = round(pad_width / 2 - 0.1)
        top = round(pad_height / 2 - 0.1)
        canvas = np.full(
            (self.image_size, self.image_size, 3), 114, dtype=np.uint8
        )
        canvas[top : top + resized_height, left : left + resized_width] = resized

        transformed = labels.clone()
        if transformed.numel():
            transformed[:, 1] = (
                transformed[:, 1] * original_width * scale + left
            ) / self.image_size
            transformed[:, 2] = (
                transformed[:, 2] * original_height * scale + top
            ) / self.image_size
            transformed[:, 3] = (
                transformed[:, 3] * original_width * scale / self.image_size
            )
            transformed[:, 4] = (
                transformed[:, 4] * original_height * scale / self.image_size
            )
            transformed[:, 1:] = transformed[:, 1:].clamp(0, 1)
        image = torch.from_numpy(canvas.copy()).permute(2, 0, 1)
        image = image.contiguous().float().div_(255.0)
        return image, transformed, str(image_path)


def collate_guide_batch(batch):
    images, label_groups, paths = zip(*batch)
    targets = []
    for batch_index, labels in enumerate(label_groups):
        if not len(labels):
            continue
        targets.append(
            torch.cat(
                (
                    torch.full(
                        (len(labels), 1), float(batch_index), dtype=torch.float32
                    ),
                    labels.float(),
                ),
                dim=1,
            )
        )
    merged = (
        torch.cat(targets, dim=0)
        if targets
        else torch.empty((0, 6), dtype=torch.float32)
    )
    return torch.stack(images), merged, list(paths)


def build_guide_metadata(
    detector,
    dataset: GuideDetectionDataset,
    dataset_yaml: str | Path | None,
    class_names: Sequence[str],
    class_mapping: Sequence[dict[str, Any]],
    image_size: int,
    scaleup: bool,
    guide_batch_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": GUIDE_CACHE_SCHEMA_VERSION,
        "checkpoint": str(detector.weights),
        "checkpoint_sha256": file_sha256(detector.weights),
        "ultralytics_version": detector.version,
        "task": "detect",
        "loss_kind": "native_detection",
        "parameter_scope": "detection_head",
        "parameters": detector.alignment_parameter_metadata(),
        "dataset_yaml": str(Path(dataset_yaml).resolve()) if dataset_yaml else None,
        "guide_images": str(dataset.image_directory),
        "guide_labels": str(dataset.label_directory),
        "guide_image_count": len(dataset.images),
        "guide_images_content_sha256": paths_content_sha256(dataset.images),
        "guide_labels_content_sha256": paths_content_sha256(dataset.labels),
        "image_size": int(image_size),
        "scaleup": bool(scaleup),
        "guide_batch_size": int(guide_batch_size),
        "class_names": list(class_names),
        "class_mapping": list(class_mapping),
    }


def _validate_gradient_list(
    gradients: Sequence[torch.Tensor], parameters: Sequence[torch.nn.Parameter], label: str
) -> None:
    if len(gradients) != len(parameters):
        raise RuntimeError(
            f"{label} gradient count {len(gradients)} does not match parameter count {len(parameters)}"
        )
    for index, (gradient, parameter) in enumerate(zip(gradients, parameters)):
        if tuple(gradient.shape) != tuple(parameter.shape):
            raise RuntimeError(
                f"{label} gradient shape mismatch at index {index}: "
                f"{tuple(gradient.shape)} != {tuple(parameter.shape)}"
            )
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"{label} gradient contains NaN or Inf at index {index}")


def compute_guide_gradients(
    detector,
    dataset: GuideDetectionDataset,
    batch_size: int,
    workers: int,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    if batch_size <= 0:
        raise ValueError("guide batch size must be positive")
    if workers < 0:
        raise ValueError("guide workers must be non-negative")
    detector.enable_alignment_parameters()
    parameters = detector.alignment_parameters()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(workers, batch_size, os.cpu_count() or 1),
        pin_memory=detector.device.type == "cuda",
        persistent_workers=workers > 0 and len(dataset) > batch_size,
        collate_fn=collate_guide_batch,
    )
    accumulator = [
        torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for parameter in parameters
    ]
    sample_count = 0
    total_loss = 0.0
    for images, targets, _ in loader:
        images = images.to(detector.device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(detector.device, dtype=torch.float32, non_blocking=True)
        if targets.numel() and int(targets[:, 1].max().item()) >= len(detector.names):
            raise RuntimeError(
                f"Guide label class id {int(targets[:, 1].max().item())} exceeds "
                f"detector class count {len(detector.names)}"
            )
        detector.clear_parameter_gradients()
        loss = detector.alignment_loss(
            images, targets, image_size=detector.image_size, scaleup=False
        ).total
        gradients = torch.autograd.grad(
            loss,
            parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        for total, parameter, gradient in zip(accumulator, parameters, gradients):
            value = (
                torch.zeros_like(parameter, dtype=torch.float32)
                if gradient is None
                else gradient.detach().float()
            )
            total.add_(value.cpu())
        current_batch = int(images.shape[0])
        sample_count += current_batch
        total_loss += float(loss.detach().item())
        detector.clear_parameter_gradients()
    if sample_count == 0:
        raise RuntimeError("Guide set has no images")
    gradients = [value.div_(sample_count) for value in accumulator]
    _validate_gradient_list(gradients, parameters, "Guide")
    norm = torch.sqrt(sum(value.square().sum() for value in gradients))
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        raise RuntimeError(f"Guide gradient norm must be finite and positive, got {float(norm)}")
    return gradients, {
        "sample_count": float(sample_count),
        "loss_total_mean": total_loss / sample_count,
        "gradient_norm": float(norm),
    }


def load_or_compute_guide_gradients(
    detector,
    image_directory: str | Path,
    label_directory: str | Path,
    cache_directory: str | Path,
    dataset_yaml: str | Path | None,
    class_names: Sequence[str],
    class_mapping: Sequence[dict[str, Any]],
    image_size: int,
    scaleup: bool,
    batch_size: int,
    workers: int,
    use_cache: bool = True,
    max_images: int | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    dataset = GuideDetectionDataset(
        image_directory,
        label_directory,
        image_size=image_size,
        scaleup=scaleup,
        max_images=max_images,
    )
    detector.enable_alignment_parameters()
    metadata = build_guide_metadata(
        detector,
        dataset,
        dataset_yaml,
        class_names,
        class_mapping,
        image_size,
        scaleup,
        batch_size,
    )
    metadata_hash = json_sha256(metadata)
    cache_directory = Path(cache_directory).expanduser().resolve()
    cache_path = cache_directory / "guide_gradients.pt"
    metadata_path = cache_directory / "guide_metadata.json"
    parameters = detector.alignment_parameters()
    gradients: list[torch.Tensor] | None = None
    stats: dict[str, Any] = {}
    cache_hit = False
    if use_cache and cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict) and payload.get("metadata") == metadata:
            candidate = payload.get("gradients")
            if isinstance(candidate, list):
                _validate_gradient_list(candidate, parameters, "Cached Guide")
                gradients = [value.detach().float().cpu() for value in candidate]
                stats = dict(payload.get("statistics", {}))
                cache_hit = True
    if gradients is None:
        gradients, stats = compute_guide_gradients(
            detector, dataset, batch_size=batch_size, workers=workers
        )
        if use_cache:
            _atomic_torch_save(
                cache_path,
                {
                    "metadata": metadata,
                    "metadata_hash": metadata_hash,
                    "gradients": gradients,
                    "statistics": stats,
                },
            )
    sidecar = {
        "metadata": metadata,
        "metadata_hash": metadata_hash,
        "cache_path": str(cache_path),
        "cache_hit": cache_hit,
        "statistics": stats,
    }
    _atomic_json_save(metadata_path, sidecar)
    device_gradients = [value.to(detector.device, dtype=torch.float32) for value in gradients]
    return device_gradients, sidecar


def global_gradient_cosine(
    generated: Sequence[torch.Tensor | None],
    guide: Sequence[torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    epsilon: float = 1e-12,
) -> GradientCosine:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (len(generated) == len(guide) == len(parameters)):
        raise ValueError("Generated, Guide, and parameter lists must have the same length")
    dot = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    generated_sq = torch.zeros_like(dot)
    guide_sq = torch.zeros_like(dot)
    for gradient, reference, parameter in zip(generated, guide, parameters):
        current = torch.zeros_like(parameter, dtype=torch.float32) if gradient is None else gradient.float()
        reference = reference.to(device=current.device, dtype=torch.float32)
        if not torch.isfinite(current).all():
            raise FloatingPointError("Generated gradient contains NaN or Inf")
        if not torch.isfinite(reference).all():
            raise FloatingPointError("Guide gradient contains NaN or Inf")
        dot = dot + (current * reference).sum()
        generated_sq = generated_sq + current.square().sum()
        guide_sq = guide_sq + reference.square().sum()
    generated_norm = generated_sq.sqrt()
    guide_norm = guide_sq.sqrt()
    if not torch.isfinite(generated_norm) or float(generated_norm.detach()) <= 0.0:
        raise RuntimeError(
            f"Generated gradient norm must be finite and positive, got {float(generated_norm.detach())}"
        )
    if not torch.isfinite(guide_norm) or float(guide_norm.detach()) <= 0.0:
        raise RuntimeError(
            f"Guide gradient norm must be finite and positive, got {float(guide_norm.detach())}"
        )
    cosine = dot / (generated_norm * guide_norm + epsilon)
    if not torch.isfinite(cosine):
        raise FloatingPointError("Gradient cosine is NaN or Inf")
    return GradientCosine(cosine, dot, generated_norm, guide_norm)


def exact_alignment_cosine(
    detector,
    image: torch.Tensor,
    targets: torch.Tensor,
    guide_gradients: Sequence[torch.Tensor],
    image_size: int,
    scaleup: bool,
    epsilon: float,
) -> tuple[Any, GradientCosine]:
    detector.enable_alignment_parameters()
    detector.clear_parameter_gradients()
    parameters = detector.alignment_parameters()
    loss_result = detector.alignment_loss(
        image, targets, image_size=image_size, scaleup=scaleup
    )
    generated = torch.autograd.grad(
        loss_result.total,
        parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    cosine = global_gradient_cosine(
        generated, guide_gradients, parameters, epsilon=epsilon
    )
    return loss_result, cosine


def build_layout_target(
    dataset,
    sample: dict[str, Any],
    device: torch.device,
    max_objects: int,
    enable_filter: bool,
    min_bbox_area_ratio: float,
) -> torch.Tensor:
    if max_objects <= 0:
        raise ValueError("max_objects must be positive")
    image_id = int(sample["image_id"])
    info = dataset.images[image_id]
    image_width = float(info["width"])
    image_height = float(info["height"])
    image_area = max(image_width * image_height, 1.0)
    annotations = list(dataset.annotations.get(image_id, []))

    annotation_id = sample.get("annotation_id")
    active_index = None
    if annotation_id is not None:
        active_index = next(
            (
                index
                for index, ann in enumerate(annotations)
                if ann.get("id") is not None and int(ann["id"]) == int(annotation_id)
            ),
            None,
        )
    if active_index is None:
        expected_bbox = [float(value) for value in sample["bbox"]]
        active_index = next(
            (
                index
                for index, ann in enumerate(annotations)
                if int(ann["class_id"]) == int(sample["class_id"])
                and all(
                    abs(float(actual) - expected) <= 1e-3
                    for actual, expected in zip(ann["bbox"], expected_bbox)
                )
            ),
            None,
        )
    if active_index is not None:
        annotations = [annotations[active_index]] + [
            ann for index, ann in enumerate(annotations) if index != active_index
        ]
    else:
        annotations.insert(
            0,
            {
                "class_id": int(sample["class_id"]),
                "bbox": [float(value) for value in sample["bbox"]],
            },
        )

    rows = []
    for ann in annotations:
        x, y, width, height = [float(value) for value in ann["bbox"]]
        ratio = max(width, 0.0) * max(height, 0.0) / image_area
        if enable_filter and ratio < min_bbox_area_ratio:
            continue
        rows.append(
            [
                0.0,
                float(ann["class_id"]),
                (x + width / 2) / image_width,
                (y + height / 2) / image_height,
                width / image_width,
                height / image_height,
            ]
        )
        if len(rows) >= max_objects:
            break
    target = torch.tensor(rows, device=device, dtype=torch.float32).reshape(-1, 6)
    if target.numel():
        target[:, 2:] = target[:, 2:].clamp(0, 1)
    return target


class AlignmentRunningStatistics:
    def __init__(self, saved: dict[str, Any] | None = None) -> None:
        self._groups: dict[str, dict[str, float]] = {}
        for group, summary in (saved or {}).get("groups", {}).items():
            if not isinstance(summary, dict) or int(summary.get("count", 0)) <= 0:
                continue
            count = int(summary["count"])
            self._groups[str(group)] = {
                "count": float(count),
                "sum": float(summary["mean"]) * count,
                "min": float(summary["min"]),
                "max": float(summary["max"]),
                "positive": float(summary["positive_rate"]) * count,
            }

    def update(self, group: str, cosine: float) -> None:
        if not math.isfinite(cosine):
            raise FloatingPointError(f"Cannot record non-finite cosine: {cosine}")
        group = str(group)
        current = self._groups.setdefault(
            group,
            {
                "count": 0.0,
                "sum": 0.0,
                "min": float(cosine),
                "max": float(cosine),
                "positive": 0.0,
            },
        )
        current["count"] += 1.0
        current["sum"] += float(cosine)
        current["min"] = min(current["min"], float(cosine))
        current["max"] = max(current["max"], float(cosine))
        current["positive"] += float(cosine > 0.0)

    @staticmethod
    def _summary(values: dict[str, float]) -> dict[str, Any]:
        count = int(values["count"])
        return {
            "count": count,
            "mean": values["sum"] / count,
            "min": values["min"],
            "max": values["max"],
            "positive_rate": values["positive"] / count,
        }

    def to_dict(self) -> dict[str, Any]:
        overall = {
            "count": sum(value["count"] for value in self._groups.values()),
            "sum": sum(value["sum"] for value in self._groups.values()),
            "min": min((value["min"] for value in self._groups.values()), default=0.0),
            "max": max((value["max"] for value in self._groups.values()), default=0.0),
            "positive": sum(value["positive"] for value in self._groups.values()),
        }
        return {
            "overall": self._summary(overall) if overall["count"] else None,
            "groups": {
                group: self._summary(values)
                for group, values in sorted(self._groups.items())
            },
        }
