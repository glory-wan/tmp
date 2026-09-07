from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import torch
import torch.nn.functional as F


SUPPORTED_ULTRALYTICS_VERSION = "8.4.115"
SUPPORTED_FAMILIES = {"yolo", "rtdetr"}


@dataclass(frozen=True)
class DetectionPrediction:
    """Model-independent detection prediction in original-image pixel coordinates."""

    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
    top2_gap: float | None = None
    class_scores: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionLoss:
    """Differentiable scalar detection loss and detached component values."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]


def normalize_family(value: str) -> str:
    family = str(value).strip().lower().replace("-", "")
    if family == "rtdetr":
        return family
    if family == "yolo":
        return family
    raise ValueError(f"Unsupported detection model family {value!r}; expected one of {sorted(SUPPORTED_FAMILIES)}")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_local_ultralytics(
    source_root: str | Path,
    expected_version: str = SUPPORTED_ULTRALYTICS_VERSION,
) -> ModuleType:
    """Import the vendored Ultralytics package and reject a different installed copy."""

    root = Path(source_root).expanduser().resolve()
    package_init = root / "ultralytics" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"Ultralytics source root is invalid; expected {package_init}")

    loaded = sys.modules.get("ultralytics")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded is not None and loaded_file and not _is_under(Path(loaded_file), root):
        raise RuntimeError(
            f"A different Ultralytics package is already imported from {loaded_file}; expected source under {root}"
        )
    if loaded is not None and not loaded_file:
        for name in [name for name in sys.modules if name == "ultralytics" or name.startswith("ultralytics.")]:
            sys.modules.pop(name, None)

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_CONFIG_DIR", os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
    module = importlib.import_module("ultralytics")
    origin = Path(module.__file__).resolve()
    if not _is_under(origin, root):
        raise RuntimeError(f"Imported Ultralytics from {origin}, expected source under {root}")
    version = str(getattr(module, "__version__", "unknown"))
    if expected_version and version != expected_version:
        raise RuntimeError(
            f"Unsupported Ultralytics version {version}; this adapter is validated against {expected_version}"
        )
    return module


def load_ultralytics_model(
    family: str,
    weights: str | Path,
    source_root: str | Path,
    expected_version: str = SUPPORTED_ULTRALYTICS_VERSION,
):
    """Load and validate a local Ultralytics PyTorch detection model."""

    family = normalize_family(family)
    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Detection model weights not found: {weights_path}")
    if weights_path.suffix.lower() != ".pt":
        raise ValueError(f"Differentiable training requires a PyTorch .pt checkpoint, got: {weights_path}")

    ultralytics = load_local_ultralytics(source_root, expected_version=expected_version)
    facade = (
        ultralytics.RTDETR(str(weights_path))
        if family == "rtdetr"
        else ultralytics.YOLO(str(weights_path), task="detect")
    )
    if str(getattr(facade, "task", "")).lower() != "detect":
        raise ValueError(
            f"Checkpoint {weights_path} is task={getattr(facade, 'task', None)!r}; only detection checkpoints are supported"
        )
    actual_name = facade.__class__.__name__.lower()
    actual_rtdetr = actual_name == "rtdetr"
    if family == "rtdetr" and not actual_rtdetr:
        raise ValueError(f"Configured family=rtdetr but checkpoint loaded as {facade.__class__.__name__}")
    if family == "yolo" and actual_name != "yolo":
        raise ValueError(
            f"Configured family=yolo but checkpoint loaded as {facade.__class__.__name__}; "
            "only standard Ultralytics YOLO detection models are supported"
        )
    if not isinstance(facade.model, torch.nn.Module):
        raise TypeError("Only native Ultralytics PyTorch checkpoints support differentiable detector feedback")
    return facade, ultralytics


def _torch_device(value: str | int | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        device = value
    else:
        text = str(value).strip().lower()
        if text in {"", "none"}:
            text = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif text.isdigit():
            text = f"cuda:{text}"
        device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def _normalize_names(names: dict | list | tuple | None) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(value) for i, value in enumerate(names)}
    return {}


class UltralyticsDetector:
    """Unified YOLO/RT-DETR inference and differentiable-loss adapter."""

    supports_top2_gap = False

    def __init__(
        self,
        family: str,
        weights: str | Path,
        source_root: str | Path,
        device: str | int | torch.device = "0",
        image_size: int = 640,
        iou: float = 0.7,
        max_det: int = 300,
        expected_version: str = SUPPORTED_ULTRALYTICS_VERSION,
    ):
        self.family = normalize_family(family)
        self.weights = Path(weights).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve()
        self.device = _torch_device(device)
        self.image_size = int(image_size)
        self.iou = float(iou)
        self.max_det = int(max_det)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.facade, self.ultralytics = load_ultralytics_model(
            self.family,
            self.weights,
            self.source_root,
            expected_version=expected_version,
        )
        self.model = self.facade.model.to(self.device)
        self.model.requires_grad_(False)
        self.names = _normalize_names(getattr(self.model, "names", None))
        self.version = str(self.ultralytics.__version__)
        self._selected_alignment_parameters: list[tuple[str, torch.nn.Parameter]] = []
        self._configure_loss_args()

    def _configure_loss_args(self) -> None:
        """Attach the default training hyperparameters expected by Ultralytics loss classes."""

        get_cfg = importlib.import_module("ultralytics.cfg").get_cfg
        current = getattr(self.model, "args", {})
        current = current if isinstance(current, dict) else vars(current)
        # Keep the checkpoint training/loss hyperparameters. This mirrors
        # build_coco_cos_subsets_ultralytics.py, which is the numerical oracle
        # for Prompt gradient alignment.
        overrides = dict(current)
        overrides.update({"task": "detect", "imgsz": self.image_size, "device": str(self.device)})
        self.model.args = get_cfg(overrides=overrides)
        head = self.model.model[-1]
        self.model.nc = int(getattr(head, "nc", len(self.names)))
        if self.family == "rtdetr" and hasattr(self.model, "criterion"):
            delattr(self.model, "criterion")
        elif self.family == "yolo":
            self.model.criterion = None
        if hasattr(self.model, "set_head_attr"):
            self.model.set_head_attr(max_det=self.max_det)

    def validate_dataset_names(self, names: list[str]) -> None:
        """Reject incompatible class spaces before mining detector failures."""

        if not self.names:
            return
        model_names = [self.names[index] for index in sorted(self.names)]
        if len(model_names) != len(names):
            raise ValueError(
                f"Dataset has {len(names)} classes but checkpoint has {len(model_names)} classes. "
                "Run warmup/fine-tuning with the target dataset before failure mining."
            )
        mismatches = [
            (index, model_name, str(names[index]))
            for index, model_name in enumerate(model_names)
            if model_name.strip().lower() != str(names[index]).strip().lower()
        ]
        if mismatches:
            preview = ", ".join(f"{i}:{model!r}!={data!r}" for i, model, data in mismatches[:5])
            raise ValueError(f"Checkpoint and dataset class names differ: {preview}")

    def predict(self, image, conf: float = 0.001) -> list[dict[str, Any]]:
        """Run public Ultralytics inference and return the legacy-compatible dictionary schema."""

        results = self.facade.predict(
            source=image,
            conf=float(conf),
            iou=self.iou,
            imgsz=self.image_size,
            device=str(self.device),
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu()
        xywh = torch.cat((xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]), dim=1).tolist()
        classes = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        return [
            DetectionPrediction(
                bbox=tuple(float(value) for value in bbox),
                class_id=int(class_id),
                confidence=float(confidence),
            ).to_dict()
            for bbox, class_id, confidence in zip(xywh, classes, confidences)
        ]

    @contextmanager
    def _loss_mode(self) -> Iterator[None]:
        """Expose raw training outputs while preventing detector-state updates."""

        module_modes = {module: module.training for module in self.model.modules()}
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        try:
            yield
        finally:
            for module, training in module_modes.items():
                module.training = training

    @contextmanager
    def _alignment_loss_mode(self) -> Iterator[None]:
        """Match the legacy gradient-scoring model state without updating buffers."""

        module_modes = {module: module.training for module in self.model.modules()}
        self.model.eval()
        try:
            yield
        finally:
            for module, training in module_modes.items():
                module.training = training

    def _preprocess_differentiable(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image tensor shaped [B,3,H,W], got {tuple(image.shape)}")
        image = image.to(device=self.device, dtype=torch.float32)
        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return image.clamp(0, 1)

    def enable_alignment_parameters(self) -> None:
        """Enable gradients only for the final detection head, in a stable order."""

        self.model.requires_grad_(False)
        head = self.model.model[-1]
        prefix = f"model.{len(self.model.model) - 1}"
        selected: list[tuple[str, torch.nn.Parameter]] = []
        for name, parameter in head.named_parameters():
            parameter.requires_grad_(True)
            selected.append((f"{prefix}.{name}" if name else prefix, parameter))
        if not selected:
            raise RuntimeError("The final detection head has no parameters for alignment")
        self._selected_alignment_parameters = selected

    def alignment_named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        if not self._selected_alignment_parameters:
            self.enable_alignment_parameters()
        return list(self._selected_alignment_parameters)

    def alignment_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for _, parameter in self.alignment_named_parameters()]

    def alignment_parameter_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
            }
            for name, parameter in self.alignment_named_parameters()
        ]

    def clear_parameter_gradients(self) -> None:
        self.model.zero_grad(set_to_none=True)

    @contextmanager
    def _alignment_parameters_disabled(self) -> Iterator[None]:
        """Keep the hard-example loss differentiable only with respect to its image."""

        selected = self.alignment_parameters()
        previous = [parameter.requires_grad for parameter in selected]
        for parameter in selected:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, requires_grad in zip(selected, previous):
                parameter.requires_grad_(requires_grad)

    def _native_detection_loss(
        self,
        image: torch.Tensor,
        targets: torch.Tensor,
        alignment_mode: bool = False,
    ) -> DetectionLoss:
        targets = targets.to(device=self.device, dtype=torch.float32)
        if targets.ndim != 2 or targets.shape[1] != 6:
            raise ValueError(f"Expected targets shaped [N,6], got {tuple(targets.shape)}")
        if targets.numel() and ((targets[:, 2:] < 0).any() or (targets[:, 2:] > 1).any()):
            raise ValueError("Differentiable-loss target boxes must be normalized to [0, 1]")
        batch = {
            "img": image,
            "batch_idx": targets[:, 0].to(dtype=torch.long),
            "cls": targets[:, 1:2],
            "bboxes": targets[:, 2:],
        }
        loss_mode = self._alignment_loss_mode if alignment_mode else self._loss_mode
        with loss_mode():
            raw_loss, raw_components = self.model(batch)
            total = raw_loss.sum()
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite {self.family} detection loss: {total.detach().cpu().item()}")
        components = {
            str(key): value.detach()
            for key, value in (raw_components.items() if isinstance(raw_components, dict) else {})
        }
        return DetectionLoss(total=total, components=components)

    def differentiable_loss(self, image: torch.Tensor, targets: torch.Tensor) -> DetectionLoss:
        """Compute native Ultralytics detection loss while preserving image gradients.

        Targets use rows ``[batch_idx, class_id, x_center, y_center, width, height]`` with normalized boxes.
        """

        image = self._preprocess_differentiable(image)
        with self._alignment_parameters_disabled():
            return self._native_detection_loss(image, targets)

    def alignment_loss(
        self,
        image: torch.Tensor,
        targets: torch.Tensor,
        image_size: int | None = None,
        scaleup: bool = False,
    ) -> DetectionLoss:
        """Compute the FP32 loss used by exact detector-head gradient alignment."""

        from ..prompt_gradient_alignment import differentiable_letterbox

        self.enable_alignment_parameters()
        image = image.to(device=self.device, dtype=torch.float32)
        targets = targets.to(device=self.device, dtype=torch.float32)
        image, targets = differentiable_letterbox(
            image,
            targets,
            image_size=int(image_size or self.image_size),
            scaleup=scaleup,
        )
        return self._native_detection_loss(image, targets, alignment_mode=True)


def create_detector(
    family: str,
    weights: str | Path,
    source_root: str | Path,
    device: str | int | torch.device = "0",
    image_size: int = 640,
    iou: float = 0.7,
    max_det: int = 300,
) -> UltralyticsDetector:
    return UltralyticsDetector(
        family=family,
        weights=weights,
        source_root=source_root,
        device=device,
        image_size=image_size,
        iou=iou,
        max_det=max_det,
    )
