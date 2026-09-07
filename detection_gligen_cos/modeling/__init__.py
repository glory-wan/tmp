"""Detection-model adapters used by the closed-loop pipeline."""

from .ultralytics_detector import (
    DetectionLoss,
    DetectionPrediction,
    UltralyticsDetector,
    create_detector,
    load_ultralytics_model,
)

__all__ = [
    "DetectionLoss",
    "DetectionPrediction",
    "UltralyticsDetector",
    "create_detector",
    "load_ultralytics_model",
]

