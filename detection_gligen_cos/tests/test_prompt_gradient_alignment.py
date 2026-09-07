from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from detection_gligen_cos.prompt_gradient_alignment import (
    AlignmentRunningStatistics,
    build_layout_target,
    differentiable_letterbox,
    global_gradient_cosine,
    parse_yolo_detection_label,
)


class _LayoutDataset:
    images = {1: {"width": 100, "height": 50}}
    annotations = {
        1: [
            {"id": 10, "class_id": 0, "bbox": [0, 0, 10, 10]},
            {"id": 11, "class_id": 1, "bbox": [20, 10, 40, 20]},
            {"id": 12, "class_id": 2, "bbox": [90, 45, 1, 1]},
        ]
    }


class PromptGradientAlignmentTest(unittest.TestCase):
    def test_letterbox_matches_scaleup_false_geometry(self):
        image = torch.zeros(1, 3, 512, 512, requires_grad=True)
        targets = torch.tensor([[0, 1, 0.5, 0.5, 0.2, 0.4]])
        output, transformed = differentiable_letterbox(
            image, targets, image_size=640, scaleup=False
        )
        self.assertEqual(tuple(output.shape), (1, 3, 640, 640))
        torch.testing.assert_close(
            transformed,
            torch.tensor([[0, 1, 0.5, 0.5, 0.16, 0.32]]),
        )
        output.sum().backward()
        self.assertIsNotNone(image.grad)

    def test_letterbox_scaleup_true_uses_full_canvas(self):
        image = torch.zeros(1, 3, 512, 512)
        targets = torch.tensor([[0, 1, 0.5, 0.5, 0.2, 0.4]])
        _, transformed = differentiable_letterbox(
            image, targets, image_size=640, scaleup=True
        )
        torch.testing.assert_close(transformed, targets)

    def test_polygon_label_is_converted_to_legacy_outer_box(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("2 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8\n", encoding="utf-8")
            labels = parse_yolo_detection_label(path)
        torch.testing.assert_close(
            labels, torch.tensor([[2.0, 0.3, 0.5, 0.4, 0.6]])
        )

    def test_layout_target_prioritizes_failure_and_matches_generation_filter(self):
        sample = {
            "image_id": 1,
            "annotation_id": 11,
            "class_id": 1,
            "bbox": [20, 10, 40, 20],
        }
        target = build_layout_target(
            _LayoutDataset(),
            sample,
            torch.device("cpu"),
            max_objects=2,
            enable_filter=True,
            min_bbox_area_ratio=0.01,
        )
        self.assertEqual(target[:, 1].tolist(), [1.0, 0.0])
        self.assertEqual(tuple(target.shape), (2, 6))

    def test_global_cosine_uses_all_parameter_elements(self):
        p1 = torch.nn.Parameter(torch.zeros(2))
        p2 = torch.nn.Parameter(torch.zeros(1))
        scale = torch.tensor(2.0, requires_grad=True)
        generated = [scale * torch.tensor([1.0, 2.0]), scale * torch.tensor([3.0])]
        guide = [torch.tensor([2.0, 0.0]), torch.tensor([1.0])]
        result = global_gradient_cosine(generated, guide, [p1, p2])
        expected = 5.0 / ((14.0**0.5) * (5.0**0.5))
        self.assertAlmostEqual(float(result.cosine), expected, places=6)
        result.cosine.backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad))

    def test_statistics_can_resume_from_saved_summary(self):
        stats = AlignmentRunningStatistics()
        stats.update("class_0", -0.5)
        stats.update("class_0", 0.5)
        resumed = AlignmentRunningStatistics(stats.to_dict())
        resumed.update("class_0", 1.0)
        summary = resumed.to_dict()["groups"]["class_0"]
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["positive_rate"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
