from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import build_coco_cos_subsets_ultralytics as legacy_alignment
from detection_gligen_cos.modeling import create_detector
from detection_gligen_cos.modeling.ultralytics_detector import load_local_ultralytics, normalize_family
from detection_gligen_cos.prompt_gradient_alignment import global_gradient_cosine


ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_ROOT = ROOT / "ultralytics"


class _FakeBoxes:
    xyxy = torch.tensor([[10.0, 20.0, 40.0, 70.0]])
    cls = torch.tensor([3.0])
    conf = torch.tensor([0.75])

    def __len__(self):
        return 1


class _FakeResult:
    boxes = _FakeBoxes()


class UltralyticsDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="gligen-sdedit-model-tests.")
        cls.tmp = Path(cls.tempdir.name)
        ultralytics = load_local_ultralytics(ULTRALYTICS_ROOT)

        cls.yolo_weights = cls.tmp / "yolo26n.pt"
        yolo_cfg = ULTRALYTICS_ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
        ultralytics.YOLO(str(yolo_cfg), task="detect").save(cls.yolo_weights)

        cls.classify_weights = cls.tmp / "yolo26n-cls.pt"
        classify_cfg = ULTRALYTICS_ROOT / "ultralytics/cfg/models/26/yolo26-cls.yaml"
        ultralytics.YOLO(str(classify_cfg), task="classify").save(cls.classify_weights)

        cls.rtdetr_weights = cls.tmp / "rtdetr-l.pt"
        rtdetr_cfg = ULTRALYTICS_ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml"
        rtdetr = ultralytics.RTDETR(str(rtdetr_cfg))
        rtdetr.model.criterion = None  # Matches optimizer-stripped checkpoints produced after training.
        rtdetr.save(cls.rtdetr_weights)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_family_normalization(self):
        self.assertEqual(normalize_family("YOLO"), "yolo")
        self.assertEqual(normalize_family("RT-DETR"), "rtdetr")
        with self.assertRaises(ValueError):
            normalize_family("segment")

    def test_yolo_end_to_end_loss_preserves_input_gradient(self):
        detector = create_detector(
            "yolo",
            self.yolo_weights,
            ULTRALYTICS_ROOT,
            device="cpu",
            image_size=64,
            max_det=10,
        )
        image = torch.rand(1, 3, 64, 64, requires_grad=True)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.25, 0.25]])
        result = detector.differentiable_loss(image, targets)
        result.total.backward()
        self.assertTrue(torch.isfinite(result.total))
        self.assertGreater(float(image.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in detector.model.parameters()))
        self.assertIn("cls_loss", result.components)

    def test_yolo_detection_head_alignment_is_exact_and_differentiable(self):
        detector = create_detector(
            "yolo",
            self.yolo_weights,
            ULTRALYTICS_ROOT,
            device="cpu",
            image_size=64,
            max_det=10,
        )
        detector.enable_alignment_parameters()
        parameters = detector.alignment_parameters()
        parameter_snapshot = [parameter.detach().clone() for parameter in parameters]
        metadata = detector.alignment_parameter_metadata()
        self.assertTrue(metadata)
        self.assertEqual(sum(item["numel"] for item in metadata), sum(p.numel() for p in parameters))
        self.assertTrue(all(item["name"].startswith("model.") for item in metadata))

        target = torch.tensor([[0, 0, 0.5, 0.5, 0.25, 0.25]])
        guide_image = torch.rand(1, 3, 64, 64)
        guide_loss = detector.alignment_loss(guide_image, target).total
        guide = torch.autograd.grad(guide_loss, parameters, allow_unused=True)
        guide = [
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, guide)
        ]

        token = torch.nn.Parameter(torch.tensor([0.1]))
        source_image = torch.sigmoid(
            torch.rand(1, 3, 64, 64)
            + token.view(1, 1, 1, 1) * torch.randn(1, 3, 64, 64)
        )
        source_image.retain_grad()
        source_loss = detector.alignment_loss(source_image, target).total
        generated = torch.autograd.grad(
            source_loss,
            parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        result = global_gradient_cosine(generated, guide, parameters)
        (-result.cosine).backward()
        self.assertTrue(torch.isfinite(result.cosine))
        self.assertIsNotNone(source_image.grad)
        self.assertTrue(torch.isfinite(source_image.grad).all())
        self.assertGreater(float(source_image.grad.abs().sum()), 0.0)
        self.assertIsNotNone(token.grad)
        self.assertTrue(torch.isfinite(token.grad).all())
        self.assertGreater(float(token.grad.abs().sum()), 0.0)
        for before, parameter in zip(parameter_snapshot, parameters):
            torch.testing.assert_close(parameter.detach(), before, rtol=0, atol=0)

    def test_alignment_parameter_order_and_cosine_match_legacy_script(self):
        legacy_alignment.torch = torch
        modules = legacy_alignment.import_local_ultralytics(ULTRALYTICS_ROOT)
        legacy = legacy_alignment.UltralyticsGradientModel(
            self.yolo_weights,
            "detect",
            torch.device("cpu"),
            modules,
        )
        detector = create_detector(
            "yolo",
            self.yolo_weights,
            ULTRALYTICS_ROOT,
            device="cpu",
            image_size=64,
            max_det=10,
        )
        detector.enable_alignment_parameters()
        self.assertEqual(
            detector.alignment_parameter_metadata(), legacy.parameter_metadata
        )

        target = torch.tensor([[0, 0, 0.5, 0.5, 0.25, 0.25]])
        images = [torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)]
        legacy_gradients = []
        current_gradients = []
        for image in images:
            batch = {
                "img": image,
                "batch_idx": target[:, 0].long(),
                "cls": target[:, 1:2],
                "bboxes": target[:, 2:],
                "paths": ["memory"],
            }
            _, legacy_gradient = legacy_alignment.compute_loss_gradients(
                legacy, batch, torch.device("cpu")
            )
            legacy_gradients.append(legacy_gradient)

            loss = detector.alignment_loss(image, target).total
            gradient = torch.autograd.grad(
                loss,
                detector.alignment_parameters(),
                allow_unused=True,
            )
            current_gradients.append(
                [
                    torch.zeros_like(parameter) if value is None else value.detach()
                    for parameter, value in zip(detector.alignment_parameters(), gradient)
                ]
            )
            for current_value, legacy_value in zip(
                current_gradients[-1], legacy_gradient
            ):
                torch.testing.assert_close(
                    current_value.cpu(), legacy_value, rtol=1e-5, atol=1e-6
                )

        legacy_cosine = legacy_alignment.gradient_statistics(
            legacy_gradients[0], legacy_gradients[1]
        )["cosine"]
        current_cosine = global_gradient_cosine(
            current_gradients[0],
            current_gradients[1],
            detector.alignment_parameters(),
        ).cosine
        self.assertAlmostEqual(float(current_cosine), legacy_cosine, places=5)

    def test_non_detection_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only detection checkpoints"):
            create_detector(
                "yolo",
                self.classify_weights,
                ULTRALYTICS_ROOT,
                device="cpu",
                image_size=64,
            )

    def test_family_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "family=yolo"):
            create_detector(
                "yolo",
                self.rtdetr_weights,
                ULTRALYTICS_ROOT,
                device="cpu",
                image_size=320,
            )

    def test_rtdetr_loss_preserves_input_gradient(self):
        detector = create_detector(
            "rtdetr",
            self.rtdetr_weights,
            ULTRALYTICS_ROOT,
            device="cpu",
            image_size=320,
            max_det=10,
        )
        image = torch.rand(1, 3, 320, 320, requires_grad=True)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.25, 0.25]])
        result = detector.differentiable_loss(image, targets)
        result.total.backward()
        self.assertTrue(torch.isfinite(result.total))
        self.assertGreater(float(image.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in detector.model.parameters()))
        self.assertEqual(set(result.components), {"cls_loss", "giou_loss", "l1_loss"})

    def test_prediction_converts_xyxy_to_top_left_xywh(self):
        detector = create_detector(
            "yolo",
            self.yolo_weights,
            ULTRALYTICS_ROOT,
            device="cpu",
            image_size=64,
            max_det=10,
        )
        detector.facade.predict = lambda **_: [_FakeResult()]
        predictions = detector.predict(object())
        self.assertEqual(predictions[0]["bbox"], (10.0, 20.0, 30.0, 50.0))
        self.assertEqual(predictions[0]["class_id"], 3)
        self.assertEqual(predictions[0]["confidence"], 0.75)
        self.assertIsNone(predictions[0]["top2_gap"])


if __name__ == "__main__":
    unittest.main()
