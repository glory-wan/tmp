from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from concurrent.futures import Future
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from detection_gligen_cos import mine_failures_ultralytics as mining
from detection_gligen_cos.modeling.ultralytics_detector import UltralyticsDetector


class MiningParallelTest(unittest.TestCase):
    def test_batch_returns_all_images_in_order_including_empty_result(self):
        detector = object.__new__(UltralyticsDetector)
        detector.device, detector.image_size = "cpu", 64
        detector.iou, detector.max_det = 0.7, 300
        boxes = SimpleNamespace(xyxy=torch.tensor([[1., 2., 4., 6.]]),
                                cls=torch.tensor([2.]), conf=torch.tensor([0.8]))
        class Boxes:
            xyxy, cls, conf = boxes.xyxy, boxes.cls, boxes.conf

            def __len__(self):
                return 1

        detector.facade = SimpleNamespace(predict=lambda **kwargs: [
            SimpleNamespace(boxes=Boxes()), SimpleNamespace(boxes=None)])
        results = detector.predict_batch([object(), object()])
        self.assertEqual(results[0][0]["bbox"], (1., 2., 3., 4.))
        self.assertEqual(results[1], [])

    def test_workers_partition_batches_preserve_order_and_bound_prefetch(self):
        pools = []

        class Pool:
            def __init__(self, **kwargs):
                self.device = kwargs["initargs"][2]
                self.batches = []
                self.closed = False
                pools.append(self)

            def submit(self, function, *args):
                result = Future()
                if function is mining._worker_metadata:
                    result.set_result({"family": "yolo"})
                else:
                    self.batches.append(args[0])
                    result.set_result(args[0])
                return result

            def shutdown(self, **kwargs):
                self.closed = True

        args = Namespace(model_family="yolo", weights="unused", ultralytics_root="unused",
                         device="0", devices="2,3", image_size=64, iou=0.7,
                         max_det=300, batch_size=2, conf=0.001)
        images = [(i, str(i), []) for i in range(9)]
        with patch.object(mining, "ProcessPoolExecutor", Pool):
            with closing(mining.mining_predictions(args, [], images)) as stream:
                next(stream)
                self.assertEqual([next(stream) for _ in range(3)], ["0", "1", "2"])
            self.assertEqual([p.device for p in pools], ["2", "3"])
            self.assertEqual([p.batches for p in pools], [[["0", "1"]], [["2", "3"]]])
            self.assertTrue(all(p.closed for p in pools))
            with closing(mining.mining_predictions(args, [], images)) as stream:
                next(stream)
                self.assertEqual(list(stream), [str(i) for i in range(9)])

    def test_early_stop_only_commits_consumed_images_to_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, state = root / "failures.json", root / "forgotten.json"
            state.write_text(json.dumps({"1:1": {"was_correct": True, "forgotten_count": 0}}))
            annotations = {i: [{"id": i, "bbox": [0, 0, 10, 10], "class_id": 0}]
                           for i in range(1, 5)}
            dataset = SimpleNamespace(
                annotation_file="unused", images_dir=root, dataset_yaml=None,
                images={i: {} for i in annotations}, annotations=annotations,
                class_names=["object"], class_mapping={}, class_to_category={0: 1},
                iter_images=lambda limit: iter((i, root / str(i), anns)
                                              for i, anns in annotations.items()))
            closed = []

            def predictions(*args):
                try:
                    yield {"family": "yolo", "version": "test", "weights": "unused",
                           "supports_top2_gap": False}
                    yield from [[], [], [], []]
                finally:
                    closed.append(True)

            argv = ["mine", "--annotations", "unused", "--images", str(root),
                    "--model-family", "yolo", "--ultralytics-root", "unused",
                    "--weights", "unused", "--output", str(output), "--state", str(state),
                    "--min-failures-per-class", "2"]
            with patch("sys.argv", argv), patch.object(mining, "CocoMini", return_value=dataset), \
                    patch.object(mining, "mining_predictions", predictions):
                mining.main()
            history = json.loads(state.read_text())
            self.assertEqual(set(history), {"1:1", "2:2"})
            self.assertEqual(history["1:1"]["forgotten_count"], 1)
            self.assertEqual(len(json.loads(output.read_text())["failures"]), 2)
            self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
