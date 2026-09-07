from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from detection_gligen_cos.common import CocoMini, load_dataset_yaml_names, load_failures


class DatasetClassMappingTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="gligen-sdedit-dataset-mapping.")
        self.root = Path(self.tempdir.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "selected.jpg").write_bytes(b"test")
        (self.images / "person_only.jpg").write_bytes(b"test")
        self.annotations = self.root / "instances.json"
        self.annotations.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "selected.jpg", "width": 100, "height": 80},
                        {"id": 2, "file_name": "person_only.jpg", "width": 100, "height": 80},
                    ],
                    "categories": [
                        {"id": 1, "name": "person"},
                        {"id": 6, "name": "bus"},
                        {"id": 18, "name": "dog"},
                        {"id": 28, "name": "umbrella"},
                    ],
                    "annotations": [
                        {"id": 10, "image_id": 1, "category_id": 6, "bbox": [10, 10, 20, 20], "area": 400},
                        {"id": 11, "image_id": 1, "category_id": 18, "bbox": [20, 20, 20, 20], "area": 400},
                        {"id": 12, "image_id": 1, "category_id": 28, "bbox": [30, 30, 20, 20], "area": 400},
                        {"id": 13, "image_id": 2, "category_id": 1, "bbox": [5, 5, 10, 10], "area": 100},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.dataset_yaml = self.root / "coco_subset.yaml"
        self.dataset_yaml.write_text(
            "path: /unused/by/this/loader\n"
            "train: images/train2017\n"
            "val: images/val2017\n"
            "names:\n"
            "  0: umbrella\n"
            "  1: dog\n"
            "  2: bus\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_yaml_names_define_model_id_order_and_coco_mapping(self):
        dataset = CocoMini(self.annotations, self.images, dataset_yaml=self.dataset_yaml)

        self.assertEqual(dataset.class_names, ["umbrella", "dog", "bus"])
        self.assertEqual(dataset.category_to_class, {28: 0, 18: 1, 6: 2})
        self.assertEqual(dataset.class_to_category, {0: 28, 1: 18, 2: 6})
        self.assertEqual(
            dataset.class_mapping,
            [
                {"model_class_id": 0, "class_name": "umbrella", "coco_category_id": 28},
                {"model_class_id": 1, "class_name": "dog", "coco_category_id": 18},
                {"model_class_id": 2, "class_name": "bus", "coco_category_id": 6},
            ],
        )
        self.assertEqual(set(dataset.images), {1})
        self.assertEqual(
            [annotation["class_id"] for annotation in dataset.annotations[1]],
            [2, 1, 0],
        )

    def test_yolo_export_uses_remapped_model_ids(self):
        dataset = CocoMini(self.annotations, self.images, dataset_yaml=self.dataset_yaml)
        dataset.export_yolo(self.root / "export", [1])

        labels = (self.root / "export/labels/selected.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual([int(line.split()[0]) for line in labels], [2, 1, 0])

    def test_class_subset_is_interpreted_in_yaml_local_id_space(self):
        dataset = CocoMini(
            self.annotations,
            self.images,
            class_subset={2},
            dataset_yaml=self.dataset_yaml,
        )

        self.assertEqual(dataset.class_names, ["umbrella", "dog", "bus"])
        self.assertEqual([annotation["class_id"] for annotation in dataset.annotations[1]], [2])
        with self.assertRaisesRegex(ValueError, "outside the 3-class space"):
            CocoMini(
                self.annotations,
                self.images,
                class_subset={5},
                dataset_yaml=self.dataset_yaml,
            )

    def test_dataset_yaml_requires_contiguous_ids_and_known_names(self):
        invalid_ids = self.root / "invalid_ids.yaml"
        invalid_ids.write_text("names:\n  0: umbrella\n  2: bus\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contiguous from 0"):
            load_dataset_yaml_names(invalid_ids)

        missing_name = self.root / "missing_name.yaml"
        missing_name.write_text("names: [umbrella, bicycle]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing from COCO categories"):
            CocoMini(self.annotations, self.images, dataset_yaml=missing_name)

    def test_failure_pool_rejects_a_different_class_order(self):
        failures = self.root / "failures.json"
        failures.write_text(
            json.dumps(
                {
                    "meta": {"class_names": ["umbrella", "dog", "bus"]},
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            load_failures(failures, expected_class_names=["umbrella", "dog", "bus"]),
            [],
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            load_failures(failures, expected_class_names=["bus", "dog", "umbrella"])


if __name__ == "__main__":
    unittest.main()
