"""Unit tests for dataset validation that do not require YOLO or image libraries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yolo_starter.dataset import (
    split_dataset,
    validate_class_names,
    validate_dataset,
    write_dataset_yaml,
)


class DatasetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            (self.root / "images" / split / "sample.jpg").touch()
            (self.root / "labels" / split / "sample.txt").write_text(
                "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_validates_yolo_labels_and_counts_boxes(self) -> None:
        report = validate_dataset(self.root, ["vehicle"])

        self.assertEqual(report["train"], {"images": 1, "objects": 1})
        self.assertEqual(report["val"], {"images": 1, "objects": 1})

    def test_rejects_class_id_outside_configured_names(self) -> None:
        label = self.root / "labels" / "train" / "sample.txt"
        label.write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "class_id must be an integer"):
            validate_dataset(self.root, ["vehicle"])

    def test_rejects_non_integer_class_id_token(self) -> None:
        label = self.root / "labels" / "train" / "sample.txt"
        label.write_text("0.0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "class_id must be an integer"):
            validate_dataset(self.root, ["vehicle"])

    def test_rejects_boxes_outside_image_boundaries(self) -> None:
        label = self.root / "labels" / "train" / "sample.txt"
        label.write_text("0 0.95 0.5 0.2 0.3\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "extends outside"):
            validate_dataset(self.root, ["vehicle"])

    def test_requires_a_label_file_for_every_image(self) -> None:
        (self.root / "labels" / "val" / "sample.txt").unlink()

        with self.assertRaisesRegex(ValueError, "missing matching label file"):
            validate_dataset(self.root, ["vehicle"])

    def test_writes_dataset_yaml_with_id_ordered_class_names(self) -> None:
        config_path = write_dataset_yaml(
            self.root, ["small vehicle", "large vehicle"], self.root / "data.yaml"
        )
        content = config_path.read_text(encoding="utf-8")

        self.assertIn("train: images/train", content)
        self.assertIn('  0: "small vehicle"', content)
        self.assertIn('  1: "large vehicle"', content)

    def test_class_names_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_class_names(["vehicle", "vehicle"])

    def test_splits_unsplit_dataset_and_keeps_source_files(self) -> None:
        source = self.root / "source"
        for index in range(10):
            (source / "images").mkdir(parents=True, exist_ok=True)
            (source / "labels").mkdir(parents=True, exist_ok=True)
            (source / "images" / f"sample-{index}.jpg").touch()
            (source / "labels" / f"sample-{index}.txt").write_text(
                "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
            )
        output = self.root / "source_split"

        report = split_dataset(source, output, ["vehicle"])

        self.assertEqual(report["train"]["images"], 8)
        self.assertEqual(report["val"]["images"], 1)
        self.assertEqual(report["test"]["images"], 1)
        self.assertTrue((source / "images" / "sample-0.jpg").is_file())
        self.assertTrue((output / "data.yaml").is_file())
        self.assertEqual(validate_dataset(output, ["vehicle"]), report)

    def test_rejects_output_directory_with_existing_files(self) -> None:
        source = self.root / "source"
        (source / "images").mkdir(parents=True)
        (source / "labels").mkdir()
        (source / "images" / "sample.jpg").touch()
        (source / "labels" / "sample.txt").write_text(
            "0 0.5 0.5 0.2 0.3\n", encoding="utf-8"
        )
        output = self.root / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "not empty"):
            split_dataset(source, output, ["vehicle"], ratios=(0.5, 0.5, 0.0))

        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
