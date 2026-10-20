"""Unit tests for the helmet dataset downloader without requiring network access."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import zipfile

from yolo_starter.download_helmet_dataset import (
    DEFAULT_CLASSES,
    build_parser,
    discover_split_pairs,
    download_archive,
    download_helmet_dataset,
    parse_yaml_classes,
)


def _create_zip_with_files(zip_path: Path, files: dict[str, str | bytes]) -> None:
    """Create a zip archive containing the given mapping of entry name to content."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            if isinstance(content, str):
                zf.writestr(name, content.encode("utf-8"))
            else:
                zf.writestr(name, content)


class DownloadHelmetDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parse_yaml_classes_formats(self) -> None:
        inline = "names: ['Helmet', 'No-Helmet']"
        self.assertEqual(parse_yaml_classes(inline), ["Helmet", "No-Helmet"])

        dict_format = "names:\n  0: Helmet\n  1: No-Helmet\n"
        self.assertEqual(parse_yaml_classes(dict_format), ["Helmet", "No-Helmet"])

        list_format = "names:\n  - Helmet\n  - No-Helmet\n"
        self.assertEqual(parse_yaml_classes(list_format), ["Helmet", "No-Helmet"])

    def test_folder_discovery_and_extraction_with_wrapped_valid(self) -> None:
        archive_path = self.root / "wrapped.zip"
        files = {
            "helmet-dataset/train/images/img1.jpg": b"\xff\xd8\xffdummy1",
            "helmet-dataset/train/labels/img1.txt": "0 0.5 0.5 0.2 0.3\n",
            "helmet-dataset/valid/images/img2.jpg": b"\xff\xd8\xffdummy2",
            "helmet-dataset/valid/labels/img2.txt": "1 0.4 0.4 0.1 0.1\n",
            "helmet-dataset/test/images/img3.jpg": b"\xff\xd8\xffdummy3",
            "helmet-dataset/test/labels/img3.txt": "0 0.6 0.6 0.2 0.2\n",
            "helmet-dataset/data.yaml": "names:\n  0: Helmet\n  1: No-Helmet\n",
        }
        _create_zip_with_files(archive_path, files)

        output_dir = self.root / "extracted"
        report = download_helmet_dataset(
            archive=archive_path,
            output=output_dir,
        )

        self.assertEqual(report["train"], {"images": 1, "objects": 1})
        self.assertEqual(report["val"], {"images": 1, "objects": 1})
        self.assertEqual(report["test"], {"images": 1, "objects": 1})

        self.assertTrue((output_dir / "images" / "train" / "img1.jpg").is_file())
        self.assertTrue((output_dir / "labels" / "train" / "img1.txt").is_file())
        self.assertTrue((output_dir / "images" / "val" / "img2.jpg").is_file())
        self.assertTrue((output_dir / "labels" / "val" / "img2.txt").is_file())
        self.assertTrue((output_dir / "images" / "test" / "img3.jpg").is_file())
        self.assertTrue((output_dir / "labels" / "test" / "img3.txt").is_file())

    def test_folder_discovery_images_split_layout(self) -> None:
        archive_path = self.root / "images_split.zip"
        files = {
            "dataset/images/train/t1.png": b"png1",
            "dataset/labels/train/t1.txt": "0 0.5 0.5 0.2 0.3\n",
            "dataset/images/val/v1.png": b"png2",
            "dataset/labels/val/v1.txt": "1 0.5 0.5 0.2 0.3\n",
        }
        _create_zip_with_files(archive_path, files)

        output_dir = self.root / "images_split_out"
        report = download_helmet_dataset(
            archive=archive_path,
            output=output_dir,
        )

        self.assertEqual(report["train"]["images"], 1)
        self.assertEqual(report["val"]["images"], 1)
        self.assertTrue((output_dir / "images" / "train" / "t1.png").is_file())
        self.assertTrue((output_dir / "labels" / "train" / "t1.txt").is_file())

    def test_subsetting_determinism(self) -> None:
        archive_path = self.root / "multi.zip"
        files: dict[str, str | bytes] = {}
        for idx in range(10):
            files[f"train/images/img_{idx:02d}.jpg"] = b"data"
            files[f"train/labels/img_{idx:02d}.txt"] = f"0 0.5 0.5 0.2 0.3\n"
        for idx in range(4):
            files[f"val/images/val_{idx:02d}.jpg"] = b"data"
            files[f"val/labels/val_{idx:02d}.txt"] = f"1 0.5 0.5 0.2 0.3\n"
        _create_zip_with_files(archive_path, files)

        out_run1 = self.root / "run1"
        out_run2 = self.root / "run2"
        out_run3 = self.root / "run3"

        # Runs 1 and 2 use the same seed
        rep1 = download_helmet_dataset(
            archive=archive_path,
            output=out_run1,
            max_train=4,
            max_val=2,
            seed=42,
        )
        rep2 = download_helmet_dataset(
            archive=archive_path,
            output=out_run2,
            max_train=4,
            max_val=2,
            seed=42,
        )
        # Run 3 uses a different seed
        rep3 = download_helmet_dataset(
            archive=archive_path,
            output=out_run3,
            max_train=4,
            max_val=2,
            seed=999,
        )

        self.assertEqual(rep1["train"]["images"], 4)
        self.assertEqual(rep1["val"]["images"], 2)

        images1 = sorted(p.name for p in (out_run1 / "images" / "train").iterdir())
        images2 = sorted(p.name for p in (out_run2 / "images" / "train").iterdir())
        images3 = sorted(p.name for p in (out_run3 / "images" / "train").iterdir())

        self.assertEqual(images1, images2)
        self.assertNotEqual(images1, images3)

    def test_pair_integrity_content_and_missing_label(self) -> None:
        archive_path = self.root / "integrity.zip"
        files: dict[str, str | bytes] = {}
        for idx in range(6):
            files[f"train/images/img_{idx}.jpg"] = b"img"
            files[f"train/labels/img_{idx}.txt"] = f"0 0.{idx + 2} 0.5 0.1 0.1\n"
        files["val/images/val.jpg"] = b"val"
        files["val/labels/val.txt"] = "1 0.5 0.5 0.2 0.2\n"
        _create_zip_with_files(archive_path, files)

        out = self.root / "integrity_out"
        download_helmet_dataset(
            archive=archive_path,
            output=out,
            max_train=3,
            seed=12,
        )

        extracted_train_images = list((out / "images" / "train").glob("*.jpg"))
        self.assertEqual(len(extracted_train_images), 3)

        for img_path in extracted_train_images:
            lbl_path = out / "labels" / "train" / f"{img_path.stem}.txt"
            self.assertTrue(lbl_path.is_file(), f"Missing label for {img_path}")
            idx = int(img_path.stem.split("_")[1])
            expected_content = f"0 0.{idx + 2} 0.5 0.1 0.1\n"
            self.assertEqual(lbl_path.read_text(encoding="utf-8"), expected_content)

        # Now test archive with missing label
        bad_archive = self.root / "bad.zip"
        _create_zip_with_files(
            bad_archive,
            {
                "train/images/orphan.jpg": b"data",
                "val/images/v.jpg": b"data",
                "val/labels/v.txt": "0 0.5 0.5 0.1 0.1\n",
            },
        )
        with self.assertRaisesRegex(ValueError, "has no matching label file"):
            download_helmet_dataset(archive=bad_archive, output=self.root / "bad_out")

    def test_data_yaml_contents(self) -> None:
        archive_path = self.root / "yaml_test.zip"
        files = {
            "train/images/t.jpg": b"data",
            "train/labels/t.txt": "0 0.5 0.5 0.2 0.2\n",
            "val/images/v.jpg": b"data",
            "val/labels/v.txt": "1 0.5 0.5 0.2 0.2\n",
            "test/images/te.jpg": b"data",
            "test/labels/te.txt": "0 0.5 0.5 0.2 0.2\n",
        }
        _create_zip_with_files(archive_path, files)

        out = self.root / "yaml_out"
        download_helmet_dataset(archive=archive_path, output=out)

        yaml_path = out / "data.yaml"
        self.assertTrue(yaml_path.is_file())
        content = yaml_path.read_text(encoding="utf-8")

        self.assertIn(f'path: "{out.resolve().as_posix()}"', content)
        self.assertIn("train: images/train", content)
        self.assertIn("val: images/val", content)
        self.assertIn("test: images/test", content)
        self.assertIn('0: "Helmet"', content)
        self.assertIn('1: "No-Helmet"', content)

    def test_refuses_non_empty_output_unless_overwrite(self) -> None:
        archive_path = self.root / "overwrite_test.zip"
        files = {
            "train/images/t.jpg": b"data",
            "train/labels/t.txt": "0 0.5 0.5 0.2 0.2\n",
            "val/images/v.jpg": b"data",
            "val/labels/v.txt": "1 0.5 0.5 0.2 0.2\n",
        }
        _create_zip_with_files(archive_path, files)

        out = self.root / "existing_dir"
        out.mkdir(parents=True)
        (out / "stale.txt").write_text("old", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            download_helmet_dataset(archive=archive_path, output=out, overwrite=False)

        # With overwrite=True, succeeds and removes stale file
        download_helmet_dataset(archive=archive_path, output=out, overwrite=True)
        self.assertFalse((out / "stale.txt").exists())
        self.assertTrue((out / "images" / "train" / "t.jpg").is_file())

    def test_skips_download_when_archive_exists(self) -> None:
        archive_path = self.root / "existing.zip"
        archive_path.write_bytes(b"placeholder")

        with patch("urllib.request.urlopen") as mock_urlopen:
            result = download_archive(url="http://dummy/test.zip", target_path=archive_path)
            self.assertEqual(result, archive_path.resolve())
            mock_urlopen.assert_not_called()

    def test_downloads_when_archive_missing(self) -> None:
        archive_path = self.root / "new.zip"
        self.assertFalse(archive_path.exists())

        fake_content = b"PK\x05\x06" + b"\x00" * 18  # valid empty zip header
        mock_response = MagicMock()
        mock_response.__enter__.return_value = io.BytesIO(fake_content)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = download_archive(url="http://dummy/test.zip", target_path=archive_path)
            self.assertEqual(result, archive_path.resolve())
            self.assertTrue(archive_path.is_file())
            mock_urlopen.assert_called_once()

    def test_rejects_unsafe_archive_entry_path(self) -> None:
        archive_path = self.root / "unsafe.zip"
        files = {
            "train/images/../../escape.jpg": b"data",
            "train/labels/../../escape.txt": "0 0.5 0.5 0.1 0.1\n",
            "val/images/val.jpg": b"data",
            "val/labels/val.txt": "1 0.5 0.5 0.1 0.1\n",
        }
        _create_zip_with_files(archive_path, files)
        with self.assertRaisesRegex(ValueError, "Unsafe path in archive entry"):
            download_helmet_dataset(archive=archive_path, output=self.root / "unsafe_out")

    def test_build_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.seed, 0)
        self.assertFalse(args.overwrite)
        self.assertIsNone(args.max_train)
        self.assertIsNone(args.max_val)
        self.assertIsNone(args.max_test)


if __name__ == "__main__":
    unittest.main()
