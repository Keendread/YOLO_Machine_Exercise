"""Unit tests for the helmet report module; no model, OpenCV or PIL needed."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yolo_starter.helmet_report import (
    CSV_COLUMNS,
    TOTAL_LABEL,
    Detection,
    ImageReport,
    summarize_image,
    summary_text,
    total_report,
    write_csv,
)


def summary_for(*detections, **kwargs) -> str:
    return summary_text(summarize_image("a.jpg", detections, **kwargs))


def read_rows(reports, **kwargs) -> list[list[str]]:
    buffer = io.StringIO(newline="")
    write_csv(reports, buffer, **kwargs)
    return list(csv.reader(io.StringIO(buffer.getvalue(), newline="")))


class SummaryWordingTests(unittest.TestCase):
    def test_no_riders(self) -> None:
        self.assertEqual(summary_for(), "No riders detected")

    def test_one_rider_with_helmet(self) -> None:
        self.assertEqual(summary_for(("Helmet", 0.9)), "1 rider: 1 with a helmet")

    def test_one_rider_without_helmet(self) -> None:
        self.assertEqual(summary_for(("No-Helmet", 0.9)), "1 rider: 1 without a helmet")

    def test_mixed_riders(self) -> None:
        self.assertEqual(
            summary_for(("Helmet", 0.9), ("No-Helmet", 0.8)),
            "2 riders: 1 with a helmet, 1 without a helmet",
        )

    def test_plural_helmets(self) -> None:
        self.assertEqual(
            summary_for(("Helmet", 0.9), ("Helmet", 0.8)), "2 riders: 2 with helmets"
        )
        self.assertEqual(
            summary_for(("No-Helmet", 0.9), ("No-Helmet", 0.8), ("Helmet", 0.7)),
            "3 riders: 1 with a helmet, 2 without helmets",
        )


class ClassMappingTests(unittest.TestCase):
    def test_names_ignore_case_and_separators(self) -> None:
        report = summarize_image(
            "a.jpg",
            [("HELMET", 0.9), ("no_helmet", 0.8), ("No Helmet", 0.7), ("no-helmet", 0.6)],
        )
        self.assertEqual((report.with_helmet, report.without_helmet), (1, 3))

    def test_unmapped_classes_are_ignored(self) -> None:
        report = summarize_image("a.jpg", [("person", 0.99), ("Helmet", 0.5), ("", 0.4)])
        self.assertEqual(report.total_riders, 1)
        self.assertEqual(report.helmet_confidences, (0.5,))
        self.assertEqual(report.no_helmet_confidences, ())

    def test_only_unmapped_classes_means_no_riders(self) -> None:
        report = summarize_image("a.jpg", [("car", 0.9), ("bicycle", 0.8)])
        self.assertEqual(report.total_riders, 0)
        self.assertEqual(summary_text(report), "No riders detected")

    def test_custom_class_map(self) -> None:
        class_map = {"hardhat": "helmet", "bare_head": "no_helmet"}
        report = summarize_image(
            "a.jpg", [("hardhat", 0.9), ("bare_head", 0.8), ("Helmet", 0.7)], class_map
        )
        self.assertEqual((report.with_helmet, report.without_helmet), (1, 1))

    def test_invalid_class_map_category_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_image("a.jpg", [("Helmet", 0.9)], {"Helmet": "hat"})

    def test_non_finite_confidence_is_ignored(self) -> None:
        report = summarize_image("a.jpg", [("Helmet", float("nan")), ("Helmet", 0.5)])
        self.assertEqual(report.helmet_confidences, (0.5,))

    def test_detection_tuple_and_plain_tuple_are_equivalent(self) -> None:
        self.assertEqual(
            summarize_image("a.jpg", [Detection("Helmet", 0.5)]),
            summarize_image("a.jpg", [("Helmet", 0.5)]),
        )


class CsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reports = [
            summarize_image(
                "one.jpg",
                [("Helmet", 0.91234), ("Helmet", 0.5), ("No-Helmet", 0.7777)],
            ),
            summarize_image("empty.jpg", []),
        ]

    def test_header_and_one_row_per_image(self) -> None:
        rows = read_rows(self.reports, include_total=False)
        self.assertEqual(rows[0], list(CSV_COLUMNS))
        self.assertEqual([row[0] for row in rows[1:]], ["one.jpg", "empty.jpg"])
        self.assertTrue(all(len(row) == len(CSV_COLUMNS) for row in rows))

    def test_row_contents_round_trip(self) -> None:
        row = dict(zip(CSV_COLUMNS, read_rows(self.reports)[1]))
        self.assertEqual(row["total_riders"], "3")
        self.assertEqual(row["with_helmet"], "2")
        self.assertEqual(row["without_helmet"], "1")
        self.assertEqual(row["helmet_conf_mean"], "0.706")  # (0.91234 + 0.5) / 2
        self.assertEqual(row["helmet_conf_min"], "0.500")
        self.assertEqual(row["no_helmet_conf_mean"], "0.778")
        self.assertEqual(row["no_helmet_conf_min"], "0.778")
        self.assertEqual(row["confidences"], "helmet:0.912;helmet:0.500;no_helmet:0.778")

    def test_no_detection_row_is_empty_but_valid(self) -> None:
        row = read_rows(self.reports, include_total=False)[2]
        self.assertEqual(row, ["empty.jpg", "0", "0", "0", "", "", "", "", ""])

    def test_only_empty_images_still_write_header_and_rows(self) -> None:
        rows = read_rows([summarize_image("x.jpg", [])])
        self.assertEqual(rows[0], list(CSV_COLUMNS))
        self.assertEqual(rows[1], ["x.jpg", "0", "0", "0", "", "", "", "", ""])
        self.assertEqual(rows[2][0], TOTAL_LABEL)

    def test_no_images_writes_header_only_without_total(self) -> None:
        self.assertEqual(read_rows([], include_total=False), [list(CSV_COLUMNS)])

    def test_total_row_is_labelled_and_sums_batch(self) -> None:
        rows = read_rows(self.reports)
        self.assertEqual(len(rows), 4)
        total = dict(zip(CSV_COLUMNS, rows[-1]))
        self.assertEqual(total["image"], TOTAL_LABEL)
        self.assertEqual(
            (total["total_riders"], total["with_helmet"], total["without_helmet"]),
            ("3", "2", "1"),
        )

    def test_total_report_sums_counts_and_confidences(self) -> None:
        total = total_report(self.reports)
        self.assertEqual(total.image, TOTAL_LABEL)
        self.assertEqual(total.total_riders, 3)
        self.assertEqual(len(total.helmet_confidences), 2)

    def test_awkward_image_names_round_trip(self) -> None:
        name = 'my "photo", day 1.jpg'
        rows = read_rows([ImageReport(image=name)], include_total=False)
        self.assertEqual(rows[1][0], name)

    def test_write_csv_to_path_uses_utf8_without_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            write_csv(self.reports, path)
            raw = path.read_bytes()
            self.assertNotIn(b"\r\r", raw)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 4)


class ImportTests(unittest.TestCase):
    def test_module_does_not_import_heavy_dependencies(self) -> None:
        code = (
            "import sys, yolo_starter.helmet_report;"
            "bad = [m for m in ('ultralytics', 'cv2', 'PIL', 'torch', 'numpy') if m in sys.modules];"
            "sys.exit(1 if bad else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
