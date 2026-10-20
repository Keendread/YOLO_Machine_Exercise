"""Turn per-image helmet detections into a rider summary and a CSV report.

This module is pure Python (standard library only) so it can be unit tested
without a model, OpenCV or PIL. Callers convert whatever their detector returns
into ``Detection(class_name, confidence)`` pairs and hand them to
``summarize_image``.

A "rider" is a detection whose class is *mapped* to ``HELMET`` or
``NO_HELMET`` by the class map. Classes that are not in the map are ignored.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, NamedTuple, TextIO

HELMET = "helmet"
NO_HELMET = "no_helmet"

TOTAL_LABEL = "TOTAL"

CSV_COLUMNS = (
    "image",
    "total_riders",
    "with_helmet",
    "without_helmet",
    "helmet_conf_mean",
    "helmet_conf_min",
    "no_helmet_conf_mean",
    "no_helmet_conf_min",
    "confidences",
)

CONFIDENCE_DECIMALS = 3


def _normalize(class_name: str) -> str:
    """Compare class names ignoring case and separators ("No-Helmet" == "no_helmet")."""
    return "".join(ch for ch in str(class_name).lower() if ch.isalnum())


# Defaults match the dataset classes (0=Helmet, 1=No-Helmet) plus common spellings.
DEFAULT_CLASS_MAP: Mapping[str, str] = {
    "Helmet": HELMET,
    "With Helmet": HELMET,
    "No-Helmet": NO_HELMET,
    "Without Helmet": NO_HELMET,
}


class Detection(NamedTuple):
    class_name: str
    confidence: float


@dataclass(frozen=True)
class ImageReport:
    """Rider counts and confidences for one image."""

    image: str
    with_helmet: int = 0
    without_helmet: int = 0
    helmet_confidences: tuple[float, ...] = ()
    no_helmet_confidences: tuple[float, ...] = ()

    @property
    def total_riders(self) -> int:
        return self.with_helmet + self.without_helmet


def _build_lookup(class_map: Mapping[str, str] | None) -> dict[str, str]:
    source = DEFAULT_CLASS_MAP if class_map is None else class_map
    lookup = {}
    for name, category in source.items():
        if category not in (HELMET, NO_HELMET):
            raise ValueError(
                f"Class {name!r} maps to {category!r}; expected "
                f"{HELMET!r} or {NO_HELMET!r}."
            )
        lookup[_normalize(name)] = category
    return lookup


def summarize_image(
    image: str,
    detections: Iterable[tuple[str, float]],
    class_map: Mapping[str, str] | None = None,
) -> ImageReport:
    """Count riders in one image. Unmapped classes and non-finite scores are ignored."""
    lookup = _build_lookup(class_map)
    helmet: list[float] = []
    no_helmet: list[float] = []
    for class_name, confidence in detections:
        category = lookup.get(_normalize(class_name))
        confidence = float(confidence)
        if category is None or not math.isfinite(confidence):
            continue
        (helmet if category == HELMET else no_helmet).append(confidence)
    return ImageReport(
        image=str(image),
        with_helmet=len(helmet),
        without_helmet=len(no_helmet),
        helmet_confidences=tuple(helmet),
        no_helmet_confidences=tuple(no_helmet),
    )


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def summary_text(report: ImageReport) -> str:
    """Human-readable summary, e.g. "2 riders: 1 with a helmet, 1 without a helmet"."""
    if report.total_riders == 0:
        return "No riders detected"
    parts = []
    if report.with_helmet:
        noun = "with a helmet" if report.with_helmet == 1 else "with helmets"
        parts.append(f"{report.with_helmet} {noun}")
    if report.without_helmet:
        noun = "without a helmet" if report.without_helmet == 1 else "without helmets"
        parts.append(f"{report.without_helmet} {noun}")
    riders = _plural(report.total_riders, "rider", "riders")
    return f"{riders}: {', '.join(parts)}"


def total_report(reports: Iterable[ImageReport]) -> ImageReport:
    """Combine reports into one batch-level report labelled ``TOTAL``."""
    reports = list(reports)
    return ImageReport(
        image=TOTAL_LABEL,
        with_helmet=sum(r.with_helmet for r in reports),
        without_helmet=sum(r.without_helmet for r in reports),
        helmet_confidences=tuple(c for r in reports for c in r.helmet_confidences),
        no_helmet_confidences=tuple(c for r in reports for c in r.no_helmet_confidences),
    )


def _round(value: float) -> str:
    return f"{round(value, CONFIDENCE_DECIMALS):.{CONFIDENCE_DECIMALS}f}"


def _mean_min(values: tuple[float, ...]) -> list[str]:
    if not values:
        return ["", ""]
    return [_round(sum(values) / len(values)), _round(min(values))]


def csv_row(report: ImageReport) -> list[str]:
    """One CSV row. Confidence cells are blank when there is nothing to report."""
    listed = [f"{HELMET}:{_round(c)}" for c in report.helmet_confidences]
    listed += [f"{NO_HELMET}:{_round(c)}" for c in report.no_helmet_confidences]
    return [
        report.image,
        str(report.total_riders),
        str(report.with_helmet),
        str(report.without_helmet),
        *_mean_min(report.helmet_confidences),
        *_mean_min(report.no_helmet_confidences),
        ";".join(listed),
    ]


def write_csv(
    reports: Iterable[ImageReport],
    destination: str | os.PathLike[str] | TextIO,
    include_total: bool = True,
) -> None:
    """Write a header, one row per image and (by default) a final ``TOTAL`` row.

    ``destination`` is a path or an open text file; a file object should be
    opened with ``newline=""`` as the csv module requires.
    """
    reports = list(reports)
    if include_total:
        reports.append(total_report(reports))

    def write(handle: TextIO) -> None:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(csv_row(report) for report in reports)

    if isinstance(destination, (str, os.PathLike)):
        with open(destination, "w", newline="", encoding="utf-8") as handle:
            write(handle)
    else:
        write(destination)
