"""Dataset checks and YOLO detection dataset configuration helpers."""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Sequence


# Ultralytics supports common image formats; keep this list broad for datasets.
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_SPLITS = ("train", "val")
OPTIONAL_SPLITS = ("test",)


def validate_class_names(class_names: Sequence[str]) -> list[str]:
    """Return cleaned class names or raise when class IDs would be ambiguous."""
    cleaned = [name.strip() for name in class_names]
    if not cleaned or any(not name for name in cleaned):
        raise ValueError("Provide at least one non-empty class name.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Class names must be unique and in the intended class-ID order.")
    return cleaned


def _validate_split(
    root: Path, split: str, class_count: int
) -> tuple[int, int, list[str]]:
    """Check image/label pairing and YOLO's normalized detection-label format."""
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    if not images_dir.is_dir():
        return 0, 0, [f"{images_dir}: required image directory is missing."]
    if not labels_dir.is_dir():
        return 0, 0, [f"{labels_dir}: required label directory is missing."]

    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    object_count, errors = _validate_image_label_pairs(images, images_dir, labels_dir, class_count)
    return len(images), object_count, errors


def _validate_image_label_pairs(
    images: Sequence[Path], images_root: Path, labels_root: Path, class_count: int
) -> tuple[int, list[str]]:
    """Validate matching YOLO labels for a collection of image paths."""
    errors: list[str] = []
    if not images:
        errors.append(f"{images_root}: no supported image files were found.")

    expected_label_paths = {
        path.relative_to(images_root).with_suffix(".txt") for path in images
    }
    present_label_paths = {
        path.relative_to(labels_root)
        for path in labels_root.rglob("*.txt")
        if path.is_file()
    }
    object_count = 0

    for relative_image in (path.relative_to(images_root) for path in images):
        relative_label = relative_image.with_suffix(".txt")
        label_path = labels_root / relative_label
        if not label_path.is_file():
            errors.append(
                f"{images_root / relative_image}: missing matching label file "
                f"{label_path} (use an empty .txt file for a true negative image)."
            )
            continue

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            errors.append(f"{label_path}: could not read label file: {exc}")
            continue

        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                errors.append(
                    f"{label_path}:{line_number}: expected 5 values "
                    "'class_id x_center y_center width height'."
                )
                continue

            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                errors.append(
                    f"{label_path}:{line_number}: class_id must be an integer "
                    "and box coordinates must be numeric."
                )
                continue

            if not 0 <= class_id < class_count:
                errors.append(
                    f"{label_path}:{line_number}: class_id must be an integer "
                    f"from 0 to {class_count - 1}."
                )
                continue
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                errors.append(
                    f"{label_path}:{line_number}: normalized coordinates must "
                    "all be finite values in [0, 1]."
                )
                continue
            x_center, y_center, width, height = coordinates
            if (
                x_center - width / 2 < 0
                or x_center + width / 2 > 1
                or y_center - height / 2 < 0
                or y_center + height / 2 > 1
            ):
                errors.append(
                    f"{label_path}:{line_number}: bounding box extends outside "
                    "the normalized image boundaries."
                )
                continue
            if coordinates[2] == 0 or coordinates[3] == 0:
                errors.append(
                    f"{label_path}:{line_number}: box width and height must be greater than 0."
                )
                continue

            object_count += 1

    for orphan in sorted(present_label_paths - expected_label_paths):
        errors.append(f"{labels_root / orphan}: label has no corresponding image.")

    return object_count, errors


def validate_unsplit_dataset(
    root: Path, class_names: Sequence[str]
) -> tuple[list[Path], dict[str, int]]:
    """Validate a source dataset with unsplit images/ and labels/ directories."""
    names = validate_class_names(class_names)
    root = root.expanduser().resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist or is not a directory: {root}")
    if not images_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"Label directory does not exist: {labels_dir}")

    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    object_count, errors = _validate_image_label_pairs(
        images, images_dir, labels_dir, len(names)
    )
    if errors:
        visible_errors = errors[:50]
        extra_count = len(errors) - len(visible_errors)
        details = "\n".join(f"  - {error}" for error in visible_errors)
        if extra_count:
            details += f"\n  - ... and {extra_count} more issue(s)."
        raise ValueError(f"Dataset validation failed:\n{details}")

    return images, {"images": len(images), "objects": object_count}


def _split_counts(
    image_count: int, ratios: Sequence[float]
) -> tuple[int, ...]:
    """Allocate integer image counts using largest remainders."""
    active_splits = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if image_count < len(active_splits):
        raise ValueError(
            "There are not enough images to populate every split with a non-zero ratio."
        )

    exact_counts = [image_count * ratio for ratio in ratios]
    counts = [math.floor(count) for count in exact_counts]
    remainders = [count - math.floor(count) for count in exact_counts]
    for index in sorted(range(len(ratios)), key=remainders.__getitem__, reverse=True)[
        : image_count - sum(counts)
    ]:
        counts[index] += 1

    # Ensure each requested split gets an image when the dataset is large enough.
    for empty_index in active_splits:
        if counts[empty_index] == 0:
            donor = max(
                (index for index in active_splits if counts[index] > 1),
                key=lambda index: counts[index] - exact_counts[index],
                default=None,
            )
            if donor is None:
                raise ValueError(
                    "There are not enough images to populate every split with a non-zero ratio."
                )
            counts[donor] -= 1
            counts[empty_index] += 1

    return tuple(counts)


def split_dataset(
    source_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, dict[str, int]]:
    """Copy an unsplit YOLO dataset into train/val/test folders and write data.yaml."""
    if len(ratios) != 3 or any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("Provide three finite, non-negative split ratios.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test ratios must add up to 1.0.")
    if ratios[0] <= 0 or ratios[1] <= 0:
        raise ValueError("Train and validation ratios must be greater than 0.")

    images, _ = validate_unsplit_dataset(source_root, class_names)
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Output directory must be separate from the source dataset directory.")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Choose another path "
            "or move its contents before splitting."
        )

    counts = _split_counts(len(images), ratios)
    shuffled_images = images.copy()
    random.Random(seed).shuffle(shuffled_images)

    split_names = ("train", "val", "test")
    report: dict[str, dict[str, int]] = {}
    offset = 0
    for split_name, count in zip(split_names, counts):
        split_images = shuffled_images[offset : offset + count]
        offset += count
        if count == 0:
            continue

        for image in split_images:
            relative_image = image.relative_to(source_root / "images")
            relative_label = relative_image.with_suffix(".txt")
            destination_image = output_root / "images" / split_name / relative_image
            destination_label = output_root / "labels" / split_name / relative_label
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination_image)
            shutil.copy2(source_root / "labels" / relative_label, destination_label)

        report[split_name] = {
            "images": count,
            "objects": sum(
                len(
                    (output_root / "labels" / split_name / image.relative_to(source_root / "images").with_suffix(".txt"))
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                for image in split_images
            ),
        }

    write_dataset_yaml(output_root, class_names, output_root / "data.yaml")
    return report


def validate_dataset(root: Path, class_names: Sequence[str]) -> dict[str, dict[str, int]]:
    """Validate required train/val splits and an optional test split."""
    names = validate_class_names(class_names)
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist or is not a directory: {root}")

    report: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    for split in REQUIRED_SPLITS:
        image_count, object_count, split_errors = _validate_split(root, split, len(names))
        report[split] = {"images": image_count, "objects": object_count}
        errors.extend(split_errors)

    test_images = root / "images" / "test"
    test_labels = root / "labels" / "test"
    if test_images.exists() or test_labels.exists():
        image_count, object_count, split_errors = _validate_split(root, "test", len(names))
        report["test"] = {"images": image_count, "objects": object_count}
        errors.extend(split_errors)

    if errors:
        visible_errors = errors[:50]
        extra_count = len(errors) - len(visible_errors)
        details = "\n".join(f"  - {error}" for error in visible_errors)
        if extra_count:
            details += f"\n  - ... and {extra_count} more issue(s)."
        raise ValueError(f"Dataset validation failed:\n{details}")

    return report


def write_dataset_yaml(
    root: Path, class_names: Sequence[str], output_path: Path
) -> Path:
    """Write an Ultralytics dataset YAML with paths relative to the dataset root."""
    names = validate_class_names(class_names)
    dataset_root = root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON string literals are valid YAML double-quoted scalars and safely escape
    # Windows paths, quotes, and non-ASCII class names.
    lines = [
        f"path: {json.dumps(dataset_root.as_posix(), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
    ]
    if (dataset_root / "images" / "test").is_dir():
        lines.append("test: images/test")
    lines.extend(["", "names:"])
    lines.extend(
        f"  {class_id}: {json.dumps(name, ensure_ascii=False)}"
        for class_id, name in enumerate(names)
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
