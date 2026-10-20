"""Download, extract, subset, and validate the DeBoniz helmet detection dataset."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence

from yolo_starter.dataset import (
    IMAGE_SUFFIXES,
    validate_class_names,
    validate_dataset,
    write_dataset_yaml,
)

DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/DeBoniz/helmet-detection-dataset/resolve/main/helmet-detection-dataset.zip"
)
DEFAULT_ARCHIVE_PATH = Path("datasets/helmet-detection-dataset.zip")
DEFAULT_OUTPUT_DIR = Path("datasets/helmet")
DEFAULT_CLASSES = ("Helmet", "No-Helmet")

SPLIT_NORMALIZATION = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
IMAGE_DIR_NAMES = ("images", "image", "img")
LABEL_DIR_NAMES = ("labels", "label", "lbl")


def download_archive(
    url: str = DEFAULT_DATASET_URL,
    target_path: Path | str = DEFAULT_ARCHIVE_PATH,
) -> Path:
    """Download the dataset zip archive if it does not already exist."""
    target = Path(target_path).expanduser().resolve()
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "yolo-detection-starter/0.1.0"},
    )
    try:
        with urllib.request.urlopen(req) as response, open(temp_target, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        temp_target.replace(target)
    except Exception:
        if temp_target.exists():
            temp_target.unlink()
        raise
    return target


def parse_yaml_classes(yaml_text: str) -> list[str] | None:
    """Extract YOLO dataset class names from YAML text without external dependencies."""
    lines = yaml_text.splitlines()
    in_names = False
    names_dict: dict[int, str] = {}
    names_list: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("names:"):
            in_names = True
            content = stripped[len("names:") :].strip()
            if content.startswith("[") and content.endswith("]"):
                items = [
                    item.strip().strip("'\"")
                    for item in content[1:-1].split(",")
                    if item.strip().strip("'\"")
                ]
                if items:
                    return items
            continue

        if in_names:
            if not (line.startswith(" ") or line.startswith("\t")):
                break

            dict_match = re.match(r"^['\"]?(\d+)['\"]?\s*:\s*['\"]?(.*?)['\"]?$", stripped)
            if dict_match:
                class_id = int(dict_match.group(1))
                class_name = dict_match.group(2).strip()
                names_dict[class_id] = class_name
                continue

            list_match = re.match(r"^-\s*['\"]?(.*?)['\"]?$", stripped)
            if list_match:
                class_name = list_match.group(1).strip()
                names_list.append(class_name)
                continue

    if names_dict:
        sorted_ids = sorted(names_dict.keys())
        return [names_dict[idx] for idx in sorted_ids]
    if names_list:
        return names_list
    return None


def extract_classes_from_archive(zf: zipfile.ZipFile) -> list[str] | None:
    """Find and parse class names from a data.yaml file inside the zip archive."""
    yaml_candidates = [
        name
        for name in zf.namelist()
        if PurePosixPath(name.replace("\\", "/")).name.lower()
        in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
    ]
    for candidate in sorted(yaml_candidates, key=len):
        try:
            raw = zf.read(candidate).decode("utf-8", errors="replace")
            classes = parse_yaml_classes(raw)
            if classes:
                return classes
        except (KeyError, OSError, ValueError):
            continue
    return None


def polygons_to_boxes(label_text: str) -> str:
    """Convert YOLO segmentation lines (class x1 y1 x2 y2 ...) to box lines.

    Some images in the source dataset carry polygon labels instead of
    ``class x_center y_center width height``. A polygon line becomes the
    bounding box around its points; box lines and malformed lines are kept
    unchanged so the dataset checker still reports anything truly broken.
    """
    lines = []
    for line in label_text.splitlines():
        values = line.split()
        if len(values) >= 7 and len(values) % 2 == 1:
            try:
                coords = [float(value) for value in values[1:]]
            except ValueError:
                lines.append(line)
                continue
            xs, ys = coords[0::2], coords[1::2]
            x_min, x_max = max(min(xs), 0.0), min(max(xs), 1.0)
            y_min, y_max = max(min(ys), 0.0), min(max(ys), 1.0)
            lines.append(
                f"{values[0]} {(x_min + x_max) / 2:.6f} {(y_min + y_max) / 2:.6f} "
                f"{x_max - x_min:.6f} {y_max - y_min:.6f}"
            )
        else:
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def discover_split_pairs(
    zf: zipfile.ZipFile,
) -> dict[str, list[tuple[str, str, PurePosixPath]]]:
    """Locate train, val (or valid), and test image+label pairs inside the archive.

    Returns a mapping of normalized split name ('train', 'val', 'test') to lists
    of (image_zip_entry, label_zip_entry, relative_subpath) tuples.
    """
    namelist = zf.namelist()
    name_map = {name.replace("\\", "/").lower(): name for name in namelist if not name.endswith("/")}

    discovered: dict[str, list[tuple[str, str, PurePosixPath]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for member_name in namelist:
        if member_name.endswith("/"):
            continue
        normalized = member_name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if any(part == ".." for part in path.parts) or path.is_absolute():
            raise ValueError(f"Unsafe path in archive entry: '{member_name}'")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # Skip macOS archive metadata (__MACOSX/ folders and ._ resource forks).
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue

        parts = path.parts
        lower_parts = [p.lower() for p in parts]

        split: str | None = None
        rel_path: PurePosixPath | None = None
        image_dir_index: int | None = None

        # Check Pattern A: ... / <split> / <images> / <relative_path>
        # Check Pattern B: ... / <images> / <split> / <relative_path>
        for i in range(len(parts) - 2):
            if lower_parts[i] in SPLIT_NORMALIZATION and lower_parts[i + 1] in IMAGE_DIR_NAMES:
                split = SPLIT_NORMALIZATION[lower_parts[i]]
                rel_path = PurePosixPath(*parts[i + 2 :])
                image_dir_index = i + 1
                break
            if lower_parts[i] in IMAGE_DIR_NAMES and lower_parts[i + 1] in SPLIT_NORMALIZATION:
                split = SPLIT_NORMALIZATION[lower_parts[i + 1]]
                rel_path = PurePosixPath(*parts[i + 2 :])
                image_dir_index = i
                break

        if split is None or rel_path is None or image_dir_index is None:
            continue

        label_candidates: list[tuple[str, ...]] = [
            parts[:image_dir_index]
            + (label_dir,)
            + parts[image_dir_index + 1 : -1]
            + (f"{path.stem}.txt",)
            for label_dir in LABEL_DIR_NAMES
        ]

        matched_label_entry: str | None = None
        for cand in label_candidates:
            cand_key = "/".join(cand).lower()
            if cand_key in name_map:
                matched_label_entry = name_map[cand_key]
                break

        if matched_label_entry is None:
            expected_cand = "/".join(label_candidates[0]) if label_candidates else f"{path.stem}.txt"
            raise ValueError(
                f"Archive image '{member_name}' in split '{split}' has no matching label "
                f"file in archive (expected '{expected_cand}')."
            )

        discovered[split].append((member_name, matched_label_entry, rel_path))

    if not discovered["train"] or not discovered["val"]:
        raise ValueError(
            "Archive must contain train and val (or valid) splits with matching image and label folders."
        )

    return discovered


def sample_split_pairs(
    pairs: Sequence[tuple[str, str, PurePosixPath]],
    max_count: int | None,
    seed: int,
    split_name: str,
) -> list[tuple[str, str, PurePosixPath]]:
    """Return a deterministic seeded sample of image/label pairs."""
    sorted_pairs = sorted(pairs, key=lambda item: str(item[2]))
    if max_count is None or max_count >= len(sorted_pairs):
        return sorted_pairs
    if max_count < 0:
        raise ValueError(f"Sample count for split '{split_name}' cannot be negative: {max_count}")
    if split_name in ("train", "val") and max_count == 0:
        raise ValueError(f"Split '{split_name}' requires at least 1 image.")

    rng = random.Random(f"{seed}_{split_name}")
    shuffled = sorted_pairs.copy()
    rng.shuffle(shuffled)
    selected = shuffled[:max_count]
    return sorted(selected, key=lambda item: str(item[2]))


def download_helmet_dataset(
    archive: Path | str | None = None,
    output: Path | str | None = None,
    url: str = DEFAULT_DATASET_URL,
    class_names: Sequence[str] | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Download, extract, subset, configure, and validate the helmet detection dataset."""
    archive_path = Path(archive) if archive is not None else DEFAULT_ARCHIVE_PATH
    output_root = Path(output) if output is not None else DEFAULT_OUTPUT_DIR

    archive_path = archive_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if not archive_path.is_file():
        download_archive(url=url, target_path=archive_path)

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. "
                "Pass overwrite=True or --overwrite to replace existing contents."
            )
        for child in output_root.iterdir():
            if child.resolve() == archive_path.resolve():
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    with zipfile.ZipFile(archive_path, "r") as zf:
        if class_names is None:
            extracted_classes = extract_classes_from_archive(zf)
            effective_classes = (
                extracted_classes if extracted_classes is not None else list(DEFAULT_CLASSES)
            )
        else:
            effective_classes = list(class_names)
        effective_classes = validate_class_names(effective_classes)

        discovered = discover_split_pairs(zf)

        split_limits = {
            "train": max_train,
            "val": max_val,
            "test": max_test,
        }

        for split_name in ("train", "val", "test"):
            raw_pairs = discovered.get(split_name, [])
            if not raw_pairs:
                continue

            limit = split_limits.get(split_name)
            if split_name == "test" and limit == 0:
                continue

            selected_pairs = sample_split_pairs(raw_pairs, limit, seed, split_name)

            dest_images = output_root / "images" / split_name
            dest_labels = output_root / "labels" / split_name
            dest_images.mkdir(parents=True, exist_ok=True)
            dest_labels.mkdir(parents=True, exist_ok=True)

            for img_entry, lbl_entry, rel_path in selected_pairs:
                target_img = dest_images / rel_path
                target_lbl = dest_labels / rel_path.with_suffix(".txt")
                target_img.parent.mkdir(parents=True, exist_ok=True)
                target_lbl.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(img_entry) as src, open(target_img, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                raw_label = zf.read(lbl_entry).decode("utf-8", errors="replace")
                target_lbl.write_text(polygons_to_boxes(raw_label), encoding="utf-8")

    write_dataset_yaml(output_root, effective_classes, output_root / "data.yaml")
    return validate_dataset(output_root, effective_classes)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and prepare the DeBoniz helmet detection dataset."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help=f"Path to local dataset zip archive (default: {DEFAULT_ARCHIVE_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to extract the prepared YOLO dataset into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DATASET_URL,
        help="URL to download the dataset archive from if not present locally.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names in class-ID order (default: read from archive or Helmet,No-Helmet).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Maximum number of training images to sample (default: all).",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Maximum number of validation images to sample (default: all).",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Maximum number of test images to sample (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible subset sampling (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output directory if it already exists and is non-empty.",
    )
    return parser


def main() -> None:
    """CLI entry point for downloading and preparing the helmet dataset."""
    args = build_parser().parse_args()
    classes = (
        validate_class_names(args.classes.split(","))
        if args.classes is not None
        else None
    )
    report = download_helmet_dataset(
        archive=args.archive,
        output=args.output,
        url=args.url,
        class_names=classes,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print("Helmet dataset prepared and validated:")
    for split, counts in report.items():
        print(
            f"  {split}: {counts['images']} image(s), "
            f"{counts['objects']} labeled object(s)"
        )
    output_path = Path(args.output).expanduser().resolve()
    print(f"Dataset root: {output_path}")
    print(f"Dataset configuration: {output_path / 'data.yaml'}")


if __name__ == "__main__":
    main()
