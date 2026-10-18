"""Split an unsplit YOLO dataset, validate it, and create its data YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo_starter.dataset import split_dataset, validate_class_names


def build_parser() -> argparse.ArgumentParser:
    """Set up arguments for dataset validation and configuration generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Split and validate an unsplit YOLO detection dataset, then write data.yaml."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Source dataset root containing images/ and matching labels/ directories.",
    )
    parser.add_argument(
        "--classes",
        required=True,
        help="Comma-separated class names in class-ID order, e.g. 'person,car,bicycle'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to create split folders (default: <source name>_split beside source).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images assigned to training (default: 0.8).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction assigned to validation (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction assigned to held-out testing (default: 0.1; use 0 to omit test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to make the random split reproducible (default: 0).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    class_names = validate_class_names(args.classes.split(","))
    output_root = args.output or args.root.parent / f"{args.root.name}_split"
    report = split_dataset(
        args.root,
        output_root,
        class_names,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )

    print("Dataset validation and split passed:")
    for split, counts in report.items():
        print(
            f"  {split}: {counts['images']} image(s), "
            f"{counts['objects']} labeled object(s)"
        )
    print(f"Split dataset created at: {output_root.resolve()}")
    print(f"Dataset configuration written to: {(output_root / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
