"""Evaluate a trained detector and report standard object-detection metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    """Set up the validation run arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO detector and report precision, recall, and mAP."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Trained weights, usually runs/detect/<run-name>/weights/best.pt.",
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset data.yaml.")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to evaluate; use test for a final held-out score.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation; keep consistent with training where practical.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Images per evaluation batch; reduce if memory is limited.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device such as '0' or 'cpu'; default is automatic selection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML file does not exist: {args.data}")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("imgsz and batch must be positive.")

    # Validation uses ground-truth labels from the chosen split to compare
    # predictions with known objects; it does not update model weights.
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
    )

    # Ultralytics exposes aggregate detection metrics through metrics.box:
    # mp/mr are mean precision/recall, map50 is AP at IoU 0.50, and map is AP
    # averaged over IoU thresholds 0.50 through 0.95 in 0.05 increments.
    box_metrics = metrics.box
    summary = {
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "mAP@0.5": float(box_metrics.map50),
        "mAP@0.5:0.95": float(box_metrics.map),
        "split": args.split,
    }
    print(json.dumps(summary, indent=2))
    print(f"Detailed validation plots and results: {metrics.save_dir}")


if __name__ == "__main__":
    main()
