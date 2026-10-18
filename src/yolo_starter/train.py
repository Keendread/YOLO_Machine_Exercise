"""Train a YOLO object detector from pretrained weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    """Define the training options and explain commonly used starter values."""
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO detector.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset data.yaml.")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help=(
            "Pretrained detection weights or a model YAML. yolo11n.pt is a small, "
            "fast starting point; use larger variants for more capacity."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training passes over the dataset; 100 is a common starting point.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size in pixels; 640 is a common default trade-off.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Images per batch; lower this if the GPU runs out of memory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device such as '0' for the first GPU or 'cpu'; default is auto.",
    )
    parser.add_argument(
        "--name",
        default="yolo-starter",
        help="Run name under runs/detect/ (weights are saved inside the run folder).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help=(
            "Parent directory for training output; defaults to Ultralytics' "
            "runs/detect directory."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early-stopping patience in epochs without validation improvement.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML file does not exist: {args.data}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.patience < 0:
        raise ValueError("epochs, imgsz, and batch must be positive; patience cannot be negative.")

    # Load pretrained detection weights: fine-tuning generally converges faster
    # on a small learning dataset than starting with random weights.
    model = YOLO(args.model)

    # 640 pixels, 100 epochs, and batch 16 are practical starting values, not
    # universal requirements. Lower batch/imgsz to fit limited GPU memory and
    # tune the epoch budget using validation performance.
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(args.project) if args.project is not None else None,
        name=args.name,
    )


if __name__ == "__main__":
    main()
