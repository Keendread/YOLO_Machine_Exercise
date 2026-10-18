"""Run object detection on images, folders, video files, or webcam streams."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _source_value(value: str) -> str | int:
    """Convert a numeric camera ID while leaving paths and stream URLs untouched."""
    if value.isdecimal():
        return int(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    """Set up model inference arguments."""
    parser = argparse.ArgumentParser(
        description="Detect objects and report each class, confidence, and bounding box."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Trained detector weights, usually the run's best.pt file.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image, image directory, video, stream URL, or camera ID (for example, 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold; 0.25 is the common prediction default.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels; 640 is a common default.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device such as '0' or 'cpu'; default is automatic selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detect/predict"),
        help="Directory in which to save annotated image/video predictions.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a live preview window (may not work in headless environments).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {args.weights}")
    if not 0 <= args.conf <= 1:
        raise ValueError("conf must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("imgsz must be positive.")

    # 0.25 is the common prediction confidence threshold. Raising it filters
    # more low-confidence boxes; lowering it returns more candidates.
    model = YOLO(str(args.weights))
    # Streaming results avoids holding every video frame's prediction in
    # memory. Saving and drawing labels/confidence make this a usable demo app.
    results = model.predict(
        source=_source_value(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        show=args.show,
        show_labels=True,
        show_conf=True,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        stream=True,
    )

    for result in results:
        boxes = result.boxes
        print(f"\n{result.path}")
        if boxes is None or len(boxes) == 0:
            print("  No detections.")
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            print(
                f"  {class_name} ({confidence:.3f}) "
                f"box_xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

    print(f"\nAnnotated output saved under: {args.output}")


if __name__ == "__main__":
    main()
