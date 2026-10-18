# YOLO Object Detection Starter

A small, commented learning project for **object detection**. A detection output
contains a class label, a confidence score, and a bounding box for each detected
object; this is different from image classification, which assigns labels to an
entire image without locating individual objects.

This project uses [Ultralytics YOLO](https://docs.ultralytics.com/) for the model
and keeps the main stages separate so each one can be learned, run, and changed
independently:

1. Prepare and validate a labeled dataset.
2. Train a detector using pretrained weights.
3. Evaluate detection metrics on validation or held-out test images.
4. Run the trained detector against images or video and save annotated output.

## Project layout

```text
src/yolo_starter/
  dataset.py          Dataset format checks and data.yaml writer
  prepare_dataset.py  Dataset preparation command
  train.py            Model training command
  evaluate.py         Precision, recall, and mAP evaluation command
  predict.py          Image/video inference application
tests/
  test_dataset.py     Dataset-helper unit tests (no model download required)
```

## 1. Install

Python 3.10 or newer is recommended. From the project directory, create and
activate a virtual environment, then install the project and its dependency:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Ultralytics will download pretrained weights the first time a model such as
`yolo11n.pt` is loaded. Training performance depends on the available hardware;
the `--device` option accepts an Ultralytics device value such as `0` for the
first CUDA GPU or `cpu`. Leaving it unspecified lets Ultralytics choose.

## 2. Prepare the dataset

This starter expects images and labels already annotated in the Ultralytics
**YOLO detection format**. Before preparing the dataset, place all annotated
images and labels in one source folder using this unsplit structure:

```text
my_dataset/
  images/
    scene-01.jpg
    scene-02.jpg
    ...
  labels/
    scene-01.txt
    scene-02.txt
    ...
```

Each image must have a matching `.txt` label file at the same relative path.
For example, `images/scene-01.jpg` pairs with `labels/scene-01.txt`. Nested
subdirectories are allowed when the label directory mirrors the image
directory. The checker supports common image formats (`jpg`,
`png`, `webp`, and others) and nested subdirectories as long as the labels
mirror the image paths.

Each non-empty label-file line represents one object and has **five values**:

```text
class_id x_center y_center width height
```

- `class_id` is a zero-based integer. Its meaning comes from the class-name
  order supplied to the preparation command: the first class is `0`, the
  second is `1`, and so on.
- `x_center`, `y_center`, `width`, and `height` are normalized to the image
  width/height, so they are finite numbers from `0` to `1`. Coordinates refer
  to the box center and box dimensions, not the top-left corner.
- An image with no objects still needs a matching **empty** `.txt` file.
- Do not include headers, confidence values, or pixel coordinates in label files.

Run the preparation command from the project directory. Class names and their
order must exactly match the IDs used by your annotations:

```powershell
python -m yolo_starter.prepare_dataset --root .\my_dataset --classes "person,car,bicycle"
```

By default, this validates the source labels and **copies** images and labels
into a sibling `my_dataset_split/` folder with an 80% train, 10% validation,
and 10% test split. The original `my_dataset/` stays unchanged. The command
also writes `my_dataset_split/data.yaml`, which is the file to pass to training.
The split is repeatable (`--seed 0` by default); it is random by image and does
not attempt class stratification. Keep related or near-duplicate images (such
as consecutive frames from one video) together before splitting to avoid
leakage between splits.

Change the ratios with `--train-ratio`, `--val-ratio`, and `--test-ratio`.
They must add up to `1.0`; train and validation must each be greater than zero.
Set the test ratio to `0` to skip creating a test split. For example, to use
80% training and 20% validation:

```powershell
python -m yolo_starter.prepare_dataset --root .\my_dataset --classes "person,car,bicycle" --train-ratio 0.8 --val-ratio 0.2 --test-ratio 0
```

You can choose a different destination with `--output .\prepared_dataset`. It
must not be inside or above the source folder, and an existing non-empty output
folder is left untouched and rejected to prevent accidental overwrites. Every
requested non-zero split needs at least one image, so very small datasets may
require fewer splits or different ratios.

## 3. Train

```powershell
python -m yolo_starter.train --data .\my_dataset_split\data.yaml
```

Common options are exposed as command-line arguments:

```powershell
python -m yolo_starter.train --data .\my_dataset_split\data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16
```

- `yolo11n.pt` is a small pretrained detection model and a practical starting
  point for an exercise. The `n` variant favors speed and smaller hardware;
  larger variants generally require more memory and compute. See the
  [Ultralytics YOLO model documentation](https://docs.ultralytics.com/models/).
- `epochs=100` is a common initial training budget, not a guarantee of
  convergence. Use validation results and the dataset size to decide whether
  to train longer or stop sooner.
- `imgsz=640` is a common image-size starting value. Larger inputs can help
  detect small objects but need more memory and time.
- `batch=16` is a common starting batch size. Reduce it if memory is exhausted;
  a larger batch is not automatically better.
- `patience=100` is the early-stopping patience in epochs without sufficient
  validation improvement. Ultralytics can stop earlier when training plateaus.

The run is stored under `runs/detect/yolo-starter/` by default. Its
`weights/best.pt` is typically the best checkpoint according to validation
fitness; `weights/last.pt` is the final epoch checkpoint.

## 4. Evaluate

Evaluate on the validation split during development:

```powershell
python -m yolo_starter.evaluate --weights .\runs\detect\yolo-starter\weights\best.pt --data .\my_dataset_split\data.yaml --split val
```

After model choices are final, evaluate once on the held-out test split (if it
exists):

```powershell
python -m yolo_starter.evaluate --weights .\runs\detect\yolo-starter\weights\best.pt --data .\my_dataset_split\data.yaml --split test
```

The program prints:

- **Precision**: proportion of predicted detections that are correct.
- **Recall**: proportion of ground-truth objects that are found.
- **mAP@0.5**: mean average precision at an intersection-over-union (IoU)
  threshold of `0.50`.
- **mAP@0.5:0.95**: mean average precision averaged across IoU thresholds from
  `0.50` through `0.95` in `0.05` increments. This stricter aggregate is often
  referred to as COCO-style mAP.

Ultralytics computes these metrics during validation and saves detailed
validation outputs and plots in the run output directory. Precision and recall
depend on the confidence threshold and validation setup; compare evaluations
made with the same dataset and settings. Keep the test split out of training and
repeated tuning so it remains a fair final check.

## 5. Use the model on images and video

Detect objects in a single image:

```powershell
python -m yolo_starter.predict --weights .\runs\detect\yolo-starter\weights\best.pt --source .\example.jpg
```

The same command accepts an image directory, video file, stream URL, or webcam
ID. For example:

```powershell
python -m yolo_starter.predict --weights .\runs\detect\yolo-starter\weights\best.pt --source .\input-video.mp4
python -m yolo_starter.predict --weights .\runs\detect\yolo-starter\weights\best.pt --source 0 --show
```

Every detection is printed with its **class name**, **confidence score**, and
pixel bounding-box coordinates `(x1, y1, x2, y2)`. Annotated predictions,
including visible labels and confidence values, are saved under
`runs/detect/predict/` by default.

### Live detection on a mirrored application window (Windows)

The desktop app can select a visible application window, including a
[scrcpy](https://github.com/Genymobile/scrcpy) tablet mirror, and show YOLO's
annotated live preview. It captures the selected application window directly
via the Windows Graphics Capture API, independently of the window's screen
position or other windows covering it. The model inference runs in a worker
thread separate from the capture callback. It keeps only the newest waiting
frame, dropping stale frames instead of accumulating delay when inference is
slower than the scrcpy refresh rate. The default confidence threshold is `0.25`.

Start scrcpy and mirror the tablet, then launch the app from the project root:

```powershell
python -m yolo_starter.window_app
```

Choose the trained `best.pt` file (the app suggests the newest one it finds in
`runs/detect`), select the scrcpy window, adjust confidence if needed, and click
**Start detection**. Click **Stop** to end the live preview. If `windows-capture`
is not installed in the environment, install the project dependencies with
`python -m pip install -e .`.

Windows Graphics Capture can capture the target window separately from the
desktop composition. Some apps pause rendering while minimized, so restore
scrcpy if the preview freezes or becomes black.

`conf=0.25` is the common Ultralytics prediction confidence threshold. Lowering
it can show more candidate detections (including more false positives);
increasing it usually filters more detections (and may miss real objects).
This is an application threshold, not a substitute for evaluation:

```powershell
python -m yolo_starter.predict --weights .\runs\detect\yolo-starter\weights\best.pt --source .\example.jpg --conf 0.4
```

## Run the lightweight tests

The dataset checks use only the Python standard library, so they do not download
model weights:

```powershell
python -m unittest discover -s tests -v
```

## Official Ultralytics references

- [Object detection dataset format](https://docs.ultralytics.com/datasets/detect/)
- [Model training](https://docs.ultralytics.com/modes/train/)
- [Model validation and metrics](https://docs.ultralytics.com/modes/val/)
- [Model prediction](https://docs.ultralytics.com/modes/predict/)
- [Configuration arguments](https://docs.ultralytics.com/usage/cfg/)
