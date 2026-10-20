"""Tkinter desktop app that screens images for riders with and without helmets.

Run it with ``python -m yolo_starter.helmet_app``.

Importing this module is cheap and side-effect free: Tk, Pillow and ultralytics
are imported only when the window is built or a model is loaded, and the model
itself is loaded only after the user clicks **Run detection**. Inference runs
in a worker thread that talks to the Tk thread through a queue polled with
``after``, so the window stays responsive.

The batch logic (``run_batch``, ``process_image``) has no Tk dependency so it
can be unit tested with a fake model.
"""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from yolo_starter.helmet_report import (
    DEFAULT_CLASS_MAP,
    ImageReport,
    summarize_image,
    summary_text,
    total_report,
    write_csv,
)

APP_NAME = "Helmet Checker"

SCREENING_NOTE = (
    "Detections are for screening only and need human review before any action is taken."
)

DEFAULT_CONFIDENCE = 0.25
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
POLL_MS = 100
MAX_STORED_SIDE = 1600  # annotated images are downsized so a big batch stays in memory


class ModelError(Exception):
    """The weights are missing or the model could not be loaded."""


@dataclass
class ImageResult:
    """Outcome for one image: a report and annotated image, or an error message."""

    path: str
    report: ImageReport | None = None
    annotated: Any = None  # PIL.Image.Image, or None when not rendered
    error: str | None = None


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def parse_confidence(text: str) -> float:
    """Parse the confidence field; raises ValueError unless it is a number in [0, 1]."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"Confidence must be a number between 0 and 1, not {text!r}.") from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Confidence must be between 0 and 1, not {text!r}.")
    return value


def fit_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """Largest size that fits the box with the same aspect ratio; never enlarges."""
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, int(width * scale)), max(1, int(height * scale))


BUNDLED_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "helmet-yolo11n.pt"


def default_weights() -> Path:
    """Suggest the newest runs/detect/**/weights/best.pt, then the bundled model,
    else the default train location."""
    candidates = list(Path("runs/detect").glob("**/weights/best.pt"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()
    if BUNDLED_WEIGHTS.is_file():
        return BUNDLED_WEIGHTS
    return Path("runs/detect/yolo-starter/weights/best.pt").resolve()


def load_model(weights: str | Path) -> Any:
    """Load YOLO weights, turning every failure into a ``ModelError`` with a clear message."""
    weights = Path(weights).expanduser()
    if not weights.is_file():
        raise ModelError(f"Model weights not found:\n{weights}")
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
    except Exception as exc:
        raise ModelError(
            f"Could not load the model from:\n{weights}\n\n{type(exc).__name__}: {exc}"
        ) from exc
    names = getattr(model, "names", None) or {}
    if not any(summarize_image("", [(name, 1.0)]).total_riders for name in names.values()):
        raise ModelError(
            f"The model has no helmet classes (it knows: {', '.join(map(str, names.values())) or 'none'}).\n"
            "Choose weights trained on the Helmet / No-Helmet dataset."
        )
    return model


def _detections(result: Any) -> list[tuple[str, float]]:
    boxes = result.boxes
    if boxes is None:
        return []
    names = result.names
    return [(names[int(cls)], float(conf)) for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist())]


def render_annotated(result: Any) -> Any:
    """Draw the detections and return a (downsized) RGB PIL image."""
    from PIL import Image

    bgr = result.plot()
    image = Image.fromarray(bgr[..., ::-1])
    image.thumbnail((MAX_STORED_SIDE, MAX_STORED_SIDE))
    return image


def process_image(
    model: Any,
    path: str,
    confidence: float,
    render: Callable[[Any], Any] = render_annotated,
) -> ImageResult:
    """Run the model on one image. Never raises: failures become ``ImageResult.error``."""
    try:
        if not Path(path).is_file():
            raise FileNotFoundError("file not found")
        results = model.predict(source=path, conf=confidence, verbose=False)
        if not results:  # ultralytics skips images it cannot decode instead of raising
            raise ValueError("the image could not be read")
        result = results[0]
        report = summarize_image(path, _detections(result))
        return ImageResult(path, report=report, annotated=render(result))
    except Exception as exc:
        return ImageResult(path, error=f"{type(exc).__name__}: {exc}")


def run_batch(
    paths: list[str],
    get_model: Callable[[], Any],
    confidence: float,
    emit: Callable[[tuple], None],
    cancel: threading.Event,
    render: Callable[[Any], Any] = render_annotated,
) -> None:
    """Worker-thread body. Reports progress through ``emit``; always ends with ("done", ...)."""
    cancelled = False
    try:
        emit(("status", "Loading model..."))
        try:
            model = get_model()
        except ModelError as exc:
            emit(("model_error", str(exc)))
            return
        for index, path in enumerate(paths):
            if cancel.is_set():
                cancelled = True
                break
            emit(("progress", index, len(paths), Path(path).name))
            emit(("result", index, process_image(model, path, confidence, render)))
    except Exception as exc:  # keep the UI from waiting forever on an unexpected failure
        emit(("model_error", f"{type(exc).__name__}: {exc}"))
    finally:
        emit(("done", cancelled))


class HelmetCheckerApp:
    """Window: weights picker, image list, annotated preview, summary and CSV export."""

    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.root = root
        root.title(APP_NAME)
        root.geometry("1100x720")
        root.minsize(760, 520)

        self.paths: list[str] = []
        self.results: dict[int, ImageResult] = {}
        self.model: Any = None
        self.model_key: tuple[str, float] | None = None
        self.messages: queue.Queue[tuple] = queue.Queue()
        self.cancel = threading.Event()
        self.running = False
        self.model_failed = False
        self.shown: Any = None  # PIL image currently previewed
        self.photo: Any = None  # keeps the Tk image alive

        self.weights = tk.StringVar(value=str(default_weights()))
        self.confidence = tk.StringVar(value=f"{DEFAULT_CONFIDENCE:.2f}")
        self.status = tk.StringVar(value="Add images, then click Run detection.")
        self.summary = tk.StringVar(value="")
        self.batch = tk.StringVar(value="")

        self._build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(POLL_MS, self._poll)

    # -- layout ----------------------------------------------------------

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Model weights:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.weights_entry = ttk.Entry(top, textvariable=self.weights)
        self.weights_entry.grid(row=0, column=1, sticky="ew")
        self.browse_button = ttk.Button(top, text="Browse...", command=self.browse_weights)
        self.browse_button.grid(row=0, column=2, padx=(8, 0))

        buttons = ttk.Frame(top)
        buttons.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.add_button = ttk.Button(buttons, text="Add images...", command=self.add_images)
        self.clear_button = ttk.Button(buttons, text="Clear", command=self.clear)
        self.add_button.pack(side=tk.LEFT)
        self.clear_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(buttons, text="Confidence:").pack(side=tk.LEFT, padx=(16, 4))
        self.confidence_box = ttk.Spinbox(
            buttons, from_=0.0, to=1.0, increment=0.05, width=6, format="%.2f",
            textvariable=self.confidence,
        )
        self.confidence_box.pack(side=tk.LEFT)
        self.run_button = ttk.Button(buttons, text="Run detection", command=self.run)
        self.run_button.pack(side=tk.LEFT, padx=(16, 0))
        self.cancel_button = ttk.Button(
            buttons, text="Cancel", command=self.cancel.set, state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(8, 0))
        self.export_button = ttk.Button(
            buttons, text="Export CSV", command=self.export_csv, state=tk.DISABLED
        )
        self.export_button.pack(side=tk.RIGHT)

        ttk.Label(
            self.root, text=SCREENING_NOTE, foreground="#8a4b00", padding=(10, 0, 10, 6)
        ).pack(fill=tk.X)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10)
        left = ttk.Frame(body)
        self.tree = ttk.Treeview(
            left, columns=("summary",), show="tree headings", selectmode="browse"
        )
        self.tree.heading("#0", text="Image")
        self.tree.heading("summary", text="Result")
        self.tree.column("#0", width=200)
        self.tree.column("summary", width=220)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        body.add(left, weight=1)

        right = ttk.Frame(body)
        self.preview = ttk.Label(
            right, text="The annotated image appears here.", anchor=tk.CENTER, relief=tk.SUNKEN
        )
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview.bind("<Configure>", lambda _event: self._show_current())
        ttk.Label(right, textvariable=self.summary, font=("TkDefaultFont", 11, "bold")).pack(
            fill=tk.X, pady=(6, 0)
        )
        ttk.Label(right, textvariable=self.batch).pack(fill=tk.X)
        body.add(right, weight=3)

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=(8, 0))
        ttk.Label(self.root, textvariable=self.status, padding=(10, 2, 10, 8)).pack(fill=tk.X)

    # -- user actions ----------------------------------------------------

    def browse_weights(self) -> None:
        chosen = self.filedialog.askopenfilename(
            title="Select trained YOLO weights",
            filetypes=[("PyTorch model weights", "*.pt"), ("All files", "*.*")],
        )
        if chosen:
            self.weights.set(chosen)

    def add_images(self) -> None:
        patterns = " ".join(f"*{ext} *{ext.upper()}" for ext in IMAGE_EXTENSIONS)
        # Some Tk builds return one selection as a plain string, not a tuple.
        chosen = self.root.tk.splitlist(self.filedialog.askopenfilenames(
            title="Add images", filetypes=[("Images", patterns), ("All files", "*.*")]
        ))
        existing = set(self.paths)
        new = []
        for path in map(str, chosen):
            if path not in existing and is_supported_image(path):
                existing.add(path)
                new.append(path)
        skipped = len(chosen) - len(new)
        for path in new:
            self.paths.append(path)
            self.tree.insert("", "end", iid=str(len(self.paths) - 1), text=Path(path).name,
                             values=("Queued",))
        self.status.set(
            f"{len(self.paths)} image(s) queued"
            + (f"; skipped {skipped} duplicate or unsupported file(s)." if skipped else ".")
        )

    def clear(self) -> None:
        self.paths.clear()
        self.results.clear()
        self.tree.delete(*self.tree.get_children())
        self.shown = None
        self._show_current()
        self.summary.set("")
        self.batch.set("")
        self.progress.configure(value=0)
        self.export_button.configure(state=self.tk.DISABLED)
        self.status.set("Add images, then click Run detection.")

    def run(self) -> None:
        if self.running:
            return
        if not self.paths:
            self.messagebox.showwarning(APP_NAME, "Add at least one image first.")
            return
        try:
            confidence = parse_confidence(self.confidence.get())
        except ValueError as exc:
            self.messagebox.showerror("Invalid confidence", str(exc))
            return
        weights = Path(self.weights.get()).expanduser()
        if not weights.is_file():
            self.messagebox.showerror("Model weights not found", f"Choose a valid .pt file:\n{weights}")
            return

        self.results.clear()
        for iid in self.tree.get_children():
            self.tree.set(iid, "summary", "Queued")
        self._on_select()
        self.batch.set("")
        self.progress.configure(value=0, maximum=len(self.paths))
        self._set_running(True)
        self.cancel.clear()
        self.model_failed = False
        threading.Thread(
            target=run_batch,
            args=(list(self.paths), lambda: self._get_model(weights), confidence,
                  self.messages.put, self.cancel),
            daemon=True,
        ).start()

    def _get_model(self, weights: Path) -> Any:
        """Runs on the worker thread; reloads only when the weights file changed."""
        key = (str(weights.resolve()), weights.stat().st_mtime)
        if self.model is None or self.model_key != key:
            self.model = None
            self.model = load_model(weights)
            self.model_key = key
        return self.model

    def export_csv(self) -> None:
        reports = [self.results[i].report for i in sorted(self.results) if self.results[i].report]
        if not reports:
            return
        destination = self.filedialog.asksaveasfilename(
            title="Export CSV", defaultextension=".csv", initialfile="helmet_report.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not destination:
            return
        try:
            write_csv(reports, destination)
        except OSError as exc:
            self.messagebox.showerror("Could not save CSV", f"{destination}\n\n{exc}")
            return
        self.status.set(f"Saved {len(reports)} row(s) plus a TOTAL row to {destination}")

    # -- worker messages -------------------------------------------------

    def _poll(self) -> None:
        try:
            while True:
                self._handle(self.messages.get_nowait())
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._poll)

    def _handle(self, message: tuple) -> None:
        kind = message[0]
        if kind == "status":
            self.status.set(message[1])
        elif kind == "progress":
            _, index, total, name = message
            self.progress.configure(value=index)
            self.status.set(f"Processing {index + 1} of {total}: {name}")
        elif kind == "result":
            self._store_result(message[1], message[2])
        elif kind == "model_error":
            self.model_failed = True
            self.messagebox.showerror("Model error", message[1])
            self.status.set("Detection failed: model error.")
        elif kind == "done":
            self._finish(cancelled=message[1])

    def _store_result(self, index: int, result: ImageResult) -> None:
        self.results[index] = result
        iid = str(index)
        if result.report is not None:
            self.tree.set(iid, "summary", summary_text(result.report))
        else:
            self.tree.set(iid, "summary", f"Error: {result.error}")
        self.progress.configure(value=index + 1)
        if not self.tree.selection():
            self.tree.selection_set(iid)
        elif self.tree.selection()[0] == iid:
            self._on_select()

    def _finish(self, cancelled: bool) -> None:
        self._set_running(False)
        failed = [r for r in self.results.values() if r.error]
        ok = len(self.results) - len(failed)
        self._update_batch()
        if ok:
            self.export_button.configure(state=self.tk.NORMAL)
        if not self.model_failed:
            self.status.set(
                f"{'Cancelled' if cancelled else 'Done'}: {ok} image(s) checked"
                + (f", {len(failed)} failed." if failed else ".")
            )
        if failed:
            lines = [f"{Path(r.path).name}: {r.error}" for r in failed[:5]]
            if len(failed) > 5:
                lines.append(f"... and {len(failed) - 5} more")
            self.messagebox.showerror("Some images could not be processed", "\n".join(lines))

    def _update_batch(self) -> None:
        reports = [r.report for r in self.results.values() if r.report]
        if reports:
            self.batch.set(
                f"Batch total ({len(reports)} images): {summary_text(total_report(reports))}"
            )

    def _set_running(self, running: bool) -> None:
        self.running = running
        idle = self.tk.DISABLED if running else self.tk.NORMAL
        for widget in (self.add_button, self.clear_button, self.run_button, self.browse_button,
                       self.weights_entry, self.confidence_box):
            widget.configure(state=idle)
        self.cancel_button.configure(state=self.tk.NORMAL if running else self.tk.DISABLED)
        if running:
            self.export_button.configure(state=self.tk.DISABLED)

    # -- preview ---------------------------------------------------------

    def _on_select(self, _event: object = None) -> None:
        selection = self.tree.selection()
        result = self.results.get(int(selection[0])) if selection else None
        if result is None:
            self.shown = None
            self.summary.set("")
        elif result.report is None:
            self.shown = None
            self.summary.set(f"Could not process {Path(result.path).name}: {result.error}")
        else:
            self.shown = result.annotated
            self.summary.set(f"{Path(result.path).name}: {summary_text(result.report)}")
        self._show_current()

    def _show_current(self) -> None:
        if self.shown is None:
            self.photo = None
            self.preview.configure(image="", text="The annotated image appears here.")
            return
        from PIL import ImageTk

        size = fit_size(*self.shown.size, max(self.preview.winfo_width() - 4, 1),
                        max(self.preview.winfo_height() - 4, 1))
        image = self.shown if size == self.shown.size else self.shown.resize(size)
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text="")

    def close(self) -> None:
        self.cancel.set()
        self.root.destroy()


def main() -> None:
    import tkinter as tk

    root = tk.Tk()
    HelmetCheckerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
