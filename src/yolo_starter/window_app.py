"""Live YOLO testing app for a selected Windows application window."""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.engine.results import Results
from windows_capture import CaptureControl, Frame, InternalCaptureControl, WindowsCapture


if os.name == "nt":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
else:
    _user32 = None


_ENUM_WINDOWS_PROC = (
    ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    if os.name == "nt"
    else None
)

if _user32 is not None:
    # HWND values are pointer-sized on 64-bit Windows; declare Win32 signatures
    # so ctypes does not truncate handles to its default 32-bit integer type.
    _user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    _user32.IsWindowVisible.restype = ctypes.c_bool
    _user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.EnumWindows.argtypes = [_ENUM_WINDOWS_PROC, ctypes.c_void_p]
    _user32.EnumWindows.restype = ctypes.c_bool


@dataclass(frozen=True)
class WindowEntry:
    """A visible desktop window, represented by its handle and display title."""

    handle: int
    title: str

    def __str__(self) -> str:
        return self.title


def list_windows() -> list[WindowEntry]:
    """Return titled, visible top-level windows on Windows, including minimized ones."""
    if _user32 is None:
        raise RuntimeError("The window testing app currently supports Windows only.")

    windows: list[WindowEntry] = []

    def collect_window(handle: int, _parameter: int) -> bool:
        if not _user32.IsWindowVisible(handle):
            return True

        title_length = _user32.GetWindowTextLengthW(handle)
        if title_length <= 0:
            return True

        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        _user32.GetWindowTextW(handle, title_buffer, len(title_buffer))
        title = title_buffer.value.strip()
        if title:
            windows.append(WindowEntry(handle=int(handle), title=title))
        return True

    callback = _ENUM_WINDOWS_PROC(collect_window)
    if not _user32.EnumWindows(callback, 0):
        error_code = ctypes.get_last_error()
        if error_code:
            raise ctypes.WinError(error_code)

    return sorted(windows, key=lambda window: window.title.casefold())


class WindowTestingApp:
    """Tkinter interface for choosing a desktop window and running YOLO."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLO Live Window Tester")
        self.root.geometry("1000x720")
        self.root.minsize(720, 480)

        self.window_choices: list[WindowEntry] = []
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.results: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=2)
        self.capture_control: CaptureControl | None = None
        self.capture_lock = threading.Lock()
        self.preview_image: ImageTk.PhotoImage | None = None
        self.running = False

        self.model_path = tk.StringVar(value=str(self._default_weights()))
        self.confidence = tk.DoubleVar(value=0.25)
        self.selected_window = tk.StringVar()
        self.status = tk.StringVar(value="Choose model weights and refresh the window list.")

        self._build_interface()
        self.refresh_windows()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._process_worker_messages)

    @staticmethod
    def _default_weights() -> Path:
        """Suggest the most recently modified best.pt beneath runs/detect."""
        candidates = list(Path("runs/detect").glob("**/weights/best.pt"))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime).resolve()
        return Path("runs/detect/yolo-starter/weights/best.pt").resolve()

    def _build_interface(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Model weights:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.weights_entry = ttk.Entry(controls, textvariable=self.model_path)
        self.weights_entry.grid(row=0, column=1, sticky="ew")
        self.browse_button = ttk.Button(controls, text="Browse...", command=self.browse_weights)
        self.browse_button.grid(row=0, column=2, padx=(8, 0))

        ttk.Label(controls, text="Application window:").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(10, 0)
        )
        self.window_picker = ttk.Combobox(
            controls, textvariable=self.selected_window, state="readonly"
        )
        self.window_picker.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        self.refresh_button = ttk.Button(
            controls, text="Refresh", command=self.refresh_windows
        )
        self.refresh_button.grid(row=1, column=2, padx=(8, 0), pady=(10, 0))

        ttk.Label(controls, text="Confidence threshold:").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(10, 0)
        )
        confidence_controls = ttk.Frame(controls)
        confidence_controls.grid(row=2, column=1, sticky="ew", pady=(10, 0))
        ttk.Scale(
            confidence_controls,
            from_=0.05,
            to=0.95,
            variable=self.confidence,
            orient=tk.HORIZONTAL,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.confidence_value = ttk.Label(confidence_controls, width=5, anchor="e")
        self.confidence_value.pack(side=tk.LEFT, padx=(8, 0))
        self.confidence.trace_add("write", self._update_confidence_label)
        self._update_confidence_label()

        self.start_button = ttk.Button(
            controls, text="Start detection", command=self.start_detection
        )
        self.start_button.grid(row=3, column=1, sticky="w", pady=(12, 0))
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self.stop_detection, state=tk.DISABLED
        )
        self.stop_button.grid(row=3, column=1, sticky="w", padx=(120, 0), pady=(12, 0))

        self.preview = ttk.Label(
            self.root,
            text="Live annotated preview appears here.",
            anchor=tk.CENTER,
            relief=tk.SUNKEN,
        )
        self.preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        ttk.Label(self.root, textvariable=self.status, padding=(10, 0, 10, 10)).pack(
            fill=tk.X
        )

    def _update_confidence_label(self, *_args: object) -> None:
        self.confidence_value.configure(text=f"{self.confidence.get():.2f}")

    def browse_weights(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select trained YOLO weights",
            filetypes=[("PyTorch model weights", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.model_path.set(selected)

    def refresh_windows(self) -> None:
        try:
            self.window_choices = list_windows()
        except (OSError, RuntimeError) as exc:
            messagebox.showerror("Could not list windows", str(exc))
            return

        self.window_picker["values"] = [window.title for window in self.window_choices]
        if self.window_choices:
            self.window_picker.current(0)
            self.status.set(f"Found {len(self.window_choices)} visible application window(s).")
        else:
            self.selected_window.set("")
            self.status.set("No visible application windows with titles were found.")

    def start_detection(self) -> None:
        if self.running:
            return
        weights = Path(self.model_path.get()).expanduser()
        if not weights.is_file():
            messagebox.showerror("Model not found", f"Choose a valid .pt weights file:\n{weights}")
            return

        index = self.window_picker.current()
        if index < 0 or index >= len(self.window_choices):
            messagebox.showerror("No window selected", "Refresh the list and select a window.")
            return

        selected = self.window_choices[index]
        self.stop_event.clear()
        self.running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.weights_entry.configure(state=tk.DISABLED)
        self.browse_button.configure(state=tk.DISABLED)
        self.window_picker.configure(state=tk.DISABLED)
        self.refresh_button.configure(state=tk.DISABLED)
        self.status.set("Loading model and starting live detection...")

        self.worker = threading.Thread(
            target=self._run_detector,
            args=(str(weights.resolve()), selected, self.confidence.get()),
            daemon=True,
        )
        self.worker.start()

    def _run_detector(self, weights: str, window: WindowEntry, confidence: float) -> None:
        inference_frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        inference_thread: threading.Thread | None = None

        def stop_capture() -> None:
            self.stop_event.set()
            with self.capture_lock:
                if self.capture_control is not None:
                    self.capture_control.stop()

        try:
            model = YOLO(weights)
            # Windows Capture targets the actual HWND using the Windows
            # Graphics Capture API; it captures the selected window regardless
            # of its desktop position or other windows covering it.
            capture = WindowsCapture(
                cursor_capture=False,
                draw_border=False,
                window_hwnd=window.handle,
            )

            @capture.event
            def on_frame_arrived(
                frame: Frame, capture_control: InternalCaptureControl
            ) -> None:
                if self.stop_event.is_set():
                    capture_control.stop()
                    return

                try:
                    # Copy the native frame before returning from the callback.
                    # The one-frame queue discards stale frames if inference
                    # cannot keep up, keeping latency bounded instead of building
                    # a growing backlog.
                    latest_frame = frame.frame_buffer[:, :, :3].copy()
                    try:
                        inference_frames.put_nowait(latest_frame)
                    except queue.Full:
                        try:
                            inference_frames.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            inference_frames.put_nowait(latest_frame)
                        except queue.Full:
                            # The inference worker consumed the old frame while
                            # this callback was replacing it; the next callback
                            # will provide another current frame.
                            pass
                except Exception as exc:
                    self._publish(("error", f"{type(exc).__name__}: {exc}"))
                    capture_control.stop()

            @capture.event
            def on_closed() -> None:
                self.stop_event.set()
                self._publish(("stopped", None))

            def run_inference() -> None:
                while not self.stop_event.is_set():
                    try:
                        bgr_frame = inference_frames.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    try:
                        prediction = next(
                            iter(
                                model.predict(
                                    source=bgr_frame,
                                    conf=confidence,
                                    imgsz=640,
                                    verbose=False,
                                    stream=False,
                                )
                            )
                        )
                        if not isinstance(prediction, Results):
                            raise TypeError(
                                "YOLO returned "
                                f"{type(prediction).__name__}, expected Results."
                            )
                        self._publish(("frame", prediction.plot()))
                    except Exception as exc:
                        self._publish(("error", f"{type(exc).__name__}: {exc}"))
                        stop_capture()
                        return

            inference_thread = threading.Thread(
                target=run_inference,
                name="yolo-inference",
                daemon=True,
            )
            inference_thread.start()

            capture_control = capture.start_free_threaded()
            with self.capture_lock:
                self.capture_control = capture_control
                if self.stop_event.is_set():
                    capture_control.stop()
            capture_control.wait()
            self.stop_event.set()
            inference_thread.join()
        except Exception as exc:
            # Send the failure to the UI thread so it is visible to the user.
            self._publish(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self.stop_event.set()
            if inference_thread is not None and inference_thread.is_alive():
                inference_thread.join()
            with self.capture_lock:
                self.capture_control = None
            self._publish(("stopped", None))

    def _publish(self, message: tuple[str, object]) -> None:
        """Keep only the latest preview frame so a slow model cannot build a backlog."""
        if message[0] != "frame":
            # Preserve failures and shutdown signals rather than letting a
            # newer preview frame overwrite them in a full queue.
            while True:
                try:
                    self.results.put_nowait(message)
                    return
                except queue.Full:
                    try:
                        self.results.get_nowait()
                    except queue.Empty:
                        pass

        try:
            self.results.put_nowait(message)
        except queue.Full:
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            try:
                self.results.put_nowait(message)
            except queue.Full:
                pass

    def _process_worker_messages(self) -> None:
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == "frame":
                    self._show_frame(payload)
                elif kind == "error":
                    self.status.set(str(payload))
                    messagebox.showerror("Detection stopped", str(payload))
                elif kind == "stopped":
                    self._set_stopped()
        except queue.Empty:
            pass

        self.root.after(100, self._process_worker_messages)

    def _show_frame(self, image: object) -> None:
        frame = image
        if not isinstance(frame, np.ndarray):
            return

        available_width = max(self.preview.winfo_width(), 1)
        available_height = max(self.preview.winfo_height(), 1)
        scale = min(available_width / frame.shape[1], available_height / frame.shape[0], 1.0)
        if scale < 1:
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # Convert directly to Tk's image type instead of PNG-encoding every
        # preview frame, which otherwise adds avoidable CPU work and latency.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.preview_image = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        self.preview.configure(image=self.preview_image, text="")
        self.status.set(
            "Detecting from the selected window. Minimized apps may pause or return frozen frames."
        )

    def stop_detection(self) -> None:
        self.stop_event.set()
        with self.capture_lock:
            if self.capture_control is not None:
                self.capture_control.stop()
        self.status.set("Stopping detection...")

    def _set_stopped(self) -> None:
        self.running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.weights_entry.configure(state=tk.NORMAL)
        self.browse_button.configure(state=tk.NORMAL)
        self.window_picker.configure(state="readonly")
        self.refresh_button.configure(state=tk.NORMAL)
        if self.status.get().startswith("Stopping"):
            self.status.set("Detection stopped.")

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def main() -> None:
    if os.name != "nt":
        raise SystemExit("The window testing app currently supports Windows only.")
    root = tk.Tk()
    WindowTestingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
