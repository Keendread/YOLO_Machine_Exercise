"""Tests for the Helmet Checker batch logic; no Tk, model, PIL or ultralytics needed."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from yolo_starter import helmet_app
from yolo_starter.helmet_app import (
    APP_NAME,
    ModelError,
    fit_size,
    is_supported_image,
    parse_confidence,
    process_image,
    run_batch,
)


class Tensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def fake_result(*detections):
    names = {0: "Helmet", 1: "No-Helmet", 2: "person"}
    classes = [cls for cls, _ in detections]
    confs = [conf for _, conf in detections]
    return SimpleNamespace(names=names, boxes=SimpleNamespace(cls=Tensor(classes), conf=Tensor(confs)))


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def predict(self, source, conf, verbose):
        self.calls.append((source, conf))
        output = self.outputs[Path(source).name]
        if isinstance(output, Exception):
            raise output
        return output


class ImportTests(unittest.TestCase):
    def test_app_name(self) -> None:
        self.assertEqual(APP_NAME, "Helmet Checker")

    def test_import_does_not_load_gui_or_model_libraries(self) -> None:
        code = (
            "import sys, yolo_starter.helmet_app;"
            "bad = [m for m in ('tkinter', 'ultralytics', 'PIL', 'torch') if m in sys.modules];"
            "sys.exit(','.join(bad) or 0)"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
        self.assertEqual(done.returncode, 0, done.stderr)


class HelperTests(unittest.TestCase):
    def test_supported_extensions(self) -> None:
        for name in ("a.jpg", "a.JPEG", "a.png", "a.webp", "a.Bmp"):
            self.assertTrue(is_supported_image(name), name)
        for name in ("a.gif", "a.txt", "jpg"):
            self.assertFalse(is_supported_image(name), name)

    def test_parse_confidence(self) -> None:
        self.assertEqual(parse_confidence("0.25"), 0.25)
        self.assertEqual(parse_confidence(" 1 "), 1.0)
        for bad in ("", "abc", "-0.1", "1.5", "nan", "inf"):
            with self.assertRaises(ValueError, msg=bad):
                parse_confidence(bad)

    def test_default_confidence(self) -> None:
        self.assertEqual(helmet_app.DEFAULT_CONFIDENCE, 0.25)

    def test_default_weights_falls_back_to_bundled_model(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            bundled = Path(empty) / "models" / "helmet-yolo11n.pt"
            bundled.parent.mkdir()
            bundled.touch()
            cwd = os.getcwd()
            original = helmet_app.BUNDLED_WEIGHTS
            try:
                os.chdir(empty)
                helmet_app.BUNDLED_WEIGHTS = bundled
                self.assertEqual(helmet_app.default_weights(), bundled)
                trained = Path("runs/detect/mine/weights/best.pt")
                trained.parent.mkdir(parents=True)
                trained.touch()
                self.assertEqual(helmet_app.default_weights(), trained.resolve())
            finally:
                helmet_app.BUNDLED_WEIGHTS = original
                os.chdir(cwd)

    def test_fit_size_shrinks_but_never_enlarges(self) -> None:
        self.assertEqual(fit_size(2000, 1000, 500, 500), (500, 250))
        self.assertEqual(fit_size(100, 50, 500, 500), (100, 50))
        self.assertEqual(fit_size(100, 50, 0, 0), (1, 1))


class ProcessImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def image(self, name: str) -> str:
        path = Path(self.dir.name) / name
        path.write_bytes(b"x")
        return str(path)

    def test_counts_riders_and_ignores_other_classes(self) -> None:
        path = self.image("a.jpg")
        model = FakeModel({"a.jpg": [fake_result((0, 0.9), (1, 0.6), (2, 0.99))]})
        result = process_image(model, path, 0.4, render=lambda r: "img")
        self.assertIsNone(result.error)
        self.assertEqual((result.report.with_helmet, result.report.without_helmet), (1, 1))
        self.assertEqual(result.annotated, "img")
        self.assertEqual(model.calls, [(path, 0.4)])

    def test_zero_detections_is_not_an_error(self) -> None:
        path = self.image("a.jpg")
        model = FakeModel({"a.jpg": [fake_result()]})
        result = process_image(model, path, 0.25, render=lambda r: None)
        self.assertIsNone(result.error)
        self.assertEqual(result.report.total_riders, 0)

    def test_missing_file_becomes_error(self) -> None:
        result = process_image(FakeModel({}), str(Path(self.dir.name) / "gone.jpg"), 0.25)
        self.assertIsNone(result.report)
        self.assertIn("not found", result.error)

    def test_undecodable_image_with_empty_prediction_becomes_error(self) -> None:
        path = self.image("bad.jpg")
        result = process_image(FakeModel({"bad.jpg": []}), path, 0.25)
        self.assertIsNone(result.report)
        self.assertIn("could not be read", result.error)

    def test_model_exception_becomes_error(self) -> None:
        path = self.image("a.jpg")
        result = process_image(FakeModel({"a.jpg": RuntimeError("boom")}), path, 0.25)
        self.assertEqual(result.error, "RuntimeError: boom")


class RunBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.messages: list[tuple] = []

    def image(self, name: str) -> str:
        path = Path(self.dir.name) / name
        path.write_bytes(b"x")
        return str(path)

    def run_batch(self, paths, get_model, cancel=None):
        run_batch(paths, get_model, 0.25, self.messages.append, cancel or threading.Event(),
                  render=lambda r: None)
        return [m[0] for m in self.messages]

    def test_bad_image_does_not_stop_the_batch(self) -> None:
        good, bad = self.image("good.jpg"), self.image("bad.jpg")
        model = FakeModel({"good.jpg": [fake_result((0, 0.9))], "bad.jpg": []})
        kinds = self.run_batch([bad, good], lambda: model)
        self.assertEqual(kinds, ["status", "progress", "result", "progress", "result", "done"])
        results = [m[2] for m in self.messages if m[0] == "result"]
        self.assertIsNotNone(results[0].error)
        self.assertEqual(results[1].report.with_helmet, 1)
        self.assertEqual(self.messages[-1], ("done", False))

    def test_model_load_failure_is_reported_then_done(self) -> None:
        def broken():
            raise ModelError("no weights")

        kinds = self.run_batch([self.image("a.jpg")], broken)
        self.assertEqual(kinds, ["status", "model_error", "done"])
        self.assertEqual(self.messages[1][1], "no weights")

    def test_unexpected_failure_still_ends_with_done(self) -> None:
        def broken():
            raise RuntimeError("surprise")

        kinds = self.run_batch([self.image("a.jpg")], broken)
        self.assertEqual(kinds[-2:], ["model_error", "done"])

    def test_cancel_stops_before_next_image(self) -> None:
        cancel = threading.Event()
        model = FakeModel({"a.jpg": [fake_result()], "b.jpg": [fake_result()]})
        paths = [self.image("a.jpg"), self.image("b.jpg")]
        original = self.messages.append

        def emit(message):
            original(message)
            if message[0] == "result":
                cancel.set()

        run_batch(paths, lambda: model, 0.25, emit, cancel, render=lambda r: None)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(self.messages[-1], ("done", True))


class LoadModelTests(unittest.TestCase):
    def test_missing_weights_raise_model_error(self) -> None:
        with self.assertRaises(ModelError) as caught:
            helmet_app.load_model(Path(tempfile.gettempdir()) / "definitely-missing-weights.pt")
        self.assertIn("not found", str(caught.exception))

    def test_unloadable_weights_raise_model_error(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "broken.pt"
            weights.write_bytes(b"not a model")
            with self.assertRaises(ModelError):
                helmet_app.load_model(weights)


if __name__ == "__main__":
    unittest.main()
