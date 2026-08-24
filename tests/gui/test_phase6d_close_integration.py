"""Phase 6D CP2: real-backend close coordination for `MainWindow`.

`tests/gui/test_main_window.py` (Phase 6C CP4) already proves the close
*coordination logic itself* (dialog, deferred close, cooperative stop
request, completion-order retries) against fake, hand-written blocking
backends. This module's only job is to prove the same coordination holds
when `TrainingPage`/`InferencePage` are wired to the *real* production
backends (`run_imagefolder_training_workflow`/`run_single_image_inference`)
-- no fake result is ever fabricated here. Real CPU training/inference is
slow to synchronize precisely against a `window.close()` call, so each
backend is wrapped with `threading.Event` gates that control only *when*
the real call is allowed to begin (and, for the cooperative-stop scenario,
when a real epoch boundary is allowed to proceed) -- the computation itself
is always the genuine production function, never a stand-in return value.

No `QThread.wait()`/`terminate()`/`time.sleep()`-as-correctness/
`processEvents()` polling/GUI-thread busy loop anywhere below -- only
bounded `threading.Event.wait(timeout=...)` (worker-thread side) and
`qtbot.waitUntil(...)` (GUI-thread side), the same primitives
`test_main_window.py` already relies on."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QMessageBox

from image_ai_studio.application.inference_controller import InferenceController
from image_ai_studio.application.training_controller import TrainingController, build_training_request
from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.inference.single_image_inference import run_single_image_inference
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.imagefolder_workflow import run_imagefolder_training_workflow

INPUT_SHAPE = (3, 8, 8)
_CLASS_COLORS = {"cat": (250, 250, 250), "dog": (5, 5, 5)}
_WAIT_TIMEOUT_MS = 20000
_EVENT_TIMEOUT_S = 15


def _make_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for class_name, color in _CLASS_COLORS.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (20, 20), color=color).save(class_dir / f"{i}.png")


def _write_model_json(path: Path) -> None:
    spec = ModelSpec(
        name="phase6d_cp2_close_coordination",
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )
    save_model_spec(spec, path)


def _train_reference_model(root: Path) -> tuple[Path, Path, Path]:
    """Genuinely trains (real CPU backend, 1 epoch, no gating) a tiny model
    outside of any `MainWindow`/`QThread` so its artifacts can be fed into a
    gated *inference* run without waiting on a concurrently-gated training
    run in the same test. Returns (output_dir, model_json_path, image_path)."""
    dataset_root = root / "dataset"
    _make_dataset(dataset_root)
    model_json_path = root / "model.json"
    _write_model_json(model_json_path)
    output_dir = root / "out"

    request = build_training_request(
        model_json_path=model_json_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        epochs=1,
        batch_size=4,
        learning_rate=1e-2,
        device="cpu",
        export_torchscript=False,
    )
    run_imagefolder_training_workflow(request)
    image_path = dataset_root / "test" / "cat" / "0.png"
    return output_dir, model_json_path, image_path


def _thread_cleaned_up(page) -> bool:
    """Same helper as `test_training_inference_integration.py` (CP1) --
    `InferencePage` resets `_thread` to `None` after cleanup while
    `TrainingPage` leaves the (about to be `deleteLater()`-ed) `QThread`
    object in place, so both "already `None`" and "no longer running" (or
    already C++-deleted, `RuntimeError`) count as cleaned up."""
    thread = page._thread
    if thread is None:
        return True
    try:
        return thread.isRunning() is False
    except RuntimeError:
        return True


def _make_gated_inference_backend(started: threading.Event, proceed: threading.Event):
    """Delegates to the real `run_single_image_inference` -- the gate only
    withholds *when* that real call is allowed to begin."""

    def backend(request):
        started.set()
        assert proceed.wait(timeout=_EVENT_TIMEOUT_S)
        return run_single_image_inference(request)

    return backend


def _make_epoch1_gated_training_backend(
    started: threading.Event, epoch1_reached: threading.Event, resume: threading.Event
):
    """Delegates to the real `run_imagefolder_training_workflow`. The gate
    intercepts the *real* per-epoch `progress_callback` (called by the real
    `run_training()` loop after epoch 1 genuinely finishes) and blocks
    inside it -- production `run_training()` only evaluates `should_stop()`
    immediately *after* that callback returns (docs: progress_callback then
    should_stop), so holding here lets the test call `window.close()`
    (which calls `TrainingController.request_stop()`) before that real
    evaluation happens, proving genuine cooperative stop rather than
    asserting on a fabricated stop_reason."""

    def backend(request, *, progress_callback=None, should_stop=None):
        started.set()

        def gated_progress_callback(progress):
            if progress_callback is not None:
                progress_callback(progress)
            if progress.run_epoch == 1:
                epoch1_reached.set()
                assert resume.wait(timeout=_EVENT_TIMEOUT_S)

        return run_imagefolder_training_workflow(
            request, progress_callback=gated_progress_callback, should_stop=should_stop
        )

    return backend


def _count_close_calls(monkeypatch) -> dict:
    """Wraps `MainWindow.close()` itself so a test can prove -- not just
    infer from final visibility -- exactly how many times `close()` is
    invoked: the initial (deferred, ignored) user request plus, later, the
    single internal retry `_maybe_finish_pending_close()` issues once every
    active page has reported `close_requested`. Delegates to the real
    `close()` every time -- this only counts, it never changes behavior."""
    counts = {"n": 0}
    original_close = MainWindow.close

    def counting_close(self):
        counts["n"] += 1
        return original_close(self)

    monkeypatch.setattr(MainWindow, "close", counting_close)
    return counts


# -- close during real inference alone -----------------------------------------


def test_close_during_real_inference_defers_until_natural_completion(tmp_path, monkeypatch, qtbot) -> None:
    output_dir, model_json_path, image_path = _train_reference_model(tmp_path / "ref")

    inference_started = threading.Event()
    resume_inference = threading.Event()
    question_calls = {"n": 0}

    def fake_question(*args, **kwargs):
        question_calls["n"] += 1
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    close_calls = _count_close_calls(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    inference_page = window._inference_page
    inference_page._controller = InferenceController(
        backend=_make_gated_inference_backend(inference_started, resume_inference)
    )
    inference_page._training_output_dir_edit.setText(str(output_dir))
    inference_page._model_json_edit.setText(str(model_json_path))
    inference_page._image_path_edit.setText(str(image_path))
    inference_page._device_combo.setCurrentText("cpu")
    inference_page._precision_combo.setCurrentText("fp32")

    inference_page._on_run_clicked()
    # GUI-thread side: poll the Qt event loop via qtbot instead of blocking
    # the test/GUI thread on threading.Event.wait() directly.
    qtbot.waitUntil(inference_started.is_set, timeout=_WAIT_TIMEOUT_MS)
    assert inference_page.is_inference_active() is True

    window.close()
    assert question_calls["n"] == 1
    assert close_calls["n"] == 1  # the deferred, ignored close request -- no retry yet
    assert window.isVisible() is True
    assert inference_page.is_inference_active() is True  # close does not cancel inference

    resume_inference.set()  # worker-thread side gate release, not a GUI-thread wait
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=_WAIT_TIMEOUT_MS)

    assert question_calls["n"] == 1  # still exactly one dialog for the whole test
    assert close_calls["n"] == 2  # exactly one retry, and it was the one that succeeded
    assert window._close_pending is False
    assert inference_page._status_label.text() == "Finished"
    # real close only happens after QThread + worker cleanup (`_on_thread_finished`)
    assert inference_page._thread is None
    assert inference_page._worker is None


# -- close while real training and real inference are both active -------------


@pytest.mark.parametrize("finish_training_first", [True, False])
def test_close_with_simultaneous_real_training_and_inference(
    tmp_path, monkeypatch, qtbot, finish_training_first
) -> None:
    # Reference artifacts for inference come from an independent, already
    # completed real training run -- inference must not wait on the
    # concurrently gated training run below to produce its own artifacts.
    ref_output_dir, ref_model_json, ref_image_path = _train_reference_model(tmp_path / "ref")

    train_dataset_root = tmp_path / "train_dataset"
    _make_dataset(train_dataset_root)
    train_model_json = tmp_path / "train_model.json"
    _write_model_json(train_model_json)
    train_output_dir = tmp_path / "train_out"

    training_started = threading.Event()
    epoch1_reached = threading.Event()
    resume_training = threading.Event()
    inference_started = threading.Event()
    resume_inference = threading.Event()
    question_calls = {"n": 0}

    def fake_question(*args, **kwargs):
        question_calls["n"] += 1
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    close_calls = _count_close_calls(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    training_page = window._training_page
    training_page._controller = TrainingController(
        backend=_make_epoch1_gated_training_backend(training_started, epoch1_reached, resume_training)
    )
    training_page._model_json_edit.setText(str(train_model_json))
    training_page._dataset_root_edit.setText(str(train_dataset_root))
    training_page._output_dir_edit.setText(str(train_output_dir))
    training_page._epochs_spin.setValue(2)  # >1 so should_stop() is genuinely evaluated after epoch 1
    training_page._batch_size_spin.setValue(4)
    training_page._learning_rate_spin.setValue(1e-2)
    training_page._export_torchscript_check.setChecked(False)
    training_page._device_combo.setCurrentText("cpu")

    inference_page = window._inference_page
    inference_page._controller = InferenceController(
        backend=_make_gated_inference_backend(inference_started, resume_inference)
    )
    inference_page._training_output_dir_edit.setText(str(ref_output_dir))
    inference_page._model_json_edit.setText(str(ref_model_json))
    inference_page._image_path_edit.setText(str(ref_image_path))
    inference_page._device_combo.setCurrentText("cpu")
    inference_page._precision_combo.setCurrentText("fp32")

    training_page._on_start_clicked()
    inference_page._on_run_clicked()

    # GUI-thread side: poll the Qt event loop via qtbot instead of blocking
    # the test/GUI thread on threading.Event.wait() directly -- the events
    # themselves are still set from the real backend's worker thread.
    qtbot.waitUntil(training_started.is_set, timeout=_WAIT_TIMEOUT_MS)
    qtbot.waitUntil(epoch1_reached.is_set, timeout=_WAIT_TIMEOUT_MS)  # real epoch 1 genuinely completed
    qtbot.waitUntil(inference_started.is_set, timeout=_WAIT_TIMEOUT_MS)
    assert training_page.is_training_active() is True
    assert inference_page.is_inference_active() is True

    window.close()
    assert question_calls["n"] == 1
    assert close_calls["n"] == 1  # the deferred, ignored close request -- no retry attempted yet
    assert window.isVisible() is True
    assert training_page._status_label.text() == "Stopping after current epoch..."
    assert inference_page.is_inference_active() is True  # inference is never cancelled

    if finish_training_first:
        resume_training.set()  # worker-thread side gate: let should_stop() evaluate True, epoch 2 never runs
        qtbot.waitUntil(lambda: training_page.is_training_active() is False, timeout=_WAIT_TIMEOUT_MS)
        assert window.isVisible() is True  # inference still active -- window must stay open
        assert question_calls["n"] == 1
        # first completion alone must not even attempt a retry: _maybe_finish_pending_close()
        # bails out early while inference is still active.
        assert close_calls["n"] == 1
        resume_inference.set()
    else:
        resume_inference.set()
        qtbot.waitUntil(lambda: inference_page.is_inference_active() is False, timeout=_WAIT_TIMEOUT_MS)
        assert window.isVisible() is True  # training still active -- window must stay open
        assert question_calls["n"] == 1
        assert close_calls["n"] == 1  # first completion alone must not attempt a retry either
        resume_training.set()

    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=_WAIT_TIMEOUT_MS)

    assert question_calls["n"] == 1  # exactly one dialog for the whole test
    assert close_calls["n"] == 2  # exactly one retry, issued only once both pages finished, and it succeeded
    assert window._close_pending is False
    # cooperative stop actually happened in the real backend (not forced termination):
    # only 1 of the requested 2 epochs ran, and stop_reason genuinely was "user_stopped".
    assert training_page._status_label.text() == "Training stopped by user"
    assert inference_page._status_label.text() == "Finished"

    qtbot.waitUntil(lambda: _thread_cleaned_up(training_page), timeout=5000)
    assert inference_page._thread is None
    assert inference_page._worker is None
