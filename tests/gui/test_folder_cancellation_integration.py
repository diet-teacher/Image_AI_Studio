"""Phase 12 CP4: real-CPU folder progress + cooperative cancellation graduation.

required-tests id: phase12_cp4_folder_cancellation_cpu_graduation

CP1 (`tests/inference/test_folder_inference.py`) / CP2
(`tests/application/test_folder_inference_controller.py`,
`tests/gui/test_qt_folder_inference_worker.py`) / CP3
(`tests/gui/test_inference_page.py` Phase 12 CP3 section,
`tests/gui/test_main_window.py`) already pin the progress/cancel contract,
the controller/worker state machine, and the `InferencePage` state
transitions against deterministic *fake* cooperative backends. This
module's single job is to prove those pieces compose in the **real
`MainWindow` + `InferencePage` asynchronous folder path**, driven by the
**real `run_folder_inference` orchestration over the real
`run_single_image_inference` CPU forward**, consuming one canonical Phase 7
portable bundle end to end (same philosophy as
`tests/gui/test_folder_inference_integration.py` and
`tests/gui/test_folder_result_export_integration.py` -- no duplicate
correctness re-verification, no fabricated result value).

The per-image forward is never faked: `_InstrumentedSingleBackend` wraps
the genuine `run_single_image_inference` and only *records* the ordered
image paths the backend is entered with / whose forward completed, and
*pauses at one chosen image's entry* so the test can act at a known point
in the sequence. Synchronisation is `threading.Event` (worker-thread side)
+ `qtbot.waitUntil` (GUI-thread side) only -- no `QThread.wait()`/
`terminate()`, no `time.sleep()`-as-correctness, no `processEvents()`
polling, no busy loop, no arbitrary race timing.

All CPU only: tiny local images / an untrained-but-real tiny state_dict,
no CUDA, no network, no external download, no screenshot comparison, no
new dependency, and no write outside the pytest `tmp_path`. The Phase 6B
single-image public API, the Phase 7 portable artifact format/paths, the
`FolderInferenceResult` type, and the Phase 11 export `format_version` 1
schema are only *consumed* here, never changed.

Idle and single-image `MainWindow` close behaviour stays covered by the
existing `tests/gui/test_main_window.py` / `tests/gui/test_inference_page.py`
/ `tests/gui/test_phase6d_close_integration.py` suites -- this module adds
only the folder cooperative-cancel-on-close path.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox

import image_ai_studio.gui.main_window as main_window_module
from image_ai_studio.application.folder_inference_controller import FolderInferenceController
from image_ai_studio.gui.inference_page import InferencePage
from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.inference.folder_inference import run_folder_inference
from image_ai_studio.inference.folder_result_export import (
    folder_result_to_csv_text,
    folder_result_to_json_dict,
    folder_result_to_json_text,
)
from image_ai_studio.inference.single_image_inference import run_single_image_inference
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.checkpoint import save_state_dict
from image_ai_studio.training.torchvision_dataset import save_class_mapping

INPUT_SHAPE = (3, 8, 8)
_CLASSES = ["cat", "dog"]
_WAIT_TIMEOUT_MS = 30000
_EVENT_TIMEOUT_S = 20

_MODE_FOLDER = "Folder"
_STATUS_SUCCESS = "Success"
_STATUS_FAILURE = "Failure"
_RESULT_PLACEHOLDER = "--"

# Phase 6C `_format_confidence` contract: 2-decimal percent. The real
# prediction value is unknown in advance, so only the *shape* is checked
# (never recomputed).
_CONFIDENCE_PATTERN = re.compile(r"^\d{1,3}\.\d{2}%$")


# -- canonical portable bundle + local image fixtures -------------------------
# (identical construction to tests/gui/test_folder_inference_integration.py --
#  established save APIs only, no new format.)


def _model_spec(name: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )


def _make_canonical_bundle(output_dir: Path, *, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _model_spec(name)
    save_model_spec(spec, output_dir / "model_definition.json")
    save_state_dict(build_model(spec), output_dir / "best_model_state_dict.pt")
    save_class_mapping(_CLASSES, {"cat": 0, "dog": 1}, output_dir / "class_mapping.json")


def _write_valid_image(path: Path, color: tuple[int, int, int] = (120, 60, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def _write_corrupt_image(path: Path) -> None:
    """Supported extension (`.png`) but not a real image -- discovery keeps
    it, the real single-image backend raises on it, and
    `run_folder_inference` isolates that one failure as a bounded per-image
    error while the batch continues."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid PNG payload -- corrupt on purpose")


# -- instrumented (but real) CPU single-image backend ------------------------


class _InstrumentedSingleBackend:
    """Wraps the genuine `run_single_image_inference`. Records the ordered
    image filenames the backend is *entered* with (`entered`) and the ones
    whose real forward *returned* (`completed`), and -- deterministically,
    via `threading.Event` only -- pauses right at the entry of the image at
    loop index `gate_index` (before its real forward begins) so a test can
    act at a known point in the sequence. Never fabricates a result: every
    image that is not gated away runs the real CPU forward."""

    def __init__(
        self,
        *,
        gate_index: int | None = None,
        gate_reached: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.entered: list[str] = []
        self.completed: list[str] = []
        self._gate_index = gate_index
        self._gate_reached = gate_reached
        self._release = release

    def __call__(self, request):
        index = len(self.entered)
        self.entered.append(request.image_path.name)
        if self._gate_index is not None and index == self._gate_index:
            if self._gate_reached is not None:
                self._gate_reached.set()
            if self._release is not None:
                assert self._release.wait(timeout=_EVENT_TIMEOUT_S)
        result = run_single_image_inference(request)  # real CPU forward, unmodified
        self.completed.append(request.image_path.name)
        return result


def _cooperative_folder_backend(single_backend: _InstrumentedSingleBackend):
    """Real `run_folder_inference` orchestration (discovery, ascending
    ordering, per-image error isolation, initial + per-completed-image
    progress, image-boundary-only cancel observation) over the instrumented
    real single-image backend. Exposes `progress_callback=`/`should_cancel=`
    keyword hooks so `FolderInferenceController` treats it as cooperative."""

    def _run(request, *, progress_callback=None, should_cancel=None):
        return run_folder_inference(
            request,
            single_backend,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
        )

    return _run


# -- recording InferencePage (deterministic GUI-thread capture of every
#    progress snapshot + terminal, in order -- a stale/duplicate signal
#    connection would show up here) ------------------------------------------


class _RecordingInferencePage(InferencePage):
    def __init__(self, *args, **kwargs) -> None:
        self.progress_snaps: list = []
        self.folder_finished_calls: list = []
        self.folder_cancelled_calls: list = []
        self.folder_failed_calls: list = []
        super().__init__(*args, **kwargs)

    def _on_folder_progress(self, snapshot) -> None:
        self.progress_snaps.append(snapshot)
        super()._on_folder_progress(snapshot)

    def _on_folder_finished(self, result) -> None:
        self.folder_finished_calls.append(result)
        super()._on_folder_finished(result)

    def _on_folder_cancelled(self, result, discovered_total) -> None:
        self.folder_cancelled_calls.append((result, discovered_total))
        super()._on_folder_cancelled(result, discovered_total)

    def _on_folder_failed(self, message) -> None:
        self.folder_failed_calls.append(message)
        super()._on_folder_failed(message)


def _install_recording_page(monkeypatch) -> None:
    """Test-only: have `MainWindow.__init__` build the recording subclass.
    No production file is changed."""
    monkeypatch.setattr(main_window_module, "InferencePage", _RecordingInferencePage)


# -- observation helpers -----------------------------------------------------


def _folder_thread_cleaned_up(page) -> bool:
    thread = page._folder_thread
    if thread is None:
        return True
    try:
        return thread.isRunning() is False
    except RuntimeError:
        return True


def _read_folder_table(page) -> list[tuple[str, ...]]:
    table = page._folder_results_table
    rows: list[tuple[str, ...]] = []
    for row in range(table.rowCount()):
        cells = []
        for col in range(table.columnCount()):
            item = table.item(row, col)
            assert item is not None, f"row {row} col {col} has no item"
            cells.append(item.text())
        rows.append(tuple(cells))
    return rows


def _connect_terminal_recorder(worker) -> list[str]:
    """Permanently connect plain observers to *this run's* worker terminal
    signals (`list.append` is atomic in CPython -- same pattern as
    `tests/gui/test_folder_inference_integration.py`) so a run can be
    asserted to emit exactly one of finished/cancelled/failed."""
    terminals: list[str] = []
    worker.finished.connect(lambda _result: terminals.append("finished"))
    worker.cancelled.connect(lambda _result, _total: terminals.append("cancelled"))
    worker.failed.connect(lambda _message: terminals.append("failed"))
    return terminals


def _fill_folder_inputs(page, *, output_dir: Path, folder: Path) -> None:
    page._mode_combo.setCurrentText(_MODE_FOLDER)
    page._training_output_dir_edit.setText(str(output_dir))
    assert page._model_json_edit.text() == ""  # auto-discovery, left blank end to end
    page._device_combo.setCurrentText("cpu")
    page._precision_combo.setCurrentText("fp32")
    page._folder_path_edit.setText(str(folder))


def _assert_controls_restored(page) -> None:
    assert page._folder_thread is None
    assert page._folder_worker is None
    assert page._folder_cancelling is False
    assert page._run_button.isEnabled() is True
    assert page._mode_combo.isEnabled() is True
    assert page._folder_path_edit.isEnabled() is True
    assert page._training_output_dir_edit.isEnabled() is True
    assert page._model_json_edit.isEnabled() is True
    assert page._device_combo.isEnabled() is True
    assert page._precision_combo.isEnabled() is True


# ===========================================================================
# 1. progress -> boundary cancel -> exact partial -> real Phase 11 export ->
#    cleanup -> clean full rerun, all real CPU through MainWindow
# ===========================================================================


def test_folder_progress_cancellation_partial_export_and_rerun_cpu_graduation(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    output_dir = tmp_path / "out"
    _make_canonical_bundle(output_dir, name="phase12_cp4_portable_bundle")

    batch = tmp_path / "batch"
    _write_valid_image(batch / "a1.png", color=(250, 250, 250))
    _write_corrupt_image(batch / "b2.png")  # isolated per-image failure inside the partial
    _write_valid_image(batch / "c3.png", color=(5, 5, 5))
    _write_valid_image(batch / "d4.png", color=(10, 200, 10))
    # Not discoverable: unsupported extension + nested folder (flat discovery).
    (batch / "notes.txt").write_text("ignored", encoding="utf-8")
    _write_valid_image(batch / "nested" / "z_ignored.png")

    _install_recording_page(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._inference_page
    window._tabs.setCurrentWidget(page)

    gate_reached = threading.Event()
    release = threading.Event()
    single_backend = _InstrumentedSingleBackend(
        gate_index=2, gate_reached=gate_reached, release=release
    )
    page._folder_controller = FolderInferenceController(
        backend=_cooperative_folder_backend(single_backend)
    )
    _fill_folder_inputs(page, output_dir=output_dir, folder=batch)

    page._on_run_clicked()
    worker = page._folder_worker
    assert worker is not None
    terminals = _connect_terminal_recorder(worker)

    # -- observe the initial 0-of-total + one per-completed-image snapshot --
    qtbot.waitUntil(gate_reached.is_set, timeout=_WAIT_TIMEOUT_MS)
    qtbot.waitUntil(
        lambda: [s.completed for s in page.progress_snaps] == [0, 1, 2],
        timeout=_WAIT_TIMEOUT_MS,
    )
    assert page.progress_snaps[0].total == 4
    assert (page.progress_snaps[-1].succeeded, page.progress_snaps[-1].failed) == (1, 1)
    # image 0 done, image 1 raised (isolated), image 2 entered and paused
    # *before* its real forward -- exactly one real image has fully completed.
    assert single_backend.entered == ["a1.png", "b2.png", "c3.png"]
    assert single_backend.completed == ["a1.png"]
    assert page._folder_cancel_button.isEnabled() is True

    # -- request cancellation only now (>= 1 real completed image) ---------
    page._folder_cancel_button.click()
    assert page._status_label.text() == "Cancelling..."
    assert page._folder_cancel_button.isEnabled() is False
    assert page._run_button.isEnabled() is False  # not restored until cleanup

    release.set()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=_WAIT_TIMEOUT_MS)
    qtbot.waitUntil(lambda: _folder_thread_cleaned_up(page), timeout=5000)
    # `_on_folder_thread_finished` (the `thread.finished` slot) is the last
    # step of the terminal-signal chain; waiting for it deterministically
    # drains a late/duplicate terminal signal instead of a fixed sleep.
    qtbot.waitUntil(lambda: page._folder_thread is None, timeout=5000)

    # -- exactly one cancelled terminal, never finished/failed ------------
    assert terminals == ["cancelled"]
    assert len(page.folder_cancelled_calls) == 1
    assert page.folder_finished_calls == []
    assert page.folder_failed_calls == []

    # -- the in-progress forward completed normally; no later image started
    assert single_backend.completed == ["a1.png", "c3.png"]
    assert single_backend.entered == ["a1.png", "b2.png", "c3.png"]  # d4.png never entered

    # -- exact completed partial rows in discovery order ------------------
    rows = _read_folder_table(page)
    assert [r[0] for r in rows] == ["a1.png", "b2.png", "c3.png"]
    assert [r[1] for r in rows] == [_STATUS_SUCCESS, _STATUS_FAILURE, _STATUS_SUCCESS]
    for idx in (0, 2):
        assert rows[idx][2] in _CLASSES
        assert _CONFIDENCE_PATTERN.match(rows[idx][3]) is not None
        assert rows[idx][4] == _RESULT_PLACEHOLDER
    assert rows[1][2] == _RESULT_PLACEHOLDER and rows[1][3] == _RESULT_PLACEHOLDER
    assert rows[1][4] not in ("", _RESULT_PLACEHOLDER)
    assert "\n" not in rows[1][4]  # table shows only the first line

    assert page._folder_summary_label.text() == "Total: 3  Succeeded: 2  Failed: 1"

    cancelled_result, discovered_total = page.folder_cancelled_calls[0]
    assert discovered_total == 4
    processed = cancelled_result.total
    assert (processed, cancelled_result.succeeded, cancelled_result.failed) == (3, 2, 1)
    assert page._status_label.text() == "Cancelled: processed 3 of 4 (1 unprocessed)"
    assert page._folder_progress_bar.maximum() == 4
    assert page._folder_progress_bar.value() == 3

    completed_series = [s.completed for s in page.progress_snaps]
    assert completed_series == sorted(completed_series)  # monotonic, never decreases
    assert completed_series[0] == 0 and completed_series[-1] == 3

    # -- real Phase 11 CSV + JSON export of the exact partial aggregate ---
    retained = page._folder_export_source
    assert retained is cancelled_result

    csv_dest = tmp_path / "partial_results.csv"
    json_dest = tmp_path / "partial_results.json"

    def _fake_save(returned: Path):
        def _inner(parent, caption, directory="", filter="", *args, **kwargs):
            return (str(returned), filter)

        return staticmethod(_inner)

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _fake_save(csv_dest))
    page._export_csv_button.click()
    assert page._status_label.text().startswith("Exported CSV")
    assert csv_dest.read_text(encoding="utf-8") == folder_result_to_csv_text(retained)

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _fake_save(json_dest))
    page._export_json_button.click()
    assert page._status_label.text().startswith("Exported JSON")
    json_text = json_dest.read_text(encoding="utf-8")
    assert json_text == folder_result_to_json_text(retained)

    payload = json.loads(json_text)
    assert payload["format_version"] == 1  # schema unchanged
    assert folder_result_to_json_dict(retained)["format_version"] == 1
    assert (payload["total"], payload["succeeded"], payload["failed"]) == (3, 2, 1)
    assert [item["status"] for item in payload["items"]] == ["success", "failed", "success"]
    assert [Path(item["image_path"]).name for item in payload["items"]] == [
        "a1.png",
        "b2.png",
        "c3.png",
    ]
    # discovered-total / unprocessed / cancelled metadata is NOT embedded in
    # format_version 1 -- it lives only on the FolderInferenceCancelled value.
    def _object_keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key.lower()
                yield from _object_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from _object_keys(nested)

    json_keys = set(_object_keys(payload))
    assert "discovered_total" not in json_keys
    assert "unprocessed" not in json_keys
    assert not any("cancel" in key for key in json_keys)

    _assert_controls_restored(page)
    assert page._folder_controller.state == "cancelled"

    # ===================================================================
    # subsequent full successful rerun in the same window -- fresh
    # progress, no stale cancel flag / rows / errors / export data
    # ===================================================================
    rerun_dir = tmp_path / "rerun"
    _write_valid_image(rerun_dir / "x1.png", color=(200, 10, 10))
    _write_valid_image(rerun_dir / "y2.png", color=(10, 10, 200))

    snap_before = len(page.progress_snaps)
    cancelled_before = len(page.folder_cancelled_calls)
    single_backend2 = _InstrumentedSingleBackend()
    page._folder_controller = FolderInferenceController(
        backend=_cooperative_folder_backend(single_backend2)
    )
    page._folder_path_edit.setText(str(rerun_dir))

    page._on_run_clicked()
    worker2 = page._folder_worker
    assert worker2 is not None
    terminals2 = _connect_terminal_recorder(worker2)

    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=_WAIT_TIMEOUT_MS)
    qtbot.waitUntil(lambda: _folder_thread_cleaned_up(page), timeout=5000)
    # deterministic drain of the terminal-signal chain -- see note above.
    qtbot.waitUntil(lambda: page._folder_thread is None, timeout=5000)

    assert terminals2 == ["finished"]
    assert len(page.folder_finished_calls) == 1
    assert len(page.folder_cancelled_calls) == cancelled_before  # no stale cancellation
    assert page.folder_failed_calls == []

    assert page._status_label.text() == "Finished"
    assert "Cancelled" not in page._status_label.text()
    assert "Failed" not in page._status_label.text()

    rerun_rows = _read_folder_table(page)
    assert [r[0] for r in rerun_rows] == ["x1.png", "y2.png"]  # no stale 3rd row
    assert [r[1] for r in rerun_rows] == [_STATUS_SUCCESS, _STATUS_SUCCESS]
    for r in rerun_rows:
        assert r[2] in _CLASSES
        assert _CONFIDENCE_PATTERN.match(r[3]) is not None
        assert r[4] == _RESULT_PLACEHOLDER  # no leftover error text
    assert page._folder_summary_label.text() == "Total: 2  Succeeded: 2  Failed: 0"

    fresh_series = [s.completed for s in page.progress_snaps[snap_before:]]
    assert fresh_series == [0, 1, 2]  # progress restarts from zero
    assert page._folder_progress_bar.maximum() == 2
    assert page._folder_progress_bar.value() == 2

    assert page._folder_export_source is page.folder_finished_calls[-1]
    assert page._folder_export_source.total == 2
    assert page._export_csv_button.isEnabled() is True
    assert page._export_json_button.isEnabled() is True

    assert single_backend2.entered == ["x1.png", "y2.png"]
    assert single_backend2.completed == ["x1.png", "y2.png"]
    assert page._folder_controller.state == "finished"
    _assert_controls_restored(page)


# ===========================================================================
# 2. close during a real folder run -> cooperative cancel, window stays alive
#    through the current forward, closes exactly once after terminal cleanup
# ===========================================================================


def test_close_during_folder_run_requests_cooperative_cancel_and_closes_once(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    output_dir = tmp_path / "out"
    _make_canonical_bundle(output_dir, name="phase12_cp4_close_bundle")

    batch = tmp_path / "batch"
    for name, color in (
        ("m1.png", (250, 250, 250)),
        ("m2.png", (5, 5, 5)),
        ("m3.png", (10, 200, 10)),
        ("m4.png", (200, 10, 10)),
    ):
        _write_valid_image(batch / name, color=color)

    question_calls = {"n": 0}

    def fake_question(*args, **kwargs):
        question_calls["n"] += 1
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    close_calls = {"n": 0}
    original_close = MainWindow.close

    def counting_close(self):
        close_calls["n"] += 1
        return original_close(self)

    monkeypatch.setattr(MainWindow, "close", counting_close)

    _install_recording_page(monkeypatch)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    page = window._inference_page
    window._tabs.setCurrentWidget(page)

    gate_reached = threading.Event()
    release = threading.Event()
    single_backend = _InstrumentedSingleBackend(
        gate_index=1, gate_reached=gate_reached, release=release
    )
    page._folder_controller = FolderInferenceController(
        backend=_cooperative_folder_backend(single_backend)
    )
    _fill_folder_inputs(page, output_dir=output_dir, folder=batch)

    page._on_run_clicked()
    worker = page._folder_worker
    assert worker is not None
    terminals = _connect_terminal_recorder(worker)

    qtbot.waitUntil(gate_reached.is_set, timeout=_WAIT_TIMEOUT_MS)
    assert page.is_inference_active() is True
    # image 0 fully completed; image 1 entered and paused before its forward.
    assert single_backend.entered == ["m1.png", "m2.png"]
    assert single_backend.completed == ["m1.png"]

    window.close()
    assert question_calls["n"] == 1
    assert close_calls["n"] == 1  # deferred + ignored, no retry yet
    assert window.isVisible() is True
    assert page._folder_cancelling is True  # cooperative cancel was requested
    assert page._status_label.text() == "Cancelling..."
    assert page.is_inference_active() is True  # window alive through the current forward

    release.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=_WAIT_TIMEOUT_MS)
    # deterministic drain of the terminal-signal chain -- see note above.
    qtbot.waitUntil(lambda: page._folder_thread is None, timeout=5000)

    assert question_calls["n"] == 1  # exactly one dialog for the whole test
    assert close_calls["n"] == 2  # exactly one retry, and it succeeded
    assert window._close_pending is False

    assert terminals == ["cancelled"]  # cooperative cancel took effect, not finished/failed
    assert len(page.folder_cancelled_calls) == 1
    assert page.folder_finished_calls == []
    assert page.folder_failed_calls == []

    # the in-progress forward completed normally; no later image started
    assert single_backend.completed == ["m1.png", "m2.png"]
    assert single_backend.entered == ["m1.png", "m2.png"]

    cancelled_result, discovered_total = page.folder_cancelled_calls[0]
    assert discovered_total == 4
    assert cancelled_result.total == 2
    assert [outcome.image_path.name for outcome in cancelled_result.items] == ["m1.png", "m2.png"]

    assert page._folder_thread is None
    assert page._folder_worker is None
    assert page._folder_cancelling is False
