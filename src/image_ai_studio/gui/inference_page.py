"""Phase 6C CP3/CP4: InferencePage result presentation + MainWindow close
coordination. Extends CP1's request-building form and CP2's async Run
Inference lifecycle -- a QThread + QtInferenceWorker pair wired around the
injected InferenceController, following the same canonical lifecycle as
TrainingPage/QtTrainingWorker (Phase 5C): started->run, finished/failed->page
slots + thread.quit + worker.deleteLater, thread.finished->thread.deleteLater
+ reference cleanup -- with a result area that presents the existing
InferenceResult fields (predicted class, confidence, class probabilities,
inference duration) after a successful run, without recalculating any
prediction values. CP4 adds `is_inference_active()`/`request_close()` so
MainWindow can defer window close until an active run finishes naturally
(no cancellation API).

Phase 7 CP2: the Model JSON field is now optional. When left blank,
`_build_request()` derives `training_output_dir/model_definition.json`
(the canonical artifact `imagefolder_workflow.py` now writes alongside
`best_model_state_dict.pt`/`class_mapping.json`) the same way it already
derives those two fixed paths. An explicit Model JSON value still wins --
this keeps outputs produced before Phase 7 (no `model_definition.json`)
working unchanged. No ModelSpec parsing happens here; a missing canonical
file still surfaces only through the existing `build_inference_request` ->
`InferenceController` -> worker `failed` path.

Phase 10 CP3: the page grows an explicit single-image vs folder mode. In
folder mode the visible shared artifact/device/precision inputs plus a
dedicated folder chooser are snapshotted into a checkpoint-1
`FolderInferenceRequest` and run through one injected
`FolderInferenceController` + `QtFolderInferenceWorker` using the exact
same canonical QThread lifecycle (bound-method slots, worker.deleteLater
on the worker's own finished/failed, thread.deleteLater on thread.finished).
Per-image outcomes are shown in discovered order, one row per image, with
aggregate total/succeeded/failed counts. A folder run with per-image
failures mixed in is a *completed* batch (worker `finished`), not a fatal
failure; `failed` is reserved for a fatal folder error. Single-image mode
is untouched -- same request, controller, worker, formatting, overlap
prevention, rerun, cleanup, and close-defer behavior.

Phase 11 CP2: the folder-result area gains two explicit export actions
(CSV / JSON). They serialize the *exact* `FolderInferenceResult` object
delivered to `_on_folder_finished` -- never text scraped back out of the
table -- through the checkpoint-1 `write_folder_result_export` boundary.
The retained aggregate is dropped (and both actions disabled) before every
new single-image or folder run and on a fatal folder failure or a mode
switch, so a stale or in-progress result can never be exported; a completed
batch with per-image failures mixed in is still fully exportable. Each
action opens a save dialog with a deterministic suggested filename,
treats cancellation as a no-op, calls the exporter exactly once, and
reports a bounded write error on the GUI thread while leaving the current
result exportable for retry. No automatic export, import, progress, or
cancellation is added, and no QThread / worker / signal ownership changes.

Phase 12 CP3: folder mode gains an accessible ``QProgressBar`` + a concise
``Cancel`` button (both inside the folder-result group, so single-image mode
still hides them). ``progress``/``cancelled`` from ``QtFolderInferenceWorker``
are connected to bound handlers *before* ``thread.start()``, and ``cancelled``
rides the same ``thread.quit`` + ``worker.deleteLater`` +
``_on_folder_thread_finished`` cleanup path as ``finished``/``failed``.
Progress text/value come straight from the delivered CP1 snapshots -- no
polling, estimating, or worker-thread reads. Cancel is nonblocking and
idempotent (one ``Event.set()`` via ``worker.request_cancel()``), shows a
concise ``Cancelling...`` status, disables further cancel clicks, and does
not restore Run/inputs until the folder thread has fully cleaned up. A
``cancelled`` terminal shows the exact completed partial outcomes in
discovery order, reports processed vs unprocessed counts (worded distinctly
from a fatal ``Failed:``), and retains that exact ``FolderInferenceResult``
for the Phase 11 CSV/JSON export. Starting a new run clears stale progress,
rows, cancel status, and export source first; a normal completion after all
images is never shown as ``Cancelled``. On an accepted ``MainWindow`` close
during an active folder run, ``request_close()`` also asks for cooperative
cancellation while still deferring ``close_requested`` to the cleanup
handler; single-image close behavior is unchanged."""
from __future__ import annotations

from pathlib import Path

import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from image_ai_studio.application.folder_inference_controller import FolderInferenceController
from image_ai_studio.application.inference_controller import InferenceController, build_inference_request
from image_ai_studio.gui.qt_folder_inference_worker import QtFolderInferenceWorker
from image_ai_studio.gui.qt_inference_worker import QtInferenceWorker
from image_ai_studio.inference.folder_inference import (
    FolderInferenceProgress,
    FolderInferenceRequest,
    FolderInferenceResult,
)
from image_ai_studio.inference.folder_result_export import (
    FolderResultExportError,
    write_folder_result_export,
)
from image_ai_studio.inference.single_image_inference import InferenceRequest, InferenceResult
from image_ai_studio.training.config import PRECISION_CHOICES


def _detect_device_choices() -> list[str]:
    choices = ["cpu"]
    if torch.cuda.is_available():
        choices.append("cuda")
        for index in range(torch.cuda.device_count()):
            choices.append(f"cuda:{index}")
    return choices


_RESULT_PLACEHOLDER = "--"

_MODEL_DEFINITION_FILENAME = "model_definition.json"
"""training output directory에 `imagefolder_workflow.py`가 저장하는 canonical
ModelSpec 파일명 -- `best_model_state_dict.pt`/`class_mapping.json`과 동일하게
고정된 이름으로 derive한다(Phase 7 CP2)."""

_MODE_SINGLE = "Single Image"
_MODE_FOLDER = "Folder"
"""`_mode_combo`의 두 항목. 명시적으로 사용자가 고르는 값이며 folder
run은 `_MODE_FOLDER`가 선택된 상태에서 Run을 눌렀을 때만 시작된다
(Phase 10 CP3)."""

_FOLDER_SUMMARY_PLACEHOLDER = "Total: --  Succeeded: --  Failed: --"
_FOLDER_RESULT_HEADERS = ("Image", "Status", "Predicted Class", "Confidence", "Error")
_FOLDER_STATUS_SUCCESS = "Success"
_FOLDER_STATUS_FAILURE = "Failure"

_FOLDER_PROGRESS_PLACEHOLDER = "Progress: 0 / 0"
"""folder 진행률 라벨의 초기(=idle) 텍스트. 새 folder run이 시작되거나
mode를 바꾸거나 fatal 실패가 나면 이 값으로 되돌린다(Phase 12 CP3)."""
_FOLDER_CANCELLING_STATUS = "Cancelling..."
"""Cancel을 누른 직후(또는 close가 협조적 취소를 요청한 직후) 상태
라벨에 보여줄 간결한 문구. terminal(`cancelled`/`finished`/`failed`)이
도착하면 덮어쓴다."""

_EXPORT_SUGGESTED_STEM = "folder_inference_results"
"""save 다이얼로그가 제시하는 결정론적 파일명 stem -- 포맷별 확장자
(`.csv`/`.json`)만 붙는다(Phase 11 CP2)."""
_EXPORT_ERROR_MAX_CHARS = 200
"""GUI thread에서 잡은 export 쓰기 오류를 상태 라벨에 보여줄 때의
길이 상한(bounded error)."""


def _format_confidence(confidence: float) -> str:
    return f"{confidence:.2%}"


def _format_duration_ms(inference_duration_seconds: float) -> str:
    return f"{inference_duration_seconds * 1000:.2f} ms"


def _format_probabilities(probabilities: dict[str, float]) -> str:
    """Class probabilities sorted by class name (not dict insertion order) so
    the displayed order is deterministic regardless of upstream ordering."""
    if not probabilities:
        return _RESULT_PLACEHOLDER
    lines = [f"{class_name}: {value:.2%}" for class_name, value in sorted(probabilities.items())]
    return "\n".join(lines)


def _folder_progress_text(total: int, completed: int, succeeded: int, failed: int) -> str:
    return f"Processed {completed} / {total}  (Succeeded: {succeeded}  Failed: {failed})"


def _folder_summary_text(result: FolderInferenceResult) -> str:
    return (
        f"Total: {result.total}  "
        f"Succeeded: {result.succeeded}  "
        f"Failed: {result.failed}"
    )


class InferencePage(QWidget):
    """Single-image inference form. Accepts optional InferenceController
    injection for tests. Run Inference builds the request (CP1), then runs
    it asynchronously via QThread + QtInferenceWorker (CP2) -- construction
    itself creates no thread/worker and starts no inference.

    CP4: exposes the minimal read-only activity query and close-coordination
    interface MainWindow needs, following TrainingPage's `close_requested`
    contract where practical. Single-image inference finishes naturally;
    folder inference requests cooperative cancellation on close. Both defer
    `close_requested` until worker/thread cleanup has completed.

    Phase 10 CP3: also accepts an optional `FolderInferenceController`
    injection and runs folder batches on a separate QThread +
    `QtFolderInferenceWorker` pair. The single-image and folder lifecycles
    are independent QThread/worker pairs but share one status label, one
    control-enable helper, and one `_close_pending` flag -- only one of the
    two can be active at a time (overlap prevention rejects the second)."""

    close_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        controller: InferenceController | None = None,
        folder_controller: FolderInferenceController | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller if controller is not None else InferenceController()
        self._folder_controller = (
            folder_controller if folder_controller is not None else FolderInferenceController()
        )
        self._thread: QThread | None = None
        self._worker: QtInferenceWorker | None = None
        self._folder_thread: QThread | None = None
        self._folder_worker: QtFolderInferenceWorker | None = None
        self._active_folder_path: Path = Path(".")
        self._close_pending = False
        # Phase 11 CP2: the exact FolderInferenceResult last delivered to
        # `_on_folder_finished`, or None when no completed aggregate is
        # retained. Never rebuilt from table text; the two export actions
        # are enabled iff this is not None and no run is active.
        self._folder_export_source: FolderInferenceResult | None = None
        # Phase 12 CP3: cooperative folder-cancellation state. `_folder_cancelling`
        # goes True the moment Cancel is clicked (or MainWindow close asks for a
        # cooperative stop) and back to False only once the folder thread has
        # fully cleaned up. `_folder_last_progress` is the most recent CP1
        # snapshot delivered to `_on_folder_progress` (None when idle) -- it is
        # only ever read for display, never polled for.
        self._folder_cancelling = False
        self._folder_last_progress: FolderInferenceProgress | None = None
        self._build_ui()

    # -- public: MainWindow.closeEvent()가 사용 ---------------------------------

    def is_inference_active(self) -> bool:
        """single-image `_thread`나 folder `_folder_thread` 중 하나라도
        cleanup(`_on_thread_finished`/`_on_folder_thread_finished`)이 끝나지
        않았으면 active다 -- `finished`/`failed` 같은 terminal worker signal을
        받은 시점이 아니라 cleanup이 실제로 끝난 시점에만 idle이 된다."""
        return self._thread is not None or self._folder_thread is not None

    def request_close(self) -> None:
        """Inference 중이 아니면 즉시 `close_requested`를 emit한다.
        단일 이미지 inference 중이면 취소 없이 자연 종료를 기다린다.
        폴더 inference 중이면 협조적 취소(`request_cancel`)만 요청하고
        (강제 종료·blocking wait 없음), 두 경우 모두 실제 emit은 해당
        thread cleanup 핸들러가 cleanup을 끝낸 뒤에만 일어난다
        (Phase 12 CP3)."""
        if not self.is_inference_active():
            self.close_requested.emit()
            return
        self._close_pending = True
        # Phase 12 CP3: an active *folder* run is asked to stop cooperatively so
        # close doesn't have to wait for the whole batch. Single-image inference
        # has no cancellation path and still finishes naturally; either way the
        # actual `close_requested` emit is deferred to the thread-cleanup
        # handler.
        if self._folder_thread is not None and self._folder_worker is not None:
            self._request_folder_cancellation()

    # -- public request builders (used by tests and Run handlers) -------------

    def _build_request(self) -> InferenceRequest:
        """Convert widget values to InferenceRequest. Derives the two fixed
        artifact paths from the training output directory. Model JSON is a
        third derived path (`training_output_dir/model_definition.json`)
        unless the user typed an explicit override -- that override always
        wins, so pre-Phase-7 outputs (no `model_definition.json`) keep
        working exactly as before (Phase 7 CP2)."""
        output_dir = self._training_output_dir_edit.text().strip()
        output_path = Path(output_dir) if output_dir else Path("")
        model_json_override = self._model_json_edit.text().strip()
        model_json_path = model_json_override or str(output_path / _MODEL_DEFINITION_FILENAME)
        return build_inference_request(
            model_json_path=model_json_path,
            state_dict_path=str(output_path / "best_model_state_dict.pt"),
            class_mapping_path=str(output_path / "class_mapping.json"),
            image_path=self._image_path_edit.text().strip(),
            device=self._device_combo.currentText(),
            precision=self._precision_combo.currentText(),
        )

    def _build_folder_request(self) -> FolderInferenceRequest:
        """Snapshot the *visible* shared artifact/device/precision inputs and
        the folder chooser into the checkpoint-1 `FolderInferenceRequest`.
        Artifact-path derivation (fixed state-dict/class-mapping filenames,
        optional-with-auto-derive Model JSON) is identical to
        `_build_request()` -- the only difference is `folder_path` in place
        of a single `image_path`."""
        output_dir = self._training_output_dir_edit.text().strip()
        output_path = Path(output_dir) if output_dir else Path("")
        model_json_override = self._model_json_edit.text().strip()
        model_json_path = model_json_override or str(output_path / _MODEL_DEFINITION_FILENAME)
        return FolderInferenceRequest(
            model_json_path=Path(model_json_path),
            state_dict_path=output_path / "best_model_state_dict.pt",
            class_mapping_path=output_path / "class_mapping.json",
            folder_path=Path(self._folder_path_edit.text().strip()),
            device=self._device_combo.currentText(),
            precision=self._precision_combo.currentText(),
        )

    # -- UI construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_inputs_group())
        layout.addLayout(self._build_mode_row())
        layout.addWidget(self._build_folder_input_container())
        layout.addLayout(self._build_actions_row())
        layout.addWidget(self._build_status_label())
        layout.addWidget(self._build_result_group())
        layout.addWidget(self._build_folder_result_group())
        layout.addStretch(1)
        self._on_mode_changed()  # apply initial (single-image) visibility

    def _build_inputs_group(self) -> QGroupBox:
        group = QGroupBox("Inference Inputs")
        form = QFormLayout(group)
        self._inputs_form = form

        self._training_output_dir_edit = QLineEdit()
        self._browse_output_button = QPushButton("Browse...")
        self._browse_output_button.clicked.connect(self._browse_training_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self._training_output_dir_edit)
        output_row.addWidget(self._browse_output_button)
        form.addRow("Training Output Dir:", output_row)

        self._model_json_edit = QLineEdit()
        self._model_json_edit.setPlaceholderText(
            f"Auto: <Training Output Dir>/{_MODEL_DEFINITION_FILENAME} (leave blank to use it)"
        )
        self._browse_model_button = QPushButton("Browse...")
        self._browse_model_button.clicked.connect(self._browse_model_json)
        model_row = QHBoxLayout()
        model_row.addWidget(self._model_json_edit)
        model_row.addWidget(self._browse_model_button)
        form.addRow("Model JSON:", model_row)

        self._image_path_edit = QLineEdit()
        self._browse_image_button = QPushButton("Browse...")
        self._browse_image_button.clicked.connect(self._browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self._image_path_edit)
        image_row.addWidget(self._browse_image_button)
        form.addRow("Input Image:", image_row)

        self._device_combo = QComboBox()
        for choice in _detect_device_choices():
            self._device_combo.addItem(choice)
        form.addRow("Device:", self._device_combo)

        self._precision_combo = QComboBox()
        for choice in PRECISION_CHOICES:
            self._precision_combo.addItem(choice)
        form.addRow("Precision:", self._precision_combo)

        return group

    def _build_mode_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem(_MODE_SINGLE)
        self._mode_combo.addItem(_MODE_FOLDER)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row.addWidget(self._mode_combo)
        row.addStretch(1)
        return row

    def _build_folder_input_container(self) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Input Folder:"))
        self._folder_path_edit = QLineEdit()
        self._browse_folder_button = QPushButton("Browse...")
        self._browse_folder_button.clicked.connect(self._browse_folder)
        row.addWidget(self._folder_path_edit)
        row.addWidget(self._browse_folder_button)
        self._folder_input_container = container
        return container

    def _build_actions_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._run_button = QPushButton("Run Inference")
        self._run_button.clicked.connect(self._on_run_clicked)
        row.addWidget(self._run_button)
        row.addStretch(1)
        return row

    def _build_status_label(self) -> QLabel:
        self._status_label = QLabel("Idle")
        return self._status_label

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("Inference Result")
        form = QFormLayout(group)
        self._results_form = form

        self._predicted_class_value_label = QLabel(_RESULT_PLACEHOLDER)
        form.addRow("Predicted Class:", self._predicted_class_value_label)

        self._confidence_value_label = QLabel(_RESULT_PLACEHOLDER)
        form.addRow("Confidence:", self._confidence_value_label)

        self._probabilities_value_label = QLabel(_RESULT_PLACEHOLDER)
        form.addRow("Probabilities:", self._probabilities_value_label)

        self._duration_value_label = QLabel(_RESULT_PLACEHOLDER)
        form.addRow("Duration:", self._duration_value_label)

        self._result_group = group
        return group

    def _build_folder_result_group(self) -> QGroupBox:
        group = QGroupBox("Folder Inference Results")
        box = QVBoxLayout(group)

        self._folder_summary_label = QLabel(_FOLDER_SUMMARY_PLACEHOLDER)
        box.addWidget(self._folder_summary_label)

        # Phase 12 CP3: folder-only progress indicator + cooperative Cancel.
        # Both live inside this group box, so `_on_mode_changed` hiding the
        # group in single-image mode hides them too. The bar starts at zero and
        # the Cancel button starts disabled (no folder run can accept a cancel
        # yet).
        progress_row = QHBoxLayout()
        self._folder_progress_bar = QProgressBar()
        self._folder_progress_bar.setAccessibleName("Folder inference progress")
        self._folder_progress_bar.setRange(0, 100)
        self._folder_progress_bar.setValue(0)
        self._folder_cancel_button = QPushButton("Cancel")
        self._folder_cancel_button.setAccessibleName("Cancel folder inference")
        self._folder_cancel_button.setEnabled(False)
        self._folder_cancel_button.clicked.connect(self._on_folder_cancel_clicked)
        progress_row.addWidget(self._folder_progress_bar)
        progress_row.addWidget(self._folder_cancel_button)
        box.addLayout(progress_row)

        self._folder_progress_label = QLabel(_FOLDER_PROGRESS_PLACEHOLDER)
        box.addWidget(self._folder_progress_label)

        table = QTableWidget(0, len(_FOLDER_RESULT_HEADERS))
        table.setHorizontalHeaderLabels(list(_FOLDER_RESULT_HEADERS))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        self._folder_results_table = table
        box.addWidget(table)

        # Phase 11 CP2: explicit CSV/JSON export actions for the retained
        # aggregate. Disabled until a completed batch is delivered.
        export_row = QHBoxLayout()
        self._export_csv_button = QPushButton("Export CSV")
        self._export_csv_button.clicked.connect(self._on_export_csv_clicked)
        self._export_json_button = QPushButton("Export JSON")
        self._export_json_button.clicked.connect(self._on_export_json_clicked)
        self._export_csv_button.setEnabled(False)
        self._export_json_button.setEnabled(False)
        export_row.addWidget(self._export_csv_button)
        export_row.addWidget(self._export_json_button)
        export_row.addStretch(1)
        box.addLayout(export_row)

        self._folder_result_group = group
        return group

    # -- slots -----------------------------------------------------------------

    def _browse_training_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Training Output Directory")
        if path:
            self._training_output_dir_edit.setText(path)

    def _browse_model_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Model JSON", filter="JSON Files (*.json)")
        if path:
            self._model_json_edit.setText(path)

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
        )
        if path:
            self._image_path_edit.setText(path)

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path:
            self._folder_path_edit.setText(path)

    def _on_mode_changed(self, *_args: object) -> None:
        """single-image / folder 모드에 따라 folder 전용 입력과 두 결과
        영역의 표시 여부만 토글한다. 공유되는 artifact/device/precision
        입력은 두 모드 모두에서 그대로 보인다."""
        folder_mode = self._is_folder_mode()
        self._folder_input_container.setVisible(folder_mode)
        self._folder_result_group.setVisible(folder_mode)
        self._result_group.setVisible(not folder_mode)
        # Switching modes leaves any previously shown folder batch behind as
        # a stale result -- drop the retained aggregate so export cannot act
        # on it (Phase 11 CP2) and reset the stale progress readout (Phase 12
        # CP3). Mode can't be switched while a run is active (controls are
        # disabled then), so this never races a live folder worker.
        self._set_folder_export_source(None)
        self._clear_folder_progress()

    def _is_folder_mode(self) -> bool:
        return self._mode_combo.currentText() == _MODE_FOLDER

    def _on_run_clicked(self) -> None:
        if (
            self._thread is not None
            or self._worker is not None
            or self._folder_thread is not None
            or self._folder_worker is not None
        ):
            # A run (single-image or folder) is already active -- overlap
            # prevention even if this handler is invoked directly while a
            # previous run has not finished cleaning up yet.
            return
        if self._is_folder_mode():
            self._start_folder_run()
        else:
            self._start_single_run()

    def _start_single_run(self) -> None:
        try:
            request = self._build_request()
        except Exception as exc:  # noqa: BLE001 -- request 조립 실패는 controller를 건드리지 않는다
            self._status_label.setText(f"Failed: {exc}")
            return

        self._thread = QThread(self)
        self._worker = QtInferenceWorker(self._controller, request)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # signal을 QObject bound method(self의 메서드)에 connect한다 --
        # plain 함수/lambda에 connect하면 GUI thread가 아니라 emit이
        # 일어난 worker thread에서 직접 실행된다(QtInferenceWorker
        # docstring/Phase 5B empirical 확인 참고).
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        # worker.deleteLater()는 worker 자신의 finished/failed에
        # connect한다(thread.finished에 connect하면 worker thread의
        # event loop가 이미 멈춘 뒤라 안전하지 않다 -- QtInferenceWorker
        # docstring의 deleteLater ordering 계약 참고).
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._clear_result_display()
        self._clear_folder_display()
        self._clear_folder_progress()
        # Drop any retained folder aggregate before async work begins so a
        # stale result can't be exported while this run is in flight
        # (Phase 11 CP2).
        self._set_folder_export_source(None)
        self._set_controls_enabled(False)
        self._status_label.setText("Running")
        self._thread.start()

    def _start_folder_run(self) -> None:
        try:
            request = self._build_folder_request()
        except Exception as exc:  # noqa: BLE001 -- request 조립 실패는 controller를 건드리지 않는다
            self._status_label.setText(f"Failed: {exc}")
            return

        self._active_folder_path = request.folder_path
        self._folder_thread = QThread(self)
        self._folder_worker = QtFolderInferenceWorker(self._folder_controller, request)
        try:
            # Establish this run's cancellation token synchronously before
            # the page exposes an active worker/thread.  An immediate Cancel
            # or close therefore cannot be overwritten by worker startup.
            self._folder_worker.prepare_run()
        except Exception as exc:  # noqa: BLE001 -- startup failure stays a bounded GUI failure
            self._folder_worker.deleteLater()
            self._folder_thread.deleteLater()
            self._folder_worker = None
            self._folder_thread = None
            self._status_label.setText(f"Failed: {type(exc).__name__}: {exc}")
            return
        self._folder_worker.moveToThread(self._folder_thread)
        self._folder_thread.started.connect(self._folder_worker.run)
        # single-image 쪽과 동일한 계약: signal은 self의 bound method에만
        # connect한다(plain 함수/lambda는 worker thread에서 직접 실행됨).
        # Phase 12 CP3: progress/cancelled도 여기서, thread.start() 전에,
        # 실제 bound QObject 핸들러에 딱 한 번씩 연결한다. cancelled는
        # finished/failed와 똑같이 thread.quit + worker.deleteLater 로 이어져
        # 동일한 cleanup lifecycle을 탄다(중복 연결 없음).
        self._folder_worker.progress.connect(self._on_folder_progress)
        self._folder_worker.finished.connect(self._on_folder_finished)
        self._folder_worker.cancelled.connect(self._on_folder_cancelled)
        self._folder_worker.failed.connect(self._on_folder_failed)
        self._folder_worker.finished.connect(self._folder_thread.quit)
        self._folder_worker.cancelled.connect(self._folder_thread.quit)
        self._folder_worker.failed.connect(self._folder_thread.quit)
        # worker.deleteLater()는 worker 자신의 finished/cancelled/failed에
        # connect (QtFolderInferenceWorker docstring의 deleteLater ordering 계약).
        self._folder_worker.finished.connect(self._folder_worker.deleteLater)
        self._folder_worker.cancelled.connect(self._folder_worker.deleteLater)
        self._folder_worker.failed.connect(self._folder_worker.deleteLater)
        self._folder_thread.finished.connect(self._folder_thread.deleteLater)
        self._folder_thread.finished.connect(self._on_folder_thread_finished)

        self._folder_cancelling = False
        self._clear_result_display()
        self._clear_folder_display()
        self._clear_folder_progress()
        # Drop any retained folder aggregate before async work begins so a
        # stale result can't be exported while this run is in flight
        # (Phase 11 CP2).
        self._set_folder_export_source(None)
        self._set_controls_enabled(False)
        self._status_label.setText("Running")
        self._folder_thread.start()
        # The run is now live and can accept a cooperative cancel.
        self._update_folder_cancel_enabled()

    # -- worker signal handlers -----------------------------------------------

    def _on_finished(self, result: InferenceResult) -> None:
        # 존재하는 InferenceResult 필드만 그대로 표시한다 -- 여기서
        # prediction 값을 재계산하지 않는다.
        self._predicted_class_value_label.setText(result.predicted_class)
        self._confidence_value_label.setText(_format_confidence(result.confidence))
        self._probabilities_value_label.setText(_format_probabilities(result.probabilities))
        self._duration_value_label.setText(_format_duration_ms(result.inference_duration_seconds))
        self._status_label.setText("Finished")

    def _on_failed(self, message: str) -> None:
        # 이전 성공 결과가 남아 있으면 stale prediction으로 보일 수
        # 있으므로 실패 시에도 결과 영역을 초기화한다.
        self._clear_result_display()
        first_line = message.splitlines()[0] if message else "Unknown error"
        self._status_label.setText(f"Failed: {first_line}")

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_controls_enabled(True)
        self._update_folder_export_enabled()
        if self._close_pending:
            self._close_pending = False
            self.close_requested.emit()

    # -- folder worker signal handlers --------------------------------------

    def _on_folder_finished(self, result: FolderInferenceResult) -> None:
        # per-image 실패가 섞여 있어도 이것은 *완료된 배치*다 -- backend가
        # 정상 반환한 aggregate를 그대로 표시할 뿐 예측값을 재계산하지
        # 않는다. stale single-image 결과가 남지 않도록 그쪽도 비운다.
        # `finished`는 모든 이미지가 처리된 뒤에만 온다 -- 취소가 뒤늦게
        # 걸려 있었더라도(마지막 이미지 경계 이후엔 관측 안 함) 정상
        # 완료이므로 절대 "Cancelled"로 표시하지 않는다(Phase 12 CP3).
        self._folder_cancel_button.setEnabled(False)
        self._clear_result_display()
        self._populate_folder_rows(result)
        self._folder_summary_label.setText(_folder_summary_text(result))
        self._folder_progress_bar.setRange(0, max(result.total, 1))
        self._folder_progress_bar.setValue(result.total)
        self._folder_progress_label.setText(
            _folder_progress_text(
                result.total, result.total, result.succeeded, result.failed
            )
        )
        # Retain the exact aggregate object (per-image failures included) so
        # the CSV/JSON export actions serialize it verbatim, never table text
        # (Phase 11 CP2).
        self._set_folder_export_source(result)
        self._status_label.setText("Finished")

    def _on_folder_failed(self, message: str) -> None:
        # fatal folder 실패 -- 이전 성공/부분 배치 표시가 남아 있으면 안
        # 되므로 folder 결과 영역과 진행률을 초기화하고 export source도
        # 버린다. 협조적 취소(`cancelled`)와 달리 이건 치명적 실패이므로
        # "Failed:" 로만 표시한다(이 구분은 Phase 12 CP3에서도 유지된다).
        self._folder_cancel_button.setEnabled(False)
        self._clear_folder_display()
        self._clear_folder_progress()
        self._set_folder_export_source(None)
        first_line = message.splitlines()[0] if message else "Unknown error"
        self._status_label.setText(f"Failed: {first_line}")

    def _on_folder_progress(self, snapshot: FolderInferenceProgress) -> None:
        """Render one CP1 progress snapshot exactly as delivered. Snapshots
        arrive on the GUI thread (bound-method slot, queued from the worker
        thread) in monotonically non-decreasing `completed` order; this
        handler never estimates, polls, or reads worker-thread state -- it
        just mirrors the numbers it is handed."""
        self._folder_last_progress = snapshot
        self._folder_progress_bar.setRange(0, max(snapshot.total, 1))
        self._folder_progress_bar.setValue(snapshot.completed)
        self._folder_progress_label.setText(
            _folder_progress_text(
                snapshot.total,
                snapshot.completed,
                snapshot.succeeded,
                snapshot.failed,
            )
        )

    def _on_folder_cancelled(
        self, result: FolderInferenceResult, discovered_total: int
    ) -> None:
        """Cooperative-cancellation terminal. Shows the exact completed
        partial outcomes (discovery order, verbatim) and reports processed
        vs unprocessed counts -- distinct wording from a fatal ``Failed:``.
        The exact partial `FolderInferenceResult` is retained for Phase 11
        CSV/JSON export, the same way a completed batch is."""
        self._folder_cancel_button.setEnabled(False)
        self._clear_result_display()
        self._populate_folder_rows(result)
        self._folder_summary_label.setText(_folder_summary_text(result))
        processed = result.total
        unprocessed = discovered_total - processed
        self._folder_progress_bar.setRange(0, max(discovered_total, 1))
        self._folder_progress_bar.setValue(processed)
        self._folder_progress_label.setText(
            _folder_progress_text(
                discovered_total, processed, result.succeeded, result.failed
            )
        )
        self._set_folder_export_source(result)
        self._status_label.setText(
            f"Cancelled: processed {processed} of {discovered_total} "
            f"({unprocessed} unprocessed)"
        )

    def _on_folder_cancel_clicked(self) -> None:
        self._request_folder_cancellation()

    def _request_folder_cancellation(self) -> None:
        """Ask the running folder worker to stop at the next image boundary.
        Nonblocking and idempotent: it only forwards to the worker's
        thread-safe `request_cancel()` (a single `Event.set()`), flips a
        concise cancelling status, and disables further cancel clicks. It
        never waits, never touches Run/inputs, and is a no-op when no folder
        run is active or a cancel is already in flight."""
        if self._folder_worker is None or self._folder_cancelling:
            return
        self._folder_cancelling = True
        self._folder_worker.request_cancel()
        self._update_folder_cancel_enabled()
        self._status_label.setText(_FOLDER_CANCELLING_STATUS)

    def _update_folder_cancel_enabled(self) -> None:
        """Cancel is offered only while a folder run is genuinely in flight
        and has not already been asked to stop."""
        self._folder_cancel_button.setEnabled(
            self._folder_thread is not None and not self._folder_cancelling
        )

    def _clear_folder_progress(self) -> None:
        self._folder_last_progress = None
        self._folder_progress_bar.setRange(0, 100)
        self._folder_progress_bar.setValue(0)
        self._folder_progress_label.setText(_FOLDER_PROGRESS_PLACEHOLDER)

    def _on_folder_thread_finished(self) -> None:
        self._folder_thread = None
        self._folder_worker = None
        # The worker thread has fully cleaned up -- only now is it safe to
        # drop the cancelling flag and restore Run/inputs (Phase 12 CP3).
        self._folder_cancelling = False
        self._update_folder_cancel_enabled()
        self._set_controls_enabled(True)
        self._update_folder_export_enabled()
        if self._close_pending:
            self._close_pending = False
            self.close_requested.emit()

    # -- helpers -----------------------------------------------------------------

    def _clear_result_display(self) -> None:
        self._predicted_class_value_label.setText(_RESULT_PLACEHOLDER)
        self._confidence_value_label.setText(_RESULT_PLACEHOLDER)
        self._probabilities_value_label.setText(_RESULT_PLACEHOLDER)
        self._duration_value_label.setText(_RESULT_PLACEHOLDER)

    def _clear_folder_display(self) -> None:
        self._folder_results_table.setRowCount(0)
        self._folder_summary_label.setText(_FOLDER_SUMMARY_PLACEHOLDER)

    def _populate_folder_rows(self, result: FolderInferenceResult) -> None:
        """Rebuild every row from scratch -- `setRowCount(0)` first so a
        rerun never leaves stale or duplicated rows. Row order follows the
        aggregate's `items` order verbatim (discovery order, CP1 contract)."""
        table = self._folder_results_table
        table.setRowCount(0)
        table.setRowCount(len(result.items))
        for row, outcome in enumerate(result.items):
            relative = self._folder_relative_display(outcome.image_path)
            if outcome.succeeded:
                cells = (
                    relative,
                    _FOLDER_STATUS_SUCCESS,
                    outcome.result.predicted_class,
                    _format_confidence(outcome.result.confidence),
                    _RESULT_PLACEHOLDER,
                )
            else:
                error_text = outcome.error or "Unknown error"
                first_line = error_text.splitlines()[0] if error_text else "Unknown error"
                cells = (
                    relative,
                    _FOLDER_STATUS_FAILURE,
                    _RESULT_PLACEHOLDER,
                    _RESULT_PLACEHOLDER,
                    first_line,
                )
            for col, text in enumerate(cells):
                table.setItem(row, col, QTableWidgetItem(text))

    def _folder_relative_display(self, image_path: Path) -> str:
        """Path relative to the folder that was run. Discovery is flat, so
        this is normally just the filename; fall back to the bare name if
        the outcome path is not under the recorded folder."""
        try:
            return str(image_path.relative_to(self._active_folder_path))
        except (ValueError, TypeError):
            return image_path.name

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._training_output_dir_edit,
            self._browse_output_button,
            self._model_json_edit,
            self._browse_model_button,
            self._image_path_edit,
            self._browse_image_button,
            self._device_combo,
            self._precision_combo,
            self._mode_combo,
            self._folder_path_edit,
            self._browse_folder_button,
            self._run_button,
        ):
            widget.setEnabled(enabled)

    # -- Phase 11 CP2: folder-result export ----------------------------------

    def _set_folder_export_source(self, result: FolderInferenceResult | None) -> None:
        """Record (or drop, with ``None``) the retained aggregate and
        re-sync the two export actions' enabled state. This is the only
        place `_folder_export_source` is assigned."""
        self._folder_export_source = result
        self._update_folder_export_enabled()

    def _update_folder_export_enabled(self) -> None:
        """Both export actions are enabled iff a completed aggregate is
        retained and no run (single-image or folder) is active."""
        enabled = self._folder_export_source is not None and not self.is_inference_active()
        self._export_csv_button.setEnabled(enabled)
        self._export_json_button.setEnabled(enabled)

    def _on_export_csv_clicked(self) -> None:
        self._export_folder_result("csv", "CSV Files (*.csv)")

    def _on_export_json_clicked(self) -> None:
        self._export_folder_result("json", "JSON Files (*.json)")

    def _export_folder_result(self, fmt: str, file_filter: str) -> None:
        """Serialize the retained `FolderInferenceResult` through the
        checkpoint-1 `write_folder_result_export` boundary.

        No-op when no aggregate is retained (guards a directly invoked
        handler even though the button is disabled then) or when the save
        dialog is cancelled. The retained aggregate and the selected path
        are passed to the exporter exactly once; a write error is caught
        here on the GUI thread, reported as a concise bounded status
        message, and leaves the current result exportable for retry
        without touching the displayed rows or starting inference."""
        result = self._folder_export_source
        if result is None:
            return
        suggested = f"{_EXPORT_SUGGESTED_STEM}.{fmt}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Folder Results as {fmt.upper()}",
            suggested,
            file_filter,
        )
        if not path:
            return
        try:
            write_folder_result_export(result, path, format=fmt)
        except (FolderResultExportError, OSError) as exc:
            detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            self._status_label.setText(
                f"Export failed: {detail[:_EXPORT_ERROR_MAX_CHARS]}"
            )
            return
        self._status_label.setText(f"Exported {fmt.upper()}: {Path(path).name}")
