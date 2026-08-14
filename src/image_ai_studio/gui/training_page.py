"""Phase 5C: actual Training GUI page. **모든 학습 실행은 Phase 5B의
`TrainingController`/`QtTrainingWorker`를 그대로 재사용한다** -- 이
모듈은 widget을 만들고 그 값을 `build_training_request()`로 넘기는
얇은 GUI 계층일 뿐, training/thread/worker architecture를 다시
구현하지 않는다.

CUDA 감지(`torch.cuda.is_available()`)는 `TrainingPage.__init__()`
안에서 device 콤보박스를 채울 때 한 번만 수행한다 -- 이 모듈을
import하는 것만으로는 어떤 side effect도 없다."""
from __future__ import annotations

import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from image_ai_studio.application.training_controller import TrainingController, build_training_request
from image_ai_studio.gui.qt_training_worker import QtTrainingWorker
from image_ai_studio.training.config import LR_SCHEDULER_CHOICES, OPTIMIZER_CHOICES, PRECISION_CHOICES
from image_ai_studio.training.imagefolder_workflow import SEED, ImageFolderWorkflowResult
from image_ai_studio.training.loop import TrainingProgress

_NO_SCHEDULER_LABEL = "None"
_STOP_REASON_TEXT = {
    "completed": "Completed",
    "early_stopped": "Training stopped by early stopping",
    "user_stopped": "Training stopped by user",
}


def parse_class_weights(text: str) -> tuple[float, ...] | None:
    """GUI 텍스트 입력(comma-separated 숫자)을 `class_weights`로 변환한다.
    빈 문자열/공백만 있으면 `None`. **숫자 형식 자체가 잘못된 경우만**
    `ValueError`로 거부한다 -- 각 weight의 semantic validity(0보다
    큰지 등)는 검증하지 않는다, 그건 `TrainingConfig`의 책임이다(순수
    textual parsing과 semantic validation을 구분)."""
    stripped = text.strip()
    if not stripped:
        return None
    parts = [p.strip() for p in stripped.split(",")]
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"invalid class weights {text!r}: expected comma-separated numbers") from exc


def _empty_to_none(text: str) -> str | None:
    """빈 문자열/공백을 `None`으로 -- `Path("")`로 잘못 변환되지 않게
    한다(optional path field 공통 처리)."""
    stripped = text.strip()
    return stripped if stripped else None


def _detect_device_choices() -> list[str]:
    choices = ["cpu"]
    if torch.cuda.is_available():
        choices.append("cuda")
        for index in range(torch.cuda.device_count()):
            choices.append(f"cuda:{index}")
    return choices


class TrainingPage(QWidget):
    """Model/Dataset/Output 선택 -> Basic/Advanced config -> Start/Stop ->
    Progress -> Result 전부를 담은 단일 페이지. `close_requested` signal은
    학습 도중 종료가 요청됐다가 실제로 안전하게 멈춘 뒤에만 emit된다
    (MainWindow가 이 signal을 받아 실제 `close()`를 수행-- §"close 처리"
    참고)."""

    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None, *, controller: TrainingController | None = None) -> None:
        super().__init__(parent)
        self._controller = controller if controller is not None else TrainingController()
        self._thread: QThread | None = None
        self._worker: QtTrainingWorker | None = None
        self._close_pending = False

        self._build_ui()
        self._on_scheduler_changed(self._scheduler_combo.currentText())
        self._reset_progress_display()

    # -- public: MainWindow.closeEvent()가 사용 ---------------------------------

    def is_training_active(self) -> bool:
        return self._controller.is_running

    def request_stop_and_close(self) -> None:
        """학습 중이 아니면 즉시 `close_requested`를 emit한다. 학습
        중이면 cooperative stop을 요청하고, 실제 close는 그 학습이
        finished/failed로 끝난 뒤(`_finish_common()`)에만 일어난다 --
        `QThread.terminate()`나 강제 종료는 쓰지 않는다."""
        if not self.is_training_active():
            self.close_requested.emit()
            return
        self._close_pending = True
        self._on_stop_clicked()

    # -- UI 구성 -----------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_dataset_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_basic_group())
        layout.addWidget(self._build_advanced_group())
        layout.addLayout(self._build_actions_row())
        layout.addWidget(self._build_progress_group())
        layout.addWidget(self._build_result_group())
        layout.addStretch(1)

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        self._model_json_edit = QLineEdit()
        self._model_browse_button = QPushButton("Browse...")
        self._model_browse_button.clicked.connect(self._on_browse_model_json)
        row = QHBoxLayout()
        row.addWidget(QLabel("Model JSON:"))
        row.addWidget(self._model_json_edit)
        row.addWidget(self._model_browse_button)
        group.setLayout(row)
        return group

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Dataset")
        self._dataset_root_edit = QLineEdit()
        self._dataset_browse_button = QPushButton("Browse...")
        self._dataset_browse_button.clicked.connect(self._on_browse_dataset_root)
        row = QHBoxLayout()
        row.addWidget(QLabel("ImageFolder root:"))
        row.addWidget(self._dataset_root_edit)
        row.addWidget(self._dataset_browse_button)
        group.setLayout(row)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Output")
        self._output_dir_edit = QLineEdit()
        self._output_browse_button = QPushButton("Browse...")
        self._output_browse_button.clicked.connect(self._on_browse_output_dir)
        row = QHBoxLayout()
        row.addWidget(QLabel("Output directory:"))
        row.addWidget(self._output_dir_edit)
        row.addWidget(self._output_browse_button)
        group.setLayout(row)
        return group

    def _build_basic_group(self) -> QGroupBox:
        group = QGroupBox("Training Settings")
        form = QFormLayout()

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 1_000_000)
        self._epochs_spin.setValue(5)
        form.addRow("Epochs:", self._epochs_spin)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 1_000_000)
        self._batch_size_spin.setValue(8)
        form.addRow("Batch size:", self._batch_size_spin)

        self._learning_rate_spin = QDoubleSpinBox()
        self._learning_rate_spin.setDecimals(8)
        self._learning_rate_spin.setRange(1e-8, 1000.0)
        self._learning_rate_spin.setSingleStep(0.0001)
        self._learning_rate_spin.setValue(1e-3)
        form.addRow("Learning rate:", self._learning_rate_spin)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(list(OPTIMIZER_CHOICES))
        form.addRow("Optimizer:", self._optimizer_combo)

        self._device_combo = QComboBox()
        self._device_combo.addItems(_detect_device_choices())
        form.addRow("Device:", self._device_combo)

        self._precision_combo = QComboBox()
        self._precision_combo.addItems(list(PRECISION_CHOICES))
        form.addRow("Precision:", self._precision_combo)

        group.setLayout(form)
        return group

    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Advanced Settings")
        form = QFormLayout()

        self._momentum_spin = QDoubleSpinBox()
        self._momentum_spin.setDecimals(4)
        self._momentum_spin.setRange(0.0, 1.0)
        self._momentum_spin.setSingleStep(0.01)
        self._momentum_spin.setValue(0.9)
        form.addRow("Momentum (SGD):", self._momentum_spin)

        self._weight_decay_spin = QDoubleSpinBox()
        self._weight_decay_spin.setDecimals(6)
        self._weight_decay_spin.setRange(0.0, 1_000_000.0)
        self._weight_decay_spin.setValue(0.0)
        form.addRow("Weight decay:", self._weight_decay_spin)

        self._gradient_clip_check, self._gradient_clip_spin = self._build_optional_double_row(
            form, "Gradient clip norm:", default=1.0, minimum=1e-8, maximum=1_000_000.0, decimals=4
        )

        self._label_smoothing_spin = QDoubleSpinBox()
        self._label_smoothing_spin.setDecimals(2)
        self._label_smoothing_spin.setRange(0.0, 1.0)
        self._label_smoothing_spin.setSingleStep(0.01)
        self._label_smoothing_spin.setValue(0.0)
        form.addRow("Label smoothing:", self._label_smoothing_spin)

        self._class_weights_edit = QLineEdit()
        self._class_weights_edit.setPlaceholderText("e.g. 1.0, 2.0, 1.5 (leave empty for None)")
        form.addRow("Class weights:", self._class_weights_edit)

        self._scheduler_combo = QComboBox()
        self._scheduler_combo.addItems([_NO_SCHEDULER_LABEL, *LR_SCHEDULER_CHOICES])
        self._scheduler_combo.currentTextChanged.connect(self._on_scheduler_changed)
        form.addRow("LR scheduler:", self._scheduler_combo)

        self._scheduler_factor_spin = QDoubleSpinBox()
        self._scheduler_factor_spin.setDecimals(4)
        self._scheduler_factor_spin.setRange(0.0001, 0.9999)
        self._scheduler_factor_spin.setValue(0.1)
        form.addRow("Scheduler factor:", self._scheduler_factor_spin)

        self._scheduler_patience_spin = QSpinBox()
        self._scheduler_patience_spin.setRange(1, 1_000_000)
        self._scheduler_patience_spin.setValue(1)
        form.addRow("Scheduler patience:", self._scheduler_patience_spin)

        self._early_stopping_check, self._early_stopping_spin = self._build_optional_int_row(
            form, "Early stopping patience:", default=5, minimum=1, maximum=1_000_000
        )

        self._checkpoint_every_check, self._checkpoint_every_spin = self._build_optional_int_row(
            form, "Checkpoint every N epochs:", default=1, minimum=1, maximum=1_000_000
        )

        self._pin_memory_check = QCheckBox("pin_memory (CUDA only)")
        form.addRow(self._pin_memory_check)

        self._non_blocking_check = QCheckBox("non_blocking (CUDA only)")
        form.addRow(self._non_blocking_check)

        self._export_torchscript_check = QCheckBox("Export TorchScript")
        self._export_torchscript_check.setChecked(True)
        form.addRow(self._export_torchscript_check)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(SEED)
        form.addRow("Seed:", self._seed_spin)

        self._resume_from_edit = QLineEdit()
        self._resume_browse_button = QPushButton("Browse...")
        self._resume_browse_button.clicked.connect(self._on_browse_resume_from)
        self._resume_clear_button = QPushButton("Clear")
        self._resume_clear_button.clicked.connect(self._resume_from_edit.clear)
        resume_row = QHBoxLayout()
        resume_row.addWidget(self._resume_from_edit)
        resume_row.addWidget(self._resume_browse_button)
        resume_row.addWidget(self._resume_clear_button)
        form.addRow("Resume from:", resume_row)

        self._checkpoint_out_edit = QLineEdit()
        self._checkpoint_out_browse_button = QPushButton("Browse...")
        self._checkpoint_out_browse_button.clicked.connect(self._on_browse_checkpoint_out)
        checkpoint_out_row = QHBoxLayout()
        checkpoint_out_row.addWidget(self._checkpoint_out_edit)
        checkpoint_out_row.addWidget(self._checkpoint_out_browse_button)
        form.addRow("Checkpoint out:", checkpoint_out_row)

        group.setLayout(form)
        return group

    def _build_optional_double_row(
        self, form: QFormLayout, label: str, *, default: float, minimum: float, maximum: float, decimals: int
    ) -> tuple[QCheckBox, QDoubleSpinBox]:
        """[Enable checkbox] + QDoubleSpinBox 조합 -- 체크 해제 상태가
        `None`을 뜻한다(값을 0 같은 sentinel로 몰래 바꾸지 않는다)."""
        checkbox = QCheckBox("Enable")
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setValue(default)
        spin.setEnabled(False)
        checkbox.toggled.connect(spin.setEnabled)
        row = QHBoxLayout()
        row.addWidget(checkbox)
        row.addWidget(spin)
        form.addRow(label, row)
        return checkbox, spin

    def _build_optional_int_row(
        self, form: QFormLayout, label: str, *, default: int, minimum: int, maximum: int
    ) -> tuple[QCheckBox, QSpinBox]:
        checkbox = QCheckBox("Enable")
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(default)
        spin.setEnabled(False)
        checkbox.toggled.connect(spin.setEnabled)
        row = QHBoxLayout()
        row.addWidget(checkbox)
        row.addWidget(spin)
        form.addRow(label, row)
        return checkbox, spin

    def _build_actions_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self._on_start_clicked)
        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        row.addWidget(self._start_button)
        row.addWidget(self._stop_button)
        return row

    def _build_progress_group(self) -> QGroupBox:
        group = QGroupBox("Progress")
        form = QFormLayout()

        self._status_label = QLabel("Idle")
        form.addRow("Status:", self._status_label)

        self._progress_bar = QProgressBar()
        form.addRow("Run progress:", self._progress_bar)

        self._global_epoch_label = QLabel("--")
        form.addRow("Global epoch:", self._global_epoch_label)

        self._train_loss_label = QLabel("--")
        form.addRow("Train loss:", self._train_loss_label)

        self._val_loss_label = QLabel("--")
        form.addRow("Validation loss:", self._val_loss_label)

        self._val_accuracy_label = QLabel("--")
        form.addRow("Validation accuracy:", self._val_accuracy_label)

        self._learning_rate_label = QLabel("--")
        form.addRow("Learning rate:", self._learning_rate_label)

        self._best_epoch_label = QLabel("--")
        form.addRow("Best epoch:", self._best_epoch_label)

        self._best_val_loss_label = QLabel("--")
        form.addRow("Best validation loss:", self._best_val_loss_label)

        self._epoch_duration_label = QLabel("--")
        form.addRow("Epoch duration:", self._epoch_duration_label)

        group.setLayout(form)
        return group

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("Result")
        form = QFormLayout()

        self._test_loss_label = QLabel("--")
        form.addRow("Test loss:", self._test_loss_label)

        self._test_accuracy_label = QLabel("--")
        form.addRow("Test accuracy:", self._test_accuracy_label)

        self._artifact_labels: dict[str, QLabel] = {}
        for key, title in (
            ("best_model_state_dict_path", "Best model:"),
            ("training_history_path", "Training history:"),
            ("class_mapping_path", "Class mapping:"),
            ("test_result_path", "Test result:"),
            ("checkpoint_path", "Checkpoint:"),
            ("torchscript_model_path", "TorchScript model:"),
        ):
            label = QLabel("--")
            label.setWordWrap(True)
            self._artifact_labels[key] = label
            form.addRow(title, label)

        self._error_summary_label = QLabel("")
        form.addRow("Error:", self._error_summary_label)

        self._details_text = QPlainTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setMaximumHeight(120)
        form.addRow("Details:", self._details_text)

        group.setLayout(form)
        return group

    # -- Browse handlers ----------------------------------------------------------

    def _on_browse_model_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Model JSON", "", "JSON files (*.json);;All files (*)")
        if path:
            self._model_json_edit.setText(path)

    def _on_browse_dataset_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select ImageFolder Root")
        if path:
            self._dataset_root_edit.setText(path)

    def _on_browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir_edit.setText(path)

    def _on_browse_resume_from(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", "", "Checkpoint files (*.pt);;All files (*)")
        if path:
            self._resume_from_edit.setText(path)

    def _on_browse_checkpoint_out(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select Checkpoint Output Path", "", "Checkpoint files (*.pt);;All files (*)")
        if path:
            self._checkpoint_out_edit.setText(path)

    def _on_scheduler_changed(self, text: str) -> None:
        enabled = text != _NO_SCHEDULER_LABEL
        self._scheduler_factor_spin.setEnabled(enabled)
        self._scheduler_patience_spin.setEnabled(enabled)

    # -- request 조립 ---------------------------------------------------------------

    def _build_request(self):
        """현재 widget 값의 snapshot에서 `ImageFolderWorkflowRequest`를
        만든다 -- `build_training_request()`(Phase 5B)를 그대로 재사용
        하고, 이 메서드는 UI 값을 그 함수의 인자 형태로 변환만 한다.
        semantic validation은 하지 않는다(그 함수/`TrainingConfig`의
        책임)."""
        scheduler_text = self._scheduler_combo.currentText()
        lr_scheduler = None if scheduler_text == _NO_SCHEDULER_LABEL else scheduler_text

        return build_training_request(
            model_json_path=self._model_json_edit.text().strip(),
            dataset_root=self._dataset_root_edit.text().strip(),
            output_dir=self._output_dir_edit.text().strip(),
            epochs=self._epochs_spin.value(),
            batch_size=self._batch_size_spin.value(),
            learning_rate=self._learning_rate_spin.value(),
            optimizer=self._optimizer_combo.currentText(),
            momentum=self._momentum_spin.value(),
            weight_decay=self._weight_decay_spin.value(),
            gradient_clip_norm=(self._gradient_clip_spin.value() if self._gradient_clip_check.isChecked() else None),
            label_smoothing=self._label_smoothing_spin.value(),
            class_weights=parse_class_weights(self._class_weights_edit.text()),
            lr_scheduler=lr_scheduler,
            lr_scheduler_factor=self._scheduler_factor_spin.value(),
            lr_scheduler_patience=self._scheduler_patience_spin.value(),
            early_stopping_patience=(
                self._early_stopping_spin.value() if self._early_stopping_check.isChecked() else None
            ),
            precision=self._precision_combo.currentText(),
            device=self._device_combo.currentText(),
            pin_memory=self._pin_memory_check.isChecked(),
            non_blocking=self._non_blocking_check.isChecked(),
            resume_from=_empty_to_none(self._resume_from_edit.text()),
            checkpoint_out=_empty_to_none(self._checkpoint_out_edit.text()),
            checkpoint_every=(
                self._checkpoint_every_spin.value() if self._checkpoint_every_check.isChecked() else None
            ),
            export_torchscript=self._export_torchscript_check.isChecked(),
            seed=self._seed_spin.value(),
        )

    # -- Start/Stop lifecycle -------------------------------------------------------

    def _on_start_clicked(self) -> None:
        try:
            request = self._build_request()
        except Exception as exc:  # noqa: BLE001 -- request 조립 실패(예: TrainingConfig validation)
            # 이 시점에는 controller.begin_run()이 아직 호출되지 않았다
            # (worker/thread도 생성되지 않음) -- application state는
            # 손대지 않고 GUI에만 실패를 표시한다(§"request 조립 실패").
            self._show_failure(f"{type(exc).__name__}: {exc}")
            return

        self._thread = QThread(self)
        self._worker = QtTrainingWorker(self._controller, request)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # signal을 QObject bound method(self의 메서드)에 connect한다 --
        # plain 함수에 connect하면 GUI thread가 아니라 emit이 일어난
        # worker thread에서 직접 실행된다(Phase 5B에서 empirical 확인,
        # docs/phase5b_..._design.md §9 참고).
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        # worker.deleteLater()는 worker 자신의 finished/failed에
        # connect한다(Qt의 canonical moveToThread 패턴 -- worker thread의
        # event loop가 아직 살아있는 동안 deferred deletion이 그
        # thread에 posting된다). thread.finished에 connect하면
        # worker thread의 event loop가 이미 quit()으로 멈춘 뒤라
        # worker에 대한 deleteLater가 안전하게 처리되지 않을 수 있다
        # (Phase 5C stabilization에서 드문 native abort의 원인으로
        # 지목됨, docs/phase5c_training_gui_design.md 참고).
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._set_controls_enabled(False)
        self._reset_progress_display()
        self._clear_result_display()
        self._status_label.setText("Running...")
        self._stop_button.setEnabled(True)
        self._thread.start()

    def _on_stop_clicked(self) -> None:
        self._controller.request_stop()
        self._status_label.setText("Stopping after current epoch...")
        self._stop_button.setEnabled(False)

    # -- worker signal handlers -------------------------------------------------------

    def _on_progress(self, progress: TrainingProgress) -> None:
        self._progress_bar.setMaximum(max(progress.total_run_epochs, 1))
        self._progress_bar.setValue(progress.run_epoch)
        self._global_epoch_label.setText(str(progress.global_epoch))
        self._train_loss_label.setText(f"{progress.train_loss:.4f}")
        self._val_loss_label.setText(f"{progress.val_loss:.4f}")
        self._val_accuracy_label.setText(f"{progress.val_accuracy:.4f}")
        self._learning_rate_label.setText(f"{progress.learning_rate:.6g}")
        self._best_epoch_label.setText(str(progress.best_epoch))
        self._best_val_loss_label.setText(f"{progress.best_val_loss:.4f}")
        self._epoch_duration_label.setText(f"{progress.epoch_duration_seconds:.2f}s")

    def _on_finished(self, result: ImageFolderWorkflowResult) -> None:
        self._status_label.setText(_STOP_REASON_TEXT.get(result.stop_reason, result.stop_reason))
        self._test_loss_label.setText(f"{result.test_loss:.4f}")
        self._test_accuracy_label.setText(f"{result.test_accuracy:.4f}")
        for key, label in self._artifact_labels.items():
            value = getattr(result, key)
            label.setText(str(value) if value is not None else "Not generated")
        self._finish_common()

    def _on_failed(self, message: str) -> None:
        first_line = message.splitlines()[0] if message else "Unknown error"
        self._show_failure(first_line, details=message)
        self._finish_common()

    def _show_failure(self, summary: str, *, details: str | None = None) -> None:
        self._status_label.setText("Failed")
        self._error_summary_label.setText(summary)
        self._details_text.setPlainText(details if details is not None else summary)

    def _finish_common(self) -> None:
        self._set_controls_enabled(True)
        self._stop_button.setEnabled(False)
        if self._close_pending:
            self._close_pending = False
            self.close_requested.emit()

    # -- helpers -------------------------------------------------------------------

    def _reset_progress_display(self) -> None:
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(1)
        self._progress_bar.setValue(0)
        for label in (
            self._global_epoch_label, self._train_loss_label, self._val_loss_label,
            self._val_accuracy_label, self._learning_rate_label, self._best_epoch_label,
            self._best_val_loss_label, self._epoch_duration_label,
        ):
            label.setText("--")

    def _clear_result_display(self) -> None:
        self._test_loss_label.setText("--")
        self._test_accuracy_label.setText("--")
        for label in self._artifact_labels.values():
            label.setText("--")
        self._error_summary_label.setText("")
        self._details_text.setPlainText("")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._model_json_edit, self._model_browse_button,
            self._dataset_root_edit, self._dataset_browse_button,
            self._output_dir_edit, self._output_browse_button,
            self._epochs_spin, self._batch_size_spin, self._learning_rate_spin,
            self._optimizer_combo, self._device_combo, self._precision_combo,
            self._momentum_spin, self._weight_decay_spin, self._gradient_clip_check,
            self._label_smoothing_spin, self._class_weights_edit, self._scheduler_combo,
            self._early_stopping_check, self._checkpoint_every_check,
            self._pin_memory_check, self._non_blocking_check, self._export_torchscript_check,
            self._seed_spin, self._resume_from_edit, self._resume_browse_button,
            self._resume_clear_button, self._checkpoint_out_edit, self._checkpoint_out_browse_button,
            self._start_button,
        ):
            widget.setEnabled(enabled)
        # optional numeric spin box들은 각자의 checkbox 상태를 따라야
        # 한다(체크 해제된 채로 다시 활성화되면 안 됨).
        self._gradient_clip_spin.setEnabled(enabled and self._gradient_clip_check.isChecked())
        self._early_stopping_spin.setEnabled(enabled and self._early_stopping_check.isChecked())
        self._checkpoint_every_spin.setEnabled(enabled and self._checkpoint_every_check.isChecked())
        scheduler_enabled = enabled and self._scheduler_combo.currentText() != _NO_SCHEDULER_LABEL
        self._scheduler_factor_spin.setEnabled(scheduler_enabled)
        self._scheduler_patience_spin.setEnabled(scheduler_enabled)
