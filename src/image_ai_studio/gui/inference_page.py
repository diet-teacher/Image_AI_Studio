"""Phase 6C CP2: InferencePage async lifecycle. Extends CP1's request-building
form with the actual Run Inference execution -- a QThread + QtInferenceWorker
pair wired around the injected InferenceController, following the same
canonical lifecycle as TrainingPage/QtTrainingWorker (Phase 5C):
started->run, finished/failed->page slots + thread.quit + worker.deleteLater,
thread.finished->thread.deleteLater + reference cleanup. No MainWindow
integration and no inference-result rendering here -- only lifecycle state
(Running/Finished/Failed) and control enablement."""
from __future__ import annotations

from pathlib import Path

import torch
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from image_ai_studio.application.inference_controller import InferenceController, build_inference_request
from image_ai_studio.gui.qt_inference_worker import QtInferenceWorker
from image_ai_studio.inference.single_image_inference import InferenceRequest, InferenceResult
from image_ai_studio.training.config import PRECISION_CHOICES


def _detect_device_choices() -> list[str]:
    choices = ["cpu"]
    if torch.cuda.is_available():
        choices.append("cuda")
        for index in range(torch.cuda.device_count()):
            choices.append(f"cuda:{index}")
    return choices


class InferencePage(QWidget):
    """Single-image inference form. Accepts optional InferenceController
    injection for tests. Run Inference builds the request (CP1), then runs
    it asynchronously via QThread + QtInferenceWorker (CP2) -- construction
    itself creates no thread/worker and starts no inference."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        controller: InferenceController | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller if controller is not None else InferenceController()
        self._thread: QThread | None = None
        self._worker: QtInferenceWorker | None = None
        self._build_ui()

    # -- public request builder (used by tests and future Run handler) ---------

    def _build_request(self) -> InferenceRequest:
        """Convert widget values to InferenceRequest. Derives the two fixed
        artifact paths from the training output directory."""
        output_dir = self._training_output_dir_edit.text().strip()
        output_path = Path(output_dir) if output_dir else Path("")
        return build_inference_request(
            model_json_path=self._model_json_edit.text().strip(),
            state_dict_path=str(output_path / "best_model_state_dict.pt"),
            class_mapping_path=str(output_path / "class_mapping.json"),
            image_path=self._image_path_edit.text().strip(),
            device=self._device_combo.currentText(),
            precision=self._precision_combo.currentText(),
        )

    # -- UI construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_inputs_group())
        layout.addLayout(self._build_actions_row())
        layout.addWidget(self._build_status_label())
        layout.addStretch(1)

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

    def _on_run_clicked(self) -> None:
        if self._worker is not None or self._thread is not None:
            # A run is already active -- overlap prevention even if this
            # handler is invoked directly while a previous run has not
            # finished cleaning up yet.
            return

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

        self._set_controls_enabled(False)
        self._status_label.setText("Running")
        self._thread.start()

    # -- worker signal handlers -------------------------------------------------

    def _on_finished(self, result: InferenceResult) -> None:
        self._status_label.setText("Finished")

    def _on_failed(self, message: str) -> None:
        first_line = message.splitlines()[0] if message else "Unknown error"
        self._status_label.setText(f"Failed: {first_line}")

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_controls_enabled(True)

    # -- helpers -----------------------------------------------------------------

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
            self._run_button,
        ):
            widget.setEnabled(enabled)
