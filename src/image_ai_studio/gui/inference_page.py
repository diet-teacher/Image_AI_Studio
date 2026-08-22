"""Phase 6C CP1: InferencePage form -- widget layout and InferenceRequest
building only. No QThread, no QtInferenceWorker, no inference execution."""
from __future__ import annotations

from pathlib import Path

import torch
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
from image_ai_studio.inference.single_image_inference import InferenceRequest
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
    injection for later lifecycle work and tests; CP1 only builds requests."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        controller: InferenceController | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller if controller is not None else InferenceController()
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
        browse_output = QPushButton("Browse...")
        browse_output.clicked.connect(self._browse_training_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self._training_output_dir_edit)
        output_row.addWidget(browse_output)
        form.addRow("Training Output Dir:", output_row)

        self._model_json_edit = QLineEdit()
        browse_model = QPushButton("Browse...")
        browse_model.clicked.connect(self._browse_model_json)
        model_row = QHBoxLayout()
        model_row.addWidget(self._model_json_edit)
        model_row.addWidget(browse_model)
        form.addRow("Model JSON:", model_row)

        self._image_path_edit = QLineEdit()
        browse_image = QPushButton("Browse...")
        browse_image.clicked.connect(self._browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self._image_path_edit)
        image_row.addWidget(browse_image)
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
        # CP1: validates request construction only; inference execution deferred to CP2+
        try:
            self._build_request()
        except Exception as exc:
            self._status_label.setText(f"Failed: {exc}")
            return
        self._status_label.setText("Ready")
