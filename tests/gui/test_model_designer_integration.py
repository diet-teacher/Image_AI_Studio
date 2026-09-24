"""Phase 14 CP3 bounded CPU graduation for the integrated Model Designer.

The test deliberately crosses the real GUI, canonical model-definition, request
builder, Qt worker, and ImageFolder training boundaries.  It uses only a tiny
local fixture and never substitutes a hand-authored model JSON or fake training
result for the workflow being graduated.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QLineEdit
import torch

from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import load_model_spec, model_spec_to_dict


_CLASS_COLORS = {"cat": (240, 240, 240), "dog": (15, 15, 15)}


def _make_tiny_imagefolder(root: Path) -> None:
    """Create deterministic two-class train/val/test splits without downloads."""
    for split in ("train", "val", "test"):
        for class_name, color in _CLASS_COLORS.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (12, 12), color=color).save(class_dir / f"{index}.png")


def _set_small_two_class_model(window: MainWindow, qtbot) -> None:
    """Edit the starter model through its live controls, including layer form."""
    designer = window._model_designer
    designer._name_edit.setText("phase14_gui_cpu_graduation")
    designer._channels_spin.setText("3")
    designer._height_spin.setText("8")
    designer._width_spin.setText("8")

    # The starter order is Conv2d -> ReLU -> Flatten -> Linear.  Select the
    # actual Linear row and edit its integer form control instead of replacing
    # the layer data with a hand-built dictionary.
    designer._layer_panel.select_row(3)
    parameter_form = designer._layer_panel._param_widget
    assert parameter_form is not None
    integer_edits = parameter_form.findChildren(QLineEdit)
    assert len(integer_edits) == 1
    integer_edits[0].selectAll()
    qtbot.keyClicks(integer_edits[0], "2")

    assert [layer["type"] for layer in designer._layer_panel.layers] == [
        "conv2d",
        "relu",
        "flatten",
        "linear",
    ]
    assert designer._layer_panel.layers[-1]["out_features"] == 2


def _thread_is_cleaned_up(window: MainWindow) -> bool:
    try:
        thread = window._training_page._thread
        return thread is None or thread.isRunning() is False
    except RuntimeError:
        # deleteLater() has released the underlying C++ QThread object.
        return True


def test_model_designer_to_real_cpu_training_graduation(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "training-output"
    invalid_target = tmp_path / "invalid.json"
    first_target = tmp_path / "designer-model.json"
    final_target = tmp_path / "designer-model-rerun.json"
    _make_tiny_imagefolder(dataset_root)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(window.isVisible)

    designer = window._model_designer
    training = window._training_page
    tabs = window._tabs
    assert tabs.count() == 3
    assert tabs.widget(0) is training
    assert tabs.widget(2) is designer
    assert tabs.tabText(2) == "Model Designer"

    tabs.setCurrentWidget(designer)
    training.set_model_json_path("preserved-before-valid-save.json")
    emitted_paths: list[str] = []
    designer.model_saved_for_training.connect(emitted_paths.append)

    dialog_targets = iter(
        [(str(invalid_target), ""), (str(first_target), ""), (str(final_target), "")]
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: next(dialog_targets),
    )

    # Invalid editor data must not reach the handoff or training controller.
    designer._name_edit.setText("")
    qtbot.mouseClick(designer._use_for_training_button, Qt.MouseButton.LeftButton)
    assert designer._status_label.text().startswith("Save failed:")
    assert emitted_paths == []
    assert not invalid_target.exists()
    assert training._model_json_edit.text() == "preserved-before-valid-save.json"
    assert tabs.currentWidget() is designer
    assert training._controller.state == "idle"

    # Correct the same editor, obtain real canonical shape feedback, and save.
    _set_small_two_class_model(window, qtbot)
    qtbot.mouseClick(designer._validate_button, Qt.MouseButton.LeftButton)
    assert designer._status_label.text() == "Valid"
    assert "Conv2d" in designer._shape_trace_label.text()
    assert "[1024] -> [2]" in designer._shape_trace_label.text()
    expected_definition = model_spec_to_dict(designer._build_model_spec())

    qtbot.mouseClick(designer._use_for_training_button, Qt.MouseButton.LeftButton)
    first_absolute = str(first_target.resolve())
    assert emitted_paths == [first_absolute]
    assert first_target.exists()
    assert training._model_json_edit.text() == first_absolute
    assert tabs.currentWidget() is training
    assert model_spec_to_dict(load_model_spec(first_target)) == expected_definition

    # A user edit stays authoritative across navigation until another explicit
    # successful designer handoff replaces it.  Re-entry must not duplicate the
    # single MainWindow connection.
    training._model_json_edit.setText("manual-override.json")
    tabs.setCurrentWidget(designer)
    tabs.setCurrentWidget(training)
    assert training._model_json_edit.text() == "manual-override.json"
    tabs.setCurrentWidget(designer)
    qtbot.mouseClick(designer._use_for_training_button, Qt.MouseButton.LeftButton)
    final_absolute = str(final_target.resolve())
    assert emitted_paths == [first_absolute, final_absolute]
    assert training._model_json_edit.text() == final_absolute
    assert tabs.currentWidget() is training
    assert model_spec_to_dict(load_model_spec(final_target)) == expected_definition

    # Prove that the canonical file builds and performs a CPU forward pass
    # before exercising that same path through TrainingPage's request boundary.
    saved_spec = load_model_spec(final_target)
    model = build_model(saved_spec).cpu().eval()
    assert [type(layer).__name__ for layer in saved_spec.layers] == [
        "Conv2dSpec",
        "ReLUSpec",
        "FlattenSpec",
        "LinearSpec",
    ]
    with torch.inference_mode():
        prediction = model(torch.zeros(1, *saved_spec.input_shape))
    assert prediction.device.type == "cpu"
    assert tuple(prediction.shape) == (1, 2)

    training._dataset_root_edit.setText(str(dataset_root))
    training._output_dir_edit.setText(str(output_dir))
    training._epochs_spin.setValue(1)
    training._batch_size_spin.setValue(2)
    training._learning_rate_spin.setValue(1e-2)
    training._device_combo.setCurrentText("cpu")
    training._export_torchscript_check.setChecked(False)

    request = training._build_request()
    assert request.model_json_path == final_target.resolve()
    assert request.dataset_root == dataset_root
    assert request.output_dir == output_dir
    assert request.device == "cpu"

    qtbot.mouseClick(training._start_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: training._start_button.isEnabled(), timeout=30_000)
    qtbot.waitUntil(lambda: _thread_is_cleaned_up(window), timeout=5_000)

    assert training._status_label.text() == "Completed"
    assert training._controller.state == "finished"
    assert training._stop_button.isEnabled() is False
    for artifact_name in (
        "model_definition.json",
        "best_model_state_dict.pt",
        "training_history.json",
        "class_mapping.json",
        "test_result.json",
    ):
        assert (output_dir / artifact_name).is_file()
    assert model_spec_to_dict(load_model_spec(output_dir / "model_definition.json")) == expected_definition
