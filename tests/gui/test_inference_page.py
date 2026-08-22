"""Phase 6C CP1: InferencePage widget wiring tests. Uses a fake backend
injected into InferenceController to avoid real model/image loading.
Verifies initial state, fixed artifact path derivation, and exact
widget-to-InferenceRequest mapping."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QFormLayout

from image_ai_studio.application.inference_controller import InferenceController
from image_ai_studio.gui.inference_page import InferencePage

pytestmark = pytest.mark.phase6c_cp1_inference_page


def _fake_controller() -> InferenceController:
    """InferenceController with a backend that should never be called in CP1."""

    def _should_not_run(request):
        raise AssertionError("inference backend must not be called in CP1 tests")

    return InferenceController(backend=_should_not_run)


def _make_page(qtbot) -> InferencePage:
    page = InferencePage(controller=_fake_controller())
    qtbot.addWidget(page)
    return page


# -- construction --------------------------------------------------------------


def test_construction_does_not_create_thread_or_worker(qtbot) -> None:
    page = _make_page(qtbot)
    assert not hasattr(page, "_thread") or getattr(page, "_thread", None) is None
    assert not hasattr(page, "_worker") or getattr(page, "_worker", None) is None


def test_optional_controller_injection_accepted(qtbot) -> None:
    controller = _fake_controller()
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    assert page._controller is controller


def test_default_controller_created_when_none_injected(qtbot) -> None:
    page = InferencePage()
    qtbot.addWidget(page)
    assert isinstance(page._controller, InferenceController)


# -- initial widget state ------------------------------------------------------


def test_initial_status_is_idle(qtbot) -> None:
    page = _make_page(qtbot)
    assert page._status_label.text() == "Idle"


def test_run_button_enabled_initially(qtbot) -> None:
    page = _make_page(qtbot)
    assert page._run_button.isEnabled() is True


def test_run_button_label(qtbot) -> None:
    page = _make_page(qtbot)
    assert page._run_button.text() == "Run Inference"


def test_all_path_fields_empty_initially(qtbot) -> None:
    page = _make_page(qtbot)
    assert page._training_output_dir_edit.text() == ""
    assert page._model_json_edit.text() == ""
    assert page._image_path_edit.text() == ""


def test_device_combo_contains_cpu(qtbot) -> None:
    page = _make_page(qtbot)
    items = [page._device_combo.itemText(i) for i in range(page._device_combo.count())]
    assert "cpu" in items


def test_precision_combo_contains_fp32(qtbot) -> None:
    page = _make_page(qtbot)
    items = [page._precision_combo.itemText(i) for i in range(page._precision_combo.count())]
    assert "fp32" in items


def test_precision_combo_default_is_fp32(qtbot) -> None:
    page = _make_page(qtbot)
    assert page._precision_combo.currentText() == "fp32"


# -- visible form labels -------------------------------------------------------


def _form_label_text(page: InferencePage, row: int) -> str:
    """Return the visible label text for the given QFormLayout row index."""
    item = page._inputs_form.itemAt(row, QFormLayout.LabelRole)
    return item.widget().text()


def test_training_output_dir_label(qtbot) -> None:
    page = _make_page(qtbot)
    assert _form_label_text(page, 0) == "Training Output Dir:"


def test_model_json_label(qtbot) -> None:
    page = _make_page(qtbot)
    assert _form_label_text(page, 1) == "Model JSON:"


def test_input_image_label(qtbot) -> None:
    page = _make_page(qtbot)
    assert _form_label_text(page, 2) == "Input Image:"


def test_device_label(qtbot) -> None:
    page = _make_page(qtbot)
    assert _form_label_text(page, 3) == "Device:"


def test_precision_label(qtbot) -> None:
    page = _make_page(qtbot)
    assert _form_label_text(page, 4) == "Precision:"


# -- artifact path derivation --------------------------------------------------


def test_state_dict_path_derived_from_training_output_dir(tmp_path, qtbot) -> None:
    page = _make_page(qtbot)
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "image.png"))

    request = page._build_request()

    assert request.state_dict_path == tmp_path / "best_model_state_dict.pt"


def test_class_mapping_path_derived_from_training_output_dir(tmp_path, qtbot) -> None:
    page = _make_page(qtbot)
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "image.png"))

    request = page._build_request()

    assert request.class_mapping_path == tmp_path / "class_mapping.json"


def test_artifact_paths_use_fixed_filenames(tmp_path, qtbot) -> None:
    page = _make_page(qtbot)
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "image.png"))

    request = page._build_request()

    assert request.state_dict_path.name == "best_model_state_dict.pt"
    assert request.class_mapping_path.name == "class_mapping.json"


def test_artifact_paths_are_children_of_output_dir(tmp_path, qtbot) -> None:
    output_dir = tmp_path / "run_001"
    output_dir.mkdir()
    page = _make_page(qtbot)
    page._training_output_dir_edit.setText(str(output_dir))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "image.png"))

    request = page._build_request()

    assert request.state_dict_path.parent == output_dir
    assert request.class_mapping_path.parent == output_dir


# -- widget-to-request mapping -------------------------------------------------


def test_model_json_path_maps_to_request(tmp_path, qtbot) -> None:
    page = _make_page(qtbot)
    model_json = tmp_path / "arch.json"
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(model_json))
    page._image_path_edit.setText(str(tmp_path / "img.png"))

    request = page._build_request()

    assert request.model_json_path == model_json


def test_image_path_maps_to_request(tmp_path, qtbot) -> None:
    page = _make_page(qtbot)
    image = tmp_path / "cat.jpg"
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(image))

    request = page._build_request()

    assert request.image_path == image


def test_device_selection_maps_to_request(tmp_path, qtbot) -> None:
    page = _make_page(qtbot)
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "img.png"))
    page._device_combo.setCurrentText("cpu")

    request = page._build_request()

    assert request.device == "cpu"


def test_precision_selection_maps_to_request(tmp_path, qtbot) -> None:
    page = _make_page(qtbot)
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "img.png"))
    page._precision_combo.setCurrentText("fp32")

    request = page._build_request()

    assert request.precision == "fp32"


def test_full_request_mapping(tmp_path, qtbot) -> None:
    """All widget values flow through _build_request() to the right fields."""
    output_dir = tmp_path / "training_out"
    output_dir.mkdir()
    model_json = tmp_path / "resnet.json"
    image = tmp_path / "test.png"

    page = _make_page(qtbot)
    page._training_output_dir_edit.setText(str(output_dir))
    page._model_json_edit.setText(str(model_json))
    page._image_path_edit.setText(str(image))
    page._device_combo.setCurrentText("cpu")
    page._precision_combo.setCurrentText("fp32")

    request = page._build_request()

    assert request.model_json_path == model_json
    assert request.image_path == image
    assert request.state_dict_path == output_dir / "best_model_state_dict.pt"
    assert request.class_mapping_path == output_dir / "class_mapping.json"
    assert request.device == "cpu"
    assert request.precision == "fp32"
