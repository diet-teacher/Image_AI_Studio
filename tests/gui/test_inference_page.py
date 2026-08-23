"""Phase 6C CP1/CP2: InferencePage widget wiring tests. Uses a fake backend
injected into InferenceController to avoid real model/image loading.
CP1 verifies initial state, fixed artifact path derivation, and exact
widget-to-InferenceRequest mapping. CP2 (below) verifies the async
Run Inference lifecycle -- Running/Finished/Failed status, control
disable/enable, overlap prevention, off-GUI-thread backend execution,
worker/thread reference cleanup, and rerun -- using fake backends plus
threading.Event synchronization (same approach as test_training_page.py)."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFormLayout

from image_ai_studio.application.inference_controller import InferenceController
from image_ai_studio.gui.inference_page import InferencePage
from image_ai_studio.inference.single_image_inference import InferenceResult

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


# ===============================================================================
# Phase 6C CP2: async Run Inference lifecycle (QThread + QtInferenceWorker)
# ===============================================================================


def _fake_inference_result() -> InferenceResult:
    return InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1},
        inference_duration_seconds=0.01,
    )


def _fill_minimum_valid_fields(page: InferencePage, tmp_path: Path) -> None:
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "image.png"))


def _start_and_wait(page: InferencePage, qtbot, timeout: int = 5000) -> None:
    """`page._on_run_clicked()`을 호출한 뒤 lifecycle cleanup
    (`_on_thread_finished`, controls 재활성화 + 참조 정리)까지 끝날
    때까지 polling한다 -- `qtbot.waitSignal()`은 쓰지 않는다
    (test_training_page.py의 `_start_and_wait()`와 동일한 이유: fake
    backend가 즉시 반환하면 signal을 놓칠 수 있고, `_run_button`
    재활성화 시점과 signal fire 시점이 반드시 같지 않다)."""
    page._on_run_clicked()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=timeout)


# -- construction stays side-effect free even with lifecycle attributes ---------


def test_construction_still_side_effect_free_with_cp2_attributes(qtbot) -> None:
    page = _make_page(qtbot)
    assert page._thread is None
    assert page._worker is None


# -- Running lifecycle ------------------------------------------------------------


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_run_disables_controls_and_shows_running(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    controller = InferenceController(backend=blocking_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    assert page._status_label.text() == "Running"
    assert page._run_button.isEnabled() is False
    assert page._training_output_dir_edit.isEnabled() is False
    assert page._model_json_edit.isEnabled() is False
    assert page._image_path_edit.isEnabled() is False
    assert page._device_combo.isEnabled() is False
    assert page._precision_combo.isEnabled() is False

    let_backend_finish.set()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=5000)
    assert page._run_button.isEnabled() is True


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_backend_runs_off_gui_thread(tmp_path, qtbot) -> None:
    main_thread_id = threading.get_ident()
    observed = {}

    def recording_backend(request):
        observed["thread_id"] = threading.get_ident()
        return _fake_inference_result()

    controller = InferenceController(backend=recording_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert observed["thread_id"] != main_thread_id


class _ThreadRecordingInferencePage(InferencePage):
    """`_on_finished`이 실제로 실행되는 thread를 기록하는 테스트 전용
    subclass -- monkeypatch로 method를 사후에 갈아끼우면 connect() 시점
    slot 해석에 영향을 줘 검증하려는 thread-affinity 결과 자체가
    바뀔 수 있다(test_training_page.py의 `_ThreadRecordingTrainingPage`
    와 동일한 이유로 진짜 subclass를 쓴다)."""

    def __init__(self, *args, **kwargs) -> None:
        self.observed_finished_thread_ids: list[int] = []
        super().__init__(*args, **kwargs)

    def _on_finished(self, result) -> None:
        self.observed_finished_thread_ids.append(threading.get_ident())
        super()._on_finished(result)


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_finished_handler_runs_on_main_qt_thread_not_worker_thread(tmp_path, qtbot) -> None:
    main_thread_id = threading.get_ident()

    def recording_backend(request):
        assert threading.get_ident() != main_thread_id  # backend 자신은 worker thread에서 돈다
        return _fake_inference_result()

    controller = InferenceController(backend=recording_backend)
    page = _ThreadRecordingInferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert page.observed_finished_thread_ids == [main_thread_id]


# -- success / failure terminal states --------------------------------------------


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_successful_run_shows_finished_and_restores_controls(tmp_path, qtbot) -> None:
    controller = InferenceController(backend=lambda request: _fake_inference_result())
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert page._status_label.text() == "Finished"
    assert controller.state == "finished"
    assert page._run_button.isEnabled() is True
    assert page._model_json_edit.isEnabled() is True
    assert page._device_combo.isEnabled() is True


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_failed_run_shows_concise_error_and_restores_controls(tmp_path, qtbot) -> None:
    def failing_backend(request):
        raise ValueError("bad model")

    controller = InferenceController(backend=failing_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert page._status_label.text() == "Failed: ValueError: bad model"
    assert "Traceback" not in page._status_label.text()  # concise, not the full traceback
    assert controller.state == "failed"
    assert page._run_button.isEnabled() is True
    assert page._model_json_edit.isEnabled() is True


# -- overlap prevention -------------------------------------------------------------


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_overlap_prevented_while_running(tmp_path, qtbot) -> None:
    """`_on_run_clicked()`을 활성 run 중에 직접 다시 호출해도(버튼이
    비활성화돼 있어 UI로는 불가능한 상황을 흉내 냄) backend가 두 번
    실행되면 안 된다."""
    backend_started = threading.Event()
    let_backend_finish = threading.Event()
    call_count = {"n": 0}

    def blocking_backend(request):
        call_count["n"] += 1
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    controller = InferenceController(backend=blocking_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    page._on_run_clicked()
    page._on_run_clicked()

    let_backend_finish.set()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=5000)

    assert call_count["n"] == 1
    assert page._status_label.text() == "Finished"


# -- worker/thread reference cleanup -----------------------------------------------


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_worker_and_thread_references_cleared_after_success(tmp_path, qtbot) -> None:
    controller = InferenceController(backend=lambda request: _fake_inference_result())
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_run_clicked()
    assert page._thread is not None
    assert page._worker is not None

    qtbot.waitUntil(lambda: page._thread is None, timeout=5000)
    assert page._worker is None


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_worker_and_thread_references_cleared_after_failure(tmp_path, qtbot) -> None:
    def failing_backend(request):
        raise RuntimeError("boom")

    controller = InferenceController(backend=failing_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_run_clicked()
    assert page._thread is not None
    assert page._worker is not None

    qtbot.waitUntil(lambda: page._thread is None, timeout=5000)
    assert page._worker is None


# -- rerun after cleanup -------------------------------------------------------------


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_repeated_successful_runs_on_same_page(tmp_path, qtbot) -> None:
    controller = InferenceController(backend=lambda request: _fake_inference_result())
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    for _ in range(3):
        _start_and_wait(page, qtbot)
        assert page._status_label.text() == "Finished"
        assert page._run_button.isEnabled() is True
        assert page._thread is None
        assert page._worker is None


@pytest.mark.phase6c_cp2_inference_page_lifecycle
def test_run_succeeds_after_a_previous_failed_run(tmp_path, qtbot) -> None:
    """§lifecycle: `InferenceController.begin_run()`은 `failed` 상태
    에서도 새 run을 허용한다(별도 reset 없음) -- 그 계약이 GUI
    lifecycle을 통해서도 실제로 성립하는지 확인한다."""
    calls = {"n": 0}

    def flaky_backend(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("first run fails")
        return _fake_inference_result()

    controller = InferenceController(backend=flaky_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)
    assert page._status_label.text().startswith("Failed")

    _start_and_wait(page, qtbot)
    assert page._status_label.text() == "Finished"


# ===============================================================================
# Phase 6C CP3: deterministic inference result presentation
# ===============================================================================


@pytest.mark.phase6c_cp3_inference_results
def test_result_area_shows_placeholders_initially(qtbot) -> None:
    page = _make_page(qtbot)
    assert page._predicted_class_value_label.text() == "--"
    assert page._confidence_value_label.text() == "--"
    assert page._probabilities_value_label.text() == "--"
    assert page._duration_value_label.text() == "--"


@pytest.mark.phase6c_cp3_inference_results
def test_successful_run_displays_predicted_class_exactly(tmp_path, qtbot) -> None:
    result = InferenceResult(
        predicted_index=2,
        predicted_class="golden_retriever",
        confidence=0.5,
        probabilities={"golden_retriever": 0.5, "poodle": 0.5},
        inference_duration_seconds=0.1,
    )
    controller = InferenceController(backend=lambda request: result)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert page._predicted_class_value_label.text() == "golden_retriever"


@pytest.mark.phase6c_cp3_inference_results
def test_successful_run_displays_confidence_in_fixed_format(tmp_path, qtbot) -> None:
    result = InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.8734,
        probabilities={"cat": 0.8734, "dog": 0.1266},
        inference_duration_seconds=0.1,
    )
    controller = InferenceController(backend=lambda request: result)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert page._confidence_value_label.text() == "87.34%"


@pytest.mark.phase6c_cp3_inference_results
def test_successful_run_displays_probabilities_sorted_by_class_name(tmp_path, qtbot) -> None:
    # Insertion order is deliberately not alphabetical -- display must sort.
    result = InferenceResult(
        predicted_index=1,
        predicted_class="zebra",
        confidence=0.6,
        probabilities={"zebra": 0.6, "aardvark": 0.25, "mongoose": 0.15},
        inference_duration_seconds=0.1,
    )
    controller = InferenceController(backend=lambda request: result)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert page._probabilities_value_label.text() == (
        "aardvark: 25.00%\nmongoose: 15.00%\nzebra: 60.00%"
    )


@pytest.mark.phase6c_cp3_inference_results
def test_successful_run_displays_duration_converted_to_milliseconds(tmp_path, qtbot) -> None:
    result = InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1},
        inference_duration_seconds=0.12345,
    )
    controller = InferenceController(backend=lambda request: result)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert page._duration_value_label.text() == "123.45 ms"


@pytest.mark.phase6c_cp3_inference_results
def test_on_finished_presents_result_without_recalculation(qtbot) -> None:
    """`_on_finished` must present the InferenceResult instance it receives
    verbatim -- calling it directly (bypassing the worker/backend entirely)
    must still produce the exact same displayed values."""
    page = _make_page(qtbot)
    result = InferenceResult(
        predicted_index=0,
        predicted_class="direct_call",
        confidence=0.3333,
        probabilities={"b": 0.6667, "a": 0.3333},
        inference_duration_seconds=0.05,
    )

    page._on_finished(result)

    assert page._predicted_class_value_label.text() == "direct_call"
    assert page._confidence_value_label.text() == "33.33%"
    assert page._probabilities_value_label.text() == "a: 33.33%\nb: 66.67%"
    assert page._duration_value_label.text() == "50.00 ms"


@pytest.mark.phase6c_cp3_inference_results
def test_failed_run_clears_previous_successful_result(tmp_path, qtbot) -> None:
    calls = {"n": 0}

    def flaky_backend(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_inference_result()
        raise ValueError("second run fails")

    controller = InferenceController(backend=flaky_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)
    assert page._predicted_class_value_label.text() == "cat"

    _start_and_wait(page, qtbot)

    assert page._status_label.text() == "Failed: ValueError: second run fails"
    assert page._predicted_class_value_label.text() == "--"
    assert page._confidence_value_label.text() == "--"
    assert page._probabilities_value_label.text() == "--"
    assert page._duration_value_label.text() == "--"


@pytest.mark.phase6c_cp3_inference_results
def test_new_run_clears_stale_result_before_showing_running(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    controller = InferenceController(backend=blocking_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_run_clicked()
    assert backend_started.wait(timeout=5)
    let_backend_finish.set()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=5000)
    assert page._predicted_class_value_label.text() == "cat"

    # Second run: result area must already be cleared by the time Running
    # is shown, before the (blocking) backend has produced a new result.
    backend_started.clear()
    let_backend_finish.clear()
    page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    assert page._status_label.text() == "Running"
    assert page._predicted_class_value_label.text() == "--"
    assert page._confidence_value_label.text() == "--"
    assert page._probabilities_value_label.text() == "--"
    assert page._duration_value_label.text() == "--"

    let_backend_finish.set()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=5000)


@pytest.mark.phase6c_cp3_inference_results
def test_new_run_clears_previous_error_presentation(tmp_path, qtbot) -> None:
    calls = {"n": 0}

    def flaky_backend(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("first run fails")
        return _fake_inference_result()

    controller = InferenceController(backend=flaky_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)
    assert page._status_label.text().startswith("Failed")

    _start_and_wait(page, qtbot)

    assert page._status_label.text() == "Finished"
    assert "Failed" not in page._status_label.text()
    assert page._predicted_class_value_label.text() == "cat"


@pytest.mark.phase6c_cp4_main_window_integration
def test_is_inference_active_false_at_construction(qtbot) -> None:
    page = _make_page(qtbot)
    assert page.is_inference_active() is False


@pytest.mark.phase6c_cp4_main_window_integration
def test_is_inference_active_true_while_running(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    controller = InferenceController(backend=blocking_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_run_clicked()
    assert backend_started.wait(timeout=5)
    assert page.is_inference_active() is True

    let_backend_finish.set()
    qtbot.waitUntil(lambda: page.is_inference_active() is False, timeout=5000)


@pytest.mark.phase6c_cp4_main_window_integration
def test_is_inference_active_stays_true_through_cleanup_not_just_terminal_signal(tmp_path, qtbot) -> None:
    """Terminal worker signal(`finished`)이 도착한 시점에는 아직
    thread/worker cleanup(`_on_thread_finished`)이 끝나지 않았을 수
    있다 -- `is_inference_active()`는 그 순간에도 여전히 `True`여야
    한다."""
    observed_active_during_finished = {}

    class _RecordingPage(InferencePage):
        def _on_finished(self, result) -> None:
            observed_active_during_finished["value"] = self.is_inference_active()
            super()._on_finished(result)

    controller = InferenceController(backend=lambda request: _fake_inference_result())
    page = _RecordingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)

    assert observed_active_during_finished["value"] is True
    assert page.is_inference_active() is False


@pytest.mark.phase6c_cp4_main_window_integration
def test_request_close_emits_immediately_when_idle(qtbot) -> None:
    page = _make_page(qtbot)
    received = []
    page.close_requested.connect(lambda: received.append(True))

    page.request_close()

    assert received == [True]


@pytest.mark.phase6c_cp4_main_window_integration
def test_request_close_defers_until_active_run_completes(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    controller = InferenceController(backend=blocking_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    received = []
    page.close_requested.connect(lambda: received.append(True))

    page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    page.request_close()
    assert received == []  # not yet -- inference must finish naturally, no cancellation

    let_backend_finish.set()
    qtbot.waitUntil(lambda: received == [True], timeout=5000)
    assert page._thread is None
    assert page._worker is None


@pytest.mark.phase6c_cp4_main_window_integration
def test_request_close_completes_after_failed_run(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def failing_blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        raise ValueError("boom")

    controller = InferenceController(backend=failing_blocking_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    received = []
    page.close_requested.connect(lambda: received.append(True))

    page._on_run_clicked()
    assert backend_started.wait(timeout=5)
    page.request_close()

    let_backend_finish.set()
    qtbot.waitUntil(lambda: received == [True], timeout=5000)
    assert page._thread is None
    assert page._worker is None


@pytest.mark.phase6c_cp4_main_window_integration
def test_repeated_request_close_does_not_duplicate_emission(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    controller = InferenceController(backend=blocking_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    received = []
    page.close_requested.connect(lambda: received.append(True))

    page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    page.request_close()
    page.request_close()
    page.request_close()

    let_backend_finish.set()
    qtbot.waitUntil(lambda: received == [True], timeout=5000)


@pytest.mark.phase6c_cp3_inference_results
def test_successful_run_removes_previous_error_and_shows_only_new_result(tmp_path, qtbot) -> None:
    calls = {"n": 0}

    def flaky_backend(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return InferenceResult(
            predicted_index=0,
            predicted_class="second_run_class",
            confidence=0.42,
            probabilities={"second_run_class": 0.42, "other": 0.58},
            inference_duration_seconds=0.2,
        )

    controller = InferenceController(backend=flaky_backend)
    page = InferencePage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, qtbot)
    assert page._status_label.text().startswith("Failed")

    _start_and_wait(page, qtbot)

    assert page._status_label.text() == "Finished"
    assert page._predicted_class_value_label.text() == "second_run_class"
    assert page._confidence_value_label.text() == "42.00%"
