"""`TrainingPage` wiring 테스트(Phase 5C). fake backend를 주입한
`TrainingController`로 실제 학습 없이: 초기 state, widget→request 매핑,
Start/Stop lifecycle, progress/finished/failed 표시, 반복 실행만
검증한다. 실제 CPU 학습 1개는 `test_training_page_integration.py`가
따로 담당한다."""
from __future__ import annotations

import threading

import pytest

from image_ai_studio.application.training_controller import TrainingController
from image_ai_studio.gui.training_page import TrainingPage, parse_class_weights
from image_ai_studio.training.imagefolder_workflow import ImageFolderWorkflowResult
from image_ai_studio.training.loop import TrainingHistory, TrainingProgress


def _fake_result(stop_reason: str = "completed") -> ImageFolderWorkflowResult:
    history = TrainingHistory(
        train_losses=[0.5], val_losses=[0.4], val_accuracies=[0.9],
        best_epoch=1, best_val_loss=0.4,
    )
    return ImageFolderWorkflowResult(
        history=history,
        test_loss=0.3,
        test_accuracy=0.95,
        best_model_state_dict_path="best.pt",
        training_history_path="history.json",
        class_mapping_path="class_mapping.json",
        test_result_path="test_result.json",
        checkpoint_path=None,
        checkpoint_metadata_path=None,
        torchscript_model_path=None,
        torchscript_metadata_path=None,
        stop_reason=stop_reason,
    )


def _make_progress(*, run_epoch: int, total_run_epochs: int, global_epoch: int) -> TrainingProgress:
    return TrainingProgress(
        run_epoch=run_epoch, total_run_epochs=total_run_epochs, global_epoch=global_epoch,
        train_loss=1.23, val_loss=1.45, val_accuracy=0.5, learning_rate=1e-2,
        best_epoch=1, best_val_loss=1.45, epochs_without_improvement=0,
        stopped_early=False, epoch_duration_seconds=0.5,
    )


def _start_and_wait(page: TrainingPage, controller: TrainingController, qtbot, timeout: int = 5000) -> None:
    """`page._on_start_clicked()`을 호출한 뒤 `_on_finished`/`_on_failed`
    가 실제로 GUI를 다 갱신할 때까지 polling으로 기다린다.

    `qtbot.waitSignal(page._worker.finished, ...)`처럼 signal 자체가
    fire하는 시점만 기다리는 방식은 두 가지 race가 있다: (1) fake
    backend가 즉시 반환하면 `_on_start_clicked()`가 끝나기도 전에
    worker thread가 이미 끝나버려 signal을 놓칠 수 있고, (2) 설령
    signal을 놓치지 않더라도, `worker.finished`에는 `page._on_finished`
    (GUI 갱신)와 `page._thread.quit`가 **각각 별도의 queued connection**
    으로 연결돼 있어, `qtbot.waitSignal()`이 감지하는 시점과
    `page._on_finished`이 실제로 GUI를 다 갱신한 시점이 반드시 같지
    않다(실측: `waitSignal` 반환 직후에도 `_start_button`이 아직
    비활성 상태인 경우를 재현함).

    `_finish_common()`(=`_on_finished`/`_on_failed`의 마지막 단계)이
    끝나야만 `_start_button`이 다시 활성화되므로, 이 조건으로 polling
    하면 GUI가 실제로 다 갱신된 뒤에만 반환된다(`waitUntil`은 조건이
    참이 될 때까지 Qt 이벤트를 계속 처리하므로 안전하다)."""
    page._on_start_clicked()
    qtbot.waitUntil(lambda: page._start_button.isEnabled(), timeout=timeout)


def _wait_for_thread_cleanup(page: TrainingPage, qtbot, timeout: int = 5000) -> None:
    """`page._thread`가 실제로 멈추거나(`isRunning() is False`) 이미
    `deleteLater()`로 해제되어 C++ 쪽 객체에 더 이상 접근할 수 없는
    상태(`RuntimeError`)가 될 때까지 기다린다 -- Start 버튼 재활성화
    시점(`_start_and_wait`)과 실제 QThread teardown 완료 시점은 서로
    다른 queued event이므로 별도로 확인해야 한다."""

    def _cleaned_up() -> bool:
        try:
            return page._thread is None or page._thread.isRunning() is False
        except RuntimeError:
            return True

    qtbot.waitUntil(_cleaned_up, timeout=timeout)


def _fill_minimum_valid_fields(page: TrainingPage, tmp_path) -> None:
    (tmp_path / "model.json").write_text("{}")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "out").mkdir()
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._dataset_root_edit.setText(str(tmp_path / "dataset"))
    page._output_dir_edit.setText(str(tmp_path / "out"))


# -- parse_class_weights() -----------------------------------------------------


def test_parse_class_weights_empty_is_none() -> None:
    assert parse_class_weights("") is None
    assert parse_class_weights("   ") is None


def test_parse_class_weights_parses_comma_separated_floats() -> None:
    assert parse_class_weights("1.0, 2.0, 1.5") == (1.0, 2.0, 1.5)


def test_parse_class_weights_invalid_number_raises() -> None:
    with pytest.raises(ValueError, match="invalid class weights"):
        parse_class_weights("1.0, notanumber")


# -- initial state --------------------------------------------------------------


def test_initial_state(qtbot) -> None:
    page = TrainingPage(controller=TrainingController(backend=lambda *a, **k: _fake_result()))
    qtbot.addWidget(page)

    assert page._status_label.text() == "Idle"
    assert page._start_button.isEnabled() is True
    assert page._stop_button.isEnabled() is False
    assert page._scheduler_factor_spin.isEnabled() is False  # scheduler 기본 None
    assert page._gradient_clip_spin.isEnabled() is False  # optional 기본 비활성
    assert page._export_torchscript_check.isChecked() is True  # backend 기본값과 일치


# -- widget -> request mapping ---------------------------------------------------


def test_basic_fields_map_to_request(tmp_path, qtbot) -> None:
    page = TrainingPage(controller=TrainingController(backend=lambda *a, **k: _fake_result()))
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)
    page._epochs_spin.setValue(3)
    page._batch_size_spin.setValue(16)
    page._learning_rate_spin.setValue(0.0005)
    page._optimizer_combo.setCurrentText("sgd")
    page._device_combo.setCurrentText("cpu")
    page._precision_combo.setCurrentText("fp32")

    request = page._build_request()

    assert request.training_config.epochs == 3
    assert request.training_config.batch_size == 16
    assert request.training_config.learning_rate == pytest.approx(0.0005)
    assert request.training_config.optimizer == "sgd"
    assert request.device == "cpu"
    assert request.training_config.precision == "fp32"


def test_cuda_device_selection_maps_to_request(tmp_path, qtbot) -> None:
    """Phase 5C §56: QThread+CUDA wiring 자체는 이미 Phase 5B
    (`test_qt_training_worker_integration.py`)가 검증했으므로 여기서는
    fake backend로 GUI의 device 선택이 request에 그대로 전달되는지만
    확인한다(전체 CUDA GUI 통합 반복 불필요)."""
    page = TrainingPage(controller=TrainingController(backend=lambda *a, **k: _fake_result()))
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    if page._device_combo.findText("cuda") < 0:
        pytest.skip("no CUDA device detected on this machine")

    page._device_combo.setCurrentText("cuda")

    request = page._build_request()

    assert request.device == "cuda"


def test_advanced_fields_map_to_request(tmp_path, qtbot) -> None:
    page = TrainingPage(controller=TrainingController(backend=lambda *a, **k: _fake_result()))
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._gradient_clip_check.setChecked(True)
    page._gradient_clip_spin.setValue(2.5)
    page._early_stopping_check.setChecked(True)
    page._early_stopping_spin.setValue(4)
    page._checkpoint_every_check.setChecked(True)
    page._checkpoint_every_spin.setValue(2)
    page._class_weights_edit.setText("1.0, 3.0")
    page._scheduler_combo.setCurrentText("plateau")
    page._scheduler_factor_spin.setValue(0.5)
    page._scheduler_patience_spin.setValue(2)
    page._pin_memory_check.setChecked(True)
    page._non_blocking_check.setChecked(True)

    request = page._build_request()

    assert request.training_config.gradient_clip_norm == pytest.approx(2.5)
    assert request.training_config.early_stopping_patience == 4
    assert request.checkpoint_every == 2
    assert request.training_config.class_weights == (1.0, 3.0)
    assert request.training_config.lr_scheduler == "plateau"
    assert request.training_config.lr_scheduler_factor == pytest.approx(0.5)
    assert request.training_config.lr_scheduler_patience == 2
    assert request.pin_memory is True
    assert request.non_blocking is True


def test_optional_fields_disabled_map_to_none(tmp_path, qtbot) -> None:
    """checkbox가 꺼져 있으면 spin box 값이 남아 있어도 None으로
    나가야 한다(값을 0 같은 sentinel로 바꾸지 않는다는 계약)."""
    page = TrainingPage(controller=TrainingController(backend=lambda *a, **k: _fake_result()))
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)
    page._gradient_clip_spin.setValue(9.9)  # 값은 채워져 있지만 checkbox는 꺼져 있음

    request = page._build_request()

    assert request.training_config.gradient_clip_norm is None
    assert request.training_config.early_stopping_patience is None
    assert request.checkpoint_every is None
    assert request.resume_from is None
    assert request.checkpoint_out is None


def test_request_construction_failure_shows_gui_failure_without_starting_controller(
    tmp_path, qtbot
) -> None:
    """§"request 조립 실패": epochs=0처럼 TrainingConfig가 거부하는
    값이면 controller.begin_run()이 아예 호출되지 않아야 한다(GUI-level
    실패와 controller state를 혼동하지 않는다)."""
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)
    page._epochs_spin.setValue(0)  # QSpinBox 최솟값이 1이므로 강제로 우회
    page._epochs_spin.setMinimum(0)
    page._epochs_spin.setValue(0)

    page._on_start_clicked()

    assert controller.state == "idle"  # begin_run()이 호출되지 않았다
    assert page._status_label.text() == "Failed"
    assert page._error_summary_label.text() != ""


# -- Start/Running lifecycle ------------------------------------------------------


def test_start_disables_controls_and_enables_stop(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request, *, progress_callback=None, should_stop=None):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_result()

    controller = TrainingController(backend=blocking_backend)
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_start_clicked()
    assert backend_started.wait(timeout=5)

    assert page._start_button.isEnabled() is False
    assert page._model_json_edit.isEnabled() is False
    assert page._stop_button.isEnabled() is True
    assert page._status_label.text() == "Running..."

    let_backend_finish.set()
    qtbot.waitUntil(lambda: page._start_button.isEnabled(), timeout=5000)

    assert page._start_button.isEnabled() is True
    assert page._stop_button.isEnabled() is False


# -- progress display ----------------------------------------------------------


def test_progress_bar_uses_run_epoch_not_global_epoch(tmp_path, qtbot) -> None:
    """progress bar semantics: resume 상황(global_epoch=4, run_epoch=1,
    total_run_epochs=2)에서도 progress bar는 1/2를 보여줘야 한다 --
    4/2처럼 잘못된 비율을 보여주면 안 된다."""

    def one_progress_backend(request, *, progress_callback=None, should_stop=None):
        progress_callback(_make_progress(run_epoch=1, total_run_epochs=2, global_epoch=4))
        return _fake_result()

    controller = TrainingController(backend=one_progress_backend)
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, controller, qtbot)

    assert page._progress_bar.maximum() == 2
    assert page._progress_bar.value() == 1
    assert page._global_epoch_label.text() == "4"
    assert page._train_loss_label.text() == "1.2300"


class _ThreadRecordingTrainingPage(TrainingPage):
    """`_on_progress`가 어느 thread에서 실제로 실행되는지 기록하는
    테스트 전용 subclass. `monkeypatch.setattr(TrainingPage, ...)`로
    class attribute를 사후에 갈아끼우는 방식은 Qt의 connect() 시점
    slot 해석에 영향을 줘 검증하려는 바로 그 thread-affinity 결과를
    바꿔버리는 것으로 실험 확인됐다 -- 실제 production과 동일하게
    class 정의 시점부터 override된 진짜 subclass를 써야 한다."""

    def __init__(self, *args, **kwargs) -> None:
        self.observed_progress_thread_ids: list[int] = []
        super().__init__(*args, **kwargs)

    def _on_progress(self, progress) -> None:
        self.observed_progress_thread_ids.append(threading.get_ident())
        super()._on_progress(progress)


def test_progress_handler_runs_on_main_qt_thread_not_worker_thread(tmp_path, qtbot) -> None:
    """Phase 5B에서 empirical하게 확인한 signal thread-affinity 함정
    (plain 함수/lambda에 connect하면 emit이 일어난 worker thread에서
    직접 실행됨, docs/phase5b_..._design.md §9)을 Phase 5C GUI에서도
    실제로 지키고 있는지 고정한다 -- `TrainingPage._on_progress`(실제
    QObject bound method)가 항상 main/GUI thread에서 실행돼야 한다."""
    main_thread_id = threading.get_ident()

    def recording_backend(request, *, progress_callback=None, should_stop=None):
        assert threading.get_ident() != main_thread_id  # backend 자신은 worker thread에서 돈다
        progress_callback(_make_progress(run_epoch=1, total_run_epochs=1, global_epoch=1))
        return _fake_result()

    controller = TrainingController(backend=recording_backend)
    page = _ThreadRecordingTrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, controller, qtbot)

    assert page.observed_progress_thread_ids == [main_thread_id]


# -- Stop lifecycle --------------------------------------------------------------


def test_stop_button_requests_cooperative_stop_and_shows_stopping(tmp_path, qtbot) -> None:
    backend_started = threading.Event()
    observed = {}

    def stoppable_backend(request, *, progress_callback=None, should_stop=None):
        backend_started.set()
        deadline = threading.Event()
        deadline.wait(timeout=0.2)  # cooperative stop이 반영될 짧은 여유
        observed["should_stop_value"] = should_stop()
        return _fake_result(stop_reason="user_stopped")

    controller = TrainingController(backend=stoppable_backend)
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    page._on_start_clicked()
    assert backend_started.wait(timeout=5)

    page._on_stop_clicked()
    assert page._status_label.text() == "Stopping after current epoch..."
    assert page._stop_button.isEnabled() is False

    qtbot.waitUntil(lambda: page._start_button.isEnabled(), timeout=5000)

    assert observed["should_stop_value"] is True
    assert page._status_label.text() == "Training stopped by user"


# -- finished stop_reason mapping ------------------------------------------------


@pytest.mark.parametrize(
    ("stop_reason", "expected_text"),
    [
        ("completed", "Completed"),
        ("early_stopped", "Training stopped by early stopping"),
        ("user_stopped", "Training stopped by user"),
    ],
)
def test_finished_status_text_matches_stop_reason(stop_reason, expected_text, tmp_path, qtbot) -> None:
    controller = TrainingController(backend=lambda *a, **k: _fake_result(stop_reason=stop_reason))
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, controller, qtbot)

    assert page._status_label.text() == expected_text
    assert page._start_button.isEnabled() is True
    assert page._stop_button.isEnabled() is False


# -- failed handler ---------------------------------------------------------------


def test_failed_backend_shows_error_and_restores_controls(tmp_path, qtbot) -> None:
    def failing_backend(request, *, progress_callback=None, should_stop=None):
        raise ValueError("bad dataset")

    controller = TrainingController(backend=failing_backend)
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    _start_and_wait(page, controller, qtbot)

    assert page._status_label.text() == "Failed"
    assert "ValueError" in page._error_summary_label.text()
    assert "bad dataset" in page._error_summary_label.text()
    assert "Traceback" in page._details_text.toPlainText()
    assert page._start_button.isEnabled() is True
    assert page._stop_button.isEnabled() is False
    assert page._model_json_edit.isEnabled() is True


# -- repeated runs ------------------------------------------------------------------


def test_repeated_runs_on_same_page(tmp_path, qtbot) -> None:
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    for _ in range(2):
        _start_and_wait(page, controller, qtbot)
        assert page._status_label.text() == "Completed"
        assert page._start_button.isEnabled() is True


def test_repeated_run_thread_lifecycle_stress(tmp_path, qtbot) -> None:
    """Phase 5C stabilization: 매 run마다 새 QThread+QtTrainingWorker를
    만들고 버리는 흐름을 짧은 시간 안에 여러 번 반복해 worker/thread
    deleteLater ordering 문제(관찰된 드문 native abort의 원인 후보)를
    누적 노출시킨다. fake backend가 거의 즉시 반환하므로 thread
    생성-실행-quit-deleteLater 주기가 촘촘하게 몰린다 -- 실제 학습을
    반복하지 않으므로 느리지 않다."""
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    page = TrainingPage(controller=controller)
    qtbot.addWidget(page)
    _fill_minimum_valid_fields(page, tmp_path)

    for _ in range(8):
        _start_and_wait(page, controller, qtbot)
        assert page._status_label.text() == "Completed"
        assert page._start_button.isEnabled() is True

    # Start 버튼 재활성화(위 loop)는 이전 QThread의 실제 teardown과는
    # 별개의 queued event다 -- 마지막 run의 thread가 실제로 멈추거나
    # 정리됐는지까지 확인해야 이 stress test가 deleteLater ordering을
    # 끝까지 검증한 것이 된다.
    _wait_for_thread_cleanup(page, qtbot)
