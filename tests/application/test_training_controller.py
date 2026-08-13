"""`TrainingController`/`build_training_request()` 테스트(Phase 5B).
PySide6를 전혀 import하지 않는다 -- 이 계층은 framework-agnostic이다.
실제 학습은 절대 돌리지 않고, 항상 fake backend를 주입해 controller의
상태 전이/backend 호출 인자/stop 전달만 검증한다(docs/
phase5b_application_qt_worker_integration_design.md §15 tests 참고)."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from image_ai_studio.application.training_controller import (
    TrainingAlreadyRunningError,
    TrainingController,
    build_training_request,
)
from image_ai_studio.training.imagefolder_workflow import (
    ImageFolderWorkflowRequest,
    ImageFolderWorkflowResult,
)
from image_ai_studio.training.loop import TrainingHistory


def _fake_result(stop_reason: str = "completed") -> ImageFolderWorkflowResult:
    history = TrainingHistory(
        train_losses=[0.5], val_losses=[0.5], val_accuracies=[0.5],
        best_epoch=1, best_val_loss=0.5,
    )
    return ImageFolderWorkflowResult(
        history=history,
        test_loss=0.5,
        test_accuracy=0.5,
        best_model_state_dict_path=Path("best.pt"),
        training_history_path=Path("history.json"),
        class_mapping_path=Path("class_mapping.json"),
        test_result_path=Path("test_result.json"),
        checkpoint_path=None,
        checkpoint_metadata_path=None,
        torchscript_model_path=None,
        torchscript_metadata_path=None,
        stop_reason=stop_reason,
    )


def _dummy_request() -> ImageFolderWorkflowRequest:
    # controller/backend 계약만 테스트하므로 실제로 존재하는 경로일 필요는
    # 없다 -- fake backend는 request를 열어보지 않고 그대로 캡처만 한다.
    from image_ai_studio.training.config import TrainingConfig

    return ImageFolderWorkflowRequest(
        model_json_path=Path("model.json"),
        dataset_root=Path("dataset"),
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=Path("out"),
    )


# -- build_training_request() ------------------------------------------------


def test_build_training_request_converts_string_paths_to_path() -> None:
    request = build_training_request(
        model_json_path="model.json",
        dataset_root="dataset",
        output_dir="out",
        epochs=3,
        batch_size=8,
        learning_rate=1e-3,
    )
    assert isinstance(request.model_json_path, Path)
    assert isinstance(request.dataset_root, Path)
    assert isinstance(request.output_dir, Path)
    assert request.training_config.epochs == 3
    assert request.resume_from is None
    assert request.checkpoint_out is None
    # 명시하지 않은 나머지는 기존 dataclass 기본값과 동일해야 한다(§9 회귀).
    assert request.device == "cpu"
    assert request.pin_memory is False
    assert request.non_blocking is False
    assert request.export_torchscript is True


def test_build_training_request_forwards_optional_paths() -> None:
    request = build_training_request(
        model_json_path="model.json",
        dataset_root="dataset",
        output_dir="out",
        epochs=1,
        batch_size=4,
        learning_rate=1e-2,
        resume_from="ckpt.pt",
        checkpoint_out="ckpt_out.pt",
        device="cuda",
        precision="fp16",
        pin_memory=True,
        non_blocking=True,
    )
    assert request.resume_from == Path("ckpt.pt")
    assert request.checkpoint_out == Path("ckpt_out.pt")
    assert request.device == "cuda"
    assert request.training_config.precision == "fp16"
    assert request.pin_memory is True
    assert request.non_blocking is True


def test_build_training_request_does_not_swallow_config_validation_errors() -> None:
    """semantic validation은 TrainingConfig 자신의 책임이다 -- request
    builder가 이를 가로채거나 완화하지 않는다(§7)."""
    with pytest.raises(ValueError):
        build_training_request(
            model_json_path="model.json",
            dataset_root="dataset",
            output_dir="out",
            epochs=0,  # TrainingConfig가 거부해야 함
            batch_size=4,
            learning_rate=1e-2,
        )


# -- TrainingController: success path -----------------------------------------


def test_controller_success_path_state_transitions_and_backend_args() -> None:
    captured: dict = {}

    def fake_backend(request, *, progress_callback=None, should_stop=None):
        captured["request"] = request
        captured["progress_callback"] = progress_callback
        captured["should_stop"] = should_stop
        return _fake_result()

    controller = TrainingController(backend=fake_backend)
    assert controller.state == "idle"
    assert controller.is_running is False

    controller.begin_run()
    assert controller.state == "running"
    assert controller.is_running is True

    request = _dummy_request()
    progresses: list = []
    result = controller.run(request, progress_callback=progresses.append)

    assert controller.state == "finished"
    assert controller.is_running is False
    assert result.stop_reason == "completed"
    assert captured["request"] is request
    # bound method는 매 attribute 접근마다 새 wrapper 객체가 생기므로(CPython)
    # `is`가 아니라 `==`(같은 __self__/__func__)로 비교한다.
    assert captured["progress_callback"] == progresses.append
    assert callable(captured["should_stop"])
    assert captured["should_stop"]() is False  # stop 요청 없었으므로 False


def test_controller_run_before_begin_run_raises() -> None:
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    with pytest.raises(RuntimeError, match="begin_run"):
        controller.run(_dummy_request())


# -- TrainingController: single active run ------------------------------------


def test_controller_rejects_second_begin_run_while_running() -> None:
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    controller.begin_run()
    with pytest.raises(TrainingAlreadyRunningError):
        controller.begin_run()
    assert controller.state == "running"  # 거부돼도 기존 상태 유지


def test_controller_allows_new_begin_run_after_finished() -> None:
    """Finished/Failed에서 곧바로 새 run을 시작할 수 있다 -- 별도
    reset 단계가 없다(docs/phase5b_application_qt_worker_integration_design.md
    §6 state model 참고)."""
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


def test_controller_allows_new_begin_run_after_failed() -> None:
    def failing_backend(request, *, progress_callback=None, should_stop=None):
        raise RuntimeError("boom")

    controller = TrainingController(backend=failing_backend)
    controller.begin_run()
    with pytest.raises(RuntimeError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


# -- TrainingController: run() lifecycle invariant -----------------------------


def test_controller_rejects_direct_run_after_finished_without_new_begin_run() -> None:
    """run()이 한 번 끝나 state가 finished가 된 뒤, 새 begin_run() 없이
    run()을 다시 호출하면 거부돼야 한다 -- backend가 두 번째로 호출되지
    않았다는 것까지 확인한다."""
    call_count = {"n": 0}

    def counting_backend(request, *, progress_callback=None, should_stop=None):
        call_count["n"] += 1
        return _fake_result()

    controller = TrainingController(backend=counting_backend)
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"
    assert call_count["n"] == 1

    with pytest.raises(RuntimeError, match="running.*stopping"):
        controller.run(_dummy_request())

    assert call_count["n"] == 1  # backend가 다시 호출되지 않았어야 한다
    assert controller.state == "finished"  # 거부돼도 상태는 그대로


def test_controller_rejects_direct_run_after_failed_without_new_begin_run() -> None:
    def failing_backend(request, *, progress_callback=None, should_stop=None):
        raise RuntimeError("boom")

    controller = TrainingController(backend=failing_backend)
    controller.begin_run()
    with pytest.raises(RuntimeError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    with pytest.raises(RuntimeError, match="running.*stopping"):
        controller.run(_dummy_request())

    assert controller.state == "failed"


# -- TrainingController: cooperative stop -------------------------------------


def test_controller_request_stop_sets_event_observed_by_backend() -> None:
    observed_should_stop = {}

    def fake_backend(request, *, progress_callback=None, should_stop=None):
        # 실제 run_training()처럼, backend는 매 epoch should_stop()을 평가한다.
        observed_should_stop["before"] = should_stop()
        return _fake_result(stop_reason="user_stopped")

    controller = TrainingController(backend=fake_backend)
    controller.begin_run()
    controller.request_stop()
    assert controller.state == "stopping"

    result = controller.run(_dummy_request())

    assert observed_should_stop["before"] is True  # stop 요청이 should_stop에 반영됨
    assert result.stop_reason == "user_stopped"
    # 최종 상태는 "stopping"이 아니라 "finished"다 -- 실제 종료는 backend
    # 반환 시점에 결정된다(중단 요청 자체가 곧바로 종료를 의미하지 않음).
    assert controller.state == "finished"


def test_controller_request_stop_is_noop_when_not_running() -> None:
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    controller.request_stop()  # idle 상태 -- 예외 없이 아무 일도 안 해야 함
    assert controller.state == "idle"


def test_controller_request_stop_is_noop_after_finished() -> None:
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"

    controller.request_stop()  # 이미 끝난 뒤 -- 예외 없이 아무 일도 안 해야 함
    assert controller.state == "finished"


def test_controller_stop_event_is_real_threading_event() -> None:
    """future Qt worker가 GUI thread에서 request_stop()을 호출하고
    worker thread가 동시에 backend를 실행하는 상황을 흉내낸다."""
    stop_event_seen: list[threading.Event] = []

    def fake_backend(request, *, progress_callback=None, should_stop=None):
        # should_stop이 실제로 threading.Event.is_set에 바인딩된
        # bound method인지 확인(간접적으로 실제 Event 객체 사용을 증명).
        assert should_stop.__self__.__class__ is threading.Event
        stop_event_seen.append(should_stop.__self__)
        return _fake_result()

    controller = TrainingController(backend=fake_backend)
    controller.begin_run()
    controller.run(_dummy_request())
    assert stop_event_seen[0].is_set() is False
