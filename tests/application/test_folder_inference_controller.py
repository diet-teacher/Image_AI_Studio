"""`FolderInferenceController` 테스트(Phase 10 CP2). PySide6를 전혀
import하지 않는다 -- 이 계층은 framework-agnostic이다. 실제 folder
inference는 절대 돌리지 않고, 항상 fake folder backend를 주입해
controller의 상태 전이/backend 호출 인자만 검증한다.

`InferenceController`(단일 이미지) 테스트 구조(Phase 6B)를 그대로
참고하되, 이 checkpoint의 핵심 계약을 추가로 고정한다: **per-image
실패가 섞인 aggregate 결과를 받으면 `finished`이고, `failed`는 폴더
연산 자체가 치명적 예외를 던졌을 때만이다.**"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from image_ai_studio.application.folder_inference_controller import (
    FolderInferenceAlreadyRunningError,
    FolderInferenceController,
)
from image_ai_studio.inference.folder_inference import (
    FolderInferenceError,
    FolderInferenceRequest,
    FolderInferenceResult,
    ImageOutcome,
    run_folder_inference,
)
from image_ai_studio.inference.single_image_inference import InferenceResult


# -- helpers ---------------------------------------------------------------


def _infer_result(index: int = 0) -> InferenceResult:
    classes = ("cat", "dog")
    return InferenceResult(
        predicted_index=index,
        predicted_class=classes[index],
        confidence=0.75,
        probabilities={"cat": 0.75, "dog": 0.25},
        inference_duration_seconds=0.001,
    )


def _dummy_request() -> FolderInferenceRequest:
    # controller/backend 계약만 테스트하므로 실제로 존재하는 경로일 필요는
    # 없다 -- fake backend는 request를 열어보지 않고 그대로 캡처만 한다.
    return FolderInferenceRequest(
        model_json_path=Path("model.json"),
        state_dict_path=Path("state_dict.pt"),
        class_mapping_path=Path("class_mapping.json"),
        folder_path=Path("images"),
        device="cpu",
        precision="fp32",
    )


def _all_success_aggregate() -> FolderInferenceResult:
    return FolderInferenceResult(
        items=(
            ImageOutcome(image_path=Path("a.png"), result=_infer_result(0), error=None),
            ImageOutcome(image_path=Path("b.png"), result=_infer_result(1), error=None),
        )
    )


def _mixed_aggregate() -> FolderInferenceResult:
    return FolderInferenceResult(
        items=(
            ImageOutcome(image_path=Path("a.png"), result=_infer_result(0), error=None),
            ImageOutcome(image_path=Path("b.png"), result=None, error="RuntimeError: boom: b.png"),
            ImageOutcome(image_path=Path("c.png"), result=_infer_result(1), error=None),
        )
    )


# -- default backend ----------------------------------------------------


def test_default_backend_is_run_folder_inference() -> None:
    default = inspect.signature(FolderInferenceController.__init__).parameters["backend"].default
    assert default is run_folder_inference


# -- success path -----------------------------------------------------


def test_controller_success_path_state_transitions_and_backend_args() -> None:
    captured: dict = {}
    aggregate = _all_success_aggregate()

    def fake_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        captured["request"] = request
        return aggregate

    controller = FolderInferenceController(backend=fake_backend)
    assert controller.state == "idle"
    assert controller.is_running is False

    controller.begin_run()
    assert controller.state == "running"
    assert controller.is_running is True

    request = _dummy_request()
    result = controller.run(request)

    assert controller.state == "finished"
    assert controller.is_running is False
    assert result is aggregate  # backend가 돌려준 바로 그 객체
    assert captured["request"] is request


def test_controller_run_before_begin_run_raises() -> None:
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    with pytest.raises(RuntimeError, match="begin_run"):
        controller.run(_dummy_request())


# -- key contract: mixed per-image outcomes still finish ---------------


def test_aggregate_with_per_image_failures_transitions_to_finished() -> None:
    """per-image 실패가 섞여 있어도 backend가 aggregate를 정상 반환하면
    controller는 `finished`다 -- `failed`가 아니다."""
    aggregate = _mixed_aggregate()
    controller = FolderInferenceController(backend=lambda request: aggregate)

    controller.begin_run()
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result is aggregate
    assert result.total == 3
    assert result.succeeded == 2
    assert result.failed == 1


def test_aggregate_with_all_images_failed_still_transitions_to_finished() -> None:
    aggregate = FolderInferenceResult(
        items=(
            ImageOutcome(image_path=Path("a.png"), result=None, error="RuntimeError: boom"),
            ImageOutcome(image_path=Path("b.png"), result=None, error="RuntimeError: boom"),
        )
    )
    controller = FolderInferenceController(backend=lambda request: aggregate)

    controller.begin_run()
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result.failed == 2
    assert result.succeeded == 0


# -- key contract: fatal exception transitions to failed --------------


def test_fatal_folder_error_transitions_to_failed_and_reraises() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("no supported images in folder: images")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()

    with pytest.raises(FolderInferenceError, match="no supported images"):
        controller.run(_dummy_request())

    assert controller.state == "failed"
    assert controller.is_running is False


def test_fatal_generic_exception_transitions_to_failed_and_reraises() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise OSError("disk vanished mid discovery")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()

    with pytest.raises(OSError, match="disk vanished"):
        controller.run(_dummy_request())

    assert controller.state == "failed"


# -- single active run -----------------------------------------------


def test_controller_rejects_second_begin_run_while_running() -> None:
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    controller.begin_run()
    with pytest.raises(FolderInferenceAlreadyRunningError):
        controller.begin_run()
    assert controller.state == "running"  # 거부돼도 기존 상태 유지


def test_controller_allows_new_begin_run_after_finished() -> None:
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


def test_controller_allows_new_begin_run_after_failed() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("boom")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()
    with pytest.raises(FolderInferenceError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


# -- run() lifecycle invariant --------------------------------------


def test_controller_rejects_direct_run_after_finished_without_new_begin_run() -> None:
    call_count = {"n": 0}

    def counting_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        call_count["n"] += 1
        return _all_success_aggregate()

    controller = FolderInferenceController(backend=counting_backend)
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"
    assert call_count["n"] == 1

    with pytest.raises(RuntimeError, match="running"):
        controller.run(_dummy_request())

    assert call_count["n"] == 1  # backend가 다시 호출되지 않았어야 한다
    assert controller.state == "finished"  # 거부돼도 상태는 그대로


def test_controller_rejects_direct_run_after_failed_without_new_begin_run() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("boom")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()
    with pytest.raises(FolderInferenceError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    with pytest.raises(RuntimeError, match="running"):
        controller.run(_dummy_request())

    assert controller.state == "failed"


# -- repeated success / fatal-failure runs leave no stale state -------


def test_repeated_alternating_success_and_fatal_runs_reset_cleanly() -> None:
    outcomes: list[str] = []

    def toggling_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        if len(outcomes) % 2 == 0:
            outcomes.append("ok")
            return _mixed_aggregate()
        outcomes.append("fatal")
        raise FolderInferenceError("folder gone")

    controller = FolderInferenceController(backend=toggling_backend)

    for _ in range(2):
        controller.begin_run()
        result = controller.run(_dummy_request())
        assert controller.state == "finished"
        assert result.total == 3

        controller.begin_run()
        with pytest.raises(FolderInferenceError):
            controller.run(_dummy_request())
        assert controller.state == "failed"
        assert controller.is_running is False

    assert outcomes == ["ok", "fatal", "ok", "fatal"]
