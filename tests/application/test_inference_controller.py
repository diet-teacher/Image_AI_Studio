"""`InferenceController`/`build_inference_request()` 테스트(Phase 6B).
PySide6를 전혀 import하지 않는다 -- 이 계층은 framework-agnostic이다.
실제 inference는 절대 돌리지 않고, 항상 fake backend를 주입해
controller의 상태 전이/backend 호출 인자만 검증한다(docs/
phase6a_inference_architecture.md §8 -- cooperative stop이 없는,
`TrainingController`보다 훨씬 단순한 state machine이므로 stop 관련
테스트는 없다)."""
from __future__ import annotations

from pathlib import Path

import pytest

from image_ai_studio.application.inference_controller import (
    InferenceAlreadyRunningError,
    InferenceController,
    build_inference_request,
)
from image_ai_studio.inference.single_image_inference import InferenceRequest, InferenceResult


def _fake_result() -> InferenceResult:
    return InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1},
        inference_duration_seconds=0.001,
    )


def _dummy_request() -> InferenceRequest:
    # controller/backend 계약만 테스트하므로 실제로 존재하는 경로일 필요는
    # 없다 -- fake backend는 request를 열어보지 않고 그대로 캡처만 한다.
    return InferenceRequest(
        model_json_path=Path("model.json"),
        state_dict_path=Path("state_dict.pt"),
        class_mapping_path=Path("class_mapping.json"),
        image_path=Path("image.png"),
        device="cpu",
        precision="fp32",
    )


# -- build_inference_request() ------------------------------------------------


def test_build_inference_request_converts_string_paths_to_path() -> None:
    request = build_inference_request(
        model_json_path="model.json",
        state_dict_path="state_dict.pt",
        class_mapping_path="class_mapping.json",
        image_path="image.png",
    )
    assert isinstance(request.model_json_path, Path)
    assert isinstance(request.state_dict_path, Path)
    assert isinstance(request.class_mapping_path, Path)
    assert isinstance(request.image_path, Path)
    # 명시하지 않은 나머지는 기존 default와 동일해야 한다.
    assert request.device == "cpu"
    assert request.precision == "fp32"


def test_build_inference_request_forwards_device_and_precision() -> None:
    request = build_inference_request(
        model_json_path="model.json",
        state_dict_path="state_dict.pt",
        class_mapping_path="class_mapping.json",
        image_path="image.png",
        device="cuda",
        precision="fp16",
    )
    assert request.device == "cuda"
    assert request.precision == "fp16"


def test_build_inference_request_does_not_swallow_downstream_validation_errors() -> None:
    """semantic validation은 이 builder의 책임이 아니다 -- 이 함수는
    타입 변환만 하고 device/precision 값 자체를 검증하지 않는다(그건
    run_single_image_inference()의 책임, Phase 6A §15)."""
    request = build_inference_request(
        model_json_path="model.json",
        state_dict_path="state_dict.pt",
        class_mapping_path="class_mapping.json",
        image_path="image.png",
        device="not-a-real-device",  # builder는 이 값을 그대로 통과시켜야 함
        precision="fp32",
    )
    assert request.device == "not-a-real-device"


# -- InferenceController: success path -----------------------------------------


def test_controller_success_path_state_transitions_and_backend_args() -> None:
    captured: dict = {}

    def fake_backend(request: InferenceRequest) -> InferenceResult:
        captured["request"] = request
        return _fake_result()

    controller = InferenceController(backend=fake_backend)
    assert controller.state == "idle"
    assert controller.is_running is False

    controller.begin_run()
    assert controller.state == "running"
    assert controller.is_running is True

    request = _dummy_request()
    result = controller.run(request)

    assert controller.state == "finished"
    assert controller.is_running is False
    assert result.predicted_class == "cat"
    assert captured["request"] is request


def test_controller_run_before_begin_run_raises() -> None:
    controller = InferenceController(backend=lambda request: _fake_result())
    with pytest.raises(RuntimeError, match="begin_run"):
        controller.run(_dummy_request())


# -- InferenceController: single active run ------------------------------------


def test_controller_rejects_second_begin_run_while_running() -> None:
    """`is_running`이 True인 동안(=`state == "running"`) 두 번째
    begin_run()은 거부된다 -- `TrainingController`의 `stopping`
    상태는 이 controller에 없으므로 `running` 하나만 확인하면 된다."""
    controller = InferenceController(backend=lambda request: _fake_result())
    controller.begin_run()
    with pytest.raises(InferenceAlreadyRunningError):
        controller.begin_run()
    assert controller.state == "running"  # 거부돼도 기존 상태 유지


def test_controller_allows_new_begin_run_after_finished() -> None:
    """Finished에서 곧바로 새 run을 시작할 수 있다 -- 별도 reset 단계가
    없다(TrainingController와 동일한 계약, Phase 6A §8)."""
    controller = InferenceController(backend=lambda request: _fake_result())
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


def test_controller_allows_new_begin_run_after_failed() -> None:
    def failing_backend(request: InferenceRequest) -> InferenceResult:
        raise RuntimeError("boom")

    controller = InferenceController(backend=failing_backend)
    controller.begin_run()
    with pytest.raises(RuntimeError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


# -- InferenceController: run() lifecycle invariant -----------------------------


def test_controller_rejects_direct_run_after_finished_without_new_begin_run() -> None:
    """run()이 한 번 끝나 state가 finished가 된 뒤, 새 begin_run() 없이
    run()을 다시 호출하면 거부돼야 한다 -- backend가 두 번째로 호출되지
    않았다는 것까지 확인한다."""
    call_count = {"n": 0}

    def counting_backend(request: InferenceRequest) -> InferenceResult:
        call_count["n"] += 1
        return _fake_result()

    controller = InferenceController(backend=counting_backend)
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"
    assert call_count["n"] == 1

    with pytest.raises(RuntimeError, match="running"):
        controller.run(_dummy_request())

    assert call_count["n"] == 1  # backend가 다시 호출되지 않았어야 한다
    assert controller.state == "finished"  # 거부돼도 상태는 그대로


def test_controller_rejects_direct_run_after_failed_without_new_begin_run() -> None:
    def failing_backend(request: InferenceRequest) -> InferenceResult:
        raise RuntimeError("boom")

    controller = InferenceController(backend=failing_backend)
    controller.begin_run()
    with pytest.raises(RuntimeError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    with pytest.raises(RuntimeError, match="running"):
        controller.run(_dummy_request())

    assert controller.state == "failed"
