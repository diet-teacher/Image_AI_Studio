"""Phase 6B: framework-agnostic application layer for single-image
inference(docs/phase6a_inference_architecture.md §8). **이 모듈은
PySide6/Qt를 절대 import하지 않는다** -- GUI framework 의존은 Phase 6C의
`image_ai_studio.gui.qt_inference_worker`에만 격리한다(Phase 5B의
`application/training_controller.py`와 동일한 경계 원칙).

`TrainingController`와 **의도적으로 다른, 훨씬 단순한** state machine을
쓴다 -- inference는 원자적 단일 forward pass라 `stopping` 상태도
cooperative stop도 없다(Phase 6A §8). `TrainingController`와 공통
base class를 만들지 않는다 -- 두 state machine의 모양 자체가 달라서
억지로 뽑은 추상화가 오히려 읽기 어렵다(YAGNI, Phase 6A §8 결론)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from image_ai_studio.inference.single_image_inference import InferenceRequest, InferenceResult, run_single_image_inference

InferenceBackend = Callable[[InferenceRequest], InferenceResult]
"""`run_single_image_inference()`와 동일한 signature(`request` 하나만
받는 callable). 테스트는 fake backend를 주입해 실제 model/이미지 없이
controller 상태 전이만 검증한다."""

InferenceControllerState = Literal["idle", "running", "finished", "failed"]


class InferenceAlreadyRunningError(RuntimeError):
    """`InferenceController.begin_run()`이 이미 `running` 상태에서
    호출됐을 때 발생한다 -- single active inference run만 지원한다."""


def build_inference_request(
    *,
    model_json_path: str | Path,
    state_dict_path: str | Path,
    class_mapping_path: str | Path,
    image_path: str | Path,
    device: str = "cpu",
    precision: str = "fp32",
) -> InferenceRequest:
    """GUI 입력값(문자열 경로 포함)과 `InferenceRequest` 사이의
    request-builder 경계(Phase 5B의 `build_training_request()`와 동일한
    철학, Phase 6A §15). 문자열 경로를 `Path`로 바꾸는 것 외에는 아무
    값도 검증/가공하지 않는다 -- semantic validation은 전부
    `run_single_image_inference()`(및 그 안에서 재사용하는 기존 training
    validator들)의 책임이다. 이 함수가 실패를 삼키는 경우는 없다."""
    return InferenceRequest(
        model_json_path=Path(model_json_path),
        state_dict_path=Path(state_dict_path),
        class_mapping_path=Path(class_mapping_path),
        image_path=Path(image_path),
        device=device,
        precision=precision,
    )


class InferenceController:
    """single active inference run의 application-level lifecycle을
    관리한다. `TrainingController`처럼 이 클래스 자신도 어떤 thread에서도
    실행될 수 있는 순수 Python 객체다 -- `begin_run()`은 caller의 현재
    thread에서 상태만 동기적으로 바꾸고(빠름), `run()`은 caller가 고른
    thread에서 실제 backend 호출을 블로킹으로 수행한다(보통 worker
    thread). cooperative stop이 없으므로 `threading.Event`도 갖지
    않는다 -- `TrainingController`와 다른 부분은 이것뿐이다."""

    def __init__(self, backend: InferenceBackend = run_single_image_inference) -> None:
        self._backend = backend
        self._state: InferenceControllerState = "idle"

    @property
    def state(self) -> InferenceControllerState:
        return self._state

    @property
    def is_running(self) -> bool:
        """`running`이면 True -- single active inference run 여부를
        판단하는 유일한 기준이다."""
        return self._state == "running"

    def begin_run(self) -> None:
        """새 inference를 시작하기 직전 호출한다(Qt worker가 실제 backend
        호출을 시작하기 전). 이미 `running`이면
        `InferenceAlreadyRunningError`를 던지고 상태를 바꾸지 않는다.
        `idle`/`finished`/`failed` 어디서든 새 run을 시작할 수 있다 --
        별도의 "reset" 단계는 없다(`TrainingController.begin_run()`과
        동일한 계약)."""
        if self.is_running:
            raise InferenceAlreadyRunningError(
                f"cannot start a new inference run while controller state is {self._state!r}"
            )
        self._state = "running"

    def run(self, request: InferenceRequest) -> InferenceResult:
        """`begin_run()` 이후 실제 backend를 블로킹으로 호출한다. 성공하면
        `state`를 `finished`로, backend가 예외를 던지면 `state`를
        `failed`로 바꾸고 그 예외를 그대로 다시 던진다(swallow하지
        않음).

        `state`가 `running`일 때만 호출할 수 있다 -- `begin_run()`을
        거치지 않았거나(`idle`), 이미 이전 run이 끝난 뒤(`finished`/
        `failed`) 새 `begin_run()` 없이 다시 호출하면 `RuntimeError`를
        던진다."""
        if self._state != "running":
            raise RuntimeError(
                "InferenceController.run() requires state 'running' "
                f"(call begin_run() first) -- got state={self._state!r}"
            )
        try:
            result = self._backend(request)
        except BaseException:
            self._state = "failed"
            raise
        self._state = "finished"
        return result
