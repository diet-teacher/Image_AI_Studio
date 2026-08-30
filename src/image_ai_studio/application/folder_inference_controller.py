"""Phase 10 CP2: framework-agnostic application layer for one folder
inference run. **이 모듈은 PySide6/Qt를 절대 import하지 않는다** -- GUI
framework 의존은 `image_ai_studio.gui.qt_folder_inference_worker`에만
격리한다(Phase 6B의 `application/inference_controller.py`와 동일한 경계
원칙).

`InferenceController`(단일 이미지)와 **의도적으로 같은 모양의** 아주
단순한 state machine을 쓴다 -- `idle`/`running`/`finished`/`failed`
네 상태뿐이고 cooperative stop도 `stopping` 상태도 없다. 두 controller가
공통 base class를 공유하지 않는 것도 동일한 이유다: 억지 추상화보다
읽기 쉬운 중복이 낫다(YAGNI, Phase 6A §8 결론).

**핵심 계약**: 한 이미지의 실패는 `run_folder_inference`가 aggregate
결과(`FolderInferenceResult`) 안에 bounded per-image error로 담아
정상적으로 반환한다 -- 그런 결과를 받으면 controller는 `finished`로
전이한다. `failed`는 폴더 연산 자체가 치명적 예외(존재하지 않는 폴더,
지원 이미지 0장 등 `FolderInferenceError`)를 던졌을 때만 쓴다."""
from __future__ import annotations

from typing import Callable, Literal

from image_ai_studio.inference.folder_inference import (
    FolderInferenceRequest,
    FolderInferenceResult,
    run_folder_inference,
)

FolderInferenceBackend = Callable[[FolderInferenceRequest], FolderInferenceResult]
"""`run_folder_inference()`와 동일한 signature(`request` 하나만 받는
callable, CP1 계약). 테스트는 fake backend를 주입해 실제 model/이미지
없이 controller 상태 전이만 검증한다."""

FolderInferenceControllerState = Literal["idle", "running", "finished", "failed"]


class FolderInferenceAlreadyRunningError(RuntimeError):
    """`FolderInferenceController.begin_run()`이 이미 `running` 상태에서
    호출됐을 때 발생한다 -- single active folder run만 지원한다(단일
    이미지 쪽 `InferenceAlreadyRunningError`와 동일한 계약)."""


class FolderInferenceController:
    """single active folder inference run의 application-level lifecycle을
    관리한다. `InferenceController`처럼 이 클래스 자신은 어떤 thread에서도
    실행될 수 있는 순수 Python 객체다 -- `begin_run()`은 caller의 현재
    thread에서 상태만 동기적으로 바꾸고(빠름), `run()`은 caller가 고른
    thread에서 folder backend 호출을 블로킹으로 수행한다(보통 worker
    thread). cooperative stop이 없으므로 `threading.Event`도 갖지 않는다."""

    def __init__(self, backend: FolderInferenceBackend = run_folder_inference) -> None:
        self._backend = backend
        self._state: FolderInferenceControllerState = "idle"

    @property
    def state(self) -> FolderInferenceControllerState:
        return self._state

    @property
    def is_running(self) -> bool:
        """`running`이면 True -- single active folder run 여부를 판단하는
        유일한 기준이다."""
        return self._state == "running"

    def begin_run(self) -> None:
        """새 folder inference를 시작하기 직전 호출한다(Qt worker가 실제
        backend 호출을 시작하기 전). 이미 `running`이면
        `FolderInferenceAlreadyRunningError`를 던지고 상태를 바꾸지
        않는다. `idle`/`finished`/`failed` 어디서든 새 run을 시작할 수
        있다 -- 별도의 "reset" 단계는 없다(`InferenceController.begin_run()`
        과 동일한 계약)."""
        if self.is_running:
            raise FolderInferenceAlreadyRunningError(
                f"cannot start a new folder inference run while controller state is {self._state!r}"
            )
        self._state = "running"

    def run(self, request: FolderInferenceRequest) -> FolderInferenceResult:
        """`begin_run()` 이후 folder backend를 블로킹으로 호출한다.
        backend가 aggregate 결과를 돌려주면 -- 그 결과에 per-image 실패가
        섞여 있더라도 -- `state`를 `finished`로 바꾸고 그 결과를 그대로
        반환한다. backend가 치명적 예외를 던지면 `state`를 `failed`로
        바꾸고 그 예외를 그대로 다시 던진다(swallow하지 않음).

        `state`가 `running`일 때만 호출할 수 있다 -- `begin_run()`을
        거치지 않았거나(`idle`), 이미 이전 run이 끝난 뒤(`finished`/
        `failed`) 새 `begin_run()` 없이 다시 호출하면 `RuntimeError`를
        던진다."""
        if self._state != "running":
            raise RuntimeError(
                "FolderInferenceController.run() requires state 'running' "
                f"(call begin_run() first) -- got state={self._state!r}"
            )
        try:
            result = self._backend(request)
        except BaseException:
            self._state = "failed"
            raise
        self._state = "finished"
        return result
