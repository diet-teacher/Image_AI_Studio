"""Phase 10 CP2: framework-agnostic application layer for one folder
inference run. **이 모듈은 PySide6/Qt를 절대 import하지 않는다** -- GUI
framework 의존은 `image_ai_studio.gui.qt_folder_inference_worker`에만
격리한다(Phase 6B의 `application/inference_controller.py`와 동일한 경계
원칙).

`InferenceController`(단일 이미지)와 **의도적으로 같은 모양의** 아주
단순한 state machine을 쓴다 -- `idle`/`running`/`finished`/`failed`에
Phase 12 CP2가 협조적(cooperative) 취소를 위한 terminal 상태
`cancelled` 하나만 더한다. 두 controller가 공통 base class를 공유하지
않는 것도 동일한 이유다: 억지 추상화보다 읽기 쉬운 중복이 낫다
(YAGNI, Phase 6A §8 결론).

**핵심 계약**: 한 이미지의 실패는 `run_folder_inference`가 aggregate
결과(`FolderInferenceResult`) 안에 bounded per-image error로 담아
정상적으로 반환한다 -- 그런 결과를 받으면 controller는 `finished`로
전이한다. `failed`는 폴더 연산 자체가 치명적 예외(존재하지 않는 폴더,
지원 이미지 0장 등 `FolderInferenceError`)를 던졌을 때만 쓴다.

**Phase 12 CP2 취소 계약**: `request_cancel()`은 GUI thread에서 몇
번이고 호출할 수 있는 thread-safe·idempotent 연산으로, backend에
넘겨줄 `threading.Event` 하나만 set한다(blocking/wait/busy loop 없음).
매 `begin_run()`은 **새 Event**를 만들어 직전 run의 취소 요청을
버린다 -- 그래서 오래된 취소 요청이 다음 run을 취소시키지 못한다.
backend가 협조적으로 멈추면(`FolderInferenceCancelled`) controller는
`cancelled`로 전이하고 그 예외를 그대로 다시 던진다 -- 부분(partial)
`FolderInferenceResult`와 discovered-total 메타데이터는 CP1이 그
예외에 실어 나른 값 그대로 보존한다. 치명적 예외와 협조적 취소는
**예외 타입으로만** 구분한다 -- 오류 문자열을 파싱하지 않는다."""
from __future__ import annotations

import inspect
import threading
from typing import Callable, Literal

from image_ai_studio.inference.folder_inference import (
    FolderInferenceCancelled,
    FolderInferenceProgress,
    FolderInferenceRequest,
    FolderInferenceResult,
    run_folder_inference,
)

FolderInferenceBackend = Callable[..., FolderInferenceResult]
"""주입되는 folder backend. 두 모양을 모두 지원한다:

* **기존 1-인자 backend** -- `run_folder_inference`의 초기 계약처럼
  `request` 하나만 받는 callable. 진행률/취소 hook을 요구하지 않는다.
* **협조적 backend** -- `run_folder_inference`처럼 `progress_callback=`/
  `should_cancel=` keyword hook을 받는 callable.

어느 쪽인지는 생성 시 `inspect.signature`로 **한 번만** 판정한다 --
backend를 호출해 보고 `TypeError` 문자열을 파싱하지 않는다."""

FolderProgressCallback = Callable[[FolderInferenceProgress], None]
"""`run()`에 넘기는 optional 진행률 콜백(CP1 `ProgressCallback`과 동일
모양). 그대로 backend로 전달된다 -- controller는 wrapping하지 않는다."""

FolderInferenceControllerState = Literal[
    "idle", "running", "finished", "failed", "cancelled"
]


class FolderInferenceAlreadyRunningError(RuntimeError):
    """`FolderInferenceController.begin_run()`이 이미 `running` 상태에서
    호출됐을 때 발생한다 -- single active folder run만 지원한다(단일
    이미지 쪽 `InferenceAlreadyRunningError`와 동일한 계약)."""


def _backend_accepts_cooperative_hooks(backend: FolderInferenceBackend) -> bool:
    """`backend`가 CP1의 `progress_callback`/`should_cancel` keyword hook을
    **명시적으로** 받는지 signature로 판정한다(둘 다 받거나 `**kwargs`가
    있으면 True). 이 경계가 "협조적 backend vs 기존 1-인자 backend"를
    가르는 유일한 지점이다 -- backend를 호출해 예외 메시지를 뜯어보는
    방식은 쓰지 않는다."""
    try:
        parameters = inspect.signature(backend).parameters
    except (TypeError, ValueError):
        return False
    values = list(parameters.values())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in values):
        return True
    keyword_names = {
        p.name
        for p in values
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {"progress_callback", "should_cancel"} <= keyword_names


class FolderInferenceController:
    """single active folder inference run의 application-level lifecycle을
    관리한다. `InferenceController`처럼 이 클래스 자신은 어떤 thread에서도
    실행될 수 있는 순수 Python 객체다 -- `begin_run()`은 caller의 현재
    thread에서 상태만 동기적으로 바꾸고(빠름), `run()`은 caller가 고른
    thread에서 folder backend 호출을 블로킹으로 수행한다(보통 worker
    thread). 취소 요청은 `threading.Event` 하나로만 표현하며(`run()`이
    반환할 때까지 상태를 `running`으로 유지, 별도 `cancelling` 상태
    없음) backend가 협조적으로 멈춘 뒤에야 `cancelled`로 전이한다."""

    def __init__(
        self, backend: FolderInferenceBackend = run_folder_inference
    ) -> None:
        self._backend = backend
        self._backend_supports_hooks = _backend_accepts_cooperative_hooks(backend)
        self._state: FolderInferenceControllerState = "idle"
        # 취소 플래그. begin_run()마다 새 Event로 교체되므로 여기 초기
        # Event는 첫 begin_run() 전에 들어온 request_cancel()만 흡수한다.
        self._cancel_event = threading.Event()

    @property
    def state(self) -> FolderInferenceControllerState:
        return self._state

    @property
    def is_running(self) -> bool:
        """`running`이면 True -- single active folder run 여부를 판단하는
        유일한 기준이다. `cancelled`는 terminal 상태이므로 False다."""
        return self._state == "running"

    @property
    def cancel_requested(self) -> bool:
        """현재 run에 대해 취소가 요청됐는지(관측용). backend가 아직
        이미지 경계에 도달하지 않았을 수 있으므로 True라고 해서 이미
        멈췄다는 뜻은 아니다."""
        return self._cancel_event.is_set()

    def begin_run(self) -> None:
        """새 folder inference를 시작하기 직전 호출한다(Qt worker가 실제
        backend 호출을 시작하기 전). 이미 `running`이면
        `FolderInferenceAlreadyRunningError`를 던지고 상태를 바꾸지
        않는다. `idle`/`finished`/`failed`/`cancelled` 어디서든 새 run을
        시작할 수 있다 -- 별도의 "reset" 단계는 없다.

        **취소 플래그만** 새 Event로 교체한다(그 외 상태는 건드리지
        않는다). 직전 run에 대해 들어와 있던 취소 요청(또는 terminal
        이후 뒤늦게 들어온 stale 요청)은 이 교체로 사라지므로, 새 run이
        오래된 취소 요청 때문에 취소되는 일은 없다."""
        if self.is_running:
            raise FolderInferenceAlreadyRunningError(
                f"cannot start a new folder inference run while controller state is {self._state!r}"
            )
        self._cancel_event = threading.Event()
        self._state = "running"

    def request_cancel(self) -> None:
        """현재(또는 다음) folder run에 대한 협조적 취소를 요청한다. GUI
        thread에서 몇 번이고 호출해도 안전하다 -- `threading.Event.set()`
        하나만 호출하므로 blocking/wait/busy loop/`terminate`가 전혀
        없고, 중복 호출은 그냥 무해하다(idempotent).

        상태는 여기서 바꾸지 않는다. 실제 취소는 backend가 다음 이미지
        경계에서 `should_cancel`을 관측해 `FolderInferenceCancelled`를
        던지고, `run()`이 그것을 받아 `cancelled`로 전이할 때 일어난다.
        진행 중인 단일 이미지 forward pass는 절대 중단되지 않는다.

        `running`이 아닐 때 호출하면(예: `idle`, terminal 이후) 플래그만
        set될 뿐이고, 다음 `begin_run()`이 그 플래그를 버린다."""
        self._cancel_event.set()

    def run(
        self,
        request: FolderInferenceRequest,
        *,
        progress_callback: FolderProgressCallback | None = None,
    ) -> FolderInferenceResult:
        """`begin_run()` 이후 folder backend를 블로킹으로 호출한다.

        backend가 aggregate 결과를 돌려주면 -- 그 결과에 per-image 실패가
        섞여 있더라도 -- `state`를 `finished`로 바꾸고 그 결과를 그대로
        반환한다. backend가 `FolderInferenceCancelled`를 던지면(협조적
        취소) `state`를 `cancelled`로 바꾸고 그 예외를 **그대로 다시
        던진다** -- 부분 `FolderInferenceResult`와 `discovered_total`은
        CP1이 예외에 실어 둔 값 그대로다. 그 밖의 치명적 예외를 던지면
        `state`를 `failed`로 바꾸고 다시 던진다(swallow하지 않음). 협조적
        취소와 치명적 실패는 예외 타입으로만 구분한다.

        `progress_callback`은 협조적 backend에만(=hook을 받는 backend)
        `progress_callback=`으로 그대로 전달된다. 기존 1-인자 backend는
        `request` 하나로만 호출되며 진행률/취소를 관측하지 않는다 --
        기존 호출자·backend 호환을 위한 명시적 경계다.

        `state`가 `running`일 때만 호출할 수 있다 -- `begin_run()`을
        거치지 않았거나(`idle`), 이미 이전 run이 끝난 뒤(`finished`/
        `failed`/`cancelled`) 새 `begin_run()` 없이 다시 호출하면
        `RuntimeError`를 던진다."""
        if self._state != "running":
            raise RuntimeError(
                "FolderInferenceController.run() requires state 'running' "
                f"(call begin_run() first) -- got state={self._state!r}"
            )
        cancel_event = self._cancel_event
        try:
            if self._backend_supports_hooks:
                result = self._backend(
                    request,
                    progress_callback=progress_callback,
                    should_cancel=cancel_event.is_set,
                )
            else:
                result = self._backend(request)
        except FolderInferenceCancelled:
            self._state = "cancelled"
            raise
        except BaseException:
            self._state = "failed"
            raise
        self._state = "finished"
        return result
