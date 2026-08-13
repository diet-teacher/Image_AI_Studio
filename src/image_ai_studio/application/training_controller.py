"""Phase 5B: framework-agnostic application layer that sits between a
future GUI and Phase 4's `run_imagefolder_training_workflow()`. **이 모듈은
PySide6/Qt를 절대 import하지 않는다** -- GUI framework 의존은
`image_ai_studio.gui.qt_training_worker`에만 격리한다(docs/
phase5b_application_qt_worker_integration_design.md §2).

이 모듈이 하는 일과 하지 않는 일:

- `build_training_request()`: UI 입력값(문자열 경로 포함)에서
  `ImageFolderWorkflowRequest`를 조립하는 얇은 경계. semantic
  validation은 전혀 하지 않는다 -- `TrainingConfig`/
  `ImageFolderWorkflowRequest`/workflow 자체의 기존 검증을 그대로
  통과시킨다(중복 구현 금지).
- `TrainingController`: single-active-run 상태(`idle`/`running`/
  `stopping`/`finished`/`failed`)와 cooperative stop을 위한
  `threading.Event`를 관리하고, 주입 가능한 backend callable
  (`TrainingBackend`, 기본값 `run_imagefolder_training_workflow`)을
  호출한다. **thread/QThread를 직접 만들지 않는다** -- 어떤 thread
  에서 `run()`을 호출할지는 전적으로 caller(예:
  `gui.qt_training_worker.QtTrainingWorker`)의 책임이다."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Literal

from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.imagefolder_workflow import (
    SEED,
    ImageFolderWorkflowRequest,
    ImageFolderWorkflowResult,
    run_imagefolder_training_workflow,
)
from image_ai_studio.training.loop import TrainingProgressCallback

TrainingBackend = Callable[..., ImageFolderWorkflowResult]
"""`run_imagefolder_training_workflow()`와 동일한 signature
(`request`, `*`, `progress_callback=`, `should_stop=`)를 갖는 아무
callable. 테스트에서는 실제 학습 없이 이 자리에 fake를 주입한다."""

ControllerState = Literal["idle", "running", "stopping", "finished", "failed"]


class TrainingAlreadyRunningError(RuntimeError):
    """`TrainingController.begin_run()`이 이미 `running`/`stopping` 상태에서
    호출됐을 때 발생한다 -- Phase 5B는 single active run만 지원한다."""


def build_training_request(
    *,
    model_json_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
    label_smoothing: float = 0.0,
    class_weights: tuple[float, ...] | None = None,
    lr_scheduler: str | None = None,
    lr_scheduler_factor: float = 0.1,
    lr_scheduler_patience: int = 1,
    early_stopping_patience: int | None = None,
    precision: str = "fp32",
    device: str = "cpu",
    pin_memory: bool = False,
    non_blocking: bool = False,
    resume_from: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    checkpoint_every: int | None = None,
    export_torchscript: bool = True,
    seed: int = SEED,
) -> ImageFolderWorkflowRequest:
    """GUI 입력값과 `ImageFolderWorkflowRequest` 사이의 request-builder
    boundary(Phase 5A 조사 결론). 문자열 경로를 `Path`로 바꾸는 것
    외에는 아무 값도 검증/가공하지 않는다 -- `TrainingConfig.__post_init__`/
    `run_imagefolder_training_workflow()`의 기존 검증이 그대로
    authoritative하다(Phase 5A 조사에서 정한 validation responsibility
    boundary). 이 함수가 실패를 삼키는 경우는 없다 -- `TrainingConfig`/
    `ImageFolderWorkflowRequest` 생성자가 던지는 예외를 그대로
    전파한다."""
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        lr_scheduler=lr_scheduler,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        early_stopping_patience=early_stopping_patience,
        precision=precision,
    )
    return ImageFolderWorkflowRequest(
        model_json_path=Path(model_json_path),
        dataset_root=Path(dataset_root),
        training_config=training_config,
        output_dir=Path(output_dir),
        resume_from=Path(resume_from) if resume_from is not None else None,
        checkpoint_out=Path(checkpoint_out) if checkpoint_out is not None else None,
        export_torchscript=export_torchscript,
        seed=seed,
        checkpoint_every=checkpoint_every,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )


class TrainingController:
    """single active training run의 application-level lifecycle을
    관리한다. 이 클래스 자신은 어떤 thread에서도 실행될 수 있는 순수
    Python 객체다 -- `begin_run()`은 caller의 현재 thread에서 상태만
    동기적으로 바꾸고(빠름), `run()`은 caller가 고른 thread에서
    실제 backend 호출을 블로킹으로 수행한다(느림, 보통 worker
    thread에서 호출됨). 이 분리 덕분에 "이미 실행 중" 거부를 별도
    thread를 만들기 전에 즉시 알 수 있다."""

    def __init__(self, backend: TrainingBackend = run_imagefolder_training_workflow) -> None:
        self._backend = backend
        self._state: ControllerState = "idle"
        self._stop_event: threading.Event | None = None

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def is_running(self) -> bool:
        """`running` 또는 `stopping`이면 True -- 이 프로퍼티가 single
        active run 여부를 판단하는 유일한 기준이다."""
        return self._state in ("running", "stopping")

    def begin_run(self) -> None:
        """새 run을 시작하기 직전 호출한다(Qt worker가 실제 backend
        호출을 시작하기 전). 이미 `running`/`stopping`이면
        `TrainingAlreadyRunningError`를 던지고 상태를 바꾸지 않는다.
        `idle`/`finished`/`failed` 어디서든 새 run을 시작할 수 있다 --
        "Finished에서 Idle로 되돌아가는" 별도 단계는 없다(docs/
        phase5b_application_qt_worker_integration_design.md §6 state
        model 참고)."""
        if self.is_running:
            raise TrainingAlreadyRunningError(
                f"cannot start a new training run while controller state is {self._state!r}"
            )
        self._stop_event = threading.Event()
        self._state = "running"

    def request_stop(self) -> None:
        """cooperative stop을 요청한다. `running`이 아니면 조용히
        아무 일도 하지 않는다(중복 클릭/이미 끝난 뒤의 요청을 에러로
        취급하지 않는다). 즉시 중단이 아니라 Phase 4의 기존 epoch
        경계 cooperative stop 그대로다 -- `state`는 `stopping`으로
        바뀌지만 실제 종료는 `run()`이 반환할 때 일어난다."""
        stop_event = self._stop_event
        if stop_event is None or not self.is_running:
            return
        stop_event.set()
        if self._state == "running":
            self._state = "stopping"

    def run(
        self,
        request: ImageFolderWorkflowRequest,
        *,
        progress_callback: TrainingProgressCallback | None = None,
    ) -> ImageFolderWorkflowResult:
        """`begin_run()` 이후 실제 backend를 블로킹으로 호출한다 --
        이 메서드를 호출한 thread에서 학습 전체(model 생성 포함)가
        수행된다(Phase 5A CUDA+thread 결론: model/CUDA context를
        미리 다른 thread에서 만들지 않는다). 성공하면 `state`를
        `finished`로, backend가 예외를 던지면 `state`를 `failed`로
        바꾸고 그 예외를 그대로 다시 던진다(swallow하지 않음 --
        Phase 4 exception 정책을 그대로 유지).

        `state`가 `running`/`stopping`일 때만 호출할 수 있다 --
        `begin_run()`을 거치지 않았거나(`idle`), 이미 이전 run이
        끝난 뒤(`finished`/`failed`) 새 `begin_run()` 없이 다시
        호출하면 `RuntimeError`를 던진다. `stopping`은 허용한다 --
        `request_stop()` 이후에도 실제 backend 호출/반환은 정상
        흐름이다(cooperative stop, §8)."""
        if self._stop_event is None or self._state not in ("running", "stopping"):
            raise RuntimeError(
                "TrainingController.run() requires state 'running' or 'stopping' "
                f"(call begin_run() first) -- got state={self._state!r}"
            )
        try:
            result = self._backend(
                request,
                progress_callback=progress_callback,
                should_stop=self._stop_event.is_set,
            )
        except BaseException:
            self._state = "failed"
            raise
        self._state = "finished"
        return result
