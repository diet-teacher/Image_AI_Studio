"""Phase 6B: PySide6 Qt worker that runs a single `InferenceRequest`
on a background `QThread` via `InferenceController`. 이 모듈이 이
repository에서 PySide6와 `image_ai_studio.application`/
`image_ai_studio.inference`를 함께 import하는 유일한 위치다 --
`InferenceController` 자신은 PySide6를 전혀 모른다(Phase 5B의
`gui/qt_training_worker.py`와 동일한 경계 원칙).

`QtTrainingWorker`와 거의 같은 모양이지만, inference에 필요 없는 것은
복사하지 않는다: `progress` signal이 없다(단일 이미지는 진행률 개념이
없다, Phase 6A §8) -- `finished`/`failed` 두 signal뿐이다.

이 모듈을 import하는 것만으로는 `QApplication`/`QThread` 생성, CUDA
초기화, inference 시작 등 어떤 side effect도 일어나지 않는다 -- 클래스
정의만 있을 뿐이다."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal

from image_ai_studio.application.inference_controller import (
    InferenceAlreadyRunningError,
    InferenceController,
)
from image_ai_studio.inference.single_image_inference import InferenceRequest


class QtInferenceWorker(QObject):
    """`QThread.moveToThread()`의 대상이 되는 `QObject`(표준 Qt worker
    패턴, `QtTrainingWorker`와 동일). 이 객체는 살아있는 model/tensor를
    전혀 소유하지 않는다 -- `run_single_image_inference()`가 모든
    inference state를 자기 내부에서만 관리하고 밖으로 반환하지 않는다.

    사용 패턴(Phase 6C가 그대로 재사용할 것을 의도함, `QtTrainingWorker`
    와 동일한 lifecycle)::

        thread = QThread()
        worker = QtInferenceWorker(controller, request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(page.on_finished)   # 아래 경고 참고
        worker.failed.connect(page.on_failed)       # 〃
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)   # worker 자신의 신호에!
        worker.failed.connect(worker.deleteLater)     # 〃
        thread.finished.connect(thread.deleteLater)
        thread.start()

    **deleteLater ordering(Phase 5C stabilization이 확정한 계약,
    docs/phase5c_training_gui_design.md §18 -- 절대 다시 어기지 않는다)**:
    `worker.deleteLater()`는 반드시 worker 자신의 `finished`/`failed`에
    연결한다. `thread.finished`에 연결하면(worker thread의 event loop가
    이미 멈춘 뒤라) deferred deletion이 안전하게 처리되지 않을 수 있다 --
    이 잘못된 패턴(`thread.finished -> worker.deleteLater`)은 Phase 6B
    에서도 다시 쓰지 않는다.

    **중요(Phase 5B에서 empirical 확인, 그대로 적용): signal을 QObject가
    아닌 평범한 함수/lambda에 connect하면 그 슬롯은 GUI thread로 자동
    queue되지 않고 emit이 일어난 이 worker thread에서 직접 실행된다.**
    안전하게 GUI를 갱신하려면 반드시 실제 QObject 인스턴스 메서드에
    connect해야 한다.

    `QThread.terminate()`는 절대 쓰지 않는다. inference는 원자적
    단일 forward pass라 cooperative stop 자체가 없다(Phase 6A §8) --
    이 worker에는 `request_stop()`에 대응하는 어떤 API도 없다."""

    finished = Signal(object)  # InferenceResult
    failed = Signal(str)  # f"{ExceptionType}: {message}\n{traceback}"

    def __init__(self, controller: InferenceController, request: InferenceRequest) -> None:
        super().__init__()
        self._controller = controller
        self._request = request

    def run(self) -> None:
        """`QThread.started`에 연결해서 쓴다 -- 이 메서드가 실행되는
        thread 안에서 `begin_run()`부터 backend(model 재구성/`.to(device)`/
        forward) 호출까지 전부 일어난다. 예외를 삼키지 않는다 --
        `failed` signal로 그대로 전달할 뿐, inference core 예외
        semantics 자체는 바꾸지 않는다."""
        try:
            self._controller.begin_run()
        except InferenceAlreadyRunningError as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        try:
            result = self._controller.run(self._request)
        except Exception as exc:  # noqa: BLE001 -- 모든 실패를 GUI로 전달한다(swallow 금지)
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return

        self.finished.emit(result)
