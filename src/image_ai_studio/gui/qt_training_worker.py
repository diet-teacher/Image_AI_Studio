"""Phase 5B: PySide6 Qt worker that runs a single
`ImageFolderWorkflowRequest` on a background `QThread` via
`TrainingController`. 이 모듈이 이 repository에서 PySide6와
`image_ai_studio.application`을 함께 import하는 유일한 위치다 --
`TrainingController` 자신은 PySide6를 전혀 모른다(docs/
phase5b_application_qt_worker_integration_design.md §2).

이 모듈을 import하는 것만으로는 `QApplication`/`QThread` 생성, CUDA
초기화, 학습 시작 등 어떤 side effect도 일어나지 않는다 -- 클래스
정의만 있을 뿐이다."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal

from image_ai_studio.application.training_controller import (
    TrainingAlreadyRunningError,
    TrainingController,
)
from image_ai_studio.training.imagefolder_workflow import ImageFolderWorkflowRequest


class QtTrainingWorker(QObject):
    """`QThread.moveToThread()`의 대상이 되는 `QObject`(표준 Qt worker
    패턴). 이 객체는 살아있는 model/optimizer/tensor를 전혀 소유하지
    않는다 -- `run_imagefolder_training_workflow()`가 모든 학습 state를
    자기 내부에서만 관리하고 밖으로 반환하지 않으므로(Phase 4 계약),
    GUI/main thread가 CUDA tensor/model을 만들어 이 worker로 넘기는
    일은 애초에 발생하지 않는다.

    사용 패턴(Phase 5C가 그대로 재사용할 것을 의도함)::

        thread = QThread()
        worker = QtTrainingWorker(controller, request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(main_window.on_progress)   # 아래 경고 참고
        worker.finished.connect(main_window.on_finished)   # 〃
        worker.failed.connect(main_window.on_failed)       # 〃
        thread.start()
        ...
        controller.request_stop()              # GUI thread에서 언제든 호출 가능
        ...
        thread.quit(); thread.wait()           # finished/failed 수신 후 정리

    **중요(empirical 확인, docs/phase5b_..._design.md §9 참고): signal을
    QObject가 아닌 평범한 함수/lambda에 connect하면 그 슬롯은 GUI
    thread로 자동 queue되지 않고 emit이 일어난 이 worker thread에서
    직접 실행된다.** 안전하게 GUI를 갱신하려면 반드시 실제 QObject
    (예: `QMainWindow`/`QWidget`) 인스턴스 메서드에 connect하거나
    `type=Qt.ConnectionType.QueuedConnection`을 명시해야 한다 --
    위 예시의 `main_window.on_progress`처럼 QObject bound method를
    쓰는 것이 그 이유다.

    `QThread.terminate()`는 절대 쓰지 않는다 -- cooperative stop만
    지원한다(Phase 4K/Phase 5A 계약)."""

    progress = Signal(object)  # TrainingProgress, worker thread에서 emit
    finished = Signal(object)  # ImageFolderWorkflowResult
    failed = Signal(str)  # f"{ExceptionType}: {message}\n{traceback}"

    def __init__(self, controller: TrainingController, request: ImageFolderWorkflowRequest) -> None:
        super().__init__()
        self._controller = controller
        self._request = request

    def run(self) -> None:
        """`QThread.started`에 연결해서 쓴다 -- 이 메서드가 실행되는
        thread 안에서 `begin_run()`부터 backend 호출까지 전부
        일어난다(model 생성/`.to(device)`/forward/backward가 이
        thread 하나로 시작·완결됨, Phase 5A CUDA+thread 결론과
        `phase5b_qthread_cuda_smoke` 실측으로 확인됨). 예외를 삼키지
        않는다 -- `failed` signal로 그대로 전달할 뿐, training core
        예외 semantics 자체는 바꾸지 않는다."""
        try:
            self._controller.begin_run()
        except TrainingAlreadyRunningError as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        try:
            result = self._controller.run(self._request, progress_callback=self.progress.emit)
        except Exception as exc:  # noqa: BLE001 -- 모든 실패를 GUI로 전달한다(swallow 금지)
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return

        self.finished.emit(result)
