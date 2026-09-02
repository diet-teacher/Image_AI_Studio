"""Phase 10 CP2: PySide6 Qt worker that runs one `FolderInferenceRequest`
on a background `QThread` via `FolderInferenceController`. 이 모듈이 이
repository에서 PySide6와 `image_ai_studio.application.folder_inference_controller`
/`image_ai_studio.inference.folder_inference`를 함께 import하는 유일한
위치다 -- `FolderInferenceController` 자신은 PySide6를 전혀 모른다
(Phase 6B의 `gui/qt_inference_worker.py`와 동일한 경계 원칙).

Phase 12 CP2: worker는 이제 `progress`/`cancelled` signal을 `finished`/
`failed`와 함께 노출한다. 한 번의 `run()` 호출은 `finished`/`cancelled`/
`failed` **셋 중 정확히 하나만, 한 번만** emit한다(terminal signal).
`progress` 스냅샷은 backend가 이미지 경계마다 순서대로 내보내며 항상
terminal signal보다 먼저 나간다.

**핵심 차이(CP1 계약)**: per-image 실패는 `FolderInferenceResult`
안에 담겨 정상 aggregate로 돌아오므로 `finished`(취소된 run이면
`cancelled`의 부분 결과)로 전달된다. `failed`는 폴더 연산 자체가
치명적 예외(존재하지 않는 폴더, 지원 이미지 0장 등
`FolderInferenceError`)를 던졌을 때만 emit된다. 협조적 취소
(`FolderInferenceCancelled`)는 `failed`가 아니라 `cancelled`로,
치명적 실패와 **예외 타입으로만** 구분한다.

이 모듈을 import하는 것만으로는 `QApplication`/`QThread` 생성, CUDA
초기화, inference 시작 등 어떤 side effect도 일어나지 않는다 -- 클래스
정의만 있을 뿐이다."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, Signal

from image_ai_studio.application.folder_inference_controller import (
    FolderInferenceAlreadyRunningError,
    FolderInferenceController,
)
from image_ai_studio.inference.folder_inference import (
    FolderInferenceCancelled,
    FolderInferenceRequest,
)


class QtFolderInferenceWorker(QObject):
    """`QThread.moveToThread()`의 대상이 되는 `QObject`(표준 Qt worker
    패턴, `QtInferenceWorker`와 동일). 이 객체는 살아있는 model/tensor를
    전혀 소유하지 않는다 -- `run_folder_inference()`가 모든 inference
    state를 자기 내부에서만 관리하고 aggregate 결과값만 반환한다.

    사용 패턴(`QtInferenceWorker`와 동일한 canonical lifecycle, Phase 12
    CP2에서 `progress`/`cancelled` 배선이 추가됨)::

        thread = QThread()
        worker = QtFolderInferenceWorker(controller, request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(page.on_folder_progress)   # 아래 경고 참고
        worker.finished.connect(page.on_folder_finished)   # 〃
        worker.cancelled.connect(page.on_folder_cancelled) # 〃
        worker.failed.connect(page.on_folder_failed)       # 〃
        worker.finished.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)   # worker 자신의 신호에!
        worker.cancelled.connect(worker.deleteLater)  # 〃
        worker.failed.connect(worker.deleteLater)     # 〃
        thread.finished.connect(thread.deleteLater)
        thread.start()
        ...
        worker.request_cancel()   # GUI thread에서 언제든, 몇 번이든

    **deleteLater ordering(Phase 5C stabilization이 확정한 계약,
    docs/phase5c_training_gui_design.md §18 -- 절대 다시 어기지 않는다)**:
    `worker.deleteLater()`는 반드시 worker 자신의 `finished`/`cancelled`/
    `failed`에 연결한다. `thread.finished`에 연결하면(worker thread의
    event loop가 이미 멈춘 뒤라) deferred deletion이 안전하게 처리되지
    않을 수 있다 -- 이 잘못된 패턴(`thread.finished -> worker.deleteLater`)
    은 여기서도 다시 쓰지 않는다. `thread.deleteLater()`만
    `thread.finished`에 연결한다.

    **중요(Phase 5B에서 empirical 확인, 그대로 적용): signal을 QObject가
    아닌 평범한 함수/lambda에 connect하면 그 슬롯은 GUI thread로 자동
    queue되지 않고 emit이 일어난 이 worker thread에서 직접 실행된다.**
    안전하게 GUI를 갱신하려면 반드시 실제 QObject 인스턴스 메서드에
    connect해야 한다.

    `QThread.terminate()`는 절대 쓰지 않는다. 취소는 `request_cancel()`
    로 요청하는 **협조적** 취소뿐이다 -- controller의 thread-safe
    `threading.Event` 하나를 set할 뿐이고, 실제 중단은 folder inference
    core가 다음 이미지 경계에서 관측한다. 진행 중인 단일 이미지 forward
    pass는 절대 중단하지 않는다."""

    progress = Signal(object)  # FolderInferenceProgress, worker thread에서 emit
    finished = Signal(object)  # FolderInferenceResult
    cancelled = Signal(object, int)  # (partial FolderInferenceResult, discovered_total)
    failed = Signal(str)  # f"{ExceptionType}: {message}\n{traceback}"

    def __init__(
        self, controller: FolderInferenceController, request: FolderInferenceRequest
    ) -> None:
        super().__init__()
        self._controller = controller
        self._request = request
        self._prepared = False

    def prepare_run(self) -> None:
        """Prepare exactly one controller run before it is exposed as active.

        ``InferencePage`` calls this synchronously before starting the worker
        thread, so an immediate Cancel/close request sets the same token that
        :meth:`run` later passes to the backend.  Direct worker users remain
        compatible because :meth:`run` calls this method as a fallback.
        """
        if self._prepared:
            return
        self._controller.begin_run()
        self._prepared = True

    def request_cancel(self) -> None:
        """GUI thread에서 협조적 취소를 요청한다 -- 몇 번이고 호출해도
        안전하다(idempotent). controller의 thread-safe 취소 플래그
        (`threading.Event`)를 set하도록 위임할 뿐이라 blocking/wait/busy
        loop/`terminate`/worker thread 객체 직접 변경이 전혀 없다. 실제
        취소 관측은 folder inference core가 다음 이미지 경계에서 한다."""
        self._controller.request_cancel()

    def run(self) -> None:
        """`QThread.started`에 연결해서 쓴다 -- 이 메서드가 실행되는
        thread 안에서 준비된 controller로 folder backend(discovery + 이미지
        순차 처리)를 호출한다. page가 미리 준비하지 않은 직접 호출은
        `prepare_run()` fallback이 먼저 `begin_run()`을 수행한다. `progress`
        스냅샷은 backend가 이미지 경계마다 순서대로 내보낸다.

        예외를 삼키지 않는다. 협조적 취소(`FolderInferenceCancelled`)는
        부분 결과와 discovered-total을 실어 `cancelled`로, 그 밖의
        치명적 예외는 `failed`로 전달한다 -- folder inference core 예외
        semantics 자체는 바꾸지 않는다. per-image 실패가 섞인 aggregate
        결과는 `finished`로 emit한다(치명적 예외가 아니다)."""
        try:
            self.prepare_run()
        except FolderInferenceAlreadyRunningError as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        try:
            result = self._controller.run(
                self._request, progress_callback=self.progress.emit
            )
        except FolderInferenceCancelled as exc:
            self.cancelled.emit(exc.result, exc.discovered_total)
            return
        except Exception as exc:  # noqa: BLE001 -- 모든 치명적 실패를 GUI로 전달한다(swallow 금지)
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return

        self.finished.emit(result)
