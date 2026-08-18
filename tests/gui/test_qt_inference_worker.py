"""`QtInferenceWorker`/`QThread` wiring 테스트(Phase 6B). `pytest-qt`의
`qtbot` fixture가 `QApplication`을 관리해 준다. 매 테스트마다 실제
inference를 돌리지 않는다 -- fake backend를 주입해 signal 전달/thread
분리/cleanup만 검증한다(실제 CPU/CUDA inference는
`test_qt_inference_worker_integration.py`가 따로 담당한다).

`QtTrainingWorker`의 테스트 구조(Phase 5B)를 그대로 참고하되, inference에
없는 것(progress signal, cooperative stop)에 대응하는 테스트는 만들지
않는다.

**Phase 6B stabilization(native abort 조사)**: 두 가지를 고쳤다.

1. 모든 테스트가 `_wire_full_lifecycle()`(worker 자신의 finished/failed에
   `deleteLater()`를 연결하는 canonical wiring, `QtInferenceWorker`의
   docstring이 규정하는 것과 동일)을 쓴다 -- 이전에는 일부 테스트가
   `thread.quit(); thread.wait(5000)`만 하고 `worker.deleteLater()`를
   전혀 연결하지 않아, `worker`(`moveToThread()`로 worker thread에
   affinity가 있는 QObject)가 스코프를 벗어날 때 Python GC가 **worker
   thread가 아닌 main thread에서** 그 C++ 객체를 파괴하는 cross-thread
   QObject 삭제가 됐다(Qt 문서상 정의되지 않은 동작).
2. **`qtbot.waitSignal(worker.finished/failed, ...)`을 더 이상 쓰지
   않는다.** worker.finished/failed에는 이미 `worker.deleteLater()`가
   연결돼 있는데, `qtbot.waitSignal()`은 같은 signal에 자신의 임시
   `SignalBlocker`를 connect했다가 `with` 블록이 끝날 때 다시
   disconnect한다 -- 이 disconnect 시도가 (worker thread에서 처리되는)
   실제 `deleteLater()` 삭제와 경합하는 것을 실측으로 재현했다:

       RuntimeWarning: libpyside: Failed to disconnect
       (<bound method SignalBlocker._quit_loop_by_signal ...>)
       from signal "finished(PyObject)". signal.disconnect(slot)

   이 경합은 이미 삭제됐거나 삭제 중인 C++ 객체의 signal에 접근하려는
   시도이므로, 드문 타이밍에서는 warning이 아니라 native access
   violation으로 이어질 수 있다는 것이 이번 조사의 결론이다(반복
   실행 중 관찰된 `Windows fatal exception: access violation`,
   "Current thread"에 Python frame이 전혀 없었던 것과 정확히 부합).
   `qtbot.waitSignal()` 대신, signal에 **영구적으로**(임시 connect/
   disconnect 없이) 연결한 plain 관찰자(`list.append`, worker thread
   에서 직접 실행되지만 CPython list.append는 GIL 덕분에 atomic이라
   안전하다) + `qtbot.waitUntil()` polling으로 완전히 대체했다 --
   이 방식은 signal에 어떤 임시 connect/disconnect도 하지 않으므로
   위 경합 자체가 구조적으로 발생할 수 없다."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from image_ai_studio.application.inference_controller import InferenceController
from image_ai_studio.gui.qt_inference_worker import QtInferenceWorker
from image_ai_studio.inference.single_image_inference import InferenceRequest, InferenceResult

MAIN_THREAD_ID = threading.get_ident()


def _dummy_request() -> InferenceRequest:
    return InferenceRequest(
        model_json_path=Path("model.json"),
        state_dict_path=Path("state_dict.pt"),
        class_mapping_path=Path("class_mapping.json"),
        image_path=Path("image.png"),
        device="cpu",
        precision="fp32",
    )


def _fake_result() -> InferenceResult:
    return InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1},
        inference_duration_seconds=0.001,
    )


def _wire_full_lifecycle(controller: InferenceController, request: InferenceRequest) -> tuple[QThread, QtInferenceWorker]:
    """`QtInferenceWorker`의 docstring이 규정하는 canonical wiring --
    `worker.deleteLater()`는 반드시 worker 자신의 finished/failed에
    연결한다(Phase 5C stabilization이 확정한 계약, `thread.finished ->
    worker.deleteLater`로 절대 되돌리지 않는다). 이 helper를 모든
    테스트가 공유해서, 어떤 테스트도 이 canonical 패턴에서 벗어나지
    않게 한다(Phase 6B stabilization)."""
    thread = QThread()
    worker = QtInferenceWorker(controller, request)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker


def _wait_for_thread_cleanup(thread: QThread, qtbot, timeout: int = 5000) -> None:
    """`QThread`의 실행이 끝날 때까지 기다린다(`isRunning() is False`).
    `thread` 객체가 이미 `deleteLater()`로 삭제된 경우(`RuntimeError`
    -- C++ 쪽 객체에 더 이상 접근할 수 없음)도 정리 완료로 간주한다.

    **주의**: `isRunning() is False`가 된 시점과 `deleteLater()`로
    예약된 deferred QObject 삭제(worker/thread 양쪽)가 실제로 처리된
    시점이 동일하다는 보장은 없다 -- 이 helper는 그 deferred deletion
    자체를 강제하거나 별도로 검증하지 않는다(blocking wait/
    `QCoreApplication.sendPostedEvents()` 강제 호출 없음). 이 함수가
    확인하는 것은 "QThread가 더 이상 실행 중이지 않다"는 사실뿐이다."""

    def _cleaned_up() -> bool:
        try:
            return thread.isRunning() is False
        except RuntimeError:
            return True

    qtbot.waitUntil(_cleaned_up, timeout=timeout)


def _start_and_capture(
    thread: QThread, worker: QtInferenceWorker, qtbot, timeout: int = 5000
) -> tuple[str, object]:
    """`thread.start()`를 호출하고 `finished`/`failed` 중 먼저 도착하는
    쪽을 반환한다. **`qtbot.waitSignal()`을 의도적으로 쓰지 않는다** --
    이 모듈 docstring에 적은 실측 경합(worker의 self-deleteLater()와
    `qtbot.waitSignal()`의 임시 SignalBlocker connect/disconnect가
    충돌) 때문이다. 대신 영구 연결(plain 함수, worker thread에서 직접
    실행되지만 CPython list.append는 atomic) + `qtbot.waitUntil()`
    polling만 쓴다 -- signal에 임시로 connect/disconnect하는 코드가
    이 함수 안 어디에도 없다."""
    events: list[tuple[str, object]] = []
    worker.finished.connect(lambda result: events.append(("finished", result)))
    worker.failed.connect(lambda message: events.append(("failed", message)))
    thread.start()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=timeout)
    return events[0]


def test_worker_runs_off_the_gui_thread_and_emits_finished(qtbot) -> None:
    thread_ids: dict = {}

    def fake_backend(request: InferenceRequest) -> InferenceResult:
        thread_ids["worker"] = threading.get_ident()
        return _fake_result()

    controller = InferenceController(backend=fake_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, payload = _start_and_capture(thread, worker, qtbot)

    assert kind == "finished"
    assert payload.predicted_class == "cat"
    assert thread_ids["worker"] != MAIN_THREAD_ID  # 실제로 다른 python thread에서 실행됨
    assert controller.state == "finished"

    _wait_for_thread_cleanup(thread, qtbot)


def test_worker_emits_failed_on_backend_exception(qtbot) -> None:
    def failing_backend(request: InferenceRequest) -> InferenceResult:
        raise ValueError("scratch injected failure")

    controller = InferenceController(backend=failing_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, message = _start_and_capture(thread, worker, qtbot)

    assert kind == "failed"
    assert "ValueError" in message
    assert "scratch injected failure" in message
    assert "Traceback" in message  # traceback.format_exc() 포함 확인
    assert controller.state == "failed"

    _wait_for_thread_cleanup(thread, qtbot)


def test_worker_rejects_second_run_while_first_is_active(qtbot) -> None:
    """single active run 계약: 같은 controller로 이미 begin_run()된
    상태에서 두 번째 worker.run()을 부르면 즉시 failed를 emit해야 한다."""
    controller = InferenceController(backend=lambda request: _fake_result())
    controller.begin_run()  # 이미 실행 중인 것처럼 미리 상태를 만들어 둔다

    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, message = _start_and_capture(thread, worker, qtbot)

    assert kind == "failed"
    assert "InferenceAlreadyRunningError" in message

    _wait_for_thread_cleanup(thread, qtbot)


def test_plain_function_slot_runs_on_emitting_worker_thread_not_gui_thread(qtbot) -> None:
    """Phase 5B에서 empirical하게 확인한 것과 동일한 경고를 이 새
    worker 클래스에도 다시 고정한다: `Signal`을 QObject가 아닌 평범한
    함수에 connect하면 emit이 일어난 worker thread에서 직접(동기)
    실행된다 -- GUI thread로 자동 queue되지 않는다."""
    execution_thread_ids: list[int] = []

    def plain_function_slot(result: object) -> None:
        execution_thread_ids.append(threading.get_ident())

    controller = InferenceController(backend=lambda request: _fake_result())
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    worker.finished.connect(plain_function_slot)  # 기본 AutoConnection, 영구 연결

    thread.start()
    qtbot.waitUntil(lambda: len(execution_thread_ids) == 1, timeout=5000)

    assert execution_thread_ids[0] != MAIN_THREAD_ID

    _wait_for_thread_cleanup(thread, qtbot)


class _RecordingReceiver(QObject):
    """실제 `QObject` receiver -- monkeypatch로 class attribute를
    갈아끼우지 않는다(Phase 5C에서 이 방식이 Qt의 connect() 시점
    thread-affinity 판정을 오염시킨다는 것이 확인됐다, docs/
    phase5c_training_gui_design.md §17). 진짜 subclass만 이 계약을
    정확히 검증할 수 있다."""

    def __init__(self) -> None:
        super().__init__()
        self.finished_thread_ids: list[int] = []
        self.failed_thread_ids: list[int] = []

    def on_finished(self, result: object) -> None:
        self.finished_thread_ids.append(threading.get_ident())

    def on_failed(self, message: str) -> None:
        self.failed_thread_ids.append(threading.get_ident())


def test_finished_delivered_to_real_qobject_receiver_runs_on_main_thread(qtbot) -> None:
    """`finished`를 실제 QObject bound method에 connect하면(plain 함수와
    달리) GUI/main thread로 안전하게 queue되어 그 thread에서 실행됨을
    고정한다."""
    receiver = _RecordingReceiver()
    controller = InferenceController(backend=lambda request: _fake_result())
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    worker.finished.connect(receiver.on_finished)  # 영구 연결

    thread.start()
    qtbot.waitUntil(lambda: len(receiver.finished_thread_ids) == 1, timeout=5000)

    assert receiver.finished_thread_ids == [MAIN_THREAD_ID]

    _wait_for_thread_cleanup(thread, qtbot)


def test_failed_delivered_to_real_qobject_receiver_runs_on_main_thread(qtbot) -> None:
    def failing_backend(request: InferenceRequest) -> InferenceResult:
        raise ValueError("boom")

    receiver = _RecordingReceiver()
    controller = InferenceController(backend=failing_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    worker.failed.connect(receiver.on_failed)  # 영구 연결

    thread.start()
    qtbot.waitUntil(lambda: len(receiver.failed_thread_ids) == 1, timeout=5000)

    assert receiver.failed_thread_ids == [MAIN_THREAD_ID]

    _wait_for_thread_cleanup(thread, qtbot)


def test_repeated_worker_run_and_cleanup_with_full_deletelater_contract(qtbot) -> None:
    """canonical lifecycle(§17)을 두 번 연속 실행해도 문제없어야 한다 --
    Phase 5C가 확정한 `worker.finished/failed -> worker.deleteLater`,
    `thread.finished -> thread.deleteLater` 순서를 그대로 실제로
    검증한다(fake backend, 반복 stress)."""
    controller = InferenceController(backend=lambda request: _fake_result())

    for _ in range(2):
        thread, worker = _wire_full_lifecycle(controller, _dummy_request())

        kind, _payload = _start_and_capture(thread, worker, qtbot)
        assert kind == "finished"
        assert controller.state == "finished"

        _wait_for_thread_cleanup(thread, qtbot)
