"""`QtFolderInferenceWorker`/`QThread` wiring 테스트(Phase 10 CP2).
`pytest-qt`의 `qtbot` fixture가 `QApplication`을 관리해 준다. 매
테스트마다 실제 folder inference를 돌리지 않는다 -- fake folder backend를
주입해 signal 전달/thread 분리/cleanup만 검증한다(실제 CPU inference는
Phase 10 CP4의 integration 테스트가 따로 담당한다).

`QtInferenceWorker`의 테스트 구조(Phase 6B stabilization 포함)를 그대로
가져온다:

1. 모든 테스트가 `_wire_full_lifecycle()`(worker 자신의 finished/failed에
   `deleteLater()`를 연결하는 canonical wiring)을 쓴다 -- cross-thread
   QObject 삭제를 피한다.
2. `qtbot.waitSignal()`을 쓰지 않는다 -- worker의 self-`deleteLater()`와
   `waitSignal()`의 임시 `SignalBlocker` connect/disconnect가 경합하는
   것이 Phase 6B에서 실측됐다. 대신 signal에 영구 연결한 plain 관찰자
   (`list.append`, CPython에서 atomic) + `qtbot.waitUntil()` polling을
   쓴다.

이 checkpoint의 핵심 계약도 여기서 고정한다: **per-image 실패가 섞인
aggregate 결과는 `finished`로 emit되고, `failed`는 폴더 연산 자체가
치명적 예외(`FolderInferenceError` 등)를 던졌을 때만 emit된다.**"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread

from image_ai_studio.application.folder_inference_controller import FolderInferenceController
from image_ai_studio.gui.qt_folder_inference_worker import QtFolderInferenceWorker
from image_ai_studio.inference.folder_inference import (
    FolderInferenceError,
    FolderInferenceRequest,
    FolderInferenceResult,
    ImageOutcome,
)
from image_ai_studio.inference.single_image_inference import InferenceResult

MAIN_THREAD_ID = threading.get_ident()


def _dummy_request() -> FolderInferenceRequest:
    return FolderInferenceRequest(
        model_json_path=Path("model.json"),
        state_dict_path=Path("state_dict.pt"),
        class_mapping_path=Path("class_mapping.json"),
        folder_path=Path("images"),
        device="cpu",
        precision="fp32",
    )


def _infer_result(index: int = 0) -> InferenceResult:
    classes = ("cat", "dog")
    return InferenceResult(
        predicted_index=index,
        predicted_class=classes[index],
        confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1},
        inference_duration_seconds=0.001,
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


def _wire_full_lifecycle(
    controller: FolderInferenceController, request: FolderInferenceRequest
) -> tuple[QThread, QtFolderInferenceWorker]:
    """`QtFolderInferenceWorker`의 docstring이 규정하는 canonical wiring --
    `worker.deleteLater()`는 반드시 worker 자신의 finished/failed에
    연결하고, `thread.deleteLater()`만 `thread.finished`에 연결한다
    (Phase 5C stabilization이 확정한 순서, `QtInferenceWorker`와 동일).
    이 helper를 모든 테스트가 공유해서 어떤 테스트도 이 패턴에서
    벗어나지 않게 한다."""
    thread = QThread()
    worker = QtFolderInferenceWorker(controller, request)
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
    `thread` 객체가 이미 `deleteLater()`로 삭제된 경우(`RuntimeError`)도
    정리 완료로 간주한다. blocking wait/`terminate`/busy loop 없이
    polling만 한다 -- deferred deletion 자체를 강제하지 않는다."""

    def _cleaned_up() -> bool:
        try:
            return thread.isRunning() is False
        except RuntimeError:
            return True

    qtbot.waitUntil(_cleaned_up, timeout=timeout)


def _start_and_capture(
    thread: QThread, worker: QtFolderInferenceWorker, qtbot, timeout: int = 5000
) -> tuple[str, object]:
    """`thread.start()`를 호출하고 `finished`/`failed` 중 먼저 도착하는
    쪽을 반환한다. `qtbot.waitSignal()`을 의도적으로 쓰지 않는다 --
    영구 연결(plain 함수) + `qtbot.waitUntil()` polling만 쓴다."""
    events: list[tuple[str, object]] = []
    worker.finished.connect(lambda result: events.append(("finished", result)))
    worker.failed.connect(lambda message: events.append(("failed", message)))
    thread.start()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=timeout)
    return events[0]


# -- success: off-thread execution + finished ------------------------


def test_worker_runs_off_the_gui_thread_and_emits_finished(qtbot) -> None:
    thread_ids: dict = {}
    aggregate = _all_success_aggregate()

    def fake_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        thread_ids["worker"] = threading.get_ident()
        return aggregate

    controller = FolderInferenceController(backend=fake_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, payload = _start_and_capture(thread, worker, qtbot)

    assert kind == "finished"
    assert payload is aggregate
    assert payload.total == 2 and payload.succeeded == 2
    assert thread_ids["worker"] != MAIN_THREAD_ID  # 실제로 다른 python thread에서 실행됨
    assert controller.state == "finished"

    _wait_for_thread_cleanup(thread, qtbot)


def test_worker_emits_exactly_one_finished_per_invocation(qtbot) -> None:
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    events: list[tuple[str, object]] = []
    worker.finished.connect(lambda result: events.append(("finished", result)))
    worker.failed.connect(lambda message: events.append(("failed", message)))
    thread.start()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)
    _wait_for_thread_cleanup(thread, qtbot)
    # cleanup까지 끝난 뒤에도 두 번째 emit이 없어야 한다.
    qtbot.wait(50)
    assert [kind for kind, _ in events] == ["finished"]


# -- key contract: per-image failures ride inside a finished aggregate


def test_worker_emits_finished_for_mixed_per_image_outcomes(qtbot) -> None:
    """per-image 실패가 섞인 aggregate는 `failed`가 아니라 `finished`로
    emit된다 -- worker는 backend가 정상 반환한 aggregate를 그대로
    전달할 뿐이다."""
    aggregate = _mixed_aggregate()
    controller = FolderInferenceController(backend=lambda request: aggregate)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, payload = _start_and_capture(thread, worker, qtbot)

    assert kind == "finished"
    assert payload is aggregate
    assert payload.total == 3
    assert payload.succeeded == 2
    assert payload.failed == 1
    assert controller.state == "finished"

    _wait_for_thread_cleanup(thread, qtbot)


# -- key contract: fatal exception -> failed -------------------------


def test_worker_emits_failed_on_fatal_folder_error(qtbot) -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("no supported images in folder: images")

    controller = FolderInferenceController(backend=fatal_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, message = _start_and_capture(thread, worker, qtbot)

    assert kind == "failed"
    assert "FolderInferenceError" in message
    assert "no supported images" in message
    assert "Traceback" in message  # traceback.format_exc() 포함 확인
    assert controller.state == "failed"

    _wait_for_thread_cleanup(thread, qtbot)


def test_worker_emits_failed_on_generic_fatal_exception(qtbot) -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise OSError("disk vanished mid discovery")

    controller = FolderInferenceController(backend=fatal_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, message = _start_and_capture(thread, worker, qtbot)

    assert kind == "failed"
    assert "OSError" in message
    assert "disk vanished" in message
    assert controller.state == "failed"

    _wait_for_thread_cleanup(thread, qtbot)


def test_worker_rejects_second_run_while_first_is_active(qtbot) -> None:
    """single active run 계약: 같은 controller로 이미 begin_run()된
    상태에서 두 번째 worker.run()을 부르면 즉시 failed를 emit해야 한다."""
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    controller.begin_run()  # 이미 실행 중인 것처럼 미리 상태를 만들어 둔다

    thread, worker = _wire_full_lifecycle(controller, _dummy_request())

    kind, message = _start_and_capture(thread, worker, qtbot)

    assert kind == "failed"
    assert "FolderInferenceAlreadyRunningError" in message

    _wait_for_thread_cleanup(thread, qtbot)


# -- signal delivery thread affinity -------------------------------


def test_plain_function_slot_runs_on_emitting_worker_thread_not_gui_thread(qtbot) -> None:
    """`Signal`을 QObject가 아닌 평범한 함수에 connect하면 emit이 일어난
    worker thread에서 직접(동기) 실행된다 -- GUI thread로 자동 queue되지
    않는다(Phase 5B에서 확인된 계약)."""
    execution_thread_ids: list[int] = []

    def plain_function_slot(result: object) -> None:
        execution_thread_ids.append(threading.get_ident())

    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    worker.finished.connect(plain_function_slot)  # 기본 AutoConnection, 영구 연결

    thread.start()
    qtbot.waitUntil(lambda: len(execution_thread_ids) == 1, timeout=5000)

    assert execution_thread_ids[0] != MAIN_THREAD_ID

    _wait_for_thread_cleanup(thread, qtbot)


class _RecordingReceiver(QObject):
    """실제 `QObject` receiver -- monkeypatch로 class attribute를 갈아끼우지
    않는다(Phase 5C에서 이 방식이 Qt의 connect() 시점 thread-affinity
    판정을 오염시킨다는 것이 확인됐다). 진짜 subclass만 이 계약을 정확히
    검증할 수 있다."""

    def __init__(self) -> None:
        super().__init__()
        self.finished_thread_ids: list[int] = []
        self.failed_thread_ids: list[int] = []

    def on_finished(self, result: object) -> None:
        self.finished_thread_ids.append(threading.get_ident())

    def on_failed(self, message: str) -> None:
        self.failed_thread_ids.append(threading.get_ident())


def test_finished_delivered_to_real_qobject_receiver_runs_on_main_thread(qtbot) -> None:
    receiver = _RecordingReceiver()
    controller = FolderInferenceController(backend=lambda request: _mixed_aggregate())
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    worker.finished.connect(receiver.on_finished)  # 영구 연결

    thread.start()
    qtbot.waitUntil(lambda: len(receiver.finished_thread_ids) == 1, timeout=5000)

    assert receiver.finished_thread_ids == [MAIN_THREAD_ID]

    _wait_for_thread_cleanup(thread, qtbot)


def test_failed_delivered_to_real_qobject_receiver_runs_on_main_thread(qtbot) -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("boom")

    receiver = _RecordingReceiver()
    controller = FolderInferenceController(backend=fatal_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    worker.failed.connect(receiver.on_failed)  # 영구 연결

    thread.start()
    qtbot.waitUntil(lambda: len(receiver.failed_thread_ids) == 1, timeout=5000)

    assert receiver.failed_thread_ids == [MAIN_THREAD_ID]

    _wait_for_thread_cleanup(thread, qtbot)


# -- repeated lifecycle cleanup ----------------------------------


def test_repeated_worker_run_and_cleanup_with_full_deletelater_contract(qtbot) -> None:
    """canonical lifecycle을 여러 번 연속 실행해도 stale controller/worker
    state가 남지 않아야 한다 -- success와 fatal-failure를 번갈아 돌린다
    (fake backend, 반복 stress). 매 run마다 새 worker/thread를 wiring한다."""
    outcomes: list[str] = []

    def toggling_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        if len(outcomes) % 2 == 0:
            outcomes.append("ok")
            return _mixed_aggregate()
        outcomes.append("fatal")
        raise FolderInferenceError("folder gone")

    controller = FolderInferenceController(backend=toggling_backend)

    for _ in range(2):
        thread, worker = _wire_full_lifecycle(controller, _dummy_request())
        kind, _payload = _start_and_capture(thread, worker, qtbot)
        assert kind == "finished"
        assert controller.state == "finished"
        _wait_for_thread_cleanup(thread, qtbot)

        thread, worker = _wire_full_lifecycle(controller, _dummy_request())
        kind, message = _start_and_capture(thread, worker, qtbot)
        assert kind == "failed"
        assert "FolderInferenceError" in message
        assert controller.state == "failed"
        _wait_for_thread_cleanup(thread, qtbot)

    assert outcomes == ["ok", "fatal", "ok", "fatal"]
