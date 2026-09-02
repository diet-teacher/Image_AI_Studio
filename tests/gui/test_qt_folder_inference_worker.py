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
    FolderInferenceCancelled,
    FolderInferenceError,
    FolderInferenceProgress,
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
    worker.cancelled.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.cancelled.connect(worker.deleteLater)
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


# ======================================================================
# Phase 12 CP2: progress / cooperative-cancellation signals
# ======================================================================


class GatedCooperativeBackend:
    """`run_folder_inference`의 CP1 hook 계약을 흉내내는 fake(worker
    thread에서 실행됨). discovery 직후 `0-of-total` 스냅샷 하나, 처리된
    이미지마다 스냅샷 하나를 `progress_callback`으로 내보내고, **이미지
    경계**에서만 `should_cancel`을 관측한다.

    deterministic 동기화: `gate_before`(loop index)에 도달하면 그 경계의
    `should_cancel` 관측 **직전에** `paused`를 set하고 `resume`를
    기다린다 -- 테스트(GUI thread)는 그 사이에 `worker.request_cancel()`
    을 호출할 수 있다. sleep/busy loop 없이 `threading.Event`로만
    동기화한다. `gate_before=None`이면 게이트 없이 끝까지 처리한다."""

    def __init__(
        self,
        image_names,
        *,
        gate_before=None,
        paused: threading.Event | None = None,
        resume: threading.Event | None = None,
        fail_names=frozenset(),
    ) -> None:
        self._names = tuple(image_names)
        self.gate_before = gate_before
        self._paused = paused
        self._resume = resume
        self._fail_names = frozenset(fail_names)
        self.calls: list[FolderInferenceRequest] = []
        self.worker_thread_id: int | None = None

    def __call__(
        self, request, *, progress_callback=None, should_cancel=None
    ) -> FolderInferenceResult:
        self.calls.append(request)
        self.worker_thread_id = threading.get_ident()
        total = len(self._names)
        outcomes: list[ImageOutcome] = []
        succeeded = 0
        failed = 0
        if progress_callback is not None:
            progress_callback(
                FolderInferenceProgress(total=total, completed=0, succeeded=0, failed=0)
            )
        for index, name in enumerate(self._names):
            if index == self.gate_before:
                if self._paused is not None:
                    self._paused.set()
                if self._resume is not None:
                    self._resume.wait(timeout=5)
            if should_cancel is not None and should_cancel():
                raise FolderInferenceCancelled(
                    FolderInferenceResult(items=tuple(outcomes)), total
                )
            if name in self._fail_names:
                outcomes.append(
                    ImageOutcome(
                        image_path=Path(name),
                        result=None,
                        error=f"RuntimeError: boom: {name}",
                    )
                )
                failed += 1
            else:
                outcomes.append(
                    ImageOutcome(image_path=Path(name), result=_infer_result(0), error=None)
                )
                succeeded += 1
            if progress_callback is not None:
                progress_callback(
                    FolderInferenceProgress(
                        total=total,
                        completed=len(outcomes),
                        succeeded=succeeded,
                        failed=failed,
                    )
                )
        return FolderInferenceResult(items=tuple(outcomes))


def _connect_all(worker: QtFolderInferenceWorker) -> tuple[list, list]:
    """terminal signal(finished/cancelled/failed)과 progress를 각각 영구
    연결(plain 함수, `list.append`는 CPython에서 atomic)한다. `qtbot.
    waitSignal()`은 쓰지 않는다 -- 기존 테스트와 동일한 이유."""
    events: list[tuple] = []
    snaps: list[FolderInferenceProgress] = []
    worker.progress.connect(lambda snap: snaps.append(snap))
    worker.finished.connect(lambda result: events.append(("finished", result)))
    worker.cancelled.connect(
        lambda result, total: events.append(("cancelled", result, total))
    )
    worker.failed.connect(lambda message: events.append(("failed", message)))
    return events, snaps


# -- signals exist ----------------------------------------------------


def test_worker_exposes_progress_and_cancelled_signals() -> None:
    for name in ("progress", "finished", "cancelled", "failed"):
        assert hasattr(QtFolderInferenceWorker, name)


def test_prepare_run_is_idempotent_and_run_reuses_prepared_token() -> None:
    begin_calls = 0

    class CountingController(FolderInferenceController):
        def begin_run(self) -> None:
            nonlocal begin_calls
            begin_calls += 1
            super().begin_run()

    def cooperative_backend(request, *, progress_callback=None, should_cancel=None):
        if should_cancel is not None and should_cancel():
            raise FolderInferenceCancelled(FolderInferenceResult(items=()), 2)
        return _all_success_aggregate()

    controller = CountingController(backend=cooperative_backend)
    worker = QtFolderInferenceWorker(controller, _dummy_request())
    events, _snaps = _connect_all(worker)

    worker.prepare_run()
    worker.prepare_run()
    worker.request_cancel()
    worker.run()

    assert begin_calls == 1
    assert [event[0] for event in events] == ["cancelled"]
    assert controller.state == "cancelled"


# -- progress ordering + finished terminal --------------------------


def test_worker_emits_ordered_progress_then_finished(qtbot) -> None:
    backend = GatedCooperativeBackend(
        ("a.png", "b.png", "c.png"), fail_names={"b.png"}
    )
    controller = FolderInferenceController(backend=backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    events, snaps = _connect_all(worker)

    thread.start()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)

    assert events[0][0] == "finished"
    # progress snapshots arrived (in order) before the terminal signal
    assert [s.completed for s in snaps] == [0, 1, 2, 3]
    assert [s.succeeded for s in snaps] == [0, 1, 1, 2]
    assert [s.failed for s in snaps] == [0, 0, 1, 1]
    assert backend.worker_thread_id not in (None, MAIN_THREAD_ID)
    assert controller.state == "finished"

    _wait_for_thread_cleanup(thread, qtbot)


# -- cancel before the first image ---------------------------------


def test_request_cancel_before_first_image_emits_cancelled_empty(qtbot) -> None:
    paused, resume = threading.Event(), threading.Event()
    backend = GatedCooperativeBackend(
        ("a.png", "b.png", "c.png"), gate_before=0, paused=paused, resume=resume
    )
    controller = FolderInferenceController(backend=backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    events, snaps = _connect_all(worker)

    thread.start()
    qtbot.waitUntil(paused.is_set, timeout=5000)  # backend gated before image 0
    worker.request_cancel()  # GUI thread, non-blocking
    resume.set()

    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)

    kind, partial, discovered_total = events[0]
    assert kind == "cancelled"
    assert partial.items == ()
    assert discovered_total == 3
    assert [s.completed for s in snaps] == [0]  # only the initial 0-of-total
    assert controller.state == "cancelled"

    _wait_for_thread_cleanup(thread, qtbot)


# -- cancel after progress; partial result preserved -------------


def test_request_cancel_after_progress_emits_cancelled_with_partial(qtbot) -> None:
    paused, resume = threading.Event(), threading.Event()
    backend = GatedCooperativeBackend(
        ("a.png", "b.png", "c.png", "d.png"),
        gate_before=2,
        paused=paused,
        resume=resume,
    )
    controller = FolderInferenceController(backend=backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    events, snaps = _connect_all(worker)

    thread.start()
    qtbot.waitUntil(paused.is_set, timeout=5000)  # 2 images processed, gated
    worker.request_cancel()
    worker.request_cancel()  # duplicate -- idempotent, must not block
    resume.set()

    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)

    kind, partial, discovered_total = events[0]
    assert kind == "cancelled"
    assert [i.image_path.name for i in partial.items] == ["a.png", "b.png"]
    assert (partial.total, partial.succeeded, partial.failed) == (2, 2, 0)
    assert discovered_total == 4
    # ordered progress: initial + one per completed image, nothing at cancel
    assert [s.completed for s in snaps] == [0, 1, 2]
    assert controller.state == "cancelled"

    # exactly one terminal signal, even after cleanup settles
    _wait_for_thread_cleanup(thread, qtbot)
    qtbot.wait(50)
    assert [e[0] for e in events] == ["cancelled"]


# -- fatal failure stays 'failed', never 'cancelled' -------------


def test_worker_emits_failed_not_cancelled_on_fatal_with_cancel_pending(qtbot) -> None:
    def fatal_backend(request, *, progress_callback=None, should_cancel=None):
        raise FolderInferenceError("no supported images in folder: images")

    controller = FolderInferenceController(backend=fatal_backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    events, _snaps = _connect_all(worker)

    thread.start()
    worker.request_cancel()  # racing cancel request; outcome must still be fatal

    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)

    kind, message = events[0]
    assert kind == "failed"
    assert "FolderInferenceError" in message
    assert "Traceback" in message
    assert controller.state == "failed"

    _wait_for_thread_cleanup(thread, qtbot)


# -- rerun after cancelled; stale cancel cannot cancel the rerun --


def test_worker_reruns_after_cancelled_with_fresh_flag(qtbot) -> None:
    paused, resume = threading.Event(), threading.Event()
    backend = GatedCooperativeBackend(
        ("a.png", "b.png", "c.png"), gate_before=0, paused=paused, resume=resume
    )
    controller = FolderInferenceController(backend=backend)

    # run 1 -- cancel before the first image
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    events, _snaps = _connect_all(worker)
    thread.start()
    qtbot.waitUntil(paused.is_set, timeout=5000)
    worker.request_cancel()
    resume.set()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)
    assert events[0][0] == "cancelled"
    assert controller.state == "cancelled"
    _wait_for_thread_cleanup(thread, qtbot)

    # run 2 -- same controller, no gate, a stale cancel request beforehand
    backend.gate_before = None
    resume.set()
    thread2, worker2 = _wire_full_lifecycle(controller, _dummy_request())
    events2, _snaps2 = _connect_all(worker2)
    worker2.request_cancel()  # stale: lands before the rerun's begin_run()
    thread2.start()
    qtbot.waitUntil(lambda: len(events2) == 1, timeout=5000)

    assert events2[0][0] == "finished"
    result = events2[0][1]
    assert [i.image_path.name for i in result.items] == ["a.png", "b.png", "c.png"]
    assert controller.state == "finished"
    assert len(backend.calls) == 2

    _wait_for_thread_cleanup(thread2, qtbot)


# -- request_cancel is a non-blocking no-op when nothing runs ----


def test_request_cancel_is_safe_before_and_after_the_run(qtbot) -> None:
    backend = GatedCooperativeBackend(("a.png", "b.png"))
    controller = FolderInferenceController(backend=backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    events, _snaps = _connect_all(worker)

    worker.request_cancel()  # before thread.start() -- discarded by begin_run()
    worker.request_cancel()  # repeated -- idempotent, non-blocking
    thread.start()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)
    assert events[0][0] == "finished"
    assert controller.state == "finished"

    _wait_for_thread_cleanup(thread, qtbot)
    # after the run the controller-level flag is still safe to poke
    controller.request_cancel()
    assert controller.state == "finished"


# -- direct worker QObject destruction observation ---------------


def test_worker_qobject_destroyed_after_cancelled_via_canonical_wiring(qtbot) -> None:
    """canonical wiring(`worker.cancelled -> worker.deleteLater`)이 취소
    경로에서도 실제로 worker QObject를 파기하는지 `destroyed` signal로
    직접 관측한다."""
    paused, resume = threading.Event(), threading.Event()
    backend = GatedCooperativeBackend(
        ("a.png", "b.png", "c.png"), gate_before=1, paused=paused, resume=resume
    )
    controller = FolderInferenceController(backend=backend)
    thread, worker = _wire_full_lifecycle(controller, _dummy_request())
    events, _snaps = _connect_all(worker)
    destroyed: list[bool] = []
    worker.destroyed.connect(lambda *_: destroyed.append(True))

    thread.start()
    qtbot.waitUntil(paused.is_set, timeout=5000)
    worker.request_cancel()
    resume.set()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=5000)
    assert events[0][0] == "cancelled"

    qtbot.waitUntil(lambda: destroyed == [True], timeout=5000)
    _wait_for_thread_cleanup(thread, qtbot)
