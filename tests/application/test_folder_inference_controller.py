"""`FolderInferenceController` 테스트(Phase 10 CP2). PySide6를 전혀
import하지 않는다 -- 이 계층은 framework-agnostic이다. 실제 folder
inference는 절대 돌리지 않고, 항상 fake folder backend를 주입해
controller의 상태 전이/backend 호출 인자만 검증한다.

`InferenceController`(단일 이미지) 테스트 구조(Phase 6B)를 그대로
참고하되, 이 checkpoint의 핵심 계약을 추가로 고정한다: **per-image
실패가 섞인 aggregate 결과를 받으면 `finished`이고, `failed`는 폴더
연산 자체가 치명적 예외를 던졌을 때만이다.**"""
from __future__ import annotations

import inspect
import threading
from pathlib import Path

import pytest

from image_ai_studio.application.folder_inference_controller import (
    FolderInferenceAlreadyRunningError,
    FolderInferenceController,
)
from image_ai_studio.inference.folder_inference import (
    FolderInferenceCancelled,
    FolderInferenceError,
    FolderInferenceProgress,
    FolderInferenceRequest,
    FolderInferenceResult,
    ImageOutcome,
    run_folder_inference,
)
from image_ai_studio.inference.single_image_inference import InferenceResult


# -- helpers ---------------------------------------------------------------


def _infer_result(index: int = 0) -> InferenceResult:
    classes = ("cat", "dog")
    return InferenceResult(
        predicted_index=index,
        predicted_class=classes[index],
        confidence=0.75,
        probabilities={"cat": 0.75, "dog": 0.25},
        inference_duration_seconds=0.001,
    )


def _dummy_request() -> FolderInferenceRequest:
    # controller/backend 계약만 테스트하므로 실제로 존재하는 경로일 필요는
    # 없다 -- fake backend는 request를 열어보지 않고 그대로 캡처만 한다.
    return FolderInferenceRequest(
        model_json_path=Path("model.json"),
        state_dict_path=Path("state_dict.pt"),
        class_mapping_path=Path("class_mapping.json"),
        folder_path=Path("images"),
        device="cpu",
        precision="fp32",
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


# -- default backend ----------------------------------------------------


def test_default_backend_is_run_folder_inference() -> None:
    default = inspect.signature(FolderInferenceController.__init__).parameters["backend"].default
    assert default is run_folder_inference


# -- success path -----------------------------------------------------


def test_controller_success_path_state_transitions_and_backend_args() -> None:
    captured: dict = {}
    aggregate = _all_success_aggregate()

    def fake_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        captured["request"] = request
        return aggregate

    controller = FolderInferenceController(backend=fake_backend)
    assert controller.state == "idle"
    assert controller.is_running is False

    controller.begin_run()
    assert controller.state == "running"
    assert controller.is_running is True

    request = _dummy_request()
    result = controller.run(request)

    assert controller.state == "finished"
    assert controller.is_running is False
    assert result is aggregate  # backend가 돌려준 바로 그 객체
    assert captured["request"] is request


def test_controller_run_before_begin_run_raises() -> None:
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    with pytest.raises(RuntimeError, match="begin_run"):
        controller.run(_dummy_request())


# -- key contract: mixed per-image outcomes still finish ---------------


def test_aggregate_with_per_image_failures_transitions_to_finished() -> None:
    """per-image 실패가 섞여 있어도 backend가 aggregate를 정상 반환하면
    controller는 `finished`다 -- `failed`가 아니다."""
    aggregate = _mixed_aggregate()
    controller = FolderInferenceController(backend=lambda request: aggregate)

    controller.begin_run()
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result is aggregate
    assert result.total == 3
    assert result.succeeded == 2
    assert result.failed == 1


def test_aggregate_with_all_images_failed_still_transitions_to_finished() -> None:
    aggregate = FolderInferenceResult(
        items=(
            ImageOutcome(image_path=Path("a.png"), result=None, error="RuntimeError: boom"),
            ImageOutcome(image_path=Path("b.png"), result=None, error="RuntimeError: boom"),
        )
    )
    controller = FolderInferenceController(backend=lambda request: aggregate)

    controller.begin_run()
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result.failed == 2
    assert result.succeeded == 0


# -- key contract: fatal exception transitions to failed --------------


def test_fatal_folder_error_transitions_to_failed_and_reraises() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("no supported images in folder: images")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()

    with pytest.raises(FolderInferenceError, match="no supported images"):
        controller.run(_dummy_request())

    assert controller.state == "failed"
    assert controller.is_running is False


def test_fatal_generic_exception_transitions_to_failed_and_reraises() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise OSError("disk vanished mid discovery")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()

    with pytest.raises(OSError, match="disk vanished"):
        controller.run(_dummy_request())

    assert controller.state == "failed"


# -- single active run -----------------------------------------------


def test_controller_rejects_second_begin_run_while_running() -> None:
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    controller.begin_run()
    with pytest.raises(FolderInferenceAlreadyRunningError):
        controller.begin_run()
    assert controller.state == "running"  # 거부돼도 기존 상태 유지


def test_controller_allows_new_begin_run_after_finished() -> None:
    controller = FolderInferenceController(backend=lambda request: _all_success_aggregate())
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


def test_controller_allows_new_begin_run_after_failed() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("boom")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()
    with pytest.raises(FolderInferenceError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    controller.begin_run()  # 예외 없이 성공해야 한다
    assert controller.state == "running"


# -- run() lifecycle invariant --------------------------------------


def test_controller_rejects_direct_run_after_finished_without_new_begin_run() -> None:
    call_count = {"n": 0}

    def counting_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        call_count["n"] += 1
        return _all_success_aggregate()

    controller = FolderInferenceController(backend=counting_backend)
    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"
    assert call_count["n"] == 1

    with pytest.raises(RuntimeError, match="running"):
        controller.run(_dummy_request())

    assert call_count["n"] == 1  # backend가 다시 호출되지 않았어야 한다
    assert controller.state == "finished"  # 거부돼도 상태는 그대로


def test_controller_rejects_direct_run_after_failed_without_new_begin_run() -> None:
    def fatal_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        raise FolderInferenceError("boom")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()
    with pytest.raises(FolderInferenceError, match="boom"):
        controller.run(_dummy_request())
    assert controller.state == "failed"

    with pytest.raises(RuntimeError, match="running"):
        controller.run(_dummy_request())

    assert controller.state == "failed"


# -- repeated success / fatal-failure runs leave no stale state -------


def test_repeated_alternating_success_and_fatal_runs_reset_cleanly() -> None:
    outcomes: list[str] = []

    def toggling_backend(request: FolderInferenceRequest) -> FolderInferenceResult:
        if len(outcomes) % 2 == 0:
            outcomes.append("ok")
            return _mixed_aggregate()
        outcomes.append("fatal")
        raise FolderInferenceError("folder gone")

    controller = FolderInferenceController(backend=toggling_backend)

    for _ in range(2):
        controller.begin_run()
        result = controller.run(_dummy_request())
        assert controller.state == "finished"
        assert result.total == 3

        controller.begin_run()
        with pytest.raises(FolderInferenceError):
            controller.run(_dummy_request())
        assert controller.state == "failed"
        assert controller.is_running is False

    assert outcomes == ["ok", "fatal", "ok", "fatal"]


# ======================================================================
# Phase 12 CP2: folder progress forwarding + cooperative cancellation
# ======================================================================


class FakeCooperativeFolderBackend:
    """`run_folder_inference`의 CP1 hook 계약을 흉내내는 fake -- 실제
    discovery/이미지 없이 미리 정한 이름 목록을 순차 처리한다. discovery
    직후 `0-of-total` 스냅샷 하나, 그 뒤 처리된 이미지마다 스냅샷 하나를
    `progress_callback`으로 내보내고, **이미지 경계**(첫 이미지 전 포함)
    에서만 `should_cancel`을 관측해 참이면 지금까지의 부분 결과와
    discovered-total을 실은 `FolderInferenceCancelled`를 던진다.

    타이머/sleep 없이 순수하게 동작한다. `gate` 콜백이 주어지면 각
    이미지 경계 직전에 `gate(index)`를 호출해 테스트가 그 지점에서
    `controller.request_cancel()`을 부를 수 있게 한다(deterministic)."""

    def __init__(self, image_names, *, fail_names=frozenset(), gate=None) -> None:
        self._names = tuple(image_names)
        self._fail_names = frozenset(fail_names)
        self._gate = gate
        self.calls: list[FolderInferenceRequest] = []
        self.processed: list[str] = []
        self.thread_ids: list[int] = []

    def __call__(
        self, request, *, progress_callback=None, should_cancel=None
    ) -> FolderInferenceResult:
        self.calls.append(request)
        self.thread_ids.append(threading.get_ident())
        total = len(self._names)
        outcomes: list[ImageOutcome] = []
        succeeded = 0
        failed = 0
        if progress_callback is not None:
            progress_callback(
                FolderInferenceProgress(total=total, completed=0, succeeded=0, failed=0)
            )
        for index, name in enumerate(self._names):
            if self._gate is not None:
                self._gate(index)
            if should_cancel is not None and should_cancel():
                raise FolderInferenceCancelled(
                    FolderInferenceResult(items=tuple(outcomes)), total
                )
            self.processed.append(name)
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


# -- new cancelled terminal state --------------------------------------


def test_controller_state_literal_includes_cancelled() -> None:
    import typing

    from image_ai_studio.application import folder_inference_controller as mod

    assert "cancelled" in typing.get_args(mod.FolderInferenceControllerState)


# -- adapter boundary: hook-aware vs one-argument backends -------------


def test_cooperative_backend_receives_progress_and_cancel_hooks() -> None:
    received: dict = {}

    def spy_backend(request, *, progress_callback=None, should_cancel=None):
        received["progress_callback"] = progress_callback
        received["should_cancel"] = should_cancel
        return _all_success_aggregate()

    controller = FolderInferenceController(backend=spy_backend)
    controller.begin_run()
    my_callback = lambda snap: None  # noqa: E731
    controller.run(_dummy_request(), progress_callback=my_callback)

    # the caller's callback is forwarded verbatim -- no wrapper
    assert received["progress_callback"] is my_callback
    # should_cancel is a thread-safe zero-arg predicate over the run's flag
    assert callable(received["should_cancel"])
    assert received["should_cancel"]() is False
    assert controller.state == "finished"


def test_one_argument_backend_still_supported_without_hook_params() -> None:
    """기존 1-인자 folder backend는 progress/cancel 파라미터를 요구하지
    않고도 그대로 동작한다 -- controller가 `request` 하나로만 호출한다."""
    calls: list = []

    def legacy_backend(request):  # 단일 파라미터, keyword hook 없음
        calls.append(request)
        return _all_success_aggregate()

    controller = FolderInferenceController(backend=legacy_backend)
    controller.begin_run()
    # progress_callback을 넘겨도 1-인자 backend는 그것을 절대 보지 않는다
    result = controller.run(_dummy_request(), progress_callback=lambda snap: None)

    assert len(calls) == 1
    assert controller.state == "finished"
    assert result.total == 2


def test_one_argument_backend_run_completes_even_with_cancel_requested() -> None:
    """1-인자 backend는 취소를 관측할 수 없다 -- request_cancel()을 해도
    run은 정상적으로 끝나고 `finished`가 된다(오류 문자열 파싱 없음)."""
    controller = FolderInferenceController(
        backend=lambda request: _all_success_aggregate()
    )
    controller.begin_run()
    controller.request_cancel()
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result.total == 2


def test_default_backend_is_detected_as_cooperative() -> None:
    controller = FolderInferenceController()  # 기본 backend = run_folder_inference
    assert controller._backend_supports_hooks is True


# -- progress forwarding ------------------------------------------------


def test_run_forwards_progress_snapshots_in_discovery_order() -> None:
    backend = FakeCooperativeFolderBackend(
        ("a.png", "b.png", "c.png"), fail_names={"b.png"}
    )
    snaps: list[FolderInferenceProgress] = []

    controller = FolderInferenceController(backend=backend)
    controller.begin_run()
    result = controller.run(_dummy_request(), progress_callback=snaps.append)

    assert controller.state == "finished"
    assert [s.completed for s in snaps] == [0, 1, 2, 3]
    assert [s.succeeded for s in snaps] == [0, 1, 1, 2]
    assert [s.failed for s in snaps] == [0, 0, 1, 1]
    for prev, cur in zip(snaps, snaps[1:]):
        assert cur.completed == prev.completed + 1
    assert (result.total, result.succeeded, result.failed) == (3, 2, 1)


def test_run_without_progress_callback_leaves_backend_hook_none() -> None:
    received: dict = {}

    def spy_backend(request, *, progress_callback="unset", should_cancel=None):
        received["progress_callback"] = progress_callback
        return _all_success_aggregate()

    controller = FolderInferenceController(backend=spy_backend)
    controller.begin_run()
    controller.run(_dummy_request())  # no progress_callback

    assert received["progress_callback"] is None


# -- cooperative cancellation: distinct terminal state ----------------


def test_cancel_before_first_image_transitions_to_cancelled_with_empty_partial() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png", "c.png"))
    controller = FolderInferenceController(backend=backend)
    controller.begin_run()
    controller.request_cancel()

    with pytest.raises(FolderInferenceCancelled) as excinfo:
        controller.run(_dummy_request())

    assert controller.state == "cancelled"
    assert controller.is_running is False
    exc = excinfo.value
    assert exc.result.items == ()
    assert exc.discovered_total == 3
    assert exc.unprocessed == 3
    assert backend.processed == []  # no image ever started


def test_cancel_after_progress_preserves_exact_partial_result() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png", "c.png", "d.png"))
    controller = FolderInferenceController(backend=backend)
    snaps: list[FolderInferenceProgress] = []

    def observer(snap: FolderInferenceProgress) -> None:
        snaps.append(snap)
        if snap.completed == 2:  # after two images done, ask to cancel
            controller.request_cancel()

    controller.begin_run()
    with pytest.raises(FolderInferenceCancelled) as excinfo:
        controller.run(_dummy_request(), progress_callback=observer)

    assert controller.state == "cancelled"
    exc = excinfo.value
    assert [i.image_path.name for i in exc.result.items] == ["a.png", "b.png"]
    assert (exc.result.total, exc.result.succeeded, exc.result.failed) == (2, 2, 0)
    assert exc.discovered_total == 4
    assert exc.unprocessed == 2
    assert backend.processed == ["a.png", "b.png"]
    # no progress snapshot is emitted for the cancelled boundary
    assert [s.completed for s in snaps] == [0, 1, 2]


def test_cancel_after_progress_carries_isolated_per_image_errors() -> None:
    backend = FakeCooperativeFolderBackend(
        ("a.png", "b.png", "c.png", "d.png"), fail_names={"a.png"}
    )
    controller = FolderInferenceController(backend=backend)

    def observer(snap: FolderInferenceProgress) -> None:
        if snap.completed == 2:
            controller.request_cancel()

    controller.begin_run()
    with pytest.raises(FolderInferenceCancelled) as excinfo:
        controller.run(_dummy_request(), progress_callback=observer)

    partial = excinfo.value.result
    assert [i.image_path.name for i in partial.items] == ["a.png", "b.png"]
    fail_a, ok_b = partial.items
    assert not fail_a.succeeded and fail_a.result is None and "boom: a.png" in fail_a.error
    assert ok_b.succeeded and ok_b.result is not None
    assert (partial.total, partial.succeeded, partial.failed) == (2, 1, 1)
    assert excinfo.value.unprocessed == 2


def test_duplicate_request_cancel_is_idempotent() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png", "c.png"))
    controller = FolderInferenceController(backend=backend)
    controller.begin_run()
    for _ in range(5):
        controller.request_cancel()  # repeated, must not raise or block
    assert controller.cancel_requested is True

    with pytest.raises(FolderInferenceCancelled) as excinfo:
        controller.run(_dummy_request())

    assert controller.state == "cancelled"
    assert excinfo.value.result.items == ()
    assert backend.processed == []
    # still safe to call again after the terminal state
    controller.request_cancel()
    assert controller.state == "cancelled"


def test_request_cancel_while_idle_is_a_noop_for_the_next_run() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png"))
    controller = FolderInferenceController(backend=backend)
    controller.request_cancel()  # no begin_run() yet

    controller.begin_run()  # clears the flag
    assert controller.cancel_requested is False
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result.total == 2
    assert backend.processed == ["a.png", "b.png"]


# -- cooperative cancellation vs fatal failure (type, not string) -----


def test_fatal_error_transitions_to_failed_even_with_cancel_requested() -> None:
    def fatal_backend(request, *, progress_callback=None, should_cancel=None):
        raise FolderInferenceError("no supported images in folder: images")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()
    controller.request_cancel()  # a cancel is pending, but the failure is fatal

    with pytest.raises(FolderInferenceError, match="no supported images"):
        controller.run(_dummy_request())

    assert controller.state == "failed"  # not 'cancelled'
    assert controller.is_running is False


def test_generic_exception_transitions_to_failed_not_cancelled() -> None:
    def fatal_backend(request, *, progress_callback=None, should_cancel=None):
        raise OSError("disk vanished mid discovery")

    controller = FolderInferenceController(backend=fatal_backend)
    controller.begin_run()
    controller.request_cancel()

    with pytest.raises(OSError, match="disk vanished"):
        controller.run(_dummy_request())

    assert controller.state == "failed"


# -- rerun after cancelled; stale cancel cannot cancel the rerun ------


def test_begin_run_allowed_after_cancelled() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png"))
    controller = FolderInferenceController(backend=backend)
    controller.begin_run()
    controller.request_cancel()
    with pytest.raises(FolderInferenceCancelled):
        controller.run(_dummy_request())
    assert controller.state == "cancelled"

    controller.begin_run()  # must not raise
    assert controller.state == "running"


def test_rerun_after_cancelled_clears_flag_and_completes_fully() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png", "c.png"))
    controller = FolderInferenceController(backend=backend)

    controller.begin_run()
    controller.request_cancel()
    with pytest.raises(FolderInferenceCancelled):
        controller.run(_dummy_request())
    assert controller.state == "cancelled"
    assert backend.processed == []

    # a stale cancel request lands *after* the terminal state
    controller.request_cancel()
    # the new run's begin_run() must discard it
    controller.begin_run()
    assert controller.cancel_requested is False
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result.total == 3
    assert backend.processed == ["a.png", "b.png", "c.png"]  # 2nd run only
    assert len(backend.calls) == 2


def test_rerun_after_finished_discards_late_cancel_request() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png"))
    controller = FolderInferenceController(backend=backend)

    controller.begin_run()
    controller.run(_dummy_request())
    assert controller.state == "finished"

    controller.request_cancel()  # stale, against a finished controller
    controller.begin_run()
    result = controller.run(_dummy_request())

    assert controller.state == "finished"
    assert result.total == 2


def test_repeated_cancel_runs_leave_no_stale_state() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png", "c.png"))
    controller = FolderInferenceController(backend=backend)

    for _ in range(3):
        controller.begin_run()
        controller.request_cancel()
        with pytest.raises(FolderInferenceCancelled) as excinfo:
            controller.run(_dummy_request())
        assert controller.state == "cancelled"
        assert excinfo.value.result.items == ()
        assert excinfo.value.discovered_total == 3

    assert backend.processed == []
    assert len(backend.calls) == 3


# -- run() lifecycle invariant still holds for cancelled --------------


def test_run_after_cancelled_without_new_begin_run_raises() -> None:
    backend = FakeCooperativeFolderBackend(("a.png", "b.png"))
    controller = FolderInferenceController(backend=backend)
    controller.begin_run()
    controller.request_cancel()
    with pytest.raises(FolderInferenceCancelled):
        controller.run(_dummy_request())
    assert controller.state == "cancelled"

    with pytest.raises(RuntimeError, match="running"):
        controller.run(_dummy_request())
    assert controller.state == "cancelled"  # rejected call leaves state intact


def test_existing_no_cancel_begin_run_behavior_is_unchanged() -> None:
    """request_cancel()을 전혀 부르지 않으면 begin/run 동작은 CP2 이전과
    똑같다 -- 성공 aggregate는 finished, 치명적 예외는 failed."""
    controller = FolderInferenceController(
        backend=lambda request: _mixed_aggregate()
    )
    controller.begin_run()
    result = controller.run(_dummy_request())
    assert controller.state == "finished"
    assert (result.total, result.succeeded, result.failed) == (3, 2, 1)
