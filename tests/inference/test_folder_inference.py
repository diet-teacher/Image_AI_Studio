"""`folder_inference` 모듈 테스트(Phase 10 CP1). 실제 모델을 로드하지
않는다 -- fake backend와 임시 이미지 파일만 쓴다. CPU 전용이며 CUDA를
요구하지 않는다. single-image core(`run_single_image_inference`)의
정확성은 tests/inference/test_single_image_inference.py가 이미 고정하고
있으므로 여기서는 discovery/조립/오류 격리/집계 계약만 검증한다."""
from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
from PIL import Image

from image_ai_studio.inference.folder_inference import (
    SUPPORTED_IMAGE_EXTENSIONS,
    FolderInferenceCancelled,
    FolderInferenceError,
    FolderInferenceProgress,
    FolderInferenceRequest,
    FolderInferenceResult,
    ImageOutcome,
    discover_supported_images,
    run_folder_inference,
)
from image_ai_studio.inference.single_image_inference import (
    InferenceRequest,
    InferenceResult,
    run_single_image_inference,
)


# -- helpers -----------------------------------------------------------------


def _write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)
    return path


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "model_json_path": root / "model.json",
        "state_dict_path": root / "state_dict.pt",
        "class_mapping_path": root / "class_mapping.json",
    }


def _request(
    folder: Path, root: Path, *, device: str = "cpu", precision: str = "fp32"
) -> FolderInferenceRequest:
    return FolderInferenceRequest(
        folder_path=folder,
        device=device,
        precision=precision,
        **_artifact_paths(root),
    )


def _result(index: int = 0) -> InferenceResult:
    classes = ("cat", "dog")
    return InferenceResult(
        predicted_index=index,
        predicted_class=classes[index],
        confidence=0.75,
        probabilities={"cat": 0.75, "dog": 0.25},
        inference_duration_seconds=0.001,
    )


class RecordingBackend:
    """모든 InferenceRequest를 순서대로 기록하고 고정된 결과 객체를
    (매번 같은 인스턴스로) 돌려준다."""

    def __init__(self, result: InferenceResult | None = None) -> None:
        self.calls: list[InferenceRequest] = []
        self.result = result if result is not None else _result()

    def __call__(self, request: InferenceRequest) -> InferenceResult:
        self.calls.append(request)
        return self.result


class FailingBackend:
    """image_path.name이 fail_names에 든 이미지에서만 예외를 던지고,
    나머지는 정상 결과를 돌려준다."""

    def __init__(self, fail_names: set[str], *, exc: Exception | None = None) -> None:
        self.calls: list[InferenceRequest] = []
        self._fail_names = fail_names
        self._exc = exc

    def __call__(self, request: InferenceRequest) -> InferenceResult:
        self.calls.append(request)
        if request.image_path.name in self._fail_names:
            raise self._exc or RuntimeError(f"boom: {request.image_path.name}")
        return _result()


class CancelAfter:
    """`should_cancel` 콜백 스텁. 이미지 경계에서 관측될 때마다 호출
    횟수를 세고, `trigger_on_check`번째(0-indexed) 관측부터 True를
    돌려준다. 예: trigger_on_check=0 이면 첫 이미지 전에 곧바로 취소,
    trigger_on_check=2 이면 이미지 두 장이 끝난 뒤(세 번째 이미지 전)
    취소된다. 타이머/sleep 없이 순수 카운터로만 동작한다."""

    def __init__(self, trigger_on_check: int) -> None:
        self.checks = 0
        self._trigger = trigger_on_check

    def __call__(self) -> bool:
        fire = self.checks >= self._trigger
        self.checks += 1
        return fire


# -- discovery: order -------------------------------------------------------


def test_discovery_returns_name_sorted_order(tmp_path: Path) -> None:
    for name in ("c.png", "a.png", "b.png"):
        _write_image(tmp_path / name)
    found = discover_supported_images(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.png", "c.png"]


def test_discovery_order_independent_of_filesystem_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("b.png", "a.png", "c.png", "d.png"):
        _write_image(tmp_path / name)

    real_iterdir = Path.iterdir

    def reverse_sorted_iterdir(self: Path):  # noqa: ANN202
        return iter(sorted(real_iterdir(self), key=lambda p: p.name, reverse=True))

    monkeypatch.setattr(Path, "iterdir", reverse_sorted_iterdir)
    found = discover_supported_images(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.png", "c.png", "d.png"]


# -- discovery: extension handling ----------------------------------------


def test_discovery_includes_only_supported_extensions_case_insensitively(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "b.jpg")
    _write_image(tmp_path / "c.jpeg")
    _write_image(tmp_path / "d.BMP")  # 대문자 확장자도 포함돼야 한다
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / "data.csv").write_text("1,2,3", encoding="utf-8")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04")
    (tmp_path / "no_extension").write_bytes(b"raw")

    found = discover_supported_images(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.jpg", "c.jpeg", "d.BMP"]


def test_discovery_excludes_subdirectories(tmp_path: Path) -> None:
    _write_image(tmp_path / "top.png")
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_image(nested / "inner.png")
    (tmp_path / "looks_like.png").mkdir()  # 확장자를 가진 디렉터리

    found = discover_supported_images(tmp_path)
    assert [p.name for p in found] == ["top.png"]


def test_discovery_returns_empty_list_when_no_supported_images(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("k: v", encoding="utf-8")
    assert discover_supported_images(tmp_path) == []


def test_supported_extensions_constant_is_lowercase_dotted_and_unique() -> None:
    assert len(SUPPORTED_IMAGE_EXTENSIONS) == len(set(SUPPORTED_IMAGE_EXTENSIONS))
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        assert ext.startswith(".")
        assert ext == ext.lower()


# -- discovery: invalid folder ------------------------------------------


def test_discovery_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FolderInferenceError, match="does not exist"):
        discover_supported_images(tmp_path / "nope")


def test_discovery_rejects_non_directory_path(tmp_path: Path) -> None:
    a_file = _write_image(tmp_path / "a.png")
    with pytest.raises(FolderInferenceError, match="not a directory"):
        discover_supported_images(a_file)


# -- request mapping ----------------------------------------------------


def test_each_image_maps_to_inference_request_sharing_artifacts(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    _write_image(folder / "one.png")
    _write_image(folder / "two.jpg")
    request = _request(folder, tmp_path, device="cpu", precision="fp32")
    backend = RecordingBackend()

    run_folder_inference(request, backend=backend)

    assert len(backend.calls) == 2
    assert [c.image_path.name for c in backend.calls] == ["one.png", "two.jpg"]
    for call in backend.calls:
        assert isinstance(call, InferenceRequest)
        assert call.model_json_path == request.model_json_path
        assert call.state_dict_path == request.state_dict_path
        assert call.class_mapping_path == request.class_mapping_path
        assert call.device == request.device
        assert call.precision == request.precision


def test_default_backend_is_run_single_image_inference() -> None:
    default = inspect.signature(run_folder_inference).parameters["backend"].default
    assert default is run_single_image_inference


# -- orchestration: all success --------------------------------------


def test_all_success_preserves_exact_results_and_counts(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    for name in ("a.png", "b.png", "c.png"):
        _write_image(folder / name)
    sentinel = _result(index=1)
    backend = RecordingBackend(result=sentinel)

    aggregate = run_folder_inference(_request(folder, tmp_path), backend=backend)

    assert isinstance(aggregate, FolderInferenceResult)
    assert [item.image_path.name for item in aggregate.items] == ["a.png", "b.png", "c.png"]
    assert aggregate.total == 3
    assert aggregate.succeeded == 3
    assert aggregate.failed == 0
    for item in aggregate.items:
        assert item.succeeded is True
        assert item.error is None
        # backend가 돌려준 바로 그 객체여야 한다(예측값 재계산 없음).
        assert item.result is sentinel


# -- orchestration: mixed success / failure -------------------------


def test_mixed_success_and_failure_isolates_errors_and_continues(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    for name in ("a.png", "b.png", "c.png", "d.png"):
        _write_image(folder / name)
    backend = FailingBackend(fail_names={"b.png", "d.png"})

    aggregate = run_folder_inference(_request(folder, tmp_path), backend=backend)

    # 실패한 이미지 뒤의 이미지도 모두 backend에 도달했다.
    assert [c.image_path.name for c in backend.calls] == ["a.png", "b.png", "c.png", "d.png"]
    assert [item.image_path.name for item in aggregate.items] == ["a.png", "b.png", "c.png", "d.png"]
    assert aggregate.total == 4
    assert aggregate.succeeded == 2
    assert aggregate.failed == 2

    ok_a, fail_b, ok_c, fail_d = aggregate.items
    assert ok_a.succeeded and ok_a.result == _result() and ok_a.error is None
    assert ok_c.succeeded and ok_c.result == _result()
    for failed in (fail_b, fail_d):
        assert not failed.succeeded
        assert failed.result is None
        assert failed.error is not None
        assert "RuntimeError" in failed.error
    assert "boom: b.png" in fail_b.error
    assert "boom: d.png" in fail_d.error


# -- orchestration: all failure ------------------------------------


def test_all_failure_produces_all_failed_outcomes_without_raising(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    names = ("a.png", "b.png", "c.png")
    for name in names:
        _write_image(folder / name)
    backend = FailingBackend(fail_names=set(names))

    aggregate = run_folder_inference(_request(folder, tmp_path), backend=backend)

    assert aggregate.total == 3
    assert aggregate.succeeded == 0
    assert aggregate.failed == 3
    assert [item.image_path.name for item in aggregate.items] == list(names)
    for item in aggregate.items:
        assert not item.succeeded
        assert item.result is None
        assert item.error is not None


def test_per_image_error_message_is_bounded(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    _write_image(folder / "a.png")
    backend = FailingBackend(
        fail_names={"a.png"}, exc=RuntimeError("detail " * 5000)
    )

    aggregate = run_folder_inference(_request(folder, tmp_path), backend=backend)

    (item,) = aggregate.items
    assert not item.succeeded
    assert len(item.error) <= 500
    assert item.error.startswith("RuntimeError")
    assert item.error.endswith("...")


# -- orchestration: empty input -----------------------------------


def test_empty_folder_raises_before_backend_invocation(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    backend = RecordingBackend()

    with pytest.raises(FolderInferenceError, match="no supported images"):
        run_folder_inference(_request(folder, tmp_path), backend=backend)
    assert backend.calls == []


def test_folder_with_only_unsupported_files_raises_before_backend(tmp_path: Path) -> None:
    folder = tmp_path / "docs_only"
    folder.mkdir()
    (folder / "a.txt").write_text("x", encoding="utf-8")
    (folder / "b.pdf").write_bytes(b"%PDF-1.4")
    backend = RecordingBackend()

    with pytest.raises(FolderInferenceError, match="no supported images"):
        run_folder_inference(_request(folder, tmp_path), backend=backend)
    assert backend.calls == []


def test_missing_folder_raises_before_backend_invocation(tmp_path: Path) -> None:
    backend = RecordingBackend()
    with pytest.raises(FolderInferenceError):
        run_folder_inference(_request(tmp_path / "missing", tmp_path), backend=backend)
    assert backend.calls == []


# -- repeatability -----------------------------------------------


def test_repeated_runs_produce_identical_order_and_counts(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    for name in ("c.png", "a.jpg", "b.png", "d.jpeg"):
        _write_image(folder / name)
    request = _request(folder, tmp_path)

    assert discover_supported_images(folder) == discover_supported_images(folder)

    first = run_folder_inference(request, backend=RecordingBackend())
    second = run_folder_inference(request, backend=RecordingBackend())

    assert [i.image_path for i in first.items] == [i.image_path for i in second.items]
    assert (first.total, first.succeeded, first.failed) == (
        second.total,
        second.succeeded,
        second.failed,
    )
    assert [i.image_path.name for i in first.items] == ["a.jpg", "b.png", "c.png", "d.jpeg"]


# -- contract: immutability & existing API unchanged -------------


def test_folder_inference_request_is_frozen(tmp_path: Path) -> None:
    request = _request(tmp_path, tmp_path)
    with pytest.raises(FrozenInstanceError):
        request.device = "cuda"  # type: ignore[misc]


def test_folder_inference_result_is_frozen() -> None:
    aggregate = FolderInferenceResult(items=())
    with pytest.raises(FrozenInstanceError):
        aggregate.items = (1,)  # type: ignore[misc]


def test_image_outcome_is_frozen() -> None:
    outcome = ImageOutcome(image_path=Path("a.png"), result=_result(), error=None)
    with pytest.raises(FrozenInstanceError):
        outcome.error = "x"  # type: ignore[misc]


def test_image_outcome_requires_exactly_one_of_result_or_error() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ImageOutcome(image_path=Path("a.png"), result=None, error=None)
    with pytest.raises(ValueError, match="exactly one"):
        ImageOutcome(image_path=Path("a.png"), result=_result(), error="boom")


def test_existing_single_image_dataclasses_are_unchanged() -> None:
    assert {f.name for f in fields(InferenceRequest)} == {
        "model_json_path",
        "state_dict_path",
        "class_mapping_path",
        "image_path",
        "device",
        "precision",
    }
    assert {f.name for f in fields(InferenceResult)} == {
        "predicted_index",
        "predicted_class",
        "confidence",
        "probabilities",
        "inference_duration_seconds",
    }


# ======================================================================
# Phase 12 CP1: folder progress + cooperative-cancellation contract
# ======================================================================


def _folder(tmp_path: Path, names: tuple[str, ...]) -> Path:
    folder = tmp_path / "images"
    for name in names:
        _write_image(folder / name)
    return folder


# -- no-hook backward compatibility ------------------------------------


def test_hooks_omitted_preserves_existing_caller_behavior(tmp_path: Path) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    sentinel = _result(index=1)
    backend = RecordingBackend(result=sentinel)

    aggregate = run_folder_inference(_request(folder, tmp_path), backend=backend)

    assert isinstance(aggregate, FolderInferenceResult)
    assert type(aggregate) is FolderInferenceResult
    assert [i.image_path.name for i in aggregate.items] == ["a.png", "b.png", "c.png"]
    assert all(i.result is sentinel for i in aggregate.items)
    assert (aggregate.total, aggregate.succeeded, aggregate.failed) == (3, 3, 0)
    assert len(backend.calls) == 3


def test_new_hook_parameters_are_keyword_only_with_none_defaults() -> None:
    params = inspect.signature(run_folder_inference).parameters
    for name in ("progress_callback", "should_cancel"):
        assert name in params
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None
    # existing positional contract is untouched
    assert params["backend"].default is run_single_image_inference


def test_should_cancel_always_false_matches_no_hook_run(tmp_path: Path) -> None:
    folder = _folder(tmp_path, ("c.png", "a.jpg", "b.png", "d.jpeg"))
    request = _request(folder, tmp_path)

    baseline = run_folder_inference(request, backend=RecordingBackend())
    with_hook = run_folder_inference(
        request, backend=RecordingBackend(), should_cancel=lambda: False
    )

    assert [i.image_path for i in baseline.items] == [i.image_path for i in with_hook.items]
    assert (baseline.total, baseline.succeeded, baseline.failed) == (
        with_hook.total,
        with_hook.succeeded,
        with_hook.failed,
    )


# -- progress: initial snapshot + one monotonic snapshot per image ----


def test_progress_emits_initial_zero_then_one_snapshot_per_completed_image(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    backend = RecordingBackend()
    snaps: list[FolderInferenceProgress] = []

    run_folder_inference(
        _request(folder, tmp_path), backend=backend, progress_callback=snaps.append
    )

    assert len(snaps) == 4  # 1 initial + 3 images
    assert all(isinstance(s, FolderInferenceProgress) for s in snaps)
    assert all(s.total == 3 for s in snaps)
    assert (snaps[0].completed, snaps[0].succeeded, snaps[0].failed) == (0, 0, 0)
    assert [s.completed for s in snaps] == [0, 1, 2, 3]
    assert [s.succeeded for s in snaps] == [0, 1, 2, 3]
    assert [s.failed for s in snaps] == [0, 0, 0, 0]
    for prev, cur in zip(snaps, snaps[1:]):
        assert cur.completed == prev.completed + 1  # strictly monotonic, step 1


def test_progress_snapshots_track_mixed_success_and_isolated_failures(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png", "d.png"))
    backend = FailingBackend(fail_names={"b.png", "c.png"})
    snaps: list[FolderInferenceProgress] = []

    aggregate = run_folder_inference(
        _request(folder, tmp_path), backend=backend, progress_callback=snaps.append
    )

    assert len(snaps) == 5
    assert [s.completed for s in snaps] == [0, 1, 2, 3, 4]
    assert [s.succeeded for s in snaps] == [0, 1, 1, 1, 2]
    assert [s.failed for s in snaps] == [0, 0, 1, 2, 2]
    for s in snaps:
        assert 0 <= s.succeeded + s.failed == s.completed <= s.total
    # the aggregate is still the ordinary, unchanged result type
    assert (aggregate.total, aggregate.succeeded, aggregate.failed) == (4, 2, 2)


def test_progress_snapshot_holds_no_qt_or_mutable_inference_state() -> None:
    field_names = {f.name for f in fields(FolderInferenceProgress)}
    assert field_names == {"total", "completed", "succeeded", "failed"}
    progress = FolderInferenceProgress(total=2, completed=1, succeeded=1, failed=0)
    for value in (progress.total, progress.completed, progress.succeeded, progress.failed):
        assert isinstance(value, int)

    import image_ai_studio.inference.folder_inference as folder_module

    source = inspect.getsource(folder_module)
    assert "PySide6" not in source
    assert "PyQt" not in source
    assert "QThread" not in source


# -- progress: immutability & invariants ------------------------------


def test_folder_inference_progress_is_frozen() -> None:
    progress = FolderInferenceProgress(total=3, completed=2, succeeded=1, failed=1)
    with pytest.raises(FrozenInstanceError):
        progress.completed = 3  # type: ignore[misc]


def test_folder_inference_progress_enforces_count_invariants() -> None:
    # valid boundary values are accepted
    FolderInferenceProgress(total=0, completed=0, succeeded=0, failed=0)
    FolderInferenceProgress(total=5, completed=5, succeeded=2, failed=3)

    with pytest.raises(ValueError):  # succeeded + failed != completed
        FolderInferenceProgress(total=3, completed=2, succeeded=1, failed=0)
    with pytest.raises(ValueError):  # completed > total
        FolderInferenceProgress(total=1, completed=2, succeeded=1, failed=1)
    with pytest.raises(ValueError):  # negative count
        FolderInferenceProgress(total=3, completed=0, succeeded=-1, failed=1)
    with pytest.raises(ValueError):  # negative completed
        FolderInferenceProgress(total=3, completed=-1, succeeded=0, failed=0)


# -- cancellation: terminal value distinct from fatal failure --------


def test_folder_inference_cancelled_is_distinct_from_fatal_error() -> None:
    assert not issubclass(FolderInferenceCancelled, FolderInferenceError)
    assert not issubclass(FolderInferenceCancelled, ValueError)
    assert issubclass(FolderInferenceCancelled, Exception)


def test_cancel_before_first_image_raises_with_empty_partial_result(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    backend = RecordingBackend()
    should_cancel = CancelAfter(trigger_on_check=0)

    with pytest.raises(FolderInferenceCancelled) as excinfo:
        run_folder_inference(
            _request(folder, tmp_path), backend=backend, should_cancel=should_cancel
        )

    exc = excinfo.value
    assert backend.calls == []  # no backend call ever started
    assert isinstance(exc.result, FolderInferenceResult)
    assert exc.result.items == ()
    assert exc.discovered_total == 3
    assert exc.unprocessed == 3
    assert exc.unprocessed == exc.discovered_total - exc.result.total
    assert should_cancel.checks == 1  # observed exactly once, before the first image


def test_cancel_after_one_image_carries_exact_partial_result(tmp_path: Path) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png", "d.png", "e.png"))
    backend = RecordingBackend()
    should_cancel = CancelAfter(trigger_on_check=1)

    with pytest.raises(FolderInferenceCancelled) as excinfo:
        run_folder_inference(
            _request(folder, tmp_path), backend=backend, should_cancel=should_cancel
        )

    exc = excinfo.value
    assert [c.image_path.name for c in backend.calls] == ["a.png"]
    assert [i.image_path.name for i in exc.result.items] == ["a.png"]
    assert exc.result.items[0].succeeded is True
    assert (exc.result.total, exc.result.succeeded, exc.result.failed) == (1, 1, 0)
    assert exc.discovered_total == 5
    assert exc.unprocessed == 4


def test_cancel_after_two_images_preserves_order_and_error_values(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png", "d.png"))
    backend = FailingBackend(fail_names={"a.png"})
    should_cancel = CancelAfter(trigger_on_check=2)

    with pytest.raises(FolderInferenceCancelled) as excinfo:
        run_folder_inference(
            _request(folder, tmp_path), backend=backend, should_cancel=should_cancel
        )

    partial = excinfo.value.result
    assert [i.image_path.name for i in partial.items] == ["a.png", "b.png"]
    fail_a, ok_b = partial.items
    assert not fail_a.succeeded and fail_a.result is None and "boom: a.png" in fail_a.error
    assert ok_b.succeeded and ok_b.result == _result()
    assert (partial.total, partial.succeeded, partial.failed) == (2, 1, 1)
    assert excinfo.value.unprocessed == 2


def test_repeated_true_cancellation_is_idempotent_across_runs(tmp_path: Path) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    request = _request(folder, tmp_path)

    for _ in range(3):
        backend = RecordingBackend()
        should_cancel = CancelAfter(trigger_on_check=0)  # always True from first check
        with pytest.raises(FolderInferenceCancelled) as excinfo:
            run_folder_inference(
                request, backend=backend, should_cancel=should_cancel
            )
        assert backend.calls == []
        assert excinfo.value.result.items == ()
        assert excinfo.value.discovered_total == 3
        assert excinfo.value.unprocessed == 3
        assert should_cancel.checks == 1


def test_cancel_requested_after_final_image_returns_full_result(tmp_path: Path) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    cancel_flag = {"value": False}

    class FlipOnLastImageBackend:
        def __init__(self) -> None:
            self.calls: list[InferenceRequest] = []

        def __call__(self, request: InferenceRequest) -> InferenceResult:
            self.calls.append(request)
            if request.image_path.name == "c.png":
                # a cancellation request arrives while / right after the final
                # image runs -- there is no post-last-image boundary to observe it
                cancel_flag["value"] = True
            return _result()

    backend = FlipOnLastImageBackend()
    aggregate = run_folder_inference(
        _request(folder, tmp_path),
        backend=backend,
        should_cancel=lambda: cancel_flag["value"],
    )

    assert isinstance(aggregate, FolderInferenceResult)
    assert type(aggregate) is FolderInferenceResult
    assert [i.image_path.name for i in aggregate.items] == ["a.png", "b.png", "c.png"]
    assert (aggregate.total, aggregate.succeeded, aggregate.failed) == (3, 3, 0)
    assert cancel_flag["value"] is True  # request was made, just never observed


def test_running_backend_call_is_never_interrupted_by_cancellation(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    state = {"cancel": False}

    class RequestsCancelMidCallBackend:
        def __init__(self) -> None:
            self.calls: list[InferenceRequest] = []

        def __call__(self, request: InferenceRequest) -> InferenceResult:
            self.calls.append(request)
            if request.image_path.name == "b.png":
                state["cancel"] = True  # cancel requested *during* b's forward pass
            return _result(index=1)

    backend = RequestsCancelMidCallBackend()
    with pytest.raises(FolderInferenceCancelled) as excinfo:
        run_folder_inference(
            _request(folder, tmp_path),
            backend=backend,
            should_cancel=lambda: state["cancel"],
        )

    exc = excinfo.value
    # b's call completed and its outcome is preserved in full; it is not discarded
    assert [c.image_path.name for c in backend.calls] == ["a.png", "b.png"]
    assert [i.image_path.name for i in exc.result.items] == ["a.png", "b.png"]
    assert all(i.succeeded and i.result == _result(index=1) for i in exc.result.items)
    # c never started -- cancellation observed only at the next image boundary
    assert not any(c.image_path.name == "c.png" for c in backend.calls)
    assert exc.discovered_total == 3
    assert exc.unprocessed == 1


def test_progress_and_cancel_together_emit_no_snapshot_on_cancel(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png", "c.png"))
    backend = RecordingBackend()
    snaps: list[FolderInferenceProgress] = []
    should_cancel = CancelAfter(trigger_on_check=2)

    with pytest.raises(FolderInferenceCancelled):
        run_folder_inference(
            _request(folder, tmp_path),
            backend=backend,
            progress_callback=snaps.append,
            should_cancel=should_cancel,
        )

    # initial 0-of-total, then one per completed image (a, b) -- nothing at cancel
    assert [s.completed for s in snaps] == [0, 1, 2]
    assert [s.succeeded for s in snaps] == [0, 1, 2]


# -- callback / discovery exceptions stay fatal ----------------------


def test_progress_callback_exception_on_initial_snapshot_is_fatal(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png"))
    backend = RecordingBackend()

    class ProgressBoom(RuntimeError):
        pass

    def callback(_progress: FolderInferenceProgress) -> None:
        raise ProgressBoom("progress boom")

    with pytest.raises(ProgressBoom, match="progress boom"):
        run_folder_inference(
            _request(folder, tmp_path), backend=backend, progress_callback=callback
        )
    assert backend.calls == []  # blew up before any backend call


def test_progress_callback_exception_after_isolated_failure_is_fatal(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, ("a.png", "b.png"))
    backend = FailingBackend(fail_names={"a.png"})
    seen: list[FolderInferenceProgress] = []

    def callback(progress: FolderInferenceProgress) -> None:
        seen.append(progress)
        if progress.completed == 1:
            raise RuntimeError("post-failure boom")

    with pytest.raises(RuntimeError, match="post-failure boom"):
        run_folder_inference(
            _request(folder, tmp_path), backend=backend, progress_callback=callback
        )

    # the isolated per-image failure was recorded and *followed by* a progress
    # emission; that emission's exception is fatal (not isolated)
    assert [p.completed for p in seen] == [0, 1]
    assert (seen[1].succeeded, seen[1].failed) == (0, 1)
    assert [c.image_path.name for c in backend.calls] == ["a.png"]


def test_discovery_exception_stays_fatal_with_hooks_supplied(tmp_path: Path) -> None:
    backend = RecordingBackend()
    snaps: list[FolderInferenceProgress] = []

    with pytest.raises(FolderInferenceError):
        run_folder_inference(
            _request(tmp_path / "missing", tmp_path),
            backend=backend,
            progress_callback=snaps.append,
            should_cancel=lambda: False,
        )
    assert backend.calls == []
    assert snaps == []  # no snapshot before discovery succeeds


# -- partial cancelled result exports via the unchanged Phase 11 schema


def test_partial_cancelled_result_is_compatible_with_phase11_exporter(
    tmp_path: Path,
) -> None:
    from image_ai_studio.inference.folder_result_export import (
        EXPORT_FORMAT_VERSION,
        folder_result_to_csv_rows,
        folder_result_to_json_dict,
    )

    folder = _folder(tmp_path, ("a.png", "b.png", "c.png", "d.png"))
    backend = FailingBackend(fail_names={"b.png"})
    should_cancel = CancelAfter(trigger_on_check=3)  # process a, b, c then cancel

    with pytest.raises(FolderInferenceCancelled) as excinfo:
        run_folder_inference(
            _request(folder, tmp_path), backend=backend, should_cancel=should_cancel
        )

    partial = excinfo.value.result
    assert [i.image_path.name for i in partial.items] == ["a.png", "b.png", "c.png"]
    assert (partial.total, partial.succeeded, partial.failed) == (3, 2, 1)

    payload = folder_result_to_json_dict(partial)
    assert EXPORT_FORMAT_VERSION == 1
    assert payload["format_version"] == 1
    # exported counts describe processed items only
    assert (payload["total"], payload["succeeded"], payload["failed"]) == (3, 2, 1)
    assert [it["image_path"] for it in payload["items"]] == [
        str(i.image_path) for i in partial.items
    ]
    # cancellation / unprocessed metadata is NOT part of the export schema
    for key in ("cancelled", "unprocessed", "discovered_total"):
        assert key not in payload

    rows = folder_result_to_csv_rows(partial)
    assert len(rows) == 3
    assert [row[0] for row in rows] == [str(i.image_path) for i in partial.items]

    # the external metadata lives only on the terminal cancellation value
    assert excinfo.value.discovered_total == 4
    assert excinfo.value.unprocessed == 1
