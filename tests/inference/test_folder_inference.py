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
    FolderInferenceError,
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
