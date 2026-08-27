"""training/artifact_io.py 테스트 (Phase 8 checkpoint 1).

내부 원자적 아티팩트 I/O primitive(``atomic_write_text`` /
``atomic_torch_save``)의 계약을 focused fault injection으로 검증한다:
성공 round-trip, 기존 목적지 보존(os.replace 실패 / serializer 실패),
helper 임시 파일 정리(정리 실패가 원래 예외를 가리지 않음), Unicode
텍스트, 그리고 shell=False / network-free 특성. 전부 CPU 전용이며
``tmp_path`` 아래 임시 디렉터리만 사용한다 -- CUDA/모델/외부 네트워크/
Bash/Git 없음.

checkpoint.py의 ``_atomic_torch_save`` 테스트(test_checkpoint.py의
Phase 4J 섹션)와 동일한 fault-injection 스타일을 따른다.
"""
from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import pytest
import torch

from image_ai_studio.training import artifact_io
from image_ai_studio.training.artifact_io import atomic_torch_save, atomic_write_text

pytestmark = pytest.mark.phase8_cp1_atomic_artifact_io_primitives


# -- 성공 round-trip ---------------------------------------------------------


def test_atomic_write_text_round_trips_through_read_text(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    atomic_write_text("hello world\n", path)

    assert path.read_text(encoding="utf-8") == "hello world\n"


def test_atomic_write_text_unicode_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "unicode.txt"
    payload = "한글 · emoji 😀 · symbols ½∑ · 日本語\n"
    atomic_write_text(payload, path)

    assert path.read_text(encoding="utf-8") == payload


def test_atomic_torch_save_round_trips_through_torch_load(tmp_path: Path) -> None:
    path = tmp_path / "payload.pt"
    obj = {"weight": torch.arange(6, dtype=torch.float32), "step": 3}
    atomic_torch_save(obj, path)

    loaded = torch.load(path, weights_only=True)
    assert loaded["step"] == 3
    assert torch.equal(loaded["weight"], torch.arange(6, dtype=torch.float32))


def test_atomic_write_text_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text("nested", nested)

    assert nested.read_text(encoding="utf-8") == "nested"


def test_atomic_torch_save_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c.pt"
    atomic_torch_save({"value": 1}, nested)

    assert torch.load(nested, weights_only=True) == {"value": 1}


def test_successful_publication_leaves_no_helper_temp_file(tmp_path: Path) -> None:
    text_path = tmp_path / "note.txt"
    pt_path = tmp_path / "payload.pt"
    atomic_write_text("done", text_path)
    atomic_torch_save({"value": 2}, pt_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["note.txt", "payload.pt"]


def test_publication_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "as_str.txt"
    atomic_write_text("via str", str(path))

    assert path.read_text(encoding="utf-8") == "via str"


# -- 임시 파일이 목적지 디렉터리에 만들어지고 폴백하지 않음 -----------------


def test_temp_file_is_created_in_destination_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """helper는 목적지와 같은 디렉터리에만 임시 파일을 만든다 -- 다른
    디렉터리로 폴백하지 않는다."""
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    real_mkstemp = tempfile.mkstemp
    seen = {}

    def spy_mkstemp(*args, **kwargs):
        seen["dir"] = kwargs.get("dir")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(artifact_io.tempfile, "mkstemp", spy_mkstemp)
    atomic_write_text("x", dest_dir / "file.txt")

    assert seen["dir"] is not None
    assert Path(seen["dir"]).resolve() == dest_dir.resolve()


# -- os.replace 실패: 기존 목적지 보존 -------------------------------------


def test_replace_failure_preserves_existing_text_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "note.txt"
    atomic_write_text("original", path)
    original_bytes = path.read_bytes()

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(artifact_io.os, "replace", failing_replace)

    with pytest.raises(OSError, match="permission denied"):
        atomic_write_text("replacement", path)

    assert path.read_bytes() == original_bytes  # byte-for-byte 보존
    assert list(tmp_path.iterdir()) == [path]  # helper 임시 파일 미잔존


def test_replace_failure_preserves_existing_torch_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.pt"
    atomic_torch_save({"value": "original"}, path)
    original_bytes = path.read_bytes()

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr(artifact_io.os, "replace", failing_replace)

    with pytest.raises(OSError, match="cross-device link"):
        atomic_torch_save({"value": "new"}, path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]
    assert torch.load(path, weights_only=True) == {"value": "original"}


# -- serializer 실패: 기존 목적지 보존 -----------------------------------


def test_torch_serializer_failure_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torch.save() 자체가 실패하면(게시 이전) 기존 파일은 전혀 바뀌지
    않고, helper 임시 파일도 남지 않는다."""
    path = tmp_path / "payload.pt"
    atomic_torch_save({"value": "original"}, path)
    original_bytes = path.read_bytes()

    def failing_torch_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(artifact_io.torch, "save", failing_torch_save)

    with pytest.raises(RuntimeError, match="disk full"):
        atomic_torch_save({"value": "new"}, path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_text_serializer_failure_preserves_existing_destination(tmp_path: Path) -> None:
    """인코딩(직렬화)이 실패하면(예: ascii 코덱 + 비-ascii 문자) 기존
    파일은 전혀 바뀌지 않고, helper 임시 파일도 남지 않는다."""
    path = tmp_path / "note.txt"
    atomic_write_text("original ascii", path)
    original_bytes = path.read_bytes()

    with pytest.raises(UnicodeEncodeError):
        atomic_write_text("비-ascii 문자", path, encoding="ascii")

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


# -- 임시 파일 정리(cleanup) --------------------------------------------------


def test_cleanup_failure_does_not_mask_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """임시 파일 정리(unlink) 자체가 실패해도, 사용자에게 보이는 예외는
    원래 직렬화 실패 예외여야 한다."""
    path = tmp_path / "payload.pt"

    def failing_torch_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("original failure")

    def failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("cleanup also failed")

    monkeypatch.setattr(artifact_io.torch, "save", failing_torch_save)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(RuntimeError, match="original failure"):
        atomic_torch_save({"value": "new"}, path)


def test_serializer_failure_without_existing_destination_leaves_directory_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.pt"

    def failing_torch_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(artifact_io.torch, "save", failing_torch_save)

    with pytest.raises(RuntimeError, match="boom"):
        atomic_torch_save({"value": "new"}, path)

    assert list(tmp_path.iterdir()) == []  # 실패한 partial 파일 미잔존


# -- 미리 존재하던 무관한 파일은 건드리지 않음 -----------------------------


def test_does_not_touch_unrelated_preexisting_files_on_success(tmp_path: Path) -> None:
    bystander = tmp_path / "unrelated.bin"
    bystander.write_bytes(b"\x00keep me\x00")

    atomic_write_text("new content", tmp_path / "target.txt")

    assert bystander.read_bytes() == b"\x00keep me\x00"


def test_does_not_touch_unrelated_preexisting_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bystander = tmp_path / "unrelated.bin"
    bystander.write_bytes(b"bystander")

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("nope")

    monkeypatch.setattr(artifact_io.os, "replace", failing_replace)

    with pytest.raises(OSError, match="nope"):
        atomic_write_text("new", tmp_path / "target.txt")

    assert bystander.read_bytes() == b"bystander"


# -- shell=False / network-free / 단일 파일 범위 -------------------------------


def test_module_does_not_use_shell_subprocess_or_network() -> None:
    """primitive는 순수 파일시스템 연산만 한다 -- subprocess/shell/소켓/
    HTTP 클라이언트를 쓰지 않는다(shell=False, network-free)."""
    source = inspect.getsource(artifact_io)
    for banned in (
        "subprocess",
        "Popen",
        "os.system",
        "shell",
        "socket",
        "urllib",
        "requests",
        "http",
    ):
        assert banned not in source, f"artifact_io must not reference {banned!r}"


def test_module_surface_is_minimal_and_internal() -> None:
    """이 checkpoint는 primitive 2개만 노출한다 -- 기존 writer(save_model_spec
    등) 이관이나 다중 파일 트랜잭션 API는 없다."""
    assert set(artifact_io.__all__) == {"atomic_write_text", "atomic_torch_save"}
    assert not hasattr(artifact_io, "save_model_spec")
    assert not hasattr(artifact_io, "save_class_mapping")
    assert not hasattr(artifact_io, "save_state_dict")
    for name in dir(artifact_io):
        assert "transaction" not in name.lower()
        assert "bundle" not in name.lower()
