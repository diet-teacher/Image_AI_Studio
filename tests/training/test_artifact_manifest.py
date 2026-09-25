"""training/artifact_manifest.py 테스트 (Phase 15 checkpoint 1).

이식 가능 아티팩트 3종(model_definition.json / best_model_state_dict.pt /
class_mapping.json)을 위한 결정적 무결성 매니페스트의 계약을 검증한다:
결정적 직렬화(반복 write 시 바이트 동일), 성공적인 build/write/load/
verify round-trip, 바이트 변조/크기 변조 탐지, 파일 누락, 손상되거나
버전이 안 맞는 매니페스트 거부, 안전하지 않은 경로/심볼릭 링크 거부,
청크 단위 스트리밍 해싱, 그리고 atomic_write_text 실패 시 기존
매니페스트 보존까지 다룬다. 전부 로컬 결정적 바이트와 tmp_path만
사용한다 -- CUDA/네트워크/실제 모델 학습/sleep 없음.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from image_ai_studio.training import artifact_manifest
from image_ai_studio.training.artifact_manifest import (
    ARTIFACT_MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ArtifactEntry,
    ArtifactManifest,
    ManifestError,
    build_manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)

_MODEL_DEFINITION_BYTES = b'{"architecture": "resnet18", "num_classes": 3}'
_STATE_DICT_BYTES = b"\x80\x02}q\x00X\x06\x00\x00\x00weightq\x01K\x01s." * 37
_CLASS_MAPPING_BYTES = b'{"classes": ["cat", "dog", "bird"]}'


def _make_bundle(bundle_dir: Path) -> None:
    (bundle_dir / "model_definition.json").write_bytes(_MODEL_DEFINITION_BYTES)
    (bundle_dir / "best_model_state_dict.pt").write_bytes(_STATE_DICT_BYTES)
    (bundle_dir / "class_mapping.json").write_bytes(_CLASS_MAPPING_BYTES)


# -- build_manifest(): 성공 및 결정적 직렬화 ---------------------------------


def test_build_manifest_records_canonical_filenames_sizes_and_sha256(tmp_path: Path) -> None:
    _make_bundle(tmp_path)

    manifest = build_manifest(tmp_path)

    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    filenames = [e.filename for e in manifest.entries]
    assert filenames == ["model_definition.json", "best_model_state_dict.pt", "class_mapping.json"]
    by_name = {e.filename: e for e in manifest.entries}
    assert by_name["model_definition.json"].size_bytes == len(_MODEL_DEFINITION_BYTES)
    assert by_name["class_mapping.json"].size_bytes == len(_CLASS_MAPPING_BYTES)
    for entry in manifest.entries:
        assert len(entry.sha256) == 64
        assert entry.sha256 == entry.sha256.lower()
        int(entry.sha256, 16)  # valid hex


def test_write_manifest_output_is_byte_identical_across_repeated_writes(tmp_path: Path) -> None:
    _make_bundle(tmp_path)

    path1 = write_manifest(tmp_path)
    first_bytes = path1.read_bytes()

    path2 = write_manifest(tmp_path)
    second_bytes = path2.read_bytes()

    assert path1 == path2
    assert first_bytes == second_bytes


def test_write_manifest_entry_order_is_fixed_canonical_order(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    write_manifest(tmp_path)

    raw = (tmp_path / ARTIFACT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    payload = json.loads(raw)
    filenames = [e["filename"] for e in payload["artifacts"]]
    assert filenames == ["model_definition.json", "best_model_state_dict.pt", "class_mapping.json"]


def test_write_manifest_uses_fixed_filename(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    dest = write_manifest(tmp_path)
    assert dest == tmp_path / "artifact_manifest.json"
    assert dest.name == ARTIFACT_MANIFEST_FILENAME


# -- load_manifest() / verify_manifest(): 성공 round-trip --------------------


def test_load_manifest_round_trips_written_manifest(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    write_manifest(tmp_path)

    loaded = load_manifest(tmp_path)

    assert loaded is not None
    assert loaded.schema_version == MANIFEST_SCHEMA_VERSION
    assert [e.filename for e in loaded.entries] == [
        "model_definition.json",
        "best_model_state_dict.pt",
        "class_mapping.json",
    ]


def test_verify_manifest_succeeds_for_unchanged_bundle(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    write_manifest(tmp_path)

    verify_manifest(tmp_path)  # no exception


def test_load_manifest_returns_none_when_file_absent(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    assert load_manifest(tmp_path) is None


def test_verify_manifest_raises_when_manifest_absent(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    with pytest.raises(ManifestError, match="does not exist"):
        verify_manifest(tmp_path)


# -- 변조 탐지: 바이트 변경 / 크기 변경 / 파일 누락 --------------------------


def test_verify_manifest_detects_changed_bytes_same_size(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    write_manifest(tmp_path)

    original = (tmp_path / "class_mapping.json").read_bytes()
    tampered = bytearray(original)
    tampered[10] ^= 0xFF  # flip a bit, keep length identical
    (tmp_path / "class_mapping.json").write_bytes(bytes(tampered))
    assert len(tampered) == len(original)

    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        verify_manifest(tmp_path)


def test_verify_manifest_detects_changed_size(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    write_manifest(tmp_path)

    (tmp_path / "model_definition.json").write_bytes(_MODEL_DEFINITION_BYTES + b"extra tail bytes")

    with pytest.raises(ManifestError, match="size mismatch"):
        verify_manifest(tmp_path)


def test_verify_manifest_detects_missing_artifact_file(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    write_manifest(tmp_path)

    (tmp_path / "best_model_state_dict.pt").unlink()

    with pytest.raises(ManifestError, match="missing"):
        verify_manifest(tmp_path)


def test_build_manifest_raises_when_artifact_missing(tmp_path: Path) -> None:
    (tmp_path / "model_definition.json").write_bytes(_MODEL_DEFINITION_BYTES)
    (tmp_path / "class_mapping.json").write_bytes(_CLASS_MAPPING_BYTES)
    # best_model_state_dict.pt intentionally absent

    with pytest.raises(ManifestError, match="missing"):
        build_manifest(tmp_path)


def test_build_manifest_raises_when_bundle_dir_not_a_directory(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "not_a_dir.txt"
    not_a_dir.write_text("x", encoding="utf-8")

    with pytest.raises(ManifestError, match="not a directory"):
        build_manifest(not_a_dir)


def test_build_manifest_raises_when_artifact_is_a_directory(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    (tmp_path / "class_mapping.json").unlink()
    (tmp_path / "class_mapping.json").mkdir()

    with pytest.raises(ManifestError, match="regular file"):
        build_manifest(tmp_path)


# -- 손상되거나 버전이 안 맞는 매니페스트 -------------------------------------


def test_load_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_non_object_root(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ManifestError, match="must be a JSON object"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    manifest = build_manifest(tmp_path)
    payload = {
        "schema_version": 999,
        "artifacts": [
            {"filename": e.filename, "size_bytes": e.size_bytes, "sha256": e.sha256} for e in manifest.entries
        ],
    }
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="unsupported manifest schema_version"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_missing_required_top_level_field(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(
        json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION}), encoding="utf-8"
    )

    with pytest.raises(ManifestError, match="missing required field"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_entry_missing_field(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts": [{"filename": "model_definition.json", "size_bytes": 10}],
    }
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="missing field"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_unknown_filename(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts": [
            {"filename": "not_a_real_artifact.bin", "size_bytes": 1, "sha256": "a" * 64},
        ],
    }
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="canonical artifact filenames"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_duplicate_entries(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    entry = {"filename": "model_definition.json", "size_bytes": 1, "sha256": "a" * 64}
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "artifacts": [entry, dict(entry)]}
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_incomplete_entry_set(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts": [{"filename": "model_definition.json", "size_bytes": 1, "sha256": "a" * 64}],
    }
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="missing entry"):
        load_manifest(tmp_path)


@pytest.mark.parametrize(
    "bad_sha256",
    [
        "not_hex" + "0" * 57,
        "A" * 64,  # uppercase not allowed
        "a" * 63,  # too short
        "a" * 65,  # too long
    ],
)
def test_load_manifest_rejects_malformed_sha256(tmp_path: Path, bad_sha256: str) -> None:
    _make_bundle(tmp_path)
    manifest = build_manifest(tmp_path)
    artifacts = [{"filename": e.filename, "size_bytes": e.size_bytes, "sha256": e.sha256} for e in manifest.entries]
    artifacts[0]["sha256"] = bad_sha256
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "artifacts": artifacts}
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="sha256"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_negative_size_bytes(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    manifest = build_manifest(tmp_path)
    artifacts = [{"filename": e.filename, "size_bytes": e.size_bytes, "sha256": e.sha256} for e in manifest.entries]
    artifacts[0]["size_bytes"] = -1
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "artifacts": artifacts}
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="size_bytes"):
        load_manifest(tmp_path)


def test_load_manifest_raises_when_manifest_path_is_a_directory(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).mkdir()

    with pytest.raises(ManifestError, match="not a regular file"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_unexpected_top_level_field(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    manifest = build_manifest(tmp_path)
    payload = {
        "schema_version": manifest.schema_version,
        "artifacts": [
            {"filename": e.filename, "size_bytes": e.size_bytes, "sha256": e.sha256} for e in manifest.entries
        ],
        "signature": "unexpected-extra-field",
    }
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="unexpected field"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_unexpected_entry_field(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    manifest = build_manifest(tmp_path)
    artifacts = [{"filename": e.filename, "size_bytes": e.size_bytes, "sha256": e.sha256} for e in manifest.entries]
    artifacts[0]["extra_field"] = "not allowed"
    payload = {"schema_version": manifest.schema_version, "artifacts": artifacts}
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="unexpected field"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    """표준 json 모듈은 기본적으로 같은 key가 두 번 나오면 나중 값으로
    조용히 덮어쓴다 -- 매니페스트 파서는 이를 malformed JSON으로
    거부해야 한다(예: sha256을 두 번 써서 검증기가 보지 못하는 값으로
    슬쩍 바꿔치기하는 것을 막는다)."""
    _make_bundle(tmp_path)
    raw = (
        '{"schema_version": 1, "artifacts": ['
        '{"filename": "model_definition.json", "size_bytes": 1, "sha256": "'
        + ("a" * 64)
        + '", "sha256": "'
        + ("b" * 64)
        + '"}'
        "]}"
    )
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(raw, encoding="utf-8")

    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_invalid_utf8(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_bytes(b"\xff\xfe\x00invalid utf-8")

    with pytest.raises(ManifestError, match="not valid UTF-8"):
        load_manifest(tmp_path)


def test_load_manifest_normalizes_deeply_nested_root_array(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    depth = sys.getrecursionlimit() + 200
    nested_array = "[" * depth + "0" + "]" * depth
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(nested_array, encoding="utf-8")

    with pytest.raises(ManifestError, match="nesting exceeds the supported depth") as excinfo:
        load_manifest(tmp_path)

    assert type(excinfo.value) is ManifestError
    assert len(str(excinfo.value)) <= 200


def test_load_manifest_normalizes_deeply_nested_unexpected_field(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    depth = sys.getrecursionlimit() + 200
    sentinel = "nested-payload-must-not-appear"
    nested_value = "[" * depth + json.dumps(sentinel) + "]" * depth
    raw = (
        '{"schema_version":1,"artifacts":[],"unexpected":'
        + nested_value
        + "}"
    )
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(raw, encoding="utf-8")

    with pytest.raises(ManifestError, match="nesting exceeds the supported depth") as excinfo:
        load_manifest(tmp_path)

    assert type(excinfo.value) is ManifestError
    assert sentinel not in str(excinfo.value)
    assert len(str(excinfo.value)) <= 200


def test_manifest_error_messages_are_bounded_for_oversized_untrusted_values(tmp_path: Path) -> None:
    """매니페스트 JSON에서 그대로 읽은 값(filename 등)이 아주 길어도,
    이를 담은 ManifestError 메시지 자체는 유계(bounded)여야 한다 --
    공격자가 임의로 긴 문자열을 넣어 에러 메시지/로그를 부풀리는 것을
    막기 위함이다."""
    _make_bundle(tmp_path)
    huge_filename = "x" * 100_000
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts": [{"filename": huge_filename, "size_bytes": 1, "sha256": "a" * 64}],
    }
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(tmp_path)

    assert len(str(excinfo.value)) < 1000  # far shorter than the 100_000-char attacker input


# -- 안전하지 않은 경로 / 심볼릭 링크 -----------------------------------------


def test_build_manifest_rejects_bundle_dir_containing_dotdot_named_file(tmp_path: Path) -> None:
    # Directly probing internal safety check: canonical names never contain
    # separators, so any attempt to point outside the fixed filename set is
    # rejected before the filesystem is even touched.
    from image_ai_studio.training.artifact_manifest import _require_safe_regular_file

    with pytest.raises(ManifestError, match="not one of the canonical"):
        _require_safe_regular_file(tmp_path, "../escape.json")


def test_require_safe_regular_file_rejects_absolute_path_as_filename(tmp_path: Path) -> None:
    from image_ai_studio.training.artifact_manifest import _require_safe_regular_file

    absolute_like = str((tmp_path / "model_definition.json").resolve())
    with pytest.raises(ManifestError):
        _require_safe_regular_file(tmp_path, absolute_like)


def _symlinks_supported(tmp_path: Path) -> bool:
    target = tmp_path / "_symlink_probe_target.txt"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "_symlink_probe_link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        return False
    finally:
        target.unlink(missing_ok=True)
        link.unlink(missing_ok=True)
    return True


def test_build_manifest_rejects_symlinked_artifact(tmp_path: Path) -> None:
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks not supported/permitted in this environment")

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    _make_bundle(real_dir)

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "model_definition.json").symlink_to(real_dir / "model_definition.json")
    (bundle_dir / "best_model_state_dict.pt").write_bytes(_STATE_DICT_BYTES)
    (bundle_dir / "class_mapping.json").write_bytes(_CLASS_MAPPING_BYTES)

    with pytest.raises(ManifestError, match="symlink"):
        build_manifest(bundle_dir)


def test_load_manifest_rejects_symlinked_manifest_path(tmp_path: Path) -> None:
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks not supported/permitted in this environment")

    _make_bundle(tmp_path)
    real_manifest = tmp_path / "real_manifest.json"
    write_manifest(tmp_path)
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).rename(real_manifest)
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).symlink_to(real_manifest)

    with pytest.raises(ManifestError, match="symlink"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_dangling_symlink_manifest_path(tmp_path: Path) -> None:
    """끊어진(dangling) 심볼릭 링크는 "매니페스트 없음"(``None``)으로
    조용히 취급되면 안 된다 -- 실제로는 무언가(아마 조작된 링크)가
    있다는 뜻이므로 명확한 ``ManifestError``로 거부해야 한다."""
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks not supported/permitted in this environment")

    _make_bundle(tmp_path)
    missing_target = tmp_path / "does_not_exist.json"
    (tmp_path / ARTIFACT_MANIFEST_FILENAME).symlink_to(missing_target)

    with pytest.raises(ManifestError, match="symlink"):
        load_manifest(tmp_path)


def test_verify_manifest_rejects_symlink_substituted_after_write(tmp_path: Path) -> None:
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks not supported/permitted in this environment")

    _make_bundle(tmp_path)
    write_manifest(tmp_path)

    elsewhere = tmp_path / "elsewhere_class_mapping.json"
    elsewhere.write_bytes(_CLASS_MAPPING_BYTES)
    (tmp_path / "class_mapping.json").unlink()
    (tmp_path / "class_mapping.json").symlink_to(elsewhere)

    with pytest.raises(ManifestError, match="symlink"):
        verify_manifest(tmp_path)


# -- 청크 단위 스트리밍 해싱 ---------------------------------------------------


def test_hash_file_streams_in_bounded_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import io

    big = tmp_path / "big.bin"
    payload = bytes(range(256)) * 5000  # 1.28MB, bigger than the test chunk size
    big.write_bytes(payload)

    monkeypatch.setattr(artifact_manifest, "_HASH_CHUNK_SIZE", 4096)

    requested_sizes: list[int] = []
    real_bytes_io = io.BytesIO(payload)

    class SpyReader:
        def read(self, n: int = -1) -> bytes:
            requested_sizes.append(n)
            return real_bytes_io.read(n)

        def __enter__(self) -> "SpyReader":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

    def fake_open(self: Path, mode: str = "r", *args: object, **kwargs: object) -> SpyReader:
        assert self == big
        assert mode == "rb"
        return SpyReader()

    monkeypatch.setattr(Path, "open", fake_open)

    size, sha256 = artifact_manifest._hash_file(big)

    assert size == len(payload)
    assert sha256 == hashlib.sha256(payload).hexdigest()
    assert len(requested_sizes) > 1  # more than one chunk was read
    assert all(n == 4096 for n in requested_sizes)  # every request is exactly the bounded chunk size


# -- atomic_write_text 실패 시 기존 매니페스트 보존 ---------------------------


def test_write_manifest_failure_preserves_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_bundle(tmp_path)
    write_manifest(tmp_path)
    original_bytes = (tmp_path / ARTIFACT_MANIFEST_FILENAME).read_bytes()

    from image_ai_studio.training import artifact_io

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(artifact_io.os, "replace", failing_replace)

    # Mutate a source file so the recomputed manifest would differ if published.
    (tmp_path / "class_mapping.json").write_bytes(_CLASS_MAPPING_BYTES + b"more")

    with pytest.raises(OSError, match="simulated replace failure"):
        write_manifest(tmp_path)

    assert (tmp_path / ARTIFACT_MANIFEST_FILENAME).read_bytes() == original_bytes


def test_write_manifest_uses_atomic_write_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_bundle(tmp_path)
    calls = []
    original = artifact_manifest.atomic_write_text

    def spy(text, path, **kwargs):
        calls.append((text, Path(path)))
        return original(text, path, **kwargs)

    monkeypatch.setattr(artifact_manifest, "atomic_write_text", spy)
    write_manifest(tmp_path)

    assert len(calls) == 1
    assert calls[0][1] == tmp_path / ARTIFACT_MANIFEST_FILENAME


# -- write_manifest()는 precomputed manifest 우회 경로를 제공하지 않는다 -----


def test_write_manifest_does_not_accept_a_manifest_override(tmp_path: Path) -> None:
    """write_manifest()는 caller가 미리 만든 ArtifactManifest를 그대로
    게시하는 지름길을 두지 않는다 -- 그런 인자를 허용하면
    build_manifest()가 강제하는 모든 검증(canonical 파일 존재/심볼릭
    링크 금지/버전/다이제스트 형식)을 우회해 임의의(버전이 다르거나,
    다이제스트가 조작된) 매니페스트를 게시할 수 있게 되기 때문이다."""
    _make_bundle(tmp_path)
    fake_manifest = ArtifactManifest(
        schema_version=999,
        entries=(ArtifactEntry(filename="model_definition.json", size_bytes=1, sha256="a" * 64),),
    )

    with pytest.raises(TypeError):
        write_manifest(tmp_path, manifest=fake_manifest)  # type: ignore[call-arg]


def test_manifest_dataclasses_are_frozen_and_comparable() -> None:
    entry_a = ArtifactEntry(filename="model_definition.json", size_bytes=1, sha256="a" * 64)
    entry_b = ArtifactEntry(filename="model_definition.json", size_bytes=1, sha256="a" * 64)
    assert entry_a == entry_b
    with pytest.raises(Exception):
        entry_a.size_bytes = 2  # type: ignore[misc]

    manifest_a = ArtifactManifest(schema_version=1, entries=(entry_a,))
    manifest_b = ArtifactManifest(schema_version=1, entries=(entry_b,))
    assert manifest_a == manifest_b
