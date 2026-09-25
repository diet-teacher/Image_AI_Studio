"""이식 가능한(portable) 학습 아티팩트 번들을 위한 결정적(deterministic)
무결성 매니페스트(Phase 15 checkpoint 1).

이 모듈은 기존 이식 가능 아티팩트 3종 -- ``model_definition.json``,
``best_model_state_dict.pt``, ``class_mapping.json`` (모두
``imagefolder_workflow.py``가 이미 만들어 온 파일들, 포맷/공개 API는
전혀 바꾸지 않는다) -- 을 담는 번들 디렉터리 옆에 함께 게시할 수 있는
작은 JSON 매니페스트 파일(``ARTIFACT_MANIFEST_FILENAME``)의 스키마와
읽기/쓰기/검증 API를 정의한다.

**이 매니페스트가 보장하는 것과 보장하지 않는 것.** 매니페스트에 담기는
SHA-256 다이제스트는 오직 "우연한 손상(디스크/전송 중 bit rot, 잘린
복사, 절반만 쓰인 파일)"과 "서로 다른 학습 실행의 아티팩트가 실수로
섞인 번들(mixed bundle)"을 잡아내기 위한 것이다. 이 매니페스트는
디지털 서명이 아니고, 신뢰할 수 없는 제3자가 의도적으로 파일과
매니페스트를 함께 변조하는 것을 막지 못한다(악의적 변조에 대한
암호학적 authenticity/서명 검증이 아니다) -- 그런 목적이 필요하면 별도의
서명 체계가 있어야 한다.

**결정적 직렬화.** 같은 3개 파일의 바이트가 바뀌지 않는 한, 몇 번을
다시 써도(재실행/재빌드해도) 매니페스트 JSON 바이트는 완전히 동일하다
-- entry 순서는 고정된 canonical 파일 이름 순서(``_CANONICAL_FILENAMES``)를
따르고, ``json.dumps``는 고정 ``sort_keys=True`` + 고정 구분자 +
줄바꿈 하나로 끝나는 포맷을 쓴다. 이것으로 재현 가능한 빌드나
바이트 단위 diff에서 매니페스트가 불필요한 잡음을 만들지 않는다.

**게시(publish)는 원자적이다.** ``write_manifest()``는 직접 파일을
열어 쓰지 않고 기존 ``training/artifact_io.py``의 ``atomic_write_text``
primitive를 그대로 재사용한다 -- 인코딩/flush/fsync/replace 중 어느
단계에서 실패해도 기존 목적지 파일은 바이트 단위로 그대로 남고, 부분
JSON이 노출되지 않는다(그 계약은 ``atomic_write_text`` 자신이 이미
검증한다 -- 이 모듈은 그 계약에 편승할 뿐 다시 구현하지 않는다).

**"매니페스트 없음"과 "매니페스트가 있는데 invalid"는 다른 신호다.**
``load_manifest()``는 파일이 아예 없으면 ``None``을 반환한다(선택적
매니페스트이므로 -- pre-Phase-15 번들에는 매니페스트 자체가 없을 수
있다). 파일은 있는데 JSON이 깨졌거나 스키마가 안 맞으면 ``None`` 대신
``ManifestError``를 던진다. 이 둘을 뭉뚱그리면 caller가 "매니페스트가
아직 없는 구버전 번들"과 "매니페스트가 손상된 번들"을 구분할 수 없게
된다.

**해싱은 파일 전체를 메모리에 올리지 않는다.** ``best_model_state_dict.pt``는
모델 크기에 따라 커질 수 있으므로, 해싱은 고정 크기 청크
(``_HASH_CHUNK_SIZE``)로 스트리밍한다. 그리고 이 모듈은 무결성
계산을 위해 ``.pt``/``.json`` 페이로드를 역직렬화(``torch.load``/
``json.load``)하지 않는다 -- 오직 바이트 스트림의 크기와 다이제스트만
본다. 이는 신뢰할 수 없거나 손상된 체크포인트를 무결성 점검 단계에서
역직렬화해 임의 코드 실행/충돌 위험을 감수하지 않기 위함이다.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from image_ai_studio.training.artifact_io import atomic_write_text

__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "ArtifactEntry",
    "ArtifactManifest",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "verify_manifest",
]

#: 매니페스트가 게시될 때 쓰는 고정 파일 이름. 번들 디렉터리 바로
#: 아래(다른 3개 아티팩트와 같은 위치)에 놓인다.
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"

#: 매니페스트 JSON 스키마의 버전. 스키마가 호환되지 않게 바뀌면 올린다.
MANIFEST_SCHEMA_VERSION = 1

#: 매니페스트가 다루는 정확히 3개의 canonical 이식 가능 아티팩트 파일
#: 이름. 순서가 매니페스트 entry의 결정적 직렬화 순서이기도 하다.
_CANONICAL_FILENAMES: tuple[str, ...] = (
    "model_definition.json",
    "best_model_state_dict.pt",
    "class_mapping.json",
)

#: 해싱 시 한 번에 읽는 바이트 수(스트리밍, 메모리에 전체를 올리지 않음).
_HASH_CHUNK_SIZE = 1024 * 1024

_SHA256_HEX_LENGTH = 64

#: 에러 메시지에 삽입하는 신뢰할 수 없는(untrusted) 값(매니페스트 JSON에서
#: 읽은 filename/size_bytes/sha256/schema_version/알 수 없는 key 이름 등)의
#: 최대 길이. 공격자가 임의로 큰 문자열/정수/키 목록을 매니페스트에 넣어
#: 에러 메시지를 통해 로그를 부풀리거나 자원을 소모시키는 것을 막는다.
_ERROR_VALUE_MAX_LEN = 80


def _bounded(value: object) -> str:
    """``value``를 ``repr()``한 뒤 ``_ERROR_VALUE_MAX_LEN``으로 잘라 에러
    메시지에 안전하게 삽입할 수 있는 문자열을 만든다. ``value``는 매니페스트
    JSON에서 그대로 읽은 신뢰할 수 없는 데이터일 수 있으므로(임의로 긴
    문자열, 임의 정밀도의 큰 정수 등) 항상 이 helper를 거쳐야 한다."""
    text = repr(value)
    if len(text) > _ERROR_VALUE_MAX_LEN:
        return text[:_ERROR_VALUE_MAX_LEN] + "...(truncated)"
    return text


def _bounded_list(values: Iterable[object]) -> str:
    """정렬 가능한 값들의 목록(예: 알 수 없는 key 이름들)을 에러 메시지에
    쓰기 안전한 문자열로 만든다. 항목 개수와 각 항목 길이를 모두 제한한다."""
    items = list(values)
    max_items = 10
    shown = [_bounded(v) for v in items[:max_items]]
    text = "[" + ", ".join(shown) + ("" if len(items) <= max_items else ", ...") + "]"
    return text


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``json.loads(..., object_pairs_hook=...)``에 넘기는 hook. JSON 객체
    literal 하나 안에 같은 key가 두 번 나오면(표준 JSON 파서는 기본적으로
    나중 값으로 조용히 덮어쓴다) ``ValueError``를 던져 매니페스트 파싱을
    거부한다 -- 이 값은 호출부(``_parse_manifest_json``)에서 다른 JSON
    파싱 실패와 동일하게 ``ManifestError``로 감싸진다."""
    seen: set[str] = set()
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate object key {_bounded(key)}")
        seen.add(key)
        result[key] = value
    return result


class ManifestError(ValueError):
    """매니페스트 생성/로드/검증 중 발견한 계약 위반을 나타낸다.

    이 예외 하나로 통일한다 -- 파일 누락, 심볼릭 링크, 안전하지 않은
    경로, 중복 entry, 손상된 JSON, 알 수 없는 스키마 버전, 예상과 다른
    구조를 모두 이 타입으로 표현해 caller가 ``except ManifestError``
    하나로 매니페스트 관련 실패를 잡을 수 있게 한다. ``ValueError``의
    서브클래스라 기존에 ``ValueError``를 잡던 코드와도 호환된다.
    """


@dataclass(frozen=True)
class ArtifactEntry:
    """매니페스트 안의 파일 하나에 대한 entry.

    ``filename``은 번들 디렉터리 기준 canonical 상대 파일 이름
    (``_CANONICAL_FILENAMES`` 중 하나)이다. ``size_bytes``는 정확한
    파일 크기, ``sha256``는 소문자 64자 hex 다이제스트다.
    """

    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    """``ARTIFACT_MANIFEST_FILENAME``에 게시되는 매니페스트 전체 내용."""

    schema_version: int
    entries: tuple[ArtifactEntry, ...]


def _hash_file(path: Path) -> tuple[int, str]:
    """``path``를 고정 크기 청크로 스트리밍하며 바이트 크기와 소문자
    SHA-256 hex 다이제스트를 함께 계산한다. 파일 내용을 역직렬화하지
    않는다 -- 순수 바이트 스트림만 본다."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _require_safe_regular_file(bundle_dir: Path, filename: str) -> Path:
    """``filename``이 ``_CANONICAL_FILENAMES``에 속하는 단순 파일
    이름이고(경로 구분자/절대 경로/상위 디렉터리 참조 없음),
    ``bundle_dir`` 바로 아래에 실제로 존재하는 심볼릭 링크가 아닌
    일반 파일임을 확인한 뒤 그 절대 경로를 반환한다. 위반 시
    ``ManifestError``로 거부한다."""
    if filename not in _CANONICAL_FILENAMES:
        raise ManifestError(
            f"{_bounded(filename)} is not one of the canonical artifact filenames {_CANONICAL_FILENAMES}"
        )
    candidate = Path(filename)
    if candidate.is_absolute() or candidate.name != filename or set(candidate.parts) & {"..", "."}:
        raise ManifestError(
            f"{_bounded(filename)} is not a safe bare filename (no separators, '..', or absolute paths)"
        )

    path = bundle_dir / filename
    if path.is_symlink():
        raise ManifestError(f"{path} must be a regular file, not a symlink")
    if not path.exists():
        raise ManifestError(f"required artifact file is missing: {path}")
    if not path.is_file():
        raise ManifestError(f"{path} must be a regular file (found a directory or other non-regular entry)")
    return path


def build_manifest(bundle_dir: str | Path) -> ArtifactManifest:
    """``bundle_dir`` 바로 아래에 있는 정확히 3개의 canonical 아티팩트
    파일(``model_definition.json``, ``best_model_state_dict.pt``,
    ``class_mapping.json``)을 읽어 ``ArtifactManifest``를 만든다.

    각 파일은 ``bundle_dir`` 바로 아래의 실제로 존재하는, 심볼릭 링크가
    아닌 일반 파일이어야 한다 -- 하나라도 없거나, 디렉터리이거나,
    심볼릭 링크이면 ``ManifestError``를 던진다. entry는 항상
    ``_CANONICAL_FILENAMES`` 순서로 정렬되어 결정적이다(디렉터리
    스캔 순서에 의존하지 않음)."""
    bundle_path = Path(bundle_dir)
    if not bundle_path.is_dir():
        raise ManifestError(f"{bundle_path} is not a directory")

    entries = []
    for filename in _CANONICAL_FILENAMES:
        file_path = _require_safe_regular_file(bundle_path, filename)
        size_bytes, sha256 = _hash_file(file_path)
        entries.append(ArtifactEntry(filename=filename, size_bytes=size_bytes, sha256=sha256))

    return ArtifactManifest(schema_version=MANIFEST_SCHEMA_VERSION, entries=tuple(entries))


def _manifest_to_json_text(manifest: ArtifactManifest) -> str:
    """``manifest``를 결정적 JSON 텍스트로 직렬화한다 -- entry 순서는
    ``manifest.entries``의 순서를 그대로 보존하고(이미
    ``_CANONICAL_FILENAMES`` 순서), 각 entry의 key는 ``sort_keys=True``로
    정렬하며, 구분자와 줄바꿈을 고정해 동일 입력에 대해 항상 동일한
    바이트를 만든다."""
    payload = {
        "schema_version": manifest.schema_version,
        "artifacts": [
            {"filename": e.filename, "size_bytes": e.size_bytes, "sha256": e.sha256} for e in manifest.entries
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def write_manifest(bundle_dir: str | Path) -> Path:
    """``bundle_dir`` 바로 아래의 3개 canonical 아티팩트 파일을 다시
    해싱해(``build_manifest()``) 만든 매니페스트를 ``bundle_dir``의
    ``ARTIFACT_MANIFEST_FILENAME``에 원자적으로 게시하고 그 경로를
    반환한다.

    이 함수는 미리 계산된 ``ArtifactManifest``를 받는 인자를 일부러 두지
    않는다 -- 그런 지름길을 허용하면 caller가 실제 번들 파일과 무관한
    (버전이 다르거나, 파일 이름이 안전하지 않거나, 다이제스트가 조작된,
    또는 순서가 뒤섞인) 매니페스트를 그대로 게시할 수 있게 되어
    ``build_manifest()``가 강제하는 모든 검증을 우회하게 된다. 항상
    ``build_manifest(bundle_dir)``를 다시 실행해 현재 디스크 상의 3개
    파일로부터 매니페스트를 새로 계산한다.

    게시는 ``training/artifact_io.py``의 ``atomic_write_text``를 그대로
    쓴다 -- 인코딩/flush/fsync/replace 중 실패하면 기존 매니페스트
    파일(있었다면)은 바이트 단위로 그대로 남고 부분 JSON이 노출되지
    않는다(재시도/폴백 없이 원래 예외가 전파된다)."""
    bundle_path = Path(bundle_dir)
    manifest = build_manifest(bundle_path)
    text = _manifest_to_json_text(manifest)
    dest = bundle_path / ARTIFACT_MANIFEST_FILENAME
    atomic_write_text(text, dest)
    return dest


_TOP_LEVEL_FIELDS = frozenset({"schema_version", "artifacts"})
_ENTRY_FIELDS = frozenset({"filename", "size_bytes", "sha256"})


def _parse_manifest_json(path: Path, raw_text: str) -> ArtifactManifest:
    """Parse and validate one manifest behind a bounded public error boundary.

    Both CPython's JSON decoder and validation-time formatting/iteration can
    raise ``RecursionError`` for attacker-controlled nesting.  Callers should
    not have to distinguish where the recursion limit was reached, so this
    narrow boundary consistently exposes the condition as ``ManifestError``
    without rendering the nested value.
    """
    try:
        return _parse_manifest_json_impl(path, raw_text)
    except RecursionError as exc:
        raise ManifestError(
            f"{_bounded(path.name)}: manifest JSON nesting exceeds the supported depth"
        ) from exc


def _parse_manifest_json_impl(path: Path, raw_text: str) -> ArtifactManifest:
    try:
        payload = json.loads(raw_text, object_pairs_hook=_no_duplicate_keys)
    except ValueError as exc:
        # json.JSONDecodeError is itself a ValueError subclass, and
        # _no_duplicate_keys() also raises plain ValueError -- both are
        # "this file's JSON is malformed" from the caller's point of view.
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ManifestError(f"{path}: manifest root must be a JSON object, got {type(payload).__name__}")

    extra_top_fields = payload.keys() - _TOP_LEVEL_FIELDS
    if extra_top_fields:
        raise ManifestError(f"{path}: manifest has unexpected field(s): {_bounded_list(sorted(extra_top_fields))}")

    missing_top_fields = _TOP_LEVEL_FIELDS - payload.keys()
    if missing_top_fields:
        raise ManifestError(
            f"{path}: manifest is missing required field(s): {_bounded_list(sorted(missing_top_fields))}"
        )

    schema_version = payload["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ManifestError(f"{path}: 'schema_version' must be an integer, got {type(schema_version).__name__}")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: unsupported manifest schema_version={_bounded(schema_version)} "
            f"(expected {MANIFEST_SCHEMA_VERSION})"
        )

    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list):
        raise ManifestError(f"{path}: 'artifacts' must be a list, got {type(artifacts).__name__}")

    seen_filenames: set[str] = set()
    entries = []
    for index, raw_entry in enumerate(artifacts):
        if not isinstance(raw_entry, dict):
            raise ManifestError(f"{path}: artifacts[{index}] must be an object, got {type(raw_entry).__name__}")

        extra_entry_fields = raw_entry.keys() - _ENTRY_FIELDS
        if extra_entry_fields:
            raise ManifestError(
                f"{path}: artifacts[{index}] has unexpected field(s): {_bounded_list(sorted(extra_entry_fields))}"
            )
        missing_entry_fields = _ENTRY_FIELDS - raw_entry.keys()
        if missing_entry_fields:
            raise ManifestError(
                f"{path}: artifacts[{index}] is missing field(s): {_bounded_list(sorted(missing_entry_fields))}"
            )

        filename = raw_entry["filename"]
        if filename not in _CANONICAL_FILENAMES:
            raise ManifestError(
                f"{path}: artifacts[{index}].filename {_bounded(filename)} is not one of the canonical "
                f"artifact filenames {_CANONICAL_FILENAMES}"
            )
        if filename in seen_filenames:
            raise ManifestError(f"{path}: duplicate manifest entry for filename {_bounded(filename)}")
        seen_filenames.add(filename)

        size_bytes = raw_entry["size_bytes"]
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ManifestError(
                f"{path}: artifacts[{index}].size_bytes must be a non-negative integer, "
                f"got {_bounded(size_bytes)}"
            )

        sha256 = raw_entry["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != _SHA256_HEX_LENGTH
            or sha256 != sha256.lower()
            or any(c not in "0123456789abcdef" for c in sha256)
        ):
            raise ManifestError(
                f"{path}: artifacts[{index}].sha256 must be a lowercase 64-character hex string, "
                f"got {_bounded(sha256)}"
            )

        entries.append(ArtifactEntry(filename=filename, size_bytes=size_bytes, sha256=sha256))

    missing_filenames = set(_CANONICAL_FILENAMES) - seen_filenames
    if missing_filenames:
        raise ManifestError(f"{path}: manifest is missing entry/entries for {sorted(missing_filenames)}")

    entries.sort(key=lambda e: _CANONICAL_FILENAMES.index(e.filename))
    return ArtifactManifest(schema_version=schema_version, entries=tuple(entries))


def load_manifest(bundle_dir: str | Path) -> ArtifactManifest | None:
    """``bundle_dir``의 ``ARTIFACT_MANIFEST_FILENAME``을 읽어
    ``ArtifactManifest``로 파싱한다.

    매니페스트 파일 자체가 없으면(선택적 매니페스트, 예: pre-Phase-15
    번들) ``None``을 반환한다 -- 이것은 "invalid 매니페스트"와는 다른
    신호다. 파일은 있지만 JSON이 깨졌거나, UTF-8로 디코딩할 수 없거나,
    스키마가 맞지 않으면(필수/예상 밖 필드, 알 수 없는 schema_version,
    canonical 목록 밖의 filename, 중복 entry, 잘못된 형식의
    sha256/size_bytes 등) ``ManifestError``를 던진다.

    매니페스트 경로 자체가 심볼릭 링크이면(가리키는 대상이 존재하든,
    끊어진(dangling) 링크이든) 절대 따라가지 않고 ``ManifestError``로
    거부한다 -- 끊어진 심볼릭 링크를 "매니페스트 없음"으로 조용히
    취급하면, 실제로는 (아마 의도적으로 조작된) 심볼릭 링크가 있는데도
    caller가 이를 pre-Phase-15 번들과 구분하지 못하게 된다."""
    path = Path(bundle_dir) / ARTIFACT_MANIFEST_FILENAME
    if path.is_symlink():
        raise ManifestError(f"{path} must be a regular file, not a symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise ManifestError(f"{path} exists but is not a regular file")

    try:
        raw_text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{path} is not valid UTF-8: {exc}") from exc
    return _parse_manifest_json(path, raw_text)


def verify_manifest(bundle_dir: str | Path) -> None:
    """``bundle_dir``의 매니페스트를 로드하고, 매니페스트에 기록된 각
    canonical 아티팩트 파일을 다시 스트리밍 해싱해 크기와 SHA-256
    다이제스트가 정확히 일치하는지 확인한다.

    불일치, 파일 누락, 또는 매니페스트 자체가 invalid하면
    ``ManifestError``를 던진다. **매니페스트 파일이 아예 없으면 아무
    것도 검증할 수 없으므로 조용히 통과시키지 않고 명시적으로
    ``ManifestError``를 던진다** -- "검증할 매니페스트가 없다"와
    "검증했는데 통과했다"를 구분해야 하는 caller는
    ``load_manifest()``를 직접 호출해 ``None``을 먼저 걸러내야 한다."""
    bundle_path = Path(bundle_dir)
    manifest = load_manifest(bundle_path)
    if manifest is None:
        raise ManifestError(
            f"{bundle_path / ARTIFACT_MANIFEST_FILENAME} does not exist -- nothing to verify "
            "(call load_manifest() first to distinguish this from an invalid manifest)"
        )

    for entry in manifest.entries:
        file_path = _require_safe_regular_file(bundle_path, entry.filename)
        actual_size, actual_sha256 = _hash_file(file_path)
        if actual_size != entry.size_bytes:
            raise ManifestError(
                f"{file_path}: size mismatch -- manifest says {_bounded(entry.size_bytes)} bytes, "
                f"actual file is {actual_size} bytes (corrupted or mixed-bundle artifact)"
            )
        if actual_sha256 != entry.sha256:
            raise ManifestError(
                f"{file_path}: SHA-256 mismatch -- manifest says {_bounded(entry.sha256)}, "
                f"actual file hashes to {actual_sha256} (corrupted or mixed-bundle artifact)"
            )
