"""내부 학습 아티팩트 I/O primitive (Phase 8 checkpoint 1).

checkpoint.py의 ``_atomic_torch_save()``와 imagefolder_resume.py의
``_atomic_write_text()``가 각자 검증한 "목적지와 같은 디렉터리에 임시 파일
생성 -> flush()/os.fsync() -> os.replace()" 안전 패턴을, 앞으로 다른
아티팩트 writer도 재사용할 수 있도록 한 곳에 모은 모듈이다. ``save_model_spec`` / ``save_class_mapping`` / ``save_state_dict`` 세 writer는
이제 각자 이 모듈의 ``atomic_write_text`` / ``atomic_torch_save`` primitive를
통해 목적지에 게시한다. 다만 원자성은 여전히 한 파일 호출 단위이며(여러
아티팩트에 걸친 다중 파일 트랜잭션이 아니다), 게시 이전 실패 시 helper의 임시
파일 정리는 best-effort다. Phase 6/7 공개 API(controller/worker/request/result/training
workflow/model-definition/dataset)도 전혀 바꾸지 않는다.

계약(두 primitive 공통):

- 목적지 디렉터리(``Path(path).parent``) 안에 ``tempfile.mkstemp()``로
  유일한 이름의 임시 파일을 만든다. 다른 디렉터리로 폴백하지 않는다.
- 임시 파일에 직렬화한 뒤 ``flush()`` + ``os.fsync()``로 완성된 바이트의
  디스크 반영을 요청하고, ``os.replace()``로 목적지에 게시(publish)한다
  (POSIX/Windows 양쪽에서 원자적 교체).
- 직렬화/flush/fsync/replace 중 게시 이전에 실패하면: 이미 존재하던
  목적지 파일은 바이트 단위로 그대로 남고, helper는 자신이 만든 임시
  파일만 best-effort로 지운 뒤(정리 실패는 삼켜서 원래 예외를 가리지
  않는다) 원래 예외를 그대로 전파한다(재시도/폴백 없음).
- 게시에 성공하면 helper 소유의 임시 파일은 남지 않고, 목적지는 각각
  표준 ``Path.read_text(encoding=...)`` / ``torch.load()`` 경로로 다시
  읽을 수 있다.

이 모듈은 미리 존재하던 임의의 파일을 지우거나 재사용하지 않으며, 여러
아티팩트에 걸친 다중 파일 트랜잭션(all-or-nothing)을 제공하지 않는다 --
한 번 호출은 정확히 한 파일의 원자적 게시만 보장한다.
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import torch

__all__ = ["atomic_write_text", "atomic_torch_save"]


def _publish_atomically(path: Path, serialize: Callable[[BinaryIO], None]) -> None:
    """``serialize(f)``가 binary 임시 파일 핸들에 바이트를 쓰게 한 뒤
    flush/fsync/os.replace로 ``path``에 원자적으로 게시한다. 모듈 docstring의
    계약을 그대로 구현한다 -- 게시 이전 실패 시 기존 ``path``를 바이트 단위로
    보존하고, helper가 만든 임시 파일만 정리하며(정리 실패는 원래 예외를
    가리지 않는다) 원래 예외를 재시도/폴백 없이 전파한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            serialize(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(text: str, path: str | Path, *, encoding: str = "utf-8") -> None:
    """``text``를 ``encoding``(기본 UTF-8)으로 인코딩해 ``path``에 원자적으로
    쓴다. 인코딩(직렬화)/flush/fsync/replace 중 게시 이전 실패 시 기존
    ``path``는 그대로 보존되고 helper의 임시 파일만 정리된다. 게시 성공 시
    ``Path(path).read_text(encoding=...)``로 그대로 읽을 수 있다."""
    dest = Path(path)
    _publish_atomically(dest, lambda f: f.write(text.encode(encoding)))


def atomic_torch_save(obj: object, path: str | Path) -> None:
    """``obj``를 ``torch.save()``로 직렬화해 ``path``에 원자적으로 쓴다.
    직렬화/flush/fsync/replace 중 게시 이전 실패 시 기존 ``path``는 그대로
    보존되고 helper의 임시 파일만 정리된다. 게시 성공 시 표준
    ``torch.load(path)`` 경로로 그대로 읽을 수 있다."""
    dest = Path(path)
    _publish_atomically(dest, lambda f: torch.save(obj, f))
