"""Phase 11 CP1: 기존 `FolderInferenceResult`를 위한 framework-independent
CSV/JSON export 계약. GUI/Qt/application controller를 전혀 모른다 -- 이미
계산된 `FolderInferenceResult`/`ImageOutcome`/`InferenceResult` 값만 읽어
텍스트로 직렬화할 뿐, inference를 다시 실행하거나 예측값을 재계산하지
않는다. Phase 6B의 single-image public API와 Phase 7 portable artifact
포맷, Phase 10의 `FolderInferenceResult` 등 기존 공개 타입은 전혀
바꾸지 않는다 -- 이 모듈은 그 위에 얹히는 순수 export 계층이다.

JSON 표현은 명시적으로 버전이 붙은 스키마다::

    {
      "format_version": 1,
      "total": <int>,
      "succeeded": <int>,
      "failed": <int>,
      "items": [
        {
          "image_path": <str>,
          "status": "success" | "failed",
          "predicted_class": <str> | null,
          "confidence": <float> | null,
          "probabilities": {<class name>: <float>, ...} | null,
          "inference_duration_seconds": <float> | null,
          "error": <str> | null
        },
        ...
      ]
    }

성공 항목은 predicted_class/confidence/probabilities/
inference_duration_seconds가 채워지고 error는 null이다. 실패 항목은
정반대로 그 넷이 null이고 error만 채워진다(값을 지어내지 않는다).
`items`의 순서는 `FolderInferenceResult.items`의 결정론적 순서를 그대로
보존한다.

CSV 표현은 고정된 헤더 한 줄만 쓴다::

    image_path,status,predicted_class,confidence,probabilities,inference_duration_seconds,error

CSV는 셀 하나에 dict를 담을 수 없으므로 `probabilities`는
class-name으로 정렬한 JSON 텍스트 문자열로 인코딩한다(JSON 표현에서는
그대로 중첩 object다). 실패 행에서 predicted_class/confidence/
probabilities/inference_duration_seconds는 빈 문자열이다.

숫자 직렬화(로케일 독립적): confidence와 inference_duration_seconds는
JSON에서는 `float`를 그대로 담아 `json.dumps`가 처리하게 하고(파이썬
`json` 모듈은 로케일을 참조하지 않고 `float.__repr__`와 동일한
round-trip 알고리즘으로 숫자를 문자열화한다), CSV에서는 동일한
알고리즘을 쓰는 `repr(value)`로 직접 문자열화한다 -- 두 표현 모두
현재 로케일(콤마 소수점 구분자 등)의 영향을 받지 않는다.

파일 게시는 기존 `image_ai_studio.training.artifact_io.atomic_write_text`
원자적 replacement primitive를 그대로 재사용한다: 목적지와 같은
디렉터리에 임시 파일을 만들어 쓴 뒤 `os.replace()`로 교체하므로,
직렬화나 교체가 실패하면 기존 목적지 파일은 바이트 단위로 보존되고
helper의 임시 파일만 정리된 뒤 원래 예외가 그대로 전파된다(재시도/
폴백 없음). 성공하면 임시 파일은 남지 않는다."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from image_ai_studio.inference.folder_inference import FolderInferenceResult, ImageOutcome
from image_ai_studio.training.artifact_io import atomic_write_text

__all__ = [
    "EXPORT_FORMAT_VERSION",
    "CSV_COLUMNS",
    "SUPPORTED_EXPORT_FORMATS",
    "FolderResultExportError",
    "folder_result_to_json_dict",
    "folder_result_to_json_text",
    "folder_result_to_csv_rows",
    "folder_result_to_csv_text",
    "write_folder_result_export",
]

# 스키마 버전. 필드 구성이나 의미가 바뀌면(필드 추가/삭제/재정의) 반드시
# 올려야 한다 -- 기존 값 1은 이 docstring이 문서화한 스키마 그대로다.
EXPORT_FORMAT_VERSION = 1

# CSV 헤더의 정확한 순서(계약의 일부 -- 재정렬 금지).
CSV_COLUMNS: tuple[str, ...] = (
    "image_path",
    "status",
    "predicted_class",
    "confidence",
    "probabilities",
    "inference_duration_seconds",
    "error",
)

SUPPORTED_EXPORT_FORMATS: tuple[str, ...] = ("csv", "json")


class FolderResultExportError(ValueError):
    """export precondition 위반 -- 지원하지 않는 포맷, 파일이 아닌
    목적지(이미 존재하는 디렉터리) 등. 파일시스템 게시를 시도하기 전에
    발생하며, 입력 `FolderInferenceResult`를 건드리지 않는다."""


def _status(outcome: ImageOutcome) -> str:
    return "success" if outcome.succeeded else "failed"


def _sorted_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """class name 오름차순으로 정렬한 새 dict를 돌려준다(원본은
    건드리지 않는다) -- JSON/CSV 양쪽 표현이 같은 순서를 쓴다."""
    return {name: probabilities[name] for name in sorted(probabilities)}


def _json_item(outcome: ImageOutcome) -> dict[str, Any]:
    item: dict[str, Any] = {
        "image_path": str(outcome.image_path),
        "status": _status(outcome),
    }
    if outcome.succeeded:
        result = outcome.result
        assert result is not None  # ImageOutcome 불변식: succeeded => result 존재
        item.update(
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            probabilities=_sorted_probabilities(result.probabilities),
            inference_duration_seconds=result.inference_duration_seconds,
            error=None,
        )
    else:
        item.update(
            predicted_class=None,
            confidence=None,
            probabilities=None,
            inference_duration_seconds=None,
            error=outcome.error,
        )
    return item


def folder_result_to_json_dict(result: FolderInferenceResult) -> dict[str, Any]:
    """`result`를 이 모듈 docstring이 정의한 JSON 스키마 dict로 바꾼다.
    `result`나 그 내부 값을 재계산/수정하지 않는다 -- 순수 읽기 변환."""
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "total": result.total,
        "succeeded": result.succeeded,
        "failed": result.failed,
        "items": [_json_item(item) for item in result.items],
    }


def folder_result_to_json_text(result: FolderInferenceResult) -> str:
    """`folder_result_to_json_dict(result)`를 결정론적 key 순서(선언
    순서 그대로, 알파벳 재정렬 없음)의 UTF-8 텍스트로 직렬화한다.
    Unicode 문자는 `\\uXXXX`로 escape하지 않고(`ensure_ascii=False`) 그대로
    담고, 끝에 개행 한 줄을 붙인다."""
    return (
        json.dumps(
            folder_result_to_json_dict(result),
            ensure_ascii=False,
            sort_keys=False,
            indent=2,
        )
        + "\n"
    )


def _csv_probabilities_cell(probabilities: dict[str, float] | None) -> str:
    if probabilities is None:
        return ""
    return json.dumps(_sorted_probabilities(probabilities), ensure_ascii=False)


def _csv_numeric_cell(value: float | None) -> str:
    if value is None:
        return ""
    # repr()는 json 모듈과 동일한 round-trip 알고리즘을 쓰므로 CSV와
    # JSON 표현의 숫자 문자열이 일치한다(로케일 영향 없음, 위 모듈
    # docstring 참고).
    return repr(value)


def folder_result_to_csv_rows(result: FolderInferenceResult) -> list[list[str]]:
    """`result.items`와 정확히 같은 순서/개수로 CSV 데이터 행(헤더
    제외)을 만든다. 각 행은 `CSV_COLUMNS` 순서와 동일한 길이-7 리스트다."""
    rows: list[list[str]] = []
    for outcome in result.items:
        if outcome.succeeded:
            res = outcome.result
            assert res is not None
            rows.append(
                [
                    str(outcome.image_path),
                    "success",
                    res.predicted_class,
                    _csv_numeric_cell(res.confidence),
                    _csv_probabilities_cell(res.probabilities),
                    _csv_numeric_cell(res.inference_duration_seconds),
                    "",
                ]
            )
        else:
            rows.append(
                [
                    str(outcome.image_path),
                    "failed",
                    "",
                    "",
                    "",
                    "",
                    outcome.error or "",
                ]
            )
    return rows


def folder_result_to_csv_text(result: FolderInferenceResult) -> str:
    """`CSV_COLUMNS` 헤더 한 줄 + `folder_result_to_csv_rows(result)`를
    RFC4180 스타일 quoting(콤마/따옴표/개행 포함 값을 자동으로
    quote/escape)으로 직렬화한 텍스트를 돌려준다."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    writer.writerows(folder_result_to_csv_rows(result))
    return buffer.getvalue()


def write_folder_result_export(
    result: FolderInferenceResult, path: str | Path, *, format: str
) -> None:
    """`result`를 `format`(`"csv"` 또는 `"json"`)으로 직렬화해 `path`에
    원자적으로 게시한다(`atomic_write_text`, 이 모듈 docstring 참고).

    `format`이 `SUPPORTED_EXPORT_FORMATS`에 없거나 `path`가 이미
    디렉터리로 존재하면(파일이 아닌 목적지) 파일시스템을 건드리기 전에
    `FolderResultExportError`를 던진다. 직렬화/게시 중 실패하면 기존
    목적지 파일은 그대로 보존되고 원래 예외가 그대로 전파된다(폴백
    없음). `result`는 어떤 경로에서도 수정되지 않는다."""
    if format not in SUPPORTED_EXPORT_FORMATS:
        raise FolderResultExportError(
            f"format must be one of {SUPPORTED_EXPORT_FORMATS}, got {format!r}"
        )
    dest = Path(path)
    if dest.is_dir():
        raise FolderResultExportError(f"destination is not a file path: {dest}")

    text = folder_result_to_json_text(result) if format == "json" else folder_result_to_csv_text(result)
    atomic_write_text(text, dest)
