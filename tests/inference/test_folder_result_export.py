"""`folder_result_export` 모듈 테스트(Phase 11 CP1). 실제 모델이나
이미지 파일을 전혀 쓰지 않는다 -- `FolderInferenceResult`/`ImageOutcome`/
`InferenceResult`를 직접 생성해서 CSV/JSON export 계약(스키마, 순서,
escaping, 원자적 게시, 실패 처리)만 검증한다. CPU 전용이며 CUDA/Qt/
network/모델 로딩을 요구하지 않는다."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from image_ai_studio.inference.folder_inference import FolderInferenceResult, ImageOutcome
from image_ai_studio.inference.folder_result_export import (
    CSV_COLUMNS,
    EXPORT_FORMAT_VERSION,
    SUPPORTED_EXPORT_FORMATS,
    FolderResultExportError,
    folder_result_to_csv_rows,
    folder_result_to_csv_text,
    folder_result_to_json_dict,
    folder_result_to_json_text,
    write_folder_result_export,
)
from image_ai_studio.inference.single_image_inference import InferenceResult

# 주의: 이 체크포인트의 allowed_files는 이 테스트 모듈과
# src/image_ai_studio/inference/folder_result_export.py 둘뿐이라
# pyproject.toml의 pytest marker 목록을 건드릴 수 없다. 채점 하네스가
# `--strict-markers`로 실행되므로 등록되지 않은 marker를 여기 붙이면
# collection 자체가 실패한다 -- 그래서 커스텀 `pytestmark`를 두지 않는다
# (다른 Phase 테스트 모듈들의 `pytestmark = pytest.mark.phaseN_cpM_...`
# 관례와 다른 점은 이 제약 때문이다).


# -- helpers -----------------------------------------------------------------


def _success(
    image_path: str | Path,
    *,
    predicted_class: str = "cat",
    confidence: float = 0.875,
    probabilities: dict[str, float] | None = None,
    duration: float = 0.0125,
) -> ImageOutcome:
    result = InferenceResult(
        predicted_index=0,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities if probabilities is not None else {"cat": confidence, "dog": 1 - confidence},
        inference_duration_seconds=duration,
    )
    return ImageOutcome(image_path=Path(image_path), result=result, error=None)


def _failure(image_path: str | Path, *, error: str = "RuntimeError: boom") -> ImageOutcome:
    return ImageOutcome(image_path=Path(image_path), result=None, error=error)


def _parse_csv_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


# -- JSON schema --------------------------------------------------------


def test_json_dict_has_versioned_top_level_schema() -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"), _failure("b.png")))
    data = folder_result_to_json_dict(aggregate)

    assert list(data.keys()) == ["format_version", "total", "succeeded", "failed", "items"]
    assert data["format_version"] == EXPORT_FORMAT_VERSION
    assert data["total"] == 2
    assert data["succeeded"] == 1
    assert data["failed"] == 1
    assert len(data["items"]) == 2


def test_json_item_field_order_and_success_values() -> None:
    aggregate = FolderInferenceResult(
        items=(
            _success(
                "images/a.png",
                predicted_class="cat",
                confidence=0.75,
                probabilities={"dog": 0.25, "cat": 0.75},
                duration=0.005,
            ),
        )
    )
    (item,) = folder_result_to_json_dict(aggregate)["items"]

    assert list(item.keys()) == [
        "image_path",
        "status",
        "predicted_class",
        "confidence",
        "probabilities",
        "inference_duration_seconds",
        "error",
    ]
    assert item["image_path"] == str(Path("images/a.png"))
    assert item["status"] == "success"
    assert item["predicted_class"] == "cat"
    assert item["confidence"] == 0.75
    assert item["probabilities"] == {"cat": 0.75, "dog": 0.25}
    assert list(item["probabilities"].keys()) == ["cat", "dog"]  # class-name 오름차순
    assert item["inference_duration_seconds"] == 0.005
    assert item["error"] is None


def test_json_item_failure_values_are_null_except_status_error() -> None:
    aggregate = FolderInferenceResult(items=(_failure("images/bad.png", error="ValueError: nope"),))
    (item,) = folder_result_to_json_dict(aggregate)["items"]

    assert item["status"] == "failed"
    assert item["predicted_class"] is None
    assert item["confidence"] is None
    assert item["probabilities"] is None
    assert item["inference_duration_seconds"] is None
    assert item["error"] == "ValueError: nope"


def test_json_text_is_utf8_ascii_unescaped_with_trailing_newline() -> None:
    unicode_path = Path("경로") / "고양이.png"
    aggregate = FolderInferenceResult(
        items=(_success(unicode_path, predicted_class="고양이", probabilities={"고양이": 0.9, "개": 0.1}),)
    )
    text = folder_result_to_json_text(aggregate)

    assert text.endswith("\n")
    assert "\\u" not in text  # ensure_ascii=False -- 유니코드 escape 없음
    assert "고양이" in text
    parsed = json.loads(text)
    assert parsed["items"][0]["image_path"] == str(unicode_path)
    assert parsed["items"][0]["predicted_class"] == "고양이"


def test_json_items_preserve_aggregate_order() -> None:
    aggregate = FolderInferenceResult(
        items=(_success("c.png"), _failure("a.png"), _success("b.png"))
    )
    items = folder_result_to_json_dict(aggregate)["items"]
    assert [item["image_path"] for item in items] == ["c.png", "a.png", "b.png"]


def test_image_path_serialized_as_full_supplied_path_not_display_name() -> None:
    nested = Path("some") / "nested" / "dir" / "photo.png"
    aggregate = FolderInferenceResult(items=(_success(nested),))
    (item,) = folder_result_to_json_dict(aggregate)["items"]
    assert item["image_path"] == str(nested)
    assert item["image_path"] != "photo.png"


# -- CSV schema ---------------------------------------------------------


def test_csv_columns_constant_matches_required_header_order() -> None:
    assert CSV_COLUMNS == (
        "image_path",
        "status",
        "predicted_class",
        "confidence",
        "probabilities",
        "inference_duration_seconds",
        "error",
    )


def test_csv_text_header_row_matches_csv_columns() -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"),))
    rows = _parse_csv_rows(folder_result_to_csv_text(aggregate))
    assert rows[0] == list(CSV_COLUMNS)


def test_csv_rows_count_matches_items_and_preserves_order() -> None:
    aggregate = FolderInferenceResult(
        items=(_success("c.png"), _failure("a.png"), _success("b.png"))
    )
    data_rows = folder_result_to_csv_rows(aggregate)
    assert len(data_rows) == 3
    assert [row[0] for row in data_rows] == ["c.png", "a.png", "b.png"]
    for row in data_rows:
        assert len(row) == len(CSV_COLUMNS)


def test_csv_success_row_values_round_trip() -> None:
    aggregate = FolderInferenceResult(
        items=(
            _success(
                "img.png",
                predicted_class="dog",
                confidence=0.6,
                probabilities={"dog": 0.6, "cat": 0.4},
                duration=0.02,
            ),
        )
    )
    rows = _parse_csv_rows(folder_result_to_csv_text(aggregate))
    header, row = rows[0], rows[1]
    record = dict(zip(header, row))

    assert record["image_path"] == "img.png"
    assert record["status"] == "success"
    assert record["predicted_class"] == "dog"
    assert float(record["confidence"]) == 0.6
    assert json.loads(record["probabilities"]) == {"cat": 0.4, "dog": 0.6}
    assert list(json.loads(record["probabilities"]).keys()) == ["cat", "dog"]
    assert float(record["inference_duration_seconds"]) == 0.02
    assert record["error"] == ""


def test_csv_failure_row_has_empty_prediction_fields() -> None:
    aggregate = FolderInferenceResult(items=(_failure("bad.png", error="boom: detail"),))
    rows = _parse_csv_rows(folder_result_to_csv_text(aggregate))
    header, row = rows[0], rows[1]
    record = dict(zip(header, row))

    assert record["status"] == "failed"
    assert record["predicted_class"] == ""
    assert record["confidence"] == ""
    assert record["probabilities"] == ""
    assert record["inference_duration_seconds"] == ""
    assert record["error"] == "boom: detail"


# -- mixed / all-failure aggregates --------------------------------


def test_mixed_aggregate_retains_exact_counts_and_rows() -> None:
    aggregate = FolderInferenceResult(
        items=(_success("a.png"), _failure("b.png"), _success("c.png"), _failure("d.png"))
    )
    data = folder_result_to_json_dict(aggregate)
    assert (data["total"], data["succeeded"], data["failed"]) == (4, 2, 2)
    statuses = [item["status"] for item in data["items"]]
    assert statuses == ["success", "failed", "success", "failed"]

    csv_rows = folder_result_to_csv_rows(aggregate)
    assert len(csv_rows) == 4
    assert [row[1] for row in csv_rows] == ["success", "failed", "success", "failed"]


def test_all_failure_aggregate_has_no_success_rows() -> None:
    aggregate = FolderInferenceResult(
        items=(_failure("a.png"), _failure("b.png"), _failure("c.png"))
    )
    data = folder_result_to_json_dict(aggregate)
    assert (data["total"], data["succeeded"], data["failed"]) == (3, 0, 3)
    assert all(item["status"] == "failed" for item in data["items"])

    for row in folder_result_to_csv_rows(aggregate):
        assert row[1] == "failed"
        assert row[2] == row[3] == row[4] == row[5] == ""


def test_all_success_aggregate_has_no_failure_rows() -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"), _success("b.png")))
    data = folder_result_to_json_dict(aggregate)
    assert (data["total"], data["succeeded"], data["failed"]) == (2, 2, 0)
    assert all(item["error"] is None for item in data["items"])


# -- determinism ----------------------------------------------------


def test_repeated_export_calls_produce_identical_text() -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"), _failure("b.png")))
    assert folder_result_to_json_text(aggregate) == folder_result_to_json_text(aggregate)
    assert folder_result_to_csv_text(aggregate) == folder_result_to_csv_text(aggregate)


def test_export_does_not_mutate_input_result() -> None:
    probabilities = {"dog": 0.4, "cat": 0.6}
    outcome = _success("a.png", probabilities=probabilities)
    aggregate = FolderInferenceResult(items=(outcome,))

    folder_result_to_json_text(aggregate)
    folder_result_to_csv_text(aggregate)

    assert outcome.result is not None
    assert outcome.result.probabilities == {"dog": 0.4, "cat": 0.6}
    assert list(outcome.result.probabilities.keys()) == ["dog", "cat"]  # 원본 순서 그대로


# -- escaping / unicode ----------------------------------------------


def test_csv_escapes_commas_quotes_and_newlines() -> None:
    tricky_path = 'weird, "path"\nwith stuff.png'
    tricky_error = 'error, with "quotes"\nand a newline'
    aggregate = FolderInferenceResult(items=(_failure(tricky_path, error=tricky_error),))

    text = folder_result_to_csv_text(aggregate)
    rows = _parse_csv_rows(text)
    header, row = rows[0], rows[1]
    record = dict(zip(header, row))

    assert record["image_path"] == tricky_path
    assert record["error"] == tricky_error


def test_csv_preserves_unicode_paths_and_classes() -> None:
    unicode_path = Path("é") / "名前" / "фото.png"
    aggregate = FolderInferenceResult(
        items=(_success(unicode_path, predicted_class="猫", probabilities={"猫": 0.9, "犬": 0.1}),)
    )
    rows = _parse_csv_rows(folder_result_to_csv_text(aggregate))
    header, row = rows[0], rows[1]
    record = dict(zip(header, row))

    assert record["image_path"] == str(unicode_path)
    assert record["predicted_class"] == "猫"
    assert json.loads(record["probabilities"]) == {"猫": 0.9, "犬": 0.1}


# -- unsupported formats / bad destinations --------------------------


def test_write_rejects_unsupported_format_without_touching_filesystem(tmp_path: Path) -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"),))
    dest = tmp_path / "out.xml"

    with pytest.raises(FolderResultExportError, match="format"):
        write_folder_result_export(aggregate, dest, format="xml")
    assert not dest.exists()


def test_write_rejects_directory_destination(tmp_path: Path) -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"),))
    directory = tmp_path / "already_a_dir"
    directory.mkdir()

    with pytest.raises(FolderResultExportError, match="not a file path"):
        write_folder_result_export(aggregate, directory, format="json")
    assert list(directory.iterdir()) == []


def test_supported_export_formats_constant() -> None:
    assert set(SUPPORTED_EXPORT_FORMATS) == {"csv", "json"}


# -- atomic write: success --------------------------------------------


def test_write_json_creates_file_matching_text_function(tmp_path: Path) -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"), _failure("b.png")))
    dest = tmp_path / "result.json"

    write_folder_result_export(aggregate, dest, format="json")

    assert dest.read_text(encoding="utf-8") == folder_result_to_json_text(aggregate)
    assert list(tmp_path.iterdir()) == [dest]  # 임시 파일이 남지 않는다


def test_write_csv_creates_file_matching_text_function(tmp_path: Path) -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"), _failure("b.png")))
    dest = tmp_path / "result.csv"

    write_folder_result_export(aggregate, dest, format="csv")

    assert dest.read_text(encoding="utf-8") == folder_result_to_csv_text(aggregate)
    assert list(tmp_path.iterdir()) == [dest]


def test_write_overwrites_existing_destination_atomically(tmp_path: Path) -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"),))
    dest = tmp_path / "result.json"
    dest.write_text("stale content", encoding="utf-8")

    write_folder_result_export(aggregate, dest, format="json")

    assert dest.read_text(encoding="utf-8") == folder_result_to_json_text(aggregate)
    assert list(tmp_path.iterdir()) == [dest]


# -- atomic write: failure preserves existing destination -------------


def test_write_failure_before_replace_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregate = FolderInferenceResult(items=(_success("a.png"),))
    dest = tmp_path / "result.json"
    dest.write_text("PRE-EXISTING", encoding="utf-8")

    def _raise_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("os.replace", _raise_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_folder_result_export(aggregate, dest, format="json")

    assert dest.read_text(encoding="utf-8") == "PRE-EXISTING"
    # helper가 만든 임시 파일도 남지 않는다.
    assert list(tmp_path.iterdir()) == [dest]


def test_write_failure_when_serializing_leaves_no_destination(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "result.json"
    # probabilities dict 안에 JSON으로 직렬화할 수 없는 값(NaN이 아니라
    # 순수 non-serializable 타입)을 넣어 json.dumps가 예외를 던지게 한다.
    bad_result = InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.5,
        probabilities={"cat": object()},  # type: ignore[dict-item]
        inference_duration_seconds=0.01,
    )
    outcome = ImageOutcome(image_path=Path("a.png"), result=bad_result, error=None)
    aggregate = FolderInferenceResult(items=(outcome,))

    with pytest.raises(TypeError):
        write_folder_result_export(aggregate, dest, format="json")
    assert not dest.exists()


# -- no Qt dependency ---------------------------------------------------


def test_module_source_does_not_reference_qt() -> None:
    import image_ai_studio.inference.folder_result_export as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "PySide" not in source
    assert "PyQt" not in source
    assert "QtCore" not in source
