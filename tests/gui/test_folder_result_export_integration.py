"""Phase 11 CP3: 실제 CPU 폴더 추론 결과 내보내기 졸업 통합 테스트.

CP1(`tests/inference/test_folder_result_export.py`)은 framework-independent
CSV/JSON export 계약(버전 붙은 JSON 스키마, 고정 CSV 헤더, 순서/escaping/
숫자 직렬화/원자적 게시/실패 보존)을 구성된 `FolderInferenceResult` 값으로
고정한다. CP2(`tests/gui/test_inference_page.py`의 Phase 11 CP2 절)는
`InferencePage`가 `_on_folder_finished`에 전달된 바로 그 `FolderInferenceResult`
객체를 -- 테이블 텍스트가 아니라 -- CP1 exporter 경계로 넘기고, 초기/Running/
fatal-failure/stale/mode-switch 상태에서 두 export 액션을 비활성화하며,
save 다이얼로그 취소를 no-op으로, GUI thread의 쓰기 오류를 bounded 메시지로
처리한다는 것을 patched 다이얼로그/exporter 경계로 고정한다.

이 모듈의 책임은 그 조각들이 **실제 `MainWindow` + `InferencePage`의 비동기
폴더 경로**에서, **fake가 아닌 실제 `run_single_image_inference` backend**와
**실제 CP1 exporter**(save 다이얼로그만 patch)와 맞물려 canonical portable
bundle 하나를 끝까지 소비하고, 화면에 보이는 것과 파일로 나간 것이 정확히
일치함을 한 번 증명하는 것이다(`tests/gui/test_folder_inference_integration.py`
의 Phase 10 CP4 통합 테스트와 동일한 철학 -- 중복 correctness 검증은 하지
않는다).

전부 CPU 전용이다: 작은 로컬 이미지/모델만 쓰고 CUDA, 외부 모델 다운로드,
네트워크, 새 의존성, 스크린샷 비교, 취소/진행률/preview/drag-and-drop,
packaging을 요구하지 않으며 pytest 임시 디렉터리 밖의 저장소 아티팩트를
만들거나 바꾸지 않는다. Phase 6B single-image public API, Phase 7 portable
artifact 포맷/경로, CP1 export 모듈, CP2 페이지 동작은 이 모듈에서 전혀
바뀌지 않는다 -- 여기서는 그것들을 소비만 한다.

**Phase 6B stabilization 계약 준수**: `qtbot.waitSignal()`을 쓰지 않고
(canonical wiring이 worker 자신의 finished/failed에 `deleteLater()`를 연결해
두어 `waitSignal()`의 임시 `SignalBlocker`가 그 삭제와 경합하는 것이
실측됐다), `InferencePage`가 노출하는 관측 가능한 상태 + `qtbot.waitUntil()`
polling만 쓴다.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QFileDialog

from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.inference.folder_result_export import (
    CSV_COLUMNS,
    EXPORT_FORMAT_VERSION,
    folder_result_to_csv_text,
    folder_result_to_json_text,
)
from image_ai_studio.inference.folder_result_export import (
    write_folder_result_export as _real_write_folder_result_export,
)
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.checkpoint import save_state_dict
from image_ai_studio.training.torchvision_dataset import save_class_mapping

INPUT_SHAPE = (3, 8, 8)
_CLASSES = ["cat", "dog"]

# Phase 6C `_format_confidence`/`_format_duration_ms` 계약: 소수점 2자리
# 퍼센트 / 밀리초. 실제 추론값은 미리 알 수 없으므로 형식만 패턴으로
# 확인한다(값을 재계산하지 않는다).
_CONFIDENCE_PATTERN = re.compile(r"^\d{1,3}\.\d{2}%$")
_DURATION_PATTERN = re.compile(r"^\d+\.\d{2} ms$")

# InferencePage가 폴더 결과 테이블 셀에 쓰는 고정 문자열(image_ai_studio/
# gui/inference_page.py의 계약).
_STATUS_SUCCESS = "Success"
_STATUS_FAILURE = "Failure"
_RESULT_PLACEHOLDER = "--"
_MODE_FOLDER = "Folder"
_MODE_SINGLE = "Single Image"

# CP2가 save 다이얼로그에 제시하는 결정론적 파일명 stem.
_EXPORT_SUGGESTED_STEM = "folder_inference_results"


# -- portable bundle / fixtures -------------------------------------------------


def _model_spec(name: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )


def _make_canonical_bundle(
    output_dir: Path, *, name: str, write_model_definition: bool = True
) -> None:
    """`output_dir` 아래에 Phase 7 canonical 파일명(`model_definition.json` /
    `best_model_state_dict.pt` / `class_mapping.json`)으로 tiny portable
    bundle을 만든다 -- established 저장 API만 쓰고 새 포맷을 도입하지 않는다.
    가중치는 학습되지 않았지만 `run_single_image_inference`가 실제로 로드/
    forward하는 진짜 state_dict다(fake 결과가 아니다)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _model_spec(name)
    if write_model_definition:
        save_model_spec(spec, output_dir / "model_definition.json")
    save_state_dict(build_model(spec), output_dir / "best_model_state_dict.pt")
    save_class_mapping(_CLASSES, {"cat": 0, "dog": 1}, output_dir / "class_mapping.json")


def _write_model_json(path: Path, name: str) -> None:
    save_model_spec(_model_spec(name), path)


def _write_valid_image(path: Path, color: tuple[int, int, int] = (120, 60, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def _write_corrupt_image(path: Path) -> None:
    """지원 확장자(`.png`)를 가졌지만 PIL이 열 수 없는 파일 -- discovery는
    포함하지만 backend가 이 한 장에서만 예외를 던지게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid PNG payload -- corrupt on purpose")


# -- lifecycle / observation helpers -----------------------------------------


def _folder_thread_cleaned_up(page) -> bool:
    """폴더 `_folder_thread`가 finished -> thread.quit() -> deleteLater()로
    정리됐는지. `_on_folder_thread_finished()`가 cleanup 직후 `_folder_thread`
    를 `None`으로 되돌리므로 `None`도 "정리 완료"로 본다. `QThread` 객체가
    아직 남았지만 C++ 쪽이 해제된 경우(`RuntimeError`)도 "살아 도는 thread
    없음" 신호이므로 정상이다."""
    thread = page._folder_thread
    if thread is None:
        return True
    try:
        return thread.isRunning() is False
    except RuntimeError:
        return True


def _single_thread_cleaned_up(page) -> bool:
    thread = page._thread
    if thread is None:
        return True
    try:
        return thread.isRunning() is False
    except RuntimeError:
        return True


def _read_folder_table(page) -> list[tuple[str, ...]]:
    """폴더 결과 테이블을 표시된 그대로(행 순서 보존) 읽어 온다."""
    table = page._folder_results_table
    rows: list[tuple[str, ...]] = []
    for row in range(table.rowCount()):
        cells = []
        for col in range(table.columnCount()):
            item = table.item(row, col)
            assert item is not None, f"row {row} col {col} has no item"
            cells.append(item.text())
        rows.append(tuple(cells))
    return rows


def _install_export_spy(monkeypatch) -> list[tuple[object, str, str]]:
    """CP1 exporter를 `inference_page` 경계에서 관찰자로 감싼다 -- 실제
    `write_folder_result_export`로 그대로 위임해 파일을 진짜로 쓰되, 매
    호출의 `(result, path, format)`을 기록해 "액션당 정확히 한 번"과 "새
    aggregate만 넘어간다"를 확인한다."""
    calls: list[tuple[object, str, str]] = []

    def _spy(result, path, *, format):
        calls.append((result, str(path), format))
        _real_write_folder_result_export(result, path, format=format)

    monkeypatch.setattr("image_ai_studio.gui.inference_page.write_folder_result_export", _spy)
    return calls


def _patch_save_dialog(monkeypatch, dest: Path) -> dict:
    """`QFileDialog.getSaveFileName`이 고정 경로를 돌려주도록 patch하고,
    다이얼로그가 제시받은 caption/suggested/filter를 기록한다. 다른 GUI
    상호작용은 patch하지 않는다."""
    seen: dict = {}

    def _fake_get_save_file_name(parent, caption, directory="", filter="", *args, **kwargs):
        seen["caption"] = caption
        seen["suggested"] = directory
        seen["filter"] = filter
        return (str(dest), filter)

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(_fake_get_save_file_name)
    )
    return seen


def _start_folder_run(page, folder: Path) -> None:
    page._folder_path_edit.setText(str(folder))
    page._on_run_clicked()


def _finish_folder_run(page, qtbot) -> None:
    """이번 실행에 새로 만들어진 `_folder_worker`의 finished/failed에 plain
    관찰자(`list.append`, CPython에서 atomic)를 붙여, 이 한 번의 실행이
    정확히 finished 1회 / failed 0회만 emit하는지 확인하며 thread cleanup
    까지 기다린다(`tests/gui/test_qt_folder_inference_worker.py`와 동일한
    패턴). cleanup 뒤 `qtbot.wait`로 늦은 중복 emit이 없음을 재확인한다."""
    worker = page._folder_worker
    assert worker is not None, "folder run did not create a worker"
    signals: list[str] = []
    worker.finished.connect(lambda _result: signals.append("finished"))
    worker.failed.connect(lambda _message: signals.append("failed"))
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=30000)
    qtbot.waitUntil(lambda: _folder_thread_cleaned_up(page), timeout=5000)
    qtbot.wait(50)
    assert signals == ["finished"], f"expected exactly one finished signal, got {signals!r}"


def _assert_controls_restored(page) -> None:
    assert page._folder_thread is None
    assert page._folder_worker is None
    assert page._run_button.isEnabled() is True
    assert page._mode_combo.isEnabled() is True
    assert page._folder_path_edit.isEnabled() is True
    assert page._training_output_dir_edit.isEnabled() is True
    assert page._model_json_edit.isEnabled() is True
    assert page._device_combo.isEnabled() is True
    assert page._precision_combo.isEnabled() is True


def _export_via_gui(
    page, monkeypatch, spy_calls, *, fmt: str, dest: Path
) -> str:
    """실제 export 액션(Export CSV / Export JSON 버튼 클릭)을 사용자처럼
    구동한다. save 다이얼로그만 patch하고 CP1 exporter는 진짜로 실행된다.
    호출 identity/경로/포맷/횟수와 상태 라벨을 검증한 뒤 실제로 기록된
    파일 텍스트를 돌려준다."""
    button = page._export_csv_button if fmt == "csv" else page._export_json_button
    assert button.isEnabled() is True, f"{fmt} export action must be enabled before a real export"
    seen = _patch_save_dialog(monkeypatch, dest)

    before = len(spy_calls)
    button.click()

    # 액션당 정확히 한 번의 exporter 호출(중복 invocation 없음).
    assert len(spy_calls) == before + 1, f"exporter must be called exactly once per {fmt} action"
    call_result, call_path, call_fmt = spy_calls[-1]
    assert call_result is page._folder_export_source, "exporter got the retained delivered aggregate"
    assert Path(call_path) == dest
    assert call_fmt == fmt
    assert Path(seen["suggested"]).name == f"{_EXPORT_SUGGESTED_STEM}.{fmt}"
    assert page._status_label.text() == f"Exported {fmt.upper()}: {dest.name}"
    assert "Export failed" not in page._status_label.text()
    text = dest.read_text(encoding="utf-8")
    assert text, "real CP1 exporter must have written the destination file"
    return text


def _assert_export_matches_delivered_and_displayed(
    source, table_rows: list[tuple[str, ...]], csv_text: str, json_text: str
) -> None:
    """파일로 나간 CSV/JSON이 (a) delivered `FolderInferenceResult`를 그대로
    직렬화한 것이고 (b) 화면 테이블에 보이는 각 행과 1:1로 대응하며 경로/
    상태/예측/confidence/probability/duration/error 의미가 동일함을,
    그리고 어떤 결과도 조용히 누락/날조되지 않았음을 확인한다."""
    items = list(source.items)

    # (a) 파일로 나간 바이트 == delivered aggregate를 verbatim 직렬화한 것
    #     (재추론/재계산 없음, 항목 누락/추가 없음).
    assert csv_text == folder_result_to_csv_text(source)
    assert json_text == folder_result_to_json_text(source)

    # CSV 구조: 고정 헤더 한 줄 + 항목당 정확히 한 데이터 행.
    csv_rows = list(csv.reader(io.StringIO(csv_text)))
    assert csv_rows[0] == list(CSV_COLUMNS)
    data_rows = csv_rows[1:]
    assert len(data_rows) == len(items) == len(table_rows)

    # JSON 구조 + 정확한 집계.
    data = json.loads(json_text)
    assert list(data.keys()) == ["format_version", "total", "succeeded", "failed", "items"]
    assert data["format_version"] == EXPORT_FORMAT_VERSION
    succeeded = sum(1 for it in items if it.succeeded)
    failed = sum(1 for it in items if not it.succeeded)
    assert (data["total"], data["succeeded"], data["failed"]) == (len(items), succeeded, failed)
    assert data["total"] == data["succeeded"] + data["failed"]
    assert len(data["items"]) == len(items)

    # delivered / displayed / exported 세 관점이 항목마다 같은 순서로 일치.
    for outcome, csv_row, jitem, shown in zip(items, data_rows, data["items"], table_rows):
        record = dict(zip(CSV_COLUMNS, csv_row))
        full_path = str(outcome.image_path)
        assert record["image_path"] == full_path
        assert jitem["image_path"] == full_path
        assert jitem["image_path"] != outcome.image_path.name  # display name이 아니라 전체 경로
        assert Path(jitem["image_path"]).name == shown[0]  # 테이블은 basename만 보여준다

        if outcome.succeeded:
            res = outcome.result
            assert res is not None
            assert record["status"] == "success"
            assert jitem["status"] == "success"
            assert shown[1] == _STATUS_SUCCESS
            assert record["predicted_class"] == res.predicted_class
            assert jitem["predicted_class"] == res.predicted_class
            assert shown[2] == res.predicted_class
            assert res.predicted_class in _CLASSES  # 날조된 클래스가 아니다
            assert float(record["confidence"]) == res.confidence
            assert jitem["confidence"] == res.confidence
            assert 0.0 <= res.confidence <= 1.0
            sorted_probs = {name: res.probabilities[name] for name in sorted(res.probabilities)}
            assert json.loads(record["probabilities"]) == sorted_probs
            assert jitem["probabilities"] == sorted_probs
            assert list(jitem["probabilities"].keys()) == sorted(res.probabilities)
            assert float(record["inference_duration_seconds"]) == res.inference_duration_seconds
            assert jitem["inference_duration_seconds"] == res.inference_duration_seconds
            assert record["error"] == ""
            assert jitem["error"] is None
            assert _CONFIDENCE_PATTERN.match(shown[3]) is not None
            assert shown[4] == _RESULT_PLACEHOLDER
        else:
            assert record["status"] == "failed"
            assert jitem["status"] == "failed"
            assert shown[1] == _STATUS_FAILURE
            assert record["predicted_class"] == ""
            assert record["confidence"] == ""
            assert record["probabilities"] == ""
            assert record["inference_duration_seconds"] == ""
            assert jitem["predicted_class"] is None
            assert jitem["confidence"] is None
            assert jitem["probabilities"] is None
            assert jitem["inference_duration_seconds"] is None
            assert outcome.error not in (None, "")
            assert record["error"] == outcome.error  # bounded 오류 전문 보존
            assert jitem["error"] == outcome.error
            assert shown[2] == _RESULT_PLACEHOLDER
            assert shown[3] == _RESULT_PLACEHOLDER
            assert shown[4] == outcome.error.splitlines()[0]  # 테이블은 첫 줄만


def _build_folder_window(qtbot, output_dir: Path, *, model_json: Path | None = None):
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._inference_page
    window._tabs.setCurrentWidget(page)

    page._mode_combo.setCurrentText(_MODE_FOLDER)
    page._training_output_dir_edit.setText(str(output_dir))
    if model_json is None:
        assert page._model_json_edit.text() == ""  # auto-discovery
    else:
        page._model_json_edit.setText(str(model_json))  # explicit legacy override
    page._device_combo.setCurrentText("cpu")
    page._precision_combo.setCurrentText("fp32")
    return window, page


# -- main graduation case ---------------------------------------------------


def test_folder_result_export_cpu_graduation_end_to_end(tmp_path: Path, qtbot, monkeypatch) -> None:
    """Phase 11 CP3의 주 사례: 학습 output 디렉터리에 canonical bundle을
    만들고(Model JSON 입력란은 끝까지 비움 -- auto-discovery), 실제
    `MainWindow`/`InferencePage` 폴더 모드로 지원 확장자 이미지 3장
    (유효 2 + 지원 확장자를 가진 깨진 1)을 처리한 뒤, 화면에 표시된
    혼합 배치를 실제 Export CSV / Export JSON 액션과 실제 CP1 exporter로
    저장한다. 검증:

    * status가 `Finished`(per-image 실패가 섞여도 *완료된 배치*다)
    * 발견 순서(파일 이름 오름차순) 그대로 이미지당 정확히 한 행
    * 중간의 깨진 이미지 하나만 격리 실패, 그 뒤 유효 이미지도 성공
    * Total/Succeeded/Failed 집계가 정확히 3/2/1
    * CSV/JSON 둘 다 표시된 폴더 결과당 정확히 한 행/항목을 같은 결정론적
      순서로 담고, 경로/상태/예측/confidence/probability/duration/error
      의미가 delivered aggregate·화면 테이블과 동일하며 JSON 집계 수가
      정확히 3/2/1
    * partial 실패가 두 export 모두에 남아 있고, 깨진 이미지 뒤의 유효
      이미지는 두 export 모두에서 성공이며, 표시/내보낸 결과가 조용히
      누락되거나 날조되지 않음
    * 각 export 액션이 CP1 exporter를 정확히 한 번, retained delivered
      aggregate와 선택 경로로만 호출
    * 같은 `MainWindow`에서 이어지는 두 번째 성공 폴더 실행이 완료 전에
      stale export 데이터를 비우고(액션 비활성화 + export source None),
      새 실행만 중복 행/이전 오류/중복 exporter 호출 없이 내보낸다
    """
    output_dir = tmp_path / "out"
    _make_canonical_bundle(output_dir, name="phase11_cp3_portable_bundle")
    assert (output_dir / "model_definition.json").exists()
    assert (output_dir / "best_model_state_dict.pt").exists()
    assert (output_dir / "class_mapping.json").exists()

    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    images_dir = tmp_path / "batch"
    _write_valid_image(images_dir / "a_first.png", color=(250, 250, 250))
    _write_corrupt_image(images_dir / "b_broken.png")
    _write_valid_image(images_dir / "c_third.png", color=(5, 5, 5))
    # 지원하지 않는 확장자와 하위 폴더는 discovery에서 빠져야 한다.
    (images_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    (images_dir / "nested").mkdir()
    _write_valid_image(images_dir / "nested" / "z_ignored.png")

    _window, page = _build_folder_window(qtbot, output_dir)

    _start_folder_run(page, images_dir)
    _finish_folder_run(page, qtbot)

    assert page._status_label.text() == "Finished"
    assert page._folder_summary_label.text() == "Total: 3  Succeeded: 2  Failed: 1"

    first_source = page._folder_export_source
    assert first_source is not None
    assert (first_source.total, first_source.succeeded, first_source.failed) == (3, 2, 1)
    # 깨진 이미지 뒤의 유효 이미지도 성공으로 남는다.
    assert [it.succeeded for it in first_source.items] == [True, False, True]

    rows = _read_folder_table(page)
    assert len(rows) == 3
    assert [row[0] for row in rows] == ["a_first.png", "b_broken.png", "c_third.png"]
    assert [row[1] for row in rows] == [_STATUS_SUCCESS, _STATUS_FAILURE, _STATUS_SUCCESS]

    assert page._export_csv_button.isEnabled() is True
    assert page._export_json_button.isEnabled() is True

    spy_calls = _install_export_spy(monkeypatch)

    csv_dest = export_dir / "run1.csv"
    json_dest = export_dir / "run1.json"
    csv_text = _export_via_gui(page, monkeypatch, spy_calls, fmt="csv", dest=csv_dest)
    json_text = _export_via_gui(page, monkeypatch, spy_calls, fmt="json", dest=json_dest)

    assert len(spy_calls) == 2  # 정확히 CSV 1회 + JSON 1회
    assert [c[2] for c in spy_calls] == ["csv", "json"]
    assert all(c[0] is first_source for c in spy_calls)

    _assert_export_matches_delivered_and_displayed(first_source, rows, csv_text, json_text)

    # partial 실패가 두 export 모두에 남아 있다(격리된 한 항목만 failed).
    data = json.loads(json_text)
    assert sum(1 for it in data["items"] if it["status"] == "failed") == 1
    assert sum(1 for it in data["items"] if it["status"] == "success") == 2
    assert data["items"][1]["status"] == "failed"
    assert data["items"][1]["error"]  # 비어 있지 않은 bounded 오류
    assert data["items"][2]["status"] == "success"  # 깨진 이미지 뒤의 유효 이미지
    csv_data_rows = list(csv.reader(io.StringIO(csv_text)))[1:]
    assert [r[1] for r in csv_data_rows] == ["success", "failed", "success"]

    # 표시/내보낸 결과가 조용히 누락되지 않는다: 발견 이미지 수 == 테이블
    # 행 수 == CSV 데이터 행 수 == JSON 항목 수.
    assert len(rows) == len(csv_data_rows) == len(data["items"]) == 3
    exported_names = [Path(it["image_path"]).name for it in data["items"]]
    assert exported_names == ["a_first.png", "b_broken.png", "c_third.png"]
    assert "z_ignored.png" not in csv_text and "z_ignored.png" not in json_text
    assert "notes.txt" not in csv_text and "notes.txt" not in json_text

    _assert_controls_restored(page)

    # -- 같은 창에서 이어지는 두 번째 성공 실행 ----------------------------
    rerun_dir = tmp_path / "rerun"
    _write_valid_image(rerun_dir / "x_one.png", color=(10, 200, 10))
    _write_valid_image(rerun_dir / "y_two.png", color=(200, 10, 10))

    _start_folder_run(page, rerun_dir)
    # 완료 전에 stale export 데이터가 비워진다(동기적으로, 배치가 끝나기 전).
    assert page._status_label.text() == "Running"
    assert page._folder_export_source is None
    assert page._export_csv_button.isEnabled() is False
    assert page._export_json_button.isEnabled() is False
    assert page._folder_results_table.rowCount() == 0

    _finish_folder_run(page, qtbot)

    assert page._status_label.text() == "Finished"
    assert "Failed" not in page._status_label.text()
    assert page._folder_summary_label.text() == "Total: 2  Succeeded: 2  Failed: 0"

    rerun_source = page._folder_export_source
    assert rerun_source is not None
    assert rerun_source is not first_source  # 새 aggregate

    rerun_rows = _read_folder_table(page)
    assert [row[0] for row in rerun_rows] == ["x_one.png", "y_two.png"]
    assert [row[1] for row in rerun_rows] == [_STATUS_SUCCESS, _STATUS_SUCCESS]

    del spy_calls[:]  # rerun export 호출만 따로 센다
    rerun_csv_dest = export_dir / "run2.csv"
    rerun_json_dest = export_dir / "run2.json"
    rerun_csv_text = _export_via_gui(page, monkeypatch, spy_calls, fmt="csv", dest=rerun_csv_dest)
    rerun_json_text = _export_via_gui(page, monkeypatch, spy_calls, fmt="json", dest=rerun_json_dest)

    # 새 실행만 내보낸다: exporter는 새 aggregate로만, 액션당 한 번만 불린다.
    assert len(spy_calls) == 2
    assert all(c[0] is rerun_source for c in spy_calls)
    assert all(c[0] is not first_source for c in spy_calls)

    _assert_export_matches_delivered_and_displayed(
        rerun_source, rerun_rows, rerun_csv_text, rerun_json_text
    )

    # 중복 행 없음(3 -> 2), stale 오류 없음, 이전 실행의 이미지 흔적 없음.
    rerun_data = json.loads(rerun_json_text)
    assert (rerun_data["total"], rerun_data["succeeded"], rerun_data["failed"]) == (2, 2, 0)
    assert all(it["status"] == "success" for it in rerun_data["items"])
    assert all(it["error"] is None for it in rerun_data["items"])
    rerun_csv_data_rows = list(csv.reader(io.StringIO(rerun_csv_text)))[1:]
    assert len(rerun_csv_data_rows) == 2
    assert all(r[6] == "" for r in rerun_csv_data_rows)  # error 열 전부 빈 문자열
    for stale_marker in ("a_first.png", "b_broken.png", "c_third.png"):
        assert stale_marker not in rerun_csv_text
        assert stale_marker not in rerun_json_text

    # 첫 실행의 export 파일은 rerun 이후에도 그대로다(새 결과만 나갔다).
    assert csv_dest.read_text(encoding="utf-8") == folder_result_to_csv_text(first_source)
    assert json_dest.read_text(encoding="utf-8") == folder_result_to_json_text(first_source)

    _assert_controls_restored(page)


# -- legacy explicit Model JSON override still works with export -------------


def test_folder_result_export_preserves_legacy_model_json_override(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    """Phase 7 이전 형태의 output 디렉터리(`model_definition.json` 없음)도
    사용자가 원본 Model JSON을 명시적으로 고르면 폴더 모드로 추론하고 그
    혼합 결과를 CSV/JSON으로 그대로 내보낼 수 있어야 한다 -- explicit
    override가 항상 우선하는 Phase 7 CP2 계약이 export 경로에서도 유지되는지
    확인한다."""
    output_dir = tmp_path / "legacy_out"
    _make_canonical_bundle(
        output_dir, name="phase11_cp3_legacy_compat", write_model_definition=False
    )
    assert not (output_dir / "model_definition.json").exists()

    model_json_path = tmp_path / "legacy_model.json"
    _write_model_json(model_json_path, "phase11_cp3_legacy_compat")

    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    images_dir = tmp_path / "batch"
    _write_valid_image(images_dir / "one.png")
    _write_corrupt_image(images_dir / "two_broken.jpg")
    _write_valid_image(images_dir / "three.jpg")

    _window, page = _build_folder_window(qtbot, output_dir, model_json=model_json_path)

    _start_folder_run(page, images_dir)
    _finish_folder_run(page, qtbot)

    assert page._status_label.text() == "Finished"
    assert page._folder_summary_label.text() == "Total: 3  Succeeded: 2  Failed: 1"

    source = page._folder_export_source
    assert source is not None
    rows = _read_folder_table(page)
    assert [row[0] for row in rows] == ["one.png", "three.jpg", "two_broken.jpg"]
    assert [row[1] for row in rows] == [_STATUS_SUCCESS, _STATUS_SUCCESS, _STATUS_FAILURE]

    spy_calls = _install_export_spy(monkeypatch)
    csv_text = _export_via_gui(
        page, monkeypatch, spy_calls, fmt="csv", dest=export_dir / "legacy.csv"
    )
    json_text = _export_via_gui(
        page, monkeypatch, spy_calls, fmt="json", dest=export_dir / "legacy.json"
    )

    assert len(spy_calls) == 2
    assert all(c[0] is source for c in spy_calls)
    _assert_export_matches_delivered_and_displayed(source, rows, csv_text, json_text)

    data = json.loads(json_text)
    assert (data["total"], data["succeeded"], data["failed"]) == (3, 2, 1)
    assert [it["status"] for it in data["items"]] == ["success", "success", "failed"]
    assert data["items"][2]["error"]

    _assert_controls_restored(page)


# -- single-image integration behavior is unaffected ------------------------


def test_single_image_inference_unaffected_by_folder_export_actions(
    tmp_path: Path, qtbot
) -> None:
    """CP3가 폴더 결과 export를 졸업시키면서도 기존 단일 이미지 통합 동작이
    그대로임을 이 fixed allowlist 안에서 함께 지킨다: 같은 canonical bundle을
    기본(Single Image) 모드로 auto-discovery 추론해 실제 결과가 표시되고
    thread가 정리되며, 폴더 결과 영역과 두 export 액션은 손대지 않은 채
    비활성/비어 있어야 한다(단일 이미지 실행은 export source를 만들지 않는다)."""
    output_dir = tmp_path / "out"
    _make_canonical_bundle(output_dir, name="phase11_cp3_single_image")

    image_path = tmp_path / "input.png"
    _write_valid_image(image_path)

    window = MainWindow()
    qtbot.addWidget(window)
    page = window._inference_page
    window._tabs.setCurrentWidget(page)

    assert page._mode_combo.currentText() == _MODE_SINGLE  # 기본 모드
    assert page._folder_export_source is None
    assert page._export_csv_button.isEnabled() is False
    assert page._export_json_button.isEnabled() is False

    page._training_output_dir_edit.setText(str(output_dir))
    assert page._model_json_edit.text() == ""
    page._image_path_edit.setText(str(image_path))
    page._device_combo.setCurrentText("cpu")
    page._precision_combo.setCurrentText("fp32")

    page._on_run_clicked()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=30000)
    qtbot.waitUntil(lambda: _single_thread_cleaned_up(page), timeout=5000)

    assert page._status_label.text() == "Finished"
    assert page._predicted_class_value_label.text() in _CLASSES
    assert _CONFIDENCE_PATTERN.match(page._confidence_value_label.text()) is not None
    assert _DURATION_PATTERN.match(page._duration_value_label.text()) is not None
    assert page._probabilities_value_label.text() != _RESULT_PLACEHOLDER

    # 단일 이미지 실행은 폴더 결과 테이블도, export 상태도 건드리지 않는다.
    assert page._folder_results_table.rowCount() == 0
    assert page._folder_export_source is None
    assert page._export_csv_button.isEnabled() is False
    assert page._export_json_button.isEnabled() is False
    assert page._thread is None
    assert page._worker is None
    assert page._run_button.isEnabled() is True
