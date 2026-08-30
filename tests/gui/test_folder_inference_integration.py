"""Phase 10 CP4: 실제 CPU 폴더 추론 졸업 통합 테스트.

CP1(`tests/inference/test_folder_inference.py`) / CP2
(`tests/application/test_folder_inference_controller.py`,
`tests/gui/test_qt_folder_inference_worker.py`) / CP3
(`tests/gui/test_inference_page.py`)은 discovery/조립/오류 격리/집계/
controller lifecycle/Qt worker thread-affinity/페이지 상태 전이를 fake
backend와 CPU 이미지로 이미 고정한다. 이 모듈의 책임은 그 조각들이
**실제 `MainWindow` + `InferencePage`의 비동기 폴더 경로**에서, **fake가
아닌 실제 `run_single_image_inference` backend**와 맞물려 canonical
portable bundle 하나를 끝까지 소비하는지 한 번 증명하는 것이다
(`tests/gui/test_training_inference_integration.py`의 단일 이미지 handoff
테스트와 동일한 철학 -- 중복 correctness 검증은 하지 않는다).

전부 CPU 전용이다: 작은 로컬 이미지/모델만 쓰고 CUDA, 외부 서비스,
네트워크, packaging 도구, 스크린샷 비교를 요구하지 않으며 pytest
임시 디렉터리 밖의 저장소 아티팩트를 건드리지 않는다. Phase 6B의
single-image public API와 Phase 7 portable artifact 포맷/경로는 이
모듈에서 전혀 바뀌지 않는다 -- 여기서는 그것들을 소비만 한다.

**Phase 6B stabilization 계약 준수**: `qtbot.waitSignal()`을 쓰지 않는다
(canonical wiring이 이미 worker 자신의 finished/failed에
`deleteLater()`를 연결해 두어, `waitSignal()`의 임시 `SignalBlocker`
connect/disconnect가 그 삭제와 경합하는 것이 실측됐다). 대신
`InferencePage`가 노출하는 관측 가능한 상태(status label, summary
label, 결과 테이블, `_run_button.isEnabled()`, `_folder_thread`) +
`qtbot.waitUntil()` polling만 쓴다.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.checkpoint import save_state_dict
from image_ai_studio.training.torchvision_dataset import save_class_mapping

INPUT_SHAPE = (3, 8, 8)
_CLASSES = ["cat", "dog"]

# Phase 6C `_format_confidence`/`_format_duration_ms` 계약: 소수점 2자리
# 퍼센트, 소수점 2자리 밀리초. 실제 추론값은 미리 알 수 없으므로 형식만
# 패턴으로 확인한다(값을 다시 계산하지 않는다).
_CONFIDENCE_PATTERN = re.compile(r"^\d{1,3}\.\d{2}%$")
_DURATION_PATTERN = re.compile(r"^\d+\.\d{2} ms$")

# InferencePage가 폴더 결과 테이블 셀에 쓰는 값(image_ai_studio/gui/
# inference_page.py의 고정 계약). 문자열 리터럴을 여기 한 곳에 모아 둔다.
_STATUS_SUCCESS = "Success"
_STATUS_FAILURE = "Failure"
_RESULT_PLACEHOLDER = "--"
_MODE_FOLDER = "Folder"
_MODE_SINGLE = "Single Image"


def _model_spec(name: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )


def _make_canonical_bundle(output_dir: Path, *, name: str, write_model_definition: bool = True) -> None:
    """`output_dir` 아래에 Phase 7 canonical 파일명으로 tiny portable
    bundle을 만든다 -- established 저장 API(`save_model_spec` /
    `save_state_dict` / `save_class_mapping`)만 쓰고, 새 포맷을 도입하지
    않는다. 가중치는 학습되지 않은 상태지만 `run_single_image_inference`가
    실제로 로드/forward하는 진짜 state_dict다(fake 결과가 아니다)."""
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
    포함하지만 backend가 이미지 한 장에서만 예외를 던지게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid PNG payload -- corrupt on purpose")


def _folder_thread_cleaned_up(page) -> bool:
    """폴더 `_folder_thread`가 finished -> thread.quit() -> deleteLater()로
    정리됐는지. `InferencePage._on_folder_thread_finished()`가 cleanup
    직후 `_folder_thread`를 `None`으로 되돌리므로 `None`도 "정리 완료"로
    본다. 아직 `QThread` 객체가 남아 있고 C++ 쪽이 이미 해제된 경우
    (`RuntimeError`)도 "살아 도는 thread 없음" 신호이므로 정상이다
    (test_qt_folder_inference_worker.py의 동일 패턴)."""
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


def _run_folder_mode(page, qtbot, *, folder: Path) -> None:
    """실제 `InferencePage` 폴더 경로를 사용자가 하듯 구동하고, 실행과
    그 뒤 thread cleanup까지 끝나기를 기다린다.

    이번 실행에 새로 만들어진 `_folder_worker`의 `finished`/`failed`에
    plain 관찰자(`list.append`, CPython에서 atomic)를 영구 연결해 --
    `tests/gui/test_qt_folder_inference_worker.py`와 동일한 패턴 -- 이
    한 번의 실행이 정확히 `finished` 1회, `failed` 0회만 emit하는지
    확인한다(중복/추가/누락 신호 전달 방지). cleanup 뒤에도 `qtbot.wait`
    으로 여유를 두어 늦은 두 번째 emit이 없음을 재확인한다. 이 helper를
    쓰는 모든 폴더 실행(첫 실행 / 같은 창 rerun / legacy override)이
    이 보장을 함께 받는다."""
    page._folder_path_edit.setText(str(folder))
    page._on_run_clicked()
    worker = page._folder_worker
    assert worker is not None, "folder run did not create a worker"
    signals: list[str] = []
    worker.finished.connect(lambda _result: signals.append("finished"))
    worker.failed.connect(lambda _message: signals.append("failed"))
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=30000)
    qtbot.waitUntil(lambda: _folder_thread_cleaned_up(page), timeout=5000)
    qtbot.wait(50)  # 늦은 중복/추가 emit이 있으면 여기서 잡힌다
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


def test_folder_inference_cpu_graduation_end_to_end(tmp_path: Path, qtbot) -> None:
    """Phase 10 CP4의 주 사례: 학습 output 디렉터리에 canonical bundle을
    만들고(Model JSON 입력란은 끝까지 비움 -- auto-discovery), 실제
    `MainWindow`/`InferencePage` 폴더 모드로 지원 확장자 이미지 3장
    (유효 2 + 깨진 1)을 처리한다. 검증:

    * status가 `Finished`(per-image 실패가 섞여도 *완료된 배치*다)
    * 발견 순서(파일 이름 오름차순) 그대로 이미지당 정확히 한 행
    * 중간의 깨진 이미지 하나만 격리 실패, 그 뒤 유효 이미지도 완료
    * Total/Succeeded/Failed 집계 값이 정확히 3/2/1
    * 성공 행은 실제 예측 클래스/confidence(재계산 아님), 실패 행은
      bounded 오류 첫 줄
    * 혼합 완료 뒤 Run/입력 복원 + worker/thread cleanup
    * 매 폴더 실행이 정확히 `finished` 1회 / `failed` 0회만 emit한다
      (`_run_folder_mode`가 실행마다 관찰자로 확인)
    * 같은 창에서 이어지는 두 번째 폴더 실행이 중복 행/이전 오류/중복
      신호 없이 성공(행 수가 3 -> 2로 줄어 leftover 행이 있으면 잡힌다)
    """
    output_dir = tmp_path / "out"
    _make_canonical_bundle(output_dir, name="phase10_cp4_portable_bundle")
    assert (output_dir / "model_definition.json").exists()
    assert (output_dir / "best_model_state_dict.pt").exists()
    assert (output_dir / "class_mapping.json").exists()

    images_dir = tmp_path / "batch"
    _write_valid_image(images_dir / "a_first.png", color=(250, 250, 250))
    _write_corrupt_image(images_dir / "b_broken.png")
    _write_valid_image(images_dir / "c_third.png", color=(5, 5, 5))
    # 지원하지 않는 확장자와 하위 폴더는 discovery에서 빠져야 한다.
    (images_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    (images_dir / "nested").mkdir()
    _write_valid_image(images_dir / "nested" / "z_ignored.png")

    window = MainWindow()
    qtbot.addWidget(window)
    page = window._inference_page
    window._tabs.setCurrentWidget(page)

    page._mode_combo.setCurrentText(_MODE_FOLDER)
    page._training_output_dir_edit.setText(str(output_dir))
    assert page._model_json_edit.text() == ""  # auto-discovery, 끝까지 비움
    page._device_combo.setCurrentText("cpu")
    page._precision_combo.setCurrentText("fp32")

    _run_folder_mode(page, qtbot, folder=images_dir)

    assert page._status_label.text() == "Finished"
    assert page._folder_summary_label.text() == "Total: 3  Succeeded: 2  Failed: 1"

    rows = _read_folder_table(page)
    assert len(rows) == 3
    assert [row[0] for row in rows] == ["a_first.png", "b_broken.png", "c_third.png"]
    assert [row[1] for row in rows] == [_STATUS_SUCCESS, _STATUS_FAILURE, _STATUS_SUCCESS]

    # 독립적으로 센 집계도 summary 라벨과 일치해야 한다.
    assert sum(1 for row in rows if row[1] == _STATUS_SUCCESS) == 2
    assert sum(1 for row in rows if row[1] == _STATUS_FAILURE) == 1

    for idx in (0, 2):  # 유효 이미지 -- 실제 추론 결과
        assert rows[idx][2] in _CLASSES
        assert _CONFIDENCE_PATTERN.match(rows[idx][3]) is not None
        assert rows[idx][4] == _RESULT_PLACEHOLDER

    broken = rows[1]  # 격리된 실패
    assert broken[2] == _RESULT_PLACEHOLDER
    assert broken[3] == _RESULT_PLACEHOLDER
    assert broken[4] not in ("", _RESULT_PLACEHOLDER)
    assert "\n" not in broken[4]  # 첫 줄만 표시

    _assert_controls_restored(page)

    # -- 같은 창에서 이어지는 두 번째 성공 실행 ----------------------------
    rerun_dir = tmp_path / "rerun"
    _write_valid_image(rerun_dir / "x_one.png", color=(10, 200, 10))
    _write_valid_image(rerun_dir / "y_two.png", color=(200, 10, 10))

    _run_folder_mode(page, qtbot, folder=rerun_dir)

    assert page._status_label.text() == "Finished"
    assert "Failed" not in page._status_label.text()
    assert page._folder_summary_label.text() == "Total: 2  Succeeded: 2  Failed: 0"

    rerun_rows = _read_folder_table(page)
    assert len(rerun_rows) == 2  # 이전 실행의 세 번째 행이 남지 않았다
    assert [row[0] for row in rerun_rows] == ["x_one.png", "y_two.png"]
    assert [row[1] for row in rerun_rows] == [_STATUS_SUCCESS, _STATUS_SUCCESS]
    for row in rerun_rows:
        assert row[2] in _CLASSES
        assert _CONFIDENCE_PATTERN.match(row[3]) is not None
        assert row[4] == _RESULT_PLACEHOLDER  # 이전 실행의 오류 텍스트가 남지 않았다

    _assert_controls_restored(page)


def test_folder_inference_explicit_model_json_override_for_legacy_bundle(tmp_path: Path, qtbot) -> None:
    """Phase 7 이전 형태의 output 디렉터리(`model_definition.json` 없음)도
    사용자가 원본 Model JSON을 명시적으로 고르면 폴더 모드에서 그대로
    쓸 수 있어야 한다 -- explicit override가 항상 우선하는 Phase 7 CP2
    계약이 폴더 경로에서도 유지되는지 확인한다."""
    output_dir = tmp_path / "legacy_out"
    _make_canonical_bundle(output_dir, name="phase10_cp4_legacy_compat", write_model_definition=False)
    assert not (output_dir / "model_definition.json").exists()

    model_json_path = tmp_path / "legacy_model.json"
    _write_model_json(model_json_path, "phase10_cp4_legacy_compat")

    images_dir = tmp_path / "batch"
    _write_valid_image(images_dir / "one.png")
    _write_valid_image(images_dir / "two.jpg")

    window = MainWindow()
    qtbot.addWidget(window)
    page = window._inference_page
    window._tabs.setCurrentWidget(page)

    page._mode_combo.setCurrentText(_MODE_FOLDER)
    page._training_output_dir_edit.setText(str(output_dir))
    page._model_json_edit.setText(str(model_json_path))  # explicit override
    page._device_combo.setCurrentText("cpu")
    page._precision_combo.setCurrentText("fp32")

    _run_folder_mode(page, qtbot, folder=images_dir)

    assert page._status_label.text() == "Finished"
    assert page._folder_summary_label.text() == "Total: 2  Succeeded: 2  Failed: 0"

    rows = _read_folder_table(page)
    assert [row[0] for row in rows] == ["one.png", "two.jpg"]
    assert [row[1] for row in rows] == [_STATUS_SUCCESS, _STATUS_SUCCESS]
    for row in rows:
        assert row[2] in _CLASSES
        assert _CONFIDENCE_PATTERN.match(row[3]) is not None

    _assert_controls_restored(page)


def test_single_image_inference_still_works_in_same_window(tmp_path: Path, qtbot) -> None:
    """CP4가 폴더 모드를 졸업시키면서도 기존 단일 이미지 통합 동작이
    그대로임을 이 fixed allowlist 안에서 함께 지킨다: 같은 canonical
    bundle을 기본(Single Image) 모드로 auto-discovery 추론해 실제 결과가
    표시되고 thread가 정리되며, 폴더 결과 영역은 비어 있어야 한다."""
    output_dir = tmp_path / "out"
    _make_canonical_bundle(output_dir, name="phase10_cp4_single_image")

    image_path = tmp_path / "input.png"
    _write_valid_image(image_path)

    window = MainWindow()
    qtbot.addWidget(window)
    page = window._inference_page
    window._tabs.setCurrentWidget(page)

    assert page._mode_combo.currentText() == _MODE_SINGLE  # 기본 모드
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

    # 단일 이미지 실행은 폴더 결과 테이블을 건드리지 않는다.
    assert page._folder_results_table.rowCount() == 0
    assert page._thread is None
    assert page._worker is None
    assert page._run_button.isEnabled() is True
