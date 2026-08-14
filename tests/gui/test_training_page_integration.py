"""실제 `TrainingPage` -> 실제 `TrainingController` -> 실제
`QtTrainingWorker` -> 실제 `run_imagefolder_training_workflow()`를 CPU에서
한 번 끝까지 돌리는 최소 통합 테스트(Phase 5C §55). Phase 4의 학습
correctness 회귀는 `tests/training/`가, application/GUI wiring 자체는
Phase 5B의 `tests/gui/test_qt_training_worker_integration.py`가 이미
전담하므로, 여기서는 "실제 GUI 화면이 실제 학습 결과를 올바르게
보여주는가"만 확인한다(중복 검증 금지)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from image_ai_studio.gui.training_page import TrainingPage
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec

INPUT_SHAPE = (3, 8, 8)
_CLASS_COLORS = {"cat": (250, 250, 250), "dog": (5, 5, 5)}


def _make_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for class_name, color in _CLASS_COLORS.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (20, 20), color=color).save(class_dir / f"{i}.png")


def _write_model_json(path: Path) -> None:
    spec = ModelSpec(
        name="phase5c_training_page_integration",
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )
    save_model_spec(spec, path)


def test_training_page_runs_real_cpu_workflow_end_to_end(tmp_path: Path, qtbot) -> None:
    _make_dataset(tmp_path)
    model_json_path = tmp_path / "model.json"
    _write_model_json(model_json_path)
    output_dir = tmp_path / "out"

    page = TrainingPage()
    qtbot.addWidget(page)

    page._model_json_edit.setText(str(model_json_path))
    page._dataset_root_edit.setText(str(tmp_path))
    page._output_dir_edit.setText(str(output_dir))
    page._epochs_spin.setValue(1)
    page._batch_size_spin.setValue(4)
    page._learning_rate_spin.setValue(1e-2)
    page._export_torchscript_check.setChecked(False)
    page._device_combo.setCurrentText("cpu")

    page._on_start_clicked()
    qtbot.waitUntil(lambda: page._start_button.isEnabled(), timeout=30000)

    assert page._status_label.text() == "Completed"
    assert page._stop_button.isEnabled() is False

    assert page._test_loss_label.text() != "--"
    assert page._test_accuracy_label.text() != "--"

    assert page._artifact_labels["best_model_state_dict_path"].text() == str(
        output_dir / "best_model_state_dict.pt"
    )
    assert page._artifact_labels["training_history_path"].text() == str(
        output_dir / "training_history.json"
    )
    assert page._artifact_labels["class_mapping_path"].text() == str(
        output_dir / "class_mapping.json"
    )
    assert page._artifact_labels["test_result_path"].text() == str(output_dir / "test_result.json")
    # torchscript export가 꺼져 있으므로 해당 artifact는 생성되지 않아야 한다.
    assert page._artifact_labels["torchscript_model_path"].text() == "Not generated"

    assert (output_dir / "best_model_state_dict.pt").exists()
    assert (output_dir / "training_history.json").exists()

    # worker/thread은 finished -> thread.quit() -> deleteLater()로 정리된다.
    # deleteLater()가 실행되고 나면 C++ 쪽 QThread 객체가 해제되어 더
    # 이상 접근할 수 없다(RuntimeError) -- 이는 "살아서 도는 thread가
    # 남아있지 않다"는 것을 보여주는 신호이므로 정상이다.
    def _thread_cleaned_up() -> bool:
        try:
            return page._thread.isRunning() is False
        except RuntimeError:
            return True

    qtbot.waitUntil(_thread_cleaned_up, timeout=5000)
