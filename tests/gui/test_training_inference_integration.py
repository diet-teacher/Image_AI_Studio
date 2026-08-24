"""Phase 6D CP1: proves a model trained through the real `TrainingPage` in a
single `MainWindow` session can immediately be used for inference through the
same window's `InferencePage`, with no fake training/inference backend
anywhere. Both `tests/gui/test_training_page_integration.py` (real CPU
training through `TrainingPage`) and `tests/gui/test_qt_inference_worker_integration.py`
(real CPU inference through `QtInferenceWorker`) already cover each half in
isolation -- this test's only job is the handoff between them inside one
`MainWindow`: the exact `best_model_state_dict.pt`/`class_mapping.json` that
training just produced must be consumable by inference without any
substitution or re-derivation (no duplicate correctness checks here)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec

INPUT_SHAPE = (3, 8, 8)
_CLASS_COLORS = {"cat": (250, 250, 250), "dog": (5, 5, 5)}

# Phase 6C's `_format_confidence`/`_format_probabilities`/`_format_duration_ms`
# (image_ai_studio/gui/inference_page.py) contract: two-decimal percentages
# and two-decimal millisecond durations. Real inference values aren't known
# ahead of time, so the formatting itself is checked by pattern instead of by
# recomputing the underlying numbers.
_CONFIDENCE_PATTERN = re.compile(r"^\d{1,3}\.\d{2}%$")
_PROBABILITY_LINE_PATTERN = re.compile(r"^(.+): (\d{1,3}\.\d{2})%$")
_DURATION_PATTERN = re.compile(r"^\d+\.\d{2} ms$")


def _make_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for class_name, color in _CLASS_COLORS.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (20, 20), color=color).save(class_dir / f"{i}.png")


def _write_model_json(path: Path) -> None:
    spec = ModelSpec(
        name="phase6d_cp1_training_to_inference",
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )
    save_model_spec(spec, path)


def _thread_cleaned_up(page) -> bool:
    """`_thread`는 finished -> thread.quit() -> deleteLater()로 정리된다.
    `InferencePage`는 `_on_thread_finished()`에서 cleanup 직후 `_thread`를
    `None`으로 되돌리므로(`TrainingPage`는 그렇게 하지 않는다) 두 페이지
    모두를 이 helper 하나로 다루려면 `None`도 "정리 완료"로 봐야 한다.
    `_thread`가 아직 `QThread` 객체로 남아 있는 경우(TrainingPage)에는
    deleteLater() 처리 후 C++ 쪽 객체가 해제되면 더 이상 접근할 수 없다
    (RuntimeError) -- 이 역시 "살아서 도는 thread가 남아있지 않다"는
    신호이므로 정상이다(test_training_page_integration.py와
    test_qt_inference_worker_integration.py의 동일 패턴 재사용)."""
    thread = page._thread
    if thread is None:
        return True
    try:
        return thread.isRunning() is False
    except RuntimeError:
        return True


def test_phase6d_cp1_training_to_inference_real_cpu(tmp_path: Path, qtbot) -> None:
    _make_dataset(tmp_path)
    model_json_path = tmp_path / "model.json"
    _write_model_json(model_json_path)
    output_dir = tmp_path / "out"

    window = MainWindow()
    qtbot.addWidget(window)

    training_page = window._training_page
    inference_page = window._inference_page

    # -- real TrainingController/QtTrainingWorker/run_imagefolder_training_workflow,
    #    driven the same way a user would from the Training tab -------------
    training_page._model_json_edit.setText(str(model_json_path))
    training_page._dataset_root_edit.setText(str(tmp_path))
    training_page._output_dir_edit.setText(str(output_dir))
    training_page._epochs_spin.setValue(1)
    training_page._batch_size_spin.setValue(4)
    training_page._learning_rate_spin.setValue(1e-2)
    training_page._export_torchscript_check.setChecked(False)
    training_page._device_combo.setCurrentText("cpu")

    training_page._on_start_clicked()
    qtbot.waitUntil(lambda: training_page._start_button.isEnabled(), timeout=30000)

    assert training_page._status_label.text() == "Completed"
    assert training_page._stop_button.isEnabled() is False

    best_state_dict_path = output_dir / "best_model_state_dict.pt"
    class_mapping_path = output_dir / "class_mapping.json"
    assert best_state_dict_path.exists()
    assert class_mapping_path.exists()

    qtbot.waitUntil(lambda: _thread_cleaned_up(training_page), timeout=5000)

    # -- switch tabs inside the same MainWindow, no second MainWindow ------
    window._tabs.setCurrentWidget(inference_page)

    # A real dataset image (untouched by training) as the inference input.
    image_path = tmp_path / "test" / "cat" / "0.png"

    inference_page._training_output_dir_edit.setText(str(output_dir))
    inference_page._model_json_edit.setText(str(model_json_path))
    inference_page._image_path_edit.setText(str(image_path))
    inference_page._device_combo.setCurrentText("cpu")
    inference_page._precision_combo.setCurrentText("fp32")

    inference_page._on_run_clicked()
    qtbot.waitUntil(lambda: inference_page._run_button.isEnabled(), timeout=30000)

    assert inference_page._status_label.text() == "Finished"

    class_mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    predicted_class = inference_page._predicted_class_value_label.text()
    assert predicted_class in class_mapping["classes"]

    # Probabilities: one "class: NN.NN%" line per class, sorted by class
    # name (Phase 6C `_format_probabilities` contract), values ~sum to 100%.
    probabilities_text = inference_page._probabilities_value_label.text()
    assert probabilities_text != "--"
    probability_lines = probabilities_text.splitlines()
    parsed_probabilities: dict[str, float] = {}
    for line in probability_lines:
        match = _PROBABILITY_LINE_PATTERN.match(line)
        assert match is not None, f"unexpected probability line format: {line!r}"
        parsed_probabilities[match.group(1)] = float(match.group(2))
    assert list(parsed_probabilities.keys()) == sorted(class_mapping["classes"])
    assert set(parsed_probabilities.keys()) == set(class_mapping["classes"])
    assert abs(sum(parsed_probabilities.values()) - 100.0) < 0.1

    # Confidence: "NN.NN%" and, per the InferenceResult contract
    # (confidence == probabilities[predicted_class]), formatted identically
    # to the predicted class's own probabilities line.
    confidence_text = inference_page._confidence_value_label.text()
    assert _CONFIDENCE_PATTERN.match(confidence_text) is not None
    predicted_line = next(line for line in probability_lines if line.startswith(f"{predicted_class}: "))
    assert predicted_line == f"{predicted_class}: {confidence_text}"

    # Duration: "NN.NN ms" (Phase 6C `_format_duration_ms` contract).
    duration_text = inference_page._duration_value_label.text()
    assert _DURATION_PATTERN.match(duration_text) is not None

    # -- controls restored after cleanup ------------------------------------
    assert inference_page._run_button.isEnabled() is True
    assert inference_page._model_json_edit.isEnabled() is True
    assert inference_page._training_output_dir_edit.isEnabled() is True
    assert inference_page._image_path_edit.isEnabled() is True
    assert inference_page._device_combo.isEnabled() is True
    assert inference_page._precision_combo.isEnabled() is True

    qtbot.waitUntil(lambda: _thread_cleaned_up(inference_page), timeout=5000)
