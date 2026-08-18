"""실제 `run_single_image_inference()`(fake 아님)를 `QtInferenceWorker`
+ 진짜 `QThread`를 통해 CPU/CUDA에서 한 번 돌리는 최소 통합 테스트
(Phase 6B, Phase 5B의 `test_qt_training_worker_integration.py`와 동일한
철학). inference core correctness 자체는 `tests/inference/`가 전담
하므로, 여기서는 application/GUI wiring이 실제 production inference
core와 맞물려 정상 동작하는지, 그리고 QThread deleteLater 계약이
지켜지는지만 확인한다(중복 검증 금지).

**Phase 6B stabilization**: `qtbot.waitSignal(worker.finished, ...)`을
쓰지 않는다 -- `worker.finished`에는 이미 `worker.deleteLater()`가
연결돼 있어(canonical wiring), `qtbot.waitSignal()`의 임시
`SignalBlocker`가 같은 signal에 connect/disconnect하는 것이 (worker
thread에서 처리되는) 실제 `deleteLater()` 삭제와 경합하는 것을
실측으로 재현했다(`tests/gui/test_qt_inference_worker.py` 모듈
docstring의 "libpyside: Failed to disconnect ... from signal
finished" RuntimeWarning 참고, 드문 native access violation의 설명
가능한 원인). 대신 영구 연결(plain 함수) + `qtbot.waitUntil()`
polling만 쓴다."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from PySide6.QtCore import QThread

from image_ai_studio.application.inference_controller import InferenceController, build_inference_request
from image_ai_studio.gui.qt_inference_worker import QtInferenceWorker
from image_ai_studio.inference.single_image_inference import InferenceResult
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.checkpoint import save_state_dict
from image_ai_studio.training.torchvision_dataset import save_class_mapping

INPUT_SHAPE = (3, 8, 8)


def _make_artifacts(root: Path) -> tuple[Path, Path, Path, Path]:
    model_spec = ModelSpec(
        name="phase6b_qt_worker_integration",
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )
    model_json_path = root / "model.json"
    save_model_spec(model_spec, model_json_path)

    model = build_model(model_spec)
    state_dict_path = root / "state_dict.pt"
    save_state_dict(model, state_dict_path)

    class_mapping_path = root / "class_mapping.json"
    save_class_mapping(["cat", "dog"], {"cat": 0, "dog": 1}, class_mapping_path)

    image_path = root / "image.png"
    Image.new("RGB", (20, 20), color=(120, 60, 200)).save(image_path)

    return model_json_path, state_dict_path, class_mapping_path, image_path


def _wire_full_lifecycle(controller: InferenceController, request) -> tuple[QThread, QtInferenceWorker]:
    thread = QThread()
    worker = QtInferenceWorker(controller, request)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)  # worker 자신의 신호에!
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker


def _start_and_capture(
    thread: QThread, worker: QtInferenceWorker, qtbot, timeout: int = 30000
) -> tuple[str, object]:
    """`qtbot.waitSignal()`을 의도적으로 쓰지 않는다(모듈 docstring
    참고) -- 영구 연결(plain 함수) + `qtbot.waitUntil()` polling만
    쓴다."""
    events: list[tuple[str, object]] = []
    worker.finished.connect(lambda result: events.append(("finished", result)))
    worker.failed.connect(lambda message: events.append(("failed", message)))
    thread.start()
    qtbot.waitUntil(lambda: len(events) == 1, timeout=timeout)
    return events[0]


def _wait_for_thread_cleanup(thread: QThread, qtbot, timeout: int = 5000) -> None:
    """`QThread`의 실행이 끝날 때까지 기다린다(`isRunning() is False`).
    `thread` 객체가 이미 `deleteLater()`로 삭제된 경우(`RuntimeError`)
    도 정리 완료로 간주한다. `isRunning() is False`가 된 시점과
    `deleteLater()`로 예약된 deferred QObject 삭제가 실제로 처리된
    시점이 동일하다는 보장은 없다 -- 이 helper는 그 deferred deletion
    자체를 강제하거나 별도로 검증하지 않는다."""

    def _cleaned_up() -> bool:
        try:
            return thread.isRunning() is False
        except RuntimeError:
            return True

    qtbot.waitUntil(_cleaned_up, timeout=timeout)


def test_qt_worker_runs_real_cpu_inference_end_to_end(tmp_path: Path, qtbot) -> None:
    model_json_path, state_dict_path, class_mapping_path, image_path = _make_artifacts(tmp_path)

    request = build_inference_request(
        model_json_path=model_json_path,
        state_dict_path=state_dict_path,
        class_mapping_path=class_mapping_path,
        image_path=image_path,
        device="cpu",
        precision="fp32",
    )

    controller = InferenceController()  # 기본 backend == run_single_image_inference (fake 아님)
    thread, worker = _wire_full_lifecycle(controller, request)

    kind, result = _start_and_capture(thread, worker, qtbot)

    assert kind == "finished"
    assert isinstance(result, InferenceResult)
    assert controller.state == "finished"
    assert result.predicted_class in ("cat", "dog")
    assert result.predicted_index in (0, 1)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
    assert result.confidence == result.probabilities[result.predicted_class]
    assert result.inference_duration_seconds >= 0.0

    _wait_for_thread_cleanup(thread, qtbot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_qt_worker_runs_real_cuda_inference_off_the_gui_thread(tmp_path: Path, qtbot) -> None:
    """Phase 6A §8/§9: QThread 안에서 실제 device="cuda" production
    inference(model 재구성부터 `.to("cuda")`, forward까지 전부)가
    정상 동작하는지 확인한다 -- CUDA inference correctness 자체는
    `tests/inference/`가 이미 검증하므로, 이 테스트는 오직 "QThread
    integration 경계가 정상인가"만 본다(Phase 5B의 CUDA wiring
    테스트와 동일한 철학, fp32 1개면 충분)."""
    model_json_path, state_dict_path, class_mapping_path, image_path = _make_artifacts(tmp_path)

    request = build_inference_request(
        model_json_path=model_json_path,
        state_dict_path=state_dict_path,
        class_mapping_path=class_mapping_path,
        image_path=image_path,
        device="cuda",
        precision="fp32",
    )

    controller = InferenceController()
    thread, worker = _wire_full_lifecycle(controller, request)

    kind, result = _start_and_capture(thread, worker, qtbot)

    assert kind == "finished"
    assert controller.state == "finished"
    assert result.predicted_class in ("cat", "dog")
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5

    _wait_for_thread_cleanup(thread, qtbot)
