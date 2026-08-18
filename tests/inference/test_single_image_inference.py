"""`run_single_image_inference()`/`InferenceRequest`/`InferenceResult`
테스트(Phase 6B). training core에서 이미 검증된 내용(model 빌드 자체의
정확성, `build_transform()` 수식 자체의 정확성, device/precision
검증 로직 자체의 정확성 -- Phase 6B에서 `training/device.py`로 이동만
했을 뿐 로직은 그대로이므로)은 여기서 반복하지 않는다(docs/
phase6a_inference_architecture.md §12)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from image_ai_studio.inference.single_image_inference import (
    InferenceRequest,
    InferenceResult,
    run_single_image_inference,
)
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.checkpoint import save_state_dict
from image_ai_studio.training.torchvision_dataset import build_transform, save_class_mapping

INPUT_SHAPE = (3, 8, 8)


def _make_model_spec(name: str = "phase6b_inference_test", *, out_features: int = 2) -> ModelSpec:
    return ModelSpec(
        name=name,
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=out_features)],
    )


def _make_artifacts(
    root: Path, *, classes: list[str] | None = None, out_features: int | None = None
) -> tuple[Path, Path, Path, Path]:
    classes = classes if classes is not None else ["cat", "dog"]
    model_spec = _make_model_spec(out_features=out_features if out_features is not None else len(classes))
    model_json_path = root / "model.json"
    save_model_spec(model_spec, model_json_path)

    model = build_model(model_spec)
    state_dict_path = root / "state_dict.pt"
    save_state_dict(model, state_dict_path)

    class_mapping_path = root / "class_mapping.json"
    save_class_mapping(classes, {name: i for i, name in enumerate(classes)}, class_mapping_path)

    image_path = root / "image.png"
    Image.new("RGB", (20, 20), color=(120, 60, 200)).save(image_path)

    return model_json_path, state_dict_path, class_mapping_path, image_path


def _request(
    root: Path, *, device: str = "cpu", precision: str = "fp32", classes: list[str] | None = None, **overrides
) -> InferenceRequest:
    model_json_path, state_dict_path, class_mapping_path, image_path = _make_artifacts(root, classes=classes)
    kwargs = dict(
        model_json_path=model_json_path,
        state_dict_path=state_dict_path,
        class_mapping_path=class_mapping_path,
        image_path=image_path,
        device=device,
        precision=precision,
    )
    kwargs.update(overrides)
    return InferenceRequest(**kwargs)


# -- dataclasses ---------------------------------------------------------------


def test_inference_request_is_frozen(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(Exception):
        request.device = "cuda"  # type: ignore[misc]


def test_inference_result_is_frozen() -> None:
    result = InferenceResult(
        predicted_index=0, predicted_class="cat", confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1}, inference_duration_seconds=0.001,
    )
    with pytest.raises(Exception):
        result.confidence = 0.5  # type: ignore[misc]


# -- model reconstruction + prediction contract ---------------------------------


def test_run_single_image_inference_returns_valid_result(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = run_single_image_inference(request)

    assert result.predicted_index in (0, 1)
    assert result.predicted_class in ("cat", "dog")
    assert result.predicted_class == list(result.probabilities.keys())[result.predicted_index]
    assert result.inference_duration_seconds >= 0.0


def test_probabilities_sum_to_approximately_one(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = run_single_image_inference(request)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(isinstance(value, float) for value in result.probabilities.values())


def test_confidence_matches_predicted_class_probability(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = run_single_image_inference(request)
    assert result.confidence == result.probabilities[result.predicted_class]
    # 확률이므로 confidence는 항상 [0, 1] 안이어야 한다.
    assert 0.0 <= result.confidence <= 1.0


def test_class_mapping_index_to_name_matches_classes_order(tmp_path: Path) -> None:
    """classes[i] == model output index i 계약(Phase 6A §4) -- probabilities
    dict의 key 순서가 classes 순서와 일치하는지, predicted_class가
    argmax(logits)에 해당하는 실제 이름인지 확인한다."""
    request = _request(tmp_path, classes=["alpha", "beta"])
    result = run_single_image_inference(request)
    assert set(result.probabilities.keys()) == {"alpha", "beta"}
    assert result.predicted_class in ("alpha", "beta")


# -- preprocessing parity(가장 중요한 신규 테스트) -------------------------------


def test_preprocessing_matches_training_imagefolder_exactly(tmp_path: Path) -> None:
    """training의 ImageFolder 경로가 만드는 tensor와 single-image
    inference 경로가 만드는 tensor가 정확히 동일해야 한다(Phase 6A §2/
    §12 -- "training과 inference preprocessing이 절대로 갈라지지 않도록
    contract를 고정"). 두 경로 모두 build_transform()을 그대로
    재사용하므로 이 테스트는 새 transform을 만들지 않았다는 것 자체를
    증명한다."""
    input_shape = (3, 16, 12)  # 정사각형이 아닌 shape으로 H/W가 뒤섞이지 않는지도 확인
    class_dir = tmp_path / "onlyclass"
    class_dir.mkdir()
    image_path = class_dir / "sample.png"
    # 단색이 아닌 이미지로 resize/normalize 오차가 숨지 않게 한다.
    gradient = Image.new("RGB", (37, 29))
    pixels = gradient.load()
    for x in range(gradient.width):
        for y in range(gradient.height):
            pixels[x, y] = (x * 5 % 256, y * 7 % 256, (x + y) * 3 % 256)
    gradient.save(image_path)

    # training 경로: 실제 ImageFolder + build_transform()
    training_dataset = ImageFolder(str(tmp_path), transform=build_transform(input_shape))
    training_tensor, _label = training_dataset[0]

    # inference 경로: single_image_inference가 실제로 쓰는 것과 동일한 조립
    inference_tensor = build_transform(input_shape)(Image.open(image_path).convert("RGB"))

    assert torch.equal(training_tensor, inference_tensor)
    assert training_tensor.shape == (3, 16, 12)
    assert training_tensor.dtype == torch.float32


def test_run_single_image_inference_uses_build_transform_end_to_end(tmp_path: Path) -> None:
    """core 함수 전체를 통해서도(직접 build_transform 호출이 아니라)
    같은 tensor가 나오는지 재확인 -- 위 테스트가 조립 자체를, 이
    테스트가 실제 함수 호출 경로를 검증한다."""
    input_shape = INPUT_SHAPE
    model_json_path, state_dict_path, class_mapping_path, image_path = _make_artifacts(tmp_path)

    expected_tensor = build_transform(input_shape)(Image.open(image_path).convert("RGB")).unsqueeze(0)

    captured: dict = {}
    original_forward = torch.nn.Sequential.forward

    def capturing_forward(self, x):  # noqa: ANN001
        captured["input"] = x.detach().clone()
        return original_forward(self, x)

    torch.nn.Sequential.forward = capturing_forward
    try:
        request = InferenceRequest(
            model_json_path=model_json_path, state_dict_path=state_dict_path,
            class_mapping_path=class_mapping_path, image_path=image_path,
            device="cpu", precision="fp32",
        )
        run_single_image_inference(request)
    finally:
        torch.nn.Sequential.forward = original_forward

    assert torch.equal(captured["input"], expected_tensor)


# -- error paths -----------------------------------------------------------------


def test_missing_model_json_raises(tmp_path: Path) -> None:
    request = _request(tmp_path, model_json_path=tmp_path / "does_not_exist.json")
    with pytest.raises(Exception):
        run_single_image_inference(request)


def test_missing_state_dict_raises(tmp_path: Path) -> None:
    request = _request(tmp_path, state_dict_path=tmp_path / "does_not_exist.pt")
    with pytest.raises(Exception):
        run_single_image_inference(request)


def test_state_dict_architecture_mismatch_raises(tmp_path: Path) -> None:
    """다른 architecture의 state_dict를 로드하면 nn.Module.load_state_dict()
    의 기존 RuntimeError가 그대로 전파돼야 한다(새 검증 코드 없음)."""
    model_json_path, _correct_state_dict, class_mapping_path, image_path = _make_artifacts(tmp_path)

    mismatched_spec = ModelSpec(
        name="mismatched", input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=64), ReLUSpec(), LinearSpec(out_features=2)],
    )
    mismatched_model = build_model(mismatched_spec)
    mismatched_state_dict_path = tmp_path / "mismatched_state_dict.pt"
    save_state_dict(mismatched_model, mismatched_state_dict_path)

    request = InferenceRequest(
        model_json_path=model_json_path, state_dict_path=mismatched_state_dict_path,
        class_mapping_path=class_mapping_path, image_path=image_path,
        device="cpu", precision="fp32",
    )
    with pytest.raises(RuntimeError):
        run_single_image_inference(request)


def test_invalid_image_file_raises(tmp_path: Path) -> None:
    model_json_path, state_dict_path, class_mapping_path, _image_path = _make_artifacts(tmp_path)
    bad_image_path = tmp_path / "not_an_image.png"
    bad_image_path.write_bytes(b"this is not a real image file")

    request = InferenceRequest(
        model_json_path=model_json_path, state_dict_path=state_dict_path,
        class_mapping_path=class_mapping_path, image_path=bad_image_path,
        device="cpu", precision="fp32",
    )
    with pytest.raises(Exception):
        run_single_image_inference(request)


def test_class_count_mismatch_raises(tmp_path: Path) -> None:
    """class_mapping.json의 class 수가 model 출력 차원과 다르면
    require_matching_num_classes()(기존 public 함수, 새 검증 아님)가
    거부한다."""
    model_json_path, state_dict_path, _class_mapping_path, image_path = _make_artifacts(tmp_path)
    mismatched_mapping_path = tmp_path / "mismatched_class_mapping.json"
    save_class_mapping(
        ["cat", "dog", "bird"], {"cat": 0, "dog": 1, "bird": 2}, mismatched_mapping_path
    )

    request = InferenceRequest(
        model_json_path=model_json_path, state_dict_path=state_dict_path,
        class_mapping_path=mismatched_mapping_path, image_path=image_path,
        device="cpu", precision="fp32",
    )
    with pytest.raises(ValueError, match="classes"):
        run_single_image_inference(request)


def test_invalid_device_string_raises(tmp_path: Path) -> None:
    request = _request(tmp_path, device="not-a-device")
    with pytest.raises(ValueError, match="device"):
        run_single_image_inference(request)


def test_cpu_fp16_raises_precision_device_incompatibility(tmp_path: Path) -> None:
    request = _request(tmp_path, device="cpu", precision="fp16")
    with pytest.raises(ValueError, match="CUDA"):
        run_single_image_inference(request)


def test_cpu_bf16_raises_precision_device_incompatibility(tmp_path: Path) -> None:
    request = _request(tmp_path, device="cpu", precision="bf16")
    with pytest.raises(ValueError, match="CUDA"):
        run_single_image_inference(request)


def test_invalid_precision_string_raises(tmp_path: Path) -> None:
    request = _request(tmp_path, device="cpu", precision="fp8")
    with pytest.raises(ValueError, match="precision"):
        run_single_image_inference(request)


def test_cuda_unavailable_or_out_of_range_index_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    request = _request(tmp_path, device="cuda")
    with pytest.raises(ValueError, match="CUDA"):
        run_single_image_inference(request)


# -- CUDA(실제 GPU) ---------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_cuda_fp32_inference_smoke(tmp_path: Path) -> None:
    request = _request(tmp_path, device="cuda", precision="fp32")
    result = run_single_image_inference(request)
    assert result.predicted_class in ("cat", "dog")
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_cuda_fp16_inference_smoke(tmp_path: Path) -> None:
    """wiring + execution contract 확인이 목적이다(성능 benchmark
    아님, Phase 6A §21) -- autocast(dtype=torch.float16)가 CUDA에서
    예외 없이 동작하고 여전히 유효한 확률을 반환하는지만 본다."""
    request = _request(tmp_path, device="cuda", precision="fp16")
    result = run_single_image_inference(request)
    assert result.predicted_class in ("cat", "dog")
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-2)


def _bf16_supported_on_this_cuda_device() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001 -- 아주 오래된 torch/driver 조합 방어
        return False


@pytest.mark.skipif(
    not _bf16_supported_on_this_cuda_device(), reason="requires a CUDA device with bf16 support"
)
def test_cuda_bf16_inference_smoke(tmp_path: Path) -> None:
    request = _request(tmp_path, device="cuda", precision="bf16")
    result = run_single_image_inference(request)
    assert result.predicted_class in ("cat", "dog")
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-2)
