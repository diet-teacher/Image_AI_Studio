"""state_dict 저장/재로드 테스트."""
from __future__ import annotations

from pathlib import Path

import torch

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import (
    BatchNorm2dSpec,
    Conv2dSpec,
    FlattenSpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.training.checkpoint import load_state_dict, save_state_dict


def _spec() -> ModelSpec:
    return ModelSpec(
        name="checkpoint_model",
        input_shape=(3, 8, 8),
        layers=[
            Conv2dSpec(out_channels=4, kernel_size=3, padding=1),
            BatchNorm2dSpec(),
            ReLUSpec(),
            FlattenSpec(),
            LinearSpec(out_features=4),
        ],
    )


def test_save_and_load_state_dict_reproduces_same_output(tmp_path: Path) -> None:
    torch.manual_seed(0)
    spec = _spec()
    original_model = build_model(spec).eval()

    example_input = torch.randn(2, *spec.input_shape)
    with torch.inference_mode():
        original_output = original_model(example_input)

    state_dict_path = tmp_path / "model_state_dict.pt"
    save_state_dict(original_model, state_dict_path)
    assert state_dict_path.exists()

    torch.manual_seed(999)  # 새 모델은 일부러 다른 초기값에서 시작
    new_model = build_model(spec).eval()
    with torch.inference_mode():
        new_output_before_load = new_model(example_input)
    # load 전에는 서로 다른 가중치라 출력도 달라야 함 (load가 실제로 뭔가
    # 바꾼다는 것을 증명하기 위한 대조군)
    assert not torch.allclose(original_output, new_output_before_load)

    load_state_dict(new_model, state_dict_path)
    with torch.inference_mode():
        reloaded_output = new_model(example_input)

    assert torch.allclose(original_output, reloaded_output)


def test_save_state_dict_creates_parent_directories(tmp_path: Path) -> None:
    model = build_model(_spec())
    nested_path = tmp_path / "nested" / "dir" / "model.pt"
    save_state_dict(model, nested_path)
    assert nested_path.exists()
