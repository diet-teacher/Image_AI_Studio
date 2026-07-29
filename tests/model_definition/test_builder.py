"""ModelSpec -> torch.nn.Module 빌드 테스트: 모듈 타입, 자동 추론 크기, forward 검증."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.shape_inference import infer_model_shapes
from image_ai_studio.model_definition.specs import (
    AdaptiveAvgPool2dSpec,
    BatchNorm2dSpec,
    Conv2dSpec,
    DropoutSpec,
    FlattenSpec,
    LinearSpec,
    MaxPool2dSpec,
    ModelSpec,
    ReLUSpec,
    ResidualBlockSpec,
)
from image_ai_studio.models.residual_block import ResidualBlock


def _example_model_spec() -> ModelSpec:
    return ModelSpec(
        name="example_model",
        input_shape=(3, 224, 224),
        layers=[
            Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2dSpec(),
            ReLUSpec(),
            MaxPool2dSpec(kernel_size=2, stride=2),
            FlattenSpec(),
            LinearSpec(out_features=10),
        ],
    )


def test_build_model_returns_sequential_with_expected_module_types() -> None:
    model = build_model(_example_model_spec())
    assert isinstance(model, nn.Sequential)
    assert [type(m) for m in model] == [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.Flatten,
        nn.Linear,
    ]


def test_conv2d_in_channels_inferred_from_model_input_shape() -> None:
    model = build_model(_example_model_spec())
    conv = model[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 3
    assert conv.out_channels == 32


def test_batch_norm_num_features_inferred_from_preceding_conv() -> None:
    model = build_model(_example_model_spec())
    bn = model[1]
    assert isinstance(bn, nn.BatchNorm2d)
    assert bn.num_features == 32


def test_linear_in_features_inferred_from_flatten() -> None:
    model = build_model(_example_model_spec())
    linear = model[-1]
    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 32 * 112 * 112
    assert linear.out_features == 10


def test_all_supported_layer_types_build_and_forward() -> None:
    spec = ModelSpec(
        name="all_layers",
        input_shape=(3, 16, 16),
        layers=[
            Conv2dSpec(out_channels=8, kernel_size=3, padding=1),
            BatchNorm2dSpec(),
            ReLUSpec(),
            MaxPool2dSpec(kernel_size=2),
            Conv2dSpec(out_channels=4, kernel_size=3, padding=1),
            AdaptiveAvgPool2dSpec(output_size=1),
            FlattenSpec(),
            DropoutSpec(p=0.5),
            LinearSpec(out_features=5),
        ],
    )
    model = build_model(spec).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 16, 16))
    assert output.shape == (2, 5)


def test_forward_pass_matches_shape_inference_output() -> None:
    spec = _example_model_spec()
    model = build_model(spec).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, *spec.input_shape))
    assert tuple(output.shape) == (1, 10)


def test_flatten_preserves_batch_dimension_matching_shape_inference() -> None:
    """nn.Flatten()도 shape_inference와 동일하게 batch 차원 유지: [N,C,H,W] -> [N,C*H*W]."""
    spec = ModelSpec(name="m", input_shape=(3, 4, 4), layers=[FlattenSpec()])
    expected_flat_shape = infer_model_shapes(spec)[0].output_shape
    assert expected_flat_shape == (3 * 4 * 4,)

    model = build_model(spec).eval()
    batch_size = 5
    with torch.inference_mode():
        output = model(torch.randn(batch_size, *spec.input_shape))
    assert tuple(output.shape) == (batch_size, *expected_flat_shape)


def test_build_model_raises_model_validation_error_for_invalid_connection() -> None:
    spec = ModelSpec(
        name="bad",
        input_shape=(3, 8, 8),
        layers=[FlattenSpec(), Conv2dSpec(out_channels=4, kernel_size=3)],
    )
    with pytest.raises(ModelValidationError, match="Conv2d"):
        build_model(spec)


# -- ResidualBlockSpec --------------------------------------------------------


def test_residual_block_builds_as_existing_residual_block_class() -> None:
    spec = ModelSpec(name="m", input_shape=(3, 8, 8), layers=[ResidualBlockSpec(out_channels=8)])
    model = build_model(spec)
    assert isinstance(model[0], ResidualBlock)


def test_residual_block_identity_shortcut_when_channels_and_stride_match() -> None:
    """in_channels == out_channels, stride == 1이면 shortcut이 nn.Identity()인지 확인."""
    spec = ModelSpec(name="m", input_shape=(8, 8, 8), layers=[ResidualBlockSpec(out_channels=8, stride=1)])
    block = build_model(spec)[0]
    assert isinstance(block.shortcut, nn.Identity)

    block = block.eval()
    with torch.inference_mode():
        output = block(torch.randn(2, 8, 8, 8))
    assert tuple(output.shape) == (2, 8, 8, 8)


@pytest.mark.parametrize(
    "in_channels,out_channels,stride",
    [(8, 16, 1), (8, 8, 2), (8, 16, 2)],
)
def test_residual_block_projection_shortcut_when_channels_or_stride_differ(
    in_channels: int, out_channels: int, stride: int
) -> None:
    """in_channels != out_channels 또는 stride != 1이면 shortcut이 Conv+BN projection인지 확인."""
    spec = ModelSpec(
        name="m",
        input_shape=(in_channels, 8, 8),
        layers=[ResidualBlockSpec(out_channels=out_channels, stride=stride)],
    )
    block = build_model(spec)[0]
    assert isinstance(block.shortcut, nn.Sequential)

    block = block.eval()
    with torch.inference_mode():
        output = block(torch.randn(2, in_channels, 8, 8))
    assert output.shape[1] == out_channels


def test_residual_block_forward_shape_matches_shape_inference() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 15, 15),
        layers=[
            Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1),
            ResidualBlockSpec(out_channels=16, stride=2),
        ],
    )
    expected_shape = infer_model_shapes(spec)[-1].output_shape

    model = build_model(spec).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, *spec.input_shape))
    assert tuple(output.shape[1:]) == expected_shape
