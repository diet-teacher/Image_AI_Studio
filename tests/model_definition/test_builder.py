"""ModelSpec -> torch.nn.Module: module types, auto-inferred sizes, and a real forward pass."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.errors import ModelValidationError
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
)


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


def test_build_model_raises_model_validation_error_for_invalid_connection() -> None:
    spec = ModelSpec(
        name="bad",
        input_shape=(3, 8, 8),
        layers=[FlattenSpec(), Conv2dSpec(out_channels=4, kernel_size=3)],
    )
    with pytest.raises(ModelValidationError, match="Conv2d"):
        build_model(spec)
