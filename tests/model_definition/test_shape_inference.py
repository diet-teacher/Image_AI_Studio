"""Shape inference: per-layer shape computation and layer-connection validation."""
from __future__ import annotations

import pytest

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.shape_inference import format_shape_trace, infer_model_shapes
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


def test_conv2d_same_padding_preserves_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 224, 224),
        layers=[Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].input_shape == (3, 224, 224)
    assert trace[0].output_shape == (32, 224, 224)
    assert trace[0].inferred == {"in_channels": 3}


def test_conv2d_stride_2_halves_spatial_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 32, 32),
        layers=[Conv2dSpec(out_channels=8, kernel_size=3, stride=2, padding=1)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (8, 16, 16)


def test_max_pool2d_default_stride_equals_kernel_size() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 224, 224),
        layers=[MaxPool2dSpec(kernel_size=2)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32, 112, 112)


def test_adaptive_avg_pool2d_shape() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 55, 55),
        layers=[AdaptiveAvgPool2dSpec(output_size=1)],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32, 1, 1)


def test_flatten_collapses_to_1d() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 112, 112),
        layers=[FlattenSpec()],
    )
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (32 * 112 * 112,)


def test_linear_infers_in_features_from_previous_layer() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(32, 112, 112),
        layers=[FlattenSpec(), LinearSpec(out_features=10)],
    )
    trace = infer_model_shapes(spec)
    flatten_info, linear_info = trace
    assert flatten_info.output_shape == (401408,)
    assert linear_info.input_shape == (401408,)
    assert linear_info.inferred == {"in_features": 401408}
    assert linear_info.output_shape == (10,)


@pytest.mark.parametrize("layer", [ReLUSpec(), DropoutSpec(p=0.3)])
def test_activation_and_dropout_preserve_shape(layer) -> None:
    spec = ModelSpec(name="m", input_shape=(3, 8, 8), layers=[layer])
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (3, 8, 8)


def test_batch_norm2d_preserves_shape_and_infers_num_features() -> None:
    spec = ModelSpec(name="m", input_shape=(16, 8, 8), layers=[BatchNorm2dSpec()])
    trace = infer_model_shapes(spec)
    assert trace[0].output_shape == (16, 8, 8)
    assert trace[0].inferred == {"num_features": 16}


def test_full_example_pipeline_matches_expected_shapes() -> None:
    """Input -> Conv2d -> MaxPool2d -> Flatten -> Linear, as in the Phase 1 spec example."""
    spec = ModelSpec(
        name="example_model",
        input_shape=(3, 224, 224),
        layers=[
            Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1),
            MaxPool2dSpec(kernel_size=2, stride=2),
            FlattenSpec(),
            LinearSpec(out_features=10),
        ],
    )
    trace = infer_model_shapes(spec)
    shapes = [(info.input_shape, info.output_shape) for info in trace]
    assert shapes == [
        ((3, 224, 224), (32, 224, 224)),
        ((32, 224, 224), (32, 112, 112)),
        ((32, 112, 112), (401408,)),
        ((401408,), (10,)),
    ]


def test_conv2d_after_flatten_raises_clear_validation_error() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[FlattenSpec(), Conv2dSpec(out_channels=4, kernel_size=3)],
    )
    with pytest.raises(ModelValidationError, match=r"Layer 1 \(Conv2d\).*3D"):
        infer_model_shapes(spec)


def test_linear_before_flatten_raises_clear_validation_error() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 8, 8),
        layers=[LinearSpec(out_features=10)],
    )
    with pytest.raises(ModelValidationError, match=r"Layer 0 \(Linear\)"):
        infer_model_shapes(spec)


def test_pooling_that_collapses_spatial_size_to_zero_raises() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 4, 4),
        layers=[MaxPool2dSpec(kernel_size=5, stride=1, padding=0)],
    )
    with pytest.raises(ModelValidationError, match="MaxPool2d"):
        infer_model_shapes(spec)


def test_conv2d_that_collapses_spatial_size_to_zero_raises() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 2, 2),
        layers=[Conv2dSpec(out_channels=4, kernel_size=5, stride=1, padding=0)],
    )
    with pytest.raises(ModelValidationError, match="Conv2d"):
        infer_model_shapes(spec)


def test_format_shape_trace_renders_layer_name_and_shapes() -> None:
    spec = ModelSpec(
        name="m",
        input_shape=(3, 224, 224),
        layers=[Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1)],
    )
    trace = infer_model_shapes(spec)
    text = format_shape_trace(trace)
    assert text == "Conv2d\n[3, 224, 224] -> [32, 224, 224]"
