"""Parameter-level validation: each LayerSpec / ModelSpec must reject bad
values in __post_init__, independent of shape context."""
from __future__ import annotations

import pytest

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.specs import (
    Conv2dSpec,
    DropoutSpec,
    LinearSpec,
    MaxPool2dSpec,
    ModelSpec,
    ReLUSpec,
)


@pytest.mark.parametrize("out_channels", [0, -1])
def test_conv2d_rejects_non_positive_out_channels(out_channels: int) -> None:
    with pytest.raises(ModelValidationError, match="out_channels"):
        Conv2dSpec(out_channels=out_channels, kernel_size=3)


@pytest.mark.parametrize("kernel_size", [0, -3])
def test_conv2d_rejects_non_positive_kernel_size(kernel_size: int) -> None:
    with pytest.raises(ModelValidationError, match="kernel_size"):
        Conv2dSpec(out_channels=16, kernel_size=kernel_size)


def test_conv2d_rejects_non_positive_stride() -> None:
    with pytest.raises(ModelValidationError, match="stride"):
        Conv2dSpec(out_channels=16, kernel_size=3, stride=0)


def test_conv2d_rejects_negative_padding() -> None:
    with pytest.raises(ModelValidationError, match="padding"):
        Conv2dSpec(out_channels=16, kernel_size=3, padding=-1)


def test_conv2d_accepts_valid_params() -> None:
    spec = Conv2dSpec(out_channels=16, kernel_size=3, stride=1, padding=1)
    assert spec.out_channels == 16
    assert spec.padding == 1


def test_conv2d_missing_required_field_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Conv2dSpec(kernel_size=3)  # type: ignore[call-arg]


@pytest.mark.parametrize("kernel_size", [0, -1])
def test_max_pool2d_rejects_non_positive_kernel_size(kernel_size: int) -> None:
    with pytest.raises(ModelValidationError, match="kernel_size"):
        MaxPool2dSpec(kernel_size=kernel_size)


def test_max_pool2d_stride_defaults_to_kernel_size() -> None:
    spec = MaxPool2dSpec(kernel_size=2)
    assert spec.stride is None
    assert spec.effective_stride == 2


def test_max_pool2d_explicit_stride_overrides_default() -> None:
    spec = MaxPool2dSpec(kernel_size=2, stride=1)
    assert spec.effective_stride == 1


@pytest.mark.parametrize("out_features", [0, -5])
def test_linear_rejects_non_positive_out_features(out_features: int) -> None:
    with pytest.raises(ModelValidationError, match="out_features"):
        LinearSpec(out_features=out_features)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_dropout_rejects_p_outside_unit_interval(p: float) -> None:
    with pytest.raises(ModelValidationError, match="'p'"):
        DropoutSpec(p=p)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_dropout_accepts_boundary_and_mid_p(p: float) -> None:
    assert DropoutSpec(p=p).p == p


def test_relu_defaults_to_not_inplace() -> None:
    assert ReLUSpec().inplace is False


def test_model_spec_rejects_empty_name() -> None:
    with pytest.raises(ModelValidationError, match="name"):
        ModelSpec(name="", input_shape=(3, 224, 224), layers=[])


def test_model_spec_rejects_blank_name() -> None:
    with pytest.raises(ModelValidationError, match="name"):
        ModelSpec(name="   ", input_shape=(3, 224, 224), layers=[])


@pytest.mark.parametrize("input_shape", [(3, 224), (3, 224, 224, 1), (0, 224, 224), (3, -1, 224)])
def test_model_spec_rejects_invalid_input_shape(input_shape: tuple) -> None:
    with pytest.raises(ModelValidationError, match="input_shape"):
        ModelSpec(name="m", input_shape=input_shape, layers=[])


def test_model_spec_converts_list_input_shape_to_tuple() -> None:
    spec = ModelSpec(name="m", input_shape=[3, 224, 224], layers=[])  # type: ignore[arg-type]
    assert spec.input_shape == (3, 224, 224)
