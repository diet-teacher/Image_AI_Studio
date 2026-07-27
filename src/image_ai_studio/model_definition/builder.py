"""Builds a torch.nn.Module from a ModelSpec.

Phase 1 only supports a flat layer list, so the result is always a
plain ``nn.Sequential``. Layer parameters that depend on the previous
layer's output shape (``Conv2d.in_channels``, ``BatchNorm2d.num_features``,
``Linear.in_features``) are filled in from ``shape_inference`` rather
than requiring the caller (or a future UI) to compute them.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from torch import nn

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.shape_inference import LayerShapeInfo
from image_ai_studio.model_definition.specs import (
    AdaptiveAvgPool2dSpec,
    BatchNorm2dSpec,
    Conv2dSpec,
    DropoutSpec,
    FlattenSpec,
    LayerSpec,
    LinearSpec,
    MaxPool2dSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.model_definition.validation import validate_model_spec


def _build_conv2d(layer: Conv2dSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.Conv2d(
        in_channels=inferred["in_channels"],
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
    )


def _build_batch_norm2d(layer: BatchNorm2dSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.BatchNorm2d(num_features=inferred["num_features"], eps=layer.eps, momentum=layer.momentum)


def _build_relu(layer: ReLUSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.ReLU(inplace=layer.inplace)


def _build_max_pool2d(layer: MaxPool2dSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.MaxPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)


def _build_adaptive_avg_pool2d(layer: AdaptiveAvgPool2dSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.AdaptiveAvgPool2d(output_size=layer.output_size)


def _build_flatten(layer: FlattenSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.Flatten()


def _build_linear(layer: LinearSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.Linear(in_features=inferred["in_features"], out_features=layer.out_features, bias=layer.bias)


def _build_dropout(layer: DropoutSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.Dropout(p=layer.p)


_BuilderFn = Callable[[LayerSpec, Dict[str, int]], nn.Module]

_BUILDERS: Dict[Type[LayerSpec], _BuilderFn] = {
    Conv2dSpec: _build_conv2d,
    BatchNorm2dSpec: _build_batch_norm2d,
    ReLUSpec: _build_relu,
    MaxPool2dSpec: _build_max_pool2d,
    AdaptiveAvgPool2dSpec: _build_adaptive_avg_pool2d,
    FlattenSpec: _build_flatten,
    LinearSpec: _build_linear,
    DropoutSpec: _build_dropout,
}


def _build_layer(info: LayerShapeInfo) -> nn.Module:
    builder_fn = _BUILDERS.get(type(info.layer))
    if builder_fn is None:
        raise ModelValidationError(
            f"Layer {info.index}: no PyTorch builder is registered for layer type "
            f"{type(info.layer).__name__!r}"
        )
    return builder_fn(info.layer, info.inferred)


def build_model(model_spec: ModelSpec) -> nn.Sequential:
    """Validate ``model_spec`` (via shape inference) and build a torch.nn.Sequential.

    Raises ModelValidationError if the spec is invalid; never returns a
    partially-built module.
    """
    shape_trace = validate_model_spec(model_spec)
    modules = [_build_layer(info) for info in shape_trace]
    return nn.Sequential(*modules)
