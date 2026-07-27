"""Per-layer tensor shape computation for a ModelSpec (batch dim excluded).

This is the primary validation surface for how layers connect to each
other: a mismatched shape between two adjacent layers (e.g. Conv2d
receiving an already-flattened 1D tensor, or a pooling layer shrinking a
spatial dimension to zero) is caught here with a message that names the
offending layer, before any torch.nn.Module is built.

It also resolves the layer parameters that depend on the previous
layer's output shape (Conv2d.in_channels, BatchNorm2d.num_features,
Linear.in_features) so callers -- currently validation.py, and
transitively builder.py -- never need to duplicate this arithmetic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from image_ai_studio.model_definition.errors import ModelValidationError
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

Shape = tuple[int, ...]


@dataclass
class LayerShapeInfo:
    """One entry in a model's shape trace.

    ``inferred`` holds any layer parameters that were derived from
    ``input_shape`` rather than supplied in the spec (e.g.
    ``{"in_features": 401408}`` for a Linear layer following a Flatten).
    """

    index: int
    layer: LayerSpec
    input_shape: Shape
    output_shape: Shape
    inferred: dict[str, int] = field(default_factory=dict)

    @property
    def layer_name(self) -> str:
        return type(self.layer).__name__.removesuffix("Spec")


def _layer_name(layer: LayerSpec) -> str:
    return type(layer).__name__.removesuffix("Spec")


def _require_rank(layer: LayerSpec, input_shape: Shape, index: int, rank: int, expected_desc: str) -> None:
    if len(input_shape) != rank:
        raise ModelValidationError(
            f"Layer {index} ({_layer_name(layer)}): expected a {rank}D input shape "
            f"{expected_desc} but got shape {tuple(input_shape)} "
            f"({len(input_shape)}D). Check the layer connected before this one."
        )


def _spatial_output_size(
    size: int, kernel_size: int, stride: int, padding: int, *, index: int, layer_name: str, dim_name: str
) -> int:
    out = math.floor((size + 2 * padding - kernel_size) / stride) + 1
    if out <= 0:
        raise ModelValidationError(
            f"Layer {index} ({layer_name}): output {dim_name} would be {out} (<= 0) given "
            f"input {dim_name} {size}, kernel_size={kernel_size}, stride={stride}, "
            f"padding={padding}. Reduce kernel_size/stride or increase padding/input size."
        )
    return out


def _conv2d_shape(layer: Conv2dSpec, input_shape: Shape, index: int) -> tuple[Shape, dict[str, int]]:
    _require_rank(layer, input_shape, index, rank=3, expected_desc="(channels, height, width)")
    in_channels, h_in, w_in = input_shape
    h_out = _spatial_output_size(
        h_in, layer.kernel_size, layer.stride, layer.padding, index=index, layer_name="Conv2d", dim_name="height"
    )
    w_out = _spatial_output_size(
        w_in, layer.kernel_size, layer.stride, layer.padding, index=index, layer_name="Conv2d", dim_name="width"
    )
    return (layer.out_channels, h_out, w_out), {"in_channels": in_channels}


def _batch_norm2d_shape(layer: BatchNorm2dSpec, input_shape: Shape, index: int) -> tuple[Shape, dict[str, int]]:
    _require_rank(layer, input_shape, index, rank=3, expected_desc="(channels, height, width)")
    return input_shape, {"num_features": input_shape[0]}


def _identity_shape(layer: LayerSpec, input_shape: Shape, index: int) -> tuple[Shape, dict[str, int]]:
    return input_shape, {}


def _max_pool2d_shape(layer: MaxPool2dSpec, input_shape: Shape, index: int) -> tuple[Shape, dict[str, int]]:
    _require_rank(layer, input_shape, index, rank=3, expected_desc="(channels, height, width)")
    channels, h_in, w_in = input_shape
    stride = layer.effective_stride
    h_out = _spatial_output_size(
        h_in, layer.kernel_size, stride, layer.padding, index=index, layer_name="MaxPool2d", dim_name="height"
    )
    w_out = _spatial_output_size(
        w_in, layer.kernel_size, stride, layer.padding, index=index, layer_name="MaxPool2d", dim_name="width"
    )
    return (channels, h_out, w_out), {}


def _adaptive_avg_pool2d_shape(
    layer: AdaptiveAvgPool2dSpec, input_shape: Shape, index: int
) -> tuple[Shape, dict[str, int]]:
    _require_rank(layer, input_shape, index, rank=3, expected_desc="(channels, height, width)")
    channels = input_shape[0]
    return (channels, layer.output_size, layer.output_size), {}


def _flatten_shape(layer: FlattenSpec, input_shape: Shape, index: int) -> tuple[Shape, dict[str, int]]:
    if len(input_shape) == 0:
        raise ModelValidationError(f"Layer {index} (Flatten): input shape must have at least one dimension")
    total = 1
    for dim in input_shape:
        total *= dim
    return (total,), {}


def _linear_shape(layer: LinearSpec, input_shape: Shape, index: int) -> tuple[Shape, dict[str, int]]:
    _require_rank(layer, input_shape, index, rank=1, expected_desc="(in_features,)")
    return (layer.out_features,), {"in_features": input_shape[0]}


_ShapeHandler = Callable[[LayerSpec, Shape, int], "tuple[Shape, dict[str, int]]"]

_SHAPE_HANDLERS: dict[type, _ShapeHandler] = {
    Conv2dSpec: _conv2d_shape,
    BatchNorm2dSpec: _batch_norm2d_shape,
    ReLUSpec: _identity_shape,
    MaxPool2dSpec: _max_pool2d_shape,
    AdaptiveAvgPool2dSpec: _adaptive_avg_pool2d_shape,
    FlattenSpec: _flatten_shape,
    LinearSpec: _linear_shape,
    DropoutSpec: _identity_shape,
}


def infer_layer_shape(layer: LayerSpec, input_shape: Shape, index: int = 0) -> tuple[Shape, dict[str, int]]:
    """Compute one layer's output shape (and any shape-derived parameters)."""
    handler = _SHAPE_HANDLERS.get(type(layer))
    if handler is None:
        raise ModelValidationError(
            f"Layer {index} ({_layer_name(layer)}): no shape inference rule is registered "
            f"for layer type {type(layer).__name__!r}"
        )
    return handler(layer, input_shape, index)


def infer_model_shapes(model_spec: ModelSpec) -> list[LayerShapeInfo]:
    """Walk a ModelSpec's layers in order, computing each layer's shape.

    Raises ModelValidationError on the first invalid connection or
    non-positive resulting dimension.
    """
    shape: Shape = tuple(model_spec.input_shape)
    trace: list[LayerShapeInfo] = []
    for index, layer in enumerate(model_spec.layers):
        output_shape, inferred = infer_layer_shape(layer, shape, index)
        trace.append(
            LayerShapeInfo(index=index, layer=layer, input_shape=shape, output_shape=output_shape, inferred=inferred)
        )
        shape = output_shape
    return trace


def format_shape_trace(trace: list[LayerShapeInfo]) -> str:
    """Render a shape trace as human-readable lines, e.g. for a future UI:

        Conv2d
        [3, 224, 224] -> [32, 224, 224]
    """
    lines = []
    for info in trace:
        lines.append(f"{info.layer_name}\n{list(info.input_shape)} -> {list(info.output_shape)}")
    return "\n".join(lines)
