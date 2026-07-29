"""ModelSpec 레이어별 텐서 shape 계산 (batch 차원 제외).

- 레이어 간 shape 연결 검증 (예: Conv2d에 1D 입력, 풀링 결과 0 이하)
- 이전 레이어 shape 의존 파라미터 계산 (Conv2d.in_channels,
  BatchNorm2d.num_features, Linear.in_features)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.specs import (
    AdaptiveAvgPool2dSpec,
    BatchNorm2dSpec,
    BranchSpec,
    Conv2dSpec,
    DropoutSpec,
    FlattenSpec,
    IdentitySpec,
    LayerSpec,
    LinearSpec,
    MaxPool2dSpec,
    ModelSpec,
    ReLUSpec,
    ResidualBlockSpec,
)

Shape = tuple[int, ...]


@dataclass
class LayerShapeInfo:
    """shape trace 한 줄. inferred: shape에서 자동 계산한 파라미터 (예: {"in_features": 401408})."""

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


def _residual_block_shape(
    layer: ResidualBlockSpec, input_shape: Shape, index: int
) -> tuple[Shape, dict[str, int]]:
    """ResidualBlock 전체 shape을 Conv2d와 동일한 공식 한 번으로 계산
    (shape 공식 근거는 docs/phase2_residual_block_design.md 4번 섹션 참고)."""
    _require_rank(layer, input_shape, index, rank=3, expected_desc="(channels, height, width)")
    in_channels, h_in, w_in = input_shape
    h_out = _spatial_output_size(
        h_in, kernel_size=3, stride=layer.stride, padding=1,
        index=index, layer_name="ResidualBlock", dim_name="height",
    )
    w_out = _spatial_output_size(
        w_in, kernel_size=3, stride=layer.stride, padding=1,
        index=index, layer_name="ResidualBlock", dim_name="width",
    )
    return (layer.out_channels, h_out, w_out), {"in_channels": in_channels}


def _branch_shape(layer: BranchSpec, input_shape: Shape, index: int) -> tuple[Shape, dict[str, int]]:
    """각 branch는 동일한 input_shape에서 시작해 기존 infer_layer_shape을 그대로
    재귀 재사용한다 (새 shape 엔진 없음). merge="add"는 모든 branch의 최종
    output_shape이 완전히 같아야 하고, merge="concat"은 channel(axis 0)만
    합산하고 나머지(H,W)는 전부 같아야 한다 (Phase 3는 channel concat만 지원,
    concat_dim은 노출하지 않음)."""
    branch_output_shapes = []
    for branch in layer.branches:
        shape = input_shape
        for sub_layer in branch:
            shape, _ = infer_layer_shape(sub_layer, shape, index)
        branch_output_shapes.append(shape)

    if layer.merge == "add":
        if len(set(branch_output_shapes)) != 1:
            raise ModelValidationError(
                f"Layer {index} (Branch): merge=\"add\" requires all branches to produce the "
                f"same output shape, got {branch_output_shapes}"
            )
        return branch_output_shapes[0], {}

    # merge == "concat"
    for shape in branch_output_shapes:
        if len(shape) != 3:
            raise ModelValidationError(
                f"Layer {index} (Branch): merge=\"concat\" requires each branch to produce a "
                f"3D (channels, height, width) output, got {shape}"
            )
    spatial_shapes = {shape[1:] for shape in branch_output_shapes}
    if len(spatial_shapes) != 1:
        raise ModelValidationError(
            f"Layer {index} (Branch): merge=\"concat\" requires all branches to share the same "
            f"(height, width), got {branch_output_shapes}"
        )
    height, width = next(iter(spatial_shapes))
    total_channels = sum(shape[0] for shape in branch_output_shapes)
    return (total_channels, height, width), {}


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
    ResidualBlockSpec: _residual_block_shape,
    BranchSpec: _branch_shape,
    IdentitySpec: _identity_shape,
}


def infer_layer_shape(layer: LayerSpec, input_shape: Shape, index: int = 0) -> tuple[Shape, dict[str, int]]:
    """레이어 output shape 계산 (자동 추론 파라미터 포함)."""
    handler = _SHAPE_HANDLERS.get(type(layer))
    if handler is None:
        raise ModelValidationError(
            f"Layer {index} ({_layer_name(layer)}): no shape inference rule is registered "
            f"for layer type {type(layer).__name__!r}"
        )
    return handler(layer, input_shape, index)


def infer_model_shapes(model_spec: ModelSpec) -> list[LayerShapeInfo]:
    """ModelSpec 레이어 순회하며 shape 계산. 연결 오류/0 이하 결과 시 ModelValidationError."""
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
    """shape trace를 UI 표시용 텍스트로 변환.

        Conv2d
        [3, 224, 224] -> [32, 224, 224]
    """
    lines = []
    for info in trace:
        lines.append(f"{info.layer_name}\n{list(info.input_shape)} -> {list(info.output_shape)}")
    return "\n".join(lines)
