"""ModelSpec -> torch.nn.Module 빌드.

- Phase 1: Sequential 구조만 지원, 결과는 항상 nn.Sequential
- in_channels/num_features/in_features 등은 shape_inference 결과 사용
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from torch import nn

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.shape_inference import LayerShapeInfo, infer_layer_shape
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
from image_ai_studio.model_definition.validation import validate_model_spec
from image_ai_studio.models.branch_block import BranchBlock
from image_ai_studio.models.residual_block import ResidualBlock


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


def _build_identity(layer: IdentitySpec, inferred: dict[str, int]) -> nn.Module:
    return nn.Identity()


def _build_residual_block(layer: ResidualBlockSpec, inferred: dict[str, int]) -> nn.Module:
    return ResidualBlock(
        in_channels=inferred["in_channels"],
        out_channels=layer.out_channels,
        stride=layer.stride,
    )


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
    ResidualBlockSpec: _build_residual_block,
    IdentitySpec: _build_identity,
}


def _build_plain_layer(layer: LayerSpec, inferred: dict[str, int], *, index: int | None = None) -> nn.Module:
    builder_fn = _BUILDERS.get(type(layer))
    if builder_fn is None:
        prefix = f"Layer {index}: " if index is not None else ""
        raise ModelValidationError(
            f"{prefix}no PyTorch builder is registered for layer type {type(layer).__name__!r}"
        )
    return builder_fn(layer, inferred)


def _build_branch(layer: BranchSpec, input_shape) -> nn.Module:
    """branch마다 기존 infer_layer_shape/_build_plain_layer를 그대로 재사용해
    nn.Sequential을 구성한 뒤 BranchBlock으로 묶는다. BranchSpec은 (다른
    레이어와 달리) 자기 input_shape 문맥이 더 필요해서 _build_layer에서
    별도로 처리한다 -- 나머지 레이어 타입의 딕셔너리 조회 방식은 그대로다."""
    branch_modules = []
    for branch in layer.branches:
        shape = input_shape
        modules = []
        for index, sub_layer in enumerate(branch):
            output_shape, inferred = infer_layer_shape(sub_layer, shape, index)
            modules.append(_build_plain_layer(sub_layer, inferred))
            shape = output_shape
        branch_modules.append(nn.Sequential(*modules))
    return BranchBlock(branch_modules, merge=layer.merge)


def _build_layer(info: LayerShapeInfo) -> nn.Module:
    if isinstance(info.layer, BranchSpec):
        return _build_branch(info.layer, info.input_shape)
    return _build_plain_layer(info.layer, info.inferred, index=info.index)


def build_model(model_spec: ModelSpec) -> nn.Sequential:
    """model_spec 검증 후 nn.Sequential로 조립. 검증 실패 시 예외, 부분 생성 모델 반환 없음."""
    shape_trace = validate_model_spec(model_spec)
    modules = [_build_layer(info) for info in shape_trace]
    return nn.Sequential(*modules)
