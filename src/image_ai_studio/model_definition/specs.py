"""모델 정의용 dataclass. UI/PyTorch와 독립적.

- Phase 1: ModelSpec + LayerSpec 순차 리스트만 지원 (Sequential, DAG 아님)
- LayerSpec: 필드/메서드 없는 마커 베이스 클래스 (향후 ResidualBlockSpec 등 확장 대비)
- 파라미터 검증: 각 dataclass __post_init__ (JSON 역직렬화 시에도 동일 적용)
- shape 연결 검증(예: Conv2d에 1D 입력)은 shape_inference.py 담당
"""
from __future__ import annotations

from dataclasses import dataclass, field

from image_ai_studio.model_definition.errors import ModelValidationError


class LayerSpec:
    """모든 레이어 spec의 마커 베이스 클래스. 필드/메서드 없음."""


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelValidationError(f"'{name}' must be a positive integer, got {value!r}")


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelValidationError(f"'{name}' must be a non-negative integer, got {value!r}")


def _require_positive_float(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise ModelValidationError(f"'{name}' must be a positive number, got {value!r}")


def _require_unit_interval(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 <= float(value) <= 1.0):
        raise ModelValidationError(f"'{name}' must be between 0.0 and 1.0, got {value!r}")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ModelValidationError(f"'{name}' must be a bool (true/false), got {value!r}")


@dataclass
class Conv2dSpec(LayerSpec):
    """2D 컨볼루션. in_channels는 이전 레이어 출력에서 자동 계산."""

    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("out_channels", self.out_channels)
        _require_positive_int("kernel_size", self.kernel_size)
        _require_positive_int("stride", self.stride)
        _require_non_negative_int("padding", self.padding)


@dataclass
class BatchNorm2dSpec(LayerSpec):
    """2D BatchNorm. num_features는 이전 레이어 출력에서 자동 계산."""

    eps: float = 1e-5
    momentum: float = 0.1

    def __post_init__(self) -> None:
        _require_positive_float("eps", self.eps)
        if not isinstance(self.momentum, (int, float)) or isinstance(self.momentum, bool) or not (
            0.0 < float(self.momentum) <= 1.0
        ):
            raise ModelValidationError(f"'momentum' must be in (0.0, 1.0], got {self.momentum!r}")


@dataclass
class ReLUSpec(LayerSpec):
    inplace: bool = False

    def __post_init__(self) -> None:
        _require_bool("inplace", self.inplace)


@dataclass
class MaxPool2dSpec(LayerSpec):
    """2D 맥스풀링. stride 생략 시 kernel_size로 대체 (torch.nn.MaxPool2d 기본 동작과 동일).

    padding은 kernel_size // 2 이하만 허용 (torch.nn.MaxPool2d 제약과 동일,
    생성 시점에 미리 검증).
    """

    kernel_size: int
    stride: int | None = None
    padding: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("kernel_size", self.kernel_size)
        if self.stride is not None:
            _require_positive_int("stride", self.stride)
        _require_non_negative_int("padding", self.padding)
        if self.padding > self.kernel_size // 2:
            raise ModelValidationError(
                f"'padding' must be at most kernel_size // 2 ({self.kernel_size // 2}), "
                f"got padding={self.padding} for kernel_size={self.kernel_size} "
                "(matches torch.nn.MaxPool2d's own constraint)"
            )

    @property
    def effective_stride(self) -> int:
        return self.stride if self.stride is not None else self.kernel_size


@dataclass
class AdaptiveAvgPool2dSpec(LayerSpec):
    """Adaptive average pooling. 정사각형 (output_size, output_size)만 지원."""

    output_size: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("output_size", self.output_size)


@dataclass
class FlattenSpec(LayerSpec):
    pass


@dataclass
class LinearSpec(LayerSpec):
    """FC 레이어. in_features는 이전 레이어 shape에서 자동 계산."""

    out_features: int
    bias: bool = True

    def __post_init__(self) -> None:
        _require_positive_int("out_features", self.out_features)
        _require_bool("bias", self.bias)


@dataclass
class DropoutSpec(LayerSpec):
    p: float = 0.5

    def __post_init__(self) -> None:
        _require_unit_interval("p", self.p)


@dataclass
class IdentitySpec(LayerSpec):
    """입력을 그대로 통과시키는 passthrough 레이어. BranchSpec의 skip path를
    명시적으로 표현하기 위한 용도 (빈 branch는 허용하지 않고 IdentitySpec()을
    명시해야 함)."""


@dataclass
class ResidualBlockSpec(LayerSpec):
    """Conv-BN-ReLU-Conv-BN + shortcut (models.residual_block.ResidualBlock 재사용).

    in_channels는 이전 레이어 출력에서 자동 계산. 내부 kernel_size/padding은
    ResidualBlock과 동일하게 고정 (설계 근거: docs/phase2_residual_block_design.md).
    """

    out_channels: int
    stride: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("out_channels", self.out_channels)
        _require_positive_int("stride", self.stride)


@dataclass
class BranchSpec(LayerSpec):
    """입력 하나를 N개 병렬 branch로 나눠 처리한 뒤 merge로 다시 하나로 합치는 composite layer.

    각 branch는 그 자체로 완결된 list[LayerSpec] 체인이다 (기존 shape_inference/builder를
    branch 단위로 재귀 재사용). 중첩 BranchSpec은 이번 Phase에서 금지한다 (설계 근거:
    docs/phase3_branch_design.md).
    """

    branches: list[list[LayerSpec]]
    merge: str = "add"

    def __post_init__(self) -> None:
        if self.merge not in ("add", "concat"):
            raise ModelValidationError(f"'merge' must be one of ('add', 'concat'), got {self.merge!r}")

        if not isinstance(self.branches, (list, tuple)):
            raise ModelValidationError(
                f"BranchSpec.branches must be a list or tuple, got {type(self.branches).__name__}"
            )
        self.branches = [self._validate_branch(index, branch) for index, branch in enumerate(self.branches)]
        if len(self.branches) < 2:
            raise ModelValidationError(
                f"BranchSpec.branches must contain at least 2 branches, got {len(self.branches)}"
            )

    def _validate_branch(self, branch_index: int, branch: object) -> list[LayerSpec]:
        if not isinstance(branch, (list, tuple)):
            raise ModelValidationError(
                f"BranchSpec.branches[{branch_index}] must be a list of LayerSpec, "
                f"got {type(branch).__name__}"
            )
        branch = list(branch)
        if not branch:
            raise ModelValidationError(
                f"BranchSpec.branches[{branch_index}] must not be empty "
                "(use a single IdentitySpec() for a passthrough branch)"
            )
        for layer_index, layer in enumerate(branch):
            if not isinstance(layer, LayerSpec):
                raise ModelValidationError(
                    f"BranchSpec.branches[{branch_index}][{layer_index}] must be a LayerSpec "
                    f"instance, got {type(layer).__name__}"
                )
            if isinstance(layer, BranchSpec):
                raise ModelValidationError(
                    f"BranchSpec.branches[{branch_index}][{layer_index}]: nested BranchSpec "
                    "is not supported in Phase 3"
                )
        return branch


@dataclass
class ModelSpec:
    """모델 정의 = 이름 + input_shape + 레이어 리스트.

    input_shape: batch 제외, (channels, height, width) 고정.
    Phase 1: 이미지 classification 입력만 지원.
    layers: 최소 1개 이상 필요 (빈 모델 금지).
    """

    name: str
    input_shape: tuple[int, int, int]
    layers: list[LayerSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelValidationError("ModelSpec.name must be a non-empty string")

        if not isinstance(self.input_shape, (list, tuple)):
            raise ModelValidationError(
                f"ModelSpec.input_shape must be a list or tuple, got {type(self.input_shape).__name__}"
            )
        self.input_shape = tuple(self.input_shape)  # type: ignore[assignment]
        if len(self.input_shape) != 3 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.input_shape
        ):
            raise ModelValidationError(
                "ModelSpec.input_shape must be 3 positive integers (channels, height, width), "
                f"got {self.input_shape!r}"
            )

        if not isinstance(self.layers, (list, tuple)):
            raise ModelValidationError(
                f"ModelSpec.layers must be a list or tuple of LayerSpec, got {type(self.layers).__name__}"
            )
        self.layers = list(self.layers)
        if not self.layers:
            raise ModelValidationError("ModelSpec.layers must contain at least one layer")
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, LayerSpec):
                raise ModelValidationError(
                    f"ModelSpec.layers[{index}] must be a LayerSpec instance, got {type(layer).__name__}"
                )
