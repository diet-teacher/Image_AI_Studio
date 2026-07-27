"""Dataclass specs describing a model, independent of any UI or PyTorch object.

Phase 1 supports a single ``ModelSpec`` holding a flat (Sequential) list
of ``LayerSpec`` objects -- no arbitrary DAG yet. ``LayerSpec`` is kept as
a plain marker base with no required methods, so a future composite
layer (e.g. a ``ResidualBlockSpec`` wrapping several sub-layers) can be
added as just another class without changing this module's shape, and
without every existing layer needing to implement a new interface.

Parameter-level validation (kernel_size > 0, etc.) runs in each
dataclass's ``__post_init__``, so an invalid spec fails immediately at
construction time -- including when built from JSON via
``serialization.py``. Shape-level validation (e.g. Conv2d receiving a
flattened 1D tensor) is a separate concern and lives in
``shape_inference.py``, since it requires knowing the output shape of
the previous layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from image_ai_studio.model_definition.errors import ModelValidationError


class LayerSpec:
    """Marker base class for all layer spec dataclasses."""


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


@dataclass
class Conv2dSpec(LayerSpec):
    """2D convolution. ``in_channels`` is inferred from the previous layer."""

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
    """2D batch normalization. ``num_features`` is inferred from the previous layer."""

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


@dataclass
class MaxPool2dSpec(LayerSpec):
    """2D max pooling. ``stride`` defaults to ``kernel_size`` when omitted, matching torch.nn.MaxPool2d."""

    kernel_size: int
    stride: int | None = None
    padding: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("kernel_size", self.kernel_size)
        if self.stride is not None:
            _require_positive_int("stride", self.stride)
        _require_non_negative_int("padding", self.padding)

    @property
    def effective_stride(self) -> int:
        return self.stride if self.stride is not None else self.kernel_size


@dataclass
class AdaptiveAvgPool2dSpec(LayerSpec):
    """Adaptive average pooling to a square (output_size, output_size) map."""

    output_size: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("output_size", self.output_size)


@dataclass
class FlattenSpec(LayerSpec):
    pass


@dataclass
class LinearSpec(LayerSpec):
    """Fully-connected layer. ``in_features`` is inferred from the previous layer's shape."""

    out_features: int
    bias: bool = True

    def __post_init__(self) -> None:
        _require_positive_int("out_features", self.out_features)


@dataclass
class DropoutSpec(LayerSpec):
    p: float = 0.5

    def __post_init__(self) -> None:
        _require_unit_interval("p", self.p)


@dataclass
class ModelSpec:
    """A full model definition: a name, an input shape, and a flat layer list.

    ``input_shape`` excludes the batch dimension and is always
    ``(channels, height, width)`` in Phase 1 -- image classification
    input only. Other input kinds (e.g. 1D sequences) can be added later
    without changing this field's meaning for existing models.
    """

    name: str
    input_shape: tuple[int, int, int]
    layers: list[LayerSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelValidationError("ModelSpec.name must be a non-empty string")

        self.input_shape = tuple(self.input_shape)  # type: ignore[assignment]
        if len(self.input_shape) != 3 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.input_shape
        ):
            raise ModelValidationError(
                "ModelSpec.input_shape must be 3 positive integers (channels, height, width), "
                f"got {self.input_shape!r}"
            )

        self.layers = list(self.layers)
