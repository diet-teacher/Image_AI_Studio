"""JSON round-trip for ModelSpec.

The JSON layer format is a flat dict with a discriminator ``"type"``
field (e.g. ``"conv2d"``) plus that layer's own fields, matching the
dataclass fields in specs.py exactly:

    {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}

Shape-derived fields (Conv2d.in_channels, BatchNorm2d.num_features,
Linear.in_features) are never part of the JSON -- they are recomputed by
shape_inference whenever the spec is built or validated.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

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

_LAYER_REGISTRY: dict[str, type[LayerSpec]] = {
    "conv2d": Conv2dSpec,
    "batch_norm2d": BatchNorm2dSpec,
    "relu": ReLUSpec,
    "max_pool2d": MaxPool2dSpec,
    "adaptive_avg_pool2d": AdaptiveAvgPool2dSpec,
    "flatten": FlattenSpec,
    "linear": LinearSpec,
    "dropout": DropoutSpec,
}
_TYPE_NAMES: dict[type[LayerSpec], str] = {cls: name for name, cls in _LAYER_REGISTRY.items()}


def _layer_to_dict(layer: LayerSpec) -> dict:
    type_name = _TYPE_NAMES.get(type(layer))
    if type_name is None:
        raise ModelValidationError(
            f"No JSON type name is registered for layer class {type(layer).__name__!r}"
        )
    return {"type": type_name, **asdict(layer)}


def _layer_from_dict(data: object, index: int) -> LayerSpec:
    if not isinstance(data, dict):
        raise ModelValidationError(f"Layer {index}: expected a JSON object, got {type(data).__name__}")
    if "type" not in data:
        raise ModelValidationError(f"Layer {index}: missing required 'type' field")

    type_name = data["type"]
    layer_cls = _LAYER_REGISTRY.get(type_name)
    if layer_cls is None:
        raise ModelValidationError(
            f"Layer {index}: unknown layer type {type_name!r}. "
            f"Supported types: {sorted(_LAYER_REGISTRY)}"
        )

    kwargs = {key: value for key, value in data.items() if key != "type"}
    try:
        return layer_cls(**kwargs)
    except TypeError as exc:
        raise ModelValidationError(f"Layer {index} ({type_name}): {exc}") from exc


def model_spec_to_dict(model_spec: ModelSpec) -> dict:
    """Convert a ModelSpec to a plain JSON-serializable dict."""
    return {
        "name": model_spec.name,
        "input_shape": list(model_spec.input_shape),
        "layers": [_layer_to_dict(layer) for layer in model_spec.layers],
    }


def model_spec_from_dict(data: dict) -> ModelSpec:
    """Build a ModelSpec from a plain dict (e.g. loaded from JSON)."""
    if not isinstance(data, dict):
        raise ModelValidationError(f"Model spec JSON must be an object, got {type(data).__name__}")

    missing = [key for key in ("name", "input_shape", "layers") if key not in data]
    if missing:
        raise ModelValidationError(f"Model spec JSON is missing required field(s): {missing}")

    if not isinstance(data["layers"], list):
        raise ModelValidationError("Model spec field 'layers' must be a list")

    layers = [_layer_from_dict(item, index) for index, item in enumerate(data["layers"])]
    return ModelSpec(name=data["name"], input_shape=tuple(data["input_shape"]), layers=layers)


def save_model_spec(model_spec: ModelSpec, path: str | Path) -> None:
    """Serialize ``model_spec`` to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_spec_to_dict(model_spec), indent=2), encoding="utf-8")


def load_model_spec(path: str | Path) -> ModelSpec:
    """Load a ModelSpec previously written by save_model_spec (or hand-authored JSON)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelValidationError(f"{path}: not valid JSON ({exc})") from exc
    return model_spec_from_dict(data)
