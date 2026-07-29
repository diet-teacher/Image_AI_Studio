"""ModelSpec <-> JSON.

레이어 JSON 포맷: "type" 필드 + 해당 dataclass 필드 (specs.py와 1:1 대응).

    {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}

shape 자동 계산 값(in_channels 등)은 JSON에 미포함 -- 빌드/검증 시 shape_inference가 재계산.
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
    ResidualBlockSpec,
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
    "residual_block": ResidualBlockSpec,
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
    if not isinstance(type_name, str):
        raise ModelValidationError(
            f"Layer {index}: 'type' must be a string, got {type(type_name).__name__}"
        )
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
    """ModelSpec -> JSON 직렬화 가능한 dict."""
    return {
        "name": model_spec.name,
        "input_shape": list(model_spec.input_shape),
        "layers": [_layer_to_dict(layer) for layer in model_spec.layers],
    }


def model_spec_from_dict(data: dict) -> ModelSpec:
    """dict -> ModelSpec 생성 (JSON 로드 결과 등)."""
    if not isinstance(data, dict):
        raise ModelValidationError(f"Model spec JSON must be an object, got {type(data).__name__}")

    missing = [key for key in ("name", "input_shape", "layers") if key not in data]
    if missing:
        raise ModelValidationError(f"Model spec JSON is missing required field(s): {missing}")

    if not isinstance(data["layers"], list):
        raise ModelValidationError("Model spec field 'layers' must be a list")

    layers = [_layer_from_dict(item, index) for index, item in enumerate(data["layers"])]
    # input_shape은 tuple() 미변환 상태로 전달 -- ModelSpec.__post_init__에서
    # 타입 검증 (raw TypeError 대신 ModelValidationError 발생시키기 위함)
    return ModelSpec(name=data["name"], input_shape=data["input_shape"], layers=layers)


def save_model_spec(model_spec: ModelSpec, path: str | Path) -> None:
    """model_spec을 JSON 파일로 저장. 상위 디렉터리 자동 생성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_spec_to_dict(model_spec), indent=2), encoding="utf-8")


def load_model_spec(path: str | Path) -> ModelSpec:
    """JSON 파일 로드 (save_model_spec 저장분 또는 수동 작성분)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelValidationError(f"{path}: not valid JSON ({exc})") from exc
    return model_spec_from_dict(data)
