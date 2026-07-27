"""JSON round-trip for ModelSpec: Python -> JSON -> Python must be semantically identical."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.serialization import (
    load_model_spec,
    model_spec_from_dict,
    model_spec_to_dict,
    save_model_spec,
)
from image_ai_studio.model_definition.shape_inference import infer_model_shapes
from image_ai_studio.model_definition.specs import (
    BatchNorm2dSpec,
    Conv2dSpec,
    FlattenSpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
)


def _example_model_spec() -> ModelSpec:
    return ModelSpec(
        name="example_model",
        input_shape=(3, 224, 224),
        layers=[
            Conv2dSpec(out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2dSpec(),
            ReLUSpec(),
            FlattenSpec(),
            LinearSpec(out_features=10),
        ],
    )


def test_model_spec_to_dict_matches_expected_json_shape() -> None:
    spec = _example_model_spec()
    assert model_spec_to_dict(spec) == {
        "name": "example_model",
        "input_shape": [3, 224, 224],
        "layers": [
            {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "batch_norm2d", "eps": 1e-5, "momentum": 0.1},
            {"type": "relu", "inplace": False},
            {"type": "flatten"},
            {"type": "linear", "out_features": 10, "bias": True},
        ],
    }


def test_round_trip_is_semantically_equal() -> None:
    original = _example_model_spec()
    restored = model_spec_from_dict(model_spec_to_dict(original))
    assert restored == original


def test_round_trip_preserves_shape_inference_result() -> None:
    original = _example_model_spec()
    restored = model_spec_from_dict(model_spec_to_dict(original))
    original_trace = [(i.input_shape, i.output_shape) for i in infer_model_shapes(original)]
    restored_trace = [(i.input_shape, i.output_shape) for i in infer_model_shapes(restored)]
    assert original_trace == restored_trace


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = _example_model_spec()
    out_path = tmp_path / "example_model.json"
    save_model_spec(original, out_path)

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["name"] == "example_model"

    restored = load_model_spec(out_path)
    assert restored == original


def test_load_model_spec_from_task_example_json(tmp_path: Path) -> None:
    """The exact JSON example from the Phase 1 spec must load successfully."""
    example_json = {
        "name": "example_model",
        "input_shape": [3, 224, 224],
        "layers": [
            {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "batch_norm2d"},
            {"type": "relu"},
            {"type": "flatten"},
            {"type": "linear", "out_features": 10},
        ],
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(example_json), encoding="utf-8")

    spec = load_model_spec(path)
    assert spec.name == "example_model"
    assert spec.input_shape == (3, 224, 224)
    assert len(spec.layers) == 5
    assert isinstance(spec.layers[0], Conv2dSpec)
    assert isinstance(spec.layers[-1], LinearSpec)


def test_unknown_layer_type_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="unknown layer type"):
        model_spec_from_dict(
            {"name": "m", "input_shape": [3, 8, 8], "layers": [{"type": "not_a_real_layer"}]}
        )


def test_missing_type_field_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="'type'"):
        model_spec_from_dict({"name": "m", "input_shape": [3, 8, 8], "layers": [{"out_channels": 4}]})


def test_non_string_type_field_raises_model_validation_error() -> None:
    """A non-string 'type' (e.g. a list) must not reach the dict.get() lookup,
    which would raise a raw TypeError (unhashable type) instead of
    ModelValidationError."""
    with pytest.raises(ModelValidationError, match="'type' must be a string"):
        model_spec_from_dict(
            {"name": "m", "input_shape": [3, 8, 8], "layers": [{"type": ["conv2d"]}]}
        )


def test_non_sequence_input_shape_in_json_raises_model_validation_error() -> None:
    """A non-iterable input_shape must not reach a bare tuple() call, which
    would raise a raw TypeError instead of ModelValidationError."""
    with pytest.raises(ModelValidationError, match="input_shape"):
        model_spec_from_dict({"name": "m", "input_shape": 123, "layers": []})


def test_missing_required_layer_param_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="conv2d"):
        model_spec_from_dict(
            {
                "name": "m",
                "input_shape": [3, 8, 8],
                # conv2d requires out_channels and kernel_size
                "layers": [{"type": "conv2d"}],
            }
        )


def test_missing_top_level_field_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="missing required field"):
        model_spec_from_dict({"name": "m", "layers": []})


def test_load_model_spec_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ModelValidationError, match="not valid JSON"):
        load_model_spec(path)
