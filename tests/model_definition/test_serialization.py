"""ModelSpec JSON round-trip 테스트 (Python -> JSON -> Python 동일성)."""
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
    BranchSpec,
    Conv2dSpec,
    FlattenSpec,
    IdentitySpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
    ResidualBlockSpec,
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
    """Phase 1 스펙에 있는 JSON 예시 그대로 로드되는지 확인."""
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
    """'type' 비문자열(예: list) 시 dict.get() 조회 전 차단 (unhashable type 방지)."""
    with pytest.raises(ModelValidationError, match="'type' must be a string"):
        model_spec_from_dict(
            {"name": "m", "input_shape": [3, 8, 8], "layers": [{"type": ["conv2d"]}]}
        )


def test_non_sequence_input_shape_in_json_raises_model_validation_error() -> None:
    """input_shape 비iterable 시 tuple() 호출 전 차단 (raw TypeError 방지)."""
    with pytest.raises(ModelValidationError, match="input_shape"):
        model_spec_from_dict({"name": "m", "input_shape": 123, "layers": []})


def test_missing_required_layer_param_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="conv2d"):
        model_spec_from_dict(
            {
                "name": "m",
                "input_shape": [3, 8, 8],
                # conv2d는 out_channels, kernel_size가 필수임
                "layers": [{"type": "conv2d"}],
            }
        )


def test_missing_top_level_field_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="missing required field"):
        model_spec_from_dict({"name": "m", "layers": []})


def test_empty_layers_list_in_json_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="at least one layer"):
        model_spec_from_dict({"name": "m", "input_shape": [3, 8, 8], "layers": []})


def test_load_model_spec_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ModelValidationError, match="not valid JSON"):
        load_model_spec(path)


# -- ResidualBlockSpec --------------------------------------------------------


def test_residual_block_model_round_trips_through_json() -> None:
    original = ModelSpec(
        name="residual_example",
        input_shape=(3, 16, 16),
        layers=[
            Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1),
            ResidualBlockSpec(out_channels=16, stride=2),
        ],
    )
    restored = model_spec_from_dict(model_spec_to_dict(original))
    assert restored == original


def test_residual_block_json_type_loads_directly() -> None:
    spec = model_spec_from_dict(
        {
            "name": "m",
            "input_shape": [3, 8, 8],
            "layers": [{"type": "residual_block", "out_channels": 16, "stride": 2}],
        }
    )
    assert isinstance(spec.layers[0], ResidualBlockSpec)
    assert spec.layers[0].out_channels == 16
    assert spec.layers[0].stride == 2


def test_residual_block_json_default_stride_restores_to_one() -> None:
    spec = model_spec_from_dict(
        {
            "name": "m",
            "input_shape": [3, 8, 8],
            "layers": [{"type": "residual_block", "out_channels": 16}],
        }
    )
    assert spec.layers[0].stride == 1


# -- IdentitySpec --------------------------------------------------------------


def test_identity_json_type_loads_directly() -> None:
    spec = model_spec_from_dict(
        {"name": "m", "input_shape": [3, 8, 8], "layers": [{"type": "identity"}]}
    )
    assert isinstance(spec.layers[0], IdentitySpec)


# -- BranchSpec ----------------------------------------------------------------


def _branch_example_model_spec() -> ModelSpec:
    return ModelSpec(
        name="branch_example",
        input_shape=(8, 8, 8),
        layers=[
            BranchSpec(
                branches=[
                    [Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1), BatchNorm2dSpec()],
                    [IdentitySpec()],
                ],
                merge="add",
            ),
            ReLUSpec(),
        ],
    )


def test_branch_model_round_trips_through_json() -> None:
    original = _branch_example_model_spec()
    restored = model_spec_from_dict(model_spec_to_dict(original))
    assert restored == original


def test_branch_model_round_trip_preserves_shape_inference_result() -> None:
    original = _branch_example_model_spec()
    restored = model_spec_from_dict(model_spec_to_dict(original))
    original_trace = [(i.input_shape, i.output_shape) for i in infer_model_shapes(original)]
    restored_trace = [(i.input_shape, i.output_shape) for i in infer_model_shapes(restored)]
    assert original_trace == restored_trace


def test_branch_json_type_loads_directly_with_nested_layers() -> None:
    spec = model_spec_from_dict(
        {
            "name": "m",
            "input_shape": [4, 8, 8],
            "layers": [
                {
                    "type": "branch",
                    "merge": "concat",
                    "branches": [
                        [{"type": "conv2d", "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}],
                        [{"type": "identity"}],
                    ],
                }
            ],
        }
    )
    branch = spec.layers[0]
    assert isinstance(branch, BranchSpec)
    assert branch.merge == "concat"
    assert isinstance(branch.branches[0][0], Conv2dSpec)
    assert isinstance(branch.branches[1][0], IdentitySpec)


def test_branch_json_default_merge_restores_to_add() -> None:
    spec = model_spec_from_dict(
        {
            "name": "m",
            "input_shape": [4, 8, 8],
            "layers": [
                {
                    "type": "branch",
                    "branches": [[{"type": "identity"}], [{"type": "identity"}]],
                }
            ],
        }
    )
    assert spec.layers[0].merge == "add"


def test_branch_json_nested_branch_raises_model_validation_error() -> None:
    with pytest.raises(ModelValidationError, match="nested BranchSpec"):
        model_spec_from_dict(
            {
                "name": "m",
                "input_shape": [4, 8, 8],
                "layers": [
                    {
                        "type": "branch",
                        "branches": [
                            [{"type": "identity"}],
                            [{"type": "branch", "branches": [[{"type": "identity"}], [{"type": "identity"}]]}],
                        ],
                    }
                ],
            }
        )
