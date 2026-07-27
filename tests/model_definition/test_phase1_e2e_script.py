"""scripts/run_phase1_e2e.py's CLI + JSON-loading layer, exercised without
needing a built run_torchscript binary.

These tests import the script itself (not a package under src/) the same
way scripts/run_phase1_e2e.py imports image_ai_studio: by inserting the
containing directory onto sys.path. They only touch parse_args() and
load_and_validate() -- both pure Python/model_definition, no torch.nn.Module
construction, no TorchScript export, no C++ subprocess -- so they run
alongside the rest of the Phase 1 unit tests without a compiled C++
runner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from image_ai_studio.model_definition.errors import ModelValidationError

import run_phase1_e2e as e2e  # noqa: E402 -- import after sys.path setup above


def test_default_model_json_argument_matches_examples_model() -> None:
    args = e2e.parse_args([])
    assert args.model_json == e2e.MODEL_JSON
    assert args.model_json == REPO_ROOT / "examples" / "models" / "phase1_e2e_model.json"


def test_custom_model_json_argument_is_parsed() -> None:
    custom_path = REPO_ROOT / "examples" / "models" / "phase1_e2e_alt_model.json"
    args = e2e.parse_args(["--model-json", str(custom_path)])
    assert args.model_json == custom_path


def test_load_and_validate_default_model_json_is_valid() -> None:
    model_spec, shape_trace = e2e.load_and_validate(e2e.MODEL_JSON)
    assert model_spec.name == "phase1_e2e_model"
    assert model_spec.input_shape == (3, 16, 16)
    assert shape_trace[-1].output_shape == (4,)


def test_load_and_validate_accepts_a_different_model_json() -> None:
    """The JSON is the single source of truth per model -- a different
    --model-json needs no matching hand-written Python ModelSpec to
    validate against, so an arbitrary second example model must also work."""
    alt_path = REPO_ROOT / "examples" / "models" / "phase1_e2e_alt_model.json"
    model_spec, shape_trace = e2e.load_and_validate(alt_path)
    assert model_spec.name == "phase1_e2e_alt_model"
    assert model_spec.input_shape == (1, 8, 8)
    assert shape_trace[-1].output_shape == (2,)


def test_load_and_validate_rejects_invalid_model_json(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad_model.json"
    bad_json.write_text(
        '{"name": "bad", "input_shape": [3, 8, 8], '
        '"layers": [{"type": "linear", "out_features": 10}]}'
    )
    with pytest.raises(ModelValidationError):
        e2e.load_and_validate(bad_json)
