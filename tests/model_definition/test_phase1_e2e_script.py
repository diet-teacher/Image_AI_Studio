"""scripts/run_phase1_e2e.py의 CLI + JSON 로딩 부분 테스트 (run_torchscript 빌드 불필요).

parse_args()/load_and_validate()만 대상 -- 순수 Python/model_definition
로직이라 torch.nn.Module 생성, TorchScript export, C++ 서브프로세스 없음.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from image_ai_studio.model_definition.errors import ModelValidationError

import run_phase1_e2e as e2e  # noqa: E402 -- sys.path 세팅 후 임포트 필요


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
    """JSON이 모델별 유일한 원본 -- 다른 --model-json도 동작해야 함."""
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
