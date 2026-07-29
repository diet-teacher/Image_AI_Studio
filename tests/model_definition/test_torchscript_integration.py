"""ModelSpec -> build_model -> torch.jit.trace -> model.pt -> reload -> parity 검증.

Phase 0 TorchScriptExporter 재사용 (export/compare 로직 중복 구현 없음,
TinyCNN/TinyResidualCNN과 동일 경로).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import (
    AdaptiveAvgPool2dSpec,
    BatchNorm2dSpec,
    Conv2dSpec,
    FlattenSpec,
    LinearSpec,
    MaxPool2dSpec,
    ModelSpec,
    ReLUSpec,
    ResidualBlockSpec,
)
from image_ai_studio.parity.compare_outputs import CPU_FP32_ATOL, CPU_FP32_RTOL, compare_outputs


def _small_model_spec() -> ModelSpec:
    # 작은 입력 크기로 테스트 속도 확보 (shape_inference/builder는 크기 무관 동작)
    return ModelSpec(
        name="phase1_torchscript_smoke",
        input_shape=(3, 16, 16),
        layers=[
            Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1),
            BatchNorm2dSpec(),
            ReLUSpec(),
            MaxPool2dSpec(kernel_size=2, stride=2),
            FlattenSpec(),
            LinearSpec(out_features=4),
        ],
    )


def test_model_spec_builds_exports_and_round_trips_through_torchscript(tmp_path: Path) -> None:
    spec = _small_model_spec()
    model = build_model(spec).eval()

    example_input = torch.randn(1, *spec.input_shape)
    output_path = tmp_path / "model.pt"
    metadata_path = tmp_path / "metadata.json"

    exporter = TorchScriptExporter()
    exporter.export(
        model,
        example_input,
        output_path,
        metadata_path,
        model_name=spec.name,
        # Phase 1 모델은 저장된 state_dict 없음 -- build_metadata()는 경로 부재 시
        # state_dict_sha256=None 처리
        state_dict_path=tmp_path / "no_state_dict.pt",
    )

    assert output_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "PASS", metadata.get("error_log")

    reloaded = torch.jit.load(str(output_path))
    reloaded.eval()

    with torch.inference_mode():
        original_output = model(example_input)
        reloaded_output = reloaded(example_input)

    parity = compare_outputs(original_output, reloaded_output, rtol=CPU_FP32_RTOL, atol=CPU_FP32_ATOL)
    assert parity.allclose, parity.to_dict()
    assert tuple(reloaded_output.shape) == (1, 4)


def _residual_model_spec() -> ModelSpec:
    return ModelSpec(
        name="phase2_torchscript_residual_smoke",
        input_shape=(3, 16, 16),
        layers=[
            Conv2dSpec(out_channels=8, kernel_size=3, stride=1, padding=1),
            ResidualBlockSpec(out_channels=8),
            ResidualBlockSpec(out_channels=16, stride=2),
            AdaptiveAvgPool2dSpec(output_size=1),
            FlattenSpec(),
            LinearSpec(out_features=4),
        ],
    )


def test_residual_block_model_builds_exports_and_round_trips_through_torchscript(tmp_path: Path) -> None:
    """위 테스트와 동일한 패턴에 ResidualBlockSpec만 추가 -- 기존
    TorchScriptExporter 그대로 trace/재로드가 정상 동작하는지 확인."""
    spec = _residual_model_spec()
    model = build_model(spec).eval()

    example_input = torch.randn(1, *spec.input_shape)
    output_path = tmp_path / "model.pt"
    metadata_path = tmp_path / "metadata.json"

    exporter = TorchScriptExporter()
    exporter.export(
        model,
        example_input,
        output_path,
        metadata_path,
        model_name=spec.name,
        state_dict_path=tmp_path / "no_state_dict.pt",
    )

    assert output_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "PASS", metadata.get("error_log")

    reloaded = torch.jit.load(str(output_path))
    reloaded.eval()

    with torch.inference_mode():
        original_output = model(example_input)
        reloaded_output = reloaded(example_input)

    parity = compare_outputs(original_output, reloaded_output, rtol=CPU_FP32_RTOL, atol=CPU_FP32_ATOL)
    assert parity.allclose, parity.to_dict()
    assert tuple(reloaded_output.shape) == (1, 4)
