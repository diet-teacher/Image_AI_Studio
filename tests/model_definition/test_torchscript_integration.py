"""ModelSpec -> build_model -> torch.jit.trace -> model.pt -> reload -> parity.

Reuses the existing Phase 0 TorchScriptExporter (image_ai_studio.export)
instead of re-implementing export/compare logic, so Phase 1 model
definitions go through the exact same, already-validated export path
that scripts/export_models.py uses for TinyCNN/TinyResidualCNN.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import (
    BatchNorm2dSpec,
    Conv2dSpec,
    FlattenSpec,
    LinearSpec,
    MaxPool2dSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.parity.compare_outputs import CPU_FP32_ATOL, CPU_FP32_RTOL, compare_outputs


def _small_model_spec() -> ModelSpec:
    # Small spatial size keeps this test fast; the shape-inference /
    # builder code path is identical regardless of input size.
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
        # Phase 1 model specs have no on-disk state_dict; build_metadata()
        # tolerates a non-existent path and records state_dict_sha256=None.
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
