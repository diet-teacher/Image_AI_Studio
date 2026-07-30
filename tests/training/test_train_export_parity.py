"""학습 -> state_dict 저장/재로드 -> TorchScript export -> parity 통합 테스트.

tests/model_definition/test_torchscript_integration.py와 동일한 패턴(Phase 0
TorchScriptExporter/parity 재사용, 신규 구현 없음)을 무작위 초기화 모델이
아니라 실제로 학습된 모델에 적용한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import (
    AdaptiveAvgPool2dSpec,
    BatchNorm2dSpec,
    Conv2dSpec,
    FlattenSpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.parity.compare_outputs import CPU_FP32_ATOL, CPU_FP32_RTOL, compare_outputs
from image_ai_studio.training.checkpoint import load_state_dict, save_state_dict
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.dataset import make_train_val_datasets
from image_ai_studio.training.loop import run_training

NUM_CLASSES = 4


def _spec() -> ModelSpec:
    return ModelSpec(
        name="train_export_smoke",
        input_shape=(3, 8, 8),
        layers=[
            Conv2dSpec(out_channels=4, kernel_size=3, padding=1),
            BatchNorm2dSpec(),
            ReLUSpec(),
            AdaptiveAvgPool2dSpec(output_size=1),
            FlattenSpec(),
            LinearSpec(out_features=NUM_CLASSES),
        ],
    )


def test_trained_model_exports_and_round_trips_through_torchscript(tmp_path: Path) -> None:
    torch.manual_seed(0)
    spec = _spec()
    model = build_model(spec)

    train_dataset, val_dataset = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=0, train_size=32, val_size=16
    )
    generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    run_training(model, train_loader, val_loader, config)

    state_dict_path = tmp_path / "state_dict.pt"
    save_state_dict(model, state_dict_path)

    reloaded_model = build_model(spec)
    load_state_dict(reloaded_model, state_dict_path)
    reloaded_model = reloaded_model.eval()

    example_input = torch.randn(1, *spec.input_shape)
    output_path = tmp_path / "model.pt"
    metadata_path = tmp_path / "metadata.json"

    TorchScriptExporter().export(
        reloaded_model,
        example_input,
        output_path,
        metadata_path,
        model_name=spec.name,
        state_dict_path=state_dict_path,
    )

    assert output_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "PASS", metadata.get("error_log")

    traced = torch.jit.load(str(output_path))
    traced.eval()

    with torch.inference_mode():
        reloaded_output = reloaded_model(example_input)
        traced_output = traced(example_input)

    parity = compare_outputs(reloaded_output, traced_output, rtol=CPU_FP32_RTOL, atol=CPU_FP32_ATOL)
    assert parity.allclose, parity.to_dict()
