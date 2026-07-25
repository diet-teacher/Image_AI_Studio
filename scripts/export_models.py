#!/usr/bin/env python
"""Run TorchScript and AOTInductor export for both models, CPU (and CUDA
if available). Cross-platform (pure Python + CMake-independent).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from image_ai_studio.export.aoti_exporter import AOTInductorExporter
from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.models.tiny_cnn import TinyCNN
from image_ai_studio.models.tiny_residual_cnn import TinyResidualCNN
from image_ai_studio.parity.tensor_io import load_tensor

ARTIFACTS_COMMON = REPO_ROOT / "artifacts" / "common"
MODELS = {
    "tiny_cnn": TinyCNN,
    "tiny_residual_cnn": TinyResidualCNN,
}


def load_model(name: str, model_cls) -> torch.nn.Module:
    state_dict_path = ARTIFACTS_COMMON / f"{name}_state_dict.pt"
    model = model_cls()
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return model.eval()


def main() -> None:
    example_input = load_tensor(
        ARTIFACTS_COMMON / "input.bin", ARTIFACTS_COMMON / "input.json"
    )

    ts_exporter = TorchScriptExporter()
    aoti_exporter = AOTInductorExporter()

    results = {}
    for name, model_cls in MODELS.items():
        state_dict_path = ARTIFACTS_COMMON / f"{name}_state_dict.pt"

        # TorchScript trace export (CPU only per spec section 15)
        ts_out = REPO_ROOT / "artifacts" / "torchscript" / name / "model.pt"
        ts_meta = REPO_ROOT / "artifacts" / "torchscript" / name / "metadata.json"
        model = load_model(name, model_cls)
        ts_exporter.export(
            model, example_input, ts_out, ts_meta, model_name=name, state_dict_path=state_dict_path
        )
        results[f"{name}_torchscript"] = ts_out

        # AOTInductor export, CPU
        aoti_cpu_out = REPO_ROOT / "artifacts" / "aoti" / name / "cpu" / "model.pt2"
        aoti_cpu_meta = REPO_ROOT / "artifacts" / "aoti" / name / "cpu" / "metadata.json"
        model = load_model(name, model_cls)
        aoti_exporter.export(
            model,
            example_input,
            aoti_cpu_out,
            aoti_cpu_meta,
            model_name=name,
            state_dict_path=state_dict_path,
            device="cpu",
        )
        results[f"{name}_aoti_cpu"] = aoti_cpu_out

        # AOTInductor export, CUDA -- only if available; a real separate
        # artifact, never a copy of the CPU one.
        if torch.cuda.is_available():
            aoti_cuda_out = REPO_ROOT / "artifacts" / "aoti" / name / "cuda" / "model.pt2"
            aoti_cuda_meta = REPO_ROOT / "artifacts" / "aoti" / name / "cuda" / "metadata.json"
            model = load_model(name, model_cls)
            aoti_exporter.export(
                model,
                example_input,
                aoti_cuda_out,
                aoti_cuda_meta,
                model_name=name,
                state_dict_path=state_dict_path,
                device="cuda",
            )
            results[f"{name}_aoti_cuda"] = aoti_cuda_out
        else:
            print(f"{name} AOTInductor CUDA export: SKIPPED (no CUDA on this machine)")

    print("\nExport summary:")
    for key, path in results.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
