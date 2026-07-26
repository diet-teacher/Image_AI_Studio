#!/usr/bin/env python
"""Run TorchScript and AOTInductor export for both models, CPU (and CUDA
if available). Cross-platform (pure Python + CMake-independent).
"""
from __future__ import annotations

import json
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
from image_ai_studio.tools.msvc_env import ensure_msvc_on_path

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


def _read_status(metadata_path: Path) -> tuple[str, str | None]:
    meta = json.loads(metadata_path.read_text())
    return meta.get("status", "UNKNOWN"), meta.get("error_log")


def main() -> None:
    if not ensure_msvc_on_path():
        print(
            "WARNING: MSVC compiler (cl.exe) not found on PATH and could not be "
            "auto-configured via vswhere/vcvarsall.bat. AOTInductor export will "
            "fail with 'Compiler: cl is not found'. Install the 'Desktop "
            "development with C++' workload, or run this script from an "
            "'x64 Native Tools Command Prompt for VS'.",
            file=sys.stderr,
        )

    example_input = load_tensor(
        ARTIFACTS_COMMON / "input.bin", ARTIFACTS_COMMON / "input.json"
    )

    ts_exporter = TorchScriptExporter()
    aoti_exporter = AOTInductorExporter()

    results: dict[str, tuple[Path, Path]] = {}
    for name, model_cls in MODELS.items():
        state_dict_path = ARTIFACTS_COMMON / f"{name}_state_dict.pt"

        # TorchScript trace export (CPU only per spec section 15)
        ts_out = REPO_ROOT / "artifacts" / "torchscript" / name / "model.pt"
        ts_meta = REPO_ROOT / "artifacts" / "torchscript" / name / "metadata.json"
        model = load_model(name, model_cls)
        ts_exporter.export(
            model, example_input, ts_out, ts_meta, model_name=name, state_dict_path=state_dict_path
        )
        results[f"{name}_torchscript"] = (ts_out, ts_meta)

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
        results[f"{name}_aoti_cpu"] = (aoti_cpu_out, aoti_cpu_meta)

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
            results[f"{name}_aoti_cuda"] = (aoti_cuda_out, aoti_cuda_meta)
        else:
            print(f"{name} AOTInductor CUDA export: SKIPPED (no CUDA on this machine)")

    print("\nExport summary:")
    failures = []
    for key, (path, meta_path) in results.items():
        status, error_log = _read_status(meta_path)
        print(f"  {key}: {status} ({path})")
        if status != "PASS":
            failures.append(key)
            print(f"    error: {error_log}")

    if failures:
        print(f"\n{len(failures)} export(s) failed: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
