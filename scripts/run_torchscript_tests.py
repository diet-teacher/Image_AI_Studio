#!/usr/bin/env python
"""Run run_torchscript for both models on CPU (and CUDA if available),
comparing each output against the Python reference. Appends every
result -- PASS, FAIL, or SKIPPED -- to results/test_matrix.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from image_ai_studio.tools.run_and_compare import run_case

MODELS = ["tiny_cnn", "tiny_residual_cnn"]
DEVICES = ["cpu", "cuda"]


def main() -> None:
    import platform

    suffix = ".exe" if platform.system() == "Windows" else ""
    runner_binary = REPO_ROOT / "build-torchscript" / "cpp" / "torchscript_runner" / f"run_torchscript{suffix}"

    for model in MODELS:
        model_artifact = REPO_ROOT / "artifacts" / "torchscript" / model / "model.pt"
        for device in DEVICES:
            result = run_case(
                runner_binary=runner_binary,
                runner_name="torchscript",
                model_name=model,
                model_artifact=model_artifact,
                device=device,
            )
            print(f"{model} torchscript {device}: {result['status']}")


if __name__ == "__main__":
    main()
