#!/usr/bin/env python
"""Run run_aoti for both models on CPU (and CUDA if available and a
CUDA .pt2 artifact was produced), comparing each output against the
Python reference. Appends every result to results/test_matrix.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from image_ai_studio.tools.run_and_compare import find_runner_binary, run_case

MODELS = ["tiny_cnn", "tiny_residual_cnn"]
DEVICES = ["cpu", "cuda"]


def main() -> None:
    runner_binary = find_runner_binary(REPO_ROOT / "build-aoti", "aoti_runner", "run_aoti")

    for model in MODELS:
        for device in DEVICES:
            model_artifact = REPO_ROOT / "artifacts" / "aoti" / model / device / "model.pt2"
            result = run_case(
                runner_binary=runner_binary,
                runner_name="aoti",
                model_name=model,
                model_artifact=model_artifact,
                device=device,
            )
            print(f"{model} aoti {device}: {result['status']}")


if __name__ == "__main__":
    main()
