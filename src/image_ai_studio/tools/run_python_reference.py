"""Run CPU (and CUDA, if available) Python inference using the shared
state_dict + input artifacts, and save the outputs as reference
tensors that C++ output will later be compared against.

CPU and CUDA references are never shared/copied between each other --
CUDA reference generation loads the same CPU state_dict then moves the
model to CUDA, exactly like the spec requires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_ai_studio.models.tiny_cnn import TinyCNN
from image_ai_studio.models.tiny_residual_cnn import TinyResidualCNN
from image_ai_studio.parity.tensor_io import load_tensor, save_tensor

ARTIFACTS_COMMON = Path("artifacts/common")
ARTIFACTS_REFERENCE = Path("artifacts/reference")

MODELS = {
    "tiny_cnn": TinyCNN,
    "tiny_residual_cnn": TinyResidualCNN,
}


def run_reference(name: str, model_cls, device: str) -> str:
    state_dict_path = ARTIFACTS_COMMON / f"{name}_state_dict.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(
            f"{state_dict_path} missing -- run prepare_test_artifacts.py first"
        )

    if device == "cuda" and not torch.cuda.is_available():
        print(f"{name} CUDA reference: SKIPPED (torch.cuda.is_available() == False)")
        return "SKIPPED"

    model = model_cls()
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    example_input = load_tensor(
        ARTIFACTS_COMMON / "input.bin", ARTIFACTS_COMMON / "input.json"
    ).to(device)

    with torch.inference_mode():
        output = model(example_input)

    out_bin = ARTIFACTS_REFERENCE / f"{name}_{device}.bin"
    out_json = ARTIFACTS_REFERENCE / f"{name}_{device}.json"
    save_tensor(output, out_bin, out_json, layout="NC")
    print(f"{name} {device} reference: PASS -> {out_bin}")
    return "PASS"


def main() -> None:
    results = {}
    for name, model_cls in MODELS.items():
        results[f"{name}_cpu"] = run_reference(name, model_cls, "cpu")
        results[f"{name}_cuda"] = run_reference(name, model_cls, "cuda")

    print("\nSummary:")
    for key, status in results.items():
        print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
