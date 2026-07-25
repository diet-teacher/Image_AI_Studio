"""Generate the single shared input tensor + state_dicts used by every
downstream step (TorchScript export, AOTInductor export, Python
reference, C++ runners). Must be run exactly once per Phase 0 run so
every consumer loads the same weights.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_ai_studio.models.tiny_cnn import TinyCNN
from image_ai_studio.models.tiny_residual_cnn import TinyResidualCNN
from image_ai_studio.parity.tensor_io import save_tensor

SEED = 20260715
ARTIFACTS_COMMON = Path("artifacts/common")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    set_seed()

    example_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    save_tensor(
        example_input,
        ARTIFACTS_COMMON / "input.bin",
        ARTIFACTS_COMMON / "input.json",
    )

    checksums = {}
    for name, model_cls in [("tiny_cnn", TinyCNN), ("tiny_residual_cnn", TinyResidualCNN)]:
        set_seed()  # re-seed so each model's init is independent of iteration order
        model = model_cls().eval()
        state_dict_path = ARTIFACTS_COMMON / f"{name}_state_dict.pt"
        state_dict_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), state_dict_path)
        checksums[name] = {
            "state_dict_path": str(state_dict_path),
            "sha256": sha256_of_file(state_dict_path),
        }
        print(f"{name}: {checksums[name]['sha256']}")

    checksums["input"] = {
        "bin_path": str(ARTIFACTS_COMMON / "input.bin"),
        "sha256": sha256_of_file(ARTIFACTS_COMMON / "input.bin"),
        "seed": SEED,
    }

    (ARTIFACTS_COMMON / "checksums.json").write_text(json.dumps(checksums, indent=2))
    print(f"\nWrote {ARTIFACTS_COMMON / 'checksums.json'}")


if __name__ == "__main__":
    main()
