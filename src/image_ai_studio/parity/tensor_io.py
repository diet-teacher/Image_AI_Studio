"""Binary tensor + JSON metadata IO shared by Python and (mirrored in) C++.

Format: <name>.bin holds raw little-endian float32 contiguous data;
<name>.json holds shape/dtype/layout metadata. No shared memory, no
pickling -- plain files so any language can read them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SUPPORTED_DTYPE = "float32"


def save_tensor(tensor: torch.Tensor, bin_path: Path, json_path: Path, layout: str = "NCHW") -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"Only float32 is supported in Phase 0, got {tensor.dtype}")

    contiguous = tensor.detach().cpu().contiguous()
    array = contiguous.numpy().astype("<f4", copy=False)

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(bin_path)

    meta = {
        "shape": list(contiguous.shape),
        "dtype": SUPPORTED_DTYPE,
        "layout": layout,
        "byte_order": "little_endian",
        "contiguous": True,
        "element_count": int(array.size),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(meta, indent=2))


def load_tensor(bin_path: Path, json_path: Path) -> torch.Tensor:
    meta = json.loads(json_path.read_text())

    if meta["dtype"] != SUPPORTED_DTYPE:
        raise ValueError(f"Unsupported dtype '{meta['dtype']}', Phase 0 only supports float32")

    expected_bytes = meta["element_count"] * 4
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{bin_path}: size mismatch, file has {actual_bytes} bytes, "
            f"element_count*4 = {expected_bytes} bytes"
        )

    array = np.fromfile(bin_path, dtype="<f4")
    tensor = torch.from_numpy(array.copy()).reshape(meta["shape"])
    return tensor
