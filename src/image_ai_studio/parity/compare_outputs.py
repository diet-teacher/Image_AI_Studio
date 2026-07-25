"""Output parity comparison: dtype/shape/element-count + error metrics."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

# Initial tolerances from the Phase 0 spec. Do not change silently --
# any change must be recorded (old value, new value, reason) in
# docs/phase0_results.md.
CPU_FP32_RTOL = 1e-5
CPU_FP32_ATOL = 1e-6
CUDA_FP32_RTOL = 1e-4
CUDA_FP32_ATOL = 1e-5


@dataclass
class ParityResult:
    dtype_match: bool
    shape_match: bool
    element_count_match: bool
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    allclose: bool
    rtol: float
    atol: float

    def to_dict(self) -> dict:
        return asdict(self)


def compare_outputs(
    reference: torch.Tensor, candidate: torch.Tensor, rtol: float, atol: float
) -> ParityResult:
    dtype_match = reference.dtype == candidate.dtype
    shape_match = list(reference.shape) == list(candidate.shape)
    element_count_match = reference.numel() == candidate.numel()

    if not (dtype_match and shape_match and element_count_match):
        return ParityResult(
            dtype_match=dtype_match,
            shape_match=shape_match,
            element_count_match=element_count_match,
            max_abs_error=float("nan"),
            mean_abs_error=float("nan"),
            max_rel_error=float("nan"),
            allclose=False,
            rtol=rtol,
            atol=atol,
        )

    ref = reference.double()
    cand = candidate.double()
    abs_error = (ref - cand).abs()
    rel_error = abs_error / ref.abs().clamp_min(1e-12)

    return ParityResult(
        dtype_match=dtype_match,
        shape_match=shape_match,
        element_count_match=element_count_match,
        max_abs_error=abs_error.max().item(),
        mean_abs_error=abs_error.mean().item(),
        max_rel_error=rel_error.max().item(),
        allclose=bool(torch.allclose(ref, cand, rtol=rtol, atol=atol)),
        rtol=rtol,
        atol=atol,
    )
