"""Shared helper for scripts/run_torchscript_tests.py and
run_aoti_tests.py: invoke a runner binary as a subprocess, then compare
its output against the matching Python reference. Backend-agnostic --
takes the binary path and model artifact path as arguments rather than
hardcoding either backend.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from image_ai_studio.parity.compare_outputs import (
    CPU_FP32_ATOL,
    CPU_FP32_RTOL,
    CUDA_FP32_ATOL,
    CUDA_FP32_RTOL,
    compare_outputs,
)
from image_ai_studio.parity.report import append_result
from image_ai_studio.parity.tensor_io import load_tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_COMMON = REPO_ROOT / "artifacts" / "common"
ARTIFACTS_REFERENCE = REPO_ROOT / "artifacts" / "reference"
RESULTS_DIR = REPO_ROOT / "results"
REPORT_LOG = RESULTS_DIR / "test_matrix.json"


def find_runner_binary(build_dir: Path, project_subdir: str, exe_name: str) -> Path:
    """Locate a built runner executable under build_dir/cpp/project_subdir.

    CMake's single-config generators (Ninja, Makefiles) place the binary
    directly in that directory; MSVC's multi-config Visual Studio generator
    nests it under a config subdirectory (Release/, Debug/, ...) instead.
    Check the plain path first, then fall back to Release/.
    """
    suffix = ".exe" if platform.system() == "Windows" else ""
    base = build_dir / "cpp" / project_subdir
    candidates = [base / f"{exe_name}{suffix}"]
    if platform.system() == "Windows":
        candidates.append(base / "Release" / f"{exe_name}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_case(
    *,
    runner_binary: Path,
    runner_name: str,
    model_name: str,
    model_artifact: Path,
    device: str,
    warmup: int = 10,
    repeat: int = 100,
) -> dict:
    record: dict = {
        "runner": runner_name,
        "model": model_name,
        "device": device,
        "repeat": repeat,
    }

    if device == "cuda" and not torch.cuda.is_available():
        record["status"] = "SKIPPED"
        record["note"] = "torch.cuda.is_available() == False on this machine"
        append_result(REPORT_LOG, record)
        return record

    if not runner_binary.exists():
        record["status"] = "BLOCKED"
        record["note"] = f"runner binary not found: {runner_binary} (build step did not produce it)"
        append_result(REPORT_LOG, record)
        return record

    if not model_artifact.exists():
        record["status"] = "BLOCKED"
        record["note"] = f"model artifact not found: {model_artifact} (export step did not produce it)"
        append_result(REPORT_LOG, record)
        return record

    out_bin = RESULTS_DIR / f"{model_name}_{runner_name}_{device}.bin"
    out_json = RESULTS_DIR / f"{model_name}_{runner_name}_{device}.json"

    cmd = [
        str(runner_binary),
        "--model", str(model_artifact),
        "--input-bin", str(ARTIFACTS_COMMON / "input.bin"),
        "--input-meta", str(ARTIFACTS_COMMON / "input.json"),
        "--output-bin", str(out_bin),
        "--output-meta", str(out_json),
        "--device", device,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    record["command"] = " ".join(cmd)
    record["stdout"] = proc.stdout.strip()
    record["stderr"] = proc.stderr.strip()

    if proc.returncode != 0:
        record["status"] = "FAIL"
        record["note"] = f"runner exited with code {proc.returncode}"
        append_result(REPORT_LOG, record)
        return record

    ref_bin = ARTIFACTS_REFERENCE / f"{model_name}_{device}.bin"
    ref_json = ARTIFACTS_REFERENCE / f"{model_name}_{device}.json"
    if not ref_bin.exists():
        record["status"] = "BLOCKED"
        record["note"] = f"Python reference not found: {ref_bin}"
        append_result(REPORT_LOG, record)
        return record

    reference = load_tensor(ref_bin, ref_json)
    candidate = load_tensor(out_bin, out_json)

    rtol, atol = (CPU_FP32_RTOL, CPU_FP32_ATOL) if device == "cpu" else (CUDA_FP32_RTOL, CUDA_FP32_ATOL)
    parity = compare_outputs(reference, candidate, rtol=rtol, atol=atol)
    record["parity"] = parity.to_dict()
    record["status"] = "PASS" if parity.allclose else "FAIL"

    append_result(REPORT_LOG, record)
    return record
