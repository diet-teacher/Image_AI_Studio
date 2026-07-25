#!/usr/bin/env python
"""End-to-end Phase 0 pipeline. Every step's status and log path is
recorded; a failure in any AOTI-related step never stops the
TorchScript steps that follow, matching the spec's no-single-point-of-
failure requirement.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"


def run_step(name: str, cmd: list[str], *, allow_fail: bool = False) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    print(f"\n=== {name} ===")
    print("  cmd:", " ".join(cmd))
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    ok = proc.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(f"  status: {status} (log: {log_path})")
    if not ok and not allow_fail:
        print(f"  NOTE: {name} failed but the pipeline continues to the next step.")
    return ok


def main() -> None:
    py = sys.executable

    run_step("01_inspect_environment", [py, "scripts/inspect_environment.py"])
    run_step("02_probe_aoti_support_headers", [py, "scripts/probe_aoti_support.py"])
    run_step("03_prepare_test_artifacts", [py, "-m", "image_ai_studio.tools.prepare_test_artifacts"])
    run_step("04_run_python_reference", [py, "-m", "image_ai_studio.tools.run_python_reference"])
    run_step("05_export_models", [py, "scripts/export_models.py"])

    # AOTI compile/link/package-load probe -- independent build dir,
    # failure here must not block step 07 below.
    run_step("06a_build_aoti_probe", [py, "scripts/build_aoti.py", "--build-dir", "build-aoti-probe", "--probe-only"])
    run_step("06b_run_aoti_probe", [
        str(REPO_ROOT / "build-aoti-probe" / "cpp" / "aoti_probe" / "probe_aoti"),
        "--package", "artifacts/aoti/tiny_cnn/cpu/model.pt2",
        "--input-bin", "artifacts/common/input.bin",
        "--input-meta", "artifacts/common/input.json",
    ])

    run_step("07_build_torchscript", [py, "scripts/build_torchscript.py"])
    run_step("08_build_aoti_runner", [py, "scripts/build_aoti.py"])

    run_step("09_run_torchscript_tests", [py, "scripts/run_torchscript_tests.py"])
    run_step("10_run_aoti_tests", [py, "scripts/run_aoti_tests.py"])

    print("\nAll steps attempted. See results/logs/ for per-step output and "
          "results/test_matrix.json for the parity/status matrix.")


if __name__ == "__main__":
    main()
