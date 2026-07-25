#!/usr/bin/env python
"""Independent Release build of run_aoti (and, optionally, probe_aoti).
A failure here must never be treated as a TorchScript failure -- see
scripts/build_torchscript.py, which this script never touches.

Usage: python scripts/build_aoti.py [--build-dir build-aoti] [--probe-only]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def torch_cmake_prefix_path() -> str:
    import torch

    return torch.utils.cmake_prefix_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build-aoti")
    parser.add_argument("--probe-only", action="store_true", help="Build probe_aoti instead of run_aoti")
    args = parser.parse_args()

    build_dir = REPO_ROOT / args.build_dir
    prefix_path = torch_cmake_prefix_path()
    target = "probe_aoti" if args.probe_only else "run_aoti"

    configure_cmd = [
        "cmake", "-S", str(REPO_ROOT), "-B", str(build_dir),
        "-DBUILD_TORCHSCRIPT_RUNNER=OFF",
        f"-DBUILD_AOTI_PROBE={'ON' if args.probe_only else 'OFF'}",
        f"-DBUILD_AOTI_RUNNER={'OFF' if args.probe_only else 'ON'}",
        f"-DCMAKE_PREFIX_PATH={prefix_path}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release", "--target", target]

    print("Configure:", " ".join(configure_cmd))
    configure = subprocess.run(configure_cmd)
    if configure.returncode != 0:
        print(f"AOTI ({target}) CONFIGURE: FAIL -- this does not affect run_torchscript", file=sys.stderr)
        return configure.returncode

    print("Build:", " ".join(build_cmd))
    build = subprocess.run(build_cmd)
    if build.returncode != 0:
        print(f"AOTI ({target}) BUILD: FAIL -- this does not affect run_torchscript", file=sys.stderr)
        return build.returncode

    print(f"AOTI ({target}) BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
