#!/usr/bin/env python
"""Independent Release build of run_torchscript. Never touches
build-aoti/ or build-aoti-probe/, and does not require them to exist or
succeed.

Usage: python scripts/build_torchscript.py [--build-dir build-torchscript]
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
    parser.add_argument("--build-dir", default="build-torchscript")
    args = parser.parse_args()

    build_dir = REPO_ROOT / args.build_dir
    prefix_path = torch_cmake_prefix_path()

    configure_cmd = [
        "cmake", "-S", str(REPO_ROOT), "-B", str(build_dir),
        "-DBUILD_TORCHSCRIPT_RUNNER=ON",
        "-DBUILD_AOTI_PROBE=OFF",
        "-DBUILD_AOTI_RUNNER=OFF",
        f"-DCMAKE_PREFIX_PATH={prefix_path}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "run_torchscript"]

    print("Configure:", " ".join(configure_cmd))
    configure = subprocess.run(configure_cmd)
    if configure.returncode != 0:
        print("TorchScript CONFIGURE: FAIL", file=sys.stderr)
        return configure.returncode

    print("Build:", " ".join(build_cmd))
    build = subprocess.run(build_cmd)
    if build.returncode != 0:
        print("TorchScript BUILD: FAIL", file=sys.stderr)
        return build.returncode

    print("TorchScript BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
