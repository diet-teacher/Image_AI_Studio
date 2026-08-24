"""Cross-platform environment inspection for the Phase 0 spike.

Collects Python/PyTorch/CUDA/compiler/CMake/git facts without assuming
Windows-only tools (dumpbin, PowerShell). Every field is filled with the
real probed value or an explicit "NOT_FOUND" / "N/A" marker -- never a
guess.
"""
from __future__ import annotations

import json
import locale
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _decode_output(data: bytes | None) -> str:
    if not data:
        return ""
    encoding = "mbcs" if sys.platform == "win32" else locale.getpreferredencoding(False)
    return data.decode(encoding, errors="replace")


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=15, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        return None

    text = (_decode_output(out.stdout) or _decode_output(out.stderr)).strip()
    return text if text else None


def _first_line(text: str | None) -> str | None:
    if not text:
        return None
    return text.splitlines()[0].strip()


def inspect_environment() -> dict:
    info: dict = {}

    info["os"] = platform.platform()
    info["os_system"] = platform.uname().system
    info["os_release"] = platform.release()
    info["architecture"] = platform.machine()

    info["python_version"] = sys.version.split()[0]
    info["python_executable"] = sys.executable

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_build_version"] = torch.version.cuda
        info["torch_cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        info["cuda_is_available"] = torch.cuda.is_available()
        info["mps_is_available"] = (
            torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        )
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["gpu_compute_capability"] = f"{major}.{minor}"
        else:
            info["gpu_name"] = None
            info["gpu_compute_capability"] = None
        info["torch_cmake_prefix_path"] = torch.utils.cmake_prefix_path
    except ImportError:
        info["torch_version"] = None
        info["torch_import_error"] = "torch is not installed in this Python environment"

    info["nvidia_smi_present"] = shutil.which("nvidia-smi") is not None
    if info["nvidia_smi_present"]:
        info["nvidia_driver_version"] = _first_line(
            _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        )
    else:
        info["nvidia_driver_version"] = None

    info["nvcc_present"] = shutil.which("nvcc") is not None
    info["cuda_toolkit_version"] = (
        _first_line(_run(["nvcc", "--version"])) if info["nvcc_present"] else None
    )

    info["cmake_version"] = _first_line(_run(["cmake", "--version"]))
    info["git_version"] = _first_line(_run(["git", "--version"]))

    if sys.platform == "win32" or os.name == "nt":
        info["compiler"] = _first_line(_run(["cl"])) or "MSVC (cl.exe) not found on PATH"
        info["visual_studio_version"] = _run(
            ["powershell", "-Command", "(Get-CimInstance MSFT_VSInstance).Version"]
        )
    else:
        compiler_cmd = "clang++" if shutil.which("clang++") else "g++"
        info["compiler"] = _first_line(_run([compiler_cmd, "--version"]))
        info["visual_studio_version"] = "N/A (not Windows)"

    return info


def main() -> None:
    info = inspect_environment()
    print(json.dumps(info, indent=2))

    out_path = Path("artifacts") / "environment_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
