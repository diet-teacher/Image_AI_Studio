"""Ensure MSVC's cl.exe is on PATH.

CMake's Visual Studio generator locates MSVC on its own, so the
cpp/*_runner builds (build_torchscript.py, build_aoti.py) never needed
cl.exe on PATH. AOTInductor's C++ codegen is different: torch._inductor
shells out to `cl` directly using the current process's PATH, so a plain
`python export_models.py` run outside a "Developer Command Prompt" fails
with "Compiler: cl is not found" even though the toolchain is installed.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

VSWHERE = (
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Microsoft Visual Studio"
    / "Installer"
    / "vswhere.exe"
)


def _find_vs_install_path() -> str | None:
    if not VSWHERE.exists():
        return None
    try:
        result = subprocess.run(
            [
                str(VSWHERE),
                "-latest",
                "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return result.stdout.strip() or None


def ensure_msvc_on_path() -> bool:
    """Make `cl.exe` resolvable on PATH for this process, if possible.

    Returns True if cl.exe is (now) on PATH, False if it could not be
    found or configured. No-op (returns True) on non-Windows platforms.
    """
    if platform.system() != "Windows":
        return True
    if shutil.which("cl") is not None:
        return True

    install_path = _find_vs_install_path()
    if install_path is None:
        return False
    vcvarsall = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.exists():
        return False

    proc = subprocess.run(
        f'cmd /c ""{vcvarsall}" x64 && set"',
        capture_output=True,
        text=True,
        shell=True,
    )
    if proc.returncode != 0:
        return False

    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            os.environ[key] = value

    return shutil.which("cl") is not None
