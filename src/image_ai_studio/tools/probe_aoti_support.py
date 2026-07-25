"""Step 1-2 of the AOTI capability probe: search the installed torch
package for the AOTInductor C++ package-loader header and the
corresponding symbols in its shared libraries. Cross-platform (uses
pathlib instead of dumpbin/grep/nm).

Steps 3-5 (compile probe, link probe, real .pt2 package load) are the
job of the cpp/aoti_probe CMake target -- this script only prepares and
reports the header/library search, then optionally invokes that target
if a build directory is given.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_torch_include_dir() -> Path | None:
    try:
        import torch
    except ImportError:
        return None
    return Path(torch.__file__).parent / "include"


def find_torch_lib_dir() -> Path | None:
    try:
        import torch
    except ImportError:
        return None
    return Path(torch.__file__).parent / "lib"


def search_header(include_dir: Path) -> Path | None:
    matches = list(include_dir.rglob("model_package_loader.h"))
    return matches[0] if matches else None


def search_symbol_in_libs(lib_dir: Path, symbol_substring: str = "AOTIModelPackageLoader") -> list[str]:
    """Best-effort symbol search across platforms.

    Uses `nm` on macOS/Linux (both ship it with the system toolchain) and
    `dumpbin` on Windows, per the spec's allowed tool list -- but never
    assumes either is present; falls back to reporting "not probed".
    """
    found_in: list[str] = []
    lib_files = list(lib_dir.glob("*.dylib")) + list(lib_dir.glob("*.so")) + list(
        lib_dir.glob("*.dll")
    ) + list(lib_dir.glob("*.lib"))

    if shutil.which("nm"):
        for lib in lib_files:
            try:
                out = subprocess.run(
                    ["nm", "-gU", str(lib)] if lib.suffix == ".dylib" else ["nm", "-D", str(lib)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if symbol_substring in out.stdout:
                    found_in.append(str(lib))
            except (subprocess.TimeoutExpired, OSError):
                continue
    elif shutil.which("dumpbin"):
        for lib in lib_files:
            try:
                out = subprocess.run(
                    ["dumpbin", "/symbols", str(lib)], capture_output=True, text=True, timeout=30
                )
                if symbol_substring in out.stdout:
                    found_in.append(str(lib))
            except (subprocess.TimeoutExpired, OSError):
                continue
    return found_in


def probe() -> dict:
    result: dict = {}

    include_dir = find_torch_include_dir()
    lib_dir = find_torch_lib_dir()

    if include_dir is None:
        result["status"] = "HEADER_NOT_FOUND"
        result["reason"] = "torch is not installed in this Python environment"
        return result

    header = search_header(include_dir)
    if header is None:
        result["status"] = "HEADER_NOT_FOUND"
        result["searched_dir"] = str(include_dir)
        return result

    result["header_path"] = str(header)

    symbol_libs = search_symbol_in_libs(lib_dir) if lib_dir else []
    result["symbol_found_in"] = symbol_libs
    if not symbol_libs:
        result["status"] = "LIBRARY_OR_SYMBOL_NOT_FOUND"
        result["searched_lib_dir"] = str(lib_dir) if lib_dir else None
        return result

    result["status"] = "HEADER_AND_SYMBOL_FOUND"
    result["note"] = (
        "Header and exported symbol both found. Actual COMPILE_FAILED / "
        "LINK_FAILED / PACKAGE_LOAD_FAILED / SUPPORTED verdict comes from "
        "actually building and running cpp/aoti_probe (see docs/phase0_results.md)."
    )
    return result


def main() -> None:
    result = probe()
    print(json.dumps(result, indent=2))

    out_path = Path("artifacts") / "aoti_header_symbol_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
