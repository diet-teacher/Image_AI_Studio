# Build Guide

The CMake project is OS-agnostic standard C++17. This document covers
Windows (the primary target), then the macOS/Linux notes needed because
this repo's C++ code is deliberately not MSVC-specific.

## Common to all platforms

- CMake >= 3.18
- A C++17 compiler
- `CMAKE_PREFIX_PATH` pointing at the LibTorch CMake config that ships
  inside the `torch` pip package:

  ```bash
  python -c "import torch; print(torch.utils.cmake_prefix_path)"
  ```

- All builds use `-DCMAKE_BUILD_TYPE=Release` / `--config Release`.
  Debug is out of scope for Phase 0 (see below).

## Windows (primary target)

Requirements:

- Windows 11
- Visual Studio 2022 (Desktop development with C++ workload), or the
  standalone MSVC Build Tools
- CMake >= 3.18 on PATH
- Python + `pip install -r requirements.txt` in the same environment
  used to derive `CMAKE_PREFIX_PATH`
- For CUDA: an NVIDIA GPU + matching CUDA-enabled `torch` build
  (`pip install torch` from the CUDA index, matching the CUDA toolkit
  actually installed)

Platform/config are fixed:

```
Platform: x64
Configuration: Release
```

Configure + build (PowerShell), TorchScript only:

```powershell
$torchCmake = python -c "import torch; print(torch.utils.cmake_prefix_path)"
cmake -S . -B build-torchscript `
  -DBUILD_TORCHSCRIPT_RUNNER=ON -DBUILD_AOTI_PROBE=OFF -DBUILD_AOTI_RUNNER=OFF `
  -DCMAKE_PREFIX_PATH="$torchCmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build-torchscript --config Release --target run_torchscript
```

AOTInductor only (separate build directory, unaffected by the above):

```powershell
cmake -S . -B build-aoti `
  -DBUILD_TORCHSCRIPT_RUNNER=OFF -DBUILD_AOTI_PROBE=OFF -DBUILD_AOTI_RUNNER=ON `
  -DCMAKE_PREFIX_PATH="$torchCmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build-aoti --config Release --target run_aoti
```

DLL/PATH: the top-level `CMakeLists.txt` already copies the LibTorch
DLLs next to each `.exe` via a post-build step (`if(MSVC) ... copy_if_different`)
so no manual `PATH` edits should be needed to run the built executables
directly from the build directory.

### Debug/Release ABI note (Windows-specific)

MSVC's Debug and Release CRT/ABI are not interchangeable. This spike
only validates Release LibTorch + Release executables. If you see link
errors that look like a Debug/Release CRT mismatch (e.g. `LNK2038`
mismatched `_ITERATOR_DEBUG_LEVEL` or `RuntimeLibrary`), that is recorded
as `INVALID_BUILD_CONFIGURATION`, not a TorchScript/AOTInductor failure
-- fix by ensuring both the executable and LibTorch are Release, x64.

## macOS / Linux

Same CMake invocations, without the `.exe` suffix and without the
`if(MSVC)` DLL-copy step (shared libraries are found via the standard
dynamic linker search path / rpath instead):

```bash
torch_cmake=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S . -B build-torchscript \
  -DBUILD_TORCHSCRIPT_RUNNER=ON -DBUILD_AOTI_PROBE=OFF -DBUILD_AOTI_RUNNER=OFF \
  -DCMAKE_PREFIX_PATH="$torch_cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build-torchscript --config Release
```

On macOS, the built binaries need `DYLD_LIBRARY_PATH` pointing at
LibTorch's `lib/` directory to find `libtorch_cpu.dylib` etc. at runtime
(no CMake `RPATH` is set up in Phase 0):

```bash
export DYLD_LIBRARY_PATH="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
```

On Linux the equivalent variable is `LD_LIBRARY_PATH`.

macOS has no NVIDIA GPU support, so `--device cuda` is expected to
report `UNSUPPORTED` (the runner does not silently fall back to CPU).
Linux is expected to support `--device cuda` with a CUDA-enabled
`torch` build, same code path as Windows -- not yet exercised on this
repo's history (see `docs/phase0_results.md`).

## AOTInductor runtime probe (any platform)

```bash
python scripts/probe_aoti_support.py
python scripts/build_aoti.py --build-dir build-aoti-probe --probe-only
./build-aoti-probe/cpp/aoti_probe/probe_aoti \
  --package artifacts/aoti/tiny_cnn/cpu/model.pt2 \
  --input-bin artifacts/common/input.bin \
  --input-meta artifacts/common/input.json
```

Exit code 0 + `result: SUPPORTED` means the header compiled, the
`torch::inductor::AOTIModelPackageLoader` symbol linked, and a real
`.pt2` package loaded and ran successfully on this machine. Any other
outcome (`HEADER_NOT_FOUND`, `LIBRARY_OR_SYMBOL_NOT_FOUND`,
`COMPILE_FAILED`, `LINK_FAILED`, `PACKAGE_LOAD_FAILED`) means
`cpp/aoti_runner` should not be expected to work and its build should be
attempted with that expectation, not blind optimism.

## Common errors

| Symptom | Likely cause |
|---|---|
| `find_package(Torch REQUIRED)` fails | `CMAKE_PREFIX_PATH` not set / wrong Python env activated when deriving it |
| `library kineto not found` CMake warning | Harmless on CPU-only LibTorch builds; not an error |
| Runner can't find `libtorch_cpu`/`torch_cpu.dll` at runtime | DLL copy step didn't run (Windows) or `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` not set (macOS/Linux) |
| `LNK2038` / CRT mismatch (Windows) | Debug/Release LibTorch or CRT mixed -- record as `INVALID_BUILD_CONFIGURATION`, rebuild both sides Release/x64 |
| `--device cuda` returns `UNSUPPORTED` | Expected when `torch::cuda::is_available()` is false on that machine -- not a bug, no fallback is performed |
