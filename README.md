# Image AI Studio

## Long-term goal

Image AI Studio is meant to let a user design an image AI model in a UI,
train it in PyTorch, export it to a C++ inference environment, and
compare the Python and C++ outputs and performance:

```
model design -> PyTorch model -> training/checkpoint -> C++ export
    -> Python inference -> C++ inference -> Python/C++ output & perf comparison
```

## Phase 0 (this repo, current state)

Phase 0 is a technical spike, not a product. It validates only the
riskiest, most uncertain part of the long-term flow:

```
Python PyTorch model -> C++ export -> load in a C++ program
    -> run on the same input tensor -> compare Python vs C++ output
```

Two deployment paths are compared, independently:

1. **TorchScript** (`torch.jit.trace`) -- deprecated upstream, validated
   here as the current compatibility path.
2. **torch.export + AOTInductor** (`.pt2` package) -- the direction
   PyTorch is pushing, validated here for real build/runtime behavior
   rather than assumed to work.

**`run_torchscript` and `run_aoti` are separate executables built from
separate CMake targets.** A failure in the AOTInductor path never blocks
building or testing the TorchScript path, and vice versa.

### Cross-platform scope

Windows 11 + Visual Studio 2022 + MSVC + an NVIDIA CUDA GPU is the
primary target. The C++ code is standard C++17 with no MSVC-specific
APIs, so the same CMakeLists.txt also builds on macOS and Linux:

- **Windows**: CPU + CUDA (primary target; not yet exercised in this
  repo's history -- see `docs/phase0_results.md` for what has actually
  been run so far).
- **Linux**: CPU + CUDA (same code, not yet exercised in this repo's
  history).
- **macOS**: CPU only. There is no NVIDIA GPU on Apple Silicon, so CUDA
  device requests are expected to report `UNSUPPORTED`, not silently
  fall back to CPU.

### What's included

- `TinyCNN` and `TinyResidualCNN` (the latter exercises a skip
  connection with BatchNorm running stats).
- Reproducible shared input tensor + `state_dict`s (fixed seed, SHA-256
  checksums).
- Python CPU (and CUDA, where available) reference outputs.
- TorchScript trace export and AOTInductor export, each with full
  environment metadata.
- Independent `run_torchscript` / `run_aoti` C++ runners: CPU FP32 (and
  CUDA FP32 where available) inference, output-parity comparison against
  the Python reference, 100-iteration repeat stability, timing stats,
  GPU memory observation.

### What's excluded

UI, model graph editor, training loop, `ImageFolder` datasets, shared
memory / socket / JSON-Lines IPC, a long-running worker process, dynamic
shapes, ONNX Runtime, TensorRT, detection/segmentation, real-time video,
model-version-management UI, Git LFS, Debug builds, building LibTorch
from source.

## Setup

### 1. Python environment (conda + requirements.txt)

```bash
conda create -n image-ai-studio python=3.11 -y
conda activate image-ai-studio
pip install -r requirements.txt
```

`requirements.txt` is a `pip freeze` snapshot of the exact versions this
spike was validated against (see `docs/phase0_results.md`). Conda is
only used to create an isolated interpreter; regenerate the same
environment on any OS with the two commands above.

Verify the Python/PyTorch/CUDA environment:

```bash
python scripts/inspect_environment.py
```

### 2. LibTorch for the C++ build

No separate LibTorch download is required. `CMAKE_PREFIX_PATH` is
derived directly from the installed `torch` pip package, so the C++
build automatically uses the exact same PyTorch version as the Python
side:

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

This keeps the Python `torch` and C++ LibTorch versions in lockstep by
construction -- there is no separate LibTorch version to drift.

Windows x64 Release is the only C++ build configuration validated by
this spike (see `docs/build.md`). Debug builds and Debug/Release LibTorch
mixing are out of scope and will be reported as
`INVALID_BUILD_CONFIGURATION`, not a backend failure.

### 3. AOTInductor capability probe

Before attempting the AOTInductor C++ runner, check whether the current
LibTorch actually ships the AOTInductor C++ runtime:

```bash
python scripts/probe_aoti_support.py   # header + exported-symbol search
python scripts/build_aoti.py --build-dir build-aoti-probe --probe-only
./build-aoti-probe/cpp/aoti_probe/probe_aoti \
  --package artifacts/aoti/tiny_cnn/cpu/model.pt2 \
  --input-bin artifacts/common/input.bin \
  --input-meta artifacts/common/input.json
```

See `docs/phase0_results.md` for the actual result on this machine.

### 4. Export models

```bash
python -m image_ai_studio.tools.prepare_test_artifacts
python -m image_ai_studio.tools.run_python_reference
python scripts/export_models.py
```

### 5. Build each runner independently

```bash
python scripts/build_torchscript.py     # -> build-torchscript/.../run_torchscript
python scripts/build_aoti.py            # -> build-aoti/.../run_aoti
```

Each uses its own build directory; a broken AOTI build never affects
`build-torchscript/`.

### 6. Run tests

```bash
python scripts/run_torchscript_tests.py   # CPU + CUDA-if-available parity
python scripts/run_aoti_tests.py
```

Or run everything end-to-end:

```bash
python scripts/run_phase0.py
```

`run_phase0.py` continues past a failed AOTI step and still runs every
TorchScript step.

## Known limitations

- TorchScript: only the static `torch.jit.trace` path was validated.
  `torch.jit.script` is out of scope for Phase 0.
- AOTInductor: `torch._inductor.aoti_compile_and_package` /
  `aoti_load_package` live under a **private** module (leading
  underscore) in the PyTorch version this was validated against -- there
  is no public non-underscore alias. Treat the API surface as subject to
  change between PyTorch releases.
- Only float32 tensors are supported by the shared bin+json tensor
  format.
- Results in this repository's history so far were produced on macOS
  (CPU only); Windows and Linux/CUDA rows in `docs/phase0_results.md`
  are marked accordingly until run on that hardware.
