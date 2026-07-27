# Image AI Studio

## Long-term goal

Image AI Studio is intended to let a user design an image AI model in a UI,
train it in PyTorch, export it to a C++ inference environment, and compare
the Python and C++ outputs and performance.

```text
model design
    -> PyTorch model
    -> training/checkpoint
    -> C++ export
    -> Python inference
    -> C++ inference
    -> Python/C++ output & performance comparison
```

## Phase 0

Phase 0 is a technical spike, not the final product.

Its purpose is to validate the riskiest and most uncertain part of the
long-term architecture:

```text
Python PyTorch model
    -> C++ export
    -> load in a C++ program
    -> run with the same input tensor
    -> compare Python and C++ outputs
```

Two deployment paths are evaluated independently:

1. **TorchScript** (`torch.jit.trace`)

   * Deprecated upstream.
   * Used here as the current compatibility path.

2. **torch.export + AOTInductor** (`.pt2`)

   * Evaluated as a newer C++ deployment path.
   * Actual build and runtime behavior is tested rather than assumed.

`run_torchscript` and `run_aoti` are separate executables built from
separate CMake targets.

A failure in the AOTInductor path must not block building or testing the
TorchScript path, and vice versa.

---

## Phase 1

Based on the Phase 0 results (`docs/phase0_results.md`), Phase 1 and
onward use **TorchScript only** as the C++ deployment/inference path.
AOTInductor is excluded from new Phase 1 work because of the Windows
CPU runtime teardown crash and the CUDA Compute Capability limitation
found in Phase 0; the existing AOTInductor code is kept for the record
but nothing new depends on it.

Phase 1 builds the **Model Definition Layer** that will sit at the
center of Image AI Studio once a UI exists:

```text
Model Definition
    -> Shape Inference / Validation
    -> PyTorch Model Builder
    -> torch.nn.Module
    -> TorchScript Export
    -> C++ Inference
```

See `docs/phase1_design.md` for the full design (supported layers,
shape inference, validation, the JSON format, and how it plugs into the
existing TorchScript exporter). Phase 1 does not include the PySide6
UI, training, IPC, or detection/segmentation.

---

## Cross-platform scope

The primary target is:

```text
Windows 11
Visual Studio 2022
MSVC
x64 Release
NVIDIA CUDA GPU
```

The C++ code is written as standard C++17 without intentionally depending
on MSVC-specific APIs, so the same CMake project can also be built on
macOS and Linux.

### Windows

Primary validation environment.

```text
CPU inference
CUDA inference
TorchScript
AOTInductor
C++ parity testing
```

### Linux

Expected to support both CPU and CUDA with the same C++ implementation.

Linux CUDA validation is not yet complete.

### macOS

CPU-only validation environment.

Apple Silicon does not provide NVIDIA CUDA support, so:

```text
--device cuda
```

should return `UNSUPPORTED`.

The runner must never silently fall back to CPU.

See `docs/phase0_results.md` for environments that have actually been
executed and validated.

---

## What's included

* `TinyCNN`
* `TinyResidualCNN`
* Residual connection testing
* BatchNorm running statistics
* Reproducible test input tensors
* Reproducible shared `state_dict`s
* Fixed random seed
* SHA-256 artifact checksums
* Python CPU reference output
* Python CUDA reference output where available
* TorchScript trace export
* AOTInductor export
* Export environment metadata
* Independent C++ TorchScript runner
* Independent C++ AOTInductor runner
* CPU FP32 inference
* CUDA FP32 inference where available
* Python/C++ output parity comparison
* 100-iteration repeat stability testing
* Inference timing statistics
* GPU memory observation

---

## What's excluded

Phase 0 intentionally does not include:

* PySide6 UI
* Model graph editor
* Training loop
* `ImageFolder` dataset integration
* Shared memory IPC
* Socket IPC
* JSON-Lines IPC
* Long-running worker process
* Dynamic shapes
* ONNX Runtime
* TensorRT
* Detection
* Segmentation
* Real-time video processing
* Model version management UI
* Git LFS
* Debug builds
* Building LibTorch from source

---

# Setup

## 1. Create a Python environment

Python 3.11 is currently used for Phase 0.

Using conda:

```bash
conda create -n ias python=3.11 pip -y
conda activate ias
```

Verify:

```bash
python --version
python -m pip --version
```

---

## 2. Install common Python dependencies

`requirements.txt` contains only platform-independent Python dependencies.

PyTorch is intentionally **not included** in `requirements.txt`, because
the required PyTorch build depends on the operating system and GPU
environment.

Install the common dependencies first:

```bash
python -m pip install -r requirements.txt
```

Current `requirements.txt`:

```text
filelock==3.32.0
fsspec==2026.6.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
packaging==26.0
sympy==1.14.0
typing_extensions==4.16.0
```

---

## 3. Install PyTorch separately

PyTorch must be installed separately according to the target environment.

This is intentional.

Do not assume that:

```bash
pip install torch
```

will install the desired CUDA-enabled build.

### Windows + NVIDIA CUDA

Install the CUDA-enabled PyTorch wheel appropriate for the environment.

Example:

```bat
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

After installation, verify:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'None')"
```

For a CUDA-enabled environment, the important result is:

```text
CUDA available: True
```

A basic CUDA operation can also be tested:

```bat
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print(y.device); print(y.mean())"
```

### macOS

Install the standard macOS PyTorch package:

```bash
python -m pip install torch==2.12.1
```

CUDA is not expected to be available on macOS.

### Linux + NVIDIA CUDA

Install the PyTorch CUDA wheel appropriate for the Linux CUDA environment.

The exact wheel should be selected based on the target PyTorch and CUDA
configuration rather than being hardcoded into `requirements.txt`.

---

## 4. Install Image AI Studio in editable mode

The project uses a `src/` Python package layout.

Install the project itself into the current Python environment:

```bash
python -m pip install -e .
```

`-e` means editable installation.

Python will reference the current source tree directly, so changes under:

```text
src/image_ai_studio/
```

are reflected without reinstalling the package after every edit.

Verify:

```bash
python -c "import image_ai_studio; print(image_ai_studio.__file__)"
```

The path should point to this repository's:

```text
src/image_ai_studio/
```

directory.

---

## 5. Inspect the environment

Run:

```bash
python scripts/inspect_environment.py
```

The environment inspection should report information including:

```text
Python version
PyTorch version
PyTorch CUDA version
CUDA availability
GPU name
GPU compute capability
CUDA Toolkit
NVIDIA driver
CMake
compiler
LibTorch location
```

---

# LibTorch for the C++ build

A separate LibTorch download is not required for the default Phase 0
workflow.

The C++ build uses the LibTorch files bundled with the installed Python
`torch` package.

Retrieve its CMake path with:

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

This allows the Python and C++ sides to use the same PyTorch installation
and avoids version drift between:

```text
Python PyTorch
C++ LibTorch
```

For example, installing a CUDA-enabled PyTorch wheel on Windows also
means the C++ build uses the LibTorch bundled with that same PyTorch
installation.

---

# Windows C++ build configuration

The primary Phase 0 C++ target is:

```text
Platform: x64
Configuration: Release
```

Debug builds are intentionally outside the Phase 0 scope.

Debug and Release MSVC CRT/ABI combinations must not be mixed.

Configuration errors caused by incompatible Debug/Release combinations
are classified as:

```text
INVALID_BUILD_CONFIGURATION
```

rather than as TorchScript or AOTInductor backend failures.

See:

```text
docs/build.md
```

for detailed build instructions.

---

# AOTInductor capability probe

Before attempting the full AOTInductor C++ runner, verify that the
installed LibTorch distribution contains the required AOTInductor C++
runtime.

Run:

```bash
python scripts/probe_aoti_support.py
```

Then build the isolated C++ probe:

```bash
python scripts/build_aoti.py --build-dir build-aoti-probe --probe-only
```

Example package test:

```bash
./build-aoti-probe/cpp/aoti_probe/probe_aoti \
  --package artifacts/aoti/tiny_cnn/cpu/model.pt2 \
  --input-bin artifacts/common/input.bin \
  --input-meta artifacts/common/input.json
```

On Windows, the executable will use the corresponding `.exe` path.

Possible capability states include:

```text
HEADER_NOT_FOUND
LIBRARY_OR_SYMBOL_NOT_FOUND
COMPILE_FAILED
LINK_FAILED
PACKAGE_LOAD_FAILED
SUPPORTED
```

A failed AOTInductor capability probe does not prevent TorchScript from
being built or tested.

See `docs/phase0_results.md` for actual results.

---

# Generate test artifacts

Create the deterministic model weights and test input:

```bash
python -m image_ai_studio.tools.prepare_test_artifacts
```

Generate the Python reference outputs:

```bash
python -m image_ai_studio.tools.run_python_reference
```

Export the models:

```bash
python scripts/export_models.py
```

This produces the artifacts required by the C++ runners.

---

# Build C++ runners

TorchScript and AOTInductor are built independently.

## TorchScript

```bash
python scripts/build_torchscript.py
```

Expected output:

```text
build-torchscript/.../run_torchscript
```

On Windows:

```text
run_torchscript.exe
```

## AOTInductor

```bash
python scripts/build_aoti.py
```

Expected output:

```text
build-aoti/.../run_aoti
```

On Windows:

```text
run_aoti.exe
```

Each backend uses its own build directory.

A broken AOTInductor build must not affect:

```text
build-torchscript/
```

---

# Run tests

## Phase 1 Model Definition Layer (unit tests)

```bash
python -m pip install -r requirements-dev.txt
pytest
```

These tests run entirely on CPU and do not require any built C++
runner. See `docs/phase1_design.md`.

## TorchScript (Phase 0 C++ parity)

```bash
python scripts/run_torchscript_tests.py
```

## AOTInductor

```bash
python scripts/run_aoti_tests.py
```

Where CUDA is available, the tests include CPU and CUDA parity.

Where CUDA is unavailable, CUDA tests are reported as skipped or
unsupported rather than silently falling back to CPU.

---

# Run the complete Phase 0 workflow

Run:

```bash
python scripts/run_phase0.py
```

The workflow performs:

```text
environment inspection
    -> AOTInductor capability probe
    -> deterministic model/input generation
    -> Python reference inference
    -> TorchScript export
    -> AOTInductor export
    -> C++ build
    -> C++ inference
    -> Python/C++ parity comparison
    -> repeat stability tests
    -> result generation
```

A failure in an AOTInductor step does not stop the TorchScript path from
continuing.

---

# Known limitations

* TorchScript validation currently covers only the static
  `torch.jit.trace` path.

* `torch.jit.script` is outside the Phase 0 scope.

* AOTInductor APIs may vary between PyTorch versions.

* Some AOTInductor compile/package/load APIs may live under
  `torch._inductor`, which is an internal/private module and should be
  treated as version-sensitive.

* Only float32 tensors are currently supported by the shared
  binary + JSON tensor format.

* macOS validation is CPU-only.

* Windows x64 Release + NVIDIA CUDA is the primary environment that must
  determine the final backend recommendation.

* A backend that works on macOS CPU does not automatically imply that the
  Windows MSVC/CUDA path is supported.

---

# Dependency policy

`requirements.txt` intentionally does not contain PyTorch.

The dependency strategy is:

```text
requirements.txt
    -> common Python dependencies

PyTorch
    -> installed separately for each OS / GPU configuration

pip install -e .
    -> installs the Image AI Studio source package itself
```

This avoids coupling the repository to one PyTorch CPU/CUDA wheel and
makes platform-specific environments explicit.
