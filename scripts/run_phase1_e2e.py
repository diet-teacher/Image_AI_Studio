#!/usr/bin/env python
"""Phase 1 end-to-end pipeline:

    Model JSON -> ModelSpec -> validate_model_spec() -> build_model()
        -> torch.nn.Module -> TorchScriptExporter (Phase 0, reused)
        -> run_torchscript.exe (Phase 0, reused) -> C++ inference
        -> parity vs. Python reference

This does not introduce a new C++ runner, a new TorchScript export path,
or new tensor I/O / parity code. It proves that an arbitrary Model
Definition Layer model (not just TinyCNN/TinyResidualCNN) can ride the
exact same Phase 0 C++ inference pipeline end to end -- run_torchscript.exe
is a generic runner that takes --model as a CLI argument, so the same
binary already built for Phase 0 also runs this model with zero changes.

Requires a built run_torchscript (see scripts/build_torchscript.py); this
script builds it automatically if missing. Never falls back to CPU for a
CUDA request -- a missing CUDA device is reported as SKIPPED via the
existing run_and_compare.run_case()/report.py status vocabulary.

Usage:

    python scripts/run_phase1_e2e.py
    python scripts/run_phase1_e2e.py --model-json examples/models/phase1_e2e_model.json
    python scripts/run_phase1_e2e.py --model-json path/to/other_model.json

--model-json defaults to examples/models/phase1_e2e_model.json, so the
no-argument form keeps working exactly as before. The JSON file is the
single source of truth for each example model -- there is no parallel
hand-written Python ModelSpec to keep in sync, so adding another
--model-json example never requires touching this script.
JSON <-> ModelSpec equivalence in general (not tied to any one example
model) is already covered by tests/model_definition/test_serialization.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.model_definition.specs import ModelSpec
from image_ai_studio.model_definition.validation import validate_model_spec
from image_ai_studio.parity.tensor_io import save_tensor
from image_ai_studio.tools.run_and_compare import find_runner_binary, run_case

SEED = 20260728
DEVICES = ["cpu", "cuda"]

MODEL_JSON = REPO_ROOT / "examples" / "models" / "phase1_e2e_model.json"
ARTIFACTS_COMMON = REPO_ROOT / "artifacts" / "common"
ARTIFACTS_TORCHSCRIPT = REPO_ROOT / "artifacts" / "torchscript"
ARTIFACTS_REFERENCE = REPO_ROOT / "artifacts" / "reference"
BUILD_DIR = REPO_ROOT / "build-torchscript"


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-json",
        type=Path,
        default=MODEL_JSON,
        help=f"Path to a ModelSpec JSON file (default: {MODEL_JSON})",
    )
    return parser.parse_args(argv)


def load_and_validate(model_json_path: Path) -> tuple[ModelSpec, list]:
    """JSON -> ModelSpec -> validate_model_spec(), with no torch.nn.Module /
    C++ runner involvement -- kept separate from main() so it can be
    exercised in pytest without needing a built run_torchscript."""
    model_spec = load_model_spec(model_json_path)
    shape_trace = validate_model_spec(model_spec)
    return model_spec, shape_trace


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_json_path: Path = args.model_json

    print("Phase 1 E2E")
    print(f"Model JSON: {model_json_path}")

    print("ModelSpec:")
    try:
        model_spec, shape_trace = load_and_validate(model_json_path)
        print(f"  PASS ({len(model_spec.layers)} layers, final shape {shape_trace[-1].output_shape})")
    except ModelValidationError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 1 E2E: FAIL")
        return 1

    print("PyTorch model build:")
    try:
        set_seed()
        model = build_model(model_spec).eval()
        print("  PASS")
    except ModelValidationError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 1 E2E: FAIL")
        return 1

    # Deterministic input, generated independently of however many random
    # draws model construction used -- re-seed first, matching the pattern
    # already used by image_ai_studio.tools.prepare_test_artifacts.
    set_seed()
    example_input = torch.randn(1, *model_spec.input_shape, dtype=torch.float32)
    input_bin = ARTIFACTS_COMMON / f"{model_spec.name}_input.bin"
    input_meta = ARTIFACTS_COMMON / f"{model_spec.name}_input.json"
    save_tensor(example_input, input_bin, input_meta)

    state_dict_path = ARTIFACTS_COMMON / f"{model_spec.name}_state_dict.pt"
    state_dict_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_dict_path)

    print("TorchScript export:")
    model_pt = ARTIFACTS_TORCHSCRIPT / model_spec.name / "model.pt"
    metadata_path = ARTIFACTS_TORCHSCRIPT / model_spec.name / "metadata.json"
    TorchScriptExporter().export(
        model,
        example_input,
        model_pt,
        metadata_path,
        model_name=model_spec.name,
        state_dict_path=state_dict_path,
    )
    export_status = json.loads(metadata_path.read_text())["status"]
    print(f"  {export_status}")
    if export_status != "PASS":
        print(f"  see {metadata_path} for error_log")
        print("\nPHASE 1 E2E: FAIL")
        return 1

    # Python reference output per device (CPU always; CUDA only if
    # available), rebuilt fresh from the saved state_dict -- same
    # independence-from-the-in-memory-model pattern as
    # image_ai_studio.tools.run_python_reference.
    for device in DEVICES:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        ref_model = build_model(model_spec).eval().to(device)
        ref_model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
        with torch.inference_mode():
            ref_output = ref_model(example_input.to(device))
        save_tensor(
            ref_output,
            ARTIFACTS_REFERENCE / f"{model_spec.name}_{device}.bin",
            ARTIFACTS_REFERENCE / f"{model_spec.name}_{device}.json",
            layout="NC",
        )

    print("C++ TorchScript runner:")
    runner_binary = find_runner_binary(BUILD_DIR, "torchscript_runner", "run_torchscript")
    if not runner_binary.exists():
        print("  runner binary not found, building via scripts/build_torchscript.py ...")
        build = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_torchscript.py")])
        if build.returncode != 0:
            print("  FAIL: scripts/build_torchscript.py failed")
            print("\nPHASE 1 E2E: FAIL")
            return 1
        runner_binary = find_runner_binary(BUILD_DIR, "torchscript_runner", "run_torchscript")

    device_status: dict[str, str] = {}
    for device in DEVICES:
        result = run_case(
            runner_binary=runner_binary,
            runner_name="torchscript",
            model_name=model_spec.name,
            model_artifact=model_pt,
            device=device,
            input_bin=input_bin,
            input_meta=input_meta,
        )
        device_status[device] = result["status"]
        print(f"  {device.upper()}: {result['status']}")

    ran_at_least_one = any(status == "PASS" for status in device_status.values())
    no_failures = all(status in ("PASS", "SKIPPED") for status in device_status.values())
    parity_ok = ran_at_least_one and no_failures
    print(f"Parity: {'PASS' if parity_ok else 'FAIL'}")

    overall_ok = parity_ok
    print(f"\nPHASE 1 E2E: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
