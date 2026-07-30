#!/usr/bin/env python
"""Phase 4A/4B 학습 엔드투엔드 파이프라인:

    Model JSON -> ModelSpec -> build_model()
        -> synthetic Train/Validation (torchvision 없음)
        -> 실제 학습 (Adam + CrossEntropyLoss, Phase 4A 고정)
        -> best epoch(최소 validation loss) 추적 (Phase 4B)
        -> training history JSON 저장 (Phase 4B)
        -> best epoch의 state_dict 저장 -> 새 model에 재로드 -> eval
        -> TorchScriptExporter (Phase 0 재사용)
        -> run_torchscript.exe (Phase 0 재사용) -> C++ 추론
        -> Python/C++ parity 비교

Phase 0~3은 전부 random-init 가중치로 parity만 검증했다 -- 이 스크립트가
이 프로젝트에서 처음으로 "실제로 학습된" 모델을 끝까지 배포하는 경로다.
Phase 4B부터는 마지막 epoch이 아니라 **best epoch(최소 validation loss)**
의 가중치를 export한다 -- run_training()은 파일을 쓰지 않고 best
state_dict를 메모리로 반환하며, 그걸 저장/재로드하는 건 이 스크립트의
책임이다 (training/loop.py, training/history.py 참고).

- model_definition/*, export/*, run_and_compare.py, C++ 러너 전부 변경 없음
- run_torchscript 빌드 필요 (scripts/build_torchscript.py), 없으면 자동 빌드
- CUDA 미가용 시 CPU 폴백 없이 SKIPPED 처리 (Phase 0/1과 동일 정책)

사용법:

    python scripts/run_training_e2e.py
    python scripts/run_training_e2e.py --model-json examples/models/phase4_training_model.json

--model-json 기본값: examples/models/phase4_training_model.json
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
from torch.utils.data import DataLoader

from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.model_definition.validation import validate_model_spec
from image_ai_studio.parity.tensor_io import save_tensor
from image_ai_studio.tools.run_and_compare import find_runner_binary, run_case
from image_ai_studio.training.checkpoint import load_state_dict, save_state_dict
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.dataset import make_train_val_datasets
from image_ai_studio.training.history import save_training_history
from image_ai_studio.training.loop import run_training

SEED = 20260730
DEVICES = ["cpu", "cuda"]
TRAINING_CONFIG = TrainingConfig(epochs=10, batch_size=8, learning_rate=2e-3)
TRAIN_SIZE = 64
VAL_SIZE = 32

MODEL_JSON = REPO_ROOT / "examples" / "models" / "phase4_training_model.json"
ARTIFACTS_COMMON = REPO_ROOT / "artifacts" / "common"
ARTIFACTS_TRAINING = REPO_ROOT / "artifacts" / "training"
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_json_path: Path = args.model_json

    print("Phase 4B Training E2E")
    print(f"Model JSON: {model_json_path}")

    print("ModelSpec:")
    try:
        model_spec = load_model_spec(model_json_path)
        shape_trace = validate_model_spec(model_spec)
        final_shape = shape_trace[-1].output_shape
        print(f"  PASS ({len(model_spec.layers)} layers, final shape {final_shape})")
    except ModelValidationError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4B E2E: FAIL")
        return 1

    print("Classification output check:")
    if len(final_shape) != 1:
        print(f"  FAIL: final output shape must be rank 1 (num_classes,), got {final_shape}")
        print("\nPHASE 4B E2E: FAIL")
        return 1
    num_classes = final_shape[0]
    if num_classes < 2:
        print(f"  FAIL: num_classes must be >= 2 for classification, got {num_classes}")
        print("\nPHASE 4B E2E: FAIL")
        return 1
    print(f"  PASS (num_classes={num_classes})")

    print("PyTorch model build:")
    set_seed()
    model = build_model(model_spec)
    print("  PASS")

    print("Synthetic dataset:")
    train_dataset, val_dataset = make_train_val_datasets(
        model_spec.input_shape, num_classes, seed=SEED, train_size=TRAIN_SIZE, val_size=VAL_SIZE
    )
    print(f"  train={len(train_dataset)} val={len(val_dataset)}")

    # DataLoader shuffle도 전용 Generator로 고정 -- 전역 RNG는 이후 example_input
    # 재생성을 위해 그대로 둔다 (run_phase1_e2e.py와 동일하게 set_seed()로 재현).
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=True,  # BatchNorm이 배치 크기 1을 못 받으므로 마지막 자투리 배치 제거
    )
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=False)

    print("Training:")
    training_result = run_training(model, train_loader, val_loader, TRAINING_CONFIG, device="cpu")
    history = training_result.history
    for epoch, (train_loss, val_loss, val_acc) in enumerate(
        zip(history.train_losses, history.val_losses, history.val_accuracies), start=1
    ):
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    loss_improved = history.train_losses[-1] < history.train_losses[0]
    print(
        f"  training loss decreased: {loss_improved} "
        f"({history.train_losses[0]:.4f} -> {history.train_losses[-1]:.4f})"
    )
    if not loss_improved:
        print("\nPHASE 4B E2E: FAIL")
        return 1

    print(f"Best epoch: {history.best_epoch}")
    print(f"Best validation loss: {history.best_val_loss:.4f}")

    history_path = ARTIFACTS_TRAINING / f"{model_spec.name}_history.json"
    save_training_history(history, history_path)
    print(f"History saved: {history_path}")

    # run_training()은 best_state_dict를 메모리로만 반환한다 (파일에 쓰지
    # 않음) -- 여기서 새 model에 로드한 뒤 기존 save_state_dict()로 저장한다.
    # 이후 export되는 건 마지막 epoch이 아니라 이 best 가중치다.
    best_model = build_model(model_spec)
    best_model.load_state_dict(training_result.best_state_dict)
    best_model = best_model.eval()

    state_dict_path = ARTIFACTS_TRAINING / f"{model_spec.name}_state_dict.pt"
    save_state_dict(best_model, state_dict_path)
    print(f"Best model saved: {state_dict_path}")

    print("Best model save/reload:")
    set_seed()  # 동일 입력 생성을 위해 seed 재설정 (run_phase1_e2e.py와 동일 패턴)
    example_input = torch.randn(1, *model_spec.input_shape, dtype=torch.float32)
    with torch.inference_mode():
        best_output = best_model(example_input)

    reloaded_model = build_model(model_spec).eval()
    load_state_dict(reloaded_model, state_dict_path)
    with torch.inference_mode():
        reloaded_output = reloaded_model(example_input)

    reload_ok = torch.allclose(best_output, reloaded_output)
    print("  PASS" if reload_ok else "  FAIL: reloaded output differs from best model output")
    if not reload_ok:
        print("\nPHASE 4B E2E: FAIL")
        return 1

    input_bin = ARTIFACTS_COMMON / f"{model_spec.name}_input.bin"
    input_meta = ARTIFACTS_COMMON / f"{model_spec.name}_input.json"
    save_tensor(example_input, input_bin, input_meta)

    print("TorchScript export:")
    model_pt = ARTIFACTS_TORCHSCRIPT / model_spec.name / "model.pt"
    metadata_path = ARTIFACTS_TORCHSCRIPT / model_spec.name / "metadata.json"
    TorchScriptExporter().export(
        reloaded_model,
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
        print("\nPHASE 4B E2E: FAIL")
        return 1

    for device in DEVICES:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        ref_model = build_model(model_spec).eval().to(device)
        load_state_dict(ref_model, state_dict_path, map_location=device)
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
            print("\nPHASE 4B E2E: FAIL")
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
    print(f"\nPHASE 4B E2E: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
