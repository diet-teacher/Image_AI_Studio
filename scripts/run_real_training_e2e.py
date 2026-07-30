#!/usr/bin/env python
"""Phase 4C 실제 이미지 학습 엔드투엔드 파이프라인:

    Model JSON -> ModelSpec -> build_model()
        -> torchvision CIFAR-10 (공식 train split -> Train/Validation 결정론적 분리)
        -> 실제 학습 (Adam + CrossEntropyLoss, Phase 4A와 동일)
        -> best epoch(최소 validation loss) 추적 (Phase 4B와 동일)
        -> training history JSON 저장
        -> best epoch의 state_dict 저장
        -> 공식 CIFAR-10 test split으로 best model 최종 평가 (Phase 4C 신규)
        -> TorchScriptExporter (Phase 0 재사용)
        -> run_torchscript.exe (Phase 0 재사용) -> C++ 추론
        -> Python/C++ parity 비교

scripts/run_training_e2e.py(synthetic, Phase 4A/4B 회귀 검증용)는 그대로
유지한다 -- 이 스크립트는 역할이 다른 별도 경로다(synthetic dataset이
아니라 실제 이미지). train_one_epoch/evaluate/run_training/TrainingHistory
/TrainingResult/state_dict save-load/TorchScriptExporter/C++ 러너는 전부
Phase 4A/4B 코드 그대로 재사용하고, 이 스크립트가 새로 하는 일은 딱
"실제 이미지 dataset을 만들어서 넘겨주는 것"과 "학습 종료 후 test
split으로 한 번 평가하는 것" 두 가지뿐이다.

**Test split은 best epoch 선택이나 학습 중 어떤 판단에도 쓰이지 않는다.**
best_state_dict가 확정된 뒤, 그 model 하나에 대해 딱 한 번만 평가한다.

- model_definition/*, export/*, run_and_compare.py, C++ 러너, training/loop.py
  전부 변경 없음
- CIFAR-10은 처음 실행 시 --data-root에 다운로드된다 (네트워크 필요).
  pytest는 이 스크립트를 호출하지 않으므로 오프라인 정책과 무관하다.
- run_torchscript 빌드 필요 (scripts/build_torchscript.py), 없으면 자동 빌드
- CUDA 미가용 시 CPU 폴백 없이 SKIPPED 처리 (Phase 0/1과 동일 정책)

사용법:

    python scripts/run_real_training_e2e.py
    python scripts/run_real_training_e2e.py --train-limit 1024 --val-limit 256 --test-limit 512
    python scripts/run_real_training_e2e.py --model-json examples/models/phase4c_cifar10_model.json

기본값은 빠른 E2E 검증을 위해 작은 subset을 사용한다 (--train-limit 등
0 이하 또는 생략 시 전체 split 사용 가능 -- 아래 parse_args 참고).
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
from image_ai_studio.training.history import save_training_history
from image_ai_studio.training.loop import evaluate, run_training
from image_ai_studio.training.torchvision_dataset import (
    NUM_CLASSES,
    limit_dataset,
    make_cifar10_test_dataset,
    make_cifar10_train_val_datasets,
)

SEED = 20260730
DEVICES = ["cpu", "cuda"]
TRAINING_CONFIG = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-3)

# 빠른 기본 E2E 실행을 위한 subset 크기. 전체 CIFAR-10(45,000/5,000/10,000)을
# 쓰려면 --train-limit 0 --val-limit 0 --test-limit 0 (제한 없음).
DEFAULT_TRAIN_LIMIT = 256
DEFAULT_VAL_LIMIT = 64
DEFAULT_TEST_LIMIT = 128

MODEL_JSON = REPO_ROOT / "examples" / "models" / "phase4c_cifar10_model.json"
DATA_ROOT = REPO_ROOT / "artifacts" / "datasets" / "cifar10"
ARTIFACTS_COMMON = REPO_ROOT / "artifacts" / "common"
ARTIFACTS_TRAINING = REPO_ROOT / "artifacts" / "training"
ARTIFACTS_TORCHSCRIPT = REPO_ROOT / "artifacts" / "torchscript"
ARTIFACTS_REFERENCE = REPO_ROOT / "artifacts" / "reference"
BUILD_DIR = REPO_ROOT / "build-torchscript"


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limit_arg(value: str) -> int | None:
    """CLI에서 0 이하는 "제한 없음"(전체 split 사용)으로 해석."""
    parsed = int(value)
    return None if parsed <= 0 else parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", type=Path, default=MODEL_JSON)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-limit", type=_limit_arg, default=DEFAULT_TRAIN_LIMIT)
    parser.add_argument("--val-limit", type=_limit_arg, default=DEFAULT_VAL_LIMIT)
    parser.add_argument("--test-limit", type=_limit_arg, default=DEFAULT_TEST_LIMIT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_json_path: Path = args.model_json

    print("Phase 4C Real-Image Training E2E")
    print(f"Model JSON: {model_json_path}")

    print("ModelSpec:")
    try:
        model_spec = load_model_spec(model_json_path)
        shape_trace = validate_model_spec(model_spec)
        final_shape = shape_trace[-1].output_shape
        print(f"  PASS ({len(model_spec.layers)} layers, final shape {final_shape})")
    except ModelValidationError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4C E2E: FAIL")
        return 1

    print("Classification output check:")
    if len(final_shape) != 1 or final_shape[0] != NUM_CLASSES:
        print(f"  FAIL: final output shape must be ({NUM_CLASSES},) for CIFAR-10, got {final_shape}")
        print("\nPHASE 4C E2E: FAIL")
        return 1
    print(f"  PASS (num_classes={NUM_CLASSES})")

    print("PyTorch model build:")
    set_seed()
    model = build_model(model_spec)
    print("  PASS")

    print("CIFAR-10 dataset:")
    try:
        train_dataset, val_dataset = make_cifar10_train_val_datasets(
            model_spec.input_shape, root=args.data_root, seed=SEED
        )
        test_dataset = make_cifar10_test_dataset(model_spec.input_shape, root=args.data_root)
    except ValueError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4C E2E: FAIL")
        return 1

    train_dataset = limit_dataset(train_dataset, args.train_limit)
    val_dataset = limit_dataset(val_dataset, args.val_limit)
    test_dataset = limit_dataset(test_dataset, args.test_limit)
    print(f"  train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    # DataLoader shuffle도 전용 Generator로 고정 -- 전역 RNG는 이후 example_input
    # 재생성을 위해 그대로 둔다 (run_training_e2e.py와 동일 패턴).
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=True,
        generator=loader_generator,
        # 작은 마지막 batch가 training-time BatchNorm 동작에 영향을 주는 것을 피하고
        # E2E의 train batch 구성을 일정하게 유지
        drop_last=True,
        num_workers=0,  # Windows에서 단순하고 안전한 값 (멀티프로세스 워커 없음)
    )
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=False, num_workers=0)

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
        print("\nPHASE 4C E2E: FAIL")
        return 1

    print(f"Best epoch: {history.best_epoch}")
    print(f"Best validation loss: {history.best_val_loss:.4f}")

    history_path = ARTIFACTS_TRAINING / f"{model_spec.name}_history.json"
    save_training_history(history, history_path)
    print(f"History saved: {history_path}")

    # run_training()은 best_state_dict를 메모리로만 반환한다 (파일에 쓰지
    # 않음) -- 여기서 새 model에 로드한 뒤 기존 save_state_dict()로 저장한다.
    best_model = build_model(model_spec)
    best_model.load_state_dict(training_result.best_state_dict)
    best_model = best_model.eval()

    state_dict_path = ARTIFACTS_TRAINING / f"{model_spec.name}_state_dict.pt"
    save_state_dict(best_model, state_dict_path)
    print(f"Best model saved: {state_dict_path}")

    # Test는 best epoch 선택에 전혀 관여하지 않는다 -- best_model이 이미
    # 확정된 뒤, 공식 test split에 대해 딱 한 번만 evaluate()를 호출한다.
    print("Test evaluation (best model, official CIFAR-10 test split):")
    test_loss, test_accuracy = evaluate(best_model, test_loader, device="cpu")
    print(f"  test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}")

    test_result_path = ARTIFACTS_TRAINING / f"{model_spec.name}_test_result.json"
    test_result_path.parent.mkdir(parents=True, exist_ok=True)
    test_result_path.write_text(
        json.dumps({"test_loss": test_loss, "test_accuracy": test_accuracy}, indent=2), encoding="utf-8"
    )
    print(f"Test result saved: {test_result_path}")

    print("Best model save/reload:")
    set_seed()  # 동일 입력 생성을 위해 seed 재설정 (run_training_e2e.py와 동일 패턴)
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
        print("\nPHASE 4C E2E: FAIL")
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
        print("\nPHASE 4C E2E: FAIL")
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
            print("\nPHASE 4C E2E: FAIL")
            return 1
        runner_binary = find_runner_binary(BUILD_DIR, "torchscript_runner", "run_torchscript")

    device_status: dict[str, str] = {}
    for device in DEVICES:
        case_result = run_case(
            runner_binary=runner_binary,
            runner_name="torchscript",
            model_name=model_spec.name,
            model_artifact=model_pt,
            device=device,
            input_bin=input_bin,
            input_meta=input_meta,
        )
        device_status[device] = case_result["status"]
        print(f"  {device.upper()}: {case_result['status']}")

    ran_at_least_one = any(status == "PASS" for status in device_status.values())
    no_failures = all(status in ("PASS", "SKIPPED") for status in device_status.values())
    parity_ok = ran_at_least_one and no_failures
    print(f"Parity: {'PASS' if parity_ok else 'FAIL'}")

    overall_ok = parity_ok
    print(f"\nPHASE 4C E2E: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
