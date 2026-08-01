#!/usr/bin/env python
"""Phase 4D 사용자 ImageFolder 데이터셋 학습 엔드투엔드 파이프라인:

    Model JSON -> ModelSpec -> build_model()
        -> 사용자가 미리 분리해 둔 train/val/test ImageFolder 폴더 로딩
        -> class_to_idx 일치 검증 (train/val/test)
        -> dataset 클래스 수 vs ModelSpec 최종 출력 shape 검증
        -> 실제 학습 (Adam + CrossEntropyLoss, Phase 4A와 동일)
        -> best epoch(최소 validation loss) 추적 (Phase 4B와 동일)
        -> training history JSON 저장
        -> best epoch의 state_dict 저장
        -> class mapping JSON 저장/재로드 확인 (Phase 4D 신규)
        -> best model의 test split 최종 평가 (Phase 4C와 동일 패턴)
        -> TorchScriptExporter (Phase 0 재사용)
        -> run_torchscript.exe (Phase 0 재사용) -> C++ 추론
        -> Python/C++ parity 비교

Phase 4C의 scripts/run_real_training_e2e.py(CIFAR-10 직접 로딩)는 그대로
유지한다 -- 이 스크립트는 "torchvision에 내장된 특정 dataset"이 아니라
"사용자가 준비한 일반 이미지 폴더"를 학습하는 별도 경로다.
train_one_epoch/evaluate/run_training/TrainingHistory/TrainingResult/
state_dict save-load/build_transform/TorchScriptExporter/C++ 러너는 전부
Phase 4A~4C 코드 그대로 재사용한다. 이 스크립트가 새로 하는 일은
"사용자 ImageFolder 폴더를 검증해서 읽어오는 것"과 "class mapping을
저장하는 것" 두 가지뿐이다.

**Test split은 best epoch 선택이나 학습 중 어떤 판단에도 쓰이지 않는다.**
best_state_dict가 확정된 뒤, 그 model 하나에 대해 딱 한 번만 평가한다
(test_loader 자체는 다른 DataLoader와 함께 미리 만들어지지만
run_training()에는 전달되지 않는다 -- Phase 4C와 동일한 정책).

지원하는 dataset 폴더 구조 (자동 split 없음, 이미 분리되어 있어야 함)::

    dataset_root/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

- model_definition/*, export/*, run_and_compare.py, C++ 러너, training/loop.py
  전부 변경 없음
- 네트워크 다운로드 없음 (사용자가 이미 준비한 로컬 폴더만 읽음).
  E2E 검증용 CIFAR-10 기반 fixture가 필요하면
  scripts/prepare_cifar10_imagefolder_fixture.py를 먼저 실행할 것.
- run_torchscript 빌드 필요 (scripts/build_torchscript.py), 없으면 자동 빌드
- CUDA 미가용 시 CPU 폴백 없이 SKIPPED 처리 (Phase 0/1과 동일 정책)

사용법 (Windows cmd/PowerShell, bash 모두 동일한 인자로 동작):

    python scripts/run_imagefolder_training_e2e.py
    python scripts/run_imagefolder_training_e2e.py --dataset-root path/to/dataset --model-json examples/models/phase4c_cifar10_model.json

기본값은 scripts/prepare_cifar10_imagefolder_fixture.py가 만드는
artifacts/datasets/cifar10_imagefolder 픽스처를 가리킨다.
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
from image_ai_studio.training.config import TrainingConfig, TrainingConfigError
from image_ai_studio.training.history import save_training_history
from image_ai_studio.training.loop import evaluate, run_training
from image_ai_studio.training.torchvision_dataset import (
    load_class_mapping,
    make_imagefolder_datasets,
    require_matching_num_classes,
    save_class_mapping,
)

SEED = 20260730
DEVICES = ["cpu", "cuda"]
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3

MODEL_JSON = REPO_ROOT / "examples" / "models" / "phase4c_cifar10_model.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "artifacts" / "datasets" / "cifar10_imagefolder"

# CIFAR-10 직접 로딩 E2E(run_real_training_e2e.py)와 artifact 경로가
# 겹치지 않도록, 파일명에는 model_spec.name 뒤에 이 접미사를 붙인다
# (같은 phase4c_cifar10_model.json을 재사용하더라도 두 E2E의 결과물이
# 서로 덮어쓰지 않아야 각각 독립적으로 회귀 검증할 수 있다).
ARTIFACT_SUFFIX = "_imagefolder"

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
    parser.add_argument("--model-json", type=Path, default=MODEL_JSON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)

    # Phase 4E: TrainingConfig 확장. 전부 생략하면 Phase 4A~4D와 동일하게
    # Adam / scheduler 없음 / early stopping 없음으로 동작한다.
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer=sgd일 때만 사용")
    parser.add_argument(
        "--lr-scheduler",
        choices=["plateau"],
        default=None,
        help="생략하면 scheduler 없음 (기본값, 기존 동작과 동일)",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-patience", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="생략하면 early stopping 없음 (기본값, 기존 동작과 동일)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_json_path: Path = args.model_json

    print("Phase 4D ImageFolder Training E2E")
    print(f"Model JSON: {model_json_path}")
    print(f"Dataset root: {args.dataset_root}")

    print("Training config:")
    try:
        training_config = TrainingConfig(
            epochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
            learning_rate=DEFAULT_LEARNING_RATE,
            optimizer=args.optimizer,
            momentum=args.momentum,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            early_stopping_patience=args.early_stopping_patience,
        )
    except TrainingConfigError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4D E2E: FAIL")
        return 1
    print(
        f"  optimizer={training_config.optimizer}"
        + (f" (momentum={training_config.momentum})" if training_config.optimizer == "sgd" else "")
    )
    print(
        "  lr_scheduler="
        + (
            f"{training_config.lr_scheduler} (factor={training_config.lr_scheduler_factor}, "
            f"patience={training_config.lr_scheduler_patience})"
            if training_config.lr_scheduler is not None
            else "None"
        )
    )
    print(f"  early_stopping_patience={training_config.early_stopping_patience}")

    print("ModelSpec:")
    try:
        model_spec = load_model_spec(model_json_path)
        shape_trace = validate_model_spec(model_spec)
        final_shape = shape_trace[-1].output_shape
        print(f"  PASS ({len(model_spec.layers)} layers, final shape {final_shape})")
    except ModelValidationError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4D E2E: FAIL")
        return 1

    artifact_name = f"{model_spec.name}{ARTIFACT_SUFFIX}"

    print("ImageFolder dataset (train/val/test):")
    try:
        splits = make_imagefolder_datasets(model_spec.input_shape, root=args.dataset_root)
    except ValueError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4D E2E: FAIL")
        return 1
    train_dataset, val_dataset, test_dataset = splits.train, splits.val, splits.test
    print(f"  classes={splits.classes}")
    print(f"  train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    print("Dataset class count vs ModelSpec output check:")
    try:
        require_matching_num_classes(len(splits.classes), final_shape)
    except ValueError as exc:
        print(f"  FAIL: {exc}")
        print("\nPHASE 4D E2E: FAIL")
        return 1
    print(f"  PASS (num_classes={len(splits.classes)})")

    print("PyTorch model build:")
    set_seed()
    model = build_model(model_spec)
    print("  PASS")

    # DataLoader shuffle도 전용 Generator로 고정 -- 전역 RNG는 이후 example_input
    # 재생성을 위해 그대로 둔다 (run_real_training_e2e.py와 동일 패턴).
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=loader_generator,
        # 작은 마지막 batch가 training-time BatchNorm 동작에 영향을 주는 것을 피하고
        # E2E의 train batch 구성을 일정하게 유지
        drop_last=True,
        num_workers=0,  # Windows에서 단순하고 안전한 값 (멀티프로세스 워커 없음)
    )
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=0)
    # test_loader는 다른 DataLoader와 함께 여기서 만들어지지만, run_training()에는
    # 전달되지 않는다 -- best_state_dict 확정 이후 evaluate()에서만 사용한다.
    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=0)

    print("Training:")
    training_result = run_training(model, train_loader, val_loader, training_config, device="cpu")
    history = training_result.history
    for epoch, (train_loss, val_loss, val_acc) in enumerate(
        zip(history.train_losses, history.val_losses, history.val_accuracies), start=1
    ):
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    print(f"  stopped_early={history.stopped_early}")

    loss_improved = history.train_losses[-1] < history.train_losses[0]
    print(
        f"  training loss decreased: {loss_improved} "
        f"({history.train_losses[0]:.4f} -> {history.train_losses[-1]:.4f})"
    )
    if not loss_improved:
        print("\nPHASE 4D E2E: FAIL")
        return 1

    print(f"Best epoch: {history.best_epoch}")
    print(f"Best validation loss: {history.best_val_loss:.4f}")

    history_path = ARTIFACTS_TRAINING / f"{artifact_name}_history.json"
    save_training_history(history, history_path)
    print(f"History saved: {history_path}")

    # run_training()은 best_state_dict를 메모리로만 반환한다 (파일에 쓰지
    # 않음) -- 여기서 새 model에 로드한 뒤 기존 save_state_dict()로 저장한다.
    best_model = build_model(model_spec)
    best_model.load_state_dict(training_result.best_state_dict)
    best_model = best_model.eval()

    state_dict_path = ARTIFACTS_TRAINING / f"{artifact_name}_state_dict.pt"
    save_state_dict(best_model, state_dict_path)
    print(f"Best model saved: {state_dict_path}")

    print("Class mapping save/reload:")
    class_mapping_path = ARTIFACTS_TRAINING / f"{artifact_name}_classes.json"
    save_class_mapping(splits.classes, splits.class_to_idx, class_mapping_path)
    reloaded_mapping = load_class_mapping(class_mapping_path)
    class_mapping_ok = reloaded_mapping["classes"] == splits.classes and (
        reloaded_mapping["class_to_idx"] == splits.class_to_idx
    )
    print("  PASS" if class_mapping_ok else "  FAIL: reloaded class mapping differs from original")
    print(f"  saved to {class_mapping_path}")
    if not class_mapping_ok:
        print("\nPHASE 4D E2E: FAIL")
        return 1

    # Test는 best epoch 선택에 전혀 관여하지 않는다 -- best_model이 이미
    # 확정된 뒤, test split에 대해 딱 한 번만 evaluate()를 호출한다.
    print("Test evaluation (best model, user-provided test split):")
    test_loss, test_accuracy = evaluate(best_model, test_loader, device="cpu")
    print(f"  test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}")

    test_result_path = ARTIFACTS_TRAINING / f"{artifact_name}_test_result.json"
    test_result_path.parent.mkdir(parents=True, exist_ok=True)
    test_result_path.write_text(
        json.dumps({"test_loss": test_loss, "test_accuracy": test_accuracy}, indent=2), encoding="utf-8"
    )
    print(f"Test result saved: {test_result_path}")

    print("Best model save/reload:")
    set_seed()  # 동일 입력 생성을 위해 seed 재설정 (Phase 4A~4C E2E와 동일 패턴)
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
        print("\nPHASE 4D E2E: FAIL")
        return 1

    input_bin = ARTIFACTS_COMMON / f"{artifact_name}_input.bin"
    input_meta = ARTIFACTS_COMMON / f"{artifact_name}_input.json"
    save_tensor(example_input, input_bin, input_meta)

    print("TorchScript export:")
    model_pt = ARTIFACTS_TORCHSCRIPT / artifact_name / "model.pt"
    metadata_path = ARTIFACTS_TORCHSCRIPT / artifact_name / "metadata.json"
    TorchScriptExporter().export(
        reloaded_model,
        example_input,
        model_pt,
        metadata_path,
        model_name=artifact_name,
        state_dict_path=state_dict_path,
    )
    export_status = json.loads(metadata_path.read_text())["status"]
    print(f"  {export_status}")
    if export_status != "PASS":
        print(f"  see {metadata_path} for error_log")
        print("\nPHASE 4D E2E: FAIL")
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
            ARTIFACTS_REFERENCE / f"{artifact_name}_{device}.bin",
            ARTIFACTS_REFERENCE / f"{artifact_name}_{device}.json",
            layout="NC",
        )

    print("C++ TorchScript runner:")
    runner_binary = find_runner_binary(BUILD_DIR, "torchscript_runner", "run_torchscript")
    if not runner_binary.exists():
        print("  runner binary not found, building via scripts/build_torchscript.py ...")
        build = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_torchscript.py")])
        if build.returncode != 0:
            print("  FAIL: scripts/build_torchscript.py failed")
            print("\nPHASE 4D E2E: FAIL")
            return 1
        runner_binary = find_runner_binary(BUILD_DIR, "torchscript_runner", "run_torchscript")

    device_status: dict[str, str] = {}
    for device in DEVICES:
        case_result = run_case(
            runner_binary=runner_binary,
            runner_name="torchscript",
            model_name=artifact_name,
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
    print(f"\nPHASE 4D E2E: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
