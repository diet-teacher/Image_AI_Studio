#!/usr/bin/env python
"""ImageFolder 학습 파이프라인 회귀 검증 E2E (Phase 4D에서 신설, Phase 4E/4G
에서 확장됐다가 Phase 4H에서 회귀 검증 전용으로 재구성됨).

**이 스크립트는 더 이상 실제 사용자 학습 CLI가 아니다** -- 그 역할은
`scripts/train_imagefolder.py`(production CLI)로 옮겨갔다. 이 스크립트는
`src/image_ai_studio/training/imagefolder_workflow.py`의
`run_imagefolder_training_workflow()`를 고정된 설정(CIFAR-10 ImageFolder
fixture, epochs=3+2, 고정 seed)으로 두 번 호출해 다음을 검증한다:

    ModelSpec 로드
        -> workflow(fresh, epochs=3, checkpoint 저장) 호출
        -> workflow(resume, epochs=2, 같은 checkpoint 경로) 호출
        -> loss가 실제로 줄었는지 확인 (Phase 4A/4B 이래의 회귀 게이트)
        -> class mapping 재검증
        -> TorchScript reload
        -> run_torchscript.exe(C++) CPU/CUDA 추론 -> Python/C++ parity 비교

`train_imagefolder.py`와 이 스크립트는 서로 import하지 않는다 -- 둘 다
`imagefolder_workflow.py`만 향해 의존한다
(docs/phase4h_production_training_cli_design.md §4/§12). C++ parity/
TorchScript export 검증은 이 스크립트에만 있고 production CLI에는 없다.

fresh(3 epoch) + resume(2 epoch) 조합은 Phase 4F/4G의 checkpoint/resume
계약(특히 "동일 경로를 --resume-from/--checkpoint-out으로 함께 써도
안전하다")을 실제 CIFAR-10 ImageFolder fixture로 매번 재현 가능하게
검증한다 -- 이전에는 이 조합을 사람이 터미널에서 직접 커맨드 두 번을
실행해서만 확인했다.

- model_definition/*, export/*, run_and_compare.py, C++ 러너 전부 변경 없음
- 네트워크 다운로드 없음 (사용자가 준비한 로컬 폴더만 읽음). CIFAR-10
  기반 fixture가 필요하면 scripts/prepare_cifar10_imagefolder_fixture.py를
  먼저 실행할 것 (자동 준비는 하지 않음)
- run_torchscript 빌드 필요 (scripts/build_torchscript.py), 없으면 자동 빌드
- CUDA 미가용 시 CPU 폴백 없이 SKIPPED 처리 (Phase 0/1과 동일 정책)

사용법:

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

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.parity.tensor_io import save_tensor
from image_ai_studio.tools.run_and_compare import find_runner_binary, run_case
from image_ai_studio.training.checkpoint import load_state_dict
from image_ai_studio.training.config import TrainingConfig, TrainingConfigError
from image_ai_studio.training.imagefolder_resume import metadata_path_for_checkpoint
from image_ai_studio.training.imagefolder_workflow import (
    ImageFolderWorkflowRequest,
    ImageFolderWorkflowResult,
    run_imagefolder_training_workflow,
)
from image_ai_studio.training.torchvision_dataset import load_class_mapping, make_imagefolder_datasets

SEED = 20260730
DEVICES = ["cpu", "cuda"]
FRESH_EPOCHS = 3
RESUME_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
ARTIFACT_NAME = "imagefolder_e2e"

MODEL_JSON = REPO_ROOT / "examples" / "models" / "phase4c_cifar10_model.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "artifacts" / "datasets" / "cifar10_imagefolder"

OUTPUT_DIR = REPO_ROOT / "artifacts" / "training" / ARTIFACT_NAME
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
ARTIFACTS_COMMON = REPO_ROOT / "artifacts" / "common"
ARTIFACTS_REFERENCE = REPO_ROOT / "artifacts" / "reference"
BUILD_DIR = REPO_ROOT / "build-torchscript"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", type=Path, default=MODEL_JSON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args(argv)


def _run_workflow_stage(
    stage_name: str, request: ImageFolderWorkflowRequest
) -> ImageFolderWorkflowResult | None:
    print(f"{stage_name}:")
    try:
        result = run_imagefolder_training_workflow(request)
    except (ModelValidationError, TrainingConfigError, ValueError, OSError) as exc:
        print(f"  FAIL: {exc}")
        return None
    for epoch, (train_loss, val_loss, val_acc) in enumerate(
        zip(result.history.train_losses, result.history.val_losses, result.history.val_accuracies), start=1
    ):
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    print(f"  stopped_early={result.history.stopped_early}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_json_path: Path = args.model_json

    print("ImageFolder Training E2E")
    print(f"Model JSON: {model_json_path}")
    print(f"Dataset root: {args.dataset_root}")

    print("ModelSpec:")
    try:
        model_spec = load_model_spec(model_json_path)
    except ModelValidationError as exc:
        print(f"  FAIL: {exc}")
        print("\nIMAGEFOLDER TRAINING E2E: FAIL")
        return 1
    print(f"  PASS ({len(model_spec.layers)} layers)")

    fresh_config = TrainingConfig(epochs=FRESH_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    resume_config = TrainingConfig(epochs=RESUME_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    # Phase 4J의 출력 경로 재사용 정책(docs/phase4j_epoch_checkpoint_design.md
    # §6-5)은 fresh 학습이 기존 checkpoint_out(+metadata sidecar)을 재사용하는
    # 것을 거부한다 -- 이 E2E는 CHECKPOINT_PATH를 고정 경로로 매번 재사용하므로,
    # 이 스크립트를 여러 번 실행할 수 있도록 stage 1(fresh) 직전에만 이 두
    # 파일을 지운다(다른 산출물은 건드리지 않음).
    CHECKPOINT_PATH.unlink(missing_ok=True)
    metadata_path_for_checkpoint(CHECKPOINT_PATH).unlink(missing_ok=True)

    fresh_result = _run_workflow_stage(
        f"Fresh training ({FRESH_EPOCHS} epochs) + checkpoint save",
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=args.dataset_root,
            training_config=fresh_config,
            output_dir=OUTPUT_DIR,
            checkpoint_out=CHECKPOINT_PATH,
            export_torchscript=True,
            seed=SEED,
        ),
    )
    if fresh_result is None:
        print("\nIMAGEFOLDER TRAINING E2E: FAIL")
        return 1

    result = _run_workflow_stage(
        f"Resume ({RESUME_EPOCHS} additional epochs) from checkpoint",
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=args.dataset_root,
            training_config=resume_config,
            output_dir=OUTPUT_DIR,
            resume_from=CHECKPOINT_PATH,
            checkpoint_out=CHECKPOINT_PATH,
            export_torchscript=True,
            seed=SEED,
        ),
    )
    if result is None:
        print("\nIMAGEFOLDER TRAINING E2E: FAIL")
        return 1

    history = result.history
    loss_improved = history.train_losses[-1] < history.train_losses[0]
    print(
        f"  training loss decreased: {loss_improved} "
        f"({history.train_losses[0]:.4f} -> {history.train_losses[-1]:.4f})"
    )
    if not loss_improved:
        print("\nIMAGEFOLDER TRAINING E2E: FAIL")
        return 1

    print(f"Best epoch: {history.best_epoch}")
    print(f"Best validation loss: {history.best_val_loss:.4f}")
    print(f"Test: loss={result.test_loss:.4f} accuracy={result.test_accuracy:.4f}")

    # -- 이하는 회귀 검증 전용 자체 재검증 (production workflow에서는 제거됨) --
    # best model save/reload를 "원본 in-memory model vs 재로드 model" 형태로
    # 비교하는 검증은 여기 두지 않는다 -- workflow는 Result에 살아있는
    # model을 전혀 반환하지 않으므로(설계 유지), 그런 비교를 하려면 같은
    # 파일을 두 번 읽어 서로 비교하는 것 밖에 할 수 없는데 그건 항상 같은
    # 결과만 내는 무의미한 검증이다. state_dict 저장/재로드 자체의 수치적
    # 정확성(원본 vs 재로드)은 이미
    # tests/training/test_imagefolder_workflow.py(§14-1, best model 평가/
    # TorchScript export 비교)와 tests/training/test_checkpoint.py가
    # 단위 테스트로 커버한다.

    print("Class mapping consistency check:")
    splits = make_imagefolder_datasets(model_spec.input_shape, root=args.dataset_root)
    reloaded_mapping = load_class_mapping(result.class_mapping_path)
    class_mapping_ok = (
        reloaded_mapping["classes"] == splits.classes and reloaded_mapping["class_to_idx"] == splits.class_to_idx
    )
    print("  PASS" if class_mapping_ok else "  FAIL: reloaded class mapping differs from dataset")
    if not class_mapping_ok:
        print("\nIMAGEFOLDER TRAINING E2E: FAIL")
        return 1

    if result.torchscript_model_path is None:
        print("TorchScript export: FAIL (workflow did not produce a TorchScript artifact)")
        print("\nIMAGEFOLDER TRAINING E2E: FAIL")
        return 1
    print(f"TorchScript export: PASS ({result.torchscript_model_path})")

    # C++ parity 비교에 쓸 입력 -- 재현 가능하도록 시드 고정.
    torch.manual_seed(SEED)
    example_input = torch.randn(1, *model_spec.input_shape, dtype=torch.float32)

    input_bin = ARTIFACTS_COMMON / f"{ARTIFACT_NAME}_input.bin"
    input_meta = ARTIFACTS_COMMON / f"{ARTIFACT_NAME}_input.json"
    save_tensor(example_input, input_bin, input_meta)

    for device in DEVICES:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        ref_model = build_model(model_spec).eval().to(device)
        load_state_dict(ref_model, result.best_model_state_dict_path, map_location=device)
        with torch.inference_mode():
            ref_output = ref_model(example_input.to(device))
        save_tensor(
            ref_output,
            ARTIFACTS_REFERENCE / f"{ARTIFACT_NAME}_{device}.bin",
            ARTIFACTS_REFERENCE / f"{ARTIFACT_NAME}_{device}.json",
            layout="NC",
        )

    print("C++ TorchScript runner:")
    runner_binary = find_runner_binary(BUILD_DIR, "torchscript_runner", "run_torchscript")
    if not runner_binary.exists():
        print("  runner binary not found, building via scripts/build_torchscript.py ...")
        build = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_torchscript.py")])
        if build.returncode != 0:
            print("  FAIL: scripts/build_torchscript.py failed")
            print("\nIMAGEFOLDER TRAINING E2E: FAIL")
            return 1
        runner_binary = find_runner_binary(BUILD_DIR, "torchscript_runner", "run_torchscript")

    device_status: dict[str, str] = {}
    for device in DEVICES:
        case_result = run_case(
            runner_binary=runner_binary,
            runner_name="torchscript",
            model_name=ARTIFACT_NAME,
            model_artifact=result.torchscript_model_path,
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
    print(f"\nIMAGEFOLDER TRAINING E2E: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
