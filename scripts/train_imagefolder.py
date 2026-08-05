#!/usr/bin/env python
"""실제 사용자용 ImageFolder 학습 CLI (Phase 4H).

`scripts/run_imagefolder_training_e2e.py`(회귀 검증 전용, CIFAR-10 fixture
기본값 + C++ CPU/CUDA parity 강제 실행)와 책임을 분리한다 -- 이 스크립트는
"학습 본질" 로직만 다루고, 그 로직 자체는
`src/image_ai_studio/training/imagefolder_workflow.py`의
`run_imagefolder_training_workflow()`에 있다. 이 스크립트와 E2E 스크립트는
서로 import하지 않고 둘 다 워크플로우 모듈만 향해 의존한다
(docs/phase4h_production_training_cli_design.md).

이 CLI는 C++ parity를 실행하지 않는다 -- 빌드된 러너 바이너리나 CUDA
가용성에 의존하지 않고 항상 순수 Python만으로 동작한다. TorchScript
export는 기본으로 포함되며 `--no-export-torchscript`로 끌 수 있다.

지원하는 dataset 폴더 구조 (자동 split 없음, 이미 분리되어 있어야 함)::

    dataset_root/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

사용법::

    # 새로 학습
    python scripts/train_imagefolder.py \\
        --model-json my_model.json --dataset-root path/to/dataset \\
        --epochs 20 --batch-size 32 --learning-rate 5e-4 \\
        --output-dir artifacts/my_run --checkpoint-out artifacts/my_run/checkpoint.pt

    # 이어서 학습 (--epochs는 "총 epoch"가 아니라 "이번에 추가로 실행할
    # epoch 수"다 -- Phase 4F/4G 계약을 그대로 따른다)
    python scripts/train_imagefolder.py \\
        --model-json my_model.json --dataset-root path/to/dataset \\
        --epochs 10 --batch-size 32 --learning-rate 5e-4 \\
        --output-dir artifacts/my_run \\
        --resume-from artifacts/my_run/checkpoint.pt --checkpoint-out artifacts/my_run/checkpoint.pt

`--seed`는 resume 시 사실상 무시된다 -- model은 곧바로 checkpoint의
model_state_dict로 덮어써지고, DataLoader shuffle 순서와 CPU RNG는
checkpoint에 저장된 상태로 복원되기 때문이다(--seed 값과 무관).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from image_ai_studio.model_definition.errors import ModelValidationError
from image_ai_studio.training.config import TrainingConfig, TrainingConfigError
from image_ai_studio.training.imagefolder_workflow import (
    SEED,
    ImageFolderWorkflowRequest,
    run_imagefolder_training_workflow,
)
from image_ai_studio.training.loop import TrainingProgress

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-json", type=Path, required=True, help="ModelSpec JSON 파일 경로")
    parser.add_argument("--dataset-root", type=Path, required=True, help="train/val/test로 이미 분리된 ImageFolder 루트")
    parser.add_argument("--output-dir", type=Path, required=True, help="best model/history/class mapping 등을 저장할 디렉터리")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="이번 실행에서 실행할 epoch 수 (resume 시에도 '추가' epoch 수)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)

    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer=sgd일 때만 사용")
    parser.add_argument(
        "--lr-scheduler",
        choices=["plateau"],
        default=None,
        help="생략하면 scheduler 없음 (기본값)",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-patience", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="생략하면 early stopping 없음 (기본값)",
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Phase 4F checkpoint 파일 경로 -- 지정하면 그 지점부터 이어서 학습한다",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help=(
            "checkpoint를 저장할 경로(학습 종료 시 항상 저장됨). metadata는 항상 같은 이름 뒤에 "
            "'.meta.json'을 붙인 경로로 자동 저장된다 (별도 플래그 없음). fresh 학습이나 "
            "--resume-from과 다른 경로로의 resume은 이 경로(와 metadata sidecar)가 완전히 "
            "비어있어야 한다 -- 기존 checkpoint를 이어서 갱신하려면 --resume-from과 같은 "
            "경로를 지정할 것"
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help=(
            "global epoch이 이 값의 배수가 될 때마다 --checkpoint-out을 자동으로 갱신한다 "
            "(1 이상의 정수, --checkpoint-out 필수). 생략하면 기존과 동일하게 학습 종료 시에만 저장된다"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="model/DataLoader 초기화 seed (resume 시에는 사실상 무시됨 -- 위 설명 참고)",
    )
    parser.add_argument(
        "--export-torchscript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="학습 후 TorchScript export 여부 (기본 포함, --no-export-torchscript로 끌 수 있음)",
    )
    return parser.parse_args(argv)


def _print_progress(progress: TrainingProgress) -> None:
    """Phase 4I: epoch이 완료될 때마다(사후 일괄 출력이 아니라) 실시간으로
    한 줄씩 찍는다. progress.global_epoch을 쓴다 -- run_epoch(이번 호출
    기준 1부터)을 쓰면 resume할 때마다 번호가 1로 되돌아가 버려 절대
    번호(체크포인트 이전 epoch까지 포함한 누적 번호)와 어긋난다. resume
    실행에서는 이 함수가 새로 완료된 epoch에만 호출되므로, 이전에 이미
    완료된 epoch은 다시 찍히지 않는다(과거 CLI처럼 history 전체를 사후에
    재출력하지 않음 -- 의도된 동작 변경,
    docs/phase4i_training_progress_and_stop_design.md §11/§17 참고)."""
    print(
        f"  epoch {progress.global_epoch}: train_loss={progress.train_loss:.4f} "
        f"val_loss={progress.val_loss:.4f} val_acc={progress.val_accuracy:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("ImageFolder Training")
    print(f"Model JSON: {args.model_json}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Resume from: {args.resume_from if args.resume_from is not None else '(none, fresh training)'}")
    print(f"Checkpoint out: {args.checkpoint_out if args.checkpoint_out is not None else '(none, not saved)'}")

    try:
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            momentum=args.momentum,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            early_stopping_patience=args.early_stopping_patience,
        )
        request = ImageFolderWorkflowRequest(
            model_json_path=args.model_json,
            dataset_root=args.dataset_root,
            training_config=training_config,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            checkpoint_out=args.checkpoint_out,
            export_torchscript=args.export_torchscript,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
        )
        result = run_imagefolder_training_workflow(request, progress_callback=_print_progress)
    except (ModelValidationError, TrainingConfigError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    history = result.history
    print(f"  stopped_early={history.stopped_early}")
    print(f"  stopped_by_user={history.stopped_by_user}")
    print(f"Best epoch: {history.best_epoch} (val_loss={history.best_val_loss:.4f})")
    print(f"Test: loss={result.test_loss:.4f} accuracy={result.test_accuracy:.4f}")

    print("Artifacts:")
    print(f"  best model:      {result.best_model_state_dict_path}")
    print(f"  training history:{result.training_history_path}")
    print(f"  class mapping:   {result.class_mapping_path}")
    print(f"  test result:     {result.test_result_path}")
    if result.checkpoint_path is not None:
        print(f"  checkpoint:      {result.checkpoint_path}")
        print(f"  checkpoint meta: {result.checkpoint_metadata_path}")
        if history.stopped_early:
            print(
                "  note: stopped_early=True -- weights/history can still be loaded, but this "
                "checkpoint cannot be used to resume training further."
            )
    if result.torchscript_model_path is not None:
        print(f"  TorchScript:     {result.torchscript_model_path}")
        print(f"  TorchScript meta:{result.torchscript_metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
