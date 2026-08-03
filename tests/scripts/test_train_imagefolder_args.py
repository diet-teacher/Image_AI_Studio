"""scripts/train_imagefolder.py의 parse_args() 테스트 (Phase 4H).

Phase 4G의 test_run_imagefolder_training_e2e_args.py와 같은 패턴 --
scripts/를 sys.path에 추가해 parse_args()를 직접 호출한다. 학습 자체는
실행하지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_imagefolder as cli  # noqa: E402

REQUIRED_ARGS = ["--model-json", "m.json", "--dataset-root", "d", "--output-dir", "o"]


def _without_flag(args: list[str], flag: str) -> list[str]:
    """args에서 flag와 그 바로 다음 값을 제거한 새 리스트를 반환."""
    result: list[str] = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            skip_next = True
            continue
        result.append(item)
    return result


@pytest.mark.parametrize("missing_flag", ["--model-json", "--dataset-root", "--output-dir"])
def test_parse_args_requires_model_json_dataset_root_output_dir(missing_flag: str) -> None:
    args = _without_flag(REQUIRED_ARGS, missing_flag)
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(args)
    assert exc_info.value.code != 0


def test_parse_args_accepts_all_required_args() -> None:
    args = cli.parse_args(REQUIRED_ARGS)
    assert args.model_json == Path("m.json")
    assert args.dataset_root == Path("d")
    assert args.output_dir == Path("o")


def test_parse_args_defaults() -> None:
    args = cli.parse_args(REQUIRED_ARGS)

    assert args.epochs == cli.DEFAULT_EPOCHS
    assert args.batch_size == cli.DEFAULT_BATCH_SIZE
    assert args.learning_rate == cli.DEFAULT_LEARNING_RATE
    assert args.optimizer == "adam"
    assert args.lr_scheduler is None
    assert args.early_stopping_patience is None
    assert args.resume_from is None
    assert args.checkpoint_out is None
    assert args.seed == cli.SEED
    assert args.export_torchscript is True


def test_parse_args_reads_batch_size_and_learning_rate() -> None:
    args = cli.parse_args([*REQUIRED_ARGS, "--batch-size", "32", "--learning-rate", "5e-4"])

    assert args.batch_size == 32
    assert args.learning_rate == 5e-4


def test_parse_args_reads_seed() -> None:
    args = cli.parse_args([*REQUIRED_ARGS, "--seed", "123"])

    assert args.seed == 123


def test_parse_args_reads_resume_from_and_checkpoint_out() -> None:
    args = cli.parse_args([*REQUIRED_ARGS, "--resume-from", "a.pt", "--checkpoint-out", "b.pt"])

    assert args.resume_from == Path("a.pt")
    assert args.checkpoint_out == Path("b.pt")


def test_parse_args_no_export_torchscript_disables_export() -> None:
    args = cli.parse_args([*REQUIRED_ARGS, "--no-export-torchscript"])

    assert args.export_torchscript is False


def test_parse_args_export_torchscript_explicit_true() -> None:
    args = cli.parse_args([*REQUIRED_ARGS, "--export-torchscript"])

    assert args.export_torchscript is True
