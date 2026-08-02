"""scripts/run_imagefolder_training_e2e.py의 parse_args() 테스트 (Phase 4G).

parse_args(argv=None)는 이미 argv를 직접 받는 함수라 parser factory로
리팩터링할 필요 없이 그대로 테스트 가능하다 (docs/
phase4g_imagefolder_resume_design.md §11-3). scripts/는 패키지가 아니므로
conftest.py의 src/ 등록과 같은 방식으로 scripts/를 sys.path에 등록한 뒤
직접 import한다 -- 학습 자체는 실행하지 않고 인자 파싱만 확인한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_imagefolder_training_e2e as e2e_script  # noqa: E402


def test_parse_args_defaults_have_no_resume_or_checkpoint() -> None:
    args = e2e_script.parse_args([])

    assert args.epochs == e2e_script.DEFAULT_EPOCHS
    assert args.resume_from is None
    assert args.checkpoint_out is None


def test_parse_args_reads_epochs() -> None:
    args = e2e_script.parse_args(["--epochs", "7"])

    assert args.epochs == 7


def test_parse_args_reads_resume_from() -> None:
    args = e2e_script.parse_args(["--resume-from", "artifacts/training/foo_checkpoint.pt"])

    assert args.resume_from == Path("artifacts/training/foo_checkpoint.pt")


def test_parse_args_reads_checkpoint_out() -> None:
    args = e2e_script.parse_args(["--checkpoint-out", "artifacts/training/foo_checkpoint.pt"])

    assert args.checkpoint_out == Path("artifacts/training/foo_checkpoint.pt")


def test_parse_args_reads_resume_from_and_checkpoint_out_together() -> None:
    args = e2e_script.parse_args(
        [
            "--epochs", "2",
            "--resume-from", "a.pt",
            "--checkpoint-out", "b.pt",
        ]
    )

    assert args.epochs == 2
    assert args.resume_from == Path("a.pt")
    assert args.checkpoint_out == Path("b.pt")
