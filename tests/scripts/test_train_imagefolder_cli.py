"""scripts/train_imagefolder.py의 실제 main(argv) 배선 통합 테스트
(Phase 4H, Phase 4G의 test_run_imagefolder_training_e2e_resume_cli.py와
같은 패턴). production CLI는 C++를 전혀 호출하지 않으므로(설계상 배제)
monkeypatch도 필요 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_imagefolder as cli  # noqa: E402

from image_ai_studio.model_definition.serialization import save_model_spec  # noqa: E402
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec  # noqa: E402
from image_ai_studio.training.checkpoint import load_training_checkpoint  # noqa: E402
from image_ai_studio.training.imagefolder_resume import metadata_path_for_checkpoint  # noqa: E402
from image_ai_studio.training.imagefolder_workflow import ImageFolderWorkflowResult  # noqa: E402
from image_ai_studio.training.loop import TrainingHistory  # noqa: E402

INPUT_SHAPE = (3, 8, 8)
_CLASS_COLORS = {"cat": (250, 250, 250), "dog": (5, 5, 5)}


def _make_split(root: Path, split: str, count_per_class: int = 4) -> None:
    for class_name, color in _CLASS_COLORS.items():
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True)
        for i in range(count_per_class):
            Image.new("RGB", (20, 20), color=color).save(class_dir / f"{i}.png")


def _make_standard_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        _make_split(root, split)


def _write_model_json(tmp_path: Path, name: str) -> Path:
    spec = ModelSpec(
        name=name,
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=16), ReLUSpec(), LinearSpec(out_features=2)],
    )
    path = tmp_path / "model.json"
    save_model_spec(spec, path)
    return path


def test_fresh_training_via_main_creates_output_dir_artifacts(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_fresh_model")
    output_dir = tmp_path / "out"

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(output_dir),
            "--epochs", "2",
            "--batch-size", "4",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "best_model_state_dict.pt").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "class_mapping.json").exists()
    assert (output_dir / "test_result.json").exists()
    assert (output_dir / "model.ts").exists()
    assert (output_dir / "model_metadata.json").exists()


def test_checkpoint_then_resume_via_main_argv(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_resume_model")
    ckpt1 = tmp_path / "checkpoint.pt"
    ckpt2 = tmp_path / "checkpoint2.pt"

    exit_code_1 = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out1"),
            "--epochs", "2",
            "--batch-size", "4",
            "--checkpoint-out", str(ckpt1),
            "--no-export-torchscript",
        ]
    )
    assert exit_code_1 == 0
    assert ckpt1.exists()
    assert metadata_path_for_checkpoint(ckpt1).exists()

    exit_code_2 = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out2"),
            "--epochs", "1",
            "--batch-size", "4",
            "--resume-from", str(ckpt1),
            "--checkpoint-out", str(ckpt2),
            "--no-export-torchscript",
        ]
    )
    assert exit_code_2 == 0
    assert ckpt2.exists()

    # --epochs는 resume 시 "총 epoch"가 아니라 "추가 epoch 수"다 -- 1회차
    # 2 epoch + 2회차 1 epoch = 누적 history 길이 3이어야 한다.
    payload = load_training_checkpoint(ckpt2)
    assert len(payload["history"]["train_losses"]) == 3


def test_resume_from_nonexistent_path_fails_cleanly_without_traceback(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_missing_resume_model")
    missing_ckpt = tmp_path / "does_not_exist" / "checkpoint.pt"

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
            "--resume-from", str(missing_ckpt),
        ]
    )

    assert exit_code == 1


def test_resume_without_metadata_sidecar_fails_clearly(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_missing_metadata_model")
    ckpt = tmp_path / "checkpoint.pt"

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out1"),
            "--epochs", "2",
            "--checkpoint-out", str(ckpt),
            "--no-export-torchscript",
        ]
    )
    assert exit_code == 0
    metadata_path_for_checkpoint(ckpt).unlink()

    exit_code_resume = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out2"),
            "--epochs", "1",
            "--resume-from", str(ckpt),
        ]
    )
    assert exit_code_resume == 1


def test_resume_with_metadata_but_missing_checkpoint_file_fails_cleanly(tmp_path: Path) -> None:
    """metadata(.meta.json)만 있고 checkpoint(.pt) 파일이 지워진 경우 --
    load_imagefolder_resume_metadata()는 성공하지만
    load_training_checkpoint()가 존재 확인 없이 곧바로 torch.load()를
    호출하므로 FileNotFoundError(OSError)가 그대로 올라올 수 있다.
    main()이 이를 traceback 없이 exit code 1로 처리해야 한다 (Phase 4G에서
    발견된 케이스, 이전 tests/scripts/test_run_imagefolder_training_e2e_resume_cli.py
    에 있던 검증 의도를 그대로 옮김)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_missing_checkpoint_file_model")
    ckpt = tmp_path / "checkpoint.pt"

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out1"),
            "--epochs", "2",
            "--checkpoint-out", str(ckpt),
            "--no-export-torchscript",
        ]
    )
    assert exit_code == 0
    assert metadata_path_for_checkpoint(ckpt).exists()

    ckpt.unlink()  # checkpoint(.pt)만 제거, metadata sidecar는 남겨 둠

    exit_code_resume = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out2"),
            "--epochs", "1",
            "--resume-from", str(ckpt),
        ]
    )
    assert exit_code_resume == 1


def test_no_export_torchscript_via_main_produces_no_torchscript_files(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_no_export_model")
    output_dir = tmp_path / "out"

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(output_dir),
            "--epochs", "1",
            "--no-export-torchscript",
        ]
    )

    assert exit_code == 0
    assert not (output_dir / "model.ts").exists()
    assert not (output_dir / "model_metadata.json").exists()


# -- Phase 4I: 실시간 progress 출력 -------------------------------------------


def _epoch_numbers_from_stdout(stdout: str) -> list[int]:
    """`  epoch {N}: train_loss=... val_loss=... val_acc=...` 형식의 줄에서
    epoch 번호만 뽑아낸다 -- 전체 문자열을 통째로 비교하지 않고, 이 테스트가
    실제로 검증하려는 계약(번호 순서/중복 여부)만 확인한다."""
    numbers = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("epoch "):
            numbers.append(int(stripped.split()[1].rstrip(":")))
    return numbers


def test_fresh_training_prints_progress_for_each_epoch_in_real_time(tmp_path: Path, capsys) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_progress_model")

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "3",
            "--batch-size", "4",
            "--no-export-torchscript",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert _epoch_numbers_from_stdout(captured.out) == [1, 2, 3]


def test_resume_via_main_prints_only_newly_completed_epochs_with_global_epoch_numbers(
    tmp_path: Path, capsys
) -> None:
    """설계 문서에 명시된 6단계 절차 그대로: (1) --checkpoint-out으로 1회차
    checkpoint 생성 (2) checkpoint.pt + checkpoint.pt.meta.json 존재 확인
    (3) 1회차 stdout 버퍼 소비 (4) --resume-from으로 2회차 실행 (5) epoch
    번호가 정확히 [3, 4]인지 확인 (6) 중복 없는지 확인. 과거 CLI는 resume 시
    누적 history 전체([1,2,3,4])를 사후 재출력했지만, Phase 4I부터는 새로
    완료된 epoch만 실시간으로 찍는다(docs/phase4i_training_progress_and_stop_design.md
    §11/§17)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_resume_progress_model")
    ckpt = tmp_path / "checkpoint.pt"

    # (1) 1회차: --checkpoint-out으로 checkpoint 생성 (2 epoch)
    exit_code_1 = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out1"),
            "--epochs", "2",
            "--batch-size", "4",
            "--checkpoint-out", str(ckpt),
            "--no-export-torchscript",
        ]
    )
    assert exit_code_1 == 0

    # (2) checkpoint.pt + metadata sidecar 존재 확인
    assert ckpt.exists()
    assert metadata_path_for_checkpoint(ckpt).exists()

    # (3) 1회차 stdout 버퍼 소비 (2회차 출력만 검사하기 위함)
    capsys.readouterr()

    # (4) 2회차: --resume-from으로 2 epoch 더 실행
    exit_code_2 = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out2"),
            "--epochs", "2",
            "--batch-size", "4",
            "--resume-from", str(ckpt),
            "--no-export-torchscript",
        ]
    )
    assert exit_code_2 == 0
    captured = capsys.readouterr()
    epoch_numbers = _epoch_numbers_from_stdout(captured.out)

    # (5) epoch 번호가 정확히 [3, 4] (누적 history 전체 재출력이 아님)
    assert epoch_numbers == [3, 4]

    # (6) 중복 없음
    assert len(epoch_numbers) == len(set(epoch_numbers))


# -- Phase 4J: --checkpoint-every ---------------------------------------------


def test_checkpoint_every_forwards_exact_value_to_workflow_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI가 `--checkpoint-every`/`--checkpoint-out` argparse 값을
    `ImageFolderWorkflowRequest`에 정확히 실어 보내는지 직접 증명한다.
    final post-hoc checkpoint는 `checkpoint_every`를 workflow에 전달하지
    않아도(즉 배선이 끊어져도) 만들어지므로, 실제 checkpoint 파일 존재
    여부만으로는 forwarding을 증명하지 못한다 -- 여기서는
    `run_imagefolder_training_workflow`를 monkeypatch로 가짜 구현으로
    바꾸고, CLI가 실제로 넘긴 `ImageFolderWorkflowRequest` 객체를
    캡처해서 필드 값을 직접 검사한다."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_checkpoint_every_forward_model")
    ckpt = tmp_path / "checkpoint.pt"

    captured: dict = {}

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        captured["request"] = request
        history = TrainingHistory(
            train_losses=[0.5], val_losses=[0.5], val_accuracies=[0.5],
            best_epoch=1, best_val_loss=0.5,
        )
        return ImageFolderWorkflowResult(
            history=history,
            test_loss=0.5,
            test_accuracy=0.5,
            best_model_state_dict_path=tmp_path / "best_model_state_dict.pt",
            training_history_path=tmp_path / "training_history.json",
            class_mapping_path=tmp_path / "class_mapping.json",
            test_result_path=tmp_path / "test_result.json",
            checkpoint_path=request.checkpoint_out,
            checkpoint_metadata_path=None,
            torchscript_model_path=None,
            torchscript_metadata_path=None,
        )

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "3",
            "--batch-size", "4",
            "--checkpoint-out", str(ckpt),
            "--checkpoint-every", "5",
            "--no-export-torchscript",
        ]
    )

    assert exit_code == 0
    assert "request" in captured  # fake_workflow가 실제로 호출됐는지 먼저 확인
    assert captured["request"].checkpoint_every == 5
    assert captured["request"].checkpoint_out == ckpt


def test_checkpoint_every_via_main_produces_final_checkpoint_end_to_end(tmp_path: Path) -> None:
    """실제 workflow를 그대로 실행해 --checkpoint-every가 있어도 학습이
    정상적으로 끝나고 최종 checkpoint가 만들어짐을 확인하는 end-to-end
    회귀 테스트(forwarding 자체의 증명은 위 request 캡처 테스트가
    담당)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_checkpoint_every_model")
    ckpt = tmp_path / "checkpoint.pt"

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "3",
            "--batch-size", "4",
            "--checkpoint-out", str(ckpt),
            "--checkpoint-every", "1",
            "--no-export-torchscript",
        ]
    )

    assert exit_code == 0
    assert ckpt.exists()
    assert metadata_path_for_checkpoint(ckpt).exists()
    payload = load_training_checkpoint(ckpt)
    assert len(payload["history"]["train_losses"]) == 3


def test_checkpoint_every_without_checkpoint_out_fails_cleanly(tmp_path: Path, capsys) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_checkpoint_every_no_out_model")

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "2",
            "--checkpoint-every", "1",
            "--no-export-torchscript",
        ]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "checkpoint_every" in stderr
    assert "checkpoint_out" in stderr


@pytest.mark.parametrize("value", ["0", "-1"])
def test_checkpoint_every_invalid_value_fails_cleanly(tmp_path: Path, value: str, capsys) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_checkpoint_every_invalid_model")

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "2",
            "--checkpoint-out", str(tmp_path / "checkpoint.pt"),
            "--checkpoint-every", value,
            "--no-export-torchscript",
        ]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "at least 1" in stderr


def test_checkpoint_every_non_integer_value_fails_argparse_parsing(tmp_path: Path, capsys) -> None:
    """`--checkpoint-every 1.5`는 workflow 검증(_validate_checkpoint_every())
    이전에, argparse의 `type=int` 변환 단계에서 이미 거부된다 -- argparse는
    파싱 오류에 SystemExit(2)를 던진다(workflow의 ValueError 경로와는
    다른 실패 지점)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_checkpoint_every_non_integer_model")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--model-json", str(model_json_path),
                "--dataset-root", str(tmp_path),
                "--output-dir", str(tmp_path / "out"),
                "--epochs", "2",
                "--checkpoint-out", str(tmp_path / "checkpoint.pt"),
                "--checkpoint-every", "1.5",
                "--no-export-torchscript",
            ]
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--checkpoint-every" in stderr


def test_fresh_run_reusing_existing_checkpoint_out_fails_cleanly_and_keeps_existing_file(
    tmp_path: Path, capsys
) -> None:
    """Phase 4J §6-5 정책의 CLI 레벨 회귀 테스트: fresh 학습이 이미
    checkpoint가 있는 --checkpoint-out 경로를 가리키면 exit code 1 +
    기존 파일은 그대로 남아야 한다."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, "cli_reuse_checkpoint_out_model")
    ckpt = tmp_path / "checkpoint.pt"

    exit_code_1 = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out1"),
            "--epochs", "1",
            "--checkpoint-out", str(ckpt),
            "--no-export-torchscript",
        ]
    )
    assert exit_code_1 == 0
    original_bytes = ckpt.read_bytes()
    capsys.readouterr()  # 1회차 stdout/stderr 버퍼 소비

    exit_code_2 = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out2"),
            "--epochs", "1",
            "--checkpoint-out", str(ckpt),
            "--no-export-torchscript",
        ]
    )

    assert exit_code_2 == 1
    assert ckpt.read_bytes() == original_bytes
    stderr = capsys.readouterr().err
    assert "already exists" in stderr
