"""scripts/train_imagefolder.py의 실제 main(argv) 배선 통합 테스트
(Phase 4H, Phase 4G의 test_run_imagefolder_training_e2e_resume_cli.py와
같은 패턴). production CLI는 C++를 전혀 호출하지 않으므로(설계상 배제)
monkeypatch도 필요 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
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


# -- Phase 4K: Graceful SIGINT and Cooperative Training Stop ------------------
#
# docs/phase4k_graceful_interruption_design.md 기준. 실제 OS signal은 전혀
# 보내지 않는다 -- controller 테스트는 handle_signal()을 직접 호출하고, CLI
# 배선 테스트는 cli.signal.getsignal/cli.signal.signal을 완전한 fake로
# 대체해 실제 pytest 프로세스의 SIGINT handler를 건드리지 않는다(§13-2).


def _make_fake_result(checkpoint_path: Path | None = None) -> ImageFolderWorkflowResult:
    """main()의 결과 출력 코드가 접근하는 최소한의 필드만 채운 fake
    ImageFolderWorkflowResult. 실제 파일을 만들지 않는다 -- main()은 경로
    문자열을 출력만 할 뿐 파일을 다시 읽지 않는다."""
    history = TrainingHistory(
        train_losses=[0.5], val_losses=[0.5], val_accuracies=[0.5],
        best_epoch=1, best_val_loss=0.5,
    )
    return ImageFolderWorkflowResult(
        history=history,
        test_loss=0.5,
        test_accuracy=0.5,
        best_model_state_dict_path=Path("best_model_state_dict.pt"),
        training_history_path=Path("training_history.json"),
        class_mapping_path=Path("class_mapping.json"),
        test_result_path=Path("test_result.json"),
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=None,
        torchscript_model_path=None,
        torchscript_metadata_path=None,
    )


def _install_fake_signal_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[object, object]], dict[str, object], object]:
    """cli.signal.getsignal()/cli.signal.signal()을 완전한 fake로 대체한다
    (docs/phase4k_graceful_interruption_design.md §13-2). fake 안에서 진짜
    signal.signal()을 호출해 위임하는 방식은 절대 쓰지 않는다 -- 그러면
    pytest 프로세스 자체의 실제 SIGINT handler가 바뀌어버린다. 반환값은
    (signal_calls, current_handler, previous_handler) -- signal_calls는
    (sig, handler) 튜플의 설치/복원 호출 기록, current_handler는 fake가
    추적하는 "현재 설치된 handler" 가변 상태, previous_handler는 설치
    이전 상태를 나타내는 sentinel이다."""
    previous_handler = object()
    current_handler: dict[str, object] = {"value": previous_handler}
    signal_calls: list[tuple[object, object]] = []

    def fake_getsignal(sig):
        assert sig == cli.signal.SIGINT
        return current_handler["value"]

    def fake_signal(sig, handler):
        assert sig == cli.signal.SIGINT
        signal_calls.append((sig, handler))
        previous = current_handler["value"]
        current_handler["value"] = handler
        return previous

    monkeypatch.setattr(cli.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(cli.signal, "signal", fake_signal)
    return signal_calls, current_handler, previous_handler


# -- controller 단위 테스트 ----------------------------------------------------


def test_sigint_controller_initial_should_stop_is_false() -> None:
    controller = cli._SigintStopController()
    assert controller.should_stop() is False


def test_sigint_controller_first_signal_sets_should_stop_true() -> None:
    controller = cli._SigintStopController()
    controller.handle_signal(cli.signal.SIGINT, None)
    assert controller.should_stop() is True


def test_sigint_controller_first_signal_writes_message_once_to_stderr_fd(capfd) -> None:
    """os.write(2, ...)는 Python의 sys.stderr 텍스트 스트림을 거치지 않고
    파일 디스크립터에 직접 쓰므로, capsys가 아니라 fd 레벨까지 캡처하는
    capfd로 검증한다(설계 문서 §13-1)."""
    controller = cli._SigintStopController()
    controller.handle_signal(cli.signal.SIGINT, None)

    captured = capfd.readouterr()
    assert captured.err.count("Interrupt requested.") == 1
    assert "Training will stop at the next safe epoch boundary." in captured.err
    assert "Press Ctrl+C again to terminate immediately." in captured.err


def test_sigint_controller_second_signal_raises_keyboard_interrupt_without_repeating_message(capfd) -> None:
    controller = cli._SigintStopController()
    controller.handle_signal(cli.signal.SIGINT, None)
    capfd.readouterr()  # 1차 안내 메시지 버퍼 소비

    with pytest.raises(KeyboardInterrupt):
        controller.handle_signal(cli.signal.SIGINT, None)

    captured = capfd.readouterr()
    assert captured.err == ""  # 2차 호출은 안내를 다시 출력하지 않음


def test_sigint_controller_should_stop_remains_true_after_multiple_reads() -> None:
    controller = cli._SigintStopController()
    controller.handle_signal(cli.signal.SIGINT, None)
    assert [controller.should_stop() for _ in range(5)] == [True] * 5


def test_sigint_controller_first_signal_does_not_consume_torch_rng() -> None:
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    controller = cli._SigintStopController()
    controller.handle_signal(cli.signal.SIGINT, None)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_sigint_controller_oswrite_failure_does_not_propagate_and_should_stop_still_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_write(fd, data):
        raise OSError("stderr is closed")

    monkeypatch.setattr(cli.os, "write", failing_write)

    controller = cli._SigintStopController()
    controller.handle_signal(cli.signal.SIGINT, None)  # 예외가 전파되면 안 됨

    assert controller.should_stop() is True


def test_sigint_controller_oswrite_called_exactly_once_on_first_signal_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """호출 횟수/인자만 검증하는 테스트이므로 실제 fd에 쓰지 않는다 --
    실제 stderr 출력 관찰은 별도의 capfd 테스트
    (test_sigint_controller_first_signal_writes_message_once_to_stderr_fd)가
    이미 담당하므로 여기서 중복 I/O를 만들지 않는다."""
    calls: list[tuple[int, bytes]] = []

    def spying_write(fd, data):
        calls.append((fd, data))
        return len(data)

    monkeypatch.setattr(cli.os, "write", spying_write)

    controller = cli._SigintStopController()
    controller.handle_signal(cli.signal.SIGINT, None)
    assert len(calls) == 1
    assert calls[0][0] == 2
    assert calls[0][1] == cli._INTERRUPT_MESSAGE_BYTES

    with pytest.raises(KeyboardInterrupt):
        controller.handle_signal(cli.signal.SIGINT, None)
    assert len(calls) == 1  # 2차 호출에서는 os.write가 다시 호출되지 않음


def test_sigint_controller_constructor_takes_no_arguments() -> None:
    """controller가 model/optimizer/generator 등 학습 객체에 대한 참조를
    전혀 갖지 않는 구조임을 생성자 시그니처 자체로 보장한다."""
    import inspect

    signature = inspect.signature(cli._SigintStopController.__init__)
    assert list(signature.parameters) == ["self"]


# -- CLI signal 배선 테스트 ------------------------------------------------------


def test_cli_first_sigint_makes_should_stop_return_true_and_restores_handler_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # workflow는 fake로 대체하므로 실제 dataset 폴더를 만들 필요가 없다
    # (dataset 로딩/검증은 run_imagefolder_training_workflow() 내부에서만
    # 일어나고, 이 테스트는 그 호출 자체를 가짜로 바꾼다).
    model_json_path = _write_model_json(tmp_path, "cli_sigint_wiring_model")
    signal_calls, current_handler, previous_handler = _install_fake_signal_module(monkeypatch)

    captured: dict = {}

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        captured["should_stop"] = should_stop
        assert callable(should_stop)
        assert should_stop() is False
        installed_handler = signal_calls[0][1]
        installed_handler(cli.signal.SIGINT, None)
        assert should_stop() is True
        return _make_fake_result()

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
            "--no-export-torchscript",
        ]
    )

    assert exit_code == 0
    assert "should_stop" in captured
    assert len(signal_calls) == 2  # 설치 1회 + 복원 1회
    assert current_handler["value"] is previous_handler


def test_cli_handler_restored_after_workflow_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_json_path = _write_model_json(tmp_path, "cli_sigint_value_error_model")
    signal_calls, current_handler, previous_handler = _install_fake_signal_module(monkeypatch)

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        raise ValueError("boom")

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
        ]
    )

    assert exit_code == 1
    assert current_handler["value"] is previous_handler
    # 설치 1회(controller.handle_signal) + 복원 1회(previous_handler) -- 그 외
    # 불필요한 추가 설치/복원 호출이 없어야 한다.
    assert len(signal_calls) == 2
    assert callable(signal_calls[0][1])
    assert signal_calls[1][1] is previous_handler


def test_cli_handler_restored_after_workflow_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_json_path = _write_model_json(tmp_path, "cli_sigint_os_error_model")
    signal_calls, current_handler, previous_handler = _install_fake_signal_module(monkeypatch)

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
        ]
    )

    assert exit_code == 1
    assert current_handler["value"] is previous_handler
    assert len(signal_calls) == 2
    assert callable(signal_calls[0][1])
    assert signal_calls[1][1] is previous_handler


def test_cli_handler_restored_after_workflow_keyboard_interrupt_and_exit_code_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """두 번째 SIGINT의 escalation을 흉내낸다 -- workflow 호출 도중
    KeyboardInterrupt가 발생해도 handler가 먼저 복원되고, exit code는
    130이어야 한다."""
    model_json_path = _write_model_json(tmp_path, "cli_sigint_keyboard_interrupt_model")
    signal_calls, current_handler, previous_handler = _install_fake_signal_module(monkeypatch)

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
        ]
    )

    assert exit_code == 130
    assert current_handler["value"] is previous_handler
    assert len(signal_calls) == 2
    assert callable(signal_calls[0][1])
    assert signal_calls[1][1] is previous_handler
    stderr = capsys.readouterr().err
    assert "Interrupted" in stderr


def test_cli_handler_install_failure_returns_exit_1_and_never_calls_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    model_json_path = _write_model_json(tmp_path, "cli_sigint_install_failure_model")

    signal_attempts: list[tuple[object, object]] = []

    def failing_getsignal(sig):
        return object()

    def failing_signal(sig, handler):
        signal_attempts.append((sig, handler))
        raise ValueError("signal only works in main thread of the main interpreter")

    monkeypatch.setattr(cli.signal, "getsignal", failing_getsignal)
    monkeypatch.setattr(cli.signal, "signal", failing_signal)

    workflow_calls = []

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        workflow_calls.append(request)
        return _make_fake_result()

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
        ]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "main thread" in stderr
    assert workflow_calls == []
    # signal.signal 호출은 실패한 설치 시도 1회뿐 -- 설치가 실패했으므로
    # 복원 시도(2번째 호출)는 있으면 안 된다.
    assert len(signal_attempts) == 1
    assert signal_attempts[0][1] is not None  # controller.handle_signal을 넘기려 시도했음


def test_cli_checkpoint_out_none_should_stop_wiring_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # workflow는 fake로 대체하므로 실제 dataset 폴더가 필요 없다.
    model_json_path = _write_model_json(tmp_path, "cli_sigint_no_checkpoint_out_model")
    signal_calls, current_handler, previous_handler = _install_fake_signal_module(monkeypatch)

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        assert request.checkpoint_out is None
        installed_handler = signal_calls[0][1]
        installed_handler(cli.signal.SIGINT, None)
        assert should_stop() is True
        return _make_fake_result(checkpoint_path=None)

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
        ]
    )

    assert exit_code == 0
    assert current_handler["value"] is previous_handler
    assert len(signal_calls) == 2
    assert callable(signal_calls[0][1])
    assert signal_calls[1][1] is previous_handler


# -- main() 전체 KeyboardInterrupt -> exit code 130 --------------------------


def test_cli_keyboard_interrupt_during_parse_args_returns_130(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def failing_parse_args(argv=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "parse_args", failing_parse_args)

    exit_code = cli.main([])

    assert exit_code == 130
    stderr = capsys.readouterr().err
    assert "Interrupted" in stderr


def test_cli_keyboard_interrupt_after_workflow_return_during_result_output_returns_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """workflow가 정상 반환된 뒤(handler가 이미 복원된 상태) 결과 출력
    코드(history = result.history)에서 KeyboardInterrupt가 발생해도
    exit code 130으로 수렴해야 한다(설계 문서 §9-3)."""
    model_json_path = _write_model_json(tmp_path, "cli_sigint_post_return_model")
    _, current_handler, previous_handler = _install_fake_signal_module(monkeypatch)

    class _ResultWithFailingHistoryAccess:
        @property
        def history(self):
            raise KeyboardInterrupt()

    def fake_workflow(request, *, progress_callback=None, should_stop=None):
        return _ResultWithFailingHistoryAccess()

    monkeypatch.setattr(cli, "run_imagefolder_training_workflow", fake_workflow)

    exit_code = cli.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "1",
        ]
    )

    assert exit_code == 130
    # workflow가 이미 정상 반환된 뒤이므로, 이 시점에는 handler가 이미
    # 복원되어 있어야 한다.
    assert current_handler["value"] is previous_handler
    stderr = capsys.readouterr().err
    assert "Interrupted" in stderr
