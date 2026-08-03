"""scripts/train_imagefolder.py의 실제 main(argv) 배선 통합 테스트
(Phase 4H, Phase 4G의 test_run_imagefolder_training_e2e_resume_cli.py와
같은 패턴). production CLI는 C++를 전혀 호출하지 않으므로(설계상 배제)
monkeypatch도 필요 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
