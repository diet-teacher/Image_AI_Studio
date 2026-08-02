"""scripts/run_imagefolder_training_e2e.py의 실제 main(argv) 배선 통합
테스트 (Phase 4G, docs/phase4g_imagefolder_resume_design.md §11-4).

metadata 단위 테스트(test_imagefolder_resume.py)와 함수 수준 exact-resume
테스트만으로는 main(argv) 내부의 실제 배선(argparse 값이 올바른 변수로
전달되는지, 호출 순서가 설계대로인지)까지는 검증하지 못한다 -- 여기서는
main(argv)를 checkpoint 저장 -> resume 순으로 두 번 실제로 호출해 그
배선을 증명한다. C++ TorchScript runner(find_runner_binary/run_case)는
monkeypatch로 대체해 실제 빌드된 바이너리 없이도 통과하도록 한다 -- C++
parity 자체의 정확성은 기존 E2E 스크립트가 이미 커버하므로 이 테스트의
책임이 아니다. exact tensor 동등성(가중치 bit-identical 여부)도 이
테스트의 책임이 아니다 -- 그건 test_imagefolder_resume.py의 함수 수준
통합 테스트가 담당하고, 이 테스트는 "배선이 맞는가"만 확인한다.
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

import run_imagefolder_training_e2e as e2e_script  # noqa: E402

from image_ai_studio.model_definition.serialization import save_model_spec  # noqa: E402
from image_ai_studio.model_definition.specs import (  # noqa: E402
    DropoutSpec,
    FlattenSpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.training.checkpoint import load_training_checkpoint  # noqa: E402
from image_ai_studio.training.imagefolder_resume import metadata_path_for_checkpoint  # noqa: E402

INPUT_SHAPE = (3, 8, 8)


_CLASS_COLORS = {"cat": (250, 250, 250), "dog": (5, 5, 5)}


def _make_split(root: Path, split: str, count_per_class: int) -> None:
    """클래스마다 고정된(거의 흑/백) 색을 써서 선형 분리가 쉬운 fixture를
    만든다 -- 이 테스트의 목적은 학습 성능 검증이 아니라 main(argv) 배선
    검증이므로, 스크립트 자체의 "training loss decreased" 게이트를
    안정적으로 통과시키기 위해 극단적으로 분리 가능한 데이터를 쓴다."""
    for class_name, color in _CLASS_COLORS.items():
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True)
        for i in range(count_per_class):
            Image.new("RGB", (20, 20), color=color).save(class_dir / f"{i}.png")


def _make_standard_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        _make_split(root, split, count_per_class=8)


def _fake_find_runner_binary(build_dir: Path, project_subdir: str, exe_name: str) -> Path:
    # main()은 이 경로가 .exists()이면 build 단계를 건너뛴다 -- 실제
    # 바이너리는 필요 없으므로 빈 파일을 하나 만들어 둔다.
    dummy = build_dir / "dummy_runner.exe"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.touch()
    return dummy


def _fake_run_case(**kwargs) -> dict:
    return {"status": "PASS"}


@pytest.fixture(autouse=True)
def _stub_cpp_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2e_script, "find_runner_binary", _fake_find_runner_binary)
    monkeypatch.setattr(e2e_script, "run_case", _fake_run_case)


def test_checkpoint_then_resume_via_main_argv(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)

    model_spec = ModelSpec(
        name="phase4g_cli_wiring_test_model",
        input_shape=INPUT_SHAPE,
        layers=[
            FlattenSpec(),
            LinearSpec(out_features=16),
            ReLUSpec(),
            DropoutSpec(p=0.3),
            LinearSpec(out_features=2),
        ],
    )
    model_json_path = tmp_path / "model.json"
    save_model_spec(model_spec, model_json_path)

    ckpt1 = tmp_path / "checkpoint.pt"
    ckpt2 = tmp_path / "checkpoint2.pt"

    # 1회차: 신규 학습 2 epoch + checkpoint 저장
    exit_code_1 = e2e_script.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--epochs", "2",
            "--checkpoint-out", str(ckpt1),
        ]
    )
    assert exit_code_1 == 0
    assert ckpt1.exists()
    assert metadata_path_for_checkpoint(ckpt1).exists()

    # 2회차: 1회차 checkpoint에서 resume, 1 epoch 추가 + 새 checkpoint 저장
    exit_code_2 = e2e_script.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--epochs", "1",
            "--resume-from", str(ckpt1),
            "--checkpoint-out", str(ckpt2),
        ]
    )
    assert exit_code_2 == 0
    assert ckpt2.exists()
    assert metadata_path_for_checkpoint(ckpt2).exists()

    # --epochs는 resume 시 "총 epoch"가 아니라 "추가 epoch 수"다 -- 1회차
    # 2 epoch + 2회차 1 epoch = 누적 history 길이 3이어야, main(argv) 내부의
    # 실제 배선이 Phase 4F 계약대로 동작함을 증명한다.
    payload = load_training_checkpoint(ckpt2)
    assert len(payload["history"]["train_losses"]) == 3


def test_resume_without_metadata_sidecar_fails_clearly(tmp_path: Path) -> None:
    """checkpoint(.pt)만 있고 metadata(.meta.json)가 없으면 main(argv)이
    FAIL로 명확히 종료해야 한다 (§8 정책)."""
    _make_standard_dataset(tmp_path)

    model_spec = ModelSpec(
        name="phase4g_cli_wiring_missing_metadata_model",
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=2)],
    )
    model_json_path = tmp_path / "model.json"
    save_model_spec(model_spec, model_json_path)

    ckpt = tmp_path / "checkpoint_no_metadata.pt"
    exit_code = e2e_script.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            # epochs=1이면 train_losses가 원소 하나뿐이라 스크립트의 기존
            # "loss decreased" 게이트(train_losses[-1] < train_losses[0])가
            # 같은 원소를 자기 자신과 비교해 항상 False가 된다 -- Phase 4G와
            # 무관한 기존 동작이므로, 이 테스트는 2 epoch를 써서 그 게이트를
            # 우회한다.
            "--epochs", "2",
            "--checkpoint-out", str(ckpt),
        ]
    )
    assert exit_code == 0
    assert ckpt.exists()

    metadata_path_for_checkpoint(ckpt).unlink()  # metadata sidecar만 제거

    exit_code_resume = e2e_script.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--epochs", "1",
            "--resume-from", str(ckpt),
        ]
    )
    assert exit_code_resume == 1


def test_resume_from_nonexistent_path_fails_cleanly_without_traceback(tmp_path: Path) -> None:
    """--resume-from이 아예 존재하지 않는 경로를 가리키면(오타, 아직 한
    번도 저장 안 함 등) traceback 없이 exit code 1 + FAIL 메시지로
    끝나야 한다."""
    _make_standard_dataset(tmp_path)

    model_spec = ModelSpec(
        name="phase4g_cli_wiring_nonexistent_resume_model",
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=2)],
    )
    model_json_path = tmp_path / "model.json"
    save_model_spec(model_spec, model_json_path)

    missing_ckpt = tmp_path / "does_not_exist" / "checkpoint.pt"

    exit_code = e2e_script.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--epochs", "1",
            "--resume-from", str(missing_ckpt),
        ]
    )
    assert exit_code == 1


def test_resume_with_metadata_but_missing_checkpoint_file_fails_cleanly(tmp_path: Path) -> None:
    """metadata(.meta.json)만 있고 checkpoint(.pt) 파일이 지워진 경우 --
    load_training_checkpoint()가 존재 확인 없이 곧바로 torch.load()를
    호출하므로 FileNotFoundError(OSError)가 그대로 올라올 수 있다. main()이
    이를 ValueError와 동일하게 traceback 없이 FAIL로 처리해야 한다."""
    _make_standard_dataset(tmp_path)

    model_spec = ModelSpec(
        name="phase4g_cli_wiring_missing_checkpoint_file_model",
        input_shape=INPUT_SHAPE,
        layers=[FlattenSpec(), LinearSpec(out_features=2)],
    )
    model_json_path = tmp_path / "model.json"
    save_model_spec(model_spec, model_json_path)

    ckpt = tmp_path / "checkpoint_missing_pt.pt"
    exit_code = e2e_script.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--epochs", "2",
            "--checkpoint-out", str(ckpt),
        ]
    )
    assert exit_code == 0
    assert metadata_path_for_checkpoint(ckpt).exists()

    ckpt.unlink()  # checkpoint(.pt)만 제거, metadata sidecar는 남겨 둠

    exit_code_resume = e2e_script.main(
        [
            "--model-json", str(model_json_path),
            "--dataset-root", str(tmp_path),
            "--epochs", "1",
            "--resume-from", str(ckpt),
        ]
    )
    assert exit_code_resume == 1
