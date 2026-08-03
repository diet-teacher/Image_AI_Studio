"""training/imagefolder_workflow.py 테스트 (Phase 4H).

`tmp_path` + PIL fixture(기존 test_imagefolder_resume.py 패턴 재사용)로
완전히 오프라인 검증한다. C++ runner는 이 모듈이 아예 import하지 않으므로
monkeypatch가 필요 없다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import save_model_spec
from image_ai_studio.model_definition.specs import (
    DropoutSpec,
    FlattenSpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.parity.compare_outputs import CPU_FP32_ATOL, CPU_FP32_RTOL, compare_outputs
from image_ai_studio.training.checkpoint import load_state_dict, load_training_checkpoint
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.imagefolder_workflow import (
    ImageFolderWorkflowRequest,
    run_imagefolder_training_workflow,
)
from image_ai_studio.training.loop import run_training
from image_ai_studio.training.torchvision_dataset import make_imagefolder_datasets

INPUT_SHAPE = (3, 8, 8)
SEED = 20260803


def _make_split(root: Path, split: str, count_per_class: int = 4) -> None:
    for class_name, base_color in (("cat", (250, 250, 250)), ("dog", (5, 5, 5))):
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True)
        for i in range(count_per_class):
            Image.new("RGB", (20, 20), color=base_color).save(class_dir / f"{i}.png")


def _make_standard_dataset(root: Path, count_per_class: int = 4) -> None:
    for split in ("train", "val", "test"):
        _make_split(root, split, count_per_class)


def _spec(name: str = "imagefolder_workflow_test_model") -> ModelSpec:
    return ModelSpec(
        name=name,
        input_shape=INPUT_SHAPE,
        layers=[
            FlattenSpec(),
            LinearSpec(out_features=16),
            ReLUSpec(),
            DropoutSpec(p=0.3),
            LinearSpec(out_features=2),
        ],
    )


def _write_model_json(tmp_path: Path, spec: ModelSpec) -> Path:
    path = tmp_path / "model.json"
    save_model_spec(spec, path)
    return path


def _make_loaders(root: Path, batch_size: int, seed: int):
    splits = make_imagefolder_datasets(INPUT_SHAPE, root)
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(splits.val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# -- fresh training ---------------------------------------------------------


def test_fresh_training_creates_expected_artifacts_and_result(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    output_dir = tmp_path / "out"

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-2),
        output_dir=output_dir,
        seed=SEED,
    )
    result = run_imagefolder_training_workflow(request)

    assert result.best_model_state_dict_path == output_dir / "best_model_state_dict.pt"
    assert result.training_history_path == output_dir / "training_history.json"
    assert result.class_mapping_path == output_dir / "class_mapping.json"
    assert result.test_result_path == output_dir / "test_result.json"
    assert result.torchscript_model_path == output_dir / "model.ts"
    assert result.torchscript_metadata_path == output_dir / "model_metadata.json"
    assert result.checkpoint_path is None
    assert result.checkpoint_metadata_path is None

    for path in (
        result.best_model_state_dict_path,
        result.training_history_path,
        result.class_mapping_path,
        result.test_result_path,
        result.torchscript_model_path,
        result.torchscript_metadata_path,
    ):
        assert path.exists()

    assert len(result.history.train_losses) == 2
    assert isinstance(result.test_loss, float)
    assert isinstance(result.test_accuracy, float)


def test_output_dir_reuse_overwrites_fixed_filenames(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    output_dir = tmp_path / "out"

    def _run(epochs: int):
        request = ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=epochs, batch_size=4, learning_rate=1e-2),
            output_dir=output_dir,
            seed=SEED,
        )
        return run_imagefolder_training_workflow(request)

    result_first = _run(epochs=1)
    result_second = _run(epochs=3)

    # 같은 output_dir, 같은 고정 파일명 -- 두 번째 실행 내용으로 덮어써진다.
    assert result_first.training_history_path == result_second.training_history_path
    history_on_disk = json.loads(result_second.training_history_path.read_text())
    assert len(history_on_disk["train_losses"]) == 3


# -- checkpoint: current model, not best model -------------------------------


def test_checkpoint_stores_current_model_not_best_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """checkpoint의 model_state_dict가 (a) 동일 설정의 순수 run_training()
    호출이 만든 마지막 epoch model과 정확히 일치하고, (b) best epoch가
    마지막 epoch가 아닌 경우 best_state_dict와는 달라야 한다는 것을
    실제 테스트로 고정한다. ImageFolderWorkflowResult는 경로/지표만
    반환하므로(설계 유지), 별도의 기준(run_training() 직접 호출) 실행과
    비교하는 방식으로 검증한다."""
    _make_standard_dataset(tmp_path)
    spec = _spec()
    model_json_path = _write_model_json(tmp_path, spec)
    config = TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-2)

    # val_loss가 epoch 2에서 최소가 되고 epoch 3에서 다시 나빠지도록
    # 강제해, best_epoch(2) != 마지막 epoch(3)인 상황을 만든다.
    val_sequence_for_reference = iter([(1.0, 1.0), (0.5, 1.0), (0.8, 1.0)])
    val_sequence_for_workflow = iter([(1.0, 1.0), (0.5, 1.0), (0.8, 1.0)])

    # (a) 기준: run_training()을 직접 호출한 마지막 epoch model.
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(val_sequence_for_reference),
    )
    # 워크플로우의 fresh 경로는 set_seed()를 한 번만 호출한다(모델 생성 직전,
    # run_training() 직전에 다시 하지 않음) -- 기준 실행도 정확히 같은
    # 순서를 따라야 RNG 소비 지점이 일치한다.
    torch.manual_seed(SEED)
    reference_model = build_model(spec)
    train_loader, val_loader = _make_loaders(tmp_path, batch_size=4, seed=SEED)
    reference_result = run_training(reference_model, train_loader, val_loader, config)
    assert reference_result.history.best_epoch == 2  # 기대한 시나리오가 실제로 만들어졌는지 확인
    reference_state = {name: tensor.clone() for name, tensor in reference_model.state_dict().items()}

    # (b) workflow: 같은 seed/설정, checkpoint_out 지정.
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(val_sequence_for_workflow),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=config,
        output_dir=tmp_path / "out",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    result = run_imagefolder_training_workflow(request)
    assert result.history.best_epoch == 2

    payload = load_training_checkpoint(checkpoint_path)
    for name, expected_tensor in reference_state.items():
        assert torch.equal(payload["model_state_dict"][name], expected_tensor), name

    differs = any(
        not torch.equal(payload["model_state_dict"][name], payload["best_state_dict"][name])
        for name in payload["model_state_dict"]
    )
    assert differs, "best epoch(2)와 마지막 epoch(3)가 다른데 model/best state_dict가 동일함"


# -- best model 평가 / test_result.json --------------------------------------


def test_best_model_evaluation_and_test_result_json(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
    )
    result = run_imagefolder_training_workflow(request)

    best_model = build_model(_spec()).eval()
    load_state_dict(best_model, result.best_model_state_dict_path)
    from image_ai_studio.training.loop import evaluate
    from image_ai_studio.training.torchvision_dataset import make_imagefolder_datasets as _make

    splits = _make(INPUT_SHAPE, tmp_path)
    test_loader = torch.utils.data.DataLoader(splits.test, batch_size=4, shuffle=False)
    expected_loss, expected_accuracy = evaluate(best_model, test_loader, device="cpu")

    assert result.test_loss == pytest.approx(expected_loss)
    assert result.test_accuracy == pytest.approx(expected_accuracy)

    on_disk = json.loads(result.test_result_path.read_text())
    assert on_disk["test_loss"] == result.test_loss
    assert on_disk["test_accuracy"] == result.test_accuracy


# -- resume exactness ---------------------------------------------------------


def test_resume_matches_continuous_run_exactly(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    spec = _spec()
    model_json_path = _write_model_json(tmp_path, spec)
    config_kwargs = dict(
        batch_size=4,
        learning_rate=1e-2,
        optimizer="sgd",
        momentum=0.9,
        lr_scheduler="plateau",
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
    )

    result_a = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=4, **config_kwargs),
            output_dir=tmp_path / "a",
            export_torchscript=False,
            seed=SEED,
        )
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    result_b1 = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=2, **config_kwargs),
            output_dir=tmp_path / "b1",
            checkpoint_out=checkpoint_path,
            export_torchscript=False,
            seed=SEED,
        )
    )
    result_b2 = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=2, **config_kwargs),
            output_dir=tmp_path / "b2",
            resume_from=checkpoint_path,
            checkpoint_out=checkpoint_path,
            export_torchscript=False,
            seed=SEED,
        )
    )

    assert len(result_b2.history.train_losses) == 4  # 2(b1) + 2(b2) = 연속 4와 동일 길이
    assert result_b2.history.train_losses == result_a.history.train_losses
    assert result_b2.history.val_losses == result_a.history.val_losses
    assert result_b2.history.val_accuracies == result_a.history.val_accuracies
    assert result_b2.history.best_epoch == result_a.history.best_epoch
    assert result_b2.history.best_val_loss == result_a.history.best_val_loss
    assert result_b2.test_loss == result_a.test_loss
    assert result_b2.test_accuracy == result_a.test_accuracy

    model_a = build_model(spec)
    load_state_dict(model_a, result_a.best_model_state_dict_path)
    model_b2 = build_model(spec)
    load_state_dict(model_b2, result_b2.best_model_state_dict_path)
    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_b2.state_dict()[name])

    del result_b1  # 배선 확인용으로만 필요, 값 자체는 비교 대상 아님


def test_resume_rejects_dataset_mismatch(tmp_path: Path) -> None:
    saved_root = tmp_path / "saved"
    current_root = tmp_path / "current"
    _make_standard_dataset(saved_root)
    _make_standard_dataset(current_root)
    (current_root / "train" / "cat" / "0.png").rename(current_root / "train" / "cat" / "renamed.png")

    spec = _spec()
    model_json_path = _write_model_json(tmp_path, spec)
    config = TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2)
    checkpoint_path = tmp_path / "checkpoint.pt"

    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=saved_root,
            training_config=config,
            output_dir=tmp_path / "o1",
            checkpoint_out=checkpoint_path,
            export_torchscript=False,
            seed=SEED,
        )
    )

    with pytest.raises(ValueError):
        run_imagefolder_training_workflow(
            ImageFolderWorkflowRequest(
                model_json_path=model_json_path,
                dataset_root=current_root,
                training_config=config,
                output_dir=tmp_path / "o2",
                resume_from=checkpoint_path,
                export_torchscript=False,
                seed=SEED,
            )
        )


# -- TorchScript export -------------------------------------------------------


def test_export_torchscript_false_produces_no_torchscript_files(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    output_dir = tmp_path / "out"

    result = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=output_dir,
            export_torchscript=False,
            seed=SEED,
        )
    )

    assert result.torchscript_model_path is None
    assert result.torchscript_metadata_path is None
    assert not (output_dir / "model.ts").exists()
    assert not (output_dir / "model_metadata.json").exists()


def test_torchscript_export_output_matches_best_model(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    spec = _spec()
    model_json_path = _write_model_json(tmp_path, spec)

    result = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "out",
            export_torchscript=True,
            seed=SEED,
        )
    )

    best_model = build_model(spec).eval()
    load_state_dict(best_model, result.best_model_state_dict_path)

    torch.manual_seed(SEED)
    example_input = torch.randn(1, *INPUT_SHAPE)

    traced = torch.jit.load(str(result.torchscript_model_path)).eval()
    with torch.inference_mode():
        reference_output = best_model(example_input)
        traced_output = traced(example_input)

    parity = compare_outputs(reference_output, traced_output, rtol=CPU_FP32_RTOL, atol=CPU_FP32_ATOL)
    assert parity.allclose, parity.to_dict()


def test_disabling_export_removes_stale_torchscript_artifacts_but_keeps_other_files(
    tmp_path: Path,
) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    user_file = output_dir / "user_notes.txt"
    user_file.write_text("keep me")

    result_first = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=output_dir,
            export_torchscript=True,
            seed=SEED,
        )
    )
    assert result_first.torchscript_model_path.exists()
    assert result_first.torchscript_metadata_path.exists()

    result_second = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=output_dir,
            export_torchscript=False,
            seed=SEED,
        )
    )

    assert result_second.torchscript_model_path is None
    assert result_second.torchscript_metadata_path is None
    assert not (output_dir / "model.ts").exists()
    assert not (output_dir / "model_metadata.json").exists()
    assert user_file.exists()
    assert user_file.read_text() == "keep me"


# -- Phase 4I: progress_callback / should_stop forwarding --------------------


def test_workflow_forwards_progress_callback_to_run_training(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
    )

    progresses: list = []
    result = run_imagefolder_training_workflow(request, progress_callback=progresses.append)

    assert len(progresses) == 3 == len(result.history.train_losses)
    assert [p.global_epoch for p in progresses] == [1, 2, 3]


def test_workflow_forwards_should_stop_and_stops_training_early(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=5, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
    )

    stop_flag = {"value": False}

    def callback(progress) -> None:
        if progress.run_epoch == 2:
            stop_flag["value"] = True

    result = run_imagefolder_training_workflow(
        request, progress_callback=callback, should_stop=lambda: stop_flag["value"]
    )

    assert len(result.history.train_losses) == 2
    assert result.history.stopped_by_user is True


def test_workflow_user_stopped_run_produces_full_artifact_set(tmp_path: Path) -> None:
    """stopped_by_user=True로 끝난 학습도 정상 완료와 동일한 아티팩트
    파이프라인을 전부 거쳐야 한다 -- 별도 분기 없음
    (docs/phase4i_training_progress_and_stop_design.md §10)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    output_dir = tmp_path / "out"
    checkpoint_path = tmp_path / "checkpoint.pt"
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=5, batch_size=4, learning_rate=1e-2),
        output_dir=output_dir,
        checkpoint_out=checkpoint_path,
        export_torchscript=True,
        seed=SEED,
    )

    result = run_imagefolder_training_workflow(request, should_stop=lambda: True)

    assert len(result.history.train_losses) == 1
    assert result.history.stopped_by_user is True
    for path in (
        result.best_model_state_dict_path,
        result.training_history_path,
        result.class_mapping_path,
        result.test_result_path,
        result.checkpoint_path,
        result.checkpoint_metadata_path,
        result.torchscript_model_path,
        result.torchscript_metadata_path,
    ):
        assert path is not None
        assert path.exists()

    payload = load_training_checkpoint(checkpoint_path)
    assert payload["history"]["stopped_by_user"] is True


def test_workflow_user_stopped_checkpoint_is_resumable(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"

    first_request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=5, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out1",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    first = run_imagefolder_training_workflow(first_request, should_stop=lambda: True)
    assert len(first.history.train_losses) == 1
    assert first.history.stopped_by_user is True

    second_request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out2",
        resume_from=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    second = run_imagefolder_training_workflow(second_request)

    assert len(second.history.train_losses) == 3
    assert second.history.stopped_by_user is False
