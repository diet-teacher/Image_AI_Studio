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
from image_ai_studio.training.imagefolder_resume import (
    load_imagefolder_resume_metadata,
    metadata_path_for_checkpoint,
)
from image_ai_studio.training.imagefolder_workflow import (
    ImageFolderWorkflowRequest,
    ImageFolderWorkflowResult,
    _is_in_place_resume,
    _normalized_path,
    _validate_checkpoint_every,
    _validate_checkpoint_output_paths,
    _validate_device,
    run_imagefolder_training_workflow,
)
from image_ai_studio.training.loop import TrainingHistory, evaluate_classification_metrics, run_training
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


# -- classification metrics / test_result.json 확장 (Phase 4O) ---------------


def test_production_result_has_classification_metrics(tmp_path: Path) -> None:
    """run_imagefolder_training_workflow()가 정상 완료한 production 결과는
    test_metrics가 항상 실제 ClassificationMetrics다(None이 아니다) --
    dataclass의 default=None은 constructor 하위호환 전용이라는 계약을
    실제 production 경로로 확인한다."""
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

    assert result.test_metrics is not None
    num_classes = 2  # _spec()의 마지막 LinearSpec(out_features=2), cat/dog
    assert len(result.test_metrics.confusion_matrix) == num_classes
    assert all(len(row) == num_classes for row in result.test_metrics.confusion_matrix)
    assert len(result.test_metrics.per_class_recall) == num_classes
    assert isinstance(result.test_metrics.macro_precision, float)
    assert isinstance(result.test_metrics.macro_recall, float)
    assert isinstance(result.test_metrics.macro_f1, float)

    # test_result.json도 기존 top-level key(test_loss/test_accuracy)는 유지한
    # 채 nested classification_metrics를 추가로 담아야 한다(additive).
    on_disk = json.loads(result.test_result_path.read_text())
    assert on_disk["test_loss"] == result.test_loss
    assert on_disk["test_accuracy"] == result.test_accuracy
    assert on_disk["classification_metrics"]["confusion_matrix"] == result.test_metrics.confusion_matrix
    assert on_disk["classification_metrics"]["per_class_recall"] == pytest.approx(
        result.test_metrics.per_class_recall
    )
    assert on_disk["classification_metrics"]["macro_precision"] == pytest.approx(
        result.test_metrics.macro_precision
    )
    assert on_disk["classification_metrics"]["macro_recall"] == pytest.approx(result.test_metrics.macro_recall)
    assert on_disk["classification_metrics"]["macro_f1"] == pytest.approx(result.test_metrics.macro_f1)

    # confusion matrix는 JSON에서도 정수로 남아야 한다(부동소수로 뭉개지지 않음).
    for row in on_disk["classification_metrics"]["confusion_matrix"]:
        for value in row:
            assert isinstance(value, int)


def test_classification_metrics_class_index_order_matches_class_mapping(tmp_path: Path) -> None:
    """confusion_matrix/per_class_recall의 class index 순서가 실제로
    class_mapping.json의 classes 순서와 일치하는지를, class별 test sample
    개수를 서로 다르게 만들어 관찰 가능한 방식으로 검증한다(shape만 보고
    order가 맞다고 주장하지 않는다). test split의 class별 sample 수를
    cat=6, dog=2로 비대칭으로 만들면, confusion matrix의 row별 합(=해당
    true class의 test sample 수)이 어느 class_mapping index가 cat/dog인지를
    직접 드러낸다."""
    for split, count_per_class in (("train", 4), ("val", 4)):
        _make_split(tmp_path, split, count_per_class)
    # test split만 cat/dog 개수를 다르게 만든다.
    cat_dir = tmp_path / "test" / "cat"
    dog_dir = tmp_path / "test" / "dog"
    cat_dir.mkdir(parents=True)
    dog_dir.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (20, 20), color=(250, 250, 250)).save(cat_dir / f"{i}.png")
    for i in range(2):
        Image.new("RGB", (20, 20), color=(5, 5, 5)).save(dog_dir / f"{i}.png")

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

    class_mapping = json.loads(result.class_mapping_path.read_text())
    assert class_mapping["classes"] == ["cat", "dog"]  # ImageFolder는 알파벳순 정렬
    cat_idx = class_mapping["class_to_idx"]["cat"]
    dog_idx = class_mapping["class_to_idx"]["dog"]

    cm = result.test_metrics.confusion_matrix
    cat_row_support = sum(cm[cat_idx])
    dog_row_support = sum(cm[dog_idx])

    assert cat_row_support == 6
    assert dog_row_support == 2


def test_imagefolder_workflow_result_constructor_backward_compatible() -> None:
    """test_metrics 없이 ImageFolderWorkflowResult(...)를 직접 생성하던
    기존 코드(테스트의 manual/fake constructor 호출)가 이번 Phase 이후에도
    그대로 성공하고, 그 경우 test_metrics는 backward-compat 전용 기본값
    None이어야 한다."""
    result = ImageFolderWorkflowResult(
        history=TrainingHistory(),
        test_loss=0.5,
        test_accuracy=0.9,
        best_model_state_dict_path=Path("best.pt"),
        training_history_path=Path("history.json"),
        class_mapping_path=Path("class_mapping.json"),
        test_result_path=Path("test_result.json"),
        checkpoint_path=None,
        checkpoint_metadata_path=None,
        torchscript_model_path=None,
        torchscript_metadata_path=None,
    )
    assert result.test_metrics is None


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


def test_resume_succeeds_from_checkpoint_file_missing_weight_decay(tmp_path: Path) -> None:
    """Phase 4L hotfix 회귀: Phase 4L 이전 형식(checkpoint 파일의
    training_config에 weight_decay 키가 없음)을 실제 checkpoint 파일에서
    흉내내, production resume 경로 전체
    (run_imagefolder_training_workflow -> _prepare_resume() ->
    load_training_checkpoint() -> require_compatible_resume_config())를
    통해 정상적으로 resume되는지 확인한다. 이 회귀는 가짜 dict가 아니라
    실제 저장된 checkpoint 파일을 수정해야만 재현됐다(load_training_checkpoint()
    가 require_compatible_resume_config()보다 먼저 실행되기 때문)."""
    _make_standard_dataset(tmp_path)
    spec = _spec()
    model_json_path = _write_model_json(tmp_path, spec)
    checkpoint_path = tmp_path / "checkpoint.pt"

    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2, weight_decay=0.0),
            output_dir=tmp_path / "a",
            checkpoint_out=checkpoint_path,
            export_torchscript=False,
            seed=SEED,
        )
    )

    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["training_config"]["weight_decay"]
    torch.save(payload, checkpoint_path)

    result = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2, weight_decay=0.0),
            output_dir=tmp_path / "b",
            resume_from=checkpoint_path,
            checkpoint_out=checkpoint_path,
            export_torchscript=False,
            seed=SEED,
        )
    )

    assert len(result.history.train_losses) == 2  # 1(fresh) + 1(resume)


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


def test_workflow_forwards_label_smoothing_to_criterion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 4N: TrainingConfig.label_smoothing이 production 경로
    (ImageFolderWorkflowRequest -> run_imagefolder_training_workflow() ->
    run_training() -> _build_criterion())를 통해 실제로 전달되는지 확인한다.
    새 E2E 스크립트 대신 기존 통합 테스트 파일의 monkeypatch/spy 패턴을
    그대로 재사용한다(이미지 데이터셋 학습 자체를 새로 검증할 필요는 없음
    -- 그건 다른 기존 테스트가 담당)."""
    calls: list[float] = []
    from image_ai_studio.training.loop import _build_criterion as real_build_criterion

    def spy_build_criterion(config: TrainingConfig, device: str = "cpu"):
        calls.append(config.label_smoothing)
        return real_build_criterion(config, device=device)

    monkeypatch.setattr("image_ai_studio.training.loop._build_criterion", spy_build_criterion)

    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2, label_smoothing=0.3),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
    )

    run_imagefolder_training_workflow(request)

    assert calls == [0.3]


def test_workflow_trains_successfully_with_matching_class_weights_length(tmp_path: Path) -> None:
    """dataset class 수(cat/dog=2)와 class_weights 길이가 일치하면 정상
    학습된다(Phase 4P)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(
            epochs=1, batch_size=4, learning_rate=1e-2, class_weights=(1.0, 2.0)
        ),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
    )

    result = run_imagefolder_training_workflow(request)

    assert len(result.history.train_losses) == 1


def test_workflow_rejects_class_weights_length_mismatch_before_training_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dataset은 class 2개(cat/dog)인데 class_weights가 1개뿐이면, 실제
    학습(run_training)이 시작되기 전에 명확한 에러로 거부돼야 한다.
    run_training()을 monkeypatch해 절대 호출되지 않음을 직접 증명한다 --
    "에러가 난다"는 것만으로는 그 에러가 학습을 실제로 시작한 뒤(예: 첫
    optimizer step 도중 PyTorch RuntimeError)에 난 것인지, 시작 전
    조기 검증에서 난 것인지 구분되지 않는다."""
    called = {"value": False}

    def fail_if_called(*args, **kwargs):
        called["value"] = True
        raise AssertionError("run_training() must not be called when class_weights length mismatches")

    monkeypatch.setattr("image_ai_studio.training.imagefolder_workflow.run_training", fail_if_called)

    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(
            epochs=1, batch_size=4, learning_rate=1e-2, class_weights=(1.0,)
        ),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
    )

    with pytest.raises(ValueError, match="class_weights"):
        run_imagefolder_training_workflow(request)

    assert called["value"] is False


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


# -- Phase 4J: checkpoint_every validation -------------------------------------


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_validate_checkpoint_every_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        _validate_checkpoint_every(value)


def test_validate_checkpoint_every_accepts_none_and_positive_int() -> None:
    _validate_checkpoint_every(None)
    _validate_checkpoint_every(1)
    _validate_checkpoint_every(5)


def test_workflow_checkpoint_every_without_checkpoint_out_raises(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
        checkpoint_every=1,
    )
    with pytest.raises(ValueError, match="checkpoint_every"):
        run_imagefolder_training_workflow(request)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_workflow_checkpoint_every_invalid_value_raises(tmp_path: Path, value: object) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        checkpoint_out=tmp_path / "checkpoint.pt",
        export_torchscript=False,
        seed=SEED,
        checkpoint_every=value,
    )
    with pytest.raises(ValueError):
        run_imagefolder_training_workflow(request)


# -- Phase 4J: checkpoint_every cadence (global epoch 기준) --------------------


def _spy_save_training_checkpoint(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """save_training_checkpoint 호출마다 그 시점의 global epoch(=
    training_result.history의 길이)을 기록하는 spy를 설치한다."""
    import image_ai_studio.training.imagefolder_workflow as workflow_module

    recorded: list[int] = []
    original = workflow_module.save_training_checkpoint

    def spy(*args, **kwargs):
        training_result = kwargs["training_result"]
        recorded.append(len(training_result.history.train_losses))
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "save_training_checkpoint", spy)
    return recorded


def test_workflow_checkpoint_every_none_saves_only_once_at_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    recorded = _spy_save_training_checkpoint(monkeypatch)

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        checkpoint_out=tmp_path / "checkpoint.pt",
        export_torchscript=False,
        seed=SEED,
    )
    run_imagefolder_training_workflow(request)

    assert recorded == [3]


def test_workflow_checkpoint_every_fresh_records_scheduled_and_final_global_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fresh 5 epoch, checkpoint_every=2 -> scheduled 저장이 global epoch
    2, 4에서, 최종 저장이 5에서 발생해야 한다."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    recorded = _spy_save_training_checkpoint(monkeypatch)

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=5, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        checkpoint_out=tmp_path / "checkpoint.pt",
        export_torchscript=False,
        seed=SEED,
        checkpoint_every=2,
    )
    run_imagefolder_training_workflow(request)

    assert recorded == [2, 4, 5]


def test_workflow_checkpoint_every_in_place_resume_uses_global_epoch_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """기존 global epoch 7에서 checkpoint_every=5로 3 epoch를 추가하면,
    scheduled 저장과 최종 저장이 둘 다 global epoch 10에서 발생해야
    한다(cadence가 hook 호출 횟수가 아니라 global epoch 기준임을
    증명) -- 총 2번 호출."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"

    seed_request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=7, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "seed_out",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    seeded = run_imagefolder_training_workflow(seed_request)
    assert len(seeded.history.train_losses) == 7

    recorded = _spy_save_training_checkpoint(monkeypatch)
    resume_request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "resume_out",
        resume_from=checkpoint_path,
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
        checkpoint_every=5,
    )
    result = run_imagefolder_training_workflow(resume_request)

    assert len(result.history.train_losses) == 10
    assert recorded == [10, 10]


# -- Phase 4J: 출력 경로 재사용 정책 --------------------------------------------


def test_fresh_training_rejects_existing_checkpoint_out(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"

    first_request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "o1",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    run_imagefolder_training_workflow(first_request)
    original_bytes = checkpoint_path.read_bytes()

    second_request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "o2",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    with pytest.raises(ValueError, match="already exists"):
        run_imagefolder_training_workflow(second_request)

    assert checkpoint_path.read_bytes() == original_bytes


def test_fresh_training_rejects_existing_metadata_sidecar_only(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"
    sidecar_path = metadata_path_for_checkpoint(checkpoint_path)
    sidecar_path.write_text('{"stale": true}', encoding="utf-8")
    original_bytes = sidecar_path.read_bytes()

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    with pytest.raises(ValueError, match="already exists"):
        run_imagefolder_training_workflow(request)

    assert sidecar_path.read_bytes() == original_bytes
    assert not checkpoint_path.exists()


def test_fresh_training_completely_new_path_saves_metadata_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"
    sidecar_path = metadata_path_for_checkpoint(checkpoint_path)

    import image_ai_studio.training.imagefolder_workflow as workflow_module

    original_save_checkpoint = workflow_module.save_training_checkpoint
    metadata_existed_before_checkpoint_save = {"value": None}

    def spy_save_checkpoint(*args, **kwargs):
        metadata_existed_before_checkpoint_save["value"] = sidecar_path.exists()
        return original_save_checkpoint(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "save_training_checkpoint", spy_save_checkpoint)

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    run_imagefolder_training_workflow(request)

    assert metadata_existed_before_checkpoint_save["value"] is True
    assert checkpoint_path.exists()
    assert sidecar_path.exists()


def test_resume_to_different_output_rejects_existing_checkpoint_at_output(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    resume_source = tmp_path / "source.pt"
    other_output = tmp_path / "other.pt"

    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path, dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "o1", checkpoint_out=resume_source,
            export_torchscript=False, seed=SEED,
        )
    )
    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path, dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "o2", checkpoint_out=other_output,
            export_torchscript=False, seed=SEED,
        )
    )
    resume_source_bytes_before = resume_source.read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        run_imagefolder_training_workflow(
            ImageFolderWorkflowRequest(
                model_json_path=model_json_path, dataset_root=tmp_path,
                training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
                output_dir=tmp_path / "o3", resume_from=resume_source, checkpoint_out=other_output,
                export_torchscript=False, seed=SEED,
            )
        )

    assert resume_source.read_bytes() == resume_source_bytes_before


def test_resume_to_different_output_rejects_existing_metadata_sidecar_only(tmp_path: Path) -> None:
    """`resume_from != checkpoint_out`이고 출력 경로에 checkpoint(.pt)는
    없지만 metadata sidecar만 남아 있는 경우도 §6-5 정책에 따라
    학습을 시작하기 전에 거부돼야 한다."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    resume_source = tmp_path / "source.pt"
    other_output = tmp_path / "other.pt"

    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path, dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "o1", checkpoint_out=resume_source,
            export_torchscript=False, seed=SEED,
        )
    )
    resume_source_bytes_before = resume_source.read_bytes()
    resume_source_metadata_bytes_before = metadata_path_for_checkpoint(resume_source).read_bytes()

    other_sidecar = metadata_path_for_checkpoint(other_output)
    other_sidecar.write_text('{"stale": true}', encoding="utf-8")
    other_sidecar_bytes_before = other_sidecar.read_bytes()
    assert not other_output.exists()

    with pytest.raises(ValueError, match="already exists"):
        run_imagefolder_training_workflow(
            ImageFolderWorkflowRequest(
                model_json_path=model_json_path, dataset_root=tmp_path,
                training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
                output_dir=tmp_path / "o2", resume_from=resume_source, checkpoint_out=other_output,
                export_torchscript=False, seed=SEED,
            )
        )

    # 학습이 시작되기도 전에 거부되므로 output checkpoint는 생성되지 않고,
    # source(resume_from)/other 경로의 sidecar 둘 다 전혀 바뀌지 않는다.
    assert not other_output.exists()
    assert other_sidecar.read_bytes() == other_sidecar_bytes_before
    assert resume_source.read_bytes() == resume_source_bytes_before
    assert metadata_path_for_checkpoint(resume_source).read_bytes() == resume_source_metadata_bytes_before


def test_resume_to_different_new_output_succeeds_and_metadata_matches_source(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    resume_source = tmp_path / "source.pt"
    new_output = tmp_path / "new_output.pt"

    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path, dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "o1", checkpoint_out=resume_source,
            export_torchscript=False, seed=SEED,
        )
    )
    source_metadata = load_imagefolder_resume_metadata(metadata_path_for_checkpoint(resume_source))

    result = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path, dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "o2", resume_from=resume_source, checkpoint_out=new_output,
            export_torchscript=False, seed=SEED,
        )
    )

    assert result.checkpoint_path == new_output
    assert new_output.exists()
    new_output_metadata = load_imagefolder_resume_metadata(metadata_path_for_checkpoint(new_output))
    assert new_output_metadata == source_metadata


def test_normalized_path_recognizes_relative_and_absolute_as_same_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    absolute = tmp_path / "checkpoint.pt"
    relative = Path("checkpoint.pt")

    assert _normalized_path(absolute) == _normalized_path(relative)


def test_is_in_place_resume_true_only_when_paths_match(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    request_same = ImageFolderWorkflowRequest(
        model_json_path=tmp_path / "m.json", dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path, resume_from=a, checkpoint_out=a,
    )
    request_different = ImageFolderWorkflowRequest(
        model_json_path=tmp_path / "m.json", dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path, resume_from=a, checkpoint_out=b,
    )
    request_fresh = ImageFolderWorkflowRequest(
        model_json_path=tmp_path / "m.json", dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path, checkpoint_out=a,
    )

    assert _is_in_place_resume(request_same) is True
    assert _is_in_place_resume(request_different) is False
    assert _is_in_place_resume(request_fresh) is False


def test_validate_checkpoint_output_paths_allows_in_place_resume_without_checking_existence(
    tmp_path: Path,
) -> None:
    """resume_from == checkpoint_out면 파일이 이미 존재해도(당연히 존재해야
    하므로) 이 검증은 그냥 통과해야 한다 -- 실제 검증은 _prepare_resume()이
    담당."""
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not a real checkpoint")
    request = ImageFolderWorkflowRequest(
        model_json_path=tmp_path / "m.json", dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path, resume_from=path, checkpoint_out=path,
    )
    _validate_checkpoint_output_paths(request)  # raise 없이 통과해야 함


def test_metadata_write_success_checkpoint_save_failure_leaves_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """완전히 새 출력 경로에서 metadata 저장은 성공하고 그 직후
    checkpoint 저장이 실패하면, 디스크에는 metadata만 남고 checkpoint는
    생기지 않아야 한다."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"
    sidecar_path = metadata_path_for_checkpoint(checkpoint_path)

    import image_ai_studio.training.imagefolder_workflow as workflow_module

    def failing_save_checkpoint(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(workflow_module, "save_training_checkpoint", failing_save_checkpoint)

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        checkpoint_out=checkpoint_path,
        export_torchscript=False,
        seed=SEED,
    )
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        run_imagefolder_training_workflow(request)

    assert sidecar_path.exists()
    assert not checkpoint_path.exists()


def test_resume_from_metadata_only_state_fails_with_clear_missing_checkpoint_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """위 상태(metadata만 있고 checkpoint 없음)에서 그 경로로 resume을
    시도하면 load_training_checkpoint()가 존재 확인 없이 곧바로
    torch.load()를 호출하므로 FileNotFoundError로 명확히 실패해야 한다
    (test_train_imagefolder_cli.py의
    test_resume_with_metadata_but_missing_checkpoint_file_fails_cleanly와
    동일한 근거)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"

    import image_ai_studio.training.imagefolder_workflow as workflow_module

    def failing_save_checkpoint(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    # save_training_checkpoint 실패 patch는 "metadata만 남기고 checkpoint
    # 없음" 상태를 만드는 이 블록 안에서만 적용한다 -- 그 상태를 만든 뒤
    # 이어지는 resume 시도는 patch가 걸리지 않은 원래 동작(실제
    # load_training_checkpoint())으로 검증해야 하므로 monkeypatch.undo()
    # 대신 범위가 명확한 context manager를 쓴다.
    with monkeypatch.context() as m:
        m.setattr(workflow_module, "save_training_checkpoint", failing_save_checkpoint)
        with pytest.raises(RuntimeError, match="simulated disk failure"):
            run_imagefolder_training_workflow(
                ImageFolderWorkflowRequest(
                    model_json_path=model_json_path, dataset_root=tmp_path,
                    training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
                    output_dir=tmp_path / "out", checkpoint_out=checkpoint_path,
                    export_torchscript=False, seed=SEED,
                )
            )

    assert metadata_path_for_checkpoint(checkpoint_path).exists()
    assert not checkpoint_path.exists()

    with pytest.raises(FileNotFoundError, match=r"checkpoint\.pt"):
        run_imagefolder_training_workflow(
            ImageFolderWorkflowRequest(
                model_json_path=model_json_path, dataset_root=tmp_path,
                training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
                output_dir=tmp_path / "out2", resume_from=checkpoint_path,
                export_torchscript=False, seed=SEED,
            )
        )


# -- Phase 4J: metadata_ready 공유(최대 한 번만 준비) --------------------------


def _spy_save_metadata(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    import image_ai_studio.training.imagefolder_workflow as workflow_module

    calls: list[None] = []
    original = workflow_module.save_imagefolder_resume_metadata

    def spy(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "save_imagefolder_resume_metadata", spy)
    return calls


def test_fresh_checkpoint_every_one_writes_metadata_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fresh 5 epoch, checkpoint_every=1 -> scheduled 저장 5번 + 최종
    저장 1번 = 6번의 checkpoint 저장 기회가 있지만, metadata는 정확히
    1번만 저장돼야 한다."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    metadata_calls = _spy_save_metadata(monkeypatch)
    checkpoint_calls = _spy_save_training_checkpoint(monkeypatch)

    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=5, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        checkpoint_out=tmp_path / "checkpoint.pt",
        export_torchscript=False,
        seed=SEED,
        checkpoint_every=1,
    )
    run_imagefolder_training_workflow(request)

    assert len(checkpoint_calls) == 6
    assert len(metadata_calls) == 1


def test_in_place_resume_never_rewrites_metadata_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"
    sidecar_path = metadata_path_for_checkpoint(checkpoint_path)

    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path, dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "o1", checkpoint_out=checkpoint_path,
            export_torchscript=False, seed=SEED,
        )
    )
    content_before = sidecar_path.read_bytes()

    metadata_calls = _spy_save_metadata(monkeypatch)
    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path, dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "o2", resume_from=checkpoint_path, checkpoint_out=checkpoint_path,
            export_torchscript=False, seed=SEED, checkpoint_every=1,
        )
    )

    # save_imagefolder_resume_metadata()가 한 번도 호출되지 않았다는 사실
    # 자체가 "재작성되지 않음"을 직접 증명한다 -- 파일시스템 timestamp
    # 정밀도 차이로 flaky해질 수 있는 mtime 비교에는 의존하지 않는다.
    assert len(metadata_calls) == 0
    assert sidecar_path.read_bytes() == content_before


# -- Phase 4Q: runtime training device exposure -------------------------------


def test_workflow_request_device_defaults_to_cpu() -> None:
    request = ImageFolderWorkflowRequest(
        model_json_path=Path("model.json"),
        dataset_root=Path("dataset"),
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=Path("out"),
    )
    assert request.device == "cpu"


def test_validate_device_accepts_cpu() -> None:
    _validate_device("cpu")  # raise 없이 통과해야 함


@pytest.mark.parametrize(
    "value",
    [
        "gpu", "CUDA", "CPU", "mps", "xpu", "hip", "cuda:", "cuda:-1", "cuda:00", "cuda: 0", "", 123, None,
        "cpu\n", "cuda\n", "cuda:0\n",
    ],
)
def test_validate_device_rejects_invalid_syntax(value: object) -> None:
    """공식 지원 syntax는 cpu/cuda/cuda:N뿐이다 -- 대소문자 변형, 다른
    backend(mps/xpu/hip), zero-padding/음수 index, 빈 문자열, 문자열이
    아닌 타입 전부 거부한다. "cpu\\n" 등 trailing newline 케이스는 정규식이
    `match()`가 아니라 `fullmatch()`를 쓴다는 계약을 직접 고정한다 --
    Python `re`의 `$`는 문자열 끝뿐 아니라 trailing newline 직전에도
    매치될 수 있어(`match()`로는 "cpu\\n"가 통과함을 실측 확인),
    `fullmatch()`가 아니면 이 케이스가 조용히 통과해버린다."""
    with pytest.raises(ValueError, match="device"):
        _validate_device(value)


def test_validate_device_rejects_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA가 사용 불가능하면 CPU로 조용히 대체하지 않고 명확히 거부한다."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="cuda"):
        _validate_device("cuda")
    with pytest.raises(ValueError, match="cuda"):
        _validate_device("cuda:0")


def test_validate_device_accepts_plain_cuda_and_valid_index_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    _validate_device("cuda")  # raise 없이 통과해야 함 (PyTorch default CUDA device 의미)
    _validate_device("cuda:0")  # raise 없이 통과해야 함
    _validate_device("cuda:1")  # raise 없이 통과해야 함


def test_validate_device_rejects_out_of_range_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.to(device)`까지 내려가면 저수준 AcceleratorError가 나는 것을
    직접 실측했다 -- 여기서 조기에 명확한 ValueError로 거부한다."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="out of range"):
        _validate_device("cuda:1")


def test_workflow_rejects_invalid_device_before_training_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """device가 유효하지 않으면 run_training()이 호출되지 않았음을
    monkeypatch로 직접 증명하며, 학습 시작 전에 조기 거부돼야 한다
    (Phase 4P의 class_weights 길이 mismatch 조기 검증과 동일한 패턴)."""
    called = {"value": False}

    def fail_if_called(*args, **kwargs):
        called["value"] = True
        raise AssertionError("run_training() must not be called when device is invalid")

    monkeypatch.setattr("image_ai_studio.training.imagefolder_workflow.run_training", fail_if_called)

    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
        device="not-a-real-device",
    )

    with pytest.raises(ValueError, match="device"):
        run_imagefolder_training_workflow(request)

    assert called["value"] is False


def test_workflow_forwards_device_to_run_training_and_moves_model_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """request.device가 실제로 run_training()의 device kwarg로 전달되고,
    run_training()에 넘어가는 model이 이미 그 device 위에 있는지(즉
    model.to(device)가 run_training() 호출보다 먼저 일어났는지) 직접
    고정한다. GPU가 없는 CI에서도 "cpu" 경로로 이 wiring 전체를 실측할
    수 있다(optional CUDA smoke test가 실제 device 이동 자체를 커버한다)."""
    captured: dict = {}
    real_run_training = run_training

    def spy_run_training(model, train_loader, val_loader, config, device="cpu", **kwargs):
        captured["device_kwarg"] = device
        captured["model_device"] = next(model.parameters()).device
        return real_run_training(model, train_loader, val_loader, config, device=device, **kwargs)

    monkeypatch.setattr("image_ai_studio.training.imagefolder_workflow.run_training", spy_run_training)

    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
        device="cpu",
    )

    run_imagefolder_training_workflow(request)

    assert captured["device_kwarg"] == "cpu"
    assert captured["model_device"] == torch.device("cpu")


def test_workflow_final_evaluation_explicitly_uses_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """workflow의 최종 detailed evaluation 호출
    (evaluate_classification_metrics)이 request.device를 그대로 전달하지
    않고 명시적으로 device="cpu"를 쓴다는 계약을 고정한다(Phase 4Q는
    training device exposure이지 evaluation device exposure가 아니다).
    이 테스트는 request.device="cpu"인 경우만 다룬다 -- CUDA training
    뒤 최종 test/export 전체 경로가 실제로 CPU에서 정상 완료되는지는
    별도의 optional CUDA smoke test
    (test_workflow_cuda_training_completes_with_cpu_final_test_and_export)
    가 담당한다."""
    captured: dict = {}
    real_evaluate_classification_metrics = evaluate_classification_metrics

    def spy_evaluate(*args, **kwargs):
        captured["device_kwarg"] = kwargs.get("device")
        return real_evaluate_classification_metrics(*args, **kwargs)

    monkeypatch.setattr(
        "image_ai_studio.training.imagefolder_workflow.evaluate_classification_metrics", spy_evaluate
    )

    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=tmp_path / "out",
        export_torchscript=False,
        seed=SEED,
        device="cpu",
    )

    run_imagefolder_training_workflow(request)

    assert captured["device_kwarg"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_workflow_cuda_training_completes_with_cpu_final_test_and_export(tmp_path: Path) -> None:
    """optional CUDA smoke test(Phase 4Q) -- 실제 CUDA가 있는 로컬 환경에서만
    실행된다(GPU 없는 CI에서는 자동 skip). 작은 ImageFolder fixture로
    device="cuda" 학습이 성공하고, 최종 test 평가와 TorchScript export까지
    (CPU best_model 기반으로) 정상 완료되는지 한 번에 확인한다 -- generic
    CUDA smoke + export boundary를 이 테스트 하나로 충분히 커버하므로
    중복 GPU 테스트를 추가하지 않는다."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    request = ImageFolderWorkflowRequest(
        model_json_path=model_json_path,
        dataset_root=tmp_path,
        training_config=TrainingConfig(
            epochs=1, batch_size=4, learning_rate=1e-2, class_weights=(1.0, 2.0)
        ),
        output_dir=tmp_path / "out",
        export_torchscript=True,
        seed=SEED,
        device="cuda",
    )

    result = run_imagefolder_training_workflow(request)

    assert len(result.history.train_losses) == 1
    assert result.test_metrics is not None
    assert result.torchscript_model_path is not None
    assert result.torchscript_model_path.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_workflow_resume_from_cpu_checkpoint_on_cuda_completes(tmp_path: Path) -> None:
    """optional CUDA resume smoke test(Phase 4Q) -- CPU에서 1 epoch 학습한
    checkpoint를 CUDA에서 resume했을 때 에러 없이 완료되는지만 확인한다
    (portability smoke). bitwise exact equality는 주장하지 않으므로
    assert하지 않는다 -- CUDA→CPU 대칭 테스트는 추가하지 않는다(한 방향
    portability smoke로 충분하다는 판단)."""
    _make_standard_dataset(tmp_path)
    model_json_path = _write_model_json(tmp_path, _spec())
    checkpoint_path = tmp_path / "checkpoint.pt"

    run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "cpu_run",
            checkpoint_out=checkpoint_path,
            export_torchscript=False,
            seed=SEED,
            device="cpu",
        )
    )

    result = run_imagefolder_training_workflow(
        ImageFolderWorkflowRequest(
            model_json_path=model_json_path,
            dataset_root=tmp_path,
            training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
            output_dir=tmp_path / "cuda_resume",
            resume_from=checkpoint_path,
            checkpoint_out=checkpoint_path,
            export_torchscript=False,
            seed=SEED,
            device="cuda",
        )
    )

    assert len(result.history.train_losses) == 2
