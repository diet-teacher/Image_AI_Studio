"""training/imagefolder_resume.py 테스트.

11-1(metadata 단위 테스트: round-trip, 해시 결정론성, 호환성 검증, 경로
유도)과 11-2(ImageFolder exact-resume 통합 테스트: 연속 실행 vs
checkpoint+resume이 정확히 일치하는지)를 같은 파일에 담는다
(docs/phase4g_imagefolder_resume_design.md §11 참고). CIFAR-10과 달리
ImageFolder는 다운로드가 필요 없으므로 tmp_path에 PIL로 직접 만든 작은
train/val/test 구조만 사용해 완전히 오프라인으로 검증한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import (
    DropoutSpec,
    FlattenSpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.imagefolder_resume import (
    _atomic_write_text,
    build_imagefolder_resume_metadata,
    hash_model_spec,
    load_imagefolder_resume_metadata,
    metadata_path_for_checkpoint,
    require_compatible_imagefolder_resume_metadata,
    save_imagefolder_resume_metadata,
)
from image_ai_studio.training.loop import TrainingHistory, TrainingResumeState, run_training
from image_ai_studio.training.torchvision_dataset import make_imagefolder_datasets

INPUT_SHAPE = (3, 8, 8)


def _make_split(root: Path, split: str, class_to_images: dict[str, int]) -> None:
    """root/split/<class>/*.png 구조를 만든다 (test_imagefolder_dataset.py와
    동일 패턴)."""
    for class_name, count in class_to_images.items():
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (20, 20), color=(10 * i, 20, 30)).save(class_dir / f"{i}.png")


def _make_standard_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        _make_split(root, split, {"cat": 4, "dog": 4})


def _dropout_classifier_spec(name: str = "imagefolder_resume_test_model") -> ModelSpec:
    """Dropout 포함 -- 전역 CPU RNG를 소비하므로 exact-resume 테스트가
    RNG state 복원 없이는 통과하지 못한다는 것을 구조적으로 보장한다
    (test_loop.py의 동일 패턴)."""
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


# -- metadata_path_for_checkpoint() -------------------------------------------


@pytest.mark.parametrize(
    ("checkpoint_name", "expected_meta_name"),
    [
        ("checkpoint.pt", "checkpoint.pt.meta.json"),
        ("foo.v2.pt", "foo.v2.pt.meta.json"),
        ("checkpoint", "checkpoint.meta.json"),
    ],
)
def test_metadata_path_for_checkpoint_derivation(
    tmp_path: Path, checkpoint_name: str, expected_meta_name: str
) -> None:
    checkpoint_path = tmp_path / checkpoint_name
    assert metadata_path_for_checkpoint(checkpoint_path) == tmp_path / expected_meta_name


# -- hash_model_spec() ----------------------------------------------------------


def test_hash_model_spec_is_deterministic_for_same_spec() -> None:
    spec_a = _dropout_classifier_spec()
    spec_b = _dropout_classifier_spec()

    assert hash_model_spec(spec_a) == hash_model_spec(spec_b)


def test_hash_model_spec_changes_when_layer_param_changes() -> None:
    spec = _dropout_classifier_spec()
    changed = _dropout_classifier_spec()
    changed.layers[3] = DropoutSpec(p=0.5)  # p=0.3 -> 0.5

    assert hash_model_spec(spec) != hash_model_spec(changed)


def test_hash_model_spec_is_unaffected_by_model_json_file_path(tmp_path: Path) -> None:
    """model_spec_to_dict()는 파일 경로를 담지 않으므로, 같은 내용을 다른
    경로에 저장해도(파일명을 바꿔도) 해시는 같아야 한다."""
    from image_ai_studio.model_definition.serialization import save_model_spec

    spec = _dropout_classifier_spec()
    path_a = tmp_path / "a" / "model.json"
    path_b = tmp_path / "b" / "renamed_model.json"
    save_model_spec(spec, path_a)
    save_model_spec(spec, path_b)

    from image_ai_studio.model_definition.serialization import load_model_spec

    assert hash_model_spec(load_model_spec(path_a)) == hash_model_spec(load_model_spec(path_b))


# -- build/save/load round-trip --------------------------------------------------


def test_metadata_round_trip(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    spec = _dropout_classifier_spec()
    splits = make_imagefolder_datasets(INPUT_SHAPE, tmp_path)

    metadata = build_imagefolder_resume_metadata(spec, splits)
    metadata_path = tmp_path / "checkpoint.pt.meta.json"
    save_imagefolder_resume_metadata(metadata, metadata_path)
    reloaded = load_imagefolder_resume_metadata(metadata_path)

    assert reloaded == metadata


def test_build_imagefolder_resume_metadata_records_sizes_and_class_to_idx(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    spec = _dropout_classifier_spec()
    splits = make_imagefolder_datasets(INPUT_SHAPE, tmp_path)

    metadata = build_imagefolder_resume_metadata(spec, splits)

    assert metadata.class_to_idx == {"cat": 0, "dog": 1}
    assert metadata.train_size == 8
    assert metadata.val_size == 8
    assert metadata.test_size == 8


# -- Phase 4J: atomic write ----------------------------------------------------


def test_atomic_write_text_creates_parent_directories(tmp_path: Path) -> None:
    nested_path = tmp_path / "nested" / "dir" / "sidecar.json"
    _atomic_write_text('{"value": 1}', nested_path)

    assert nested_path.read_text(encoding="utf-8") == '{"value": 1}'


def test_atomic_write_text_no_leftover_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.json"
    _atomic_write_text('{"value": 1}', path)

    assert list(tmp_path.iterdir()) == [path]


def test_save_imagefolder_resume_metadata_atomic_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_imagefolder_resume_metadata()가 내부적으로 원자적 저장을
    쓰므로, 저장 도중 실패(완전히 새 경로에서 metadata 쓰기가 실패하는
    시나리오)해도 그 자리에 checkpoint가 아직 생기지 않은 상태(즉 아무
    파일도 없는 상태)가 그대로 유지되어야 한다."""
    dataset_root = tmp_path / "dataset"
    _make_standard_dataset(dataset_root)
    spec = _dropout_classifier_spec()
    splits = make_imagefolder_datasets(INPUT_SHAPE, dataset_root)

    metadata_path = tmp_path / "checkpoint.pt.meta.json"
    metadata = build_imagefolder_resume_metadata(spec, splits)

    def failing_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("image_ai_studio.training.imagefolder_resume.os.replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        save_imagefolder_resume_metadata(metadata, metadata_path)

    assert not metadata_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []  # 임시 파일 미잔존


def test_atomic_write_text_cleanup_failure_does_not_mask_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sidecar.json"

    def failing_replace(*args, **kwargs):
        raise OSError("original failure")

    def failing_unlink(self, *args, **kwargs):
        raise OSError("cleanup also failed")

    monkeypatch.setattr("image_ai_studio.training.imagefolder_resume.os.replace", failing_replace)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(OSError, match="original failure"):
        _atomic_write_text('{"value": 1}', path)


# -- load_imagefolder_resume_metadata() 에러 계약 --------------------------------


def test_load_imagefolder_resume_metadata_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_imagefolder_resume_metadata(tmp_path / "no_such_file.meta.json")


def test_load_imagefolder_resume_metadata_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "broken.meta.json"
    path.write_text('{"metadata_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field"):
        load_imagefolder_resume_metadata(path)


def test_load_imagefolder_resume_metadata_rejects_unsupported_version(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    spec = _dropout_classifier_spec()
    splits = make_imagefolder_datasets(INPUT_SHAPE, tmp_path)
    metadata = build_imagefolder_resume_metadata(spec, splits)

    path = tmp_path / "checkpoint.pt.meta.json"
    save_imagefolder_resume_metadata(metadata, path)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    data["metadata_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported metadata_version"):
        load_imagefolder_resume_metadata(path)


# -- require_compatible_imagefolder_resume_metadata() ----------------------------


def _build_metadata(root: Path, spec: ModelSpec | None = None) -> "object":
    spec = spec or _dropout_classifier_spec()
    splits = make_imagefolder_datasets(INPUT_SHAPE, root)
    return build_imagefolder_resume_metadata(spec, splits)


def test_require_compatible_imagefolder_resume_metadata_accepts_identical_metadata(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    metadata = _build_metadata(tmp_path)

    require_compatible_imagefolder_resume_metadata(metadata, metadata)  # raise 없이 통과해야 함


def test_require_compatible_imagefolder_resume_metadata_rejects_model_spec_change(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    saved = _build_metadata(tmp_path)
    changed_spec = _dropout_classifier_spec()
    changed_spec.layers[3] = DropoutSpec(p=0.9)
    current = _build_metadata(tmp_path, spec=changed_spec)

    with pytest.raises(ValueError, match="model_spec_hash"):
        require_compatible_imagefolder_resume_metadata(saved, current)


def test_require_compatible_imagefolder_resume_metadata_rejects_class_to_idx_change(tmp_path: Path) -> None:
    saved_root = tmp_path / "saved"
    current_root = tmp_path / "current"
    _make_split(saved_root, "train", {"cat": 4, "dog": 4})
    _make_split(saved_root, "val", {"cat": 4, "dog": 4})
    _make_split(saved_root, "test", {"cat": 4, "dog": 4})
    # class 이름을 바꿔 class_to_idx 배정이 달라지도록 함 (dog -> zebra, 알파벳 순 index 변경)
    _make_split(current_root, "train", {"cat": 4, "zebra": 4})
    _make_split(current_root, "val", {"cat": 4, "zebra": 4})
    _make_split(current_root, "test", {"cat": 4, "zebra": 4})

    saved = _build_metadata(saved_root)
    current = _build_metadata(current_root)

    with pytest.raises(ValueError):
        require_compatible_imagefolder_resume_metadata(saved, current)


def test_require_compatible_imagefolder_resume_metadata_rejects_size_change(tmp_path: Path) -> None:
    saved_root = tmp_path / "saved"
    current_root = tmp_path / "current"
    _make_standard_dataset(saved_root)
    for split in ("train", "val", "test"):
        _make_split(current_root, split, {"cat": 5, "dog": 4})  # train/val/test 전부 cat 1장 추가

    saved = _build_metadata(saved_root)
    current = _build_metadata(current_root)

    with pytest.raises(ValueError, match="train_size"):
        require_compatible_imagefolder_resume_metadata(saved, current)


def test_require_compatible_imagefolder_resume_metadata_rejects_file_content_swap(tmp_path: Path) -> None:
    """개수/클래스는 그대로지만 파일 하나를 rename해서 상대경로가 달라진
    경우(예: 0.png <-> renamed.png) train_files_hash가 달라져야 한다."""
    saved_root = tmp_path / "saved"
    current_root = tmp_path / "current"
    _make_standard_dataset(saved_root)
    _make_standard_dataset(current_root)
    (current_root / "train" / "cat" / "0.png").rename(current_root / "train" / "cat" / "renamed.png")

    saved = _build_metadata(saved_root)
    current = _build_metadata(current_root)

    with pytest.raises(ValueError, match="train_files_hash"):
        require_compatible_imagefolder_resume_metadata(saved, current)


# -- 11-2. ImageFolder exact-resume 통합 테스트 -----------------------------------


def _assert_deep_equal(a: object, b: object, path: str = "value") -> None:
    """optimizer/scheduler state_dict처럼 텐서와 스칼라가 섞인 중첩
    dict/list를 재귀적으로 정확히 비교하는 테스트 전용 헬퍼
    (test_loop.py의 동일 헬퍼와 같은 패턴)."""
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"tensor mismatch at {path}"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"dict keys mismatch at {path}: {a.keys()} != {b.keys()}"
        for key in a:
            _assert_deep_equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"length mismatch at {path}"
        for index, (x, y) in enumerate(zip(a, b)):
            _assert_deep_equal(x, y, f"{path}[{index}]")
    else:
        assert a == b, f"value mismatch at {path}: {a!r} != {b!r}"


def _make_loaders(
    root: Path, input_shape: tuple[int, int, int], batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    splits = make_imagefolder_datasets(input_shape, root)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True
    )
    val_loader = DataLoader(splits.val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, generator


def test_imagefolder_checkpoint_resume_matches_continuous_run_exactly(tmp_path: Path) -> None:
    """핵심 계약: 연속 4 epoch 실행과, 2 epoch 실행 후 checkpoint 저장 ->
    resume으로 2 epoch를 더 실행한 결과가 model parameters/optimizer
    state/scheduler state/history/best_state_dict/epochs_without_improvement
    전부에서 정확히 일치해야 한다. Dropout이 있는 모델을 쓰므로(전역 CPU
    RNG 소비), 이 테스트는 §3-2에서 설계한 RNG 복원 순서 없이는 통과할 수
    없다. lr_scheduler="plateau"도 함께 켜서 scheduler_state_dict 왕복까지
    검증한다 (test_loop.py의 동일 계약 테스트와 같은 패턴)."""
    _make_standard_dataset(tmp_path)
    spec = _dropout_classifier_spec()
    seed = 20260802
    config_kwargs = dict(
        batch_size=4,
        learning_rate=1e-2,
        optimizer="sgd",
        momentum=0.9,
        lr_scheduler="plateau",
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
    )

    # (a) 연속 4 epoch
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = _make_loaders(tmp_path, INPUT_SHAPE, config_kwargs["batch_size"], seed)
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=4, **config_kwargs))

    # (b) 2 epoch 실행 -> checkpoint 저장
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, generator_b = _make_loaders(
        tmp_path, INPUT_SHAPE, config_kwargs["batch_size"], seed
    )
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=2, **config_kwargs)
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model_b,
        training_result=result_b1,
        training_config=first_config,
        loader_generator_state=generator_b.get_state().clone(),
        cpu_rng_state=torch.get_rng_state().clone(),
    )
    metadata_path = metadata_path_for_checkpoint(checkpoint_path)
    save_imagefolder_resume_metadata(
        build_imagefolder_resume_metadata(spec, make_imagefolder_datasets(INPUT_SHAPE, tmp_path)),
        metadata_path,
    )

    # -- resume: "새 프로세스"를 흉내 --------------------------------------------
    saved_metadata = load_imagefolder_resume_metadata(metadata_path)
    current_metadata = build_imagefolder_resume_metadata(spec, make_imagefolder_datasets(INPUT_SHAPE, tmp_path))
    require_compatible_imagefolder_resume_metadata(saved_metadata, current_metadata)

    payload = load_training_checkpoint(checkpoint_path)

    model_b2 = build_model(spec)
    model_b2.load_state_dict(payload["model_state_dict"])
    train_loader_b2_base, val_loader_b2, _ = _make_loaders(
        tmp_path, INPUT_SHAPE, config_kwargs["batch_size"], seed
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(payload["loader_generator_state"])
    train_loader_b2 = DataLoader(
        train_loader_b2_base.dataset,
        batch_size=config_kwargs["batch_size"],
        shuffle=True,
        generator=restored_generator,
        drop_last=True,
    )

    resume_state = TrainingResumeState(
        optimizer_state_dict=payload["optimizer_state_dict"],
        scheduler_state_dict=payload["scheduler_state_dict"],
        history=TrainingHistory(**payload["history"]),
        epochs_without_improvement=payload["epochs_without_improvement"],
        best_state_dict=payload["best_state_dict"],
        training_config=payload["training_config"],
    )

    resume_config = TrainingConfig(epochs=2, **config_kwargs)
    torch.set_rng_state(payload["cpu_rng_state"])
    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, resume_config, resume_state=resume_state
    )

    assert result_b2.history.train_losses == result_a.history.train_losses
    assert result_b2.history.val_losses == result_a.history.val_losses
    assert result_b2.history.val_accuracies == result_a.history.val_accuracies
    assert result_b2.history.best_epoch == result_a.history.best_epoch
    assert result_b2.history.best_val_loss == result_a.history.best_val_loss
    assert result_b2.epochs_without_improvement == result_a.epochs_without_improvement

    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_b2.state_dict()[name])
    for name, tensor in result_a.best_state_dict.items():
        assert torch.equal(tensor, result_b2.best_state_dict[name])

    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict, "optimizer")
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict, "scheduler")
