"""state_dict / full training checkpoint 저장/재로드 테스트."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import (
    BatchNorm2dSpec,
    Conv2dSpec,
    FlattenSpec,
    LinearSpec,
    ModelSpec,
    ReLUSpec,
)
from image_ai_studio.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_state_dict,
    load_training_checkpoint,
    require_compatible_resume_config,
    save_state_dict,
    save_training_checkpoint,
)
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.dataset import make_train_val_datasets
from image_ai_studio.training.loop import TrainingHistory, TrainingResult, run_training


def _spec() -> ModelSpec:
    return ModelSpec(
        name="checkpoint_model",
        input_shape=(3, 8, 8),
        layers=[
            Conv2dSpec(out_channels=4, kernel_size=3, padding=1),
            BatchNorm2dSpec(),
            ReLUSpec(),
            FlattenSpec(),
            LinearSpec(out_features=4),
        ],
    )


def test_save_and_load_state_dict_reproduces_same_output(tmp_path: Path) -> None:
    torch.manual_seed(0)
    spec = _spec()
    original_model = build_model(spec).eval()

    example_input = torch.randn(2, *spec.input_shape)
    with torch.inference_mode():
        original_output = original_model(example_input)

    state_dict_path = tmp_path / "model_state_dict.pt"
    save_state_dict(original_model, state_dict_path)
    assert state_dict_path.exists()

    torch.manual_seed(999)  # 새 모델은 일부러 다른 초기값에서 시작
    new_model = build_model(spec).eval()
    with torch.inference_mode():
        new_output_before_load = new_model(example_input)
    # load 전에는 서로 다른 가중치라 출력도 달라야 함 (load가 실제로 뭔가
    # 바꾼다는 것을 증명하기 위한 대조군)
    assert not torch.allclose(original_output, new_output_before_load)

    load_state_dict(new_model, state_dict_path)
    with torch.inference_mode():
        reloaded_output = new_model(example_input)

    assert torch.allclose(original_output, reloaded_output)


def test_save_state_dict_creates_parent_directories(tmp_path: Path) -> None:
    model = build_model(_spec())
    nested_path = tmp_path / "nested" / "dir" / "model.pt"
    save_state_dict(model, nested_path)
    assert nested_path.exists()


# -- Phase 4F: full training checkpoint ---------------------------------------

NUM_CLASSES = 4
INPUT_SHAPE = (3, 8, 8)


def _mlp_spec() -> ModelSpec:
    return ModelSpec(
        name="checkpoint_resume_model",
        input_shape=INPUT_SHAPE,
        layers=[
            FlattenSpec(),
            LinearSpec(out_features=16),
            ReLUSpec(),
            LinearSpec(out_features=NUM_CLASSES),
        ],
    )


def _make_loaders(seed: int, batch_size: int = 8) -> tuple[DataLoader, DataLoader, torch.Generator]:
    train_dataset, val_dataset = make_train_val_datasets(
        INPUT_SHAPE, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, generator


def _run_and_save_checkpoint(
    tmp_path: Path, config: TrainingConfig, seed: int = 0
) -> tuple[Path, TrainingResult, torch.nn.Module]:
    torch.manual_seed(seed)
    spec = _mlp_spec()
    model = build_model(spec)
    train_loader, val_loader, generator = _make_loaders(seed)

    result = run_training(model, train_loader, val_loader, config)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        training_result=result,
        training_config=config,
        loader_generator_state=generator.get_state(),
        cpu_rng_state=torch.get_rng_state(),
    )
    return checkpoint_path, result, model


def test_save_and_load_training_checkpoint_round_trips(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    checkpoint_path, result, model = _run_and_save_checkpoint(tmp_path, config)

    payload = load_training_checkpoint(checkpoint_path)

    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["history"]["train_losses"] == result.history.train_losses
    assert payload["epochs_without_improvement"] == result.epochs_without_improvement
    assert payload["training_config"]["batch_size"] == config.batch_size
    for name, tensor in model.state_dict().items():
        assert torch.equal(payload["model_state_dict"][name], tensor)


def test_load_training_checkpoint_rejects_missing_format_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}}, path)

    with pytest.raises(ValueError, match="format_version"):
        load_training_checkpoint(path)


def test_load_training_checkpoint_rejects_unsupported_format_version(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["format_version"] = 999
    bad_path = tmp_path / "bad_version.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="unsupported checkpoint format_version"):
        load_training_checkpoint(bad_path)


def test_load_training_checkpoint_rejects_bare_state_dict(tmp_path: Path) -> None:
    model = build_model(_mlp_spec())
    path = tmp_path / "bare.pt"
    save_state_dict(model, path)

    with pytest.raises(ValueError, match="does not look like a full training checkpoint"):
        load_training_checkpoint(path)


def test_load_state_dict_rejects_full_training_checkpoint(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    model = build_model(_mlp_spec())
    with pytest.raises(ValueError, match="full training checkpoint"):
        load_state_dict(model, checkpoint_path)


def test_load_training_checkpoint_rejects_mismatched_history_lengths(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["history"]["val_losses"] = payload["history"]["val_losses"] + [0.5]
    bad_path = tmp_path / "bad_history.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="mismatched lengths"):
        load_training_checkpoint(bad_path)


def test_load_training_checkpoint_allows_stopped_early_for_weight_extraction(tmp_path: Path) -> None:
    """checkpoint 파일 자체가 stopped_early=True여도 load_training_checkpoint()
    는 거부하지 않고 payload를 그대로 반환해야 한다 -- "이 파일을 조회/
    가중치 추출할 수 있는가"와 "이 payload로 resume을 실행할 수 있는가"는
    별개다. 후자를 거부하는 것은 TrainingResumeState/run_training()의
    책임이다(tests/training/test_loop.py 참고). 여기서는 stopped_early인
    checkpoint에서도 model_state_dict/best_state_dict를 공식 API로 정상
    추출할 수 있음을 확인한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["history"]["stopped_early"] = True
    stopped_path = tmp_path / "stopped_early.pt"
    torch.save(payload, stopped_path)

    loaded = load_training_checkpoint(stopped_path)  # raise 없이 통과해야 함

    assert loaded["history"]["stopped_early"] is True
    fresh_model = build_model(_mlp_spec())
    fresh_model.load_state_dict(loaded["best_state_dict"])  # 공식 API로 가중치 추출 가능


def test_load_training_checkpoint_accepts_legacy_history_without_stopped_by_user(tmp_path: Path) -> None:
    """Phase 4H까지 저장된 checkpoint의 history payload에는 stopped_by_user
    키가 없다. load_training_checkpoint()의 구조 검증(_REQUIRED_HISTORY_FIELDS)
    은 그 키를 요구하지 않으므로(checkpoint.py 무수정) 여전히 통과해야
    하고, TrainingHistory(**payload["history"])로 복원하면 dataclass
    기본값(False)이 채워져야 한다 -- history.py의 legacy history.json
    하위 호환(tests/training/test_history.py)과 동일한 메커니즘을
    checkpoint 경로에서도 확인한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    assert "stopped_by_user" in payload["history"]  # 현재 저장 형식에는 있음을 먼저 확인
    del payload["history"]["stopped_by_user"]  # Phase 4H까지의 실제 저장 형식을 흉내
    legacy_path = tmp_path / "legacy_no_stopped_by_user.pt"
    torch.save(payload, legacy_path)

    loaded = load_training_checkpoint(legacy_path)  # 구조 검증 통과 (필수 필드 목록에 없음)

    restored_history = TrainingHistory(**loaded["history"])
    assert restored_history.stopped_by_user is False


def test_load_training_checkpoint_rejects_non_dict_history(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["history"] = None
    bad_path = tmp_path / "history_none.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="'history' must be a dict"):
        load_training_checkpoint(bad_path)


def test_load_training_checkpoint_rejects_history_missing_field(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["history"]["val_accuracies"]
    bad_path = tmp_path / "history_missing_field.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="'history' is missing required field"):
        load_training_checkpoint(bad_path)


def test_load_training_checkpoint_rejects_non_dict_training_config(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["training_config"] = None
    bad_path = tmp_path / "config_none.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="'training_config' must be a dict"):
        load_training_checkpoint(bad_path)


def test_load_training_checkpoint_rejects_training_config_missing_field(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["training_config"]["momentum"]
    bad_path = tmp_path / "config_missing_field.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="'training_config' is missing required field"):
        load_training_checkpoint(bad_path)


def test_load_training_checkpoint_rejects_negative_epochs_without_improvement(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["epochs_without_improvement"] = -1
    bad_path = tmp_path / "negative_counter.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="'epochs_without_improvement' must be a non-negative integer"):
        load_training_checkpoint(bad_path)


@pytest.mark.parametrize("field_name", ["loader_generator_state", "cpu_rng_state"])
def test_load_training_checkpoint_rejects_non_tensor_rng_fields(tmp_path: Path, field_name: str) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload[field_name] = [1, 2, 3]  # 텐서가 아님
    bad_path = tmp_path / f"{field_name}_not_tensor.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match=f"'{field_name}' must be a torch.Tensor"):
        load_training_checkpoint(bad_path)


def test_load_training_checkpoint_rejects_scheduler_inconsistency(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, lr_scheduler="plateau")
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["scheduler_state_dict"] = None  # training_config는 여전히 lr_scheduler="plateau"
    bad_path = tmp_path / "inconsistent_scheduler.pt"
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match="internally inconsistent"):
        load_training_checkpoint(bad_path)


def test_loader_generator_state_and_cpu_rng_state_round_trip_exactly(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    torch.manual_seed(0)
    spec = _mlp_spec()
    model = build_model(spec)
    train_loader, val_loader, generator = _make_loaders(seed=0)
    result = run_training(model, train_loader, val_loader, config)

    loader_state = generator.get_state()
    cpu_state = torch.get_rng_state()

    checkpoint_path = tmp_path / "rng.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        training_result=result,
        training_config=config,
        loader_generator_state=loader_state,
        cpu_rng_state=cpu_state,
    )

    payload = load_training_checkpoint(checkpoint_path)

    assert torch.equal(payload["loader_generator_state"], loader_state)
    assert torch.equal(payload["cpu_rng_state"], cpu_state)


def test_checkpoint_distinguishes_current_model_from_best_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """best epoch가 마지막 epoch가 아닌 상황에서, checkpoint의
    model_state_dict(현재/마지막 epoch)와 best_state_dict(과거 best epoch)가
    서로 다른 값을 보존해야 한다 -- 실수로 best_state_dict를 로드한 model을
    save_training_checkpoint()에 넘기면 이 둘이 같아져 버그가 된다."""
    torch.manual_seed(0)
    spec = _mlp_spec()
    model = build_model(spec)
    train_loader, val_loader, generator = _make_loaders(seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    call_count = {"value": 0}

    def fake_train_one_epoch(model, loader, optimizer, device="cpu"):
        call_count["value"] += 1
        epoch_value = float(call_count["value"])
        for param in model.parameters():
            param.data.fill_(epoch_value)
        return epoch_value

    # epoch 1: 1.0(best) / epoch 2: 0.5(best) / epoch 3: 0.8(개선 없음) -> best_epoch=2, 마지막 epoch=3
    fixed_val_results = iter([(1.0, 1.0), (0.5, 1.0), (0.8, 1.0)])
    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)
    assert result.history.best_epoch == 2  # 마지막 epoch(3)이 아님을 먼저 확인

    checkpoint_path = tmp_path / "current_vs_best.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,  # run_training()에 실제로 쓰인 "현재"(epoch 3) model
        training_result=result,
        training_config=config,
        loader_generator_state=generator.get_state(),
        cpu_rng_state=torch.get_rng_state(),
    )

    payload = load_training_checkpoint(checkpoint_path)

    for tensor in payload["model_state_dict"].values():
        assert torch.all(tensor == 3.0)  # 마지막으로 완료된 epoch
    for tensor in payload["best_state_dict"].values():
        assert torch.all(tensor == 2.0)  # best epoch(2)의 snapshot

    # 핵심 주장: 이 둘은 서로 다른 텐서 값이어야 한다
    any_key = next(iter(payload["model_state_dict"]))
    assert not torch.equal(payload["model_state_dict"][any_key], payload["best_state_dict"][any_key])


def test_require_compatible_resume_config_passes_when_fields_match(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.8)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    payload = load_training_checkpoint(checkpoint_path)

    resume_config = TrainingConfig(epochs=99, batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.8)
    require_compatible_resume_config(payload["training_config"], resume_config)  # raise 없이 통과해야 함


def test_require_compatible_resume_config_allows_epochs_and_early_stopping_to_differ(tmp_path: Path) -> None:
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    payload = load_training_checkpoint(checkpoint_path)

    resume_config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=5)
    require_compatible_resume_config(payload["training_config"], resume_config)  # raise 없이 통과해야 함


@pytest.mark.parametrize(
    "override",
    [
        {"optimizer": "sgd"},
        {"learning_rate": 1e-4},
        {"momentum": 0.5},
        {"batch_size": 4},
        {"lr_scheduler": "plateau"},
    ],
)
def test_require_compatible_resume_config_rejects_mismatched_fields(
    tmp_path: Path, override: dict
) -> None:
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    payload = load_training_checkpoint(checkpoint_path)

    base_kwargs = dict(epochs=3, batch_size=8, learning_rate=1e-2)
    base_kwargs.update(override)
    mismatched_config = TrainingConfig(**base_kwargs)

    with pytest.raises(ValueError, match="cannot resume"):
        require_compatible_resume_config(payload["training_config"], mismatched_config)


@pytest.mark.parametrize(
    "override",
    [
        {"lr_scheduler": None},
        {"lr_scheduler_factor": 0.5},
        {"lr_scheduler_patience": 2},
    ],
)
def test_require_compatible_resume_config_rejects_mismatched_scheduler_fields(
    tmp_path: Path, override: dict
) -> None:
    """factor/patience mismatch는 scheduler가 실제로 켜진 checkpoint를
    기준으로 확인해야 의미가 분명하다 (scheduler가 꺼져 있으면 이 값들이
    실제로 쓰이지 않으므로)."""
    scheduler_config = TrainingConfig(
        epochs=3, batch_size=8, learning_rate=1e-2,
        lr_scheduler="plateau", lr_scheduler_factor=0.1, lr_scheduler_patience=1,
    )
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, scheduler_config)
    payload = load_training_checkpoint(checkpoint_path)

    base_kwargs = dict(
        epochs=3, batch_size=8, learning_rate=1e-2,
        lr_scheduler="plateau", lr_scheduler_factor=0.1, lr_scheduler_patience=1,
    )
    base_kwargs.update(override)
    mismatched_config = TrainingConfig(**base_kwargs)

    with pytest.raises(ValueError, match="cannot resume"):
        require_compatible_resume_config(payload["training_config"], mismatched_config)
