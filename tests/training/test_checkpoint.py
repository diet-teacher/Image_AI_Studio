"""state_dict / full training checkpoint 저장/재로드 테스트."""
from __future__ import annotations

import os
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
    _atomic_torch_save,
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

    def fake_train_one_epoch(model, loader, optimizer, device="cpu", gradient_clip_norm=None, criterion=None):
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
        {"weight_decay": 0.3},
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


# -- Phase 4L hotfix: weight_decay 누락 legacy checkpoint 파일 회귀 -----------
#
# 아래 테스트들은 require_compatible_resume_config()에 가짜 dict를 직접
# 넘기지 않는다 -- save_training_checkpoint()로 실제 파일을 저장한 뒤 그
# 파일에서 weight_decay 키를 직접 지워 Phase 4L 이전 형식을 흉내내고,
# load_training_checkpoint()로 그 파일을 다시 읽는다. 이렇게 해야
# "load_training_checkpoint()가 require_compatible_resume_config()보다
# 먼저 실행되며 RESUME_CONFIG_FIELDS 전체 존재를 요구한다"는, 가짜 dict로는
# 드러나지 않았던 실제 계층 경계(checkpoint file -> load_training_checkpoint
# -> require_compatible_resume_config)를 검증할 수 있다.


def _strip_weight_decay_from_saved_checkpoint(checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["training_config"]["weight_decay"]
    torch.save(payload, checkpoint_path)


def test_load_training_checkpoint_accepts_legacy_file_missing_weight_decay(tmp_path: Path) -> None:
    """Phase 4L 이전 checkpoint 파일에는 weight_decay 키가 없다 --
    load_training_checkpoint()는 구조적 검사만 담당하므로 이 파일을 정상
    로드해야 한다(호환성 판단은 require_compatible_resume_config()의 몫)."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, weight_decay=0.0)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    _strip_weight_decay_from_saved_checkpoint(checkpoint_path)

    payload = load_training_checkpoint(checkpoint_path)  # 실패하면 안 됨

    assert "weight_decay" not in payload["training_config"]


def test_legacy_checkpoint_file_resumes_when_current_weight_decay_is_zero(tmp_path: Path) -> None:
    """weight_decay 키가 없는 실제 checkpoint 파일 + 현재 config
    weight_decay=0.0 조합은, 실제 파일 경로를 거쳐도 resume 호환성 검사를
    통과해야 한다(Phase 4L이 원래 약속한 migration 정책)."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, weight_decay=0.0)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    _strip_weight_decay_from_saved_checkpoint(checkpoint_path)

    payload = load_training_checkpoint(checkpoint_path)
    resume_config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2, weight_decay=0.0)

    require_compatible_resume_config(payload["training_config"], resume_config)  # raise 없이 통과해야 함


def test_legacy_checkpoint_file_rejects_resume_when_current_weight_decay_is_nonzero(tmp_path: Path) -> None:
    """weight_decay 키가 없는 실제 checkpoint 파일 + 현재 config
    weight_decay>0.0 조합은 값이 실제로 달라지므로 여전히 거부돼야 한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, weight_decay=0.0)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    _strip_weight_decay_from_saved_checkpoint(checkpoint_path)

    payload = load_training_checkpoint(checkpoint_path)
    resume_config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2, weight_decay=0.1)

    with pytest.raises(ValueError, match="weight_decay"):
        require_compatible_resume_config(payload["training_config"], resume_config)


def test_checkpoint_file_missing_other_required_field_is_still_rejected_at_load(tmp_path: Path) -> None:
    """weight_decay 외의 다른 RESUME_CONFIG_FIELDS(예: momentum)가 실제
    checkpoint 파일에서 누락되면, 여전히 load_training_checkpoint() 단계에서
    즉시 거부돼야 한다 -- weight_decay 예외를 다른 필드로 일반화하지
    않았음을 확인한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["training_config"]["momentum"]
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=r"missing required field\(s\).*momentum"):
        load_training_checkpoint(checkpoint_path)


def test_checkpoint_file_with_weight_decay_present_still_checks_it_normally(tmp_path: Path) -> None:
    """Phase 4L 이후 정상 checkpoint(weight_decay 키가 실제로 존재)는 기존과
    동일하게 저장값과 현재 config를 정확히 비교해야 한다 -- legacy 예외는
    이 경우 전혀 개입하지 않는다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, weight_decay=0.25)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    payload = load_training_checkpoint(checkpoint_path)

    assert payload["training_config"]["weight_decay"] == 0.25

    mismatched_resume_config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2, weight_decay=0.5)
    with pytest.raises(ValueError, match="weight_decay"):
        require_compatible_resume_config(payload["training_config"], mismatched_resume_config)

    matching_resume_config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2, weight_decay=0.25)
    require_compatible_resume_config(payload["training_config"], matching_resume_config)  # raise 없이 통과해야 함


# -- Phase 4P: class_weights -- legacy checkpoint(키 자체가 없음) 회귀 --------
#
# class_weights는 weight_decay와 달리 RESUME_CONFIG_FIELDS에 아예 들어있지
# 않으므로(RESUME_CONFIG_LEGACY_DEFAULTS에 넣을 필요 자체가 없음), Phase 4L의
# "예외를 두지 않으면 legacy checkpoint가 구조 검증 단계에서부터 거부된다"는
# 문제가 애초에 발생할 수 없다는 것을 실제 checkpoint 파일 경로로 직접
# 확인한다 -- weight_decay 회귀 테스트와 동일한 패턴(가짜 dict가 아니라 실제
# 저장된 파일에서 키를 지운다).


def _strip_class_weights_from_saved_checkpoint(checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["training_config"]["class_weights"]
    torch.save(payload, checkpoint_path)


def test_load_training_checkpoint_accepts_legacy_file_missing_class_weights(tmp_path: Path) -> None:
    """Phase 4P 이전 checkpoint 파일에는 class_weights 키가 아예 없다 --
    class_weights는 RESUME_CONFIG_FIELDS에 없으므로 load_training_checkpoint()의
    구조적 필수 필드 목록에도 없다. 이 파일을 정상 로드해야 한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    _strip_class_weights_from_saved_checkpoint(checkpoint_path)

    payload = load_training_checkpoint(checkpoint_path)  # 실패하면 안 됨

    assert "class_weights" not in payload["training_config"]


def test_legacy_checkpoint_file_missing_class_weights_resumes_regardless_of_new_value(tmp_path: Path) -> None:
    """class_weights 키가 없는 실제 checkpoint 파일이더라도, 새 resume
    config의 class_weights 값(None이든 tuple이든)과 무관하게 resume 호환성
    검사를 통과해야 한다 -- class_weights는 애초에 비교 대상(RESUME_CONFIG_FIELDS)이
    아니므로 weight_decay 같은 legacy-default 예외 자체가 필요 없다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    _strip_class_weights_from_saved_checkpoint(checkpoint_path)
    payload = load_training_checkpoint(checkpoint_path)

    resume_config_without_weights = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)
    require_compatible_resume_config(payload["training_config"], resume_config_without_weights)

    resume_config_with_weights = TrainingConfig(
        epochs=5, batch_size=8, learning_rate=1e-2, class_weights=(1.0, 2.0, 0.5, 3.0)
    )
    require_compatible_resume_config(payload["training_config"], resume_config_with_weights)


def test_checkpoint_round_trips_class_weights_when_set(tmp_path: Path) -> None:
    """class_weights가 설정된 정상 checkpoint는 저장된 값을 그대로
    round-trip해야 한다(단순 asdict() 자동 반영 확인, 별도 직렬화 로직
    없음)."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, class_weights=(1.0, 2.0, 0.5, 3.0))
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = load_training_checkpoint(checkpoint_path)

    assert payload["training_config"]["class_weights"] == (1.0, 2.0, 0.5, 3.0)


# -- Phase 4R: cuda_rng_state ---------------------------------------------------
#
# cuda_rng_state는 cpu_rng_state/loader_generator_state와 대칭적인 순수 실행
# state다(training config가 아님) -- GPU 없이도 CPU tensor로 직렬화 round-trip을
# 전부 검증할 수 있다(GPU 필요한 부분은 torch.cuda.get_rng_state()/
# set_rng_state()를 실제로 호출하는 부분뿐이고, checkpoint.py는 그 호출을
# 전혀 하지 않는다 -- caller가 채취한 값을 그대로 저장/반환만 한다).


def test_checkpoint_round_trips_cuda_rng_state_none(tmp_path: Path) -> None:
    """cuda_rng_state를 생략하면(기존 CPU checkpoint 호출부와 동일) 기본값
    None이 그대로 저장/로드된다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)  # cuda_rng_state 생략

    payload = load_training_checkpoint(checkpoint_path)

    assert payload["cuda_rng_state"] is None


def test_checkpoint_round_trips_cuda_rng_state_tensor(tmp_path: Path) -> None:
    """실제 CUDA 없이도(GPU 없는 CI에서도) cuda_rng_state가 진짜
    torch.cuda.get_rng_state()가 반환하는 것과 같은 형태(uint8 1D
    Tensor)의 가짜 값으로 정확히 round-trip됨을 확인한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    torch.manual_seed(0)
    spec = _mlp_spec()
    model = build_model(spec)
    train_loader, val_loader, generator = _make_loaders(seed=0)
    result = run_training(model, train_loader, val_loader, config)

    fake_cuda_rng_state = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.uint8)
    checkpoint_path = tmp_path / "checkpoint_with_cuda_rng.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        training_result=result,
        training_config=config,
        loader_generator_state=generator.get_state(),
        cpu_rng_state=torch.get_rng_state(),
        cuda_rng_state=fake_cuda_rng_state,
    )

    payload = load_training_checkpoint(checkpoint_path)

    assert torch.equal(payload["cuda_rng_state"], fake_cuda_rng_state)


def _strip_cuda_rng_state_from_saved_checkpoint(checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["cuda_rng_state"]
    torch.save(payload, checkpoint_path)


def test_load_training_checkpoint_accepts_legacy_file_missing_cuda_rng_state(tmp_path: Path) -> None:
    """Phase 4R 이전 checkpoint 파일에는 cuda_rng_state 키가 아예 없다 --
    cuda_rng_state는 structural 필수 key가 아니므로 이 파일을 정상
    로드해야 하고, payload에 없는 키는 그냥 없는 채로 반환되어야 한다
    (checkpoint.py가 임의로 None을 채워 넣지 않는다 -- 그건 caller,
    즉 imagefolder_workflow.py의 payload.get() 책임이다)."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)
    _strip_cuda_rng_state_from_saved_checkpoint(checkpoint_path)

    payload = load_training_checkpoint(checkpoint_path)  # 실패하면 안 됨

    assert "cuda_rng_state" not in payload


def test_new_checkpoint_writer_always_writes_cuda_rng_state_key(tmp_path: Path) -> None:
    """새 writer는 CPU checkpoint에도 cuda_rng_state key를 명시적으로
    쓴다(값은 None) -- "legacy라 키가 아예 없음"과 "새 checkpoint인데
    CPU라서 값이 None임"을 명확히 구분할 수 있다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    raw_payload = torch.load(checkpoint_path, weights_only=True)

    assert "cuda_rng_state" in raw_payload
    assert raw_payload["cuda_rng_state"] is None


def test_load_training_checkpoint_rejects_invalid_cuda_rng_state_type(tmp_path: Path) -> None:
    """cuda_rng_state가 존재하는데 None도 torch.Tensor도 아니면 명확한
    ValueError로 거부한다. shape/dtype처럼 PyTorch 버전에 따라 달라질 수
    있는 값까지는 검증하지 않는다(cpu_rng_state/loader_generator_state와
    동일한 최소 검증 철학)."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, _, _ = _run_and_save_checkpoint(tmp_path, config)

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["cuda_rng_state"] = "not-a-tensor"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="cuda_rng_state"):
        load_training_checkpoint(checkpoint_path)


# -- Phase 4J: atomic save -----------------------------------------------------


def test_atomic_torch_save_creates_parent_directories(tmp_path: Path) -> None:
    nested_path = tmp_path / "nested" / "dir" / "payload.pt"
    _atomic_torch_save({"value": 1}, nested_path)

    assert nested_path.exists()
    assert torch.load(nested_path, weights_only=True) == {"value": 1}


def test_atomic_torch_save_no_leftover_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "payload.pt"
    _atomic_torch_save({"value": 1}, path)

    assert list(tmp_path.iterdir()) == [path]


def test_atomic_torch_save_failure_before_replace_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torch.save() 자체가 실패하면(os.replace() 이전) 기존 파일은 전혀
    바뀌지 않고, 임시 파일도 남지 않아야 한다."""
    path = tmp_path / "payload.pt"
    _atomic_torch_save({"value": "original"}, path)
    original_bytes = path.read_bytes()

    def failing_torch_save(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("image_ai_studio.training.checkpoint.torch.save", failing_torch_save)

    with pytest.raises(RuntimeError, match="disk full"):
        _atomic_torch_save({"value": "new"}, path)

    assert path.read_bytes() == original_bytes  # 기존 파일 보존
    assert list(tmp_path.iterdir()) == [path]  # 임시 파일 미잔존


def test_atomic_torch_save_os_replace_failure_propagates_and_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.replace() 자체가 실패해도(예: 권한 문제) 예외가 재시도/폴백 없이
    그대로 전파되고, 기존 파일은 보존되어야 한다."""
    path = tmp_path / "payload.pt"
    _atomic_torch_save({"value": "original"}, path)
    original_bytes = path.read_bytes()

    def failing_replace(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("image_ai_studio.training.checkpoint.os.replace", failing_replace)

    with pytest.raises(OSError, match="permission denied"):
        _atomic_torch_save({"value": "new"}, path)

    assert path.read_bytes() == original_bytes  # 기존 파일 보존
    assert list(tmp_path.iterdir()) == [path]  # 임시 파일 미잔존


def test_atomic_torch_save_cleanup_failure_does_not_mask_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """임시 파일 정리(unlink) 자체가 실패해도, 사용자에게 보이는 예외는
    원래 저장 실패 예외여야 한다(정리 실패가 원래 예외를 가리면 안 됨)."""
    path = tmp_path / "payload.pt"

    def failing_torch_save(*args, **kwargs):
        raise RuntimeError("original failure")

    def failing_unlink(self, *args, **kwargs):
        raise OSError("cleanup also failed")

    monkeypatch.setattr("image_ai_studio.training.checkpoint.torch.save", failing_torch_save)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(RuntimeError, match="original failure"):
        _atomic_torch_save({"value": "new"}, path)


def test_save_training_checkpoint_atomic_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_training_checkpoint()가 내부적으로 원자적 저장을 쓰므로,
    저장 도중 실패해도 기존 checkpoint 파일이 손상되지 않아야 한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)
    checkpoint_path, result, model = _run_and_save_checkpoint(tmp_path, config)
    original_bytes = checkpoint_path.read_bytes()

    def failing_torch_save(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("image_ai_studio.training.checkpoint.torch.save", failing_torch_save)

    with pytest.raises(RuntimeError, match="boom"):
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            training_result=result,
            training_config=config,
            loader_generator_state=torch.get_rng_state(),
            cpu_rng_state=torch.get_rng_state(),
        )

    assert checkpoint_path.read_bytes() == original_bytes
    assert {p.name for p in tmp_path.iterdir()} == {checkpoint_path.name}


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
