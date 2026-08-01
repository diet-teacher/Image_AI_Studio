"""train_one_epoch / evaluate / run_training 테스트.

여기서는 학습 루프 메커니즘 자체(파라미터 변화, loss 감소, evaluate의 부작용
없음)만 검증한다. BatchNorm/Dropout/ResidualBlockSpec/BranchSpec 각각의
학습 동작 검증은 test_model_definition_integration.py에서 다룬다.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.dataset import make_train_val_datasets
from image_ai_studio.training.loop import (
    TrainingHistory,
    TrainingResult,
    _build_optimizer,
    _build_scheduler,
    evaluate,
    run_training,
    train_one_epoch,
)

NUM_CLASSES = 4
INPUT_SHAPE = (3, 8, 8)


def _mlp_classifier_spec() -> ModelSpec:
    """분류가 쉬운 합성 데이터셋에 맞춘 최소 MLP -- loss 감소 테스트가 흔들리지
    않도록 일부러 단순하게 구성 (Conv/BatchNorm 등 조합 테스트는 별도 파일)."""
    return ModelSpec(
        name="mlp_classifier",
        input_shape=INPUT_SHAPE,
        layers=[
            FlattenSpec(),
            LinearSpec(out_features=16),
            ReLUSpec(),
            LinearSpec(out_features=NUM_CLASSES),
        ],
    )


def _make_loaders(spec: ModelSpec, seed: int, batch_size: int = 8) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_train_one_epoch_changes_parameters() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, _ = _make_loaders(spec, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = copy.deepcopy(model.state_dict())
    train_one_epoch(model, train_loader, optimizer)
    after = model.state_dict()

    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_evaluate_does_not_change_parameters() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    before = copy.deepcopy(model.state_dict())
    evaluate(model, val_loader)
    after = model.state_dict()

    assert all(torch.equal(before[name], after[name]) for name in before)


def test_evaluate_returns_loss_and_accuracy_in_valid_range() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    loss, accuracy = evaluate(model, val_loader)
    assert loss > 0.0
    assert 0.0 <= accuracy <= 1.0


def test_train_one_epoch_raises_on_empty_loader() -> None:
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, *spec.input_shape), torch.empty(0, dtype=torch.long)), batch_size=8
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    with pytest.raises(ValueError, match="empty DataLoader"):
        train_one_epoch(model, empty_loader, optimizer)


def test_evaluate_raises_on_empty_loader() -> None:
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, *spec.input_shape), torch.empty(0, dtype=torch.long)), batch_size=8
    )

    with pytest.raises(ValueError, match="empty DataLoader"):
        evaluate(model, empty_loader)


def test_run_training_reduces_training_loss() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=8, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config)
    history = result.history

    assert len(history.train_losses) == config.epochs
    assert len(history.val_losses) == config.epochs
    assert len(history.val_accuracies) == config.epochs
    assert history.train_losses[-1] < history.train_losses[0]


# -- Phase 4B: best epoch 추적 --------------------------------------------------


def test_run_training_returns_training_result_with_history_and_best_state_dict() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config)

    assert isinstance(result, TrainingResult)
    assert isinstance(result.history, TrainingHistory)
    assert set(result.best_state_dict.keys()) == set(model.state_dict().keys())


def test_run_training_best_epoch_matches_min_val_loss() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=6, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config)
    history = result.history

    assert history.best_val_loss == min(history.val_losses)
    # best_epoch은 1-indexed이므로 val_losses 인덱스는 -1
    assert history.val_losses[history.best_epoch - 1] == history.best_val_loss
    assert 1 <= history.best_epoch <= config.epochs


def test_run_training_tie_keeps_first_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """val_loss가 동일하면(strict `<` 비교이므로 자연히) 먼저 나온 epoch이 유지된다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    # epoch 1: 0.5 (best) / epoch 2: 0.5 (동률, 갱신 안 됨) / epoch 3: 0.6 (더 나쁨)
    fixed_val_results = iter([(0.5, 1.0), (0.5, 1.0), (0.6, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)

    assert result.history.best_epoch == 1
    assert result.history.best_val_loss == 0.5


def test_run_training_first_epoch_fallback_when_never_improves(monkeypatch: pytest.MonkeyPatch) -> None:
    """validation loss가 첫 epoch 이후 한 번도 개선되지 않아도 best_epoch==1,
    best_state_dict가 채워져 있어야 한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    # epoch을 거듭할수록 val_loss가 계속 나빠지는 시나리오
    fixed_val_results = iter([(0.1, 1.0), (0.5, 1.0), (0.9, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)

    assert result.history.best_epoch == 1
    assert result.history.best_val_loss == 0.1
    assert result.best_state_dict is not None
    assert len(result.best_state_dict) > 0


def test_run_training_best_state_dict_is_snapshot_not_reference() -> None:
    """best epoch 이후 model이 더 학습되어도 보관된 best_state_dict는
    바뀌지 않아야 한다 (deep copy snapshot, 참조 아님)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config)
    snapshot_before = {name: tensor.clone() for name, tensor in result.best_state_dict.items()}

    # best epoch 이후에도 동일한 model 객체를 계속 학습시켜본다
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        train_one_epoch(model, train_loader, optimizer)

    assert all(
        torch.equal(snapshot_before[name], result.best_state_dict[name]) for name in snapshot_before
    )


# -- Phase 4E: _build_optimizer / _build_scheduler ---------------------------


def test_build_optimizer_default_config_returns_adam() -> None:
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3)

    optimizer = _build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 1e-3


def test_build_optimizer_sgd_config_returns_sgd_with_matching_momentum() -> None:
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=5e-3, optimizer="sgd", momentum=0.77)

    optimizer = _build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 5e-3
    assert optimizer.param_groups[0]["momentum"] == 0.77


def test_build_scheduler_returns_none_when_not_configured() -> None:
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3)
    optimizer = _build_optimizer(model, config)

    assert _build_scheduler(optimizer, config) is None


def test_build_scheduler_plateau_matches_config_factor_and_patience() -> None:
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-3, lr_scheduler="plateau", lr_scheduler_factor=0.25, lr_scheduler_patience=3
    )
    optimizer = _build_optimizer(model, config)

    scheduler = _build_scheduler(optimizer, config)

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.factor == 0.25
    assert scheduler.patience == 3


def test_build_scheduler_reduces_lr_after_patience_bad_steps() -> None:
    """ReduceLROnPlateau의 실제 동작을 그대로 고정한다: 첫 step()이 기준값을
    세우고, 그 뒤로 patience번 연속 개선이 없어야(= patience+1번째 step) LR이
    줄어든다. patience=2, 동일한 loss를 계속 넣으면 4번째 step()에서 처음
    LR이 바뀐다 (실제로 PyTorch로 직접 실행해 확인한 호출 순번)."""
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1.0, lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=2
    )
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    lrs_after_each_step = []
    for _ in range(4):
        scheduler.step(1.0)  # 개선 없는 동일한 loss를 계속 전달
        lrs_after_each_step.append(optimizer.param_groups[0]["lr"])

    assert lrs_after_each_step == [1.0, 1.0, 1.0, 0.5]


# -- Phase 4E: run_training()의 optimizer 선택 -------------------------------


def test_run_training_with_sgd_optimizer_reduces_training_loss() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=8, batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.9)

    result = run_training(model, train_loader, val_loader, config)

    assert result.history.train_losses[-1] < result.history.train_losses[0]


# -- Phase 4E: early stopping --------------------------------------------------


def test_run_training_early_stopping_disabled_runs_all_epochs() -> None:
    """early_stopping_patience=None(기본값)이면 지금까지와 동일하게
    config.epochs를 전부 실행하고 stopped_early는 False여야 한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config)

    assert len(result.history.train_losses) == config.epochs
    assert result.history.stopped_early is False


def test_run_training_stops_exactly_after_patience_non_improving_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patience=2, val_loss 시퀀스 [1.0, 1.0, 1.0, 0.5]:
    epoch 1 -> best(1.0), 카운터 0
    epoch 2 -> 개선 없음, 카운터 1
    epoch 3 -> 개선 없음, 카운터 2 == patience -> epoch 3 종료 후 중단
    (epoch 4의 0.5는 절대 실행되지 않아야 한다 -- off-by-one 방지)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=2)

    fixed_val_results = iter([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.5, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)
    history = result.history

    assert len(history.train_losses) == 3
    assert history.stopped_early is True
    assert history.best_epoch == 1
    assert history.best_val_loss == 1.0
    assert result.best_state_dict is not None
    assert len(result.best_state_dict) > 0


def test_run_training_early_stopping_tie_is_not_improvement(monkeypatch: pytest.MonkeyPatch) -> None:
    """동률(val_loss가 기존 best와 같음)은 개선으로 취급하지 않는다 --
    기존 best epoch 선택 계약(strict `<`)과 동일해야 한다. patience=1,
    시퀀스 [1.0, 1.0]이면 epoch 2에서 카운터가 1 == patience가 되어
    epoch 2 종료 직후 중단된다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=1)

    fixed_val_results = iter([(1.0, 1.0), (1.0, 1.0), (0.1, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)
    history = result.history

    assert len(history.train_losses) == 2
    assert history.stopped_early is True
    assert history.best_epoch == 1
    assert history.best_val_loss == 1.0


def test_run_training_early_stopping_patience_one_stops_after_first_non_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patience=1의 최소 경계: 개선이 한 번만 없어도(2 epoch째) 즉시 중단."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=1)

    fixed_val_results = iter([(0.9, 1.0), (1.0, 1.0), (0.1, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)

    assert len(result.history.train_losses) == 2
    assert result.history.stopped_early is True
