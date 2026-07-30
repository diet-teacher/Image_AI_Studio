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
from image_ai_studio.training.loop import evaluate, run_training, train_one_epoch

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

    history = run_training(model, train_loader, val_loader, config)

    assert len(history.train_losses) == config.epochs
    assert len(history.val_losses) == config.epochs
    assert len(history.val_accuracies) == config.epochs
    assert history.train_losses[-1] < history.train_losses[0]
