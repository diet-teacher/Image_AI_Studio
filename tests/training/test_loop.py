"""train_one_epoch / evaluate / run_training 테스트.

여기서는 학습 루프 메커니즘 자체(파라미터 변화, loss 감소, evaluate의 부작용
없음)만 검증한다. BatchNorm/Dropout/ResidualBlockSpec/BranchSpec 각각의
학습 동작 검증은 test_model_definition_integration.py에서 다룬다.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.specs import DropoutSpec, FlattenSpec, LinearSpec, ModelSpec, ReLUSpec
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.dataset import make_train_val_datasets
from image_ai_studio.training.loop import (
    EpochCheckpointView,
    TrainingHistory,
    TrainingResult,
    TrainingResumeState,
    _build_criterion,
    _build_optimizer,
    _build_precision_execution,
    _build_scheduler,
    evaluate,
    evaluate_classification_metrics,
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


# -- evaluate_classification_metrics() (Phase 4O) -----------------------------
#
# evaluate()의 기존 단위 테스트(위)를 복제하지 않는다 -- evaluate() 자체는
# 이 Phase에서 무수정이므로 그 테스트들은 그대로 유효하다. 여기서는
# evaluate_classification_metrics()가 evaluate()와 다르게 새로 제공하는
# 부분(confusion matrix 기반 상세 metric)과, evaluate()와 반드시 같아야
# 하는 부분(loss/accuracy의 의미)만 검증한다.


def test_evaluate_classification_metrics_matches_evaluate_loss_and_accuracy() -> None:
    """evaluate_classification_metrics()의 (loss, accuracy)가 같은
    model/loader에 대한 기존 evaluate()의 반환값과 정확히 일치해야 한다 --
    두 함수가 같은 의미(unsmoothed CrossEntropyLoss, argmax accuracy,
    sample-weighted 평균)를 계산한다는 계약을 고정한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    expected_loss, expected_accuracy = evaluate(model, val_loader)
    loss, accuracy, _metrics = evaluate_classification_metrics(model, val_loader, num_classes=NUM_CLASSES)

    assert loss == pytest.approx(expected_loss)
    assert accuracy == pytest.approx(expected_accuracy)


def test_evaluate_classification_metrics_accuracy_matches_confusion_matrix_diagonal() -> None:
    """accuracy == sum(diagonal(confusion_matrix)) / sum(confusion_matrix) --
    accuracy와 confusion matrix가 같은 예측 결과로부터 계산됐음을 보장하는
    핵심 회귀 계약."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    _loss, accuracy, metrics = evaluate_classification_metrics(model, val_loader, num_classes=NUM_CLASSES)

    cm = torch.tensor(metrics.confusion_matrix)
    diagonal_sum = cm.diagonal().sum().item()
    total = cm.sum().item()
    assert diagonal_sum / total == pytest.approx(accuracy)


def test_evaluate_classification_metrics_returns_expected_shapes() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    _loss, _accuracy, metrics = evaluate_classification_metrics(model, val_loader, num_classes=NUM_CLASSES)

    assert len(metrics.confusion_matrix) == NUM_CLASSES
    assert all(len(row) == NUM_CLASSES for row in metrics.confusion_matrix)
    assert len(metrics.per_class_recall) == NUM_CLASSES
    assert isinstance(metrics.macro_precision, float)
    assert isinstance(metrics.macro_recall, float)
    assert isinstance(metrics.macro_f1, float)


def test_evaluate_classification_metrics_metrics_are_finite() -> None:
    """zero-division=0.0 정책 덕분에 모든 지표가 항상 finite여야 한다(NaN/
    +-inf 없음) -- json.dumps()의 비표준 NaN 허용에 의존하지 않는다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    _loss, _accuracy, metrics = evaluate_classification_metrics(model, val_loader, num_classes=NUM_CLASSES)

    assert all(math.isfinite(value) for row in metrics.confusion_matrix for value in row)
    assert all(math.isfinite(value) for value in metrics.per_class_recall)
    assert math.isfinite(metrics.macro_precision)
    assert math.isfinite(metrics.macro_recall)
    assert math.isfinite(metrics.macro_f1)


def test_evaluate_classification_metrics_does_not_change_parameters() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    before = copy.deepcopy(model.state_dict())
    evaluate_classification_metrics(model, val_loader, num_classes=NUM_CLASSES)
    after = model.state_dict()

    assert all(torch.equal(before[name], after[name]) for name in before)


def test_evaluate_classification_metrics_raises_on_empty_loader() -> None:
    """evaluate()와 동일한 empty-loader 정책(ValueError) -- zero-division으로
    조용히 0 loss/accuracy를 반환하지 않는다."""
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, *spec.input_shape), torch.empty(0, dtype=torch.long)), batch_size=8
    )

    with pytest.raises(ValueError, match="empty DataLoader"):
        evaluate_classification_metrics(model, empty_loader, num_classes=NUM_CLASSES)


@pytest.mark.parametrize("invalid_num_classes", [0, -1])
def test_evaluate_classification_metrics_rejects_non_positive_num_classes(invalid_num_classes: int) -> None:
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _, val_loader = _make_loaders(spec, seed=0)

    with pytest.raises(ValueError, match="num_classes"):
        evaluate_classification_metrics(model, val_loader, num_classes=invalid_num_classes)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_run_training_on_cuda_completes_one_epoch() -> None:
    """optional CUDA smoke test(Phase 4Q) -- model.to("cuda") 후
    run_training(..., device="cuda")가 정상 완료되는지 최소 확인한다
    (GPU 없는 CI에서는 자동 skip). model이 이미 target device에 있어야
    한다는 run_training()의 기존 계약을 caller가 그대로 지킨다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec).to("cuda")
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config, device="cuda")

    assert next(model.parameters()).device.type == "cuda"
    assert len(result.history.train_losses) == 1
    assert math.isfinite(result.history.train_losses[0])
    assert math.isfinite(result.history.val_losses[0])


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
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
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
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
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


def test_build_optimizer_adamw_config_returns_adamw() -> None:
    """Phase 4L: optimizer="adamw"는 torch.optim.AdamW 인스턴스를 반환해야 한다."""
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=2e-3, optimizer="adamw")

    optimizer = _build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 2e-3


@pytest.mark.parametrize("optimizer_name", ["adam", "sgd", "adamw"])
def test_build_optimizer_passes_weight_decay_to_param_groups(optimizer_name: str) -> None:
    """Phase 4L: weight_decay는 Adam/SGD/AdamW 세 optimizer 모두에 동일하게 전달된다."""
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-3, optimizer=optimizer_name, weight_decay=0.37
    )

    optimizer = _build_optimizer(model, config)

    assert optimizer.param_groups[0]["weight_decay"] == 0.37


def test_build_optimizer_default_weight_decay_is_zero_in_param_groups() -> None:
    """weight_decay를 지정하지 않으면(기본값 0.0) 기존 동작과 동일하게
    param_groups에도 0.0이 반영되어야 한다."""
    model = build_model(_mlp_classifier_spec())
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3)

    optimizer = _build_optimizer(model, config)

    assert optimizer.param_groups[0]["weight_decay"] == 0.0


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
    """ReduceLROnPlateau의 실제 동작을 그대로 고정한다: 첫 step() 호출은
    baseline(기준값)을 세울 뿐 LR을 바꾸지 않는다. 그 이후로 개선되지
    않는(bad) epoch 수가 patience를 초과하는 step() 호출에서 LR이
    감소한다. patience=2, 동일한 loss를 계속 넣으면:

        call 1: baseline 설정 (bad epoch 아님)
        call 2: bad epoch 1
        call 3: bad epoch 2
        call 4: bad epoch 3 > patience(2) -> LR 감소

    즉 4번째 step()에서 처음 LR이 바뀐다 (실제로 PyTorch로 직접 실행해
    확인한 호출 순번)."""
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
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)
    history = result.history

    assert len(history.train_losses) == 3
    assert history.stopped_early is True
    assert history.best_epoch == 1
    assert history.best_val_loss == 1.0
    assert result.best_state_dict is not None
    assert len(result.best_state_dict) > 0


def test_run_training_early_stopping_preserves_best_epoch_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """best_state_dict가 실제로 best epoch(1)의 파라미터 값을 담고 있는지
    직접 검증한다 -- "비어있지 않다"는 것만으로는 어느 epoch의 snapshot인지
    증명하지 못하므로, train_one_epoch을 monkeypatch해 매 epoch 파라미터
    전체를 서로 다른 상수(epoch 번호)로 명시적으로 채운다. 이 스펙
    (_mlp_classifier_spec)에는 BatchNorm이 없어 state_dict의 모든
    텐서가 float32 파라미터뿐이므로 전체를 안전하게 상수로 채울 수 있다.

    patience=2, val_loss 시퀀스 [1.0, 1.0, 1.0, 0.5]:
    epoch 1 -> best, 파라미터를 전부 1.0으로 채움
    epoch 2 -> 개선 없음, 파라미터를 전부 2.0으로 채움 (best_state_dict는 안 바뀌어야 함)
    epoch 3 -> 개선 없음, 파라미터를 전부 3.0으로 채움 -> 중단
    (epoch 4는 실행되지 않으므로 파라미터가 4.0이 될 일도 없다)

    best_state_dict는 epoch 1에서 채운 1.0을 그대로 유지해야 한다.
    """
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=2)

    call_count = {"value": 0}

    def fake_train_one_epoch(
        model, loader, optimizer, device="cpu", gradient_clip_norm=None, criterion=None,
        autocast_dtype=None, scaler=None, non_blocking=False,
    ):
        call_count["value"] += 1
        epoch_value = float(call_count["value"])
        for param in model.parameters():
            param.data.fill_(epoch_value)
        return epoch_value  # train_loss 값 자체는 이 테스트의 관심사가 아님

    fixed_val_results = iter([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.5, 1.0)])
    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)
    history = result.history

    # 중단 시점/best epoch 자체는 기존 테스트와 동일한 계약 재확인
    assert len(history.train_losses) == 3
    assert history.stopped_early is True
    assert history.best_epoch == 1

    # 핵심 검증: best_state_dict가 epoch 2/3의 값(2.0/3.0)이 아니라
    # epoch 1에서 채운 값(1.0)을 그대로 담고 있는지 확인
    for tensor in result.best_state_dict.values():
        assert torch.all(tensor == 1.0)


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
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
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
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    result = run_training(model, train_loader, val_loader, config)

    assert len(result.history.train_losses) == 2
    assert result.history.stopped_early is True


# -- Phase 4F: resume ----------------------------------------------------------


def _dropout_mlp_classifier_spec() -> ModelSpec:
    """Dropout이 포함된 스펙. Dropout은 DataLoader shuffle generator가 아니라
    전역 CPU RNG를 쓰므로(직접 실행해 확인, docs/phase4f 참고), exact
    resume 테스트에서 CPU RNG state 복원이 실제로 필요함을 증명하는 데
    쓴다."""
    return ModelSpec(
        name="dropout_mlp_classifier",
        input_shape=INPUT_SHAPE,
        layers=[
            FlattenSpec(),
            LinearSpec(out_features=16),
            ReLUSpec(),
            DropoutSpec(p=0.3),
            LinearSpec(out_features=NUM_CLASSES),
        ],
    )


def _assert_deep_equal(a: object, b: object, path: str = "value") -> None:
    """optimizer/scheduler state_dict처럼 텐서와 스칼라가 섞인 중첩
    dict/list를 재귀적으로 정확히 비교하는 테스트 전용 헬퍼."""
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


def _make_resume_state(result: TrainingResult, config: TrainingConfig) -> TrainingResumeState:
    """config는 result를 만들어 낸 바로 그 TrainingConfig여야 한다 --
    training_config가 채워져 있어야 run_training()의 core config
    호환성 검증(require_compatible_resume_config)을 통과한다."""
    return TrainingResumeState(
        optimizer_state_dict=result.optimizer_state_dict,
        scheduler_state_dict=result.scheduler_state_dict,
        history=result.history,
        epochs_without_improvement=result.epochs_without_improvement,
        best_state_dict=result.best_state_dict,
        training_config=asdict(config),
        scaler_state_dict=result.scaler_state_dict,
    )


def test_run_training_resume_appends_history_after_prior_epochs() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)

    second = run_training(
        model, train_loader, val_loader,
        TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2),
        resume_state=_make_resume_state(first, first_config),
    )

    assert len(second.history.train_losses) == 5
    assert second.history.train_losses[:3] == first.history.train_losses
    assert second.history.val_losses[:3] == first.history.val_losses
    assert second.history.val_accuracies[:3] == first.history.val_accuracies


def test_run_training_resume_records_best_epoch_using_absolute_epoch_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """completed_epochs=3에서 resume 후 첫 epoch(절대 번호 4)에서 새로운
    best가 나오면 best_epoch는 4여야 한다 -- 1이 되면 안 된다(off-by-reset
    방지)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)

    first_val_results = iter([(1.0, 1.0), (0.9, 1.0), (0.8, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(first_val_results),
    )
    first_config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)
    assert first.history.best_epoch == 3

    second_val_results = iter([(0.5, 1.0), (0.99, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(second_val_results),
    )
    second = run_training(
        model, train_loader, val_loader,
        TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2),
        resume_state=_make_resume_state(first, first_config),
    )

    assert len(second.history.train_losses) == 5
    assert second.history.best_epoch == 4
    assert second.history.best_val_loss == 0.5


def test_run_training_resume_preserves_best_state_dict_when_no_new_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resume 이후 epoch들이 전부 이전 best보다 개선되지 않으면,
    best_state_dict는 resume 이전 값(과거 best epoch의 snapshot)을 그대로
    유지해야 한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)

    first_val_results = iter([(1.0, 1.0), (0.5, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(first_val_results),
    )
    first_config = TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)
    first_best_snapshot = {name: tensor.clone() for name, tensor in first.best_state_dict.items()}

    second_val_results = iter([(0.9, 1.0), (0.8, 1.0)])  # 둘 다 0.5보다 나쁨 -> 개선 없음
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(second_val_results),
    )
    second = run_training(
        model, train_loader, val_loader,
        TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2),
        resume_state=_make_resume_state(first, first_config),
    )

    assert second.history.best_epoch == 2  # resume 이전 epoch 그대로
    assert second.history.best_val_loss == 0.5
    for name, tensor in second.best_state_dict.items():
        assert torch.equal(tensor, first_best_snapshot[name])


def test_run_training_resume_restores_early_stopping_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """checkpoint 시점에 이미 patience-1만큼 카운트된 상태로 resume하면,
    resume 후 단 1번만 개선에 실패해도 즉시 중단돼야 한다 (카운터가
    0부터 다시 시작하면 안 됨)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)

    # epoch1=best(1.0), epoch2=개선없음(카운터1), epoch3=개선없음(카운터2) -- early_stopping_patience=3이라 아직 중단 안 됨
    first_val_results = iter([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(first_val_results),
    )
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2, early_stopping_patience=3)
    first = run_training(model, train_loader, val_loader, config)
    assert first.history.stopped_early is False
    assert first.epochs_without_improvement == 2

    # resume 후 1 epoch만 더 개선 실패하면 카운터가 3 == patience가 되어 즉시 중단돼야 함
    second_val_results = iter([(1.0, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(second_val_results),
    )
    second = run_training(
        model, train_loader, val_loader, config, resume_state=_make_resume_state(first, config)
    )

    assert len(second.history.train_losses) == 4
    assert second.history.stopped_early is True


def test_run_training_resume_does_not_mutate_resume_state_history() -> None:
    """resume_state.history 원본 객체는 run_training() 호출로 변형되면
    안 된다 (caller가 들고 있는 checkpoint payload/TrainingResumeState를
    나중에 다시 쓸 수 있어야 하므로)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)
    resume_state = _make_resume_state(first, first_config)

    original_length = len(resume_state.history.train_losses)
    original_losses = list(resume_state.history.train_losses)

    run_training(
        model, train_loader, val_loader,
        TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2),
        resume_state=resume_state,
    )

    assert len(resume_state.history.train_losses) == original_length
    assert resume_state.history.train_losses == original_losses


def test_training_resume_state_rejects_stopped_early_history() -> None:
    stopped_history = TrainingHistory(
        train_losses=[1.0], val_losses=[1.0], val_accuracies=[0.5],
        best_epoch=1, best_val_loss=1.0, stopped_early=True,
    )
    with pytest.raises(ValueError, match="stopped_early"):
        TrainingResumeState(
            optimizer_state_dict={}, scheduler_state_dict=None,
            history=stopped_history, epochs_without_improvement=0, best_state_dict={},
            training_config=asdict(TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)),
        )


def test_training_resume_state_rejects_mismatched_history_lengths() -> None:
    bad_history = TrainingHistory(
        train_losses=[1.0, 0.9], val_losses=[1.0], val_accuracies=[0.5, 0.6],
        best_epoch=1, best_val_loss=1.0,
    )
    with pytest.raises(ValueError, match="equal length"):
        TrainingResumeState(
            optimizer_state_dict={}, scheduler_state_dict=None,
            history=bad_history, epochs_without_improvement=0, best_state_dict={},
            training_config=asdict(TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)),
        )


def test_training_resume_state_rejects_empty_history() -> None:
    empty_history = TrainingHistory(train_losses=[], val_losses=[], val_accuracies=[])
    with pytest.raises(ValueError, match="must not be empty"):
        TrainingResumeState(
            optimizer_state_dict={}, scheduler_state_dict=None,
            history=empty_history, epochs_without_improvement=0, best_state_dict={},
            training_config=asdict(TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)),
        )


@pytest.mark.parametrize("value", [None, "2", -1, True])
def test_training_resume_state_rejects_invalid_epochs_without_improvement(value: object) -> None:
    """int가 아니거나(None/문자열/bool) 음수면 TypeError가 아니라 명확한
    ValueError여야 한다. bool은 Python에서 int의 서브클래스라 명시적으로
    제외한다(TrainingConfig의 기존 검증 스타일과 동일)."""
    valid_history = TrainingHistory(
        train_losses=[1.0], val_losses=[1.0], val_accuracies=[0.5], best_epoch=1, best_val_loss=1.0
    )
    with pytest.raises(ValueError, match="epochs_without_improvement must be a non-negative integer"):
        TrainingResumeState(
            optimizer_state_dict={}, scheduler_state_dict=None,
            history=valid_history, epochs_without_improvement=value, best_state_dict={},
            training_config=asdict(TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)),
        )


def test_run_training_resume_rejects_internally_inconsistent_scheduler_state() -> None:
    """resume_state.training_config는 scheduler가 켜진 것으로 되어 있고
    새 config도 그와 완전히 일치하는데(= require_compatible_resume_config
    통과), resume_state.scheduler_state_dict 자체가 없으면(내부적으로
    손상된 resume_state) run_training()이 이 불일치를 별도로 잡아야
    한다 -- config 비교만으로는 이 손상을 발견할 수 없다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    scheduler_config = TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2, lr_scheduler="plateau")
    first = run_training(model, train_loader, val_loader, scheduler_config)
    resume_state = _make_resume_state(first, scheduler_config)
    resume_state.scheduler_state_dict = None  # 저장/조립 과정에서 손상된 상황을 흉내

    with pytest.raises(ValueError, match="scheduler_state_dict"):
        run_training(model, train_loader, val_loader, scheduler_config, resume_state=resume_state)


def test_run_training_resume_rejects_incompatible_saved_config() -> None:
    """caller가 require_compatible_resume_config()를 별도로 호출하지 않고
    바로 run_training()을 호출해도, config가 checkpoint 저장 당시와
    다르면 core API 자체가 거부해야 한다 (caller helper 호출에 의존하지
    않음을 증명)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)
    resume_state = _make_resume_state(first, first_config)

    incompatible_config = TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-4)  # learning_rate만 다름
    with pytest.raises(ValueError, match="learning_rate"):
        run_training(model, train_loader, val_loader, incompatible_config, resume_state=resume_state)


def test_run_training_resume_matches_continuous_run_exactly() -> None:
    """핵심 계약: 연속 5 epoch 실행과, 3 epoch 실행 후 checkpoint 저장을
    흉내낸 뒤 2 epoch를 resume한 결과가 다음 전부에서 정확히 일치해야
    한다 -- model parameters, optimizer state, scheduler state, history,
    best_state_dict, best_epoch, best_val_loss, epochs_without_improvement.

    Dropout이 포함된 모델을 쓴다 -- Dropout은 DataLoader의 shuffle
    generator가 아니라 전역 CPU RNG에 의존하므로(직접 실행해 확인),
    이 테스트가 실제로 CPU RNG state 복원 없이는 통과할 수 없다는 것을
    (train_one_epoch 안에서 model.train() 상태로 forward가 일어나므로)
    구조적으로 검증한다.
    """
    seed = 20260801
    spec = _dropout_mlp_classifier_spec()

    def make_loaders() -> tuple[DataLoader, DataLoader, torch.Generator]:
        train_dataset, val_dataset = make_train_val_datasets(
            spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader, generator

    config_kwargs = dict(
        batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.9,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    # (a) 연속 5 epoch
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = make_loaders()
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=5, **config_kwargs))

    # (b) 3 epoch 실행
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, generator_b = make_loaders()
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=3, **config_kwargs)
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)

    # checkpoint 시점 상태를 caller가 직접 채취 (run_training()이 파일을 쓰지 않으므로)
    loader_generator_state = generator_b.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    resume_state = _make_resume_state(result_b1, first_config)

    # "새 프로세스"를 흉내: 새 model/DataLoader/generator를 만들고 저장된 상태를 복원
    model_b2 = build_model(spec)
    model_b2.load_state_dict(model_b.state_dict())  # 3 epoch 완료 시점의 "현재" model
    train_dataset2, val_dataset2 = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(loader_generator_state)
    train_loader_b2 = DataLoader(
        train_dataset2, batch_size=8, shuffle=True, generator=restored_generator, drop_last=True
    )
    val_loader_b2 = DataLoader(val_dataset2, batch_size=8, shuffle=False)
    torch.set_rng_state(cpu_rng_state)

    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, TrainingConfig(epochs=2, **config_kwargs),
        resume_state=resume_state,
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

    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


def test_run_training_resume_matches_continuous_run_exactly_with_adamw_weight_decay() -> None:
    """Phase 4L: optimizer="adamw" + weight_decay>0 조합에서도 exact-resume
    계약(test_run_training_resume_matches_continuous_run_exactly와 동일한
    근거)이 깨지지 않아야 한다."""
    seed = 20260801
    spec = _dropout_mlp_classifier_spec()

    def make_loaders() -> tuple[DataLoader, DataLoader, torch.Generator]:
        train_dataset, val_dataset = make_train_val_datasets(
            spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader, generator

    config_kwargs = dict(
        batch_size=8, learning_rate=1e-2, optimizer="adamw", weight_decay=0.05,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    # (a) 연속 5 epoch
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = make_loaders()
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=5, **config_kwargs))

    # (b) 3 epoch 실행
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, generator_b = make_loaders()
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=3, **config_kwargs)
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)

    loader_generator_state = generator_b.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    resume_state = _make_resume_state(result_b1, first_config)

    model_b2 = build_model(spec)
    model_b2.load_state_dict(model_b.state_dict())
    train_dataset2, val_dataset2 = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(loader_generator_state)
    train_loader_b2 = DataLoader(
        train_dataset2, batch_size=8, shuffle=True, generator=restored_generator, drop_last=True
    )
    val_loader_b2 = DataLoader(val_dataset2, batch_size=8, shuffle=False)
    torch.set_rng_state(cpu_rng_state)

    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, TrainingConfig(epochs=2, **config_kwargs),
        resume_state=resume_state,
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

    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


# -- Phase 4M: gradient_clip_norm ----------------------------------------------


def _total_grad_norm(model: torch.nn.Module) -> float:
    """optimizer.step() 이후에도 .grad는 다음 zero_grad() 전까지 그대로
    남아있으므로, train_one_epoch()가 반환된 직후(loader에 batch가 정확히
    1개뿐이라 그게 곧 "마지막으로 step() 직전에 쓰인 gradient") 이 값을 읽으면
    실제로 clipping이 적용됐는지 production API를 바꾸지 않고도 직접 검증할
    수 있다."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return total_sq**0.5


def _single_batch_loader(spec: ModelSpec, batch_size: int = 8, scale: float = 1.0) -> DataLoader:
    """batch가 정확히 1개인 DataLoader -- scale을 키우면 backward() 직후의
    gradient가 커지도록 만들어, clipping이 실제로 norm을 줄이는지(대조군
    없이는 "원래도 작아서 우연히 통과"와 구분되지 않으므로) 검증할 수 있다."""
    torch.manual_seed(0)
    images = torch.randn(batch_size, *spec.input_shape) * scale
    labels = torch.randint(0, NUM_CLASSES, (batch_size,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_train_one_epoch_clips_gradient_norm_to_max_norm() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    loader = _single_batch_loader(spec, scale=1000.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_one_epoch(model, loader, optimizer, gradient_clip_norm=1e-3)

    assert _total_grad_norm(model) <= 1e-3 + 1e-6  # 부동소수 오차만 허용


def test_train_one_epoch_without_clipping_leaves_gradient_norm_unbounded() -> None:
    """위 테스트와 동일한 입력(scale=1000.0)을 clipping 없이 실행하면
    gradient norm이 max_norm(1e-3)보다 훨씬 커야 한다 -- 대조군 없이는
    위 테스트가 "원래도 norm이 작아서 우연히 통과"인지 구분할 수 없다."""
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    loader = _single_batch_loader(spec, scale=1000.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_one_epoch(model, loader, optimizer, gradient_clip_norm=None)

    assert _total_grad_norm(model) > 1e-3


def test_train_one_epoch_default_gradient_clip_norm_matches_explicit_none() -> None:
    """gradient_clip_norm 생략(기본값)과 명시적 None이 완전히 동일한 결과를
    내야 한다 -- clipping 비활성화 시 기존 계산 경로가 그대로 사용된다는
    회귀 계약."""
    spec = _mlp_classifier_spec()

    torch.manual_seed(0)
    model_a = build_model(spec)
    loader_a = _single_batch_loader(spec, scale=1.0)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
    loss_a = train_one_epoch(model_a, loader_a, optimizer_a)  # 인자 생략

    torch.manual_seed(0)
    model_b = build_model(spec)
    loader_b = _single_batch_loader(spec, scale=1.0)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)
    loss_b = train_one_epoch(model_b, loader_b, optimizer_b, gradient_clip_norm=None)  # 명시적 None

    assert loss_a == loss_b
    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_b.state_dict()[name])


def test_run_training_applies_new_gradient_clip_norm_after_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """resume 시 gradient_clip_norm을 바꾸면(여기서는 None -> 0.5), 실제로
    resume된 구간에서 새 값이 clip_grad_norm_()에 전달되는지 확인한다 --
    require_compatible_resume_config()가 이 값을 비교 대상으로 삼지
    않는다는 것만으로는 "새 값이 실제로 적용된다"는 것까지는 증명하지
    못하므로, torch.nn.utils.clip_grad_norm_ 호출을 spy로 감싸 실제
    max_norm 인자를 기록한다."""
    calls: list[float | None] = []
    real_clip_grad_norm_ = torch.nn.utils.clip_grad_norm_

    def spy_clip_grad_norm_(parameters, max_norm, **kwargs):
        calls.append(max_norm)
        return real_clip_grad_norm_(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip_grad_norm_)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, gradient_clip_norm=None)
    first = run_training(model, train_loader, val_loader, first_config)

    assert calls == []  # gradient_clip_norm=None이면 clip 호출 자체가 없다

    second_config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, gradient_clip_norm=0.5)
    run_training(
        model, train_loader, val_loader, second_config,
        resume_state=_make_resume_state(first, first_config),
    )

    assert calls  # resume 구간에서는 clip이 호출됐다
    assert all(call == 0.5 for call in calls)  # 그것도 새 config의 값으로


def test_run_training_resume_matches_continuous_run_exactly_with_gradient_clip_norm() -> None:
    """Phase 4M: gradient_clip_norm != None에서도 exact-resume 계약이
    깨지지 않아야 한다(clip_grad_norm_()는 RNG를 소비하지 않는 결정론적
    연산이므로 tensor-level exact equality를 기대한다)."""
    seed = 20260801
    spec = _dropout_mlp_classifier_spec()

    def make_loaders() -> tuple[DataLoader, DataLoader, torch.Generator]:
        train_dataset, val_dataset = make_train_val_datasets(
            spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader, generator

    config_kwargs = dict(
        batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.9, gradient_clip_norm=0.5,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    # (a) 연속 5 epoch
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = make_loaders()
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=5, **config_kwargs))

    # (b) 3 epoch 실행
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, generator_b = make_loaders()
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=3, **config_kwargs)
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)

    loader_generator_state = generator_b.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    resume_state = _make_resume_state(result_b1, first_config)

    model_b2 = build_model(spec)
    model_b2.load_state_dict(model_b.state_dict())
    train_dataset2, val_dataset2 = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(loader_generator_state)
    train_loader_b2 = DataLoader(
        train_dataset2, batch_size=8, shuffle=True, generator=restored_generator, drop_last=True
    )
    val_loader_b2 = DataLoader(val_dataset2, batch_size=8, shuffle=False)
    torch.set_rng_state(cpu_rng_state)

    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, TrainingConfig(epochs=2, **config_kwargs),
        resume_state=resume_state,
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

    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


# -- Phase 4N: label_smoothing --------------------------------------------------


def test_build_criterion_default_matches_plain_cross_entropy_loss() -> None:
    """label_smoothing=0.0(기본값)이면 인자 없는 nn.CrossEntropyLoss()와
    bitwise 동일해야 한다 -- 기존 동작을 완전히 재현한다는 회귀 계약."""
    torch.manual_seed(0)
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (4,))
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3)

    criterion = _build_criterion(config)

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.0
    assert torch.equal(criterion(logits, targets), nn.CrossEntropyLoss()(logits, targets))


def test_build_criterion_applies_label_smoothing_matching_pytorch_reference() -> None:
    """하드코딩된 magic number가 아니라 PyTorch reference 구현
    (nn.CrossEntropyLoss(label_smoothing=0.1))과 직접 비교한다."""
    torch.manual_seed(0)
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (4,))
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, label_smoothing=0.1)

    criterion = _build_criterion(config)

    assert criterion.label_smoothing == 0.1
    reference = nn.CrossEntropyLoss(label_smoothing=0.1)
    assert torch.equal(criterion(logits, targets), reference(logits, targets))


def test_build_criterion_label_smoothing_changes_loss_value() -> None:
    """smoothing>0이 실제로 unsmoothed CE와 다른 값을 낸다는 대조군 --
    이게 없으면 위 테스트가 "우연히 같은 값"인지 구분할 수 없다."""
    torch.manual_seed(0)
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (4,))

    unsmoothed = _build_criterion(TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3))
    smoothed = _build_criterion(
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, label_smoothing=0.1)
    )

    assert not torch.equal(unsmoothed(logits, targets), smoothed(logits, targets))


def test_train_one_epoch_default_criterion_matches_plain_cross_entropy() -> None:
    """criterion 생략(기존 호출 `train_one_epoch(model, loader, optimizer)`)
    이 명시적으로 unsmoothed CrossEntropyLoss()를 넘긴 것과 완전히 동일한
    결과를 내야 한다 -- 기존 public 호출의 backward compatibility."""
    spec = _mlp_classifier_spec()

    torch.manual_seed(0)
    model_a = build_model(spec)
    train_loader_a, _ = _make_loaders(spec, seed=0)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
    loss_a = train_one_epoch(model_a, train_loader_a, optimizer_a)  # criterion 생략

    torch.manual_seed(0)
    model_b = build_model(spec)
    train_loader_b, _ = _make_loaders(spec, seed=0)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)
    loss_b = train_one_epoch(
        model_b, train_loader_b, optimizer_b, criterion=nn.CrossEntropyLoss()
    )  # 명시적 unsmoothed CE

    assert loss_a == loss_b
    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_b.state_dict()[name])


def test_train_one_epoch_actually_uses_provided_criterion() -> None:
    """넘긴 criterion이 mock으로 호출 여부만 확인되는 게 아니라, 매 batch
    실제로 사용됨을 직접 확인한다."""
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, _ = _make_loaders(spec, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    call_count = {"value": 0}

    class RecordingCriterion(nn.Module):
        def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            call_count["value"] += 1
            return nn.functional.cross_entropy(outputs, labels, label_smoothing=0.2)

    train_one_epoch(model, train_loader, optimizer, criterion=RecordingCriterion())

    assert call_count["value"] == len(train_loader)  # 매 batch마다 정확히 한 번씩


def test_train_one_epoch_criterion_and_gradient_clip_norm_coexist() -> None:
    """Phase 4M(gradient_clip_norm)과 Phase 4N(criterion)이 동시에 있어도
    호출 순서가 깨지지 않고 정상적으로 완료돼야 한다."""
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, _ = _make_loaders(spec, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    loss = train_one_epoch(
        model, train_loader, optimizer,
        gradient_clip_norm=1.0, criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
    )

    assert loss > 0.0


def test_run_training_passes_build_criterion_result_to_train_one_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_training()이 _build_criterion(config)로 만든 바로 그 criterion
    객체를 실제로 train_one_epoch()에 전달한다는 연결 계약을 직접 고정한다.
    다른 테스트들(criterion factory가 올바른 CrossEntropyLoss를 만드는지,
    train_one_epoch()가 넘겨받은 criterion을 실제로 쓰는지, resume 시
    새 값으로 다시 만들어지는지)은 각 구간을 따로 검증하지만, 그 사이를
    잇는 `train_one_epoch(..., criterion=criterion)` 한 줄이 production
    코드에서 실수로 빠지는 회귀는 어느 것도 직접 잡지 못한다 -- 예를 들어
    run_training()이 criterion을 만들어 놓고 실제로는 넘기지 않아도(즉
    train_one_epoch()가 항상 자체 기본 unsmoothed CrossEntropyLoss를
    쓰게 되어도) 그 테스트들은 각자의 관찰 지점에서는 여전히 통과할 수
    있다. train_one_epoch() 자체를 monkeypatch해 실제로 전달받은
    `criterion` keyword 인자를 검사하는 것으로 이 연결을 직접 증명한다."""
    captured: dict = {}

    def fake_train_one_epoch(
        model, loader, optimizer, device="cpu", gradient_clip_norm=None, criterion=None,
        autocast_dtype=None, scaler=None, non_blocking=False,
    ):
        captured["criterion"] = criterion
        return 0.5

    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: (0.5, 0.5),
    )

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, label_smoothing=0.3)

    run_training(model, train_loader, val_loader, config)

    assert captured["criterion"] is not None
    assert isinstance(captured["criterion"], nn.CrossEntropyLoss)
    assert captured["criterion"].label_smoothing == 0.3


def test_run_training_evaluate_ignores_label_smoothing() -> None:
    """label_smoothing>0으로 학습해도 run_training()이 history에 기록하는
    val_loss/val_accuracy는 (수정되지 않은) evaluate()를 그대로 호출한
    결과와 정확히 일치해야 한다 -- 즉 이 테스트는 "run_training()의
    validation 경로가 무수정 evaluate()를 그대로 쓴다"는 배선을
    증명한다. evaluate() 자체가 항상 ordinary(unsmoothed)
    CrossEntropyLoss를 쓴다는 계약은 evaluate()가 이번 Phase에서 무수정
    production 코드라는 사실과, 그 계약을 이미 검증하는 기존 evaluate
    테스트들이 담당한다(이 테스트가 manual reference와 독립적으로
    "unsmoothed임"을 다시 증명하는 것은 아니다)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, label_smoothing=0.5)

    result = run_training(model, train_loader, val_loader, config)
    expected_val_loss, expected_val_accuracy = evaluate(model, val_loader)

    assert result.history.val_losses[-1] == expected_val_loss
    assert result.history.val_accuracies[-1] == expected_val_accuracy


def test_run_training_uses_new_label_smoothing_after_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """resume 시 label_smoothing을 바꾸면(여기서는 0.0 -> 0.1), 실제로
    resume된 구간에서 새 값으로 만든 criterion이 쓰이는지 확인한다 --
    require_compatible_resume_config()가 이 값을 비교 대상으로 삼지
    않는다는 것만으로는 "새 값이 실제로 적용된다"는 것까지는 증명하지
    못하므로, _build_criterion() 호출을 spy로 감싸 실제로 어떤
    label_smoothing으로 호출되는지 기록한다."""
    calls: list[float] = []
    real_build_criterion = _build_criterion

    def spy_build_criterion(config: TrainingConfig, device: str = "cpu") -> nn.Module:
        calls.append(config.label_smoothing)
        return real_build_criterion(config, device=device)

    monkeypatch.setattr("image_ai_studio.training.loop._build_criterion", spy_build_criterion)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, label_smoothing=0.0)
    first = run_training(model, train_loader, val_loader, first_config)

    assert calls == [0.0]

    second_config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, label_smoothing=0.1)
    run_training(
        model, train_loader, val_loader, second_config,
        resume_state=_make_resume_state(first, first_config),
    )

    assert calls == [0.0, 0.1]  # resume 구간에서도 새 config의 값으로 criterion이 다시 만들어짐


def test_run_training_resume_matches_continuous_run_exactly_with_label_smoothing() -> None:
    """Phase 4N: label_smoothing != 0에서도 exact-resume 계약이 깨지지
    않아야 한다(CrossEntropyLoss(label_smoothing=...)는 RNG를 소비하지
    않는 결정론적 연산이므로 tensor-level exact equality를 기대한다).
    continuous run과 resume run 양쪽에 동일한 label_smoothing 값을
    쓴다 -- resume 도중 값을 바꾸는 케이스는 별도 spy 테스트(위)가
    담당한다."""
    seed = 20260801
    spec = _dropout_mlp_classifier_spec()

    def make_loaders() -> tuple[DataLoader, DataLoader, torch.Generator]:
        train_dataset, val_dataset = make_train_val_datasets(
            spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader, generator

    config_kwargs = dict(
        batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.9, label_smoothing=0.3,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    # (a) 연속 5 epoch
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = make_loaders()
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=5, **config_kwargs))

    # (b) 3 epoch 실행
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, generator_b = make_loaders()
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=3, **config_kwargs)
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)

    loader_generator_state = generator_b.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    resume_state = _make_resume_state(result_b1, first_config)

    model_b2 = build_model(spec)
    model_b2.load_state_dict(model_b.state_dict())
    train_dataset2, val_dataset2 = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(loader_generator_state)
    train_loader_b2 = DataLoader(
        train_dataset2, batch_size=8, shuffle=True, generator=restored_generator, drop_last=True
    )
    val_loader_b2 = DataLoader(val_dataset2, batch_size=8, shuffle=False)
    torch.set_rng_state(cpu_rng_state)

    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, TrainingConfig(epochs=2, **config_kwargs),
        resume_state=resume_state,
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

    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


# -- Phase 4P: class_weights ---------------------------------------------------


def test_build_criterion_default_class_weights_matches_plain_cross_entropy_loss() -> None:
    """class_weights=None(기본값)이면 인자 없는 nn.CrossEntropyLoss()와
    bitwise 동일해야 한다 -- 기존 동작을 완전히 재현한다는 회귀 계약."""
    torch.manual_seed(0)
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (4,))
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3)

    criterion = _build_criterion(config)

    assert criterion.weight is None
    assert torch.equal(criterion(logits, targets), nn.CrossEntropyLoss()(logits, targets))


def test_build_criterion_applies_class_weights_matching_pytorch_reference() -> None:
    """하드코딩된 magic number가 아니라 PyTorch reference 구현
    (nn.CrossEntropyLoss(weight=torch.tensor(...)))과 직접 비교한다."""
    torch.manual_seed(0)
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (4,))
    weights = (1.0, 2.0, 0.5, 3.0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, class_weights=weights)

    criterion = _build_criterion(config)

    reference = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    assert torch.equal(criterion(logits, targets), reference(logits, targets))


def test_build_criterion_class_weights_changes_loss_value() -> None:
    """class_weights가 실제로 unweighted CE와 다른 값을 낸다는 대조군.

    random fixture 대신 손으로 구성한 deterministic 값을 쓴다 -- weighted
    CrossEntropyLoss의 기본 reduction="mean"은 각 샘플의 loss를 그 샘플의
    target class weight로 가중 평균하므로(weight 합으로 정규화), batch의
    모든 샘플이 우연히 같은 per-sample loss를 내면 weight를 아무리
    비대칭으로 줘도 weighted/unweighted 평균이 같아질 수 있다(동일한 값의
    가중 평균은 weight와 무관하게 그 값 자체이므로). 이 fixture는 두
    샘플의 target class(0, 1)가 서로 다른 per-sample loss를 내도록
    logits를 직접 골라(target 0은 다른 클래스보다 훨씬 큰 logit을 가져
    loss가 작고, target 1은 그렇지 않아 loss가 큼) class 1에 훨씬 큰
    weight(4.0)를 줬다 -- 어떤 seed가 나오든 항상 이 두 값이 다르다는
    것을 보장한다(random RNG에 의존하지 않음)."""
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],  # target 0: 다른 클래스보다 logit이 훨씬 커서 loss가 작음
            [2.0, 1.0, 0.0, 0.0],  # target 1: 1등 클래스가 아니라서 loss가 상대적으로 큼
        ]
    )
    targets = torch.tensor([0, 1])
    # class 1(loss가 큰 샘플의 target)에 훨씬 큰 weight를 줘, weighted 평균이
    # unweighted 평균(단순 산술 평균)보다 명확히 커지도록 만든다.
    weights = (1.0, 4.0, 1.0, 1.0)

    unweighted = _build_criterion(TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3))
    weighted = _build_criterion(
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, class_weights=weights)
    )

    assert not torch.equal(unweighted(logits, targets), weighted(logits, targets))


def test_build_criterion_weight_tensor_dtype_and_device() -> None:
    """weight tensor의 dtype은 항상 float32로 고정되고(PyTorch가 정수 dtype을
    거부함을 실측 확인), device는 _build_criterion()에 전달한 device와
    일치해야 한다. GPU가 CI에 없다고 CPU 검증을 생략하지 않는다 -- 여기서는
    CPU 경로만 검증하고 GPU는 필수로 요구하지 않는다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, class_weights=(1.0, 2.0))

    criterion = _build_criterion(config, device="cpu")

    assert criterion.weight.dtype == torch.float32
    assert criterion.weight.device.type == "cpu"
    assert torch.equal(criterion.weight, torch.tensor([1.0, 2.0], dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_build_criterion_weight_tensor_on_cuda_device() -> None:
    """optional CUDA smoke test(Phase 4Q) -- class_weights tensor가
    _build_criterion(config, device="cuda")로 실제 CUDA device 위에
    생성되는지 확인한다(GPU 없는 CI에서는 자동 skip)."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, class_weights=(1.0, 2.0))

    criterion = _build_criterion(config, device="cuda")

    assert criterion.weight.dtype == torch.float32
    assert criterion.weight.device.type == "cuda"


def test_build_criterion_class_weights_and_label_smoothing_combination_matches_pytorch_reference() -> None:
    """weight+label_smoothing 조합이 PyTorch reference와 정확히 일치해야
    한다(둘의 조합을 PyTorch가 제약 없이 지원함을 실측 확인)."""
    torch.manual_seed(0)
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (4,))
    weights = (1.0, 2.0, 0.5, 3.0)
    config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-3, class_weights=weights, label_smoothing=0.1
    )

    criterion = _build_criterion(config)

    reference = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), label_smoothing=0.1)
    assert torch.equal(criterion(logits, targets), reference(logits, targets))


def test_run_training_passes_class_weights_criterion_to_train_one_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_training()이 _build_criterion(config, device)로 만든 weighted
    criterion을 실제로 train_one_epoch()에 전달하는지 직접 고정한다
    (Phase 4N의 동일 목적 테스트와 같은 이유 -- criterion factory가 옳고
    train_one_epoch()가 넘겨받은 criterion을 쓴다는 것만으로는 그 사이를
    잇는 한 줄이 production 코드에서 실수로 빠지는 회귀를 잡지 못한다)."""
    captured: dict = {}

    def fake_train_one_epoch(
        model, loader, optimizer, device="cpu", gradient_clip_norm=None, criterion=None,
        autocast_dtype=None, scaler=None, non_blocking=False,
    ):
        captured["criterion"] = criterion
        return 0.5

    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: (0.5, 0.5),
    )

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    weights = (1.0, 2.0, 0.5, 3.0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, class_weights=weights)

    run_training(model, train_loader, val_loader, config)

    assert captured["criterion"] is not None
    assert torch.equal(captured["criterion"].weight, torch.tensor(weights, dtype=torch.float32))


def test_run_training_evaluate_ignores_class_weights() -> None:
    """class_weights를 써서 학습해도 run_training()이 history에 기록하는
    val_loss/val_accuracy는 (수정되지 않은) evaluate()를 그대로 호출한
    결과와 정확히 일치해야 한다 -- label_smoothing과 동일한 검증 목적."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-2, class_weights=(1.0, 2.0, 0.5, 3.0)
    )

    result = run_training(model, train_loader, val_loader, config)
    expected_val_loss, expected_val_accuracy = evaluate(model, val_loader)

    assert result.history.val_losses[-1] == expected_val_loss
    assert result.history.val_accuracies[-1] == expected_val_accuracy


def test_run_training_uses_new_class_weights_after_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """resume 시 class_weights를 바꾸면, 실제로 resume된 구간에서 새 값으로
    만든 criterion이 쓰이는지 확인한다 -- compatibility 통과만으로는 새
    값이 실제 적용된다는 것을 증명하지 못하므로 _build_criterion() 호출을
    spy로 감싼다."""
    calls: list[tuple[float, ...] | None] = []
    real_build_criterion = _build_criterion

    def spy_build_criterion(config: TrainingConfig, device: str = "cpu") -> nn.Module:
        calls.append(config.class_weights)
        return real_build_criterion(config, device=device)

    monkeypatch.setattr("image_ai_studio.training.loop._build_criterion", spy_build_criterion)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-2, class_weights=(1.0, 2.0, 0.5, 3.0)
    )
    first = run_training(model, train_loader, val_loader, first_config)

    assert calls == [(1.0, 2.0, 0.5, 3.0)]

    second_config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-2, class_weights=(3.0, 0.5, 2.0, 1.0)
    )
    run_training(
        model, train_loader, val_loader, second_config,
        resume_state=_make_resume_state(first, first_config),
    )

    assert calls == [(1.0, 2.0, 0.5, 3.0), (3.0, 0.5, 2.0, 1.0)]


def test_run_training_resume_matches_continuous_run_exactly_with_class_weights() -> None:
    """Phase 4P: class_weights != None에서도 exact-resume 계약이 깨지지
    않아야 한다(weight tensor 생성은 RNG를 소비하지 않는 결정론적 연산이므로
    tensor-level exact equality를 기대한다). continuous run과 resume run
    양쪽에 동일한 class_weights 값을 쓴다 -- resume 도중 값을 바꾸는
    케이스는 별도 spy 테스트(위)가 담당한다."""
    seed = 20260810
    spec = _dropout_mlp_classifier_spec()

    def make_loaders() -> tuple[DataLoader, DataLoader, torch.Generator]:
        train_dataset, val_dataset = make_train_val_datasets(
            spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader, generator

    config_kwargs = dict(
        batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.9,
        class_weights=(1.0, 2.0, 0.5, 3.0),
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    # (a) 연속 5 epoch
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = make_loaders()
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=5, **config_kwargs))

    # (b) 3 epoch 실행
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, generator_b = make_loaders()
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=3, **config_kwargs)
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)

    loader_generator_state = generator_b.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    resume_state = _make_resume_state(result_b1, first_config)

    model_b2 = build_model(spec)
    model_b2.load_state_dict(model_b.state_dict())
    train_dataset2, val_dataset2 = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(loader_generator_state)
    train_loader_b2 = DataLoader(
        train_dataset2, batch_size=8, shuffle=True, generator=restored_generator, drop_last=True
    )
    val_loader_b2 = DataLoader(val_dataset2, batch_size=8, shuffle=False)
    torch.set_rng_state(cpu_rng_state)

    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, TrainingConfig(epochs=2, **config_kwargs),
        resume_state=resume_state,
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

    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


# -- Phase 4J: checkpoint_hook / EpochCheckpointView ---------------------------


def test_run_training_checkpoint_hook_none_no_behavior_change() -> None:
    """checkpoint_hook=None(기본값)이면 checkpoint_hook을 아예 넘기지 않는
    것과 동일한 결과를 내야 한다 -- history 전체, 최종 model, best_state_dict,
    optimizer/scheduler state, epochs_without_improvement까지 observable
    training result 전부를 비교한다(early stopping/should_stop이 관여할
    여지가 있는 필드도 포함)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model_a = build_model(spec)
    train_loader_a, val_loader_a = _make_loaders(spec, seed=0)
    config = TrainingConfig(
        epochs=5, batch_size=8, learning_rate=1e-2, lr_scheduler="plateau",
        lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )
    torch.manual_seed(0)
    result_a = run_training(model_a, train_loader_a, val_loader_a, config)

    torch.manual_seed(0)
    model_b = build_model(spec)
    train_loader_b, val_loader_b = _make_loaders(spec, seed=0)
    torch.manual_seed(0)
    result_b = run_training(model_b, train_loader_b, val_loader_b, config, checkpoint_hook=None)

    assert result_a.history.train_losses == result_b.history.train_losses
    assert result_a.history.val_losses == result_b.history.val_losses
    assert result_a.history.val_accuracies == result_b.history.val_accuracies
    assert result_a.history.best_epoch == result_b.history.best_epoch
    assert result_a.history.stopped_early == result_b.history.stopped_early
    assert result_a.history.stopped_by_user == result_b.history.stopped_by_user
    assert result_a.epochs_without_improvement == result_b.epochs_without_improvement
    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_b.state_dict()[name])
    for name, tensor in result_a.best_state_dict.items():
        assert torch.equal(tensor, result_b.best_state_dict[name])
    _assert_deep_equal(result_a.optimizer_state_dict, result_b.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b.scheduler_state_dict)


def test_run_training_checkpoint_hook_called_once_per_completed_epoch() -> None:
    """EpochCheckpointView는 hook 호출 범위에서만 유효한 ephemeral view라
    view 객체 자체를 hook 밖으로 들고 나오면 안 된다 -- hook 안에서
    `len(view.history.train_losses)`(global epoch)라는 immutable 파생
    값만 기록해 매 epoch 정확히 한 번씩 호출됐는지 확인한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=4, batch_size=8, learning_rate=1e-2)

    global_epochs: list[int] = []

    def hook(view: EpochCheckpointView) -> None:
        global_epochs.append(len(view.history.train_losses))

    result = run_training(model, train_loader, val_loader, config, checkpoint_hook=hook)

    assert global_epochs == [1, 2, 3, 4]
    assert len(global_epochs) == config.epochs == len(result.history.train_losses)


def test_run_training_checkpoint_hook_non_scheduled_epoch_skips_state_dict_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EpochCheckpointView는 살아있는 참조만 담으므로, view 생성 자체와
    hook 호출 사이의 core 코드(train_one_epoch/evaluate/history 기록/
    scheduler.step()/view 조립)는 optimizer.state_dict()를 전혀 호출하지
    않아야 한다 -- hook이 스스로 "non-scheduled epoch"라 판단해 즉시
    반환한 경우, 그 사실만으로 (a) 실제로 .state_dict()가 호출되지
    않았음과 (b) view 생성 과정 자체도 이를 호출하지 않았음을
    torch.optim.Adam.state_dict()에 실제 spy를 걸어 증명한다(hook 내부
    카운터만으로는 core/view 쪽의 실수를 검출하지 못하므로)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=4, batch_size=8, learning_rate=1e-2)

    call_count = {"value": 0}
    original_state_dict = torch.optim.Adam.state_dict

    def spying_state_dict(self, *args, **kwargs):
        call_count["value"] += 1
        return original_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "state_dict", spying_state_dict)

    # 각 hook 호출 "직전"(count_before)과 "직후"(count_after)의 누적 호출
    # 횟수를 기록한다. count_before[i]가 직전 epoch의 count_after와 다르면,
    # 그 사이(core 코드 + view 생성)에서 몰래 state_dict()가 호출됐다는
    # 뜻이다.
    records: list[tuple[int, int]] = []

    def selective_hook(view: EpochCheckpointView) -> None:
        count_before = call_count["value"]
        global_epoch = len(view.history.train_losses)
        if global_epoch % 2 == 0:  # scheduled epoch만 명시적으로 조회
            view.optimizer.state_dict()
        records.append((count_before, call_count["value"]))

    run_training(model, train_loader, val_loader, config, checkpoint_hook=selective_hook)

    assert len(records) == config.epochs
    # scheduled epoch(2, 4)에서만 hook 자신이 정확히 1번 호출.
    assert [after - before for before, after in records] == [0, 1, 0, 1]
    # 연속된 hook 호출 사이(core 코드 + 다음 view 생성)에는 호출이 전혀
    # 없어야 한다 -- 이전 hook이 끝난 시점의 누적 값과 다음 hook 진입
    # 시점의 누적 값이 정확히 같아야 함.
    for previous, current in zip(records, records[1:]):
        assert previous[1] == current[0]
    # 학습 종료 후 TrainingResult 조립 시 발생하는 최종 1회 호출까지 포함해
    # 총 호출 횟수는 "scheduled 2회 + 최종 1회" = 3이어야 한다(hook 구간
    # 호출과 명확히 구분).
    assert call_count["value"] == records[-1][1] + 1 == 3


def test_run_training_checkpoint_hook_non_scheduled_epoch_skips_scheduler_state_dict_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """위 optimizer 테스트(test_run_training_checkpoint_hook_non_scheduled_epoch_skips_state_dict_calls)와
    동일한 수준으로 scheduler에 대해서도 재확인한다 -- 단순히 총 호출
    횟수만 보지 않고, 각 hook 진입 전후의 누적 호출 수를 기록해 (a)
    scheduled epoch(2, 4)에서만 hook 자신이 정확히 1번 호출하고,
    non-scheduled epoch(1, 3)에서는 증가가 없으며, (b) 연속된 hook 호출
    사이(core 코드 + 다음 view 생성)에서는 몰래 호출되지 않고, (c)
    학습 종료 후 TrainingResult 조립 시 최종 1회만 추가로 발생함을
    각각 검증한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(
        epochs=4, batch_size=8, learning_rate=1e-2,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    call_count = {"value": 0}
    original_state_dict = torch.optim.lr_scheduler.ReduceLROnPlateau.state_dict

    def spying_state_dict(self, *args, **kwargs):
        call_count["value"] += 1
        return original_state_dict(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "state_dict", spying_state_dict)

    records: list[tuple[int, int]] = []

    def selective_hook(view: EpochCheckpointView) -> None:
        count_before = call_count["value"]
        global_epoch = len(view.history.train_losses)
        if global_epoch % 2 == 0:  # scheduled epoch만 명시적으로 조회
            view.scheduler.state_dict()
        records.append((count_before, call_count["value"]))

    run_training(model, train_loader, val_loader, config, checkpoint_hook=selective_hook)

    assert len(records) == config.epochs
    # epoch 1, 3(non-scheduled)에서는 증가량 0, epoch 2, 4(scheduled)에서는 증가량 1.
    assert [after - before for before, after in records] == [0, 1, 0, 1]
    # 연속된 hook 호출 사이(core 코드 + 다음 view 생성)에는 호출이 전혀
    # 없어야 한다 -- 이전 hook이 끝난 시점의 누적 값과 다음 hook 진입
    # 시점의 누적 값이 정확히 같아야 함.
    for previous, current in zip(records, records[1:]):
        assert previous[1] == current[0]
    # 학습 종료 후 TrainingResult 조립 시 발생하는 최종 1회 호출까지 포함해
    # 총 호출 횟수는 "scheduled 2회 + 최종 1회" = 3이어야 한다.
    assert call_count["value"] == records[-1][1] + 1 == 3


def test_run_training_checkpoint_hook_view_matches_epoch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate()를 monkeypatch해 val_loss 개선/비개선 순서를 결정론적으로
    고정하고(1.0, 0.8, 0.9, 1.1 -> best는 epoch 1, 2에서만 갱신), 각 hook
    호출 시점의 EpochCheckpointView가 정확한 global_epoch/
    epochs_without_improvement/best_epoch/best_state_dict를 담고 있는지
    직접 검증한다. train_one_epoch()도 monkeypatch해 매 epoch 모델
    파라미터 전체를 epoch 번호(1.0, 2.0, 3.0, 4.0)로 채워, best_state_dict
    snapshot이 실제로 어느 epoch의 값인지 명확히 구분한다(test_loop.py의
    test_run_training_early_stopping_preserves_best_epoch_parameters와
    동일한 기법)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=4, batch_size=8, learning_rate=1e-2)

    call_count = {"value": 0}

    def fake_train_one_epoch(
        model, loader, optimizer, device="cpu", gradient_clip_norm=None, criterion=None,
        autocast_dtype=None, scaler=None, non_blocking=False,
    ):
        call_count["value"] += 1
        epoch_value = float(call_count["value"])
        for param in model.parameters():
            param.data.fill_(epoch_value)
        return epoch_value

    fixed_val_results = iter([(1.0, 1.0), (0.8, 1.0), (0.9, 1.0), (1.1, 1.0)])
    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    recorded: list[dict] = []

    def hook(view: EpochCheckpointView) -> None:
        recorded.append(
            {
                "global_epoch": len(view.history.train_losses),
                "epochs_without_improvement": view.epochs_without_improvement,
                "best_epoch": view.history.best_epoch,
                "best_state_dict": {name: tensor.clone() for name, tensor in view.best_state_dict.items()},
            }
        )

    result = run_training(model, train_loader, val_loader, config, checkpoint_hook=hook)
    history = result.history

    assert [r["global_epoch"] for r in recorded] == [1, 2, 3, 4]
    assert [r["epochs_without_improvement"] for r in recorded] == [0, 0, 1, 2]
    assert [r["best_epoch"] for r in recorded] == [1, 2, 2, 2]
    expected_best_param_value = [1.0, 2.0, 2.0, 2.0]
    for record, expected_value in zip(recorded, expected_best_param_value):
        for tensor in record["best_state_dict"].values():
            assert torch.all(tensor == expected_value)

    assert history.best_epoch == 2
    assert history.best_val_loss == 0.8
    # 마지막 hook 호출의 best_state_dict는 최종 best_state_dict와 일치해야 한다.
    last_best = recorded[-1]["best_state_dict"]
    for name, tensor in result.best_state_dict.items():
        assert torch.equal(tensor, last_best[name])


def test_run_training_checkpoint_hook_view_is_ephemeral_live_reference() -> None:
    """view.model/view.history는 매 호출마다 run_training() 내부의 같은
    객체를 가리키는 살아있는 참조여야 한다(identity 기반, 매 epoch 새로
    복사되지 않음). view는 hook 호출 범위에서만 유효한 ephemeral view라는
    계약이 있으므로, view.model/view.history 객체 자체를 hook 밖의
    리스트에 보관하지 않는다 -- 대신 hook 내부에서 identity 비교 결과
    (bool)와 `id(view.history)`처럼 이후에도 안전하게 재사용 가능한
    immutable 값만 기록한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    model_identity_results: list[bool] = []
    history_ids: list[int] = []

    def hook(view: EpochCheckpointView) -> None:
        model_identity_results.append(view.model is model)
        history_ids.append(id(view.history))

    run_training(model, train_loader, val_loader, config, checkpoint_hook=hook)

    assert len(model_identity_results) == config.epochs
    assert all(model_identity_results)
    assert len(set(history_ids)) == 1  # 매 호출이 같은 history 객체를 가리킴


def test_run_training_checkpoint_hook_called_before_progress_callback() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    call_order: list[str] = []

    run_training(
        model, train_loader, val_loader, config,
        checkpoint_hook=lambda view: call_order.append("hook"),
        progress_callback=lambda progress: call_order.append("progress"),
    )

    assert call_order == ["hook", "progress"] * config.epochs


def test_run_training_checkpoint_hook_runs_before_progress_callback_exception() -> None:
    """progress_callback이 예외를 던져도, 그 직전에 이미 실행된
    checkpoint_hook의 side effect는 그대로 남아 있어야 한다(§3-1의
    "hook을 먼저 실행하는 이유")."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)

    hook_calls = {"value": 0}

    def hook(view: EpochCheckpointView) -> None:
        hook_calls["value"] += 1

    def failing_progress_callback(progress) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_training(
            model, train_loader, val_loader, config,
            checkpoint_hook=hook, progress_callback=failing_progress_callback,
        )

    assert hook_calls["value"] == 1


def test_run_training_checkpoint_hook_stopped_by_user_always_false_in_view() -> None:
    """hook은 should_stop() 평가 이전에 실행되므로, view.history.stopped_by_user는
    should_stop이 True를 반환해 학습이 멈추는 epoch에서도 항상 False로
    보여야 한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2)

    seen_stopped_by_user: list[bool] = []
    stop_flag = {"value": False}

    def hook(view: EpochCheckpointView) -> None:
        seen_stopped_by_user.append(view.history.stopped_by_user)
        if len(view.history.train_losses) == 3:
            stop_flag["value"] = True

    result = run_training(
        model, train_loader, val_loader, config,
        checkpoint_hook=hook, should_stop=lambda: stop_flag["value"],
    )

    assert result.history.stopped_by_user is True
    assert all(value is False for value in seen_stopped_by_user)


def test_run_training_checkpoint_hook_loader_generator_matches_train_loader() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2)

    seen_generators = []

    def hook(view: EpochCheckpointView) -> None:
        seen_generators.append(view.loader_generator)

    run_training(model, train_loader, val_loader, config, checkpoint_hook=hook)

    assert all(generator is train_loader.generator for generator in seen_generators)
    assert train_loader.generator is not None


def test_run_training_checkpoint_hook_loader_generator_none_when_loader_has_no_generator() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_dataset, val_dataset = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=0, train_size=32, val_size=16
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)

    seen_generators = []
    run_training(
        model, train_loader, val_loader, config,
        checkpoint_hook=lambda view: seen_generators.append(view.loader_generator),
    )

    assert seen_generators == [None]


def test_run_training_checkpoint_hook_early_stopping_epoch_reports_stopped_early_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=2)
    fixed_val_results = iter([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.5, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    seen_stopped_early: list[bool] = []
    run_training(
        model, train_loader, val_loader, config,
        checkpoint_hook=lambda view: seen_stopped_early.append(view.history.stopped_early),
    )

    assert seen_stopped_early == [False, False, True]


def test_run_training_checkpoint_hook_exception_propagates_and_no_result_returned() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)

    def failing_hook(view: EpochCheckpointView) -> None:
        raise RuntimeError("checkpoint save failed")

    with pytest.raises(RuntimeError, match="checkpoint save failed"):
        run_training(model, train_loader, val_loader, config, checkpoint_hook=failing_hook)


def test_run_training_checkpoint_hook_called_exactly_once_when_epochs_is_one() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)

    call_count = {"value": 0}
    run_training(
        model, train_loader, val_loader, config,
        checkpoint_hook=lambda view: call_count.__setitem__("value", call_count["value"] + 1),
    )

    assert call_count["value"] == 1


def test_run_training_checkpoint_hook_captured_resume_state_matches_continuous_run() -> None:
    """자동 checkpoint(hook)에서 캡처한 상태로 resume한 결과가 continuous
    run과 값 기준으로 정확히 일치해야 한다. hook은 view를 읽기만 하는
    순수 구현이고, progress_callback/should_stop은 쓰지 않는다(§3-5
    전제)."""
    seed = 20260804
    spec = _dropout_mlp_classifier_spec()

    def make_loaders() -> tuple[DataLoader, DataLoader, torch.Generator]:
        train_dataset, val_dataset = make_train_val_datasets(
            spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader, generator

    config_kwargs = dict(
        batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.9,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    # (a) 연속 5 epoch
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = make_loaders()
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=5, **config_kwargs))

    # (b) 3 epoch 실행, checkpoint_hook으로 매 epoch 상태를 캡처(마지막 캡처만 사용)
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, _ = make_loaders()
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=3, **config_kwargs)

    captured: dict = {}

    def capture_hook(view: EpochCheckpointView) -> None:
        # config가 scheduler를 켰고 train_loader가 generator를 갖고
        # 있으므로, 이 두 필드가 None이면 캡처 로직 자체가 잘못된
        # 것이다 -- .state_dict()/.get_state()를 부르기 전에 명시적으로
        # 확인한다.
        assert view.scheduler is not None
        assert view.loader_generator is not None
        captured["history"] = copy.deepcopy(view.history)
        captured["best_state_dict"] = copy.deepcopy(view.best_state_dict)
        captured["epochs_without_improvement"] = view.epochs_without_improvement
        captured["optimizer_state_dict"] = copy.deepcopy(view.optimizer.state_dict())
        captured["scheduler_state_dict"] = copy.deepcopy(view.scheduler.state_dict())
        captured["loader_generator_state"] = view.loader_generator.get_state().clone()
        captured["cpu_rng_state"] = torch.get_rng_state().clone()

    run_training(model_b, train_loader_b, val_loader_b, first_config, checkpoint_hook=capture_hook)

    resume_state = TrainingResumeState(
        optimizer_state_dict=captured["optimizer_state_dict"],
        scheduler_state_dict=captured["scheduler_state_dict"],
        history=captured["history"],
        epochs_without_improvement=captured["epochs_without_improvement"],
        best_state_dict=captured["best_state_dict"],
        training_config=asdict(first_config),
    )

    model_b2 = build_model(spec)
    model_b2.load_state_dict(model_b.state_dict())
    train_dataset2, val_dataset2 = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(captured["loader_generator_state"])
    train_loader_b2 = DataLoader(
        train_dataset2, batch_size=8, shuffle=True, generator=restored_generator, drop_last=True
    )
    val_loader_b2 = DataLoader(val_dataset2, batch_size=8, shuffle=False)
    torch.set_rng_state(captured["cpu_rng_state"])

    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, TrainingConfig(epochs=2, **config_kwargs),
        resume_state=resume_state,
    )

    assert result_b2.history.train_losses == result_a.history.train_losses
    assert result_b2.history.val_losses == result_a.history.val_losses
    assert result_b2.history.val_accuracies == result_a.history.val_accuracies
    assert result_b2.history.best_epoch == result_a.history.best_epoch
    assert result_b2.epochs_without_improvement == result_a.epochs_without_improvement
    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_b2.state_dict()[name])
    for name, tensor in result_a.best_state_dict.items():
        assert torch.equal(tensor, result_b2.best_state_dict[name])
    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


def test_run_training_resume_scheduler_lr_reduction_crosses_checkpoint_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scheduler resume 계약은 "state_dict가 왕복된다"만으로는 부족하다 --
    checkpoint 이전에 쌓인 bad epoch 수가 resume 후에도 정확히 이어져서,
    실제로 같은 시점에 LR이 감소해야 한다.

    lr_scheduler_patience=2, val_loss를 계속 1.0으로 고정하면:
        epoch 1: baseline (아직 bad epoch 아님)
        epoch 2: bad 1
        epoch 3: bad 2
        checkpoint
        epoch 4: bad 3 > patience(2) -> LR 감소 (1.0 -> 0.5)

    연속 4 epoch와 (3 epoch 실행 + resume 1 epoch)를 비교해, 4번째
    epoch에서만 LR이 줄어들고 그 값과 scheduler 내부 상태가 두 경로에서
    정확히 같은지 확인한다 (실제로 PyTorch로 미리 실행해 4번째 step에서
    LR이 처음 바뀜을 확인한 순번 -- test_build_scheduler_reduces_lr_after_
    patience_bad_steps와 동일한 근거)."""
    spec = _mlp_classifier_spec()
    scheduler_kwargs = dict(
        batch_size=8, learning_rate=1.0, lr_scheduler="plateau",
        lr_scheduler_factor=0.5, lr_scheduler_patience=2,
    )

    # (a) 연속 4 epoch
    torch.manual_seed(0)
    model_a = build_model(spec)
    train_loader_a, val_loader_a = _make_loaders(spec, seed=0)
    val_results_a = iter([(1.0, 1.0)] * 4)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(val_results_a),
    )
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=4, **scheduler_kwargs))
    assert result_a.optimizer_state_dict["param_groups"][0]["lr"] == 0.5

    # (b) 3 epoch 실행 -- 아직 bad epoch 2개뿐이라 LR은 그대로여야 함
    torch.manual_seed(0)
    model_b = build_model(spec)
    train_loader_b, val_loader_b = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(epochs=3, **scheduler_kwargs)
    val_results_b1 = iter([(1.0, 1.0)] * 3)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(val_results_b1),
    )
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)
    assert result_b1.optimizer_state_dict["param_groups"][0]["lr"] == 1.0  # 아직 감소 전

    # resume 1 epoch -- 이 epoch가 3번째 bad epoch가 되어 LR이 감소해야 함
    resume_state = _make_resume_state(result_b1, first_config)
    resume_config = TrainingConfig(epochs=1, **scheduler_kwargs)
    val_results_b2 = iter([(1.0, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(val_results_b2),
    )
    result_b2 = run_training(model_b, train_loader_b, val_loader_b, resume_config, resume_state=resume_state)

    assert result_b2.optimizer_state_dict["param_groups"][0]["lr"] == 0.5
    _assert_deep_equal(result_a.optimizer_state_dict["param_groups"], result_b2.optimizer_state_dict["param_groups"])
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


# -- Phase 4I: progress_callback / should_stop --------------------------------


def test_run_training_callback_and_should_stop_none_by_default_no_behavior_change() -> None:
    """progress_callback/should_stop을 아예 넘기지 않는 것과 명시적으로
    None을 넘기는 것이 동일한 결과를 내야 한다 (기존 caller 전부가
    이 경로를 그대로 쓰므로, Phase 4I 도입으로 기존 동작이 바뀌면 안 됨)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model_a = build_model(spec)
    train_loader_a, val_loader_a = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)
    torch.manual_seed(0)
    result_a = run_training(model_a, train_loader_a, val_loader_a, config)

    torch.manual_seed(0)
    model_b = build_model(spec)
    train_loader_b, val_loader_b = _make_loaders(spec, seed=0)
    torch.manual_seed(0)
    result_b = run_training(
        model_b, train_loader_b, val_loader_b, config, progress_callback=None, should_stop=None
    )

    assert result_a.history.train_losses == result_b.history.train_losses
    assert result_a.history.val_losses == result_b.history.val_losses
    assert result_a.history.val_accuracies == result_b.history.val_accuracies
    for name, tensor in result_a.best_state_dict.items():
        assert torch.equal(tensor, result_b.best_state_dict[name])


def test_run_training_progress_callback_called_once_per_completed_epoch_with_matching_metrics() -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=4, batch_size=8, learning_rate=1e-2)

    progresses: list = []
    result = run_training(model, train_loader, val_loader, config, progress_callback=progresses.append)
    history = result.history

    assert len(progresses) == config.epochs
    for index, progress in enumerate(progresses):
        assert progress.run_epoch == index + 1
        assert progress.global_epoch == index + 1
        assert progress.total_run_epochs == config.epochs
        assert progress.train_loss == history.train_losses[index]
        assert progress.val_loss == history.val_losses[index]
        assert progress.val_accuracy == history.val_accuracies[index]
    last = progresses[-1]
    assert last.best_epoch == history.best_epoch
    assert last.best_val_loss == history.best_val_loss
    assert last.epochs_without_improvement == result.epochs_without_improvement
    assert last.stopped_early == history.stopped_early


def test_run_training_progress_callback_reports_global_epoch_on_resume() -> None:
    """resume 시 progress.run_epoch는 이번 호출 기준(1부터)이지만
    global_epoch은 절대 번호(이전 completed_epochs + run_epoch)여야 한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    first_config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)

    progresses: list = []
    second_config = TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2)
    run_training(
        model, train_loader, val_loader, second_config,
        resume_state=_make_resume_state(first, first_config),
        progress_callback=progresses.append,
    )

    assert [p.run_epoch for p in progresses] == [1, 2]
    assert [p.global_epoch for p in progresses] == [4, 5]
    assert [p.total_run_epochs for p in progresses] == [2, 2]


def test_run_training_progress_callback_learning_rate_captured_before_scheduler_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """learning_rate는 그 epoch의 train_one_epoch()가 실제로 사용한 값(=
    scheduler.step() 호출 전)이어야 한다. patience=2, val_loss를 계속
    1.0으로 고정하면 4번째 step()에서 처음 LR이 줄어든다
    (test_build_scheduler_reduces_lr_after_patience_bad_steps와 동일한 근거) --
    그 4번째 epoch의 progress.learning_rate는 여전히 줄어들기 *전* 값(1.0)을
    보고해야 하고, 그 이후(optimizer_state_dict)에는 줄어든 값(0.5)이 남아야 한다."""
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(
        epochs=4, batch_size=8, learning_rate=1.0,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=2,
    )
    fixed_val_results = iter([(1.0, 1.0)] * 4)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    progresses: list = []
    result = run_training(model, train_loader, val_loader, config, progress_callback=progresses.append)

    assert [p.learning_rate for p in progresses] == [1.0, 1.0, 1.0, 1.0]
    assert result.optimizer_state_dict["param_groups"][0]["lr"] == 0.5


def test_run_training_progress_callback_fires_even_on_early_stopping_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=2)
    fixed_val_results = iter([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.5, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    progresses: list = []
    result = run_training(model, train_loader, val_loader, config, progress_callback=progresses.append)

    assert len(progresses) == 3 == len(result.history.train_losses)
    assert progresses[-1].stopped_early is True
    assert progresses[0].stopped_early is False
    assert progresses[1].stopped_early is False


def test_run_training_progress_callback_exception_propagates_and_stops_immediately() -> None:
    """콜백이 예외를 던지면 그대로 전파되고(감싸지 않음), 그 epoch 이후로는
    더 이상 학습이 진행되지 않는다. 검증 가능한 외부 상태(콜백 호출 횟수,
    호출자가 들고 있는 model 참조)만 확인한다 -- 예외 시 TrainingResult가
    반환되지 않으므로 내부 history 객체는 애초에 접근할 수 없다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    initial_state = copy.deepcopy(model.state_dict())
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)

    call_count = {"value": 0}

    def failing_callback(progress) -> None:
        call_count["value"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_training(model, train_loader, val_loader, config, progress_callback=failing_callback)

    assert call_count["value"] == 1
    # model은 호출자가 넘긴 바로 그 참조이므로 예외 후에도 접근 가능하고,
    # 첫 epoch만큼은 실제로 학습되어 초기 상태와 달라져 있어야 한다.
    assert any(not torch.equal(initial_state[name], model.state_dict()[name]) for name in initial_state)


def test_run_training_should_stop_set_inside_callback_prevents_next_epoch() -> None:
    """콜백 안에서 동기적으로 stop 플래그를 세팅하면, should_stop()이 콜백
    직후 같은 epoch 경계에서 바로 평가되어 다음 epoch이 시작되지 않아야
    한다 (지연 없이 즉시 반영 -- 콜백이 should_stop보다 먼저 호출되는
    순서 자체를 검증)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2)

    stop_flag = {"value": False}

    def callback(progress) -> None:
        if progress.run_epoch == 3:
            stop_flag["value"] = True

    result = run_training(
        model, train_loader, val_loader, config,
        progress_callback=callback, should_stop=lambda: stop_flag["value"],
    )

    assert len(result.history.train_losses) == 3
    assert result.history.stopped_by_user is True
    assert result.history.stopped_early is False


def test_run_training_should_stop_true_before_first_epoch_still_runs_one_epoch() -> None:
    """should_stop이 처음부터(첫 콜백 전부터) True를 반환해도, epoch 시작
    전에는 절대 평가되지 않으므로 최소 1개 epoch은 항상 완료된다
    ("0 epoch 결과"는 불가능)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config, should_stop=lambda: True)

    assert len(result.history.train_losses) == 1
    assert result.history.stopped_by_user is True


def test_run_training_should_stop_not_evaluated_once_early_stopping_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """early stopping이 우선한다: 이미 stopped_early=True로 끝난 epoch에서는
    should_stop()을 평가하지 않는다. patience=2, val_loss=[1.0,1.0,1.0,0.5]면
    epoch 3에서 early stopping이 발동하므로 should_stop()은 epoch 1, 2에서만
    (2번) 평가되고, epoch 3에서는 호출되지 않아야 한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=10, batch_size=8, learning_rate=1e-2, early_stopping_patience=2)
    fixed_val_results = iter([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.5, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu", non_blocking=False: next(fixed_val_results),
    )

    should_stop_calls = {"value": 0}

    def should_stop() -> bool:
        should_stop_calls["value"] += 1
        return False

    result = run_training(model, train_loader, val_loader, config, should_stop=should_stop)

    assert len(result.history.train_losses) == 3
    assert result.history.stopped_early is True
    assert result.history.stopped_by_user is False
    assert should_stop_calls["value"] == 2


def test_run_training_should_stop_not_evaluated_on_last_requested_epoch() -> None:
    """이번 호출의 마지막 요청 epoch에서는 더 이상 건너뛸 epoch이 없으므로
    should_stop()을 평가하지 않는다. config.epochs=3이면 epoch 1, 2에서만
    (2번) 평가되고, epoch 3(마지막)에서는 호출되지 않는다 (should_stop은
    매번 False를 반환하므로 3 epoch 전부 실행되고, 호출 횟수만으로 "마지막
    epoch에서는 평가 자체가 없다"는 계약을 확인한다)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)

    should_stop_calls = {"value": 0}

    def should_stop() -> bool:
        should_stop_calls["value"] += 1
        return False

    result = run_training(model, train_loader, val_loader, config, should_stop=should_stop)

    assert len(result.history.train_losses) == 3
    assert should_stop_calls["value"] == 2
    assert result.history.stopped_by_user is False


def test_run_training_should_stop_not_evaluated_when_epochs_is_one() -> None:
    """config.epochs=1이면 유일한 epoch이 곧 마지막 요청 epoch이므로
    should_stop()은 절대 호출되지 않는다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)

    def should_stop() -> bool:
        raise AssertionError("should_stop must not be called when config.epochs == 1")

    result = run_training(model, train_loader, val_loader, config, should_stop=should_stop)

    assert len(result.history.train_losses) == 1
    assert result.history.stopped_by_user is False


def test_training_resume_state_accepts_stopped_by_user_history() -> None:
    """stopped_early와 달리 stopped_by_user=True인 history는 resume 대상으로
    거부되지 않아야 한다 (사용자가 잠시 멈춘 것뿐이므로 resume이 항상
    가능해야 함 -- Phase 4I의 핵심 목적)."""
    history = TrainingHistory(
        train_losses=[1.0], val_losses=[1.0], val_accuracies=[0.5],
        best_epoch=1, best_val_loss=1.0, stopped_by_user=True,
    )
    resume_state = TrainingResumeState(
        optimizer_state_dict={}, scheduler_state_dict=None,
        history=history, epochs_without_improvement=0, best_state_dict={},
        training_config=asdict(TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)),
    )
    assert resume_state.history.stopped_by_user is True


def test_run_training_resume_resets_stopped_by_user() -> None:
    """resume_state.history.stopped_by_user=True로 들어와도, 이번 호출은
    아직 멈춘 적이 없으므로 즉시 False로 리셋되어야 한다 -- 그러지 않으면
    이번 호출이 끝까지 다 돌았는데도(should_stop 없이) 이전 중단 상태가
    잘못 남는다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=5, batch_size=8, learning_rate=1e-2)

    stop_flag = {"value": False}

    def callback(progress) -> None:
        if progress.run_epoch == 2:
            stop_flag["value"] = True

    first = run_training(
        model, train_loader, val_loader, config,
        progress_callback=callback, should_stop=lambda: stop_flag["value"],
    )
    assert first.history.stopped_by_user is True
    assert len(first.history.train_losses) == 2

    second = run_training(
        model, train_loader, val_loader,
        TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2),
        resume_state=_make_resume_state(first, config),
    )

    assert second.history.stopped_by_user is False
    assert len(second.history.train_losses) == 4


def test_run_training_user_stop_then_resume_matches_continuous_run_exactly() -> None:
    """stopped_by_user 경로의 checkpoint/resume도 stopped_early 경로
    (test_run_training_resume_matches_continuous_run_exactly)와 동일한
    exact-resume 계약을 만족해야 한다 -- 중단 방식만 config.epochs로 자르는
    대신 should_stop 콜백으로 3 epoch 후 중단시킨다. Dropout이 포함된
    모델을 써서(전역 CPU RNG 소비) RNG state 복원이 실제로 필요함을 함께
    검증한다."""
    seed = 20260803
    spec = _dropout_mlp_classifier_spec()

    def make_loaders() -> tuple[DataLoader, DataLoader, torch.Generator]:
        train_dataset, val_dataset = make_train_val_datasets(
            spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        return train_loader, val_loader, generator

    config_kwargs = dict(
        batch_size=8, learning_rate=1e-2, optimizer="sgd", momentum=0.9,
        lr_scheduler="plateau", lr_scheduler_factor=0.5, lr_scheduler_patience=1,
    )

    # (a) 연속 5 epoch (should_stop 없이)
    torch.manual_seed(seed)
    model_a = build_model(spec)
    train_loader_a, val_loader_a, _ = make_loaders()
    torch.manual_seed(seed)
    result_a = run_training(model_a, train_loader_a, val_loader_a, TrainingConfig(epochs=5, **config_kwargs))

    # (b) should_stop으로 3 epoch 후 중단 (config.epochs=10이지만 실제로는 3에서 멈춤)
    torch.manual_seed(seed)
    model_b = build_model(spec)
    train_loader_b, val_loader_b, generator_b = make_loaders()
    torch.manual_seed(seed)
    first_config = TrainingConfig(epochs=10, **config_kwargs)

    stop_flag = {"value": False}

    def progress_callback(progress) -> None:
        if progress.run_epoch == 3:
            stop_flag["value"] = True

    result_b1 = run_training(
        model_b, train_loader_b, val_loader_b, first_config,
        progress_callback=progress_callback, should_stop=lambda: stop_flag["value"],
    )
    assert len(result_b1.history.train_losses) == 3
    assert result_b1.history.stopped_by_user is True
    assert result_b1.history.stopped_early is False

    # checkpoint 시점 상태를 caller가 직접 채취 (run_training()이 파일을 쓰지 않으므로)
    loader_generator_state = generator_b.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    resume_state = _make_resume_state(result_b1, first_config)

    # "새 프로세스"를 흉내: 새 model/DataLoader/generator를 만들고 저장된 상태를 복원
    model_b2 = build_model(spec)
    model_b2.load_state_dict(model_b.state_dict())
    train_dataset2, val_dataset2 = make_train_val_datasets(
        spec.input_shape, NUM_CLASSES, seed=seed, train_size=32, val_size=16
    )
    restored_generator = torch.Generator()
    restored_generator.set_state(loader_generator_state)
    train_loader_b2 = DataLoader(
        train_dataset2, batch_size=8, shuffle=True, generator=restored_generator, drop_last=True
    )
    val_loader_b2 = DataLoader(val_dataset2, batch_size=8, shuffle=False)
    torch.set_rng_state(cpu_rng_state)

    result_b2 = run_training(
        model_b2, train_loader_b2, val_loader_b2, TrainingConfig(epochs=2, **config_kwargs),
        resume_state=resume_state,
    )

    assert result_b2.history.stopped_by_user is False
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

    _assert_deep_equal(result_a.optimizer_state_dict, result_b2.optimizer_state_dict)
    _assert_deep_equal(result_a.scheduler_state_dict, result_b2.scheduler_state_dict)


# -- Phase 4S/4T: precision / autocast_dtype / GradScaler -----------------------


def test_build_precision_execution_returns_none_none_for_fp32_precision() -> None:
    """precision="fp32"(기본값)이면 device 문자열이 "cuda"여도 CUDA API를
    전혀 건드리지 않고 (None, None)을 반환한다 -- config.precision 검사가
    device 검사보다 먼저이므로 CUDA 하드웨어 없이도 이 분기만은 항상
    안전하게 실행 가능하다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, precision="fp32")
    assert _build_precision_execution(config, "cpu") == (None, None)
    assert _build_precision_execution(config, "cuda") == (None, None)


def test_build_precision_execution_rejects_unsupported_precision_value() -> None:
    """`_build_precision_execution()`은 "fp32"/"fp16"/"bf16" 세 값을
    명시적으로 분기하고, 그 외 값은 "fp32도 fp16도 아니면 bf16"처럼
    implicit dispatch로 조용히 처리하지 않고 ValueError로 명확히
    거부해야 한다 -- 향후 TrainingConfig.PRECISION_CHOICES에 새 값이
    추가되는데 이 함수 수정이 누락되면, 새 precision이 엉뚱한 기존
    분기로 조용히 처리되는 회귀를 방지하기 위한 fail-fast 계약이다.
    TrainingConfig.__post_init__이 이미 정상 값만 허용하므로, 이 값은
    검증을 우회해 인위적으로 만든다(production에서는 도달하지 않는
    경로를 이 함수 자신의 invariant로 직접 고정)."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, precision="fp16")
    config.precision = "fp8"  # TrainingConfig.__post_init__ 검증을 우회
    with pytest.raises(ValueError, match="precision"):
        _build_precision_execution(config, "cuda")


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_build_precision_execution_rejects_cpu_device(precision: str) -> None:
    """precision="fp16"/"bf16"인데 device="cpu"면 조용히 FP32로 대체하지
    않고 ValueError로 명확히 거부한다(CPU AMP는 이번 Phase 범위 밖) --
    ImageFolderWorkflowRequest를 거치지 않고 이 함수(또는
    run_training())를 직접 호출하는 generic caller까지 이 규칙을
    강제하기 위한 lower-level invariant다. 이 분기는 실제
    torch.amp.GradScaler("cuda")를 생성하지 않으므로 CUDA 하드웨어 없이
    안전하다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, precision=precision)
    with pytest.raises(ValueError, match="precision"):
        _build_precision_execution(config, "cpu")


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("device", ["mps", "xpu"])
def test_build_precision_execution_rejects_non_cuda_backend_strings(precision: str, device: str) -> None:
    """precision="fp16"/"bf16"인데 device가 "cpu"가 아닌 다른 non-CUDA
    backend 문자열(예: "mps"/"xpu")이어도 ValueError로 명확히 거부해야
    한다 -- Phase 4S 최초 구현은 `device == "cpu"`만 검사해서, CUDA가
    전혀 아닌 이런 backend 문자열이 조용히 torch.amp.GradScaler("cuda")
    를 받아버리는 버그가 있었다(실측으로 재현). fp16/bf16 둘 다 이
    함수 하나의 동일한 CUDA 판별을 거치므로 같은 실수가 bf16에서
    반복되지 않는다. 실제 MPS/XPU 하드웨어 없이도 이 문자열 비교만으로
    안전하게 검증 가능하다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, precision=precision)
    with pytest.raises(ValueError, match="precision"):
        _build_precision_execution(config, device)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_run_training_rejects_non_cuda_backend_with_amp_precision_without_workflow(
    precision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generic run_training() 회귀 계약(CPU 케이스의 일반화, fp16/bf16
    공통) -- device="mps"처럼 이 프로젝트가 인식하지 않는 non-CUDA
    backend 문자열로 TrainingConfig(precision=...)+run_training()을
    직접 호출해도 silent fallback 없이 _build_precision_execution()
    단계에서 거부되어야 한다. train_one_epoch()가 호출되지 않음을
    monkeypatch로 직접 증명해 실제 batch forward 전에 거부됨을
    보장한다."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError(f"train_one_epoch() must not be called when precision={precision!r}+device='mps'")

    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fail_if_called)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, precision=precision)

    with pytest.raises(ValueError, match="precision"):
        run_training(model, train_loader, val_loader, config, device="mps")


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_run_training_rejects_cpu_device_with_amp_precision_without_workflow(
    precision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generic run_training() 회귀 계약(fp16/bf16 공통) --
    ImageFolderWorkflowRequest/_validate_precision_device_compatibility()
    를 거치지 않고 TrainingConfig(precision=...)+device="cpu"로
    run_training()을 직접 호출해도 silent FP32 fallback이 일어나면 안
    된다. 실제 batch forward(train_one_epoch)가 시작되기 전에 거부됨을
    monkeypatch로 직접 증명한다."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError(f"train_one_epoch() must not be called when precision={precision!r}+device='cpu'")

    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fail_if_called)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, precision=precision)

    with pytest.raises(ValueError, match="precision"):
        run_training(model, train_loader, val_loader, config, device="cpu")


def test_train_one_epoch_default_scaler_none_matches_omitted_argument() -> None:
    """scaler=None을 명시하는 것과 아예 생략하는 것이 완전히 동일한 FP32
    경로를 타야 한다(하위호환 계약) -- 같은 seed로 두 번 학습해 손실이
    정확히 일치하는지로 증명한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model_a = build_model(spec)
    train_loader_a, _ = _make_loaders(spec, seed=0)

    torch.manual_seed(0)
    model_b = build_model(spec)
    train_loader_b, _ = _make_loaders(spec, seed=0)

    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    torch.manual_seed(1)
    loss_omitted = train_one_epoch(model_a, train_loader_a, optimizer_a)
    torch.manual_seed(1)
    loss_explicit_none = train_one_epoch(model_b, train_loader_b, optimizer_b, scaler=None)

    assert loss_omitted == loss_explicit_none
    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_b.state_dict()[name])


def test_train_one_epoch_fp32_path_never_calls_amp_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """scaler=None(FP32, 기본값)이면 train_one_epoch()이 torch.amp.autocast를
    전혀 호출하지 않아야 한다 -- monkeypatch로 "호출되면 fail"을 강제해
    production CPU/CUDA FP32 경로가 새 AMP 코드 경로를 아예 거치지 않는다는
    계약을 직접 고정한다."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("torch.amp.autocast must not be called when scaler is None")

    monkeypatch.setattr("torch.amp.autocast", _fail_if_called)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, _ = _make_loaders(spec, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, train_loader, optimizer)  # scaler 생략 -- 기본값 None

    assert math.isfinite(loss)


def test_run_training_default_precision_produces_no_scaler_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """production run_training()을 기본 TrainingConfig(precision="fp32")로
    호출하면 TrainingResult.scaler_state_dict가 None이고, AMP API가 전혀
    호출되지 않아야 한다."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("torch.amp API must not be called when precision='fp32'")

    monkeypatch.setattr("torch.amp.autocast", _fail_if_called)
    monkeypatch.setattr("torch.amp.GradScaler", _fail_if_called)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config)

    assert result.scaler_state_dict is None


def test_training_result_default_scaler_state_is_none() -> None:
    """TrainingResult를 기존(Phase 4S 이전) 위치/키워드 인자만으로 만들어도
    scaler_state_dict가 기본값 None으로 채워져야 한다(하위호환 계약)."""
    result = TrainingResult(
        history=TrainingHistory(),
        best_state_dict={},
        optimizer_state_dict={},
        scheduler_state_dict=None,
        epochs_without_improvement=0,
    )
    assert result.scaler_state_dict is None


def test_training_resume_state_default_scaler_state_is_none() -> None:
    """TrainingResumeState도 마찬가지로 기존 keyword 인자만으로 생성 가능하고
    scaler_state_dict 기본값이 None이어야 한다(하위호환 계약)."""
    history = TrainingHistory(
        train_losses=[1.0], val_losses=[1.0], val_accuracies=[0.5], best_epoch=1, best_val_loss=1.0
    )
    resume_state = TrainingResumeState(
        optimizer_state_dict={},
        scheduler_state_dict=None,
        history=history,
        epochs_without_improvement=0,
        best_state_dict={},
        training_config={},
    )
    assert resume_state.scaler_state_dict is None


def test_epoch_checkpoint_view_default_scaler_state_is_none() -> None:
    """EpochCheckpointView도 기존 keyword 인자만으로 생성 가능하고
    scaler_state_dict 기본값이 None이어야 한다(하위호환 계약)."""
    view = EpochCheckpointView(
        model=build_model(_mlp_classifier_spec()),
        history=TrainingHistory(),
        best_state_dict={},
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]),
        scheduler=None,
        epochs_without_improvement=0,
        loader_generator=None,
    )
    assert view.scaler_state_dict is None


def test_build_precision_execution_passes_device_type_only_to_gradscaler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """precision="fp16"+ordinal이 붙은 training device("cuda:0"/"cuda:1"
    등)를 받아도 `torch.amp.GradScaler`에는 공식 public API contract에
    맞는 device type `"cuda"`만 전달돼야 한다(GradScaler.__init__
    docstring: "Possible values are: 'cuda' and 'cpu'" -- ordinal이
    붙은 문자열은 문서화된 값이 아니다). PyTorch private attribute
    (`scaler._device`)에 의존하지 않고, `torch.amp.GradScaler` 생성자에
    실제로 전달된 인자를 monkeypatch로 직접 가로채 검증한다 -- 이렇게
    하면 실제 CUDA 하드웨어 없이도(fake GradScaler라 GPU를 전혀
    건드리지 않음) 이 계약을 CPU-only로 안전하게 고정할 수 있다.
    production code는 이 테스트를 위해 변경하지 않았다."""
    captured: dict = {}

    class _FakeGradScaler:
        def __init__(self, device, *args, **kwargs):
            captured["device"] = device

    monkeypatch.setattr(torch.amp, "GradScaler", _FakeGradScaler)

    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, precision="fp16")
    for device in ("cuda", "cuda:0", "cuda:1"):
        captured.clear()
        autocast_dtype, scaler = _build_precision_execution(config, device)
        assert autocast_dtype is torch.float16
        assert isinstance(scaler, _FakeGradScaler)
        assert captured["device"] == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_build_precision_execution_returns_float16_and_gradscaler_for_cuda_fp16() -> None:
    """optional CUDA smoke test -- precision="fp16"+device="cuda"/"cuda:0"
    양쪽 모두 실제(fake가 아닌) `torch.amp.GradScaler` 인스턴스가
    정상적으로 만들어짐을 실제 GPU에서 확인한다(ordinal이 붙은 device
    문자열도 그대로 받아들이는지 실측 확인 -- 별도 ordinal-specific
    scaler 설계는 없다). GradScaler에 어떤 device 인자가 실제로
    전달되는지의 contract 자체는 CPU-only monkeypatch 테스트
    (`test_build_precision_execution_passes_device_type_only_to_gradscaler`)
    가 검증한다 -- 이 테스트는 그 결과로 실제 GradScaler가 정상
    생성/동작하는지의 functional 확인만 담당한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, precision="fp16")
    for device in ("cuda", "cuda:0"):
        autocast_dtype, scaler = _build_precision_execution(config, device)
        assert autocast_dtype is torch.float16
        assert isinstance(scaler, torch.amp.GradScaler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_build_precision_execution_returns_bfloat16_and_no_scaler_for_cuda_bf16() -> None:
    """optional CUDA smoke test(Phase 4T) -- precision="bf16"+device="cuda"
    /"cuda:0"는 (torch.bfloat16, None)을 반환해야 한다 -- BF16은
    GradScaler를 쓰지 않는다는 핵심 계약을 직접 고정한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, precision="bf16")
    for device in ("cuda", "cuda:0"):
        autocast_dtype, scaler = _build_precision_execution(config, device)
        assert autocast_dtype is torch.bfloat16
        assert scaler is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_run_training_cuda_fp16_amp_completes_one_epoch_with_clipping() -> None:
    """optional CUDA smoke test(Phase 4S) -- CUDA FP16 AMP + gradient
    clipping이 함께 정상 완료되는지 최소 확인한다(Phase 4M의 clip_grad_norm_
    이 scaler.unscale_() 이후 정확히 호출되는 실제 production 경로).
    same-device exact-resume 전체 증명은 test_imagefolder_workflow.py의
    production workflow 테스트가 담당한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec).to("cuda")
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-2, gradient_clip_norm=1.0, precision="fp16"
    )

    result = run_training(model, train_loader, val_loader, config, device="cuda")

    assert len(result.history.train_losses) == 1
    assert math.isfinite(result.history.train_losses[0])
    assert math.isfinite(result.history.val_losses[0])
    # scaler_state_dict의 정확한 내부 key 구성(scale/growth_factor/...)은
    # PyTorch 버전에 따라 달라질 수 있는 implementation detail이라
    # regression contract로 강제하지 않는다(checkpoint loader의 최소
    # 검증 철학과 동일) -- 여기서는 "실제로 non-empty scaler state가
    # 존재한다"만 확인한다.
    assert result.scaler_state_dict  # not None, not empty


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_run_training_cuda_fp16_amp_resume_accepts_missing_scaler_state() -> None:
    """optional CUDA smoke test -- resume_state.scaler_state_dict가 None이어도
    (legacy 또는 FP32 checkpoint를 AMP로 resume하는 상황을 흉내) config가
    precision="fp16"이면 에러 없이 fresh scaler로 진행되어야 한다(비대칭
    처리 계약, scheduler의 엄격한 mismatch 검증과 다름)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec).to("cuda")
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, precision="fp16")

    result_a = run_training(model, train_loader, val_loader, config, device="cuda")
    resume_state = _make_resume_state(result_a, config)
    resume_state.scaler_state_dict = None  # legacy/FP32 checkpoint를 흉내

    model_b = build_model(spec).to("cuda")
    model_b.load_state_dict(model.state_dict())
    train_loader_b, val_loader_b = _make_loaders(spec, seed=0)

    result_b = run_training(
        model_b, train_loader_b, val_loader_b, TrainingConfig(epochs=1, **{
            k: v for k, v in asdict(config).items() if k != "epochs"
        }),
        device="cuda", resume_state=resume_state,
    )

    assert result_b.scaler_state_dict is not None  # fresh scaler로 시작했지만 여전히 존재


# -- Phase 4U: non_blocking wiring (CUDA H2D transfer optimization) ---------


def _spy_tensor_to(monkeypatch: pytest.MonkeyPatch, sink: list[bool | None]) -> None:
    """`torch.Tensor.to()` 호출에 실제로 전달되는 `non_blocking` kwarg를
    `sink`에 기록만 하고 원래 동작으로 위임한다(Phase 4U). 이 프로젝트의
    최소 CPU fixture(nn.Linear 등)는 forward 안에서 `.to()`를 호출하지
    않으므로, `train_one_epoch()`/`evaluate()`의 `images.to(...)`/
    `labels.to(...)` 두 호출만 안전하게 가로챌 수 있다."""
    original_to = torch.Tensor.to

    def spying_to(self, *args, **kwargs):
        sink.append(kwargs.get("non_blocking"))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spying_to)


@pytest.mark.parametrize("non_blocking", [False, True])
def test_train_one_epoch_forwards_non_blocking_to_tensor_to(
    monkeypatch: pytest.MonkeyPatch, non_blocking: bool
) -> None:
    """`train_one_epoch(..., non_blocking=...)`이 실제로 `images.to(device,
    non_blocking=...)`/`labels.to(device, non_blocking=...)` 호출에 그
    값을 그대로 전달하는지 CPU에서 직접 고정한다(CPU에서도 인자 자체는
    전달되어야 한다 -- 실제 비동기 여부와 무관하게 wiring 계약)."""
    recorded: list[bool | None] = []
    _spy_tensor_to(monkeypatch, recorded)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, _val_loader = _make_loaders(spec, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_one_epoch(model, train_loader, optimizer, non_blocking=non_blocking)

    calls_with_kwarg = [value for value in recorded if value is not None]
    assert len(calls_with_kwarg) > 0, "images.to()/labels.to() must pass non_blocking explicitly"
    assert all(value == non_blocking for value in calls_with_kwarg)


@pytest.mark.parametrize("non_blocking", [False, True])
def test_evaluate_forwards_non_blocking_to_tensor_to(
    monkeypatch: pytest.MonkeyPatch, non_blocking: bool
) -> None:
    """`evaluate(..., non_blocking=...)`도 `train_one_epoch()`과 동일한
    의미로 `.to()` 호출에 그 값을 전달하는지 직접 고정한다(validation도
    training epoch 중 같은 effective 값을 받는다는 §5 wiring 계약의
    evaluate() 쪽 절반)."""
    recorded: list[bool | None] = []
    _spy_tensor_to(monkeypatch, recorded)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    _train_loader, val_loader = _make_loaders(spec, seed=0)

    evaluate(model, val_loader, non_blocking=non_blocking)

    calls_with_kwarg = [value for value in recorded if value is not None]
    assert len(calls_with_kwarg) > 0, "images.to()/labels.to() must pass non_blocking explicitly"
    assert all(value == non_blocking for value in calls_with_kwarg)


def test_train_one_epoch_non_blocking_defaults_to_false() -> None:
    """기본값(`non_blocking` 생략)에서 Phase 4A~4T까지의 기존 FP32 CPU
    경로가 그대로 재현되는지 확인한다 -- `non_blocking=False`로 명시한
    호출과 결과가 완전히 동일해야 한다(existing caller 하위호환)."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model_a = build_model(spec)
    train_loader_a, _val_loader_a = _make_loaders(spec, seed=0)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-2)
    loss_omitted = train_one_epoch(model_a, train_loader_a, optimizer_a)  # non_blocking 생략

    torch.manual_seed(0)
    model_b = build_model(spec)
    train_loader_b, _val_loader_b = _make_loaders(spec, seed=0)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)
    loss_explicit_false = train_one_epoch(model_b, train_loader_b, optimizer_b, non_blocking=False)

    assert loss_omitted == loss_explicit_false


def test_run_training_forwards_non_blocking_to_train_one_epoch_and_evaluate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_training()이 자신의 `non_blocking` 인자를 매 epoch의
    `train_one_epoch()`/`evaluate()` 호출 둘 다에 그대로 전달하는
    연결 계약을 직접 고정한다(criterion 배선 테스트와 동일한 근거 --
    각 함수가 자기 자신의 non_blocking 인자를 올바르게 처리하는지와,
    run_training()이 실제로 그 값을 넘기는지는 서로 다른 실패 지점이다)."""
    captured: dict = {}

    def fake_train_one_epoch(
        model, loader, optimizer, device="cpu", gradient_clip_norm=None, criterion=None,
        autocast_dtype=None, scaler=None, non_blocking=False,
    ):
        captured["train_non_blocking"] = non_blocking
        return 0.5

    def fake_evaluate(model, loader, device="cpu", non_blocking=False):
        captured["evaluate_non_blocking"] = non_blocking
        return 0.5, 0.5

    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr("image_ai_studio.training.loop.evaluate", fake_evaluate)

    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec)
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)

    run_training(model, train_loader, val_loader, config, non_blocking=True)

    assert captured["train_non_blocking"] is True
    assert captured["evaluate_non_blocking"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_run_training_cuda_non_blocking_completes_one_epoch_with_finite_loss() -> None:
    """optional CUDA smoke test(Phase 4U) -- non_blocking=True로 generic
    run_training()이 한 epoch 정상 완료되는지 최소 확인한다. production
    workflow 경로의 pin_memory+non_blocking exact-resume 전체 증명은
    test_imagefolder_workflow.py의
    test_workflow_cuda_pin_memory_non_blocking_resume_boundary_option_change_exact_resume
    가 담당한다."""
    torch.manual_seed(0)
    spec = _mlp_classifier_spec()
    model = build_model(spec).to("cuda")
    train_loader, val_loader = _make_loaders(spec, seed=0)
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2)

    result = run_training(model, train_loader, val_loader, config, device="cuda", non_blocking=True)

    assert len(result.history.train_losses) == 1
    assert math.isfinite(result.history.train_losses[0])
    assert math.isfinite(result.history.val_losses[0])
