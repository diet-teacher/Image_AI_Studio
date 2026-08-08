"""train_one_epoch / evaluate / run_training 테스트.

여기서는 학습 루프 메커니즘 자체(파라미터 변화, loss 감소, evaluate의 부작용
없음)만 검증한다. BatchNorm/Dropout/ResidualBlockSpec/BranchSpec 각각의
학습 동작 검증은 test_model_definition_integration.py에서 다룬다.
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import pytest
import torch
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

    def fake_train_one_epoch(model, loader, optimizer, device="cpu", gradient_clip_norm=None):
        call_count["value"] += 1
        epoch_value = float(call_count["value"])
        for param in model.parameters():
            param.data.fill_(epoch_value)
        return epoch_value  # train_loss 값 자체는 이 테스트의 관심사가 아님

    fixed_val_results = iter([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.5, 1.0)])
    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
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
        lambda model, loader, device="cpu": next(first_val_results),
    )
    first_config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)
    assert first.history.best_epoch == 3

    second_val_results = iter([(0.5, 1.0), (0.99, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(second_val_results),
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
        lambda model, loader, device="cpu": next(first_val_results),
    )
    first_config = TrainingConfig(epochs=2, batch_size=8, learning_rate=1e-2)
    first = run_training(model, train_loader, val_loader, first_config)
    first_best_snapshot = {name: tensor.clone() for name, tensor in first.best_state_dict.items()}

    second_val_results = iter([(0.9, 1.0), (0.8, 1.0)])  # 둘 다 0.5보다 나쁨 -> 개선 없음
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(second_val_results),
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
        lambda model, loader, device="cpu": next(first_val_results),
    )
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-2, early_stopping_patience=3)
    first = run_training(model, train_loader, val_loader, config)
    assert first.history.stopped_early is False
    assert first.epochs_without_improvement == 2

    # resume 후 1 epoch만 더 개선 실패하면 카운터가 3 == patience가 되어 즉시 중단돼야 함
    second_val_results = iter([(1.0, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(second_val_results),
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

    def fake_train_one_epoch(model, loader, optimizer, device="cpu", gradient_clip_norm=None):
        call_count["value"] += 1
        epoch_value = float(call_count["value"])
        for param in model.parameters():
            param.data.fill_(epoch_value)
        return epoch_value

    fixed_val_results = iter([(1.0, 1.0), (0.8, 1.0), (0.9, 1.0), (1.1, 1.0)])
    monkeypatch.setattr("image_ai_studio.training.loop.train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(fixed_val_results),
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
        lambda model, loader, device="cpu": next(fixed_val_results),
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
        lambda model, loader, device="cpu": next(val_results_a),
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
        lambda model, loader, device="cpu": next(val_results_b1),
    )
    result_b1 = run_training(model_b, train_loader_b, val_loader_b, first_config)
    assert result_b1.optimizer_state_dict["param_groups"][0]["lr"] == 1.0  # 아직 감소 전

    # resume 1 epoch -- 이 epoch가 3번째 bad epoch가 되어 LR이 감소해야 함
    resume_state = _make_resume_state(result_b1, first_config)
    resume_config = TrainingConfig(epochs=1, **scheduler_kwargs)
    val_results_b2 = iter([(1.0, 1.0)])
    monkeypatch.setattr(
        "image_ai_studio.training.loop.evaluate",
        lambda model, loader, device="cpu": next(val_results_b2),
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
        lambda model, loader, device="cpu": next(fixed_val_results),
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
        lambda model, loader, device="cpu": next(fixed_val_results),
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
        lambda model, loader, device="cpu": next(fixed_val_results),
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
