"""학습/평가 루프. loss=CrossEntropyLoss는 고정. optimizer(Adam/SGD)와 LR
scheduler(없음/ReduceLROnPlateau)는 Phase 4E부터 TrainingConfig로 선택
가능 -- 선택지가 2개/1개뿐이라 registry 없이 이 모듈의 private helper
(_build_optimizer/_build_scheduler)로 충분하다."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_ai_studio.training.config import TrainingConfig


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """config.optimizer에 따라 Adam 또는 SGD를 생성. TrainingConfig.__post_init__이
    이미 "adam"/"sgd" 외의 값을 거부하므로 그 외 분기는 없다."""
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """config.lr_scheduler가 None이면 scheduler를 아예 만들지 않는다 (기존
    동작 재현). "plateau"만 지원 -- StepLR 등 metric에 의존하지 않는
    scheduler는 이번 Phase 범위 밖."""
    if config.lr_scheduler is None:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience
    )


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu"
) -> float:
    """1 epoch 학습 (model.train() -> forward -> CrossEntropyLoss -> backward -> step). 반환값: epoch 평균 loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    if total_samples == 0:
        raise ValueError("train_one_epoch: loader produced no samples (empty DataLoader)")

    return total_loss / total_samples


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[float, float]:
    """model.eval()로 validation loss/accuracy 계산 (parameter 변경 없음). 반환값: (avg_loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    if total_samples == 0:
        raise ValueError("evaluate: loader produced no samples (empty DataLoader)")

    return total_loss / total_samples, total_correct / total_samples


@dataclass
class TrainingHistory:
    """run_training()의 epoch별 결과 + best epoch 추적.

    JSON으로 그대로 직렬화 가능한 순수 메트릭만 담는다 (텐서 값인
    state_dict는 여기 두지 않고 TrainingResult가 별도로 가진다 --
    training/history.py의 save_training_history()가 이 dataclass를
    그대로 asdict()해서 저장한다).

    best_epoch는 1-indexed. val_loss가 strict하게(<) 더 낮아질 때만
    갱신되므로, 동률이면 먼저 나온 epoch가 유지된다.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    stopped_early: bool = False


@dataclass
class TrainingResult:
    """run_training()의 전체 반환값.

    history(JSON 직렬화 가능)와 best_state_dict(텐서, 메모리에만 존재)를
    분리했다 -- run_training()은 파일을 쓰지 않으므로, 파일 저장은 이
    두 필드를 받은 호출자(예: run_training_e2e.py)의 책임이다.
    """

    history: TrainingHistory
    best_state_dict: dict[str, torch.Tensor]


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: str = "cpu",
) -> TrainingResult:
    """train_one_epoch/evaluate를 config.epochs만큼 반복하는 얇은 조립 함수.

    매 epoch validation loss가 지금까지의 최솟값보다 strict하게 낮아지면
    (`val_loss < best_val_loss`), 그 시점의 model.state_dict()를 깊은
    복사로 메모리에 보관한다 -- 이후 epoch에서도 model은 계속 학습되어
    바뀌므로, best 시점의 가중치를 남기려면 그 순간 복사해야 한다
    (참조만 보관하면 마지막 epoch 값으로 덮어써진다). 파일 저장은 하지
    않는다 (호출자 책임).

    optimizer/scheduler는 config에 따라 여기서 1회만 생성한다 (resume
    없음 -- optimizer/scheduler state는 저장/재로드되지 않고 이 함수
    호출 안에서만 존재한다). model은 호출 전에 이미 device로 옮겨져
    있어야 함.

    매 epoch의 순서는 train -> validation -> history 기록 -> best
    model/개선 카운터 갱신 -> scheduler.step(val_loss) -> early stopping
    조건 확인이다. 즉 마지막으로 실행된 epoch에서 scheduler가 LR을
    바꿨더라도, 그 직후 early stopping으로 멈추면 바뀐 LR은 실제로
    쓰이지 않을 수 있다 -- 이는 의도된 동작이다(다음 epoch이 없으므로
    바뀐 LR을 쓸 기회 자체가 없을 뿐, 계산 자체는 정상 수행됨).
    `config.early_stopping_patience`와 `config.lr_scheduler_patience`를
    함께 쓸 때는 early_stopping_patience를 lr_scheduler_patience보다
    크게 잡는 것을 권장한다 -- 그래야 LR이 줄어든 뒤에도 실제로 몇
    epoch 더 학습할 기회가 생긴다 (강제 검증 규칙은 아님).
    """
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    history = TrainingHistory()
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        history.train_losses.append(train_one_epoch(model, train_loader, optimizer, device=device))
        val_loss, val_accuracy = evaluate(model, val_loader, device=device)
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_accuracy)

        if history.best_val_loss is None or val_loss < history.best_val_loss:
            history.best_epoch = epoch
            history.best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            history.stopped_early = True
            break

    # config.epochs >= 1은 TrainingConfig.__post_init__이 이미 보장하므로
    # 루프가 최소 1회는 돌고, epoch 1에서 best_val_loss가 None이라 항상
    # best_state_dict가 채워진다.
    return TrainingResult(history=history, best_state_dict=best_state_dict)
