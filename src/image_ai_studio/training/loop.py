"""학습/평가 루프. optimizer=Adam, loss=CrossEntropyLoss로 고정 (Phase 4A 범위, 선택 불가)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_ai_studio.training.config import TrainingConfig


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

    optimizer(Adam, config.learning_rate)는 여기서 1회만 생성 (resume 없음).
    model은 호출 전에 이미 device로 옮겨져 있어야 함.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = TrainingHistory()
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        history.train_losses.append(train_one_epoch(model, train_loader, optimizer, device=device))
        val_loss, val_accuracy = evaluate(model, val_loader, device=device)
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_accuracy)

        if history.best_val_loss is None or val_loss < history.best_val_loss:
            history.best_epoch = epoch
            history.best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

    # config.epochs >= 1은 TrainingConfig.__post_init__이 이미 보장하므로
    # 루프가 최소 1회는 돌고, epoch 1에서 best_val_loss가 None이라 항상
    # best_state_dict가 채워진다.
    return TrainingResult(history=history, best_state_dict=best_state_dict)
