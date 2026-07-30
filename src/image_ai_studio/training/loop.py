"""학습/평가 루프. optimizer=Adam, loss=CrossEntropyLoss로 고정 (Phase 4A 범위, 선택 불가)."""
from __future__ import annotations

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
    """run_training()의 epoch별 결과."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: str = "cpu",
) -> TrainingHistory:
    """train_one_epoch/evaluate를 config.epochs만큼 반복하는 얇은 조립 함수.

    optimizer(Adam, config.learning_rate)는 여기서 1회만 생성 (resume 없음).
    model은 호출 전에 이미 device로 옮겨져 있어야 함.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = TrainingHistory()

    for _ in range(config.epochs):
        history.train_losses.append(train_one_epoch(model, train_loader, optimizer, device=device))
        val_loss, val_accuracy = evaluate(model, val_loader, device=device)
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_accuracy)

    return history
