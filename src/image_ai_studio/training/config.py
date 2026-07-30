"""학습 설정. optimizer=Adam, loss=CrossEntropyLoss로 고정 (선택 registry 없음, Phase 4A 범위)."""
from __future__ import annotations

from dataclasses import dataclass


class TrainingConfigError(ValueError):
    """TrainingConfig 값이 잘못됐을 때 발생."""


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingConfigError(f"'{name}' must be a positive integer, got {value!r}")


def _require_positive_float(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise TrainingConfigError(f"'{name}' must be a positive number, got {value!r}")


@dataclass
class TrainingConfig:
    """epochs/batch_size/learning_rate만 지원. optimizer/loss는 loop.py에 Adam/CrossEntropyLoss로 고정."""

    epochs: int
    batch_size: int
    learning_rate: float

    def __post_init__(self) -> None:
        _require_positive_int("epochs", self.epochs)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_float("learning_rate", self.learning_rate)
