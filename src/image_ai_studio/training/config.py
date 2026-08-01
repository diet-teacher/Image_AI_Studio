"""학습 설정. loss=CrossEntropyLoss는 여전히 고정(Phase 4E도 classification
전용, 선택 registry 없음). Phase 4E부터 optimizer(Adam/SGD)와 LR scheduler
(없음/ReduceLROnPlateau), early stopping을 이 dataclass에서 선택할 수 있다
-- 실제 생성은 registry가 아니라 loop.py의 private helper(_build_optimizer/
_build_scheduler)가 담당한다 (선택지가 2개/1개뿐이라 registry는 과설계)."""
from __future__ import annotations

from dataclasses import dataclass

OPTIMIZER_CHOICES = ("adam", "sgd")
LR_SCHEDULER_CHOICES = ("plateau",)


class TrainingConfigError(ValueError):
    """TrainingConfig 값이 잘못됐을 때 발생."""


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingConfigError(f"'{name}' must be a positive integer, got {value!r}")


def _require_positive_float(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise TrainingConfigError(f"'{name}' must be a positive number, got {value!r}")


def _require_fraction(name: str, value: object, *, low_inclusive: bool) -> None:
    """0과 1 사이의 값만 허용 (momentum=[0,1), factor=(0,1))."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigError(f"'{name}' must be a number, got {value!r}")
    value = float(value)
    lower_ok = value >= 0.0 if low_inclusive else value > 0.0
    if not (lower_ok and value < 1.0):
        bound = "[0.0, 1.0)" if low_inclusive else "(0.0, 1.0)"
        raise TrainingConfigError(f"'{name}' must be in {bound}, got {value!r}")


def _require_one_of(name: str, value: object, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise TrainingConfigError(f"'{name}' must be one of {choices}, got {value!r}")


@dataclass
class TrainingConfig:
    """epochs/batch_size/learning_rate는 필수. optimizer/scheduler/early
    stopping은 전부 선택(옵션)이며, 기본값은 Phase 4A~4D의 기존 동작
    (Adam, scheduler 없음, early stopping 없음)을 그대로 재현한다."""

    epochs: int
    batch_size: int
    learning_rate: float

    optimizer: str = "adam"  # "adam" | "sgd"
    momentum: float = 0.9  # optimizer="sgd"일 때만 사용 (Adam이어도 항상 유효 범위 검증)

    lr_scheduler: str | None = None  # None | "plateau"
    lr_scheduler_factor: float = 0.1  # lr_scheduler="plateau"일 때만 사용
    lr_scheduler_patience: int = 1  # 〃

    early_stopping_patience: int | None = None  # None => 비활성화 (현재 동작)

    def __post_init__(self) -> None:
        _require_positive_int("epochs", self.epochs)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_float("learning_rate", self.learning_rate)

        _require_one_of("optimizer", self.optimizer, OPTIMIZER_CHOICES)
        # optimizer가 "adam"이어도 momentum을 항상 검증한다 -- TrainingConfig가
        # 어떤 optimizer를 고르든 항상 일관되게 유효한 값을 갖도록 하기 위함
        # (GUI에서 momentum을 먼저 조절하고 optimizer를 나중에 바꾸는 순서도
        # 자연스럽게 허용됨).
        _require_fraction("momentum", self.momentum, low_inclusive=True)

        if self.lr_scheduler is not None:
            _require_one_of("lr_scheduler", self.lr_scheduler, LR_SCHEDULER_CHOICES)
        _require_fraction("lr_scheduler_factor", self.lr_scheduler_factor, low_inclusive=False)
        _require_positive_int("lr_scheduler_patience", self.lr_scheduler_patience)

        if self.early_stopping_patience is not None:
            _require_positive_int("early_stopping_patience", self.early_stopping_patience)
