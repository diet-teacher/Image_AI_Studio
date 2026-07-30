"""TrainingConfig 검증 테스트."""
from __future__ import annotations

import pytest

from image_ai_studio.training.config import TrainingConfig, TrainingConfigError


@pytest.mark.parametrize("epochs", [0, -1])
def test_rejects_non_positive_epochs(epochs: int) -> None:
    with pytest.raises(TrainingConfigError, match="epochs"):
        TrainingConfig(epochs=epochs, batch_size=8, learning_rate=1e-3)


@pytest.mark.parametrize("batch_size", [0, -4])
def test_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(TrainingConfigError, match="batch_size"):
        TrainingConfig(epochs=1, batch_size=batch_size, learning_rate=1e-3)


@pytest.mark.parametrize("learning_rate", [0.0, -0.1])
def test_rejects_non_positive_learning_rate(learning_rate: float) -> None:
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=learning_rate)


def test_rejects_bool_epochs() -> None:
    """bool은 int의 서브클래스라서 isinstance(True, int)가 True로 나옴 -- 별도 차단 필요."""
    with pytest.raises(TrainingConfigError, match="epochs"):
        TrainingConfig(epochs=True, batch_size=8, learning_rate=1e-3)  # type: ignore[arg-type]


def test_accepts_valid_config() -> None:
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-3)
    assert config.epochs == 3
    assert config.batch_size == 8
    assert config.learning_rate == 1e-3
