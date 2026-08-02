"""TrainingConfig 검증 테스트."""
from __future__ import annotations

import pytest

from image_ai_studio.training.config import (
    TrainingConfig,
    TrainingConfigError,
    require_compatible_resume_config,
)


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


def test_default_config_reproduces_phase_4a_4d_behavior() -> None:
    """Phase 4E 신규 필드는 전부 기존 동작(Adam, scheduler 없음, early
    stopping 없음)을 그대로 재현하는 기본값이어야 한다 -- 기존 호출부가
    수정 없이 그대로 동작한다는 회귀 계약."""
    config = TrainingConfig(epochs=3, batch_size=8, learning_rate=1e-3)
    assert config.optimizer == "adam"
    assert config.lr_scheduler is None
    assert config.early_stopping_patience is None


# -- Phase 4E: optimizer -----------------------------------------------------


def test_accepts_adam_optimizer() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, optimizer="adam")
    assert config.optimizer == "adam"


def test_accepts_sgd_optimizer() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, optimizer="sgd", momentum=0.5)
    assert config.optimizer == "sgd"
    assert config.momentum == 0.5


def test_rejects_unknown_optimizer() -> None:
    with pytest.raises(TrainingConfigError, match="optimizer"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, optimizer="rmsprop")


# -- Phase 4E: momentum (optimizer와 무관하게 항상 검증) ----------------------


def test_rejects_negative_momentum() -> None:
    with pytest.raises(TrainingConfigError, match="momentum"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, momentum=-0.1)


def test_rejects_momentum_greater_or_equal_to_one() -> None:
    with pytest.raises(TrainingConfigError, match="momentum"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, momentum=1.0)


def test_accepts_zero_momentum() -> None:
    """momentum=0.0은 경계값(0.0 <= momentum)이라 허용되어야 한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, momentum=0.0)
    assert config.momentum == 0.0


def test_momentum_is_validated_even_when_optimizer_is_adam() -> None:
    """optimizer="adam"이라 momentum이 실제로 쓰이지 않아도 검증은
    동일하게 적용된다 -- TrainingConfig가 항상 일관된 값을 갖도록 함."""
    with pytest.raises(TrainingConfigError, match="momentum"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, optimizer="adam", momentum=2.0)


# -- Phase 4E: lr_scheduler ---------------------------------------------------


def test_accepts_no_scheduler() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, lr_scheduler=None)
    assert config.lr_scheduler is None


def test_accepts_plateau_scheduler() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, lr_scheduler="plateau")
    assert config.lr_scheduler == "plateau"


def test_rejects_unknown_scheduler() -> None:
    with pytest.raises(TrainingConfigError, match="lr_scheduler"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, lr_scheduler="step")


@pytest.mark.parametrize("factor", [0.0, 1.0, -0.5, 1.5])
def test_rejects_out_of_range_scheduler_factor(factor: float) -> None:
    with pytest.raises(TrainingConfigError, match="lr_scheduler_factor"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, lr_scheduler_factor=factor)


@pytest.mark.parametrize("patience", [0, -1])
def test_rejects_non_positive_scheduler_patience(patience: int) -> None:
    with pytest.raises(TrainingConfigError, match="lr_scheduler_patience"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, lr_scheduler_patience=patience)


# -- Phase 4E: early_stopping_patience ---------------------------------------


def test_accepts_no_early_stopping() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, early_stopping_patience=None)
    assert config.early_stopping_patience is None


def test_accepts_positive_early_stopping_patience() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, early_stopping_patience=3)
    assert config.early_stopping_patience == 3


@pytest.mark.parametrize("patience", [0, -1])
def test_rejects_non_positive_early_stopping_patience(patience: int) -> None:
    with pytest.raises(TrainingConfigError, match="early_stopping_patience"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, early_stopping_patience=patience)


# -- Phase 4F: require_compatible_resume_config -------------------------------


def test_require_compatible_resume_config_rejects_non_dict_checkpoint_config() -> None:
    """checkpoint_config가 dict가 아니면(예: TrainingResumeState를 파일
    경유 없이 직접 만들면서 training_config=None을 넘긴 경우) TypeError가
    아니라 명확한 ValueError를 내야 한다."""
    with pytest.raises(ValueError, match="training_config must be a dict"):
        require_compatible_resume_config(None, TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3))
