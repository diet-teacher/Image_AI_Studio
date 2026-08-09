"""TrainingConfig 검증 테스트."""
from __future__ import annotations

import pytest

from image_ai_studio.training.config import (
    RESUME_CONFIG_FIELDS,
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


@pytest.mark.parametrize("learning_rate", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_learning_rate(learning_rate: float) -> None:
    """_require_positive_float()는 예전에 float(value) <= 0.0만 검사해서
    NaN/+inf를 조용히 통과시켰다(NaN과의 비교, +inf와 0.0의 비교가 각각
    False라서) -- math.isfinite() 검사를 추가해 셋 다 거부한다."""
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=learning_rate)


def test_rejects_bool_learning_rate() -> None:
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=True)  # type: ignore[arg-type]


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


def test_accepts_adamw_optimizer() -> None:
    """Phase 4L: AdamW 추가."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, optimizer="adamw")
    assert config.optimizer == "adamw"


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


# -- Phase 4L: weight_decay (optimizer와 무관하게 항상 검증, 상한 없음) -------


def test_weight_decay_defaults_to_zero() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3)
    assert config.weight_decay == 0.0


def test_accepts_zero_weight_decay() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, weight_decay=0.0)
    assert config.weight_decay == 0.0


@pytest.mark.parametrize("weight_decay", [1e-4, 0.5, 1.0, 5.0])
def test_accepts_positive_weight_decay_with_no_upper_bound(weight_decay: float) -> None:
    """임의의 상한을 두지 않는다 -- 1.0 이상의 값도 허용해야 한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, weight_decay=weight_decay)
    assert config.weight_decay == weight_decay


def test_rejects_negative_weight_decay() -> None:
    with pytest.raises(TrainingConfigError, match="weight_decay"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, weight_decay=-0.1)


@pytest.mark.parametrize("weight_decay", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_weight_decay(weight_decay: float) -> None:
    with pytest.raises(TrainingConfigError, match="weight_decay"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, weight_decay=weight_decay)


def test_rejects_bool_weight_decay() -> None:
    with pytest.raises(TrainingConfigError, match="weight_decay"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, weight_decay=True)  # type: ignore[arg-type]


def test_weight_decay_is_validated_even_when_optimizer_is_adam() -> None:
    with pytest.raises(TrainingConfigError, match="weight_decay"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, optimizer="adam", weight_decay=-1.0)


# -- Phase 4M: gradient_clip_norm (None=비활성화, 0 초과 유한값만 허용, 상한 없음) --


def test_gradient_clip_norm_defaults_to_none() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3)
    assert config.gradient_clip_norm is None


def test_accepts_none_gradient_clip_norm() -> None:
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, gradient_clip_norm=None)
    assert config.gradient_clip_norm is None


@pytest.mark.parametrize("gradient_clip_norm", [1e-12, 0.1, 1.0, 5.0, 1000.0])
def test_accepts_positive_gradient_clip_norm_with_no_upper_bound(gradient_clip_norm: float) -> None:
    """임의의 상한을 두지 않는다 -- 1.0 이상의 값도 허용해야 한다."""
    config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, gradient_clip_norm=gradient_clip_norm)
    assert config.gradient_clip_norm == gradient_clip_norm


def test_rejects_zero_gradient_clip_norm() -> None:
    with pytest.raises(TrainingConfigError, match="gradient_clip_norm"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, gradient_clip_norm=0.0)


def test_rejects_negative_gradient_clip_norm() -> None:
    with pytest.raises(TrainingConfigError, match="gradient_clip_norm"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, gradient_clip_norm=-1.0)


def test_rejects_bool_gradient_clip_norm() -> None:
    with pytest.raises(TrainingConfigError, match="gradient_clip_norm"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, gradient_clip_norm=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("gradient_clip_norm", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_gradient_clip_norm(gradient_clip_norm: float) -> None:
    """gradient_clip_norm은 양의 유한값만 허용하며 NaN/+inf/-inf를
    모두 거부한다."""
    with pytest.raises(TrainingConfigError, match="gradient_clip_norm"):
        TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, gradient_clip_norm=gradient_clip_norm)


def test_gradient_clip_norm_is_not_a_resume_config_field() -> None:
    """gradient_clip_norm은 optimizer의 param_groups에 속하지 않는 순수
    runtime 파라미터라 optimizer.load_state_dict()가 조용히 덮어쓸 위험이
    없다 -- weight_decay/momentum/learning_rate와 달리 RESUME_CONFIG_FIELDS
    에 포함되지 않는다는 public contract를 직접 고정한다(설계 문서 §7)."""
    assert "gradient_clip_norm" not in RESUME_CONFIG_FIELDS


@pytest.mark.parametrize(
    ("saved_gradient_clip_norm", "resume_gradient_clip_norm"),
    [(None, 0.5), (1.0, 0.5), (1.0, None)],
)
def test_require_compatible_resume_config_allows_gradient_clip_norm_to_differ(
    saved_gradient_clip_norm: float | None, resume_gradient_clip_norm: float | None
) -> None:
    """gradient_clip_norm은 resume 시 자유롭게 바꿀 수 있어야 한다(양방향).
    실제 checkpoint에는 TrainingConfig 전체가 asdict()로 그대로 저장되므로
    이 필드도 저장되지만, RESUME_CONFIG_FIELDS에 없으므로 비교 대상이
    아니다 -- 아래 checkpoint_config에 이 필드를 넣어도(가짜 dict가
    실제 저장 형태를 흉내낸 것) 값이 다르면 거부될 것이라고 착각하지
    않도록, 실제로는 무시된 채 통과함을 이 테스트가 고정한다."""
    checkpoint_config = {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "lr_scheduler": None,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 1,
        "batch_size": 8,
        # 참고 목적으로만 저장되고 비교에는 쓰이지 않는다 (RESUME_CONFIG_FIELDS 밖).
        "gradient_clip_norm": saved_gradient_clip_norm,
    }
    resume_config = TrainingConfig(
        epochs=1, batch_size=8, learning_rate=1e-3, gradient_clip_norm=resume_gradient_clip_norm
    )

    require_compatible_resume_config(checkpoint_config, resume_config)  # raise 없이 통과해야 함


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


def _base_checkpoint_config_without_weight_decay() -> dict:
    """Phase 4L 이전에 저장된 checkpoint의 training_config를 흉내낸다 --
    weight_decay 키가 아예 없다."""
    return {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "momentum": 0.9,
        "lr_scheduler": None,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 1,
        "batch_size": 8,
    }


def test_require_compatible_resume_config_allows_missing_weight_decay_when_resume_config_is_zero() -> None:
    """Phase 4L: weight_decay 키가 없는 과거 checkpoint는 weight_decay=0.0으로
    학습된 것으로 간주하므로, 새 config도 0.0이면 resume이 허용되어야 한다."""
    checkpoint_config = _base_checkpoint_config_without_weight_decay()
    original = dict(checkpoint_config)
    resume_config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, weight_decay=0.0)

    require_compatible_resume_config(checkpoint_config, resume_config)

    assert checkpoint_config == original  # checkpoint_config 자체는 mutate되지 않는다


def test_require_compatible_resume_config_rejects_missing_weight_decay_when_resume_config_is_nonzero() -> None:
    """weight_decay 키가 없는 과거 checkpoint(=0.0으로 간주)를 weight_decay>0.0
    으로 resume하려 하면 값이 실제로 달라지므로 거부해야 한다."""
    checkpoint_config = _base_checkpoint_config_without_weight_decay()
    original = dict(checkpoint_config)
    resume_config = TrainingConfig(epochs=1, batch_size=8, learning_rate=1e-3, weight_decay=0.1)

    with pytest.raises(ValueError, match="weight_decay"):
        require_compatible_resume_config(checkpoint_config, resume_config)

    assert checkpoint_config == original  # checkpoint_config 자체는 mutate되지 않는다
