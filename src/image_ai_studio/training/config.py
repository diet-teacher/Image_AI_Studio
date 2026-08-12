"""학습 설정. loss=CrossEntropyLoss는 여전히 고정(Phase 4E도 classification
전용, 선택 registry 없음). Phase 4E부터 optimizer(Adam/SGD)와 LR scheduler
(없음/ReduceLROnPlateau), early stopping을 이 dataclass에서 선택할 수 있다
-- 실제 생성은 registry가 아니라 loop.py의 private helper(_build_optimizer/
_build_scheduler)가 담당한다 (선택지가 2개/1개뿐이라 registry는 과설계).
Phase 4L부터 optimizer에 AdamW가 추가되고, weight_decay(Adam/SGD/AdamW
공통 적용)를 선택할 수 있다. Phase 4M부터 gradient norm clipping
(gradient_clip_norm)을 선택할 수 있다 -- optimizer의 param_groups와
무관한 순수 runtime 파라미터라 RESUME_CONFIG_FIELDS에 포함되지 않는다.
Phase 4N부터 label smoothing(label_smoothing)을 선택할 수 있다 --
training loss에만 적용되고(loop.py의 evaluate()는 항상 unsmoothed
CrossEntropyLoss를 쓴다), gradient_clip_norm과 동일한 이유로
RESUME_CONFIG_FIELDS에 포함되지 않는다. Phase 4P부터 class별 명시적
weight(class_weights)를 선택할 수 있다 -- label_smoothing과 마찬가지로
training loss(CrossEntropyLoss(weight=...))에만 적용되고, criterion은
매번 config로 새로 생성될 뿐 checkpoint subsystem이 그 state를 저장/
복원하지 않으므로 RESUME_CONFIG_FIELDS에 포함되지 않는다. Phase 4S부터
precision("fp32"|"fp16")을 선택할 수 있다 -- CUDA training에서만 의미가
있고, non-CUDA device+"fp16" 조합은 ImageFolderWorkflowRequest를
거치는 workflow 레벨(`_validate_precision_device_compatibility()`,
이 프로젝트가 다루는 device 값 중 "cpu"만 해당)과, workflow를 우회한
generic `run_training()` 호출(`"cpu"`뿐 아니라 이 프로젝트가 인식하지
않는 다른 backend 문자열까지 포함) 양쪽 모두에서 거부된다 -- loop.py의
`_build_precision_execution()` 참고. optimizer/scheduler의
load_state_dict() 호환성에 영향을 주지 않음을 실측으로 확인했으므로
(현재 지원 optimizer와 실측한 AMP 경로에서는 momentum buffer 등
optimizer state가 precision과 무관하게 float32로 유지됨)
gradient_clip_norm/label_smoothing/class_weights와 같은 이유로
RESUME_CONFIG_FIELDS에 포함되지 않는다. Phase 4T부터 precision("bf16")
도 선택할 수 있다 -- fp16과 달리 CUDA BF16 autocast만 쓰고
`torch.amp.GradScaler`는 쓰지 않는다(BF16은 FP32와 같은 exponent-bit
폭을 가져 FP16의 좁은 dynamic range 문제를 크게 완화하고, 이
프로젝트의 실제 BF16 학습 경로에서 GradScaler 없이 정상 학습/
exact-resume이 실측으로 확인됨 -- BF16에 이론적으로 절대 underflow가
없다는 뜻은 아니다). device 조합 검증 정책은 fp16과 완전히
동일하다(non-CUDA device는 workflow/generic run_training() 양쪽에서
거부)."""
from __future__ import annotations

import math
from dataclasses import dataclass

OPTIMIZER_CHOICES = ("adam", "sgd", "adamw")
LR_SCHEDULER_CHOICES = ("plateau",)
# Phase 4S: CUDA FP16 autocast + GradScaler 지원. Phase 4T: CUDA BF16
# autocast(GradScaler 없음) 추가 -- CPU AMP는 여전히 non-goal(docs/
# phase4t_cuda_bf16_mixed_precision_design.md 참고).
PRECISION_CHOICES = ("fp32", "fp16", "bf16")

# Phase 4F: resume 시 checkpoint와 반드시 일치해야 하는 필드.
# optimizer.load_state_dict()/scheduler.load_state_dict()가 다른 종류의
# optimizer/scheduler 사이에서 안전하게 동작하지 않으므로(값이 잘못
# 섞이거나 저장된 param group 값이 새 값을 조용히 덮어쓰므로), 그 구조를
# 결정하는 필드들을 여기서 명시적으로 관리한다. epochs/early_stopping_patience는
# 의도적으로 제외 -- resume마다 자유롭게 바꿀 수 있다.
#
# 이 helper는 config.py에 둔다 -- loop.py(run_training())와 checkpoint.py
# (checkpoint 파일 조회) 둘 다 이 검증이 필요한데, config.py는 이미 둘 중
# 어느 쪽도 import하지 않는 가장 아래 계층이라 여기 두면 순환 의존 없이
# 양쪽에서 그대로 가져다 쓸 수 있다.
RESUME_CONFIG_FIELDS = (
    "optimizer",
    "learning_rate",
    "momentum",
    "weight_decay",
    "lr_scheduler",
    "lr_scheduler_factor",
    "lr_scheduler_patience",
    "batch_size",
)

# Phase 4L: weight_decay는 RESUME_CONFIG_FIELDS에 속하면서도 Phase 4L
# 이전에 저장된 checkpoint에는 키 자체가 없을 수 있는 유일한 필드다 -- 그런
# checkpoint는 weight_decay=0.0으로 학습된 것으로 간주한다. 이 dict가 "어떤
# 필드가 이 예외를 갖는지"와 "누락 시 어떤 값으로 간주하는지"의 단일 출처다
# -- require_compatible_resume_config()(아래)와 checkpoint.py의
# load_training_checkpoint() 둘 다 이 하나의 dict를 참조해야 한다. 두 곳이
# 각자 "weight_decay"를 하드코딩해 서로 다른 예외 목록을 갖게 되면(Phase 4L
# 최초 구현에서 실제로 발생했던 회귀), checkpoint 파일을 통한 실제 resume
# 경로에서 이 migration 정책이 조용히 깨질 수 있다. 다른 필드를 여기 추가하지
# 말 것 -- 이 예외는 weight_decay 하나에만 좁게 적용되도록 의도됐다.
RESUME_CONFIG_LEGACY_DEFAULTS: dict[str, object] = {
    "weight_decay": 0.0,
}


class TrainingConfigError(ValueError):
    """TrainingConfig 값이 잘못됐을 때 발생."""


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingConfigError(f"'{name}' must be a positive integer, got {value!r}")


def _require_positive_float(name: str, value: object) -> None:
    # math.isfinite()를 명시적으로 검사한다 -- NaN/+inf는 <= 0.0과 비교하면
    # 둘 다 False가 되어(NaN과의 비교는 항상 False, +inf는 0.0보다 큼) 그냥
    # float(value) <= 0.0만 검사하면 조용히 통과해버린다.
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise TrainingConfigError(f"'{name}' must be a finite positive number, got {value!r}")


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


def _require_non_negative_finite_float(name: str, value: object) -> None:
    """0 이상의 유한한 실수만 허용 (weight_decay -- 상한은 두지 않는다,
    실제 값이 학습에 적절한지는 사용자의 hyperparameter 선택 책임)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigError(f"'{name}' must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise TrainingConfigError(f"'{name}' must be a finite number >= 0.0, got {value!r}")


def _require_closed_unit_interval(name: str, value: object) -> None:
    """[0.0, 1.0] 양끝 포함 (label_smoothing). 상/하한 비교 자체가 NaN을
    걸러내므로(NaN과의 모든 비교는 False) math.isfinite()가 따로 필요
    없다 -- +inf/-inf도 각각 상한/하한 비교에서 자연히 거부된다."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 <= float(value) <= 1.0):
        raise TrainingConfigError(f"'{name}' must be in [0.0, 1.0], got {value!r}")


def _require_positive_finite_float(name: str, value: object) -> None:
    """0보다 큰 유한한 실수만 허용 (gradient_clip_norm -- 상한은 두지
    않는다). _require_positive_float()와 달리 math.isfinite()로 NaN/+inf도
    명시적으로 거부한다 -- NaN <= 0.0과 inf <= 0.0이 파이썬에서 둘 다
    False라서, `float(value) <= 0.0`만 검사하는 _require_positive_float()는
    이 두 값을 조용히 통과시킨다(이 helper가 그 함수를 대체하지 않는 이유)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigError(f"'{name}' must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise TrainingConfigError(f"'{name}' must be a finite positive number, got {value!r}")


def _require_class_weights(name: str, value: object) -> None:
    """class_weights 전용 검증(Phase 4P) -- None(비활성)이 아니면 반드시
    tuple이어야 하고(공식 representation, list 등 다른 sequence는 거부 --
    canonicalization은 CLI 경계의 책임이지 이 dataclass의 책임이 아니다),
    비어 있지 않아야 하며, 각 원소는 finite + strictly positive(0 이하/
    NaN/+-inf 거부)여야 한다. PyTorch CrossEntropyLoss(weight=...)는 이
    값들을 constructor/forward 어디서도 검증하지 않고(0/음수/NaN/inf
    전부 조용히 통과시켜 NaN loss나 잘못된 부호의 finite loss를 낼 수
    있음을 직접 실측 확인함), 그 방어를 이 helper가 대신한다."""
    if not isinstance(value, tuple):
        raise TrainingConfigError(f"'{name}' must be a tuple of positive numbers or None, got {value!r}")
    if len(value) == 0:
        raise TrainingConfigError(f"'{name}' must not be empty, got {value!r}")
    for index, element in enumerate(value):
        if (
            not isinstance(element, (int, float))
            or isinstance(element, bool)
            or not math.isfinite(float(element))
            or float(element) <= 0.0
        ):
            raise TrainingConfigError(f"'{name}[{index}]' must be a finite positive number, got {element!r}")


@dataclass
class TrainingConfig:
    """epochs/batch_size/learning_rate는 필수. optimizer/scheduler/early
    stopping은 전부 선택(옵션)이며, 기본값은 Phase 4A~4D의 기존 동작
    (Adam, scheduler 없음, early stopping 없음)을 그대로 재현한다."""

    epochs: int
    batch_size: int
    learning_rate: float

    optimizer: str = "adam"  # "adam" | "sgd" | "adamw"
    momentum: float = 0.9  # optimizer="sgd"일 때만 사용 (Adam/AdamW여도 항상 유효 범위 검증)
    weight_decay: float = 0.0  # Adam/SGD/AdamW 공통 적용 (Phase 4L)

    gradient_clip_norm: float | None = None  # None => clipping 비활성화 (Phase 4M)

    label_smoothing: float = 0.0  # [0.0, 1.0], training loss에만 적용 (Phase 4N)

    class_weights: tuple[float, ...] | None = None  # class별 명시적 weight, training loss에만 적용 (Phase 4P)

    lr_scheduler: str | None = None  # None | "plateau"
    lr_scheduler_factor: float = 0.1  # lr_scheduler="plateau"일 때만 사용
    lr_scheduler_patience: int = 1  # 〃

    early_stopping_patience: int | None = None  # None => 비활성화 (현재 동작)

    precision: str = "fp32"  # "fp32" | "fp16" | "bf16", CUDA에서만 "fp16"/"bf16" 허용 (Phase 4S/4T)

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
        # optimizer와 무관하게 항상 검증 (momentum과 같은 이유).
        _require_non_negative_finite_float("weight_decay", self.weight_decay)

        if self.gradient_clip_norm is not None:
            _require_positive_finite_float("gradient_clip_norm", self.gradient_clip_norm)

        # optimizer/scheduler와 무관하게 항상 검증 (momentum/weight_decay와 같은 이유).
        _require_closed_unit_interval("label_smoothing", self.label_smoothing)

        if self.class_weights is not None:
            _require_class_weights("class_weights", self.class_weights)

        if self.lr_scheduler is not None:
            _require_one_of("lr_scheduler", self.lr_scheduler, LR_SCHEDULER_CHOICES)
        _require_fraction("lr_scheduler_factor", self.lr_scheduler_factor, low_inclusive=False)
        _require_positive_int("lr_scheduler_patience", self.lr_scheduler_patience)

        if self.early_stopping_patience is not None:
            _require_positive_int("early_stopping_patience", self.early_stopping_patience)

        # device와 무관하게 항상 검증 (momentum/weight_decay와 같은 이유) --
        # device 조합("cpu"+"fp16" 거부)은 device를 모르는 이 dataclass가
        # 아니라 workflow 레벨(imagefolder_workflow.py)의 책임이다.
        _require_one_of("precision", self.precision, PRECISION_CHOICES)


def require_compatible_resume_config(checkpoint_config: dict, resume_config: TrainingConfig) -> None:
    """resume에 사용할 TrainingConfig가 checkpoint 저장 당시의 config와
    optimizer/scheduler 구조 관련 필드(RESUME_CONFIG_FIELDS)에서 일치하는지
    확인한다.

    PyTorch의 optimizer.load_state_dict()/scheduler.load_state_dict()는
    다른 종류의 optimizer/scheduler 사이에서 안전하게 동작하지 않고,
    저장된 param group 값(learning_rate/momentum 등)을 그대로 복원해
    새 config 값을 조용히 덮어쓰므로, 여기서 먼저 명확한 에러로 걸러낸다.
    batch_size가 바뀌면 같은 sample 순서라도 batch 구성과 optimizer step
    수가 달라지므로 이것도 강제 일치 대상이다. epochs/early_stopping_patience
    는 의도적으로 비교 대상에서 제외한다 -- resume마다 자유롭게 바꿀 수 있다.

    이 함수는 `run_training()`이 resume_state를 받을 때 항상 스스로
    호출한다 (loop.py) -- caller가 별도로 이 함수를 부르는 것은 조기
    검증(fail fast)을 위한 선택 사항일 뿐, config 호환성이 실제로
    강제되는 지점은 이 함수를 항상 거치는 `run_training()` 내부다.

    `checkpoint_config`가 dict가 아닌 경우(예: TrainingResumeState를
    파일 경유 없이 직접 만들면서 training_config=None을 넘긴 경우)를
    가장 먼저 검사한다 -- 그러지 않으면 아래 `in checkpoint_config`에서
    TypeError가 나서, 이 함수가 항상 명확한 ValueError만 낸다는 계약이
    깨진다.

    RESUME_CONFIG_LEGACY_DEFAULTS(위)에 있는 필드(현재 weight_decay만)는
    예외다: Phase 4L 이전에 저장된 checkpoint에는 이 키가 없을 수 있으므로,
    그런 경우에는 checkpoint가 해당 기본값으로 학습된 것으로 간주한다
    (checkpoint_config 자체는 mutate하지 않는다). 다른 필드가 누락된 경우는
    기존과 동일하게 항상 거부한다 -- 이 예외를 다른 필드로 일반화하지 말 것.
    """
    if not isinstance(checkpoint_config, dict):
        raise ValueError(
            f"checkpoint training_config must be a dict, got {type(checkpoint_config).__name__}"
        )

    strictly_required_fields = [
        name for name in RESUME_CONFIG_FIELDS if name not in RESUME_CONFIG_LEGACY_DEFAULTS
    ]
    missing = [name for name in strictly_required_fields if name not in checkpoint_config]
    if missing:
        raise ValueError(f"checkpoint training_config is missing required field(s): {missing}")

    for field_name in RESUME_CONFIG_FIELDS:
        if field_name in RESUME_CONFIG_LEGACY_DEFAULTS and field_name not in checkpoint_config:
            saved_value = RESUME_CONFIG_LEGACY_DEFAULTS[field_name]
        else:
            saved_value = checkpoint_config[field_name]
        new_value = getattr(resume_config, field_name)
        if saved_value != new_value:
            raise ValueError(
                f"cannot resume: checkpoint was saved with {field_name}={saved_value!r} "
                f"but resume config uses {field_name}={new_value!r}"
            )
