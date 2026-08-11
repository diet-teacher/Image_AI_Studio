"""모델 상태 저장/재로드. 두 가지 서로 다른 용도를 구분한다:

- save_state_dict()/load_state_dict() (Phase 4A) -- bare model state_dict만.
  best model 배포/export용 아티팩트. optimizer/scheduler/epoch 등은
  포함하지 않는다. 기존 동작/포맷 변경 없음.
- save_training_checkpoint()/load_training_checkpoint() (Phase 4F) --
  중단된 학습을 이어서(resume) 실행하기 위한 self-contained 체크포인트.
  model/optimizer/scheduler state, history, best model, early stopping
  카운터, DataLoader shuffle generator 상태, CPU RNG 상태까지 하나의
  파일에 담는다. Phase 4R부터 CUDA training의 same-device exact-resume을
  위한 `cuda_rng_state`(optional, CPU checkpoint에서는 `None`)도 담는다
  -- `cpu_rng_state`와 대칭적인 execution state이며 `TrainingConfig`
  필드가 아니다.

두 포맷은 서로 다른 용도이므로 섞어 쓰면 안 된다 -- load_training_checkpoint()/
load_state_dict() 둘 다 상대방 포맷이 들어오면 명확한 에러로 거부한다
(아래 참고).

**checkpoint "조회"와 "resume 가능 여부"는 별개다.** load_training_checkpoint()
는 구조적으로 정상인 파일이면(포맷/필수 key/타입) history.stopped_early
값과 무관하게 항상 payload를 반환한다 -- 사용자가 model_state_dict나
best_state_dict만 꺼내 쓰는 것도 정당한 용도이기 때문이다. resume 실행
자체를 거부하는 지점은 training/loop.py의 TrainingResumeState 생성
시점과 run_training() 진입 시점이다.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from image_ai_studio.training.config import (
    RESUME_CONFIG_FIELDS,
    RESUME_CONFIG_LEGACY_DEFAULTS,
    TrainingConfig,
    require_compatible_resume_config,
)
from image_ai_studio.training.loop import TrainingResult

# require_compatible_resume_config는 config.py가 단일 구현이다 (loop.py의
# run_training()도 같은 함수를 가져다 쓴다) -- 여기서는 checkpoint 관련
# 코드를 쓰는 caller가 `from image_ai_studio.training.checkpoint import
# require_compatible_resume_config`로 자연스럽게 찾을 수 있도록 재노출만
# 한다. 로직을 다시 구현하지 않는다.
__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "save_state_dict",
    "load_state_dict",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "require_compatible_resume_config",
]

CHECKPOINT_FORMAT_VERSION = 1

_REQUIRED_HISTORY_FIELDS = (
    "train_losses",
    "val_losses",
    "val_accuracies",
    "best_epoch",
    "best_val_loss",
    "stopped_early",
)


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    """payload를 path에 원자적으로 쓴다(Phase 4J, docs/
    phase4j_epoch_checkpoint_design.md §7-2) -- 목적지와 같은 디렉터리에
    tempfile.mkstemp()로 임시 파일을 만들고, torch.save() 후
    flush()/os.fsync()로 디스크에 반영을 요청한 뒤 os.replace()로
    교체한다(POSIX/Windows 양쪽에서 원자적). os.replace() 이전에
    실패하면 목적지 파일은 전혀 건드려지지 않는다. os.replace() 자체가
    실패하면 예외가 그대로 전파된다(재시도/폴백 없음). 어떤 예외든
    임시 파일 삭제를 시도하지만, 그 정리가 실패해도 원래 예외를
    가리지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def save_state_dict(model: nn.Module, path: str | Path) -> None:
    """model.state_dict()를 파일로 저장. 상위 디렉터리 자동 생성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state_dict(model: nn.Module, path: str | Path, *, map_location: str = "cpu") -> nn.Module:
    """저장된 state_dict를 model에 로드 (in-place) 후 그대로 반환."""
    path = Path(path)
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(state_dict, dict) and "format_version" in state_dict:
        raise ValueError(
            f"{path} looks like a full training checkpoint (has a 'format_version' key), "
            "not a bare model state_dict -- use load_training_checkpoint() instead"
        )
    model.load_state_dict(state_dict)
    return model


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    training_result: TrainingResult,
    training_config: TrainingConfig,
    loader_generator_state: torch.Tensor,
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor | None = None,
) -> None:
    """중단된 학습을 이어서(resume) 실행하기 위한 self-contained checkpoint를
    저장한다.

    `model`은 run_training()이 실제로 학습에 사용한 **현재(마지막으로
    완료된 epoch)** model이어야 한다 -- `training_result.best_state_dict`
    (지금까지 validation loss가 가장 낮았던 epoch의 snapshot)와 다른
    시점일 수 있으므로 절대 혼동하면 안 된다. 예를 들어 현재 epoch=5,
    best epoch=3이라면 이 checkpoint의 model_state_dict는 epoch 5,
    best_state_dict는 epoch 3의 가중치를 담는다.

    loader_generator_state/cpu_rng_state는 run_training()이 알지 못하는
    값이라 caller가 직접 채취해서 넘겨야 한다 (DataLoader의 shuffle
    generator는 caller가 만들었으므로 caller만 갖고 있고, CPU RNG는
    torch.get_rng_state()로 언제든 caller가 직접 읽을 수 있다):

        loader_generator_state = train_loader.generator.get_state()
        cpu_rng_state = torch.get_rng_state()

    cuda_rng_state(Phase 4R, 기본값 None)는 CUDA training의 same-device
    exact-resume에 필요한 현재 training device의 CUDA RNG snapshot이다
    -- CPU training이면 caller가 이 인자를 생략해 `None`으로 저장한다
    (기존 CPU checkpoint 호출부는 이 키워드 인자를 몰라도 계속 동작한다).
    CUDA training이면 caller가 `torch.cuda.get_rng_state(device)`로 직접
    채취해서 넘겨야 한다(`cpu_rng_state`와 동일한 책임 분리 -- 이 함수는
    device를 모르므로 스스로 채취하지 않는다). `RESUME_CONFIG_FIELDS`와
    무관한 순수 실행 state이므로 `training_config`에는 나타나지 않는다.

    저장은 원자적이다(Phase 4J, docs/phase4j_epoch_checkpoint_design.md
    §7-2) -- 임시 파일에 다 쓴 뒤 os.replace()로 교체하므로, 이 함수가
    예외를 던지면 `path`의 기존 내용은 전혀 바뀌지 않는다(재시도/
    폴백 없이 예외가 그대로 전파된다).
    """
    path = Path(path)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": training_result.optimizer_state_dict,
        "scheduler_state_dict": training_result.scheduler_state_dict,
        "history": asdict(training_result.history),
        "best_state_dict": training_result.best_state_dict,
        "epochs_without_improvement": training_result.epochs_without_improvement,
        "training_config": asdict(training_config),
        "loader_generator_state": loader_generator_state,
        "cpu_rng_state": cpu_rng_state,
        "cuda_rng_state": cuda_rng_state,
    }
    _atomic_torch_save(payload, path)


def load_training_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict:
    """save_training_checkpoint()로 저장한 payload를 dict로 반환한다.

    구조적으로 잘못된 파일(형식 판별자 없음, format_version 불일치, 필수
    key 누락, history/training_config가 dict가 아니거나 그 안의 필수
    필드 누락, history 길이 불일치, epochs_without_improvement가 음수,
    scheduler 관련 필드 불일치)은 여기서 명확한 ValueError로 거부한다.
    `training_config`의 "필수 필드"는 `RESUME_CONFIG_LEGACY_DEFAULTS`
    (config.py)에 있는 필드(현재 weight_decay만)는 제외한다 -- Phase 4L
    이전 checkpoint에는 그 키가 없을 수 있고, 그 경우의 처리는 이 함수가
    아니라 require_compatible_resume_config()의 책임이다.

    `cuda_rng_state`(Phase 4R)는 structural 필수 key가 아니다 -- pre-4R
    checkpoint에는 이 키 자체가 없다. 있으면 `None` 또는 `torch.Tensor`
    인지만 확인한다(shape/dtype은 검증하지 않는다 -- `cpu_rng_state`/
    `loader_generator_state`도 shape을 검증하지 않는 기존 스타일과 동일,
    CUDA RNG state의 실제 크기는 PyTorch/CUDA 버전에 따라 달라질 수 있는
    implementation detail이라 여기서 강하게 가정하면 안 된다). **이 함수
    자신은 키가 없어도 payload에 `"cuda_rng_state": None`을 새로
    삽입하지 않는다** -- 반환하는 `dict`는 파일에 저장된 key 구성을
    그대로 유지한다. 부재를 `None`으로 취급하는 것은 호출 측(예:
    `imagefolder_workflow.py`의 `payload.get("cuda_rng_state")`)의
    책임이다. 이 값의 부재는 "CPU checkpoint" 또는 "same-device CUDA
    exact 계약이 없는 pre-4R CUDA checkpoint"라는 뜻이며, 어느 쪽이든
    resume 자체를 막지 않는다(portable-only).

    **`history.stopped_early`가 True여도 거부하지 않는다** -- 이 함수의
    책임은 "이 파일이 구조적으로 유효한 checkpoint인가"까지다. 사용자가
    `payload["best_state_dict"]`나 `payload["model_state_dict"]`를 꺼내
    새 모델에 `model.load_state_dict(...)`로 로드해 쓰는 것은 정당한
    용도이므로 막을 이유가 없다. "이 payload로 resume을 실행할 수
    있는가"는 완전히 다른 질문이고, 그 답은 여기가 아니라
    training/loop.py의 TrainingResumeState(생성 시점 + run_training()
    진입 시점)가 낸다.

    **학습 설정(optimizer/learning_rate 등) 호환성도 여기서 검사하지
    않는다** -- 이 함수는 이번에 resume에 사용할 새 TrainingConfig를
    모르기 때문이다. caller가 새 config를 정한 뒤
    require_compatible_resume_config()로 확인하거나(조기 검증용),
    TrainingResumeState를 만들어 run_training()에 넘기면 그 안에서
    항상 강제된다.

    반환값은 아직 순수 dict다 -- `payload["history"]`는 TrainingHistory
    인스턴스가 아니라 dict이므로, TrainingResumeState로 조립하는 것은
    caller 책임이다 (checkpoint.py는 그 조립을 대신 해주는 함수를 두지
    않는다 -- 현재 이걸 쓰는 caller가 하나뿐이라 추상화가 아직 정당화되지
    않는다).
    """
    path = Path(path)
    loaded = torch.load(path, map_location=map_location, weights_only=True)

    if not isinstance(loaded, dict) or "format_version" not in loaded:
        raise ValueError(
            f"{path} does not look like a full training checkpoint (missing 'format_version') -- "
            "if this is a bare model state_dict, use load_state_dict() instead"
        )
    if loaded["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} has unsupported checkpoint format_version={loaded['format_version']!r} "
            f"(expected {CHECKPOINT_FORMAT_VERSION})"
        )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "history",
        "best_state_dict",
        "epochs_without_improvement",
        "training_config",
        "loader_generator_state",
        "cpu_rng_state",
    }
    missing = required_keys - loaded.keys()
    if missing:
        raise ValueError(f"{path} is missing required checkpoint field(s): {sorted(missing)}")

    history = loaded["history"]
    if not isinstance(history, dict):
        raise ValueError(f"{path}: 'history' must be a dict, got {type(history).__name__}")
    missing_history_fields = [name for name in _REQUIRED_HISTORY_FIELDS if name not in history]
    if missing_history_fields:
        raise ValueError(f"{path}: 'history' is missing required field(s): {missing_history_fields}")

    lengths = {len(history["train_losses"]), len(history["val_losses"]), len(history["val_accuracies"])}
    if len(lengths) != 1:
        raise ValueError(
            f"{path}: history metric lists have mismatched lengths "
            f"(train_losses={len(history['train_losses'])}, val_losses={len(history['val_losses'])}, "
            f"val_accuracies={len(history['val_accuracies'])})"
        )

    training_config = loaded["training_config"]
    if not isinstance(training_config, dict):
        raise ValueError(f"{path}: 'training_config' must be a dict, got {type(training_config).__name__}")
    # RESUME_CONFIG_LEGACY_DEFAULTS(config.py)에 있는 필드(현재 weight_decay만)는
    # Phase 4L 이전 checkpoint에 키 자체가 없을 수 있으므로 여기서 필수로
    # 요구하지 않는다 -- require_compatible_resume_config()가 이 dict를
    # 그대로 참조해 누락 시 기본값으로 간주한다. 이 구조적 검사가
    # RESUME_CONFIG_FIELDS 전체를 그대로 요구하면, 그 뒤의 migration 정책이
    # 실제 checkpoint 파일 경로에서는 결코 실행되지 못한다(load 단계에서
    # 이미 거부되므로).
    strictly_required_config_fields = [
        name for name in RESUME_CONFIG_FIELDS if name not in RESUME_CONFIG_LEGACY_DEFAULTS
    ]
    missing_config_fields = [name for name in strictly_required_config_fields if name not in training_config]
    if missing_config_fields:
        raise ValueError(f"{path}: 'training_config' is missing required field(s): {missing_config_fields}")

    epochs_without_improvement = loaded["epochs_without_improvement"]
    if not isinstance(epochs_without_improvement, int) or epochs_without_improvement < 0:
        raise ValueError(
            f"{path}: 'epochs_without_improvement' must be a non-negative integer, "
            f"got {epochs_without_improvement!r}"
        )

    for tensor_field in ("loader_generator_state", "cpu_rng_state"):
        if not isinstance(loaded[tensor_field], torch.Tensor):
            raise ValueError(
                f"{path}: '{tensor_field}' must be a torch.Tensor, got {type(loaded[tensor_field]).__name__}"
            )

    # cuda_rng_state(Phase 4R)는 optional -- 키 자체가 없는 pre-4R checkpoint는
    # 여기서 아무것도 검증하지 않고 그대로 통과시킨다(위 docstring 참고).
    if "cuda_rng_state" in loaded:
        cuda_rng_state = loaded["cuda_rng_state"]
        if cuda_rng_state is not None and not isinstance(cuda_rng_state, torch.Tensor):
            raise ValueError(
                f"{path}: 'cuda_rng_state' must be None or a torch.Tensor, "
                f"got {type(cuda_rng_state).__name__}"
            )

    scheduler_configured = training_config.get("lr_scheduler") is not None
    scheduler_state_present = loaded["scheduler_state_dict"] is not None
    if scheduler_configured != scheduler_state_present:
        raise ValueError(
            f"{path}: training_config.lr_scheduler={training_config.get('lr_scheduler')!r} but "
            f"scheduler_state_dict is {'set' if scheduler_state_present else 'None'} -- checkpoint "
            "is internally inconsistent"
        )

    return loaded
