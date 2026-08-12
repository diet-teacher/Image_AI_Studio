"""ImageFolder 학습 orchestration (Phase 4H). `scripts/train_imagefolder.py`
(production CLI)와 `scripts/run_imagefolder_training_e2e.py`(회귀 검증
E2E) 둘 다 이 모듈의 `run_imagefolder_training_workflow()`만 호출한다 --
"학습 본질" 로직(ModelSpec/dataset 검증, model build/resume, 학습 실행,
checkpoint/history/best model/class mapping/test 결과 저장, TorchScript
export)은 여기 한 곳에만 있고, 두 호출자는 이걸 향해서만 의존한다(스크립트
끼리 서로 import하지 않음 -- docs/phase4h_production_training_cli_design.md
§4/§12).

이 모듈은 E2E 전용 로직(loss-decreased 게이트, class mapping/best model
reload 자체 검증, C++ parity)을 전혀 모른다 -- 그건 호출자(E2E)가 반환된
`ImageFolderWorkflowResult`를 받은 뒤 자기 책임으로 수행한다. 마찬가지로
`image_ai_studio.tools.run_and_compare`(C++ runner)는 이 모듈에서 아예
import하지 않는다 -- production CLI 경로가 실수로도 C++ 빌드/CUDA 가용성에
의존하지 않도록 하기 위함.

예외는 삼키거나 재포장하지 않는다 -- ModelValidationError/
TrainingConfigError/ValueError/OSError를 그대로 전파한다(전부 이미
ValueError의 서브클래스). TorchScript export 실패만 예외로 승격한다
(TorchScriptExporter.export()는 예외 대신 metadata.json의 status 필드로
실패를 표현하므로, 워크플로우의 "성공하면 Result, 실패하면 예외"라는
단일 출력 계약에 맞춘다).
"""
from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_ai_studio.export.torchscript_exporter import TorchScriptExporter
from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.model_definition.specs import ModelSpec
from image_ai_studio.model_definition.validation import validate_model_spec
from image_ai_studio.training.checkpoint import (
    load_training_checkpoint,
    save_state_dict,
    save_training_checkpoint,
)
from image_ai_studio.training.config import TrainingConfig, require_compatible_resume_config
from image_ai_studio.training.history import save_training_history
from image_ai_studio.training.imagefolder_resume import (
    build_imagefolder_resume_metadata,
    load_imagefolder_resume_metadata,
    metadata_path_for_checkpoint,
    require_compatible_imagefolder_resume_metadata,
    save_imagefolder_resume_metadata,
)
from image_ai_studio.training.loop import (
    CheckpointHook,
    EpochCheckpointView,
    ShouldStopCallback,
    TrainingHistory,
    TrainingProgressCallback,
    TrainingResult,
    TrainingResumeState,
    TrainingStopReason,
    evaluate_classification_metrics,
    run_training,
)
from image_ai_studio.training.metrics import ClassificationMetrics
from image_ai_studio.training.torchvision_dataset import (
    ImageFolderSplits,
    make_imagefolder_datasets,
    require_matching_num_classes,
    save_class_mapping,
)

SEED = 20260730

_TORCHSCRIPT_MODEL_FILENAME = "model.ts"
_TORCHSCRIPT_METADATA_FILENAME = "model_metadata.json"


@dataclass
class ImageFolderWorkflowRequest:
    """워크플로우 호출에 필요한 전부. `training_config`는 이미 검증된
    `TrainingConfig` 인스턴스를 그대로 받는다(호출자가 CLI argparse에서
    조립하든, E2E가 고정 상수로 조립하든 이 dataclass는 신경 쓰지 않는다).

    checkpoint_every(Phase 4J, docs/phase4j_epoch_checkpoint_design.md
    §6/§11)는 global epoch이 이 값의 배수가 될 때마다 `checkpoint_out`을
    자동으로 갱신한다. `None`(기본값)이면 학습 도중 자동 저장을 하지
    않고, 기존과 동일하게 학습 종료 시 최종 저장만 수행한다.
    `checkpoint_every`를 켜려면 `checkpoint_out`이 함께 있어야 한다.

    device(Phase 4Q)는 `"cpu"`/`"cuda"`/`"cuda:N"`만 허용하는 순수 runtime
    실행 파라미터다 -- `seed`처럼 `TrainingConfig` 밖에서 관리되는
    run-level 파라미터와 같은 계층이지만(둘 다 학습 objective를 바꾸는
    hyperparameter가 아니므로 `TrainingConfig`에 두지 않는다), `seed`는
    RNG 결과에 영향을 주고 `device`는 execution backend를 정한다는
    차이가 있다(checkpoint의 `training_config`/`RESUME_CONFIG_FIELDS`와는
    둘 다 무관). training(fresh/resume)에만 적용되고, 최종 test 평가/
    TorchScript export/C++ parity는 이 값과 무관하게 항상 CPU를
    유지한다(아래 `run_imagefolder_training_workflow()` 참고).

    pin_memory/non_blocking(Phase 4U)은 CUDA training의 host->device
    batch 전송을 최적화하는 순수 runtime 실행 힌트다 -- `device`와 같은
    계층(학습 objective를 바꾸지 않으므로 `TrainingConfig`에 두지 않고,
    checkpoint의 `training_config`/`RESUME_CONFIG_FIELDS`와도 무관하다).
    사전 조사(docs/phase4u_cuda_h2d_transfer_optimization_design.md)에서
    두 값 모두 resume 전후로 자유롭게 바뀌어도 bitwise exact-resume이
    깨지지 않음을 실측으로 확인했다. `device == "cpu"`면 이 두 값은 항상
    무시되고 effective pin_memory/non_blocking은 강제로 `False`다 --
    CPU에는 host->device 전송 자체가 없어 순수 optimization hint가
    의미를 갖지 못하므로(`run_imagefolder_training_workflow()`가 이
    effective 값을 계산해 실제 DataLoader/`run_training()`에 전달한다).
    최종 test 평가(항상 CPU)에는 이 값들이 전달되지 않는다. 둘은
    서로 독립적인 runtime optimization hint로 설정 가능하다(한쪽만
    켜는 조합을 거부하지 않는다) -- `non_blocking=True`는 host-side
    synchronization을 줄일 수 있어 pageable(unpinned) source에서도
    의미가 있을 수 있고, `pin_memory=True`는 DataLoader가 반환하는
    host tensor를 page-locked memory에 배치해 CUDA H2D transfer를 더
    효율적으로 만들 수 있다. 다만 이 프로젝트는 항상 default CUDA
    stream만 쓰므로, 두 값을 함께 켜도 H2D copy와 model kernel
    execution의 GPU-side overlap을 보장하지는 않는다(정확성 관점에서는
    어떤 조합도 안전함이 실측됐다 -- docs/
    phase4u_cuda_h2d_transfer_optimization_design.md 참고)."""

    model_json_path: Path
    dataset_root: Path
    training_config: TrainingConfig
    output_dir: Path
    resume_from: Path | None = None
    checkpoint_out: Path | None = None
    export_torchscript: bool = True
    seed: int = SEED
    checkpoint_every: int | None = None
    device: str = "cpu"
    pin_memory: bool = False
    non_blocking: bool = False


@dataclass
class ImageFolderWorkflowResult:
    """저장된 파일 경로와 학습 결과 지표만 담는다 -- 살아있는 nn.Module/
    텐서 객체는 담지 않는다(호출자가 필요하면 저장된 파일에서 다시
    읽으면 된다). export/checkpoint를 하지 않았으면 해당 경로는 None."""

    history: TrainingHistory
    test_loss: float
    test_accuracy: float
    best_model_state_dict_path: Path
    training_history_path: Path
    class_mapping_path: Path
    test_result_path: Path
    checkpoint_path: Path | None
    checkpoint_metadata_path: Path | None
    torchscript_model_path: Path | None
    torchscript_metadata_path: Path | None
    # Phase 4O: 최종 test 평가의 confusion matrix/macro precision/recall/F1
    # (test-only, validation/TrainingHistory/checkpoint는 무수정). 기본값
    # None은 "test 평가가 생략될 수 있다"는 뜻이 아니라, 이 dataclass를
    # 이 필드 없이 직접 생성하던 기존 코드(테스트의 manual/fake
    # constructor 호출)와의 생성자 하위호환을 위한 것이다 -- 마지막 필드로
    # 둬야 그 앞의 필드들이 여전히 기본값 없이 위치/키워드 인자로 채워질
    # 수 있다. `run_imagefolder_training_workflow()`가 정상 완료해 반환하는
    # production 결과의 test_metrics는 항상 실제 ClassificationMetrics다.
    test_metrics: ClassificationMetrics | None = None
    # Phase 4V: run_training()이 이미 계산한 `TrainingResult.stop_reason`을
    # 그대로 전달만 한다(single source of truth -- 이 workflow가
    # `history.stopped_early`/`stopped_by_user`로부터 다시 계산하지
    # 않는다). `run_imagefolder_training_workflow()`가 이 프로젝트의
    # GUI-facing public entrypoint이므로, 향후 caller(GUI 포함)가 최종
    # 종료 사유를 알기 위해 `TrainingResult`까지 직접 들여다볼 필요가
    # 없게 한다. `test_metrics`와 같은 이유로 기본값을 둬 기존 manual/fake
    # constructor 호출과의 생성자 하위호환을 유지한다 -- production
    # 결과는 항상 실제 "completed"/"early_stopped"/"user_stopped" 중
    # 하나다.
    stop_reason: TrainingStopReason = "completed"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_cuda_rng_state(device: str) -> torch.Tensor | None:
    """현재 training device의 CUDA RNG state를 캡처한다(Phase 4R,
    `cpu_rng_state`(`torch.get_rng_state()`) 캡처와 대칭적인 역할).
    `device == "cpu"`면 CUDA API를 전혀 호출하지 않고 `None`을 반환한다
    -- CPU 경로가 CUDA API를 절대 건드리지 않는다는 계약을 코드로
    강제한다. `torch.cuda.get_rng_state_all()`은 쓰지 않는다 -- Phase
    4Q가 single-device training만 지원하므로, 이 학습이 실제로 쓰는
    device 하나만 캡처하는 것으로 충분하다(multi-GPU RNG 저장은 범위
    밖)."""
    if device == "cpu":
        return None
    return torch.cuda.get_rng_state(device).clone()


def _restore_cuda_rng_state(state: torch.Tensor | None, device: str) -> None:
    """캡처된 CUDA RNG state를 복원한다(Phase 4R). `state`가 `None`이면
    (fresh training이거나, pre-4R legacy checkpoint에 이 키 자체가 없던
    경우) 아무 것도 하지 않고 조용히 넘어간다 -- warning/error 없이
    portable-only resume을 그대로 허용한다(기존 `cpu_rng_state`의
    fresh-path 처리와 동일한 정책). `device == "cpu"`면 `state`와
    무관하게 CUDA API를 절대 호출하지 않는다."""
    if device == "cpu" or state is None:
        return
    torch.cuda.set_rng_state(state, device)


@contextmanager
def _cuda_deterministic_context(enabled: bool) -> Iterator[None]:
    """CUDA training 구간에서만(Phase 4R) `torch.use_deterministic_algorithms(True)`
    /`cudnn.deterministic=True`/`cudnn.benchmark=False`를 적용하고, 정상
    종료든 예외 종료든 caller의 기존 전역 설정을 정확히 복원한다(로컬
    CUDA 실측으로 양쪽 경로 모두 sentinel 값이 정확히 복원됨을 직접
    확인함). 이 설정들은 process-global이라 scoped하게 관리하지 않으면
    이 workflow 호출 한 번이 caller의 이후 모든 PyTorch 실행에 영향을
    준다.

    `enabled=False`(CPU training)면 이 전역 설정을 읽지도 쓰지도 않는다
    -- CPU 경로는 deterministic 설정과 완전히 무관해야 한다는 계약을
    코드로 강제한다(실측: 이 분기를 타면 sentinel 값이 함수 진입 전과
    후에 완전히 동일함을 확인).

    context 내부 정책은 `warn_only=False`(strict fail-fast)이다 --
    지원 ModelSpec에 향후 deterministic 구현이 없는 CUDA 연산이
    추가되면, `warn_only=True`로 경고만 내고 계속 실행해 exact-resume
    계약을 조용히 깨뜨리는 대신 명확한 RuntimeError로 즉시 실패해야
    한다(의도된 strict behavior). 이 `warn_only=False`는 context
    내부에서만 강제되는 정책이며, caller가 context 진입 전에
    `warn_only=True`를 쓰고 있었다면 context 종료 시 그 원래 값을
    정확히 복원한다(진입 전 값을 그대로 덮어써 기본값 `False`로
    고정하지 않는다). `CUBLAS_WORKSPACE_CONFIG` 환경변수는 이 함수가
    건드리지 않는다 -- 로컬 실측(subprocess)에서 이 프로젝트가 실제
    쓰는 연산은 설정 유무와 무관하게 동일하게 동작했고, CUDA context
    초기화 이후 환경변수를 바꾸는 것은 반영이 보장되지 않아 근거 없는
    설계이기 때문이다."""
    if not enabled:
        yield
        return

    previous_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms_enabled, warn_only=previous_warn_only
        )
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark


def _prepare_resume(
    request: ImageFolderWorkflowRequest,
    model_spec: ModelSpec,
    splits: ImageFolderSplits,
) -> tuple[nn.Module, torch.Generator, TrainingResumeState | None, torch.Tensor | None, torch.Tensor | None]:
    """`request.resume_from`이 None이면 (신규 model, 신규 generator, None,
    None, None)을 반환한다. 있으면 metadata 로드/검증 -> model build+load ->
    generator 복원 -> ResumeState 조립 -> config 검증까지 전부 수행한 뒤
    (model, restored_generator, resume_state, payload["cpu_rng_state"],
    payload.get("cuda_rng_state"))를 반환한다.

    다섯 번째 반환값(cuda_rng_state, Phase 4R)은 pre-4R checkpoint처럼
    이 키 자체가 없으면 `None`이다 -- `dict.get()`을 쓰므로 legacy
    checkpoint에서도 KeyError 없이 그냥 `None`이 된다(구조적 필수
    key가 아님, checkpoint.py의 structural validation도 이 키를 요구하지
    않는다).

    scaler_state_dict(Phase 4S)는 이 함수의 반환 tuple에 별도로 담기지
    않는다 -- `cuda_rng_state`와 달리 이 값은 `TrainingResumeState`의
    필드이므로(위 `resume_state` 조립에서 `payload.get("scaler_state_dict")`
    로 채워짐), 이미 반환하는 `resume_state` 안에 포함돼 있다. 반환
    tuple의 arity를 Phase 4R에서 이미 5로 확장했으므로, Phase 4S는 이
    함수의 반환 형태를 다시 바꿀 필요가 없다.

    **이 함수는 전역 CPU/CUDA RNG를 절대 건드리지 않는다.** 네 번째/
    다섯 번째 반환값(cpu_rng_state/cuda_rng_state)은 호출자가 DataLoader
    생성을 전부 마친 뒤, run_training() 호출 바로 직전에
    torch.set_rng_state()/torch.cuda.set_rng_state()로 직접 적용해야
    한다 -- 이 함수 안에서 미리 복원하면, 함수가 반환된 뒤 호출자가 하는
    DataLoader 생성이 복원 시점과 run_training() 사이에 끼어들게 되어
    "RNG 복원은 항상 마지막"이라는 불변조건이 함수 경계 때문에 깨진다."""
    if request.resume_from is None:
        _set_seed(request.seed)
        model = build_model(model_spec)
        loader_generator = torch.Generator().manual_seed(request.seed)
        return model, loader_generator, None, None, None

    saved_metadata = load_imagefolder_resume_metadata(metadata_path_for_checkpoint(request.resume_from))
    payload = load_training_checkpoint(request.resume_from)

    current_metadata = build_imagefolder_resume_metadata(model_spec, splits)
    require_compatible_imagefolder_resume_metadata(saved_metadata, current_metadata)

    _set_seed(request.seed)
    model = build_model(model_spec)
    # payload["best_state_dict"]가 아니라 model_state_dict를 쓴다 --
    # best_state_dict를 쓰면 "최고 성능 epoch"에서 재개하게 되어 resume
    # 시작점 계약(마지막으로 완료된 epoch에서 이어간다)을 깬다.
    model.load_state_dict(payload["model_state_dict"])

    restored_generator = torch.Generator()
    restored_generator.set_state(payload["loader_generator_state"])

    resume_state = TrainingResumeState(
        optimizer_state_dict=payload["optimizer_state_dict"],
        scheduler_state_dict=payload["scheduler_state_dict"],
        history=TrainingHistory(**payload["history"]),
        epochs_without_improvement=payload["epochs_without_improvement"],
        best_state_dict=payload["best_state_dict"],
        training_config=payload["training_config"],
        # Phase 4S: pre-4S checkpoint처럼 이 키 자체가 없으면 .get()이
        # None을 반환한다 -- run_training()이 scaler=None(FP32)이거나
        # precision="fp16"인데 이 값이 None이면 fresh GradScaler로
        # 시작한다(portable-only, 위 checkpoint.py의 최소 검증 철학과
        # 동일하게 반환값 arity/tuple을 늘리지 않는다 -- cuda_rng_state와
        # 달리 scaler state는 TrainingResumeState의 필드이지 이 함수의
        # 반환 tuple에 별도로 담지 않는다).
        scaler_state_dict=payload.get("scaler_state_dict"),
    )
    require_compatible_resume_config(resume_state.training_config, request.training_config)

    return model, restored_generator, resume_state, payload["cpu_rng_state"], payload.get("cuda_rng_state")


def _validate_checkpoint_every(value: int | None) -> None:
    """checkpoint_every 유효성 검증(Phase 4J, §6-2/§11-2). `config.py`의
    private `_require_positive_int()`는 재사용하지 않는다 -- 이 모듈
    자체의 validator로 둔다."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint_every must be an integer or None, got {value!r}")
    if value < 1:
        raise ValueError(f"checkpoint_every must be at least 1, got {value!r}")


# Phase 4Q: 이 프로젝트가 명시적으로 지원하는 문법은 "cpu"/"cuda"/"cuda:N"
# 뿐이다(N은 leading zero 없는 음이 아닌 정수). torch.device()도 대소문자
# 구분, zero-padding, 그 외 backend(mps/xpu/hip 등)를 이미 엄격하게
# 거부함을 실측했지만, torch.device() 혼자서는 이 프로젝트가 검증하지
# 않은 backend까지 그대로 통과시키므로(mps/xpu/hip...), 이 프로젝트는
# 자체 허용 목록을 별도로 관리한다 -- 과도한 자체 parser 대신 최소
# 정규식 하나로 충분하다.
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda|cuda:(0|[1-9][0-9]*))$")


def _validate_device(value: str) -> None:
    """device 문자열을 검증한다(Phase 4Q). CUDA 계열이면 추가로
    `torch.cuda.is_available()`/`torch.cuda.device_count()`로 조기
    검증한다 -- 그러지 않으면 `.to(device)`까지 내려가서야 저수준
    `AcceleratorError`("CUDA error: invalid device ordinal")가 나는데,
    이 경우가 실제로 재현됨을 로컬에서 직접 확인했다(require_matching_num_classes()
    와 동일한 "깊은 곳 대신 조기에 명확한 에러" 패턴). CUDA 미가용 시
    CPU로 조용히 대체하지 않는다 -- 사용자가 명시한 실행 의도를 그대로
    존중한다."""
    if not isinstance(value, str) or not _DEVICE_PATTERN.fullmatch(value):
        raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:N', got {value!r}")
    if value == "cpu":
        return
    if not torch.cuda.is_available():
        raise ValueError(f"device={value!r} requires CUDA, but torch.cuda.is_available() is False")
    if ":" in value:
        index = int(value.split(":", 1)[1])
        device_count = torch.cuda.device_count()
        if index >= device_count:
            raise ValueError(
                f"device={value!r} is out of range -- torch.cuda.device_count()=={device_count}"
            )


def _is_cuda_device(device: str) -> bool:
    """`device`가 CUDA 계열("cuda"/"cuda:N")인지 판별한다(Phase 4U). 이미
    `_validate_device()`가 통과시킨 값에만 적용되므로 별도 형식 검증은
    하지 않는다. `loop.py`의 `_build_precision_execution()`이 쓰는 것과
    동일한 판별 predicate다(그쪽은 fp16/bf16 CUDA 전용 여부, 이쪽은
    pin_memory/non_blocking effective 여부 -- 판별 기준 자체는 같다)."""
    return device == "cuda" or device.startswith("cuda:")


_CUDA_ONLY_PRECISIONS = ("fp16", "bf16")


def _validate_precision_device_compatibility(precision: str, device: str) -> None:
    """`precision="fp16"`(Phase 4S)/`"bf16"`(Phase 4T)은 CUDA에서만
    허용한다. `TrainingConfig` 자신은 device를 모르므로(`_require_one_of()`
    가 "fp32"/"fp16"/"bf16" 값 자체만 검증) 이 cross-field 검증은 device를
    이미 아는 workflow 레벨에서 한다 -- `class_weights` 길이를 dataset
    크기와 여기서 검증하는 것과 같은 이유(`require_matching_num_classes()`
    근처 참고). CPU AMP(`torch.amp.autocast(device_type="cpu", ...)`)는
    이번 Phase의 범위 밖이라 silent CPU fallback 없이 명확히 거부한다.

    이 함수는 `run_training()`이 내부적으로 강제하는 `loop.py`의
    `_build_precision_execution()`(같은 조합을 `ValueError`로 거부)와
    같은 검증을 중복하는 것이 아니라, 서로 다른 경계를 보호하는
    defense-in-depth다 -- 이 함수는 dataset/model 준비 전 user-facing
    fail-fast(workflow 진입 즉시 거부)이고, `_build_precision_execution()`
    는 이 workflow를 거치지 않고 `TrainingConfig`+`run_training()`을
    직접 호출하는 generic caller까지 보호한다. `fp16`/`bf16` 둘 다
    `_CUDA_ONLY_PRECISIONS`에 속하는 값이면 device="cpu"일 때 동일하게
    거부한다 -- fp16만 검사하고 bf16을 빠뜨리면 Phase 4S stabilization
    에서 발견한 "특정 precision 값만 하드코딩 검사"와 같은 종류의 누락이
    반복된다."""
    if precision in _CUDA_ONLY_PRECISIONS and device == "cpu":
        raise ValueError(
            f"precision={precision!r} requires a CUDA device, but device={device!r} -- "
            "CPU AMP is not supported in this phase"
        )


def _normalized_path(path: str | Path) -> str:
    """두 경로가 같은 파일을 가리키는지 비교하기 위한 정규화(Phase 4J,
    §11-2). Path.resolve()로 상대/절대 표기 차이를 없애고,
    os.path.normcase()로 Windows의 대소문자 비구분 파일시스템에서의
    오탐/누락을 줄인다(POSIX에서 normcase는 no-op)."""
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _is_in_place_resume(request: ImageFolderWorkflowRequest) -> bool:
    """resume_from과 checkpoint_out이 정확히 같은 파일을 가리키면(Phase
    4J, §6-4/§11-2) True -- 이 경우에만 기존 checkpoint_out 경로를
    갱신하는 것이 허용된다. §7-3의 metadata_ready 초기값도 이 함수
    하나를 그대로 재사용한다."""
    if request.resume_from is None or request.checkpoint_out is None:
        return False
    return _normalized_path(request.resume_from) == _normalized_path(request.checkpoint_out)


def _validate_checkpoint_output_paths(request: ImageFolderWorkflowRequest) -> None:
    """출력 경로 재사용 정책(Phase 4J, §6-5): in-place resume(resume_from
    == checkpoint_out)만 기존 checkpoint_out 경로를 갱신할 수 있다.
    그 외(fresh 또는 다른 경로로의 resume)는 checkpoint_out과 그
    metadata sidecar가 완전히 비어있는 새 경로여야 한다 -- 기존 파일이
    있으면 학습을 시작하기 전에 거부한다(기존 파일을 지우거나 바꾸지
    않는다)."""
    if request.checkpoint_out is None:
        return
    if _is_in_place_resume(request):
        return

    checkpoint_path = Path(request.checkpoint_out)
    metadata_path = metadata_path_for_checkpoint(checkpoint_path)
    if checkpoint_path.exists():
        raise ValueError(
            f"{checkpoint_path} already exists -- a fresh training run (or a resume "
            "that writes to a different path than --resume-from) must use a new, "
            "unused checkpoint_out path. To continue training this exact checkpoint, "
            "pass it as both --resume-from and --checkpoint-out."
        )
    if metadata_path.exists():
        raise ValueError(
            f"{metadata_path} already exists -- a fresh training run (or a resume "
            "that writes to a different path than --resume-from) must use a new, "
            "unused checkpoint_out path."
        )


def _make_checkpoint_hook(
    request: ImageFolderWorkflowRequest,
    ensure_checkpoint_metadata: Callable[[], None],
) -> CheckpointHook:
    """global epoch 기준 cadence로 동작하는 checkpoint_hook을 만든다
    (Phase 4J, §11-3). `model_spec`/`splits`는 다시 캡처하지 않는다 --
    `ensure_checkpoint_metadata`가 이미 그것들을 캡처했으므로 이 hook은
    그 함수 하나만 공유해서 쓴다."""

    def hook(view: EpochCheckpointView) -> None:
        global_epoch = len(view.history.train_losses)
        if global_epoch % request.checkpoint_every != 0:
            return  # non-scheduled epoch -- state_dict()/RNG 조회를 전혀 하지 않는다

        if view.loader_generator is None:
            raise ValueError(
                "auto checkpoint requires an explicit DataLoader generator for exact "
                "resume, but loader_generator is None"
            )

        ensure_checkpoint_metadata()  # §7-3, checkpoint보다 먼저

        training_result = TrainingResult(
            history=view.history,
            best_state_dict=view.best_state_dict,
            optimizer_state_dict=view.optimizer.state_dict(),
            scheduler_state_dict=(view.scheduler.state_dict() if view.scheduler is not None else None),
            epochs_without_improvement=view.epochs_without_improvement,
            # Phase 4S: view.scaler_state_dict는 run_training()의 epoch
            # 루프가 이 hook 호출 직전에 이미 scaler.state_dict()로 채워 온
            # 읽기 전용 snapshot이다(EpochCheckpointView 참고) -- 여기서
            # 다시 계산하지 않는다.
            scaler_state_dict=view.scaler_state_dict,
        )
        save_training_checkpoint(  # 원자적(§7-2)
            request.checkpoint_out,
            model=view.model,
            training_result=training_result,
            training_config=request.training_config,
            loader_generator_state=view.loader_generator.get_state(),
            cpu_rng_state=torch.get_rng_state(),
            # Phase 4R: cpu_rng_state와 동일하게 이 hook이 호출되는 시점(=
            # epoch train+validate가 이미 끝난 뒤)의 CUDA RNG state를 그대로
            # 읽기만 한다(torch.cuda.get_rng_state()도 읽기 전용 호출이라
            # EpochCheckpointView 계약을 위반하지 않는다). device="cpu"면
            # _capture_cuda_rng_state()가 CUDA API를 호출하지 않고 None을
            # 반환한다.
            cuda_rng_state=_capture_cuda_rng_state(request.device),
        )

    return hook


def run_imagefolder_training_workflow(
    request: ImageFolderWorkflowRequest,
    *,
    progress_callback: TrainingProgressCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
) -> ImageFolderWorkflowResult:
    _validate_checkpoint_every(request.checkpoint_every)
    if request.checkpoint_every is not None and request.checkpoint_out is None:
        raise ValueError("checkpoint_every requires checkpoint_out to be set")
    _validate_checkpoint_output_paths(request)
    _validate_device(request.device)
    _validate_precision_device_compatibility(request.training_config.precision, request.device)
    # Phase 4U: pin_memory/non_blocking은 CUDA에서만 의미 있는 순수 runtime
    # 최적화 힌트다 -- device="cpu"면 request 값과 무관하게 항상 False로
    # 강제한다(§4 CPU/CUDA effective option contract). CPU DataLoader에
    # pin_memory=True를 그대로 넘기면 PyTorch가 "no accelerator found"
    # 경고를 내므로, 여기서 미리 걸러 그 경고 자체가 발생하지 않게 한다.
    is_cuda = _is_cuda_device(request.device)
    effective_pin_memory = request.pin_memory if is_cuda else False
    effective_non_blocking = request.non_blocking if is_cuda else False

    model_spec = load_model_spec(request.model_json_path)
    shape_trace = validate_model_spec(model_spec)
    final_shape = shape_trace[-1].output_shape

    splits = make_imagefolder_datasets(model_spec.input_shape, root=request.dataset_root)
    require_matching_num_classes(len(splits.classes), final_shape)

    # Phase 4P: class_weights가 설정돼 있으면 dataset의 실제 class 수와 길이가
    # 일치하는지 여기서 조기 검증한다(require_matching_num_classes()와 대칭적인
    # 위치/스타일) -- generic run_training()/TrainingConfig는 class 이름도
    # dataset도 모르므로 이 검증을 할 수 없고(PyTorch CrossEntropyLoss의
    # forward-time shape 검증이 그 경로의 backstop), ImageFolder workflow는
    # 이미 splits.classes를 알고 있으므로 학습을 시작하기 전에 명확한 에러로
    # 거부할 수 있는 유일한 지점이다.
    if request.training_config.class_weights is not None and len(
        request.training_config.class_weights
    ) != len(splits.classes):
        raise ValueError(
            f"class_weights has {len(request.training_config.class_weights)} value(s) but dataset has "
            f"{len(splits.classes)} classes {splits.classes} -- class_weights order must match "
            "class_mapping.json's classes/class_to_idx order"
        )

    # metadata_ready/ensure_checkpoint_metadata는 이 workflow 호출 하나당
    # 정확히 한 번 만들어지는 closure 상태다 -- scheduled checkpoint_hook과
    # 아래의 학습 종료 후 최종 저장이 이 하나를 함께 공유해서, metadata
    # sidecar를 이번 실행 동안 최대 한 번만 쓴다(Phase 4J, §7-3/§11-3).
    # in-place resume은 _prepare_resume()이 이미 metadata를 로드/검증했으므로
    # True로 시작해 절대 다시 쓰지 않는다.
    metadata_ready = _is_in_place_resume(request)

    def ensure_checkpoint_metadata() -> None:
        nonlocal metadata_ready
        if metadata_ready:
            return
        metadata_path = metadata_path_for_checkpoint(request.checkpoint_out)
        current_metadata = build_imagefolder_resume_metadata(model_spec, splits)
        save_imagefolder_resume_metadata(current_metadata, metadata_path)  # 원자적(§7-2)
        metadata_ready = True

    model, loader_generator, resume_state, cpu_rng_state, cuda_rng_state = _prepare_resume(
        request, model_spec, splits
    )
    # Phase 4Q: _prepare_resume()은 항상 CPU model을 build/load한다
    # (map_location 기본값과 build_model()의 기존 동작 유지) -- 여기서
    # 학습에 실제로 쓸 device로 옮긴다. run_training()의 기존 계약("model은
    # 호출 전에 이미 device로 옮겨져 있어야 함")을 만족해야 하고, PyTorch
    # semantics상 optimizer는 model.parameters()가 가리키는 실제 tensor를
    # 참조하므로 run_training() 내부에서 optimizer가 생성되기 전에 이
    # 지점에서 target device로 이동한다.
    model = model.to(request.device)

    batch_size = request.training_config.batch_size
    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=True,
        num_workers=0,
        pin_memory=effective_pin_memory,
    )
    # val은 training 동안 request.device에서 평가되므로(run_training() 내부
    # evaluate() 호출) train과 동일한 effective_pin_memory를 적용한다.
    val_loader = DataLoader(
        splits.val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=effective_pin_memory
    )
    # test는 Phase 4Q부터 항상 CPU 고정 평가라(아래 evaluate_classification_metrics()
    # 호출 참고) Phase 4U optimization을 적용하지 않는다 -- pin_memory=True를
    # 줘도 CUDA로 옮겨질 일이 없어 항상 무의미하다.
    test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_hook = (
        _make_checkpoint_hook(request, ensure_checkpoint_metadata) if request.checkpoint_every is not None else None
    )
    # Phase 4R: CUDA training 구간(RNG 복원 -> run_training() -> 최종 RNG
    # 캡처)만 scoped deterministic context로 감싼다 -- device="cpu"면
    # 이 context는 전역 설정을 전혀 읽거나 쓰지 않는다. model JSON
    # 검증/dataset 스캔이나 아래의 best_model CPU 최종 test/export는
    # 이 context 밖이다 -- deterministic 설정은 CUDA stochastic training
    # 자체에만 의미가 있고, CPU 전용 코드에 적용할 이유가 없다.
    with _cuda_deterministic_context(enabled=request.device != "cpu"):
        # DataLoader 생성이 전부 끝난 뒤, 다른 RNG 소비 작업 없이 즉시
        # run_training()을 호출한다 -- fresh 경로에서는 cpu_rng_state/
        # cuda_rng_state가 둘 다 None이라 이 블록이 아무 일도 하지 않는다.
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state)
        _restore_cuda_rng_state(cuda_rng_state, request.device)

        training_result = run_training(
            model, train_loader, val_loader, request.training_config, device=request.device,
            resume_state=resume_state, progress_callback=progress_callback, should_stop=should_stop,
            checkpoint_hook=checkpoint_hook, non_blocking=effective_non_blocking,
        )
        # checkpoint 저장에 쓸 RNG snapshot -- 이후 코드(TorchScript export의
        # set_seed() 등)가 전역 RNG를 다시 바꾸기 전에, 학습이 실제로 끝난
        # 시점의 상태를 독립적인 snapshot으로 캡처해 둔다. 다음 epoch
        # continuation 지점을 나타내야 하므로, deterministic context를
        # 벗어나기 전, final test/export가 전역 RNG를 건드리기 전에 캡처한다.
        cpu_rng_state_after = torch.get_rng_state().clone()
        cuda_rng_state_after = _capture_cuda_rng_state(request.device)
    loader_generator_state_after = loader_generator.get_state().clone()
    history = training_result.history

    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_history_path = output_dir / "training_history.json"
    save_training_history(history, training_history_path)

    class_mapping_path = output_dir / "class_mapping.json"
    save_class_mapping(splits.classes, splits.class_to_idx, class_mapping_path)

    # checkpoint는 `model`(현재/마지막 epoch 가중치)이 아직 어떤 방식으로도
    # best 가중치로 대체되지 않은 이 시점에 저장한다 -- 아래 best_model
    # 생성(별도 인스턴스) 이전에 두어야, best_state_dict를 현재 모델로
    # 착각해서 저장하는 버그가 애초에 발생할 수 없다. 학습 도중
    # checkpoint_hook이 이미 몇 번 저장했더라도, 이 최종 저장은 항상
    # 실행된다(Phase 4J, §6-4) -- should_stop() 평가 이후의 정확한
    # stopped_by_user 값을 반영하는 저장은 이 최종 저장뿐이기 때문이다
    # (마지막 epoch이 scheduled epoch였다면 같은 global epoch이 두 번
    # 저장되는 것은 의도된 동작이다, §9-4).
    checkpoint_path: Path | None = None
    checkpoint_metadata_path: Path | None = None
    if request.checkpoint_out is not None:
        checkpoint_path = request.checkpoint_out
        checkpoint_metadata_path = metadata_path_for_checkpoint(checkpoint_path)
        ensure_checkpoint_metadata()  # §7-3 -- 이미 준비됐으면(scheduled 저장이 있었으면) 아무 것도 안 함
        save_training_checkpoint(  # 원자적(§7-2)
            checkpoint_path,
            model=model,
            training_result=training_result,
            training_config=request.training_config,
            loader_generator_state=loader_generator_state_after,
            cpu_rng_state=cpu_rng_state_after,
            cuda_rng_state=cuda_rng_state_after,
        )

    # run_training()은 best_state_dict를 메모리로만 반환한다 -- 여기서 새
    # model에 로드한 뒤 저장한다.
    best_model = build_model(model_spec)
    best_model.load_state_dict(training_result.best_state_dict)
    best_model = best_model.eval()

    best_model_state_dict_path = output_dir / "best_model_state_dict.pt"
    save_state_dict(best_model, best_model_state_dict_path)

    # class_mapping.json(위에서 저장)의 classes 순서가 confusion_matrix/
    # per_class_recall의 class index 순서와 동일하다는 계약을 위해, 여기서
    # 쓰는 num_classes도 그 순서를 만드는 len(splits.classes)를 그대로
    # 재사용한다(위의 require_matching_num_classes() 호출이 이미 이 값과
    # model 출력 차원의 일치를 검증했으므로 여기서 다시 검증하지 않는다).
    # Phase 4Q: 학습이 request.device(예: "cuda")에서 수행됐더라도 최종 test
    # 평가는 의도적으로 항상 "cpu"를 그대로 쓴다(request.device를 여기로
    # 전달하지 않는다) -- best_model은 build_model()로 새로 만들어져 항상
    # CPU이고(위), TorchScriptExporter도 example_input을 강제로 CPU로
    # 옮긴 뒤 model(example_input)을 호출하므로 model이 CPU가 아니면 export
    # 단계에서 device mismatch가 난다. 이번 Phase는 "training device
    # exposure"이지 "evaluation device exposure"가 아니다.
    test_loss, test_accuracy, test_metrics = evaluate_classification_metrics(
        best_model, test_loader, num_classes=len(splits.classes), device="cpu"
    )
    test_result_path = output_dir / "test_result.json"
    test_result_path.write_text(
        json.dumps(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "classification_metrics": asdict(test_metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    torchscript_model_path: Path | None = None
    torchscript_metadata_path: Path | None = None
    ts_model_path = output_dir / _TORCHSCRIPT_MODEL_FILENAME
    ts_metadata_path = output_dir / _TORCHSCRIPT_METADATA_FILENAME
    if request.export_torchscript:
        _set_seed(request.seed)
        example_input = torch.randn(1, *model_spec.input_shape, dtype=torch.float32)
        TorchScriptExporter().export(
            best_model,
            example_input,
            ts_model_path,
            ts_metadata_path,
            model_name=model_spec.name,
            state_dict_path=best_model_state_dict_path,
        )
        export_metadata = json.loads(ts_metadata_path.read_text())
        if export_metadata["status"] != "PASS":
            raise ValueError(f"TorchScript export failed: {export_metadata.get('error_log')}")
        torchscript_model_path = ts_model_path
        torchscript_metadata_path = ts_metadata_path
    else:
        # 이전 실행이 같은 output_dir에 남긴 TorchScript 산출물을 제거한다
        # -- 지우지 않으면 사용자가 이번 실행 결과로 착각할 수 있다.
        # 워크플로우가 고정 이름으로 관리하는 이 두 파일만 지우고, output_dir의
        # 다른 파일은 건드리지 않는다. 삭제 실패(권한 등)는 감싸지 않고
        # 그대로 전파한다.
        ts_model_path.unlink(missing_ok=True)
        ts_metadata_path.unlink(missing_ok=True)

    return ImageFolderWorkflowResult(
        history=history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        best_model_state_dict_path=best_model_state_dict_path,
        training_history_path=training_history_path,
        class_mapping_path=class_mapping_path,
        test_result_path=test_result_path,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        torchscript_model_path=torchscript_model_path,
        torchscript_metadata_path=torchscript_metadata_path,
        test_metrics=test_metrics,
        stop_reason=training_result.stop_reason,
    )
