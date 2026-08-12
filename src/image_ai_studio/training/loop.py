"""학습/평가 루프. loss=CrossEntropyLoss는 고정(label smoothing 계수는
Phase 4N부터, class별 명시적 weight는 Phase 4P부터 TrainingConfig로 선택
가능). optimizer(Adam/SGD)와 LR scheduler(없음/ReduceLROnPlateau)는
Phase 4E부터 TrainingConfig로 선택 가능 -- 선택지가 2개/1개뿐이라 registry
없이 이 모듈의 private helper(_build_optimizer/_build_scheduler/
_build_criterion)로 충분하다. Phase 4S/4T부터 `_build_precision_execution()`
이 config.precision("fp32"|"fp16"|"bf16")+device로부터 `autocast_dtype`
(forward+loss를 감쌀 dtype, fp32면 None)과 `scaler`(scaled backward용
`torch.amp.GradScaler`, fp16만 사용하고 fp32/bf16은 None)라는 독립된
두 값을 계산하고, `train_one_epoch()`이 이 둘을 각자 다른 축으로
받아 분기한다 -- precision="fp32"(기본값)에서는 둘 다 항상 `None`이므로
AMP API를 전혀 호출하지 않고, `train_one_epoch()`의 두 `else` branch가
Phase 4A~4S의 기존 FP32 forward/backward/[clip]/optimizer.step 계산
semantics를 그대로 실행한다(numerical anchor 무변경 -- 코드 자체에는
새 분기가 추가됐지만 그 분기를 타지 않는 한 실행 경로/계산 결과는
동일하다). `config.precision`이 `"fp16"`/`"bf16"`인데 `device`가
CUDA가 아니면(`"cpu"`뿐 아니라 `"mps"`/`"xpu"` 등 이 프로젝트가 다루지
않는 다른 backend 포함) `_build_precision_execution()`이 `ValueError`를
낸다 -- silent FP32 fallback(또는 잘못된 backend용 scaler 생성) 없이
명확히 거부한다."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_ai_studio.training.config import TrainingConfig, require_compatible_resume_config
from image_ai_studio.training.metrics import ClassificationMetrics, compute_classification_metrics


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """config.optimizer에 따라 Adam, AdamW, 또는 SGD를 생성. TrainingConfig.__post_init__이
    이미 "adam"/"sgd"/"adamw" 외의 값을 거부하므로 그 외 분기는 없다.
    weight_decay는 세 optimizer 모두에 공통으로 전달한다 (Phase 4L)."""
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """config.lr_scheduler가 None이면 scheduler를 아예 만들지 않는다 (기존
    동작 재현). "plateau"만 지원 -- StepLR 등 metric에 의존하지 않는
    scheduler는 이번 Phase 범위 밖."""
    if config.lr_scheduler is None:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience
    )


def _build_criterion(config: TrainingConfig, device: str = "cpu") -> nn.Module:
    """config.label_smoothing/config.class_weights로 CrossEntropyLoss를
    생성한다(Phase 4N/4P). 선택지가 CrossEntropy 하나뿐이라
    _build_optimizer()/_build_scheduler()와 동일한 근거로 registry나 loss
    이름 선택 필드는 두지 않는다. 이 criterion은 training(train_one_epoch)
    에서만 쓴다 -- evaluate()는 validation/test loss의 의미
    (ReduceLROnPlateau/early stopping/best model selection/test_loss)를
    그대로 지키기 위해 항상 별도의 unsmoothed/unweighted CrossEntropyLoss를
    자체적으로 쓴다(무수정).

    config.class_weights가 None이 아니면 `device` 위에 바로 weight tensor를
    생성한다(Phase 4P) -- model/입력이 이미 이 device에 있으므로, weight만
    다른 device에 있으면 forward에서 device mismatch 에러가 난다. dtype은
    항상 float32로 고정한다 -- PyTorch가 정수 dtype weight tensor를 거부하고
    (실측 확인), config.class_weights의 원소가 Python int/float가 섞여
    있어도 이 tensor 생성 시점에 항상 올바른 dtype으로 정규화된다."""
    weight = (
        torch.tensor(config.class_weights, dtype=torch.float32, device=device)
        if config.class_weights is not None
        else None
    )
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)


def _build_precision_execution(
    config: TrainingConfig, device: str
) -> tuple[torch.dtype | None, torch.amp.GradScaler | None]:
    """config.precision/device로부터 `train_one_epoch()`이 실제로 필요로
    하는 두 개의 독립적인 저수준 실행 정보를 계산한다(Phase 4T가
    `_build_grad_scaler()`를 대체): `autocast_dtype`(forward+loss를
    감쌀 dtype, `None`이면 autocast 자체를 쓰지 않음)과 `scaler`
    (`torch.amp.GradScaler`, `None`이면 scaled backward를 쓰지 않음).

    Phase 4S 최초 구현은 "scaler가 있는가"라는 단일 신호로 "AMP를
    쓰는가"와 "AMP dtype이 무엇인가"를 동시에 판별했다 -- FP16만
    지원할 때는 이 결합이 우연히 맞았지만("scaler 있음"="FP16 AMP
    사용"), BF16은 autocast는 필요하면서 GradScaler는 필요 없는
    조합이라(아래 GradScaler 불필요성 설명 참고) 이 결합을 그대로
    유지하면 BF16이 `scaler is None` 때문에 FP32 branch로 조용히
    빠지는 새로운 silent fallback을 만든다. 그래서 두 값을 독립된
    tuple로 분리했다:

    ```text
    precision=fp32          -> (None, None)
    precision=fp16 + CUDA   -> (torch.float16, GradScaler)
    precision=bf16 + CUDA   -> (torch.bfloat16, None)
    ```

    `config.precision == "fp32"`이면 device가 무엇이든(CUDA 포함)
    `(None, None)`을 반환한다 -- `train_one_epoch()`이 AMP API를 전혀
    호출하지 않는 기존(Phase 4A~4S) FP32 경로를 그대로 실행한다.

    `config.precision`이 `"fp16"`/`"bf16"`인데 `device`가 CUDA가
    아니면(`"cpu"`뿐 아니라 이 프로젝트가 인식하지 않는 다른 backend
    문자열, 예: `"mps"`/`"xpu"` 포함) **`ValueError`를 낸다.** Phase 4S
    stabilization에서 `device == "cpu"`만 검사해 `"mps"`/`"xpu"`
    같은 다른 non-CUDA backend가 조용히 CUDA용 scaler를 받아버리는
    버그를 발견/수정한 적이 있다 -- Phase 4T도 같은 실수를 반복하지
    않도록 fp16/bf16 둘 다 동일한 CUDA 판별을 거친다.
    `ImageFolderWorkflowRequest`(imagefolder_workflow.py의
    `_validate_precision_device_compatibility()`)를 거치지 않고 이
    함수(또는 `run_training()`)를 직접 호출하는 generic caller가
    있을 수 있는데, 거기서도 silent fallback이 일어나면 안 된다 --
    사용자가 명시한 실행 의도를 그대로 존중한다는 `_validate_device()`
    (Phase 4Q)의 기존 원칙과 동일하다. workflow 레벨의 조기 검증과 이
    함수의 검증은 서로 다른 경계를 보호하는 defense-in-depth다
    (workflow: dataset/model 준비 전 user-facing fail-fast, 이 함수:
    workflow를 우회한 generic `run_training()` 호출도 보호).

    CUDA 판별은 이 프로젝트가 이미 다루는 `"cuda"`/`"cuda:N"` 형태만
    인식한다(`device == "cuda" or device.startswith("cuda:")`) --
    `imagefolder_workflow.py`의 `_DEVICE_PATTERN`(ordinal 범위까지 검증)
    을 여기서 다시 구현하지 않는다. fp16/bf16 둘 다 이 판별 로직
    하나를 공유하므로(각 precision마다 따로 구현하지 않음) 중복이
    없다.

    BF16은 GradScaler를 쓰지 않는다 -- FP32와 동일한 8-bit exponent
    range를 가져(FP16의 5-bit보다 넓음) FP16에서 loss scaling이 특히
    필요했던 좁은 dynamic range 문제를 크게 완화한다(BF16이 이론적으로
    절대 underflow가 없다는 뜻은 아니다 -- exponent range 밖의 값은
    BF16에도 여전히 존재한다). 이 프로젝트의 실제 BF16 학습 경로에서
    GradScaler 없이 정상 학습(forward/backward/optimizer.step 정상
    동작) 및 same-device continuous-vs-split+resume bitwise exact가
    실측으로 확인됐으므로, **production contract로 BF16에는 GradScaler
    를 사용하지 않는다** -- API 레벨에서는 BF16+GradScaler 조합도 에러
    없이 동작하지만, 불필요한 checkpoint state/복잡도만 늘어나므로
    채택하지 않는다(docs/phase4t_cuda_bf16_mixed_precision_design.md
    참고).

    PyTorch 실측 확인: `torch.cuda.amp.GradScaler`는 FutureWarning으로
    deprecated이며 `torch.amp.GradScaler("cuda", ...)`가 권장 API다 -- 이
    함수는 항상 후자를 쓴다. `GradScaler`의 첫 인자는 **device type**
    (`"cuda"` 또는 `"cpu"`)만 공식 contract다(설치된 PyTorch의
    `GradScaler.__init__` docstring: "Possible values are: 'cuda' and
    'cpu'. The type is the same as the `type` attribute of a
    `torch.device`") -- ordinal이 붙은 `"cuda:0"`/`"cuda:1"` 등은
    문서화된 값이 아니다. 실제로 `GradScaler.__init__` 소스를 확인한
    결과 `self._device == "cuda"`라는 정확한 문자열 비교로 CUDA
    availability 경고 분기를 타는데, `"cuda:0"`을 넘기면 이 비교가
    항상 거짓이 되어 그 분기가 조용히 스킵되고, `update(new_scale=...)`
    내부의 `new_scale.device.type != self._device` assertion도
    `tensor.device.type`이 항상 ordinal 없는 `"cuda"`이므로
    `"cuda:0"`과 비교하면 항상 어긋난다(이 프로젝트는 `update()`를
    인자 없이만 호출하므로 이 경로를 실제로 타지는 않지만, `self._device`
    가 device type만 담아야 한다는 내부 불변식을 보여준다). scale
    tensor의 실제 device/ordinal 배치는 `scale()`이
    `outputs.device`(즉 loss tensor가 이미 올라가 있는 실제 device)로
    lazy 초기화하므로 `self._device`와 무관하다 -- 즉 `device`의
    ordinal을 `GradScaler`에 전달해도 실질적인 기능 이득이 없다.
    따라서 이 함수는 `GradScaler`에 **항상 `"cuda"`(device type)만
    전달**하고, model/tensor의 실제 ordinal 배치는 기존과 동일하게
    `model.to(device)`/`images.to(device)`가 전담한다 -- ordinal별
    별도 scaler 설계를 추가할 필요가 없다는 결론은 유지되지만, 그
    이유가 "ordinal을 그대로 전달해도 된다"가 아니라 "애초에 전달할
    필요가 없다"로 정정됐다. `init_scale`/`growth_factor`/
    `backoff_factor`/`growth_interval` 등 tuning parameter는 PyTorch
    기본값을 그대로 쓴다 -- 이번 Phase도 이 값들을 config/CLI에
    노출하지 않는다(non-goal)."""
    if config.precision == "fp32":
        return None, None
    if config.precision not in ("fp16", "bf16"):
        # TrainingConfig.PRECISION_CHOICES가 이미 "fp32"/"fp16"/"bf16" 외의
        # 값을 거부하므로 정상 경로에서는 도달하지 않는다 -- 이 함수 자신이
        # "fp32도 fp16도 아니면 무조건 bf16"처럼 implicit dispatch로 새
        # precision을 조용히 잘못 처리하지 않도록 명시적 fail-fast로 막아
        # 둔다(향후 PRECISION_CHOICES에 값이 추가되는데 이 함수 수정이
        # 누락되는 경우를 대비).
        raise ValueError(f"unsupported precision: {config.precision!r}")
    if not (device == "cuda" or device.startswith("cuda:")):
        raise ValueError(f"precision={config.precision!r} requires a CUDA device, but device={device!r}")
    if config.precision == "fp16":
        return torch.float16, torch.amp.GradScaler("cuda")
    return torch.bfloat16, None  # config.precision == "bf16"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
    criterion: nn.Module | None = None,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """1 epoch 학습 (model.train() -> forward -> CrossEntropyLoss -> backward
    -> [gradient_clip_norm이 있으면 L2 norm clipping] -> step). 반환값: epoch
    평균 loss. gradient_clip_norm=None(기본값)이면 clip 호출 자체가 없어
    Phase 4A~4L의 기존 동작과 완전히 동일하다(Phase 4M). criterion=None
    (기본값)이면 기존과 동일하게 unsmoothed/unweighted CrossEntropyLoss를
    내부에서 생성한다 -- run_training()은 _build_criterion(config, device)로
    만든 criterion을 넘겨 label smoothing(Phase 4N)/class weight(Phase 4P)를
    적용한다.

    `autocast_dtype`/`scaler`는 서로 독립적인 두 축이다(Phase 4T,
    `_build_precision_execution()` 참고) -- `autocast_dtype`은 forward+
    loss를 `torch.amp.autocast`로 감쌀지/어떤 dtype으로 감쌀지를
    결정하고, `scaler`는 scaled backward(`scaler.scale(loss).backward()`
    -> `scaler.step()` -> `scaler.update()`)를 쓸지를 결정한다. **이
    둘을 하나의 신호로 겸용하지 않는다** -- `scaler is None`은 더 이상
    "AMP를 쓰지 않는다"는 뜻이 아니라 "scaled backward를 쓰지 않는다"는
    뜻일 뿐이다(BF16은 `autocast_dtype=torch.bfloat16`이면서
    `scaler=None`인 조합).

    둘 다 기본값 `None`이면(Phase 4A~4S 이전과 동일) `torch.amp.autocast`
    /`GradScaler` API를 이 함수가 전혀 호출하지 않는 기존 FP32 경로가
    그대로 실행된다(기존 caller 전부 하위호환, CPU/CUDA FP32 회귀 없음).

    `autocast_dtype`이 주어지면 그 dtype으로 `torch.amp.autocast(
    device_type="cuda", dtype=autocast_dtype)` 안에서 forward+loss만
    계산한다(backward/clip/step/update는 그 밖에서 호출 -- PyTorch
    권장 사용법). 그 뒤 `scaler`가 주어지면(Phase 4S, CUDA FP16 AMP)
    `scaler.scale(loss).backward()` -> (gradient_clip_norm이 있으면)
    `scaler.unscale_(optimizer)` 후 `clip_grad_norm_` -> `scaler.step(
    optimizer)` -> `scaler.update()` 순서로 실행한다. `unscale_()`은
    clipping이 있을 때만, 그리고 이 step에서 정확히 한 번만 호출한다
    -- 실측 확인: `unscale_()`을 생략하면 grad norm이 현재 `scale`배
    (예: 1024배)로 부풀어 있어 clipping 임계값이 사실상 무의미해지고,
    같은 step에서 두 번 호출하면 `RuntimeError`가 난다. `scaler`가
    `None`이면(Phase 4T, CUDA BF16) autocast로 감싼 forward+loss
    이후 `loss.backward()` -> (clipping 있으면) `clip_grad_norm_` ->
    `optimizer.step()`을 그대로 쓴다 -- FP32 경로와 동일한 순서이며
    scaled-backward 관련 API를 전혀 호출하지 않는다."""
    model.train()
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if autocast_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
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


def evaluate_classification_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str = "cpu",
) -> tuple[float, float, ClassificationMetrics]:
    """evaluate()와 같은 의미의 (loss, accuracy)를 계산하면서, 같은 forward
    pass 안에서 confusion matrix도 배치 단위로 함께 누적해 상세
    classification metric까지 반환한다(Phase 4O). loss/accuracy 계산 방식
    (unsmoothed CrossEntropyLoss, argmax accuracy, sample-weighted 평균)은
    evaluate()와 정확히 동일하다 -- 이 함수는 evaluate()를 대체하지 않는다
    (evaluate()는 무수정이며 training-loop validation 경로가 계속 그대로
    쓴다); confusion matrix까지 필요한 소비자(현재는 ImageFolder 최종 test
    평가)가 이 함수를 대신 쓴다. 같은 데이터셋을 evaluate()로 한 번, 이
    함수로 또 한 번 -- 두 번 순회하지 않도록, loss/accuracy/confusion
    matrix를 전부 이 함수 하나의 순회에서 함께 계산한다.

    confusion matrix 누적 tensor는 `device` 위에 그대로 두고 배치마다
    더한다 -- GPU 평가 시에도 배치마다 CPU로 옮기는 동기화 없이 텐서
    상에서만 누적한다. evaluation이 끝난 뒤 `[num_classes, num_classes]`
    matrix 하나를 `compute_classification_metrics()`가 딱 한 번만 CPU로
    옮겨 상세 metric을 계산한다(그 함수 내부에서 class별 `.item()`을
    반복하므로, 매 배치가 아니라 여기서 한 번만 옮겨야 반복 GPU
    synchronization을 피할 수 있다). empty loader 정책은 evaluate()와
    동일하게 ValueError."""
    if num_classes <= 0:
        raise ValueError(
            f"evaluate_classification_metrics: num_classes must be a positive integer, got {num_classes!r}"
        )

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

            indices = labels * num_classes + predictions
            confusion_matrix += torch.bincount(indices, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            )

    if total_samples == 0:
        raise ValueError("evaluate_classification_metrics: loader produced no samples (empty DataLoader)")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    metrics = compute_classification_metrics(confusion_matrix)
    return avg_loss, accuracy, metrics


@dataclass
class TrainingHistory:
    """run_training()의 epoch별 결과 + best epoch 추적.

    JSON으로 그대로 직렬화 가능한 순수 메트릭만 담는다 (텐서 값인
    state_dict는 여기 두지 않고 TrainingResult가 별도로 가진다 --
    training/history.py의 save_training_history()가 이 dataclass를
    그대로 asdict()해서 저장한다).

    best_epoch는 1-indexed. val_loss가 strict하게(<) 더 낮아질 때만
    갱신되므로, 동률이면 먼저 나온 epoch가 유지된다.

    stopped_by_user(Phase 4I)는 should_stop() 콜백 때문에 아직 실행할
    epoch가 남은 채로 멈췄으면 True다(docs/phase4i_training_progress_and_stop_design.md
    §7/§9 참고). stopped_early와 달리 TrainingResumeState는 이 값을
    거부하지 않는다 -- 사용자가 잠시 멈춘 것뿐이므로 resume이 항상
    가능해야 한다. 기본값이 있어 옛 checkpoint/JSON history(이 필드가
    없는)를 읽어도 그대로 False로 채워진다.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    stopped_early: bool = False
    stopped_by_user: bool = False


@dataclass
class TrainingResult:
    """run_training()의 전체 반환값.

    history(JSON 직렬화 가능)와 best_state_dict(텐서, 메모리에만 존재)를
    분리했다 -- run_training()은 파일을 쓰지 않으므로, 파일 저장은 이
    필드들을 받은 호출자(예: run_training_e2e.py)의 책임이다.

    optimizer_state_dict/scheduler_state_dict/epochs_without_improvement는
    Phase 4F(resume)를 위해 추가됐다 -- run_training() 내부에서만 존재하다
    함수가 끝나면 사라지던 상태를, 호출자가 나중에 이어서 학습을 재개할 때
    쓸 수 있도록 밖으로 내보낸다. 전부 함수 종료 시점의 독립적인 snapshot
    (deepcopy)이며, 반환 이후 model/optimizer가 더 학습되어도 바뀌지 않는다.

    scaler_state_dict(Phase 4S)는 config.precision="fp16"+CUDA device일
    때만 값이 있고, 그 외(FP32 training, CPU training)에는 `None`이다 --
    optimizer_state_dict/scheduler_state_dict와 같은 위치/역할의 필드다.
    이 값이 checkpoint에 그대로 저장되어 same-device AMP exact-resume에
    쓰인다(positive/negative control 실측으로 GradScaler state 복원이
    exact-resume에 필수임을 확인함 -- docs/
    phase4s_amp_mixed_precision_design.md 참고).
    """

    history: TrainingHistory
    best_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict
    scheduler_state_dict: dict | None
    epochs_without_improvement: int
    scaler_state_dict: dict | None = None


@dataclass(frozen=True)
class TrainingProgress:
    """Phase 4I: 한 epoch이 완료될 때마다 run_training()이
    progress_callback에 넘기는 읽기 전용 스냅샷(docs/
    phase4i_training_progress_and_stop_design.md §3/§4/§7).

    관찰/UI 갱신 전용이다 -- model 객체, state_dict, optimizer/scheduler
    상태와 시간 정보는 담지 않고, 완료된 epoch의 지표만 전달한다(나중에
    필요해지면 기본값 있는 필드를 추가해도 하위 호환이 깨지지 않는다).
    완전한 학습 상태(model.state_dict()가 아니라 best_state_dict,
    optimizer_state_dict, scheduler_state_dict, history 전체)는 이
    dataclass가 아니라 학습이 끝난 뒤 반환되는 TrainingResult가 담당한다.
    run_training()을 직접 호출하는 코드는 자신이 넘긴 model 참조를 계속
    들고 있으므로 필요하면 그쪽에서 별도로 들여다볼 수 있지만, 그건 이
    dataclass의 책임이 아니다.

    run_epoch: 이번 run_training() 호출 안에서 몇 번째 epoch인지(1부터).
    total_run_epochs: 이번 호출에서 실행 예정인 전체 epoch 수(config.epochs).
    global_epoch: resume을 포함한 전체 이력에서 이 epoch의 절대 번호
        (loop.py의 for 루프가 도는 epoch 변수와 동일한 값).
    learning_rate: 이번 epoch의 train_one_epoch()가 실제로 사용한 값
        (scheduler.step() 호출 전에 캡처됨). scheduler가 이번 epoch에
        LR을 바꿨다면 그 바뀐 값은 다음 콜백에서 보인다.
    stopped_early: 이번 epoch에서 early stopping이 발동했으면 True.
        사용자 중단 여부는 이 dataclass가 아니라 학습이 끝난 뒤
        TrainingResult.history.stopped_by_user로만 확인한다(콜백
        호출 시점에는 아직 should_stop()을 평가하지 않았으므로 이
        dataclass에는 그 값을 담을 수 없다, §3/§4/§7).
    """

    run_epoch: int
    total_run_epochs: int
    global_epoch: int

    train_loss: float
    val_loss: float
    val_accuracy: float
    learning_rate: float

    best_epoch: int
    best_val_loss: float
    epochs_without_improvement: int

    stopped_early: bool


TrainingProgressCallback = Callable[[TrainingProgress], None]
"""epoch 완료마다 정확히 한 번 호출된다(early stopping으로 끝난
epoch에서도 마찬가지). 예외를 던지면 run_training()이 그대로 전파하고
TrainingResult를 반환하지 않는다 -- 의도적으로 학습을 중단하려면
callback에서 예외를 던지지 말고 ShouldStopCallback을 쓸 것을 권장한다.
그래야 run_training()이 정상적인 TrainingResult를 반환하고, 상위
workflow(예: run_imagefolder_training_workflow())가 checkpoint_out 등
기존 artifact 후처리를 계속 수행할 수 있다(docs/
phase4i_training_progress_and_stop_design.md §5).

side-effect 계약(Phase 4J, docs/phase4j_epoch_checkpoint_design.md §3-5):
관찰/출력/UI 전달 전용이다 -- PyTorch CPU RNG를 소비하면 안 되고
(torch.rand()/torch.randn() 등 호출 금지), model/optimizer/scheduler/
DataLoader generator를 변경하면 안 된다(closure나 전역 변수로 이
객체들에 접근할 수 있더라도 마찬가지). checkpoint_hook이 이 callback
보다 먼저 실행되므로(Phase 4J), 이 계약을 어기면 그 순간 저장된
checkpoint가 실제로 다음 epoch가 시작할 때의 상태와 달라져
exact-resume이 깨질 수 있다."""

ShouldStopCallback = Callable[[], bool]
"""인자 없이 bool을 반환하는 아무 callable이나 가능하다 (예:
threading.Event().is_set). 이번 호출의 마지막 요청 epoch거나
config.epochs == 1이면 절대 평가되지 않는다(docs/
phase4i_training_progress_and_stop_design.md §7 "특수 사례").

side-effect 계약(Phase 4J, docs/phase4j_epoch_checkpoint_design.md §3-5):
외부 stop flag를 읽어 bool을 반환하는 용도로만 쓴다 -- PyTorch RNG를
소비하면 안 되고, model/optimizer/scheduler/DataLoader generator를
변경하면 안 되며, 반환값 외의 학습 상태 side effect를 만들면 안 된다.
TrainingProgressCallback과 같은 이유(exact-resume)로 이 계약이
요구된다."""


@dataclass(frozen=True)
class EpochCheckpointView:
    """checkpoint_hook 호출 동안만 유효한 읽기 전용 뷰(synchronous
    ephemeral view) -- 독립 snapshot이 아니다.

    model/history/optimizer/scheduler/loader_generator는 전부
    run_training() 내부의 살아있는 참조다. hook은 이 view가 유효한
    동안(자기 자신의 동기 호출 범위 안)에서 필요한 조회와 직렬화를
    전부 끝내야 하고, 이 view나 그 어떤 참조도 나중에 쓰려고
    보관하면 안 된다. 비동기/백그라운드 저장에는 쓸 수 없다.

    optimizer/scheduler/loader_generator는 읽기 전용으로만 접근해야
    한다 -- 이 객체들을 변형하면(특히 loader_generator) exact-resume이
    깨진다. hook은 또한 학습에 사용되는 RNG를 소비해서도 안 된다
    (torch.rand()/torch.randn() 등 호출 금지) -- .state_dict()/
    .get_state()/torch.get_rng_state() 같은 읽기 전용 호출 자체는
    RNG를 소비하지 않으므로 허용되지만, 그 밖의 RNG 소비 연산은
    다음 epoch가 시작할 때의 실제 상태와 checkpoint에 저장된 상태를
    어긋나게 만든다(docs/phase4j_epoch_checkpoint_design.md §3-5,
    exact-resume 계약의 일부).

    best_state_dict/epochs_without_improvement는 run_training()이
    이미 매 epoch 유지하는 값이라 담는 데 추가 비용이 없다.

    scaler_state_dict(Phase 4S)는 optimizer/scheduler와 달리 **살아있는
    GradScaler 객체 참조가 아니라 이미 계산된 dict snapshot**이다 -- 이
    view가 만들어지는 매 epoch마다 `scaler.state_dict()`를 미리 호출해
    담아 둔다. GradScaler.state_dict()는 텐서가 아니라 5개의 순수 Python
    float/int만 담은 작은 dict라(optimizer.state_dict()처럼 잠재적으로
    큰 텐서를 복사하는 비용이 없음) 이 eager 호출이 "무거운 계산은
    hook이 scheduled epoch로 판단한 뒤에만"이라는 이 클래스의 기존
    비용 원칙을 깨지 않는다. 이렇게 하면 checkpoint_hook이 살아있는
    GradScaler 객체를 실수로 변형할 가능성이 애초에 없다(읽기 전용
    계약을 타입으로 강제). scaler가 없으면(FP32/CPU training) `None`."""

    model: nn.Module
    history: TrainingHistory
    best_state_dict: dict[str, torch.Tensor]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None
    epochs_without_improvement: int
    loader_generator: torch.Generator | None
    scaler_state_dict: dict | None = None


CheckpointHook = Callable[[EpochCheckpointView], None]
"""완료된 epoch마다(있으면) 정확히 한 번 호출된다 -- 호출 자체는
저렴하고(view는 살아있는 참조만 담아 조립 비용이 거의 없다), 무거운
계산(state_dict 조회, RNG 조회, 디스크 I/O)은 hook이 scheduled epoch로
판단한 뒤에만 수행해야 한다. progress_callback/should_stop보다 먼저
호출된다(docs/phase4j_epoch_checkpoint_design.md §3-1).

side-effect 계약(§3-5, EpochCheckpointView 계약과 통합): view의 상태를
읽고 동기적으로 저장하는 용도로만 쓴다 -- model/optimizer/scheduler/
loader_generator를 변경하면 안 되고, torch.get_rng_state()/
generator.get_state()/.state_dict() 같은 읽기 전용 호출 외에 학습에
사용되는 RNG를 소비하는 연산을 호출하면 안 된다. hook이 예외를 던지면
run_training()이 그대로 전파한다(감싸지 않음)."""


@dataclass
class TrainingResumeState:
    """중단된 run_training() 호출을 이어서 실행하기 위해 필요한, 이전
    호출이 끝나며 사라졌을 상태의 스냅샷. loop.py(TrainingHistory/
    TrainingResult와 같은 파일)에 둔다 -- checkpoint.py가 아니라 이
    파일이 정의해야, "resume에 무엇이 필요한가"라는 질문의 답이 항상
    run_training()의 실제 구현과 함께 붙어 있다.

    completed_epochs는 별도 필드로 두지 않는다 -- len(history.train_losses)
    가 유일한 출처다. 이 dataclass의 __post_init__이 그 전제(train_losses/
    val_losses/val_accuracies 길이 일치, 비어있지 않음)를 검증한다.

    training_config는 checkpoint 저장 당시 TrainingConfig를 asdict()한
    순수 dict다 -- run_training()이 resume_state를 받으면 이 값과 새
    config를 항상 require_compatible_resume_config()로 비교한다(그
    호출을 caller가 빼먹어도 우회할 수 없다).

    scaler_state_dict(Phase 4S, 기본값 None)는 scheduler_state_dict와
    달리 **엄격한 양방향 mismatch 검증을 받지 않는다** -- `precision`이
    RESUME_CONFIG_FIELDS에 없기 때문에(자유롭게 바뀔 수 있는 training
    semantics 필드, gradient_clip_norm/label_smoothing과 동일한 범주),
    "config가 fp16을 요청하는데 이 값이 None"이거나 "이 값이 있는데
    config가 fp32"인 조합 둘 다 에러가 아니다: 전자는 fresh GradScaler로
    시작(legacy checkpoint 또는 FP32 checkpoint에서 AMP로 resume,
    portable-only), 후자는 이 값을 그냥 쓰지 않는다(AMP checkpoint에서
    FP32로 resume). 이 비대칭 처리가 precision 변경 resume을 허용하는
    Phase 4S의 공식 정책이다(docs/phase4s_amp_mixed_precision_design.md
    참고).
    """

    optimizer_state_dict: dict
    scheduler_state_dict: dict | None
    history: TrainingHistory
    epochs_without_improvement: int
    best_state_dict: dict[str, torch.Tensor]
    training_config: dict
    scaler_state_dict: dict | None = None

    def __post_init__(self) -> None:
        lengths = {
            len(self.history.train_losses),
            len(self.history.val_losses),
            len(self.history.val_accuracies),
        }
        if len(lengths) != 1:
            raise ValueError(
                "TrainingResumeState.history metric lists must have equal length, got "
                f"train_losses={len(self.history.train_losses)}, "
                f"val_losses={len(self.history.val_losses)}, "
                f"val_accuracies={len(self.history.val_accuracies)}"
            )
        completed_epochs = len(self.history.train_losses)
        if completed_epochs == 0:
            raise ValueError("TrainingResumeState.history must not be empty (nothing completed to resume from)")
        if self.history.best_epoch is None or not (1 <= self.history.best_epoch <= completed_epochs):
            raise ValueError(
                f"TrainingResumeState.history.best_epoch={self.history.best_epoch!r} is out of range "
                f"for {completed_epochs} completed epoch(s)"
            )
        if self.history.best_val_loss is None:
            raise ValueError("TrainingResumeState.history.best_val_loss must not be None")
        if (
            not isinstance(self.epochs_without_improvement, int)
            or isinstance(self.epochs_without_improvement, bool)
            or self.epochs_without_improvement < 0
        ):
            raise ValueError(
                "epochs_without_improvement must be a non-negative integer, "
                f"got {self.epochs_without_improvement!r}"
            )
        if self.history.stopped_early:
            raise ValueError(
                "cannot resume a TrainingHistory with stopped_early=True -- early stopping already "
                "decided this training run is finished. To keep training from these weights, load "
                "the desired weights with model.load_state_dict(...), then start a fresh "
                "TrainingConfig/run_training() call instead of resuming."
            )


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: str = "cpu",
    resume_state: TrainingResumeState | None = None,
    *,
    progress_callback: TrainingProgressCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
    checkpoint_hook: CheckpointHook | None = None,
) -> TrainingResult:
    """train_one_epoch/evaluate를 config.epochs만큼 반복하는 얇은 조립 함수.

    매 epoch validation loss가 지금까지의 최솟값보다 strict하게 낮아지면
    (`val_loss < best_val_loss`), 그 시점의 model.state_dict()를 깊은
    복사로 메모리에 보관한다 -- 이후 epoch에서도 model은 계속 학습되어
    바뀌므로, best 시점의 가중치를 남기려면 그 순간 복사해야 한다
    (참조만 보관하면 마지막 epoch 값으로 덮어써진다). 파일 저장은 하지
    않는다 (호출자 책임 -- resume_state를 만들거나 checkpoint를 저장하는
    것도 마찬가지로 호출자 책임이다. training/checkpoint.py의
    save_training_checkpoint() 참고).

    optimizer/scheduler/criterion(Phase 4N/4P, _build_criterion)은 config에
    따라 여기서 생성한다. resume_state가 None(기본값)이면 Phase 4E까지의
    동작과 완전히 동일 -- 매번 새로 생성하고, epoch 1부터, 빈 history로
    시작한다. criterion은 학습(train_one_epoch)에만 쓰인다 -- evaluate()
    는 validation/test loss의 기존 의미(ReduceLROnPlateau/early
    stopping/best model selection/test_loss)를 그대로 지키기 위해 항상
    별도의 unsmoothed/unweighted CrossEntropyLoss를 자체적으로 쓴다
    (무수정, Phase 4N/4P에서도 변경하지 않음). label_smoothing/class_weights
    둘 다 optimizer의 param_groups와 무관한 순수 criterion 생성자 인자다.
    resume 시 자유롭게 바꿀 수 있는 이유(gradient_clip_norm과 동일한 결론,
    아래 require_compatible_resume_config 설명 참고)는 "criterion의
    state_dict()가 항상 비어서"가 아니다 -- weight가 설정된 CrossEntropyLoss는
    실제로 weight buffer를 가져 state_dict()가 비어있지 않다. 진짜 이유는
    checkpoint subsystem(training/checkpoint.py)이 criterion의 state 자체를
    애초에 저장하지도 복원하지도 않기 때문이다 -- optimizer/scheduler처럼
    checkpoint에서 load_state_dict()로 복원되어 새 config 값을 조용히
    덮어쓸 경로가 criterion에는 없다.

    resume_state가 주어지면(Phase 4F), 다음 순서로 진행한다:
    1. resume_state.history.stopped_early=True는 여기서 거부한다
       (TrainingResumeState 생성 시점의 __post_init__에서도 이미 거부
       하지만, dataclass가 frozen이 아니라 생성 후 mutate될 수 있으므로
       재확인한다). early stopping으로 끝난 학습은 재개 대상이 아니다.
       (checkpoint 파일 자체는 stopped_early=True여도
       checkpoint.load_training_checkpoint()로 조회/가중치 추출은
       가능하다 -- 거부되는 것은 "resume 실행"뿐이다.)
    2. resume_state.training_config와 이번 config가 optimizer/scheduler
       구조 관련 필드에서 일치하는지 확인한다
       (config.require_compatible_resume_config, 아래 참고) -- caller가
       이 검증을 별도로 호출하는 것은 조기 실패(fail fast)를 위한 선택
       사항일 뿐, 실제 계약은 여기서 항상 강제된다. caller가 검증을
       빼먹고 바로 run_training()을 호출해도 우회할 수 없다.
    **위 두 검증은 optimizer/scheduler를 생성하기 전에 끝난다** -- config가
    호환되지 않으면 그 즉시 거부하고, 불필요한 optimizer/scheduler
    객체를 만들지 않는다.
    3. optimizer/scheduler를 config로 새로 생성한 뒤, resume_state에 저장된
       state_dict를 로드한다 (resume_state의 텐서를 그대로 aliasing하지
       않도록 deepcopy 후 로드 -- 그러지 않으면 이후 최적화 스텝이
       호출자가 들고 있는 resume_state/checkpoint payload의 텐서를 조용히
       변형시킬 수 있다). config.precision="fp16"+CUDA device면 같은
       원칙으로 `_build_precision_execution()`이 만든 GradScaler에도
       resume_state.scaler_state_dict가 있으면 로드한다(Phase 4S) --
       단 scheduler와 달리 존재/부재 불일치를 에러로 취급하지 않는다.
       `precision`은 RESUME_CONFIG_FIELDS가 아니라 자유롭게 바뀔 수 있는
       필드이므로, FP32 checkpoint를 AMP로 resume(scaler는 fresh로 시작)
       하거나 AMP checkpoint를 FP32로 resume(scaler_state_dict를 그냥
       쓰지 않음)하는 것 모두 정상 동작이다(portable, exact는 same
       precision끼리만 보장 -- 아래 checkpoint_hook 문단 및 docs/
       phase4s_amp_mixed_precision_design.md 참고).
    4. history/best_state_dict는 resume_state의 값을 deepcopy해서 이어받는다
       (resume_state.history 원본 객체를 직접 수정하지 않는다).
    5. epoch 번호는 completed_epochs = len(history.train_losses) 이후부터
       시작한다 (`config.epochs`는 "이번 호출에서 추가로 실행할 epoch 수"라는
       의미를 resume 여부와 무관하게 항상 그대로 유지한다 -- resume라고
       "총 목표 epoch"로 재해석하지 않는다). 즉 이전에 3 epoch를 완료하고
       resume_state로 이어받은 뒤 config.epochs=2면, 실행되는 epoch 번호는
       4, 5이고 best_epoch도 이 절대 번호 기준으로 기록된다.

    model은 호출 전에 이미 device로 옮겨져 있어야 함.

    매 epoch의 순서는 train -> validation -> history 기록 -> best
    model/개선 카운터 갱신 -> scheduler.step(val_loss) -> early stopping
    조건 확인 -> checkpoint_hook 호출 -> progress_callback 호출 ->
    (early stopping이 아니고 다음 epoch가 남아 있을 때만) should_stop()
    평가다(Phase 4I/4J, docs/phase4i_training_progress_and_stop_design.md
    §4/§7, docs/phase4j_epoch_checkpoint_design.md §3-1). 즉 마지막으로
    실행된 epoch에서 scheduler가 LR을 바꿨더라도, 그 직후 early
    stopping으로 멈추면 바뀐 LR은 실제로 쓰이지 않을 수 있다 -- 이는
    의도된 동작이다(다음 epoch이 없으므로 바뀐 LR을 쓸 기회 자체가
    없을 뿐, 계산 자체는 정상 수행됨).
    `config.early_stopping_patience`와 `config.lr_scheduler_patience`를
    함께 쓸 때는 early_stopping_patience를 lr_scheduler_patience보다
    크게 잡는 것을 권장한다 -- 그래야 LR이 줄어든 뒤에도 실제로 몇
    epoch 더 학습할 기회가 생긴다 (강제 검증 규칙은 아님).

    progress_callback(Phase 4I)이 주어지면, 완료된 epoch마다(early
    stopping으로 끝난 epoch 포함) 정확히 한 번 TrainingProgress를
    만들어 호출한다. 콜백이 예외를 던지면 그대로 전파한다(감싸지
    않음) -- 이 경우 TrainingResult는 반환되지 않으므로,
    optimizer/scheduler/history를 하나로 묶어 받을 방법이 없다(단
    `model`은 호출자가 넘긴 바로 그 참조이므로 예외 후에도 호출자
    쪽에서 계속 접근 가능하다). 의도적으로 학습을 중단하려면 콜백에서
    예외를 던지지 말고 should_stop을 쓸 것 -- 그래야 run_training()이
    정상적인 TrainingResult를 반환하고, 상위 workflow(예:
    run_imagefolder_training_workflow())가 checkpoint_out 등 기존
    artifact 후처리를 계속 수행할 수 있다.

    should_stop(Phase 4I)이 주어지면, "이번 epoch이 early stopping으로
    끝난 게 아니고, 이번 호출에서 아직 실행할 epoch가 하나라도 남아
    있을 때만" progress_callback 호출 **직후** 평가한다 -- epoch
    시작 전에는 절대 평가하지 않으므로 항상 최소 1개의 새 epoch이
    완료된다(0 epoch 결과는 불가능). should_stop()이 True이면
    `history.stopped_by_user = True`를 설정하고 멈춘다. 이 순서
    (콜백을 먼저 호출한 뒤 should_stop을 평가) 덕분에, 콜백 안에서
    동기적으로 stop 플래그를 세팅하는 UI 패턴은 지연 없이 같은 epoch
    경계에서 바로 반영된다. 이번 호출의 마지막 요청 epoch이거나
    `config.epochs == 1`이면 should_stop()은 절대 평가되지 않는다
    (남길 epoch이 없으므로 "중단"이 의미가 없음) -- 이 경우
    `stopped_by_user`는 `False`로 남는다. early stopping과
    `stopped_by_user`는 동시에 True가 될 수 없다(early stopping이
    항상 우선).

    checkpoint_hook(Phase 4J, docs/phase4j_epoch_checkpoint_design.md
    §3/§4)이 주어지면, 완료된 epoch마다(early stopping으로 끝난 epoch
    포함) `scheduler.step()`과 early stopping 판정 이후, progress_callback
    보다 먼저 정확히 한 번 `EpochCheckpointView`를 만들어 호출한다.
    이 시점에는 아직 should_stop()을 평가하지 않았으므로 view가 담는
    `history.stopped_by_user`는 항상 `False`다. hook이 예외를 던지면
    그대로 전파한다(감싸지 않음) -- progress_callback과 같은 이유로
    `TrainingResult`는 반환되지 않는다. `checkpoint_hook=None`(기본값)이면
    이 호출 자체가 없으므로 기존 동작과 완전히 동일하다.
    `loader_generator`는 `train_loader.generator`를 그대로 전달한다
    (DataLoader가 생성자에서 받은 값을 그대로 보관하는 attribute이며,
    전달하지 않았으면 `None`이다).
    """
    if resume_state is not None:
        # resume 관련 사전 검증은 optimizer/scheduler를 만들기 **전에** 끝낸다 --
        # config가 호환되지 않으면 잘못된 optimizer/scheduler 객체를 만들
        # 이유가 없다 (생성 자체가 비싸지는 않지만, "검증 실패 시 아무
        # 부수효과 없이 즉시 거부"라는 fail-fast 원칙에 맞춘다).
        #
        # resume_state.history.stopped_early는 TrainingResumeState.__post_init__이
        # 생성 시점에 이미 거부하지만, dataclass는 frozen이 아니라 생성 이후
        # resume_state.history.stopped_early = True처럼 직접 mutate될 수 있으므로
        # run_training() 진입 시에도 다시 한번 방어한다.
        if resume_state.history.stopped_early:
            raise ValueError(
                "cannot resume: resume_state.history.stopped_early is True -- early stopping "
                "already decided this training run is finished. To keep training from these "
                "weights, call model.load_state_dict(resume_state.best_state_dict) or load the "
                "checkpoint payload's model_state_dict, then start a fresh TrainingConfig/"
                "run_training() call."
            )

        # config 호환성도 caller가 별도로 확인하지 않아도 여기서 항상 강제된다
        # (require_compatible_resume_config를 caller가 빼먹어도 우회 불가).
        require_compatible_resume_config(resume_state.training_config, config)

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    # criterion은 optimizer/scheduler와 마찬가지로 epoch 루프 진입 전 한 번만
    # 생성해 재사용한다 -- resume 여부와 무관하게 매번 config로 새로 만든다.
    # weight가 설정된 CrossEntropyLoss는 실제로 weight buffer를 가지므로
    # state_dict()가 비어있지 않을 수 있다(Phase 4P) -- 그럼에도 resume 시
    # 자유롭게 값을 바꿀 수 있는 이유는 "state_dict가 항상 비어서"가 아니라,
    # checkpoint subsystem(training/checkpoint.py)이 criterion의 state 자체를
    # 애초에 저장/복원하지 않기 때문이다(optimizer_state_dict/scheduler_state_dict
    # 만 checkpoint에 저장됨) -- 그래서 optimizer/scheduler처럼 resume_state에서
    # 로드할 criterion state가 없다.
    criterion = _build_criterion(config, device=device)
    # autocast_dtype/scaler는 optimizer/scheduler/criterion과 마찬가지로
    # epoch 루프 진입 전 config로 한 번만 계산한다(Phase 4S/4T). scheduler와
    # 달리 config 호환성이 엄격히 강제되지 않는다 -- precision은
    # RESUME_CONFIG_FIELDS가 아니므로(아래 resume 블록의 scaler 처리 참고)
    # 여기서 계산하는 값은 "새 config가 요청한" precision을 그대로 따른다.
    autocast_dtype, scaler = _build_precision_execution(config, device)

    if resume_state is not None:
        optimizer.load_state_dict(copy.deepcopy(resume_state.optimizer_state_dict))

        if scheduler is not None:
            if resume_state.scheduler_state_dict is None:
                raise ValueError(
                    "config.lr_scheduler enables a scheduler but resume_state.scheduler_state_dict "
                    "is None -- the checkpoint was saved with a different (or no) scheduler"
                )
            scheduler.load_state_dict(copy.deepcopy(resume_state.scheduler_state_dict))
        elif resume_state.scheduler_state_dict is not None:
            raise ValueError(
                "resume_state.scheduler_state_dict is set but config.lr_scheduler is None -- "
                "the checkpoint was saved with a scheduler that this resume config does not enable"
            )

        # scaler는 scheduler와 달리 존재/부재 불일치를 에러로 취급하지
        # 않는다(Phase 4S) -- precision은 RESUME_CONFIG_FIELDS가 아니라
        # 자유롭게 바뀔 수 있는 필드이므로, "이번 config는 fp16인데
        # checkpoint에는 scaler state가 없음"(legacy 또는 FP32 checkpoint를
        # AMP로 resume)과 "checkpoint에는 scaler state가 있는데 이번
        # config는 fp32"(AMP checkpoint를 FP32로 resume) 둘 다 정상
        # 케이스로 허용한다(portable, exact는 미보장). 전자는 scaler가
        # 방금 fresh 생성된 그대로 진행하고, 후자는 그 값을 그냥 쓰지
        # 않는다(scaler가 None이므로 로드할 대상이 없음).
        if scaler is not None and resume_state.scaler_state_dict is not None:
            scaler.load_state_dict(copy.deepcopy(resume_state.scaler_state_dict))

        history = copy.deepcopy(resume_state.history)
        # 이전 checkpoint가 사용자 중단으로 저장된 것이었다면
        # (resume_state.history.stopped_by_user == True) 그 값이 그대로
        # 복사되어 온다 -- 이번 호출은 아직 멈춘 적이 없으므로 명시적으로
        # 되돌린다. stopped_early는 TrainingResumeState.__post_init__이
        # True인 경우를 이미 거부하므로 이 리셋이 필요 없다.
        history.stopped_by_user = False
        best_state_dict = copy.deepcopy(resume_state.best_state_dict)
        epochs_without_improvement = resume_state.epochs_without_improvement
        completed_epochs = len(history.train_losses)
    else:
        history = TrainingHistory()
        best_state_dict = None
        epochs_without_improvement = 0
        completed_epochs = 0

    for epoch in range(completed_epochs + 1, completed_epochs + config.epochs + 1):
        run_epoch = epoch - completed_epochs

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device=device,
            gradient_clip_norm=config.gradient_clip_norm, criterion=criterion,
            autocast_dtype=autocast_dtype, scaler=scaler,
        )
        history.train_losses.append(train_loss)
        val_loss, val_accuracy = evaluate(model, val_loader, device=device)
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_accuracy)
        learning_rate = optimizer.param_groups[0]["lr"]

        if history.best_val_loss is None or val_loss < history.best_val_loss:
            history.best_epoch = epoch
            history.best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            history.stopped_early = True

        if checkpoint_hook is not None:
            checkpoint_hook(
                EpochCheckpointView(
                    model=model,
                    history=history,
                    best_state_dict=best_state_dict,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs_without_improvement=epochs_without_improvement,
                    loader_generator=train_loader.generator,
                    scaler_state_dict=(scaler.state_dict() if scaler is not None else None),
                )
            )

        if progress_callback is not None:
            progress_callback(
                TrainingProgress(
                    run_epoch=run_epoch,
                    total_run_epochs=config.epochs,
                    global_epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    learning_rate=learning_rate,
                    best_epoch=history.best_epoch,
                    best_val_loss=history.best_val_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    stopped_early=history.stopped_early,
                )
            )

        # should_stop()은 "이 요청이 실제로 뭔가를 단축시킬 수 있을 때"만
        # 평가한다 -- early stopping으로 이미 끝났거나, 이번이 이번 호출의
        # 마지막 요청 epoch라면(더 이상 건너뛸 epoch가 없음) 평가해도
        # 의미 있는 조기 종료가 아니다.
        has_next_epoch = run_epoch < config.epochs
        if (
            not history.stopped_early
            and has_next_epoch
            and should_stop is not None
            and should_stop()
        ):
            history.stopped_by_user = True

        if history.stopped_early or history.stopped_by_user:
            break

    # config.epochs >= 1은 TrainingConfig.__post_init__이 이미 보장하므로
    # 루프가 최소 1회는 돌고, resume이 아닌 한(또는 resume_state가 이미
    # best_state_dict를 채워 온 경우) epoch 1에서 best_val_loss가 None이라
    # 항상 best_state_dict가 채워진다.
    return TrainingResult(
        history=history,
        best_state_dict=best_state_dict,
        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
        scheduler_state_dict=copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        epochs_without_improvement=epochs_without_improvement,
        scaler_state_dict=copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
    )
