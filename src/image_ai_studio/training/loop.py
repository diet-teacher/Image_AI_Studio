"""학습/평가 루프. loss=CrossEntropyLoss는 고정. optimizer(Adam/SGD)와 LR
scheduler(없음/ReduceLROnPlateau)는 Phase 4E부터 TrainingConfig로 선택
가능 -- 선택지가 2개/1개뿐이라 registry 없이 이 모듈의 private helper
(_build_optimizer/_build_scheduler)로 충분하다."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_ai_studio.training.config import TrainingConfig, require_compatible_resume_config


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """config.optimizer에 따라 Adam 또는 SGD를 생성. TrainingConfig.__post_init__이
    이미 "adam"/"sgd" 외의 값을 거부하므로 그 외 분기는 없다."""
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


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
    """run_training()의 epoch별 결과 + best epoch 추적.

    JSON으로 그대로 직렬화 가능한 순수 메트릭만 담는다 (텐서 값인
    state_dict는 여기 두지 않고 TrainingResult가 별도로 가진다 --
    training/history.py의 save_training_history()가 이 dataclass를
    그대로 asdict()해서 저장한다).

    best_epoch는 1-indexed. val_loss가 strict하게(<) 더 낮아질 때만
    갱신되므로, 동률이면 먼저 나온 epoch가 유지된다.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    stopped_early: bool = False


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
    """

    history: TrainingHistory
    best_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict
    scheduler_state_dict: dict | None
    epochs_without_improvement: int


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
    """

    optimizer_state_dict: dict
    scheduler_state_dict: dict | None
    history: TrainingHistory
    epochs_without_improvement: int
    best_state_dict: dict[str, torch.Tensor]
    training_config: dict

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

    optimizer/scheduler는 config에 따라 여기서 생성한다. resume_state가
    None(기본값)이면 Phase 4E까지의 동작과 완전히 동일 -- 매번 새로
    생성하고, epoch 1부터, 빈 history로 시작한다.

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
       변형시킬 수 있다).
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
    조건 확인이다. 즉 마지막으로 실행된 epoch에서 scheduler가 LR을
    바꿨더라도, 그 직후 early stopping으로 멈추면 바뀐 LR은 실제로
    쓰이지 않을 수 있다 -- 이는 의도된 동작이다(다음 epoch이 없으므로
    바뀐 LR을 쓸 기회 자체가 없을 뿐, 계산 자체는 정상 수행됨).
    `config.early_stopping_patience`와 `config.lr_scheduler_patience`를
    함께 쓸 때는 early_stopping_patience를 lr_scheduler_patience보다
    크게 잡는 것을 권장한다 -- 그래야 LR이 줄어든 뒤에도 실제로 몇
    epoch 더 학습할 기회가 생긴다 (강제 검증 규칙은 아님).
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

        history = copy.deepcopy(resume_state.history)
        best_state_dict = copy.deepcopy(resume_state.best_state_dict)
        epochs_without_improvement = resume_state.epochs_without_improvement
        completed_epochs = len(history.train_losses)
    else:
        history = TrainingHistory()
        best_state_dict = None
        epochs_without_improvement = 0
        completed_epochs = 0

    for epoch in range(completed_epochs + 1, completed_epochs + config.epochs + 1):
        history.train_losses.append(train_one_epoch(model, train_loader, optimizer, device=device))
        val_loss, val_accuracy = evaluate(model, val_loader, device=device)
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_accuracy)

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
    )
