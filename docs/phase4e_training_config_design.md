# Phase 4E: TrainingConfig 확장 (optimizer / scheduler / early stopping)

이 문서는 설계 검토와 실제 구현/실행 결과를 함께 정리한다. Phase 4A~4D는
optimizer=Adam, loss=CrossEntropyLoss, scheduler 없음, early stopping
없음으로 전부 고정되어 있었다. Phase 4E의 목표는 이 학습 경로에 **사용자가
지정할 수 있는 최소한의 학습 설정 계층**을 추가하는 것이다:

```text
기존: TrainingConfig(epochs, batch_size, learning_rate) -> 항상 Adam, 항상 config.epochs만큼 실행

Phase 4E: TrainingConfig에 optimizer(Adam/SGD), LR scheduler(없음/
ReduceLROnPlateau), early stopping(없음/patience)을 추가로 지정 가능
```

## 1. 범위

지원하는 것:

* `optimizer`: `"adam"` | `"sgd"` (SGD는 `momentum`도 함께 지정)
* `lr_scheduler`: `None` | `"plateau"` (`ReduceLROnPlateau`, `factor`/`patience` 지정)
* `early_stopping_patience`: `None` | 양의 정수
* `TrainingHistory.stopped_early: bool`

지원하지 않는 것(의도적, 17절 참고): loss 선택, Adam betas/weight decay,
SGD dampening/nesterov, scheduler threshold/cooldown/min_lr, StepLR 등
추가 scheduler, full checkpoint/resume, augmentation, 자동 split,
registry/factory, GUI.

`src/image_ai_studio` 아래 **production code**에서는
`training/config.py`와 `training/loop.py`(`run_training()`/
`TrainingConfig`/`TrainingHistory`가 정의된 두 파일)만 수정했다.
`training/checkpoint.py`(모델 가중치만 저장, optimizer/scheduler state는
여전히 범위 밖), `training/history.py`의 저장/로드 함수(`TrainingHistory`를
그대로 직렬화/역직렬화하므로 코드 수정 불필요), `training/dataset.py`/
`torchvision_dataset.py`, `model_definition/*`, `export/*`, C++ 코드,
`scripts/run_training_e2e.py`/`scripts/run_real_training_e2e.py`(Phase
4A/4B/4C 회귀 앵커)는 전부 수정하지 않았다.

이 범위 밖에서 실제로 변경/추가된 파일은 다음과 같다 (production
code가 아니라 CLI/테스트/문서): `scripts/run_imagefolder_training_e2e.py`
(새 CLI 플래그), `tests/training/test_config.py`/`test_loop.py`/
`test_history.py`(신규 테스트), `README.md`, 그리고 이 문서
(`docs/phase4e_training_config_design.md`) 자체.

## 2. `TrainingConfig` 신규 필드

```python
OPTIMIZER_CHOICES = ("adam", "sgd")
LR_SCHEDULER_CHOICES = ("plateau",)

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float

    optimizer: str = "adam"
    momentum: float = 0.9

    lr_scheduler: str | None = None
    lr_scheduler_factor: float = 0.1
    lr_scheduler_patience: int = 1

    early_stopping_patience: int | None = None
```

새 registry/factory 모듈은 만들지 않았다 -- 선택지가 optimizer 2개,
scheduler 1개뿐이라 `loop.py`의 private 함수(`_build_optimizer`/
`_build_scheduler`) 안의 `if`/`return` 몇 줄로 충분하다(3절).

### Validation 계약

```text
epochs > 0, batch_size > 0, learning_rate > 0        (변경 없음)
optimizer ∈ {"adam", "sgd"}
0.0 <= momentum < 1.0
lr_scheduler ∈ {None, "plateau"}
0.0 < lr_scheduler_factor < 1.0
lr_scheduler_patience > 0
early_stopping_patience is None 또는 > 0
```

`_require_one_of`(문자열 선택지 검증)와 `_require_fraction`(0~1 사이 값
검증) 두 헬퍼를 `config.py`에 추가했다. `_require_one_of`는
`model_definition/specs.py`의 `BranchSpec.merge ∈ {"add", "concat"}`
검증과 동일한 패턴이라 새로운 검증 스타일이 아니다.

**`momentum`은 `optimizer="adam"`이어도 항상 검증한다.** SGD를 쓰지
않으면 실제로 사용되지 않는 값이지만, `TrainingConfig` 인스턴스가 항상
전체적으로 유효한 상태를 유지하도록 하기 위해서다 -- 예를 들어 GUI에서
사용자가 momentum 슬라이더를 먼저 조절하고 optimizer 드롭다운을 나중에
바꾸는 순서도 자연스럽게 허용된다. `lr_scheduler_factor`/
`lr_scheduler_patience`도 같은 이유로 `lr_scheduler`가 `None`이어도
항상 검증한다.

## 3. optimizer / scheduler 생성 (`loop.py`, private helper)

```python
def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def _build_scheduler(optimizer, config: TrainingConfig):
    if config.lr_scheduler is None:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience
    )
```

`config.optimizer == "adam"`(기본값)일 때 `_build_optimizer`는 Phase
4A~4D가 항상 실행하던 `torch.optim.Adam(model.parameters(),
lr=config.learning_rate)`와 정확히 같은 호출을 반환한다 -- 이게 이번
Phase의 핵심 회귀 계약이다(5절에서 실측으로 확인).

## 4. Early stopping의 정확한 의미

`early_stopping_patience=N`은 다음을 뜻한다:

> validation loss가 N회 연속 strict improvement(`val_loss <
> best_val_loss`)를 만들지 못하면, N번째 non-improving epoch를 **완료한
> 뒤** 학습을 중단한다.

`run_training()`은 매 epoch마다 `epochs_without_improvement` 카운터를
갱신한다: `val_loss`가 개선되면 0으로 리셋, 아니면 +1. 카운터가
`early_stopping_patience` 이상이 되면 그 epoch를 끝낸 직후 루프를
`break`한다. 동률(`val_loss == best_val_loss`)은 기존 best epoch 선택
계약과 동일하게 개선으로 취급하지 않는다(strict `<`).

예 (patience=2):

```text
epoch 1: val_loss=1.0 -> best (카운터 0)
epoch 2: val_loss=1.0 -> 개선 없음 (카운터 1)
epoch 3: val_loss=1.0 -> 개선 없음 (카운터 2 == patience) -> epoch 3 종료 후 중단
```

`epoch 4`는 절대 실행되지 않는다 -- 이 off-by-one 경계를
`test_run_training_stops_exactly_after_patience_non_improving_epochs`가
정확히 이 시퀀스로 고정한다(`history.stopped_early is True`,
`len(history.train_losses) == 3`, `history.best_epoch == 1`).
`early_stopping_patience=0`은 `TrainingConfigError`로 거부한다(최솟값 1).

`early_stopping_patience=None`(기본값)이면 이 카운터는 계산은 되지만
중단 조건에서 절대 쓰이지 않으므로, `config.epochs`를 항상 전부
실행하는 기존 동작이 그대로 유지된다.

## 5. Scheduler와 early stopping의 epoch 내 실행 순서

매 epoch는 다음 순서로 실행된다:

```text
train_one_epoch()
    -> evaluate()
    -> history 기록 (train_loss/val_loss/val_accuracy)
    -> best model 갱신 또는 epochs_without_improvement 증가
    -> scheduler.step(val_loss)      (scheduler가 있으면)
    -> early stopping 조건 확인       (early_stopping_patience가 있으면)
```

이 순서를 선택한 이유:

* `evaluate()`가 이미 매 epoch validation loss를 계산해 두므로,
  `ReduceLROnPlateau`가 필요로 하는 metric을 별도로 다시 계산할 필요가
  없다 -- scheduler는 그 값을 그대로 받아 쓴다.
* best epoch/early stopping 판단을 먼저 마친 뒤에 scheduler를
  건드리므로, "이번 epoch가 best인지"와 "LR을 줄일지"가 서로의 계산에
  끼어들지 않는다(순서를 바꿔도 두 판단은 독립적이라 실제로는 영향이
  없지만, 읽는 순서와 실행 순서를 일치시켜 두는 편이 코드를 더 예측
  가능하게 만든다).

**이 순서의 부수 효과**: 마지막으로 실행된 epoch에서 scheduler가 LR을
줄였더라도, 그 직후 early stopping으로 학습이 끝나면 줄어든 LR은 실제로
쓰일 기회가 없다(다음 epoch가 없으므로). 이건 **의도된 동작**이다 --
`scheduler.step()` 자체는 정상적으로 실행되고 optimizer 내부 상태도
정상적으로 바뀌지만, 그 상태를 소비할 다음 학습 스텝이 없을 뿐이다.
LR 값 자체는 `run_training()`이 저장하거나 반환하지 않으므로(4절/6절
참고, 이 정보는 checkpoint/resume 범위) 사용자에게 혼란을 주는
불일치는 없다.

**권장 사항(강제 아님)**: `early_stopping_patience`와
`lr_scheduler_patience`를 함께 쓸 때는 `early_stopping_patience >
lr_scheduler_patience`로 두는 것을 권장한다 -- 그래야 LR이 줄어든
뒤에도 실제로 몇 epoch 더 학습할 기회가 생긴다. 이 관계를 강제하는
validation 규칙은 만들지 않았다(예: `lr_scheduler_patience`가 더 큰
조합도 유효한 설정일 수 있고, 이를 금지할 만큼 명백히 항상 잘못된
조합은 아니기 때문).

## 6. `TrainingHistory.stopped_early`

```python
@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    stopped_early: bool = False
```

`completed_epochs` 같은 별도 필드는 만들지 않았다 -- 실제 실행된 epoch
수는 `len(history.train_losses)`로 이미 알 수 있다. `run_training()`의
로컬 변수인 `epochs_without_improvement`도 `TrainingHistory`에 넣지
않았다 -- 이건 사용자에게 보여줄 결과 metric이 아니라 학습 도중의
실행 상태(resume 시에나 필요한 정보)이므로, "checkpoint/resume는 이번
Phase 범위가 아니다"라는 원칙에 따라 넣지 않았다.

### 하위 호환

`load_training_history()`는 여전히 `TrainingHistory(**data)`로 복원한다
(코드 수정 없음). Phase 4D까지 저장된 JSON에는 `stopped_early` 키가
없는데, `**data`가 그 키를 안 넘기면 dataclass 기본값(`False`)이 그대로
적용된다 -- 실제로 `stopped_early` 키가 없는 legacy 형식 JSON을 손으로
만들어 `load_training_history()`로 읽어서 `stopped_early is False`,
나머지 필드는 정상 복원됨을 확인했다
(`test_load_training_history_defaults_stopped_early_when_key_is_missing`).

## 7. `run_training()` 호환성

외부 시그니처는 변경하지 않았다:

```python
def run_training(model, train_loader, val_loader, config, device="cpu") -> TrainingResult:
```

인자 개수/순서/타입, 반환 타입(`TrainingResult`) 전부 Phase 4A~4D와
동일하다. 함수 내부만 `torch.optim.Adam(...)` 한 줄을
`_build_optimizer(model, config)` 호출로, 그리고 scheduler/early
stopping 로직을 추가로 갖도록 확장했다.

기존 호출 패턴(`TrainingConfig(epochs=..., batch_size=...,
learning_rate=...)`, optimizer/scheduler/early_stopping 인자를 전혀
지정하지 않음)은 코드 수정 없이 그대로 컴파일/실행되고, 기본값이
`optimizer="adam"`, `lr_scheduler=None`, `early_stopping_patience=None`
이므로 **기존과 동일한 학습 경로**를 그대로 탄다.

## 8. 사용자 대면 CLI (`run_imagefolder_training_e2e.py`)

`scripts/run_imagefolder_training_e2e.py`에 새 CLI 플래그를 추가했다
(Phase 4C `run_real_training_e2e.py`, Phase 4A/4B `run_training_e2e.py`
회귀 앵커는 변경하지 않음):

```bash
python scripts/run_imagefolder_training_e2e.py \
    --optimizer sgd --momentum 0.9 \
    --lr-scheduler plateau --lr-scheduler-factor 0.5 --lr-scheduler-patience 1 \
    --early-stopping-patience 3
```

모든 플래그는 생략 가능하고, 생략 시 `optimizer=adam`,
`lr_scheduler=None`(scheduler 없음), `early_stopping_patience=None`
(early stopping 없음)으로 기존 동작을 그대로 재현한다. `--lr-scheduler`
는 `argparse`의 `choices=["plateau"]` + `default=None`으로 표현했다 --
플래그를 아예 안 주면 `None`이 되는 표준 argparse 관례를 그대로
따랐고, 별도의 `"none"` 문자열 sentinel 등을 도입하지 않았다.

## 9. 신규 unit test

**`tests/training/test_config.py`** (8 -> 29, +21):
optimizer 정상/거부, momentum 범위(0 포함/1 미포함 경계 포함),
optimizer="adam"이어도 momentum이 검증됨, scheduler 정상/거부, scheduler
factor/patience 범위, early_stopping_patience 정상/거부, 기본값이
Phase 4A~4D 동작을 재현하는지.

**`tests/training/test_loop.py`** (11 -> 21, +10):
`_build_optimizer`/`_build_scheduler` 직접 단위 테스트(타입, lr/momentum/
factor/patience 값 일치), `ReduceLROnPlateau`가 실제로 몇 번째
`step()` 호출에서 LR을 바꾸는지(아래 참고), SGD로 `run_training()`이
여전히 loss를 감소시키는지, early stopping 비활성화 시 전체 epoch 실행,
patience 경계값(off-by-one) 고정, 동률은 개선이 아님, patience=1
최소 경계.

**`tests/training/test_history.py`** (4 -> 7, +3):
`stopped_early=True`/`False` round-trip, `stopped_early` 키가 없는
legacy JSON의 하위 호환 로드.

## 10. Scheduler 동작 실측

`ReduceLROnPlateau(factor=0.5, patience=2)`에 동일한 loss(1.0)를 계속
`step()`으로 넣었을 때 실제 PyTorch 동작을 직접 실행해 확인했다:

```text
call 1: lr 1.0 -> 1.0 (기준값 설정, 아직 "나쁜" epoch 아님)
call 2: lr 1.0 -> 1.0 (나쁜 epoch 1회)
call 3: lr 1.0 -> 1.0 (나쁜 epoch 2회)
call 4: lr 1.0 -> 0.5 (나쁜 epoch 3회 > patience=2 -> 감소)
```

즉 `patience=P`는 "첫 `step()`은 baseline(기준값)을 세울 뿐 LR을 바꾸지
않고, 그 이후로 개선되지 않는(bad) epoch 수가 `P`를 초과하는 `step()`
호출에서 LR이 감소한다"는 PyTorch 자체의 동작이다. 이 순번을
`test_build_scheduler_reduces_lr_after_patience_bad_steps`에서
`lr_scheduler_patience=2`로 그대로 고정했다(4번째 `step()`에서 최초로
LR이 바뀜을 assert).

## 11. 실제 실행 검증 결과

Windows 11, PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, GTX 1080에서
전부 실제로 실행하여 확인했다 (추정치 없음):

* **신규 unit test**: `test_config.py` 29 passed, `test_loop.py` 21
  passed, `test_history.py` 7 passed
* **`tests/training/` 전체**: 95 passed (Phase 4D까지의 61 + 신규 34)
* **전체 `pytest`**: 252 passed (Phase 4D까지의 218 + 신규 34)
* **Phase 0 regression** (`scripts/run_torchscript_tests.py`):
  `tiny_cnn`/`tiny_residual_cnn` CPU/CUDA 전부 PASS
* **Phase 1~3 E2E regression** (4개 예시 JSON): 전부 PASS
* **Phase 4A/4B synthetic E2E** (`scripts/run_training_e2e.py`, 수정
  없음, 기본 설정): train loss 1.3386 -> 0.2867, best epoch 10 --
  Phase 4D 시점에 기록된 것과 동일한 epoch별 train/val loss, best
  epoch가 콘솔 출력 기준으로 재현되었다(개별 텐서/artifact 파일을
  바이트 단위로 비교한 것은 아니다). `optimizer="adam"`(기본값)이
  Phase 4D와 동일한 `torch.optim.Adam(model.parameters(),
  lr=config.learning_rate)` 호출을 그대로 생성하므로(3절), 이 재현은
  우연이 아니라 코드 경로가 실제로 바뀌지 않았다는 근거다
* **Phase 4C CIFAR-10 E2E** (`scripts/run_real_training_e2e.py`, 수정
  없음, 기본 설정): best epoch 4, test_accuracy=0.1953 -- 마찬가지로
  Phase 4D 시점 출력과 동일한 수치가 재현되었다
* **Phase 4D ImageFolder E2E, 기본 설정** (`--optimizer`/`--lr-scheduler`/
  `--early-stopping-patience` 전부 생략): best epoch 5,
  test_accuracy=0.2600 -- 역시 Phase 4D 시점 출력과 동일한 수치가
  재현되었다
* **Phase 4D ImageFolder E2E, 신규 설정 조합** (`--optimizer sgd
  --momentum 0.9 --lr-scheduler plateau --lr-scheduler-factor 0.5
  --lr-scheduler-patience 1 --early-stopping-patience 3`):

  ```text
  Training config:
    optimizer=sgd (momentum=0.9)
    lr_scheduler=plateau (factor=0.5, patience=1)
    early_stopping_patience=3
  epoch 1: train_loss=2.4249 val_loss=2.2890 val_acc=0.1200
  epoch 2: train_loss=2.3541 val_loss=2.2510 val_acc=0.1200
  epoch 3: train_loss=2.2836 val_loss=2.2115 val_acc=0.1800
  epoch 4: train_loss=2.2428 val_loss=2.1912 val_acc=0.2200
  epoch 5: train_loss=2.2179 val_loss=2.1730 val_acc=0.2800
  stopped_early=False
  Best epoch: 5, Best validation loss: 2.1730
  Class mapping save/reload: PASS
  test_loss=2.2420 test_accuracy=0.1800
  Best model save/reload: PASS
  TorchScript export: PASS
  C++ TorchScript runner: CPU PASS, CUDA PASS
  Parity: PASS

  PHASE 4D E2E: PASS
  ```

  이 실행에서는 5 epoch 동안 val_loss가 계속 개선되어 scheduler/early
  stopping이 실제로 발동하지는 않았다(`stopped_early=False`) -- 이는
  fixture가 작고 학습이 짧기 때문이며 예상된 결과다. scheduler 감소
  시점과 early stopping의 정확한 경계 동작은 10절/4절의 deterministic
  unit test에서 이미 별도로, 우연에 기대지 않고 고정해 검증했다.

## 12. 이번 Phase 4E에서 의도적으로 구현하지 않은 것

* loss function 선택 (여전히 CrossEntropyLoss 고정)
* Adam betas, weight decay
* SGD dampening, nesterov
* scheduler threshold/cooldown/min_lr
* StepLR/CosineAnnealingLR 등 `"plateau"` 외 scheduler
* full checkpoint(optimizer/scheduler state, epoch 번호), resume
* RNG/DataLoader 반복 상태 저장
* augmentation, 자동 train/val/test split
* dataset/optimizer/scheduler registry 또는 factory
* PySide6 UI

이 목록은 필요성이 구체적으로 확인되기 전까지 보류하며, `TrainingConfig`
구조나 `run_training()`의 외부 시그니처를 바꾸지 않고 확장할 수 있는
지점(선택지 추가, `_build_optimizer`/`_build_scheduler` 분기 추가)에
위치시켰다.
