# Phase 4M: Gradient Norm Clipping — 설계안

**상태: 구현 완료, 전체 단위/통합 테스트 및 기존 E2E 회귀 검증 완료.**
`TrainingConfig`에 `gradient_clip_norm`을 추가하고, `train_one_epoch()`의
`loss.backward()`와 `optimizer.step()` 사이에 `torch.nn.utils.clip_grad_norm_()`
기반 L2 gradient norm clipping을 넣었다. Phase 4L이 확립한 "config 필드 →
core 계산 지점 → CLI" 패턴을 그대로 재사용했고, `checkpoint.py`/
`imagefolder_workflow.py`는 계획대로 무수정으로 끝났다.

**전제**: 이 문서는 별도 조사/계획 라운드(Phase 4L 이후 다음 Phase
후보 A~G 비교, gradient clipping을 Phase 4M으로 채택하고 상세 설계까지
마친 라운드)에서 확정된 정책을 실제 구현에 맞춰 정리한 것이다. 정책
자체의 재검토는 이 문서의 목적이 아니다.

---

## 1. 목표

- **해결하려는 문제**: 현재 `ResidualBlock`/`Branch`(Add/Concat) 등을
  포함한 비교적 깊은 모델 구성도 이미 지원하는데, gradient explosion을
  완화할 수단이 전혀 없다.
- **새로 지원할 기능**: `torch.nn.utils.clip_grad_norm_()` 기반 L2 norm
  clipping. `TrainingConfig.gradient_clip_norm: float | None = None`으로
  켜고 끈다.
- **기본 동작**: 옵션을 지정하지 않으면(`None`) clipping이 전혀 일어나지
  않아 Phase 4A~4L의 기존 동작을 완전히 재현한다.
- **사용자 입장에서 보이는 변화**: `scripts/train_imagefolder.py`에
  `--gradient-clip-norm FLOAT` 플래그가 새로 생긴다.
- **non-goals**: gradient value clipping(`clip_grad_value_`), 사용자
  노출 `norm_type`(L2 고정), `error_if_nonfinite` 노출, gradient
  norm을 history/metric으로 기록, loss/scheduler/metric/GPU 관련 변경,
  checkpoint 정책 변경.

---

## 2. 범위 / non-goals

지원하는 것:

- `TrainingConfig.gradient_clip_norm: float | None = None`
- CLI `--gradient-clip-norm FLOAT`(기본 `None` = clipping 비활성화)
- `gradient_clip_norm`이 설정되면 매 batch, `optimizer.step()` 직전에
  `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)`
  실행(L2 norm, `norm_type` 기본값 2.0 그대로 사용)

지원하지 않는 것(의도적, 이번 Phase 범위 밖):

- `clip_grad_value_()` 기반 value clipping
- `norm_type` 사용자 노출(L2 고정)
- `error_if_nonfinite` config/CLI 노출(PyTorch 기본 동작 그대로 사용)
- clip 전/후 gradient norm을 `TrainingHistory`/progress에 기록
- loss function 선택, 추가 LR scheduler, 평가 metric 확장, GPU/device
  노출, AMP, checkpoint 정책 변경(latest/best 분리, overwrite 등)
- `gradient_clip_norm`을 `RESUME_CONFIG_FIELDS`에 추가(§7에서 이유
  상세 설명)

`src/image_ai_studio` 아래 **production code**에서는 `training/config.py`
와 `training/loop.py`(`train_one_epoch()`/`run_training()` 호출부)만
수정했다. `training/checkpoint.py`, `training/imagefolder_workflow.py`,
`training/history.py`, `_build_optimizer()`, `_build_scheduler()`,
`evaluate()`, `model_definition/*`, `export/*`, `parity/*`, C++ 코드,
`scripts/run_imagefolder_training_e2e.py`(기본값만 쓰므로 anchor 불변)는
전부 수정하지 않았다 -- 실제 구현도 그대로 이 경계를 지켰다
(`git diff --stat`으로 확인).

---

## 3. Phase 4M 구현 전 gradient 계산 구조

이 절은 Phase 4M 착수 시점(구현 전) 조사 내용을 그대로 남긴 기록이다 --
아래 코드는 **구현 전** `train_one_epoch()`(`loop.py`)의 실제 순서였다:

```python
def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu"
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    ...
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ...
```

Phase 4M 구현 전에는 `loss.backward()`와 `optimizer.step()` 사이에
아무 코드도 없었고, 이 지점이 gradient clipping을 넣을 정확한 삽입
위치였다(실제로 이 지점에 삽입됐다 -- §6에 구현 후 코드). `device`는
구현 전에도 keyword-or-positional 파라미터였고, 기존 호출부는 전부
`train_one_epoch(model, loader, optimizer)` 또는 `train_one_epoch(model,
loader, optimizer, device=device)` 형태였다 -- 그래서 새 파라미터를
시그니처 맨 끝에 기본값 있는 keyword로 추가하는 결정을 내렸다(§6에서
상세, 이 결정이 production 호출 호환성과 test double 시그니처에 각각
어떤 영향을 줬는지는 §6에 구분해 정리했다).

구현 전 `run_training()`은 매 epoch `train_one_epoch(model, train_loader,
optimizer, device=device)`를 호출했다 -- 여기서 `config.gradient_clip_norm`
을 그대로 넘기도록 확장하는 것이 계획이었다(§6에 구현 후 실제 코드).
`evaluate()`는 `torch.inference_mode()` 안에서 gradient를 아예 만들지
않으므로 이 기능과 무관해 구현 전후 모두 무수정이다.

---

## 4. `TrainingConfig` 설계

```python
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float

    optimizer: str = "adam"  # "adam" | "sgd" | "adamw"
    momentum: float = 0.9
    weight_decay: float = 0.0

    gradient_clip_norm: float | None = None  # None => clipping 비활성화 (Phase 4M)

    lr_scheduler: str | None = None
    lr_scheduler_factor: float = 0.1
    lr_scheduler_patience: int = 1

    early_stopping_patience: int | None = None
```

필드 위치는 optimizer 관련 필드(`momentum`/`weight_decay`) 바로 다음,
`lr_scheduler` 앞에 둔다 -- gradient clipping은 optimizer 선택과
무관하게 매 backward step에 적용되는 training-stability 파라미터라는
의미를 그대로 반영한다. 기본값 `None`은 기존 동작(clipping 없음)을
그대로 재현한다.

---

## 5. Validation 정책

### 5-1. 정책(확정)

- `None` 허용(비활성화)
- `0.0` 거부
- 음수 거부
- bool 거부
- `NaN`/`+inf`/`-inf` 거부
- 유한한 양수는 **상한 없이** 전부 허용(`1e-12`, `0.1`, `1.0`, `1000.0` 등)

### 5-2. 기존 helper 재사용 가능 여부 (직접 코드로 확인)

`config.py`의 기존 숫자 검증 helper 3개를 실제로 읽고 검증한 결과:

```python
def _require_positive_float(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise TrainingConfigError(f"'{name}' must be a positive number, got {value!r}")
```

**이 helper는 NaN과 +inf를 거부하지 못한다** -- `float("nan") <= 0.0`과
`float("inf") <= 0.0`은 파이썬에서 둘 다 `False`이므로(NaN과의 비교는
항상 False, +inf는 0.0보다 크므로), `isinstance`/`bool` 검사만 통과하면
`or` 체인 전체가 `False`가 되어 예외가 발생하지 않는다 -- 즉 현재
`_require_positive_float()`는 `NaN`과 `+inf`를 **조용히 허용**한다
(`-inf`는 `<=0.0`이 `True`라 정상적으로 거부됨). 이 사실은
`learning_rate`(현재 이 helper로 검증됨)에도 이미 해당되는 기존 코드의
잠재적 결함이지만, 이번 Phase의 범위(gradient_clip_norm 하나)를 벗어나는
"단순 편의를 위한 refactoring"이므로 `_require_positive_float()` 자체는
고치지 않는다.

```python
def _require_non_negative_finite_float(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigError(f"'{name}' must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise TrainingConfigError(f"'{name}' must be a finite number >= 0.0, got {value!r}")
```

이 helper는 `math.isfinite()`로 NaN/±inf를 정확히 거부하지만, 하한이
`>= 0.0`이라 `0.0`을 **허용**한다 -- `gradient_clip_norm`은 `0.0`을
반드시 거부해야 하므로(스펙상 "양의 유한 실수"), 이 helper도 그대로
재사용할 수 없다.

**결론**: 어느 기존 helper도 계약을 정확히 만족하지 않으므로, 이름만
보고 억지로 재사용하지 않고 `_require_non_negative_finite_float()`와
같은 스타일로 새 helper `_require_positive_finite_float()`를 추가한다
(하한만 `>= 0.0` 대신 `> 0.0`으로 바꾼 것과 동치):

```python
def _require_positive_finite_float(name: str, value: object) -> None:
    """0보다 큰 유한한 실수만 허용 (gradient_clip_norm -- 상한은 두지 않는다).
    _require_positive_float()와 달리 NaN/+inf도 math.isfinite()로 명시적으로
    거부한다(NaN <= 0.0과 inf <= 0.0이 둘 다 False라서 _require_positive_float()
    는 이 값들을 조용히 통과시킨다)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigError(f"'{name}' must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise TrainingConfigError(f"'{name}' must be a finite positive number, got {value!r}")
```

`__post_init__()`에는 `early_stopping_patience`와 동일한 패턴으로
"`None`이 아닐 때만 검증"을 적용한다:

```python
if self.gradient_clip_norm is not None:
    _require_positive_finite_float("gradient_clip_norm", self.gradient_clip_norm)
```

---

## 6. Core training loop 변경

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    ...
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        ...
```

`gradient_clip_norm`을 시그니처 맨 끝에 기본값 `None`으로 추가하므로,
기존 production 호출(`train_one_epoch(model, loader, optimizer)`,
`train_one_epoch(model, loader, optimizer, device=device)`)은 전혀
깨지지 않는다 -- 이 둘은 새 keyword 인자를 아예 넘기지 않으므로
기본값 `None`을 그대로 쓴다.

다만 `run_training()`은 아래처럼 `train_one_epoch()`를 호출할 때
`gradient_clip_norm=config.gradient_clip_norm`을 **항상** 명시적으로
넘긴다(`total_norm`(반환값)은 사용하지 않고, `norm_type`/
`error_if_nonfinite`는 PyTorch 기본값 그대로 둔다 -- §2 non-goals):

```python
train_loss = train_one_epoch(
    model, train_loader, optimizer, device=device, gradient_clip_norm=config.gradient_clip_norm
)
```

`run_training()`의 이 호출 계약이 확장됐기 때문에, 기존에
`image_ai_studio.training.loop.train_one_epoch`를 monkeypatch하던
`tests/training/test_loop.py`(2곳)와 `tests/training/test_checkpoint.py`
(1곳)의 fake 함수는 이 새 keyword를 받을 수 있도록 시그니처를

```python
def fake_train_one_epoch(model, loader, optimizer, device="cpu"):
```

에서

```python
def fake_train_one_epoch(model, loader, optimizer, device="cpu", gradient_clip_norm=None):
```

로 갱신해야 했다(§10-2/§14에 실제 조치 기록). **이것은 production
backward compatibility가 깨진 것이 아니다** -- production 코드를
호출하는 실제 caller는 전부 그대로 동작한다. 다만 `run_training()`을
통째로 대체하는 test double은 그 함수의 실제 호출 계약(이제
`gradient_clip_norm`을 항상 keyword로 넘김)을 반영해야 하므로, 이
시그니처 갱신은 test double이 확장된 함수 호출 계약을 뒤따라간
결과다.

Phase 4I~4K가 고정한 epoch-level 순서(train → validation → history →
best model 갱신 → scheduler.step → early stopping → checkpoint_hook →
progress_callback → should_stop)는 전혀 건드리지 않는다 -- 이 기능은
`train_one_epoch()` **내부**, 그것도 batch 루프 안에서 끝나고
`run_training()`의 epoch 루프 구조에는 아무 영향이 없다. `_build_optimizer()`
/`_build_scheduler()`/`evaluate()`도 무수정이다(gradient가 없는 평가
경로와는 애초에 무관).

---

## 7. Checkpoint/resume 정책

`gradient_clip_norm`은 **`RESUME_CONFIG_FIELDS`에 추가하지 않는다.**

이유: `RESUME_CONFIG_FIELDS`가 존재하는 이유는 PyTorch의
`optimizer.load_state_dict()`가 저장된 `param_groups` 값(learning_rate/
momentum/weight_decay)을 그대로 복원해 **생성자에 넘긴 새 값을 조용히
덮어쓰기** 때문이다(`config.py`에 이미 문서화된 근거). `gradient_clip_norm`
은 PyTorch optimizer의 `param_groups`에 전혀 속하지 않는, `train_one_epoch()`
가 매 호출마다 현재 `TrainingConfig`에서 그대로 읽는 순수 runtime
파라미터다. `optimizer.load_state_dict()`가 이 값을 덮어쓸 수 없으므로,
resume 시 새 config에 다른 값을 넣으면 그 새 값이 그대로, 정확하게
적용된다 -- silent override 위험이 없다. `epochs`/`early_stopping_patience`
가 `RESUME_CONFIG_FIELDS`에서 제외된 것과 같은 이유로, `gradient_clip_norm`
도 resume마다 자유롭게 바꿀 수 있도록 둔다. 이것을 명시적 public
contract로 문서화하고(이 절) 테스트로 고정한다(§10).

구체적으로 다음 두 방향 모두 허용된다:

```text
기존 checkpoint: clipping 없음(gradient_clip_norm=None)
resume config:  gradient_clip_norm=1.0
  → 허용, resume 구간부터 clipping 적용

기존 checkpoint: gradient_clip_norm=1.0
resume config:  gradient_clip_norm=0.5
  → 허용, resume 구간부터 새 max_norm 적용
```

**결론적으로 이번 Phase는 다음을 전부 수정하지 않는다**: `checkpoint.py`,
`RESUME_CONFIG_FIELDS`, `RESUME_CONFIG_LEGACY_DEFAULTS`, checkpoint format
version, `require_compatible_resume_config()`. Phase 4M 이전 checkpoint
migration rule도 필요 없다 --애초에 `RESUME_CONFIG_FIELDS`에 없는
필드라 "누락 시 거부"라는 문제 자체가 발생하지 않는다(Phase 4L이
`weight_decay`에서 겪었던 문제의 근본 원인을 구조적으로 피해간다).

`TrainingConfig`에 필드를 추가했으므로 `asdict(training_config)`를
그대로 저장하는 `save_training_checkpoint()`를 통해 `gradient_clip_norm`
값 자체는 checkpoint의 `training_config`에 자동으로 포함된다(관찰
목적으로는 남지만, resume 호환성 강제에는 쓰이지 않는다) -- 이 역시
`checkpoint.py` 무수정으로 얻어지는 부수 효과다.

---

## 8. Exact-resume / determinism 분석

- `torch.nn.utils.clip_grad_norm_()`는 이미 계산된 `.grad` 텐서에 대한
  순수 산술 연산(L2 norm 계산 후 필요하면 in-place scaling)이며,
  `torch.rand()`/`torch.randn()` 등 RNG를 소비하는 연산을 전혀 호출하지
  않는다.
- 따라서 동일한 `gradient_clip_norm` 값을 continuous 구간과 resume
  구간 양쪽에 동일하게 적용하면, 결정론적 연산이므로 tensor-level exact
  equality가 유지되어야 한다.
- 이 기대를 기존 exact-resume 테스트 패턴(`_dropout_mlp_classifier_spec()`
  + `_assert_deep_equal()` + CPU RNG/DataLoader generator 상태 캡처/복원)
  을 그대로 재사용해 `gradient_clip_norm != None` 조합에서 검증한다(§10).
- 만약 실제로 tensor-level exact equality가 깨진다면, tolerance를
  낮추는 방식으로 우회하지 않고 원인을 조사해 별도로 보고한다.

---

## 9. CLI 설계

| 항목 | 내용 |
|---|---|
| 이름 | `--gradient-clip-norm` |
| type | `float` |
| choices | 없음(자유 float) |
| default | `None`(생략 시 clipping 비활성화) |
| 의미 | gradient L2 norm clipping 최대값. 생략하면 clipping 비활성화. |
| backward compatibility | 생략하면 기존 명령어와 완전히 동일하게 동작 |

argparse에서는 복잡한 validation을 하지 않는다(`type=float`만 지정) --
실제 validation은 전부 `TrainingConfig.__post_init__()`이 담당한다.
흐름은 기존과 동일하게 `argparse → TrainingConfig(gradient_clip_norm=
args.gradient_clip_norm) → ImageFolderWorkflowRequest → run_training()
→ train_one_epoch()`이며, `imagefolder_workflow.py`는 `training_config`
를 불투명하게 전달하므로 무수정이다.

---

## 10. 테스트 및 검증 결과

기존 전체 테스트 467개에 Phase 4M 신규 테스트 24개를 더해 `pytest -q`
전체 **491 passed**(신규 실패/skip 없음)를 확인했다.

### 10-1. `tests/training/test_config.py`

- 기본값 `None`
- `None` 허용
- 작은 양수(`1e-12`)/일반 양수(`0.1`, `1.0`)/`1.0` 초과(`5.0`, `1000.0`,
  상한 없음 확인) 허용
- `0.0` 거부, 음수 거부, bool 거부, `NaN`/`+inf`/`-inf` 거부
- **실제 결과**: 위 항목 전부 테스트로 작성해 통과를 확인했다. 추가로
  `gradient_clip_norm`이 `RESUME_CONFIG_FIELDS`에 없다는 사실 자체를
  고정하는 테스트, 그리고 `require_compatible_resume_config()`가
  checkpoint/resume config의 `gradient_clip_norm`이 서로 달라도(양방향,
  `None`↔값 포함) 거부하지 않는다는 계약을 3가지 조합으로 고정하는
  테스트를 추가했다.

### 10-2. `tests/training/test_loop.py` -- 실제 수치 효과

mock으로 "호출됐는지"만 확인하지 않고, 실제 gradient가 clip되는지를
검증한다. `train_one_epoch()`는 각 batch가 끝나면 `optimizer.step()`을
호출하지만 `.grad`는 다음 `zero_grad()` 전까지 파라미터에 남아있으므로,
loader에 batch가 정확히 1개뿐인 DataLoader를 쓰면 함수가 반환된 시점의
`model.parameters()[...].grad`가 "마지막(=유일한) batch의 step 직전
gradient"와 동일하다 -- 이 사실을 이용해 production API를 변경하지
않고도 실제 clip 효과를 검증한다.

- **clipping 활성화**: 매우 작은 `gradient_clip_norm`을 주고
  `train_one_epoch()` 실행 후 `model.parameters()`의 gradient 전체
  L2 norm이 `<= max_norm`(부동소수 오차 허용)임을 직접 계산해 확인.
  대조군으로 clipping 없이 실행했을 때의 norm이 그보다 훨씬 큼을 함께
  확인(그래야 "원래도 작아서 우연히 통과"가 아님을 증명).
- **clipping 비활성화**: `gradient_clip_norm=None`이면 clip 호출 자체가
  없어 기존 `train_one_epoch()` 반환값(loss)이 완전히 동일함을 회귀로
  고정.
- **실제 결과**: `scale=1000.0`으로 부풀린 입력 + `max_norm=1e-3`
  조합에서 clip 후 실제 gradient L2 norm이 `1e-3 + 1e-6`(부동소수
  오차) 이하임을 확인했고, 동일 입력을 clipping 없이 실행하면 norm이
  `1e-3`보다 훨씬 큼을 대조군으로 확인했다. `gradient_clip_norm` 인자를
  생략한 호출과 명시적으로 `None`을 넘긴 호출이 완전히 동일한 loss/
  model state를 만듦도 회귀로 고정했다. `run_training()` 내부의
  `train_one_epoch()` 호출을 monkeypatch하던 기존 테스트 2개(`test_loop.py`)
  + `test_checkpoint.py` 1개의 fake 함수 시그니처에 `gradient_clip_norm=None`
  파라미터를 추가해야 했다(새 keyword 인자를 `run_training()`이 항상
  전달하므로) -- 프로덕션 로직 변경은 아니고, 시그니처가 실제로 늘어난
  데 따른 필수적인 테스트 더블 갱신이다.

### 10-3. Resume 정책 테스트 -- "자유롭게 변경 가능"이 public contract

이번 Phase는 checkpoint serialization/load 구조를 전혀 바꾸지 않으므로,
실제 `.pt` checkpoint 파일 save/load를 거치는 새 테스트를 추가하지
않는다. 대신 다음 두 계층에서 각각 검증한다:

- **`require_compatible_resume_config()` 자체의 비교 계약**
  (`tests/training/test_config.py`): `weight_decay` 없는 checkpoint를
  다룰 때와 같은 패턴으로, `training_config` dict를 직접 만들어
  `gradient_clip_norm` 값이 checkpoint config와 resume config 사이에
  달라도(`None`↔값 양방향 포함) 거부되지 않음을 확인한다.
  `require_compatible_resume_config()` 자체는 수정하지 않는다.
- **`run_training()`의 실제 resume 흐름에서 새 값이 적용되는지**
  (`tests/training/test_loop.py`): 아래 순서로, checkpoint 파일이
  아니라 `run_training()`이 반환한 결과로부터 직접 조립한
  `TrainingResumeState`를 사용한다.

  ```text
  fresh run (gradient_clip_norm=None)
    → run_training()의 반환값으로 resume_state 구성 (_make_resume_state)
    → gradient_clip_norm=0.5로 바꾼 새 TrainingConfig로
      run_training(..., resume_state=resume_state) 호출
  ```

  이 흐름을 "production resume 흐름(checkpoint 파일이 아니라
  `run_training()`이 실제로 지원하는 in-memory resume 경로)"이라고는
  부를 수 있지만, checkpoint 파일 I/O를 통과하는 integration test는
  아니다.

**실제 결과**: `test_config.py`에서 `(saved, resume)` =
`(None, 0.5)`/`(1.0, 0.5)`/`(1.0, None)` 세 조합 모두
`require_compatible_resume_config()`가 거부하지 않음을 확인했다.
`test_loop.py`에서는 위 in-memory resume 흐름 위에
`torch.nn.utils.clip_grad_norm_`을 spy로 감싸, resume 구간에서 새
`gradient_clip_norm` 값(`None` → `0.5`)이 실제로 `max_norm` 인자로
전달됨을 확인했다(gradient norm을 직접 재계산하는 대신, production
API를 바꾸지 않고 실제 적용 여부를 증명하는 방식을 택했다).

### 10-4. Exact-resume 회귀

`gradient_clip_norm != None`(동일 값, continuous/resume 양쪽) 조합에서
기존 `test_run_training_resume_matches_continuous_run_exactly` 패턴을
재사용해 model parameter/optimizer state/scheduler state/history/best
state가 tensor-level로 정확히 일치함을 고정한다.

**실제 결과**: `optimizer="sgd"` + `gradient_clip_norm=0.5` +
`lr_scheduler="plateau"` 조합으로 continuous 5 epoch과 3+2 split/resume을
비교해, model parameter/optimizer state/scheduler state/history/best
state 전부 tensor-level exact equality가 그대로 유지됨을 확인했다 --
`clip_grad_norm_()`가 RNG를 소비하지 않는다는 §8의 분석이 실측으로도
확인됐다.

### 10-5. CLI wiring

`tests/scripts/test_train_imagefolder_cli.py`에 기존 request 캡처
패턴을 재사용해 `--gradient-clip-norm 1.5` → `TrainingConfig.
gradient_clip_norm == 1.5`, 플래그 생략 → `None` 두 가지만 검증한다
(과도한 CLI 테스트 추가 지양). **실제 결과**: 두 테스트 모두 통과했다.

---

## 11. E2E/회귀 검증 결과

새 E2E 스크립트는 추가하지 않았다. 기존 4개 E2E(`run_training_e2e.py`/
`run_real_training_e2e.py`/`run_resume_training_e2e.py`/
`run_imagefolder_training_e2e.py`)는 전부 `gradient_clip_norm` 미지정
(기본값 `None`) 조합만 쓰므로, 재실행으로 기존 수치 anchor가 완전히
동일하게 유지되는지, resume/TorchScript export/C++ CPU·CUDA parity가
전부 PASS인지 확인하는 것으로 충분했다. clipping의 실제 계산 효과와
exact-resume은 §10의 unit/integration 테스트가 담당했다.

**실제 결과**: 4개 전부 재실행해 기존 수치 anchor가 완전히 동일하게
유지된 채 PASS했다(예: `run_training_e2e.py`의 epoch 1 train_loss=1.3386
→ epoch 10 train_loss=0.2867, `run_resume_training_e2e.py`의 continuous
vs split+resume 비교 전부 PASS). `run_imagefolder_training_e2e.py`의
TorchScript export/C++(LibTorch) CPU·CUDA parity도 기존과 동일하게
PASS했다.

---

## 12. 위험 요소

- 기존 default 동작 변경 위험: 낮음(`None` 기본값).
- old checkpoint compatibility: 위험 없음(§7, `RESUME_CONFIG_FIELDS`
  미포함이라 Phase 4L에서 겪은 누락-필드 문제 자체가 발생하지 않음).
- silent wiring bug: CLI → TrainingConfig → train_one_epoch 경로가
  짧아 위험 낮음, CLI wiring 테스트로 방어.
- exact-resume 깨짐: 이론상 위험 낮음(결정론적 연산), §10-4 회귀
  테스트로 실측 검증.
- RNG consumption 변화: 없음(`clip_grad_norm_`는 RNG 미사용).
- `_require_positive_float()`의 NaN/+inf 미검출: 기존 코드의 잠재적
  결함이지만 이번 Phase 범위 밖이라 수정하지 않음(§5-2) -- 별도 향후
  검토 후보로 남긴다.
- CLI 복잡도 증가: 플래그 1개, 낮음.
- Phase 범위가 커지는 문제: value clipping/`norm_type` 노출을 함께
  넣지 않도록 주의(§2 non-goals에 명시).

---

## 13. Acceptance Criteria

- [x] `TrainingConfig.gradient_clip_norm` 기본값 `None`이 기존 동작을
      완전히 재현한다 -- 인자 생략/명시적 `None`이 동일 결과를 냄을 확인.
- [x] `gradient_clip_norm`이 설정되면 실제 학습 중 gradient의 L2 norm이
      `max_norm` 이하로 clip된다 -- 수치 계산으로 직접 검증, 대조군 포함.
- [x] `gradient_clip_norm=0`, 음수, bool, NaN, +inf, -inf는 거부된다 --
      `_require_positive_finite_float()`로 검증 완료.
- [x] `--gradient-clip-norm`이 `TrainingConfig`와 workflow request까지
      정확히 배선된다 -- CLI wiring 테스트로 검증 완료.
- [x] `gradient_clip_norm`은 `RESUME_CONFIG_FIELDS`에 포함되지 않으며,
      resume 시 자유롭게 다른 값으로 바꿀 수 있다 -- 3가지 조합
      compatibility 테스트 + 실제 resume 시 새 값이 적용됨을 spy로
      검증 완료.
- [x] `gradient_clip_norm != None` 조합에서도 continuous run과 resume
      run이 tensor-level exact equality를 유지한다 -- 회귀 테스트로
      검증 완료.
- [x] `checkpoint.py`/`imagefolder_workflow.py`가 무수정임을
      `git diff --stat`으로 확인했다.
- [x] 전체 pytest 통과 -- 기존 467개 + 신규 24개 = **491 passed**.
- [x] 기존 4개 E2E anchor가 깨지지 않는다 -- 4개 전부 재실행해 PASS,
      TorchScript/C++ CPU·CUDA parity도 기존과 동일하게 PASS.
- [x] README에 Phase 4M 절이 반영된다 -- 반영 완료.

---

## 14. 구현 순서

1. [x] `config.py`: `_require_positive_finite_float()` 추가, `TrainingConfig.
   gradient_clip_norm` 필드 + `__post_init__()` 검증 추가.
2. [x] `test_config.py`: §10-1 검증 테스트 작성/통과.
3. [x] `loop.py`: `train_one_epoch()`에 `gradient_clip_norm` 파라미터 +
   clip 호출 추가, `run_training()` 호출부 수정.
4. [x] `test_loop.py`: §10-2(실제 수치 효과)/§10-3(resume 자유 변경)/
   §10-4(exact-resume) 테스트 작성/통과 -- 기존 `train_one_epoch`
   monkeypatch fake 3곳의 시그니처 업데이트 포함(§10-2 참고).
5. [x] `scripts/train_imagefolder.py`: `--gradient-clip-norm` 플래그 +
   `TrainingConfig` 조립부 배선.
6. [x] `test_train_imagefolder_cli.py`: §10-5 CLI wiring 테스트 작성/통과.
7. [x] 전체 pytest 재실행(491 passed).
8. [x] 기존 4개 E2E 재실행(anchor 불변 확인, 전부 PASS).
9. [x] `docs/phase4m_gradient_clipping_design.md`/README 최종 상태로 갱신.

위 9단계 전부 이 순서 그대로 완료됐다 -- 계획과 실제 구현 순서 사이에
차이는 없었다. 유일하게 계획 대비 추가로 필요했던 작업은 4단계에서
발견된 기존 monkeypatch fake 시그니처 갱신(§10-2)이며, 이는 새 Phase의
로직 변경이 아니라 `train_one_epoch()`의 새 keyword 인자를 기존
test double이 받아들이도록 하는 필수적인 부수 작업이었다.

---

## 15. 향후 확장과의 연결

- **gradient value clipping**: 동일한 `train_one_epoch()`의 같은
  지점에 `torch.nn.utils.clip_grad_value_()`를 추가하는 형태로,
  norm clipping과 나란히(또는 상호 배타적으로) 확장 가능한 별도
  소규모 Phase 후보.
- **gradient norm 관측/로깅**: clip 전/후 norm을 `TrainingProgress`에
  노출하는 확장은 Phase 4I의 progress callback 계약을 다시 열어야
  하므로 별도 검토가 필요.
- **loss function 확장(label smoothing)**: `criterion` 생성이
  `train_one_epoch()`/`evaluate()`에 중복 하드코딩된 지점을 그대로
  물려받는 다음 후보.
