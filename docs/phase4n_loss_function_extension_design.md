# Phase 4N: CrossEntropy Label Smoothing — 설계안

**상태: 구현 완료, 전체 단위/통합 테스트 및 기존 E2E 회귀 검증 완료.**
`TrainingConfig`에 `label_smoothing`을 추가하고, `loop.py`에
`_build_optimizer()`/`_build_scheduler()`와 나란한 `_build_criterion()`
을 신설해 `train_one_epoch()`에 주입했다. label smoothing은 **training
loss에만** 적용했고, validation/test loss와 scheduler/early
stopping/best-model-selection 의미는 기존 unsmoothed
`nn.CrossEntropyLoss()` 그대로 유지했다(`evaluate()` 무수정).
전체 pytest **535 passed**(기존 506 + 신규 29), 기존 4개 E2E anchor
불변을 실측 확인했다(§10-1). 커밋 전 마지막 검토 라운드에서
`run_training() → train_one_epoch()` criterion 전달 계약을 직접
고정하는 테스트 1개를 추가했고(§10-1), PyTorch의 실제 forward-level
검증 범위(§3-1)와 `CrossEntropyLoss.state_dict()`(§4/§7) 등 문서
표현을 실측 결과에 맞춰 정밀화했다 -- production 실행 로직은
변경하지 않았다(주석/docstring만 수정).

**전제**: 이 문서는 별도 조사/설계 라운드(Phase 4M 이후 다음 Phase
후보 비교, `evaluate()` monkeypatch 약 20곳이라는 실측 사실을 근거로
"training-only smoothing" 정책을 확정한 라운드)에서 결정된 정책을
실제 구현에 맞춰 정리한 것이다. 정책 자체의 재검토는 이 문서의
목적이 아니다.

---

## 1. 목적

- **해결하려는 문제**: 현재 loss는 `nn.CrossEntropyLoss()`로 완전히
  고정되어 있고, overconfidence를 완화하는 label smoothing 같은 흔한
  정규화 옵션이 전혀 없다.
- **새로 지원할 기능**: `TrainingConfig.label_smoothing: float = 0.0`
  으로 `nn.CrossEntropyLoss(label_smoothing=...)`를 켜고 끈다.
- **기본 동작**: `0.0`(기본값)이면 기존 `nn.CrossEntropyLoss()`와
  bitwise 동일한 결과를 낸다(직접 실측 확인, §3).
- **사용자 입장에서 보이는 변화**: `scripts/train_imagefolder.py`에
  `--label-smoothing FLOAT` 플래그가 새로 생긴다.
- **non-goals**: BCE/BCEWithLogitsLoss, multilabel, focal loss, class
  weight, custom loss plugin, regression loss, `loss: str` 선택
  필드, reduction/ignore_index 변경, validation/test smoothing,
  scheduler/metric/GPU 관련 변경.

---

## 2. 기존 loss 구조 (재확인)

`train_one_epoch()`와 `evaluate()`(둘 다 `loop.py`)는 각각 독립적으로
`criterion = nn.CrossEntropyLoss()`를 함수 최상단에서 생성한다.
`run_training()`은 criterion을 전혀 모르는 채로 두 함수를 호출한다.
`imagefolder_workflow.py`의 최종 `test_loss` 계산도 `evaluate()`를
그대로 재사용한다 -- 즉 validation loss와 test loss는 이미 동일한
코드 경로를 공유한다.

**설계에 결정적인 실측 사실**: `tests/training/test_loop.py`/
`test_checkpoint.py`를 grep한 결과, `train_one_epoch()`를
monkeypatch하는 곳은 3곳(Phase 4M에서 `gradient_clip_norm=None`을
받도록 이미 갱신됨)인 반면, `evaluate()`를 monkeypatch하는 lambda는
**20곳**(`lambda model, loader, device="cpu": ...` 형태, `**kwargs`
없음)이다. `evaluate()`에 새 keyword를 추가하면 이 20곳이 전부
깨진다. 이 사실이 "training-only smoothing" 정책(§6)의 핵심 근거다.

TorchScript/C++ parity는 inference output(forward pass)만 비교하고
loss 계산과는 무관함을 `test_train_export_parity.py`/
`test_torchscript_integration.py`로 재확인했다 -- label smoothing은
export/추론 그래프에 전혀 나타나지 않는다.

---

## 3. `label_smoothing` config 계약

### 3-1. PyTorch 실제 동작 (로컬에서 constructor + forward + backward까지 직접 확인)

**constructor는 전혀 검증하지 않는다** -- 아래 다섯 값 전부 생성
자체는 성공한다:

```python
torch.nn.CrossEntropyLoss(label_smoothing=-0.1)          # 생성 성공
torch.nn.CrossEntropyLoss(label_smoothing=1.1)            # 생성 성공
torch.nn.CrossEntropyLoss(label_smoothing=float("nan"))   # 생성 성공
torch.nn.CrossEntropyLoss(label_smoothing=float("inf"))   # 생성 성공
torch.nn.CrossEntropyLoss(label_smoothing=float("-inf"))  # 생성 성공
```

**하지만 forward 시점에는 부분적인 검증이 있다** -- 실제 logits/target
으로 forward/backward까지 직접 실행해 확인한 결과:

| 값 | constructor | forward | backward |
|---|---|---|---|
| `-0.1` | 성공 | 성공, loss=1.392834(=unsmoothed와 동일값) | 성공 |
| `1.1` | 성공 | **`RuntimeError: label_smoothing must be between 0.0 and 1.0. Got: 1.1`** | 해당 없음 |
| `NaN` | 성공 | 성공, loss=1.392834(=unsmoothed와 동일값) | 성공 |
| `+inf` | 성공 | **`RuntimeError: label_smoothing must be between 0.0 and 1.0. Got: inf`** | 해당 없음 |
| `-inf` | 성공 | 성공, loss=1.392834(=unsmoothed와 동일값) | 성공 |

관찰된 패턴은 PyTorch의 forward 구현이 `label_smoothing > 0`일 때만
내부 범위 검사를 활성화하는 것과 일치한다 -- `-0.1`/`NaN`/`-inf`는
`label_smoothing > 0` 비교 자체가 `False`가 되어(NaN과의 비교는 항상
`False`) 이 검사를 아예 거치지 않고, **예외 없이 마치
`label_smoothing=0.0`인 것처럼(스무딩 없이) 조용히 동작한다** -- 즉
음수/`NaN`/`-inf`를 지정해도 사용자가 의도한 효과가 조용히 사라진다.
반면 `1.1`/`+inf`는 `label_smoothing > 0`이 `True`라 검사가
활성화되고, 그 안에서 `<= 1.0` 위반이 적발되어 `RuntimeError`로
거부된다.

**`TrainingConfig` validation의 실질적 가치는 다음 두 가지다**:
(1) PyTorch가 forward에서야 뒤늦게(그것도 우리 도메인과 무관한 일반
`RuntimeError`로) 거부하는 `>1.0`/`+inf` 케이스를 학습 시작 전, 일관된
`TrainingConfigError`로 조기에 차단하고, (2) PyTorch가 아예 조용히
무시해버리는 음수/`NaN`/`-inf` 케이스도 명확한 오류로 전환한다 --
optimizer/loss backend별로 검증 시점과 방식이 제각각인 것에 의존하지
않고, config 단계에서 항상 같은 방식으로 걸러낸다는 것이 핵심이다.

`label_smoothing=1.0`(정상 상한)도 직접 forward/backward를 실행해
수치적으로 완전히 유효함(NaN 없음, gradient 정상)을 확인했다.
`label_smoothing=0.0`과 kwarg를 아예 생략한 기본 `CrossEntropyLoss()`
가 `torch.equal()` 수준으로 **bitwise 동일**함도 확인했다 -- 기본값이
기존 동작을 완전히 재현한다는 배경이다.

### 3-2. 검증 정책(확정)

```text
bool          → 거부
음수          → 거부
0.0           → 허용
0.1           → 허용
1.0           → 허용
> 1.0         → 거부
NaN/+inf/-inf → 거부
```

`[0.0, 1.0]` 양끝 포함.

### 3-3. 기존 helper 재사용 불가 확인

`_require_fraction(name, value, *, low_inclusive)`의 실제 구현:

```python
lower_ok = value >= 0.0 if low_inclusive else value > 0.0
if not (lower_ok and value < 1.0):
    ...
```

상한이 `low_inclusive`와 무관하게 **항상 `< 1.0`(배타적)**으로
고정되어 있어 `label_smoothing=1.0`을 거부한다 -- 재사용 불가.
`_require_positive_finite_float`/`_require_non_negative_finite_float`
는 상한 자체가 없어 부적합. 어느 기존 helper도 `[0.0, 1.0]` 양끝
포함 계약을 갖지 않으므로 새 helper가 필요하다:

```python
def _require_closed_unit_interval(name: str, value: object) -> None:
    """[0.0, 1.0] 양끝 포함 (label_smoothing). 상/하한 비교 자체가
    NaN을 걸러내므로(NaN과의 모든 비교는 False) math.isfinite()가
    따로 필요 없다 -- +inf/-inf도 각각 상한/하한 비교에서 자연히
    거부된다."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 <= float(value) <= 1.0):
        raise TrainingConfigError(f"'{name}' must be in [0.0, 1.0], got {value!r}")
```

---

## 4. `_build_criterion()` 구조

```python
def _build_criterion(config: TrainingConfig) -> nn.Module:
    """config.label_smoothing으로 CrossEntropyLoss를 생성. 선택지가
    CrossEntropy 하나뿐이라 registry/loss-name 필드는 두지 않는다
    (_build_optimizer()/_build_scheduler()와 동일한 근거)."""
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
```

`_build_optimizer()`/`_build_scheduler()`와 나란한 위치에 둔다.
`loss: str = "cross_entropy"` 같은 필드는 만들지 않는다 -- 선택지가
하나뿐인 상태에서 그런 필드는 검증 표면만 늘릴 뿐 실질적 유연성을
주지 못하는 과설계다(module docstring이 이미 이 근거를 `_build_scheduler`
에 대해 명시하고 있다).

이번 Phase의 `CrossEntropyLoss(weight=None)`는 `nn.Module`이므로
`state_dict()` 메서드 자체는 존재하지만, 실제로 직접 확인한 결과
`criterion.state_dict()`가 빈 `OrderedDict()`다(`weight=None`이라
등록된 파라미터/버퍼가 전혀 없음 -- `named_parameters()`/
`named_buffers()` 둘 다 빈 리스트). `label_smoothing`은 Python scalar
설정값으로 매번 `TrainingConfig`에서 다시 만들어지므로, 이 criterion을
checkpoint에 별도로 저장/복원할 필요가 없다(`optimizer`/`scheduler`
와 다른 점 -- 이들은 실제로 비어있지 않은 `state_dict()`를 갖고
checkpoint에 저장/복원된다).

---

## 5. `train_one_epoch()` API 변경

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
    criterion: nn.Module | None = None,
) -> float:
    ...
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    ...
```

`criterion`을 시그니처 맨 끝에 기본값 `None`으로 추가한다 -- 기존
직접 호출(`train_one_epoch(model, loader, optimizer)`,
`train_one_epoch(model, loader, optimizer, device=device)`)은 전혀
깨지지 않는다. `run_training()`은 `criterion = _build_criterion(config)`
를 **epoch 루프 진입 전 한 번만** 생성해 매 epoch `train_one_epoch()`
호출에 그대로 전달한다(optimizer/scheduler와 같은 생명주기).

**`evaluate()`는 시그니처와 내부 로직을 전혀 수정하지 않는다** -- 계속
`nn.CrossEntropyLoss()`(unsmoothed)를 내부에서 생성한다.

---

## 6. training-only smoothing 정책

```text
train_loss = smoothed CrossEntropyLoss  (label_smoothing 적용)
val_loss   = ordinary CrossEntropyLoss  (항상 unsmoothed)
test_loss  = ordinary CrossEntropyLoss  (항상 unsmoothed, evaluate() 재사용)
```

이 정책을 택한 이유, 우선 의미적 근거부터:

1. label smoothing은 원래 "training 시 overconfidence를 줄이는
   정규화 기법"으로 설계된 것이지 평가 지표를 바꾸기 위한 것이
   아니다 -- training objective와 validation/test metric을 분리하는
   것이 표준적인 실무 관행과 일치한다.
2. val_loss/test_loss를 항상 ordinary(unsmoothed) CrossEntropyLoss로
   고정하면, 서로 다른 `label_smoothing` 설정으로 학습한 실행들 사이에서도
   validation/test 지표가 안정적인(stable) 공통 기준으로 유지된다.
3. Phase 4E의 `ReduceLROnPlateau`(val_loss 기반), early stopping
   (`epochs_without_improvement`, val_loss 기반), Phase 4B의 best
   model selection(`val_loss < best_val_loss`) 세 계약의 의미가
   label smoothing과 무관하게 그대로 보존된다.

여기에 더해, 부수적이지만 실무적으로 중요한 호환성 이점도 있다(§2의
call graph 조사에 근거):

4. `evaluate()`가 val_loss/test_loss 양쪽에 재사용되므로, 이 정책은
   `evaluate()`의 API를 전혀 바꾸지 않고도 성립한다 -- 그 결과 이
   함수를 monkeypatch하는 기존 테스트 약 20곳도 전혀 영향받지 않는다.

위 3번의 세 계약(`ReduceLROnPlateau`/early stopping/best model
selection)은 이 정책 덕분에 해당 production 로직을 전혀 변경할
필요가 없었다 -- 이 사실은 기존 pytest 전체와 4개 E2E를 재실행해
회귀가 없음을 실측으로 확인했다(§10-1/§11).

**주의(사용자에게 명확히 알려야 하는 부분)**: `label_smoothing > 0`
일 때 `history.train_losses`와 `history.val_losses`는 **서로 다른
objective의 숫자**다 -- 직접 비교(예: "train_loss가 val_loss보다
낮으니 overfitting")가 무의미해질 수 있다. 이것은 의도된 trade-off이며
버그가 아니다.

---

## 7. Resume 정책(확정: 자유 변경)

`label_smoothing`은 **`RESUME_CONFIG_FIELDS`에 추가하지 않는다.**

이유(mechanical override 기준): `RESUME_CONFIG_FIELDS`가 존재하는
이유는 `optimizer.load_state_dict()`/`scheduler.load_state_dict()`가
저장된 `param_groups` 값을 그대로 복원해 새 config 값을 조용히
덮어쓰기 때문이다(`config.py`에 이미 문서화된, 좁게 한정된 근거).
`label_smoothing`은 어떤 `*.load_state_dict()`에도 관여하지 않는
순수 criterion 생성자 인자이고, 이번 Phase가 쓰는
`CrossEntropyLoss(weight=None)`에는 checkpoint로 저장·복원해야 할
tensor state가 없다(`state_dict()`가 빈 `OrderedDict()`임을 직접
확인, §4) -- silent override 위험이 구조적으로 존재하지 않는다
(`gradient_clip_norm`과 동일한 논리, Phase 4M).

```text
fresh run:  label_smoothing=0.0
resume:     label_smoothing=0.1
  → 허용, resume 구간부터 새 값 적용
```

**`checkpoint.py`, `RESUME_CONFIG_LEGACY_DEFAULTS`, checkpoint format
version은 전혀 수정하지 않는다** -- `RESUME_CONFIG_FIELDS`에 없는
필드라 "누락 시 거부"라는 Phase 4L식 문제 자체가 발생하지 않는다.
`TrainingConfig`에 필드를 추가했으므로 `asdict(training_config)`를
통해 `label_smoothing` 값 자체는 checkpoint의 `training_config`에
**저장은 되지만**, `RESUME_CONFIG_FIELDS` 밖이라 resume compatibility
비교 대상은 아니다(저장되지 않는다는 표현은 부정확하므로 쓰지 않는다).

---

## 8. Exact-resume / determinism

로컬에서 직접 확인:

```text
동일 logits/targets로 label_smoothing=0.1을 두 번 호출 → loss 완전히 동일(torch.equal)
label_smoothing 적용 전후 RNG state 완전히 동일(torch.get_rng_state() 불변)
```

`label_smoothing`은 RNG를 전혀 소비하지 않는 순수 결정론적 수식(one-hot과
uniform 분포를 정적 가중치로 섞는 계산)이다. 따라서 **continuous run과
resume run이 동일한 `label_smoothing` 값을 쓰면** 기존 exact-resume
계약(model parameter/optimizer state/scheduler state/history/best
state 전부 tensor-level exact equality) 이 그대로 유지되어야 한다.
이 계약은 "동일한 값을 쓸 때만" 성립하며, resume 도중 값을 바꾸면
그 시점부터 학습 궤적 자체가 달라지는 것이 당연하다(의도된 동작).

---

## 9. CLI

| 항목 | 내용 |
|---|---|
| 이름 | `--label-smoothing` |
| type | `float` |
| default | `0.0` |
| 의미 | label smoothing 계수 (CrossEntropyLoss, `[0.0, 1.0]`, 기본값 0.0 = 미적용) |
| backward compatibility | 생략하면 기존 명령어와 완전히 동일하게 동작 |

argparse는 `type=float`만 지정하고, 실제 validation은
`TrainingConfig.__post_init__()`이 전담한다(기존 컨벤션과 동일).
흐름: `argparse → TrainingConfig(label_smoothing=...) →
ImageFolderWorkflowRequest → run_training() → _build_criterion() →
train_one_epoch()`. `imagefolder_workflow.py`는 `training_config`를
불투명하게 전달하므로 무수정이다.

---

## 10. 테스트 전략

- **config**: 기본값 0.0, `[0.0, 1.0]` 정상 범위(경계값 포함), bool/음수/
  `>1.0`/NaN/+inf/-inf 거부, `label_smoothing`이 `RESUME_CONFIG_FIELDS`에
  없음, `require_compatible_resume_config()`가 값이 달라도 거부하지
  않음(저장은 되지만 비교 대상이 아니라는 계약을 정확한 표현으로 테스트).
- **criterion factory**: 기본/0.1 각각에서 `nn.CrossEntropyLoss` 타입과
  `label_smoothing` 속성값 확인.
- **numerical behavior**: `label_smoothing=0.0`이 `nn.CrossEntropyLoss()`
  와 `torch.equal()` 수준으로 동일, `label_smoothing=0.1`이 PyTorch
  reference(`nn.CrossEntropyLoss(label_smoothing=0.1)`)와 정확히 일치
  (하드코딩된 magic number가 아니라 reference 구현과 직접 비교).
- **train_one_epoch**: `criterion=None`이 기존 동작 유지, custom
  criterion이 실제로 쓰임, gradient_clip_norm과 동시에 있어도 정상 동작.
- **run_training → train_one_epoch 연결 계약**: `_build_criterion(config)`
  가 만든 바로 그 criterion 객체가 실제로 `train_one_epoch()`에
  전달되는지 `train_one_epoch()` 자체를 monkeypatch해 직접 확인한다 --
  criterion factory가 올바르게 동작한다는 것과 train_one_epoch()가
  받은 criterion을 쓴다는 것은 각각 검증되지만, 그 사이를 잇는
  `train_one_epoch(..., criterion=criterion)` 배선 자체가 실수로
  빠지는 회귀는 이 둘만으로는 잡히지 않는다.
- **evaluate 정책 회귀**: label_smoothing>0인 config로 학습해도
  `run_training()`이 history에 기록하는 val_loss/val_accuracy가
  (무수정) `evaluate()`를 그대로 호출한 결과와 정확히 일치함을
  확인한다 -- 이 테스트는 "run_training()의 validation 경로가
  무수정 evaluate()를 그대로 쓴다"는 배선을 증명하는 것이지,
  `evaluate()` 자체가 unsmoothed CE를 쓴다는 계약을 독립적으로 다시
  증명하는 것은 아니다(그 계약은 무수정 production 코드와 기존
  evaluate 테스트가 담당한다).
- **resume**: 다른 `label_smoothing` 값으로 resume이 거부되지 않음
  + `run_training()`의 resume 흐름에서 새 값으로 만든 criterion이
  실제로 쓰이는지 확인(Phase 4M의 `gradient_clip_norm` resume 테스트와
  유사한 구조).
- **exact-resume**: `label_smoothing > 0`(continuous vs split+resume,
  동일 값)에서 tensor-level exact equality.
- **ImageFolder workflow integration**: 새 E2E 스크립트 대신, 기존
  `test_imagefolder_workflow.py`에 가벼운 integration 테스트 1개로
  production wiring 연결을 확인.
- **CLI**: 생략 시 0.0, `--label-smoothing 0.1` 전달 확인, 잘못된 값이
  `TrainingConfigError` 경로에서 exit code 1로 정확히 거부되는지 확인.

### 10-1. 실제 결과

기존 전체 테스트 506개에 Phase 4N 신규 테스트 29개를 더해 `pytest -q`
전체 **535 passed**(신규 실패/skip 없음)를 확인했다. 내역: `test_config.py`
+15(validation 11 + resume 자유변경 3 + `RESUME_CONFIG_FIELDS` 미포함
확인 1), `test_loop.py` +10(criterion factory 3 + train_one_epoch 3 +
run_training→train_one_epoch 연결 계약 1 + evaluate 정책 회귀 1 +
resume spy 1 + exact-resume 1), `test_imagefolder_workflow.py`
+1(production wiring integration), `test_train_imagefolder_cli.py` +3
(정상 전달/기본값/invalid 값 거부). 기존 `train_one_epoch()` monkeypatch
fake 3곳(`test_loop.py` 2곳, `test_checkpoint.py` 1곳)에
`criterion=None` 파라미터를 추가해야 했다(Phase 4M 때
`gradient_clip_norm=None`을 추가했던 것과 동일한 종류의 필수 시그니처
갱신 -- production 로직 변경 아님). `evaluate()`를 monkeypatch하는
기존 20곳은 전혀 건드리지 않았다(설계대로).

---

## 11. 기존 E2E 회귀

새 E2E 스크립트를 추가하지 않는다. 기존 4개 E2E(`run_training_e2e.py`/
`run_real_training_e2e.py`/`run_resume_training_e2e.py`/
`run_imagefolder_training_e2e.py`)는 전부 `label_smoothing` 미지정
(기본값 `0.0`) 조합만 쓰므로, 재실행으로 기존 수치 anchor가 완전히
동일하게 유지되는지 확인하는 것으로 충분하다. TorchScript/C++ parity는
§2의 근거로 label smoothing과 무관하지만, 기존 E2E 안에 이미 포함된
parity 검증은 자연스럽게 함께 재실행된다.

**실제 결과**: 4개 전부 재실행해 기존 수치 anchor가 완전히 동일하게
유지된 채 PASS했다(예: `run_training_e2e.py`의 epoch 1 train_loss=1.3386
→ epoch 10 train_loss=0.2867, `run_resume_training_e2e.py`의 continuous
vs split+resume 비교 전부 PASS). `run_imagefolder_training_e2e.py`의
TorchScript export/C++(LibTorch) CPU·CUDA parity도 기존과 동일하게
PASS했다.

---

## 12. 제외 범위 (재확인)

BCE/BCEWithLogitsLoss, multilabel classification, output/target
representation 변경, focal loss, class weights, per-class weights
CLI, custom loss plugin, regression loss, 임의 이름 선택, reduction
변경, ignore_index 변경, validation/test smoothing, scheduler 추가,
metric 추가, GPU/device 관련 변경.

---

## 13. 향후 확장 후보

- **class weights**: `_build_criterion()`이 확장점으로 이미 존재하므로,
  `nn.CrossEntropyLoss(weight=...)`를 추가하는 형태로 자연스럽게
  확장 가능한 다음 후보.
- **validation/test 시에도 동일 objective로 평가하고 싶은 사용자를
  위한 옵션**: 현재는 항상 unsmoothed로 고정했지만, 필요성이 실제로
  확인되면 `evaluate()`에 opt-in criterion 파라미터를 추가하는 별도
  Phase로 검토 가능(이번 Phase에서는 20곳 monkeypatch 비용 대비
  가치가 낮다고 판단해 보류).
- **Evaluation Metric Extension**: per-class accuracy/confusion
  matrix 등 Phase 4N 조사 라운드에서 함께 검토됐던 후보.
