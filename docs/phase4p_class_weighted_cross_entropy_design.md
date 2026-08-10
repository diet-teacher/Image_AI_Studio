# Phase 4P: Explicit Class-weighted CrossEntropy — 설계안

## 1. 목적

Phase 4O가 제공하는 confusion matrix/per-class recall로 관찰한 클래스
불균형/특정 클래스 성능 저하를, 사용자가 명시적으로 지정한 class별
weight로 직접 개선할 수 있게 한다. **explicit weight만** 지원한다 --
automatic(inverse-frequency 등) 계산과 `WeightedRandomSampler`는 이번
Phase 범위 밖이다(§13).

## 2. 기존 구조 (재확인)

* `TrainingConfig`(`config.py`): `epochs/batch_size/learning_rate`(필수)
  + `optimizer, momentum, weight_decay, gradient_clip_norm,
  label_smoothing, lr_scheduler, lr_scheduler_factor,
  lr_scheduler_patience, early_stopping_patience`(전부 선택).
  `RESUME_CONFIG_FIELDS`는 `optimizer/learning_rate/momentum/
  weight_decay/lr_scheduler/lr_scheduler_factor/lr_scheduler_patience/
  batch_size` 8개 -- `gradient_clip_norm`/`label_smoothing`은 제외.
* `loop.py::_build_criterion(config) -> nn.CrossEntropyLoss(label_smoothing=...)`,
  `run_training()`이 epoch 루프 진입 전 `criterion = _build_criterion(config)`를
  호출한다 -- 이 호출 지점에서 `device`는 이미 `run_training()`의
  파라미터로 존재한다.
* `checkpoint.py::load_training_checkpoint()`의 구조적 검증은
  `training_config` dict에 대해 `RESUME_CONFIG_FIELDS`(레거시 예외
  제외)만 필수로 요구하고, `TrainingConfig`의 전체 필드를 요구하지
  않는다. `save_training_checkpoint()`의 payload는
  `model_state_dict`/`optimizer_state_dict`/`scheduler_state_dict`만
  저장하고, criterion의 state는 어디에도 저장/복원하지 않는다.
* `imagefolder_workflow.py`는 `require_matching_num_classes(len(splits.classes),
  final_shape)`로 dataset class 수와 model 출력 shape 일치를 이미
  조기 검증하고 있다(`run_imagefolder_training_workflow()` 안에서
  dataset/model 로드 직후).

## 3. PyTorch 실측(constructor/forward/backward)

로컬 PyTorch(`2.12.1+cu126`)로 직접 실행해 확인:

```
all positive        constructor OK  forward OK   backward OK
one zero            constructor OK  forward OK   backward OK   (조용히 통과)
all zero            constructor OK  forward=nan  backward(grad=nan)
one negative        constructor OK  forward OK(이상한 부호)  backward OK
one NaN/+inf/-inf   constructor OK  forward=nan  backward(grad=nan)
wrong length         constructor OK  forward FAILED: RuntimeError(shape mismatch)
int dtype            constructor OK  forward FAILED: RuntimeError(dtype mismatch)
```

한 batch가 우연히 zero-weight class 샘플로만 구성되면 `NaN` loss가
직접 재현됨(`weight=[1.0, 0.0, 1.0]`, 그 batch의 label이 전부 weight=0인
class). weight+label_smoothing 조합은 PyTorch가 제약 없이 지원하며
(둘 다 켰을 때/각각만 켰을 때/둘 다 껐을 때 4가지 값이 모두 다름을
직접 확인), `state_dict()`는 weight가 설정되면 **비어있지 않다**
(`OrderedDict([('weight', tensor(...))])`) -- label_smoothing과 달리
buffer가 실제로 존재한다.

**결론**: PyTorch는 constructor에서 아무것도 검증하지 않고 forward에서도
NaN/inf/negative/zero 값 자체는 막지 않는다(shape/dtype 불일치만
RuntimeError) -- 이 프로젝트가 자체 validation을 둬야 한다.

## 4. `class_weights` representation

```python
class_weights: tuple[float, ...] | None = None
```

공식 representation은 **tuple뿐**이다. `TrainingConfig` 생성 시 list 등
다른 sequence를 넘기면 `TrainingConfigError`로 거부한다(canonicalize하지
않는다) -- CLI 경계(`scripts/train_imagefolder.py`)에서만
`tuple(args.class_weights)`로 변환한다. mutable aliasing 방지, config
representation을 하나로 고정, CLI parsing 관심사와 TrainingConfig
representation 관심사 분리가 목적이다.

## 5. validation 계약

`config.py`에 `_require_class_weights(name, value)` helper를 신설했다
(기존 scalar validator들과 별개 -- tuple/empty/index별 오류 메시지가
필요해 기존 helper에 억지로 끼워넣지 않았다):

* `None` → 비활성(기본값)
* `tuple`이 아니면 거부(list 포함)
* 빈 tuple 거부
* 각 원소: `bool` 거부, `int`/`float`만 허용, finite + strictly positive
  (`> 0`)만 허용 -- 0/음수/NaN/`+inf`/`-inf` 거부
* 오류 메시지에 index 포함: `'class_weights[1]' must be a finite positive
  number, got ...`

zero를 허용하지 않는 이유(§3의 실측 근거): all-zero/단일 zero-weight
class에 배치가 쏠리는 경우 NaN이 실제로 재현되고, negative weight는
크래시 없이 조용히 이상한 finite loss를 만들어 디버깅이 극히 어렵다.

## 6. `_build_criterion(config, device)` 확장

```python
def _build_criterion(config: TrainingConfig, device: str = "cpu") -> nn.Module:
    weight = (
        torch.tensor(config.class_weights, dtype=torch.float32, device=device)
        if config.class_weights is not None
        else None
    )
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
```

`device` 인자를 추가한 이유: `run_training()`이 `_build_criterion(config)`
를 호출하는 지점에 `device`가 이미 `run_training()` 자신의 파라미터로
존재하므로, 배관 비용이 거의 0이다. `_build_criterion`은
private 함수라 시그니처 변경이 공개 API를 깨지 않는다 -- 영향받는
곳은 `test_loop.py`의 직접 호출 3곳 + monkeypatch spy 2곳
(`test_loop.py`, `test_imagefolder_workflow.py`)뿐이며, 전부 실제로
수정했다(§14).

criterion lifecycle은 Phase 4N의 계약을 그대로 유지한다: `run_training()`
이 epoch 루프 진입 전 한 번만 생성하고, `train_one_epoch(...,
criterion=criterion)`으로 넘긴다 -- 배치/epoch마다 재생성하지 않는다.
`train_one_epoch()`의 기존 `criterion=None` fallback(무수정)은 여전히
unsmoothed/unweighted CE를 쓴다.

## 7. weight tensor dtype/device

dtype은 항상 `torch.float32`로 고정한다 -- PyTorch가 정수 dtype weight
tensor를 거부함을 실측 확인했고(§3), `config.class_weights`의 원소가
Python `int`/`float` 혼재여도 이 지점에서 항상 올바른 dtype으로
정규화된다. device는 `_build_criterion()`에 전달된 `device`(model/
입력과 동일)에 직접 생성한다 -- CPU 고정 현재 ImageFolder workflow에
국한되지 않고, 향후 GPU 노출 Phase가 오더라도 재사용 가능하도록
설계했다.

## 8. label_smoothing 조합

`class_weights=None+label_smoothing=0.0`(기존 numerical anchor, 반드시
유지), `class_weights=set+label_smoothing=0.0`,
`class_weights=None+label_smoothing>0`, `class_weights=set+label_smoothing>0`
네 조합 전부 PyTorch reference와 직접 비교하는 테스트로 고정했다
(§3의 실측 근거).

## 9. generic `run_training()`의 의도된 제한사항

`run_training()`은 `ModelSpec`도 ImageFolder class mapping도 모르고
이미 만들어진 `nn.Module`만 받는다. 따라서 `class_weights` 길이와
model 출력 class 수의 일치 여부를 generic 경로에서 별도로
introspection하지 않는다:

* **ImageFolder workflow**: 학습 시작 전에 명시적으로 검증(§10).
* **generic `run_training()` 경로**: 별도 검증 없음 -- 길이가 어긋나면
  PyTorch `CrossEntropyLoss`의 forward-time `RuntimeError`(§3에서 실측
  확인된 shape 검증)가 최종 backstop이다.

**"class weight 길이 mismatch는 항상 사전 검증된다"는 표현은 부정확하다**
-- ImageFolder 경로에 한해서만 사전 검증된다.

## 10. ImageFolder class-count mismatch 조기 검증

`imagefolder_workflow.py`의 `require_matching_num_classes(len(splits.classes),
final_shape)` 호출 바로 다음에, `class_weights`가 설정돼 있으면
`len(class_weights) == len(splits.classes)`를 확인하는 대칭적인 체크를
추가했다:

```python
if request.training_config.class_weights is not None and len(
    request.training_config.class_weights
) != len(splits.classes):
    raise ValueError(...)
```

기존 `require_matching_num_classes()`와 같은 `ValueError` 스타일을
재사용했다(새 예외 클래스를 만들지 않음). `run_training()`이 호출되기
전에 거부됨을 `run_training`을 monkeypatch해 직접 증명하는 테스트로
고정했다(§17).

## 11. class order 계약

`--class-weights`의 순서는 `class_mapping.json`의 `classes`/
`class_to_idx` 순서와 반드시 일치해야 한다(예: `classes=["cat","dog"]`
이면 `--class-weights 1.0 3.0`은 `cat=1.0, dog=3.0`). `TrainingConfig`/
generic training core에는 class 이름을 전혀 넣지 않는다 -- Phase 4O가
확립한 "generic core는 class 이름을 모르고, ImageFolder workflow만
class_mapping을 안다"는 분리를 그대로 유지한다.

## 12. resume 정책 -- mechanical vs semantic

**mechanical policy**(이 저장소가 실제로 코드화한 좁은 기준): criterion
관련 필드가 checkpoint의 `optimizer_state_dict`/`scheduler_state_dict`
처럼 `load_state_dict()`로 복원되어 새 config 값을 조용히 덮어쓸 위험이
있는가? `checkpoint.py`는 criterion의 state를 애초에 저장/복원하지
않으므로(§2), `class_weights`는 `gradient_clip_norm`/`label_smoothing`
과 동일한 처지다.

**semantic policy**: class_weights는 학습 objective를 바꾸는 값이므로
resume 중 바꾸는 게 위험하다는 시각. 하지만 이 우려는 label_smoothing
에도 동일하게 적용됐었고, 이미 이 프로젝트가 "`val_loss`(best-model
선택/early stopping/scheduler가 쓰는 값)는 항상 무수정 `evaluate()`
경로만 쓰므로 training-loss-only 변경은 이 결정 로직을 오염시키지
않는다"는 근거로 resume-free-change를 택했다. class_weights도 같은
구조(training-only, `evaluate()`/Phase 4O metric은 무영향)이므로 semantic
우려도 label_smoothing 선례로 해소된다.

**중요 -- 문구 정확성**: label_smoothing의 "criterion state_dict는
항상 빈 dict"라는 근거를 class_weights에 그대로 재사용하면 안 된다.
실측 결과 weight가 설정된 `CrossEntropyLoss`의 `state_dict()`는
**비어있지 않다**. 이 프로젝트의 실제 mechanical 근거는 "state_dict가
항상 비어서"가 아니라 "**checkpoint subsystem이 criterion의 state
자체를 저장/복원하지 않아서**"다 -- `loop.py`/`config.py`의 주석과
README를 이 정확한 표현으로 작성했다.

**결론**: `class_weights`는 `RESUME_CONFIG_FIELDS`에 포함하지 않고
resume 시 자유롭게 변경 가능하게 한다.

## 13. WeightedRandomSampler / automatic weighting 제외

명시적으로 범위 밖: automatic(inverse-frequency 등) 계산,
`WeightedRandomSampler`/oversampling/undersampling, class-name 기반
weight 지정 문법. class-weighted CE는 loss contribution을 조절하고
weighted sampler는 batch sampling 분포를 조절하는 서로 다른 개입
지점이라, 동시에 도입하면 overcompensation 위험이 있고 상호작용 튜닝
검증까지 범위에 들어와 과설계가 된다 -- 향후 별도 Phase 후보다.

## 14. 테스트 전략

아래 개수는 **pytest collected case 기준**이다(`pytest --collect-only -q`
로 실측) -- 일부 테스트가 `@pytest.mark.parametrize`를 쓰므로 Python
test 함수 개수와 실제 collected case 개수가 다르다. Phase 4O 시점의
파일별 baseline과 Phase 4P 구현 후 collected case 수를 실측 비교했다.

* `tests/training/test_config.py`: baseline 81 -> 96 (**+15 case**):
  default `None`, 정상 tuple, 빈 tuple 거부, list 거부, bool 원소 거부,
  zero/negative 거부, `NaN`/`+inf`/`-inf` 거부(parametrize 3 case), 오류
  메시지 index 포함, `RESUME_CONFIG_FIELDS` 미포함, `require_compatible_resume_config`
  가 차이를 허용(`None<->tuple` 양방향 포함, parametrize 3 case).
* `tests/training/test_loop.py`: baseline 96 -> 105 (**+9 case**):
  기존 Phase 4N `test_build_criterion_*` 스타일 재사용 -- default 회귀,
  PyTorch reference와 정확히 일치(weight only/weight+smoothing), 대조군
  (weighted가 unweighted와 다른 값 -- random seed가 아니라 두 target
  class의 per-sample loss가 서로 다르도록 손으로 구성한 deterministic
  logits/targets + 비대칭 weight를 써서, 어떤 실행에서도 항상 다른 값이
  나옴을 보장), weight tensor dtype/device, `run_training()`이 실제 weighted
  criterion을 `train_one_epoch()`에 전달하는지 spy로 고정(Phase 4N에서
  "criterion factory는 맞는데 실제로 안 넘겨진다"는 wiring gap을
  나중에야 잡았던 경험을 이번엔 처음부터 막음), `evaluate()`가
  class_weights를 무시하는지, resume 시 새 값이 실제 적용되는지(spy),
  동일 weights에서 exact-resume이 유지되는지.
* `tests/training/test_imagefolder_workflow.py`: baseline 44 -> 46
  (**+2 case**): 정상 길이 학습 성공, 길이 불일치 시 `run_training()`이
  호출되지 않았음을 monkeypatch로 직접 증명하며 조기 거부(class order
  자체는 Phase 4O에서 이미 강하게 고정했으므로 중복 테스트를 추가하지
  않았다).
* `tests/scripts/test_train_imagefolder_cli.py`: baseline 39 -> 42
  (**+3 case**): 정확한 tuple forwarding, 생략 시 `None`, invalid 값이
  기존 `TrainingConfigError -> exit code 1` 경로를 타는지.
* `tests/training/test_checkpoint.py`: baseline 42 -> 45 (**+3 case**):
  `weight_decay` legacy 회귀 테스트와 동일 패턴(가짜 dict가 아니라 실제
  저장 파일에서 키를 지움)으로, `class_weights` 키가 없는 legacy 파일이
  정상 로드되고 resume 호환성 검사도 통과하며(신규 값과 무관), 값이
  설정된 경우 round-trip이 보존됨을 확인(함수 3개, parametrize 없음).
  **`checkpoint.py` production 코드는 무수정** -- `class_weights`가
  `RESUME_CONFIG_FIELDS`에 없으므로 Phase 4L 같은 legacy-default 예외
  자체가 필요 없음을 이 테스트들이 실제 파일 경로로 증명한다.
* 전체 회귀: **Phase 4O baseline 558 -> Phase 4P 590 (+32 collected
  case)** = config 15 + loop 9 + workflow 2 + cli 3 + checkpoint 3,
  590 passed.

## 15. E2E 전략

새 Phase 4P 전용 E2E 스크립트는 만들지 않았다. 기존 5개
(`run_phase1_e2e.py`, `run_training_e2e.py`, `run_real_training_e2e.py`,
`run_resume_training_e2e.py`, `run_imagefolder_training_e2e.py`)를
기본 `class_weights=None` 경로로 재실행해 numerical anchor가 전부
그대로 유지됨을 확인했다(PASS, 수치 완전 동일). non-default class
weight 경로는 `test_imagefolder_workflow.py`의 통합 테스트 레벨에서
실제 학습까지 검증했다(§14) -- E2E 스크립트 레벨까지 새로 만들 필요는
없었다.

## 16. 제외 범위 (재확인)

automatic(inverse-frequency 등) class weight 계산, `WeightedRandomSampler`
/oversampling/undersampling, class-name 기반 weight 지정 문법, zero/
negative weight, validation/test weighting, Phase 4O metric 로직 변경,
GPU/device CLI 노출, AMP, 추가 LR scheduler, `checkpoint.py`/`history.py`/
`metrics.py`(전부 무수정).
