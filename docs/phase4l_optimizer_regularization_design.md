# Phase 4L: Optimizer Regularization Extension — 설계안

**상태: 구현 완료, 전체 단위/통합 테스트 및 기존 E2E 회귀 검증 완료.**
`TrainingConfig`에 `weight_decay`를 추가하고 optimizer 선택지에
`AdamW`를 더해, README가 "아직 미지원"으로 명시해 온 정규화 공백
(weight decay)을 메웠다. Phase 4A~4K가 쌓아 온 `config.py` →
`loop.py`(`_build_optimizer`) → CLI라는 3계층 구조를 그대로 확장했고,
`checkpoint.py`/`imagefolder_workflow.py`는 계획대로 무수정으로
끝났다.

**전제**: 이 문서는 Phase 4K 완료 후 별도 검토 라운드(다음 Phase 후보
A~J 비교, weight_decay/optimizer 확장을 Phase 4L로 채택)를 거쳐 확정된
정책을 그대로 문서화한 것이다. 정책 자체의 재검토는 이 문서의 목적이
아니다.

---

## 1. 범위

지원하는 것:

- `TrainingConfig.weight_decay: float = 0.0` — Adam/SGD/AdamW 공통 적용.
- `optimizer` 선택지에 `"adamw"` 추가(`torch.optim.AdamW`). 기본값은
  기존과 동일하게 `"adam"`.
- CLI `--weight-decay FLOAT`(기본 `0.0`).
- resume 호환성: `weight_decay`를 `RESUME_CONFIG_FIELDS`에 추가하되,
  **Phase 4L 이전 checkpoint에 한해** `weight_decay` 키가 없으면
  `0.0`으로 저장됐던 것으로 간주하는 좁은 하위 호환 규칙(§5).

지원하지 않는 것(의도적, 이번 Phase 범위 밖):

- gradient clipping
- loss function 선택(label smoothing 등)
- `"plateau"` 외 LR scheduler(StepLR/CosineAnnealingLR 등)
- Adam `betas`/`eps` 노출
- SGD `dampening`/`nesterov`
- GPU/device 노출(`imagefolder_workflow.py`의 `device="cpu"` 하드코딩은
  그대로 유지)
- mixed precision
- checkpoint 정책 확장(latest+best 분리, overwrite 등)

`src/image_ai_studio` 아래 **production code**에서는 `training/config.py`와
`training/loop.py`(`_build_optimizer()`)만 수정했다. `training/checkpoint.py`
(payload를 `asdict(training_config)`로 그대로 저장하므로 무수정),
`training/imagefolder_workflow.py`(`training_config`를 불투명하게
통과시키므로 무수정), `training/history.py`, `model_definition/*`,
`export/*`, `parity/*`, C++ 코드, `scripts/run_imagefolder_training_e2e.py`
(기본값만 쓰므로 anchor 불변)는 전부 수정하지 않았다 -- 실제 구현도
그대로 이 경계를 지켰다(`git diff --stat`으로 확인).

---

## 2. 현재 구조 분석 (재확인)

### 2-1. `TrainingConfig`와 검증 helper (`config.py`)

```python
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float

    optimizer: str = "adam"  # "adam" | "sgd"
    momentum: float = 0.9

    lr_scheduler: str | None = None
    lr_scheduler_factor: float = 0.1
    lr_scheduler_patience: int = 1

    early_stopping_patience: int | None = None
```

검증 helper는 용도별로 이미 분화되어 있다: `_require_positive_int`,
`_require_positive_float`(0 초과), `_require_fraction`(0~1 범위,
momentum/lr_scheduler_factor용), `_require_one_of`(선택지 검증). **이
중 `weight_decay`(0 이상, 상한 없음, NaN/inf 거부)에 맞는 helper는
없다** — `_require_positive_float`은 `>0`을 요구해 `0.0`을 거부하므로
부적합, `_require_fraction`은 `<1.0` 상한이 있어 부적합. 새 helper가
필요하다(§4-2).

### 2-2. `_build_optimizer()` (`loop.py`)

```python
def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """config.optimizer에 따라 Adam 또는 SGD를 생성. TrainingConfig.__post_init__이
    이미 "adam"/"sgd" 외의 값을 거부하므로 그 외 분기는 없다."""
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
```

이 함수가 정확히 이번 Phase의 확장 지점이었다 — `weight_decay` 전달과
`AdamW` 분기 둘 다 이 함수 안에서 끝났다. `run_training()`의 epoch
루프/`checkpoint_hook`/`progress_callback`/`should_stop` 순서(Phase
4I~4K가 정밀 검증한 부분)는 전혀 건드리지 않았다(§6에 실제 구현 결과).

### 2-3. checkpoint payload 흐름 (`checkpoint.py`, 무수정 이유)

```python
payload = {
    "format_version": CHECKPOINT_FORMAT_VERSION,
    ...
    "training_config": asdict(training_config),
    ...
}
```

`training_config`를 `asdict()`로 그대로 저장하므로, `TrainingConfig`에
필드를 추가해도 `checkpoint.py`는 **한 줄도 고칠 필요가 없다**(Phase 4I가
`stopped_by_user`를 이 방식으로 무수정 추가한 전례와 동일). `optimizer.
state_dict()`의 `param_groups`에도 PyTorch가 `weight_decay`를 자동으로
포함한다.

### 2-4. resume 호환성 강제 지점 (`config.py`)

```python
def require_compatible_resume_config(checkpoint_config: dict, resume_config: TrainingConfig) -> None:
    ...
    missing = [name for name in RESUME_CONFIG_FIELDS if name not in checkpoint_config]
    if missing:
        raise ValueError(f"checkpoint training_config is missing required field(s): {missing}")

    for field_name in RESUME_CONFIG_FIELDS:
        saved_value = checkpoint_config[field_name]
        new_value = getattr(resume_config, field_name)
        if saved_value != new_value:
            raise ValueError(...)
```

`run_training()`이 resume 시 항상 이 함수를 호출하므로(caller가
빼먹어도 우회 불가), `RESUME_CONFIG_FIELDS`에 `weight_decay`를 추가하면
이 지점 하나로 강제력이 생긴다. 이번 Phase에서 **resume compatibility와
관련된** 실제 로직 변경은 이 함수 안에만 존재한다(§5) -- config
validation(`_require_non_negative_finite_float`)과 optimizer 생성
(`_build_optimizer()`), CLI wiring에도 각각 별도의 실제 로직 변경이
있다(§4, §6, §7).

### 2-5. `ImageFolderWorkflowRequest`/CLI 데이터 흐름

```text
CLI argparse(args.weight_decay)
  → TrainingConfig(weight_decay=...)          (scripts/train_imagefolder.py main())
  → ImageFolderWorkflowRequest.training_config (불투명하게 통과, 무수정)
  → run_imagefolder_training_workflow()
  → run_training()
  → _build_optimizer(model, config)
  → torch.optim.Adam/SGD/AdamW(..., weight_decay=config.weight_decay)
```

`ImageFolderWorkflowRequest`는 `training_config: TrainingConfig` 필드
하나로 이미 이 값을 감싸 전달하므로, **`imagefolder_workflow.py`는
이번 Phase에서 단 한 줄도 수정하지 않았다.**

### 2-6. 기존 테스트 구조

`tests/training/test_config.py`(필드별 독립 `TrainingConfigError`
테스트), `tests/training/test_loop.py`(`_build_optimizer`/
`_build_scheduler` 단위 테스트 + exact-resume 비교 테스트),
`tests/training/test_checkpoint.py`(`require_compatible_resume_config`
테스트), `tests/scripts/test_train_imagefolder_cli.py`(CLI 배선 테스트,
`cli.main()` 직접 호출 + request 캡처 패턴).

---

## 3. `TrainingConfig` 변경

```python
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float

    optimizer: str = "adam"  # "adam" | "sgd" | "adamw"
    momentum: float = 0.9  # optimizer="sgd"일 때만 사용

    weight_decay: float = 0.0  # Adam/SGD/AdamW 공통 적용

    lr_scheduler: str | None = None
    lr_scheduler_factor: float = 0.1
    lr_scheduler_patience: int = 1

    early_stopping_patience: int | None = None
```

필드 위치는 `momentum` 바로 다음(둘 다 optimizer 관련 하이퍼파라미터)에
둔다. 기본값 `0.0`은 기존 동작(weight decay 없음)을 그대로 재현한다.

```python
OPTIMIZER_CHOICES = ("adam", "sgd", "adamw")
```

---

## 4. Validation

### 4-1. 정책(확정)

- bool 거부
- 유한한 숫자(NaN/±inf 거부)
- `0.0` 이상(음수 거부)
- **임의의 상한을 두지 않는다** — `1.0` 이상의 유한한 값도 허용(실제
  적절성은 사용자의 hyperparameter 선택 책임).

### 4-2. 신규 helper: `_require_non_negative_finite_float()`

기존 helper 중 이 요구사항(하한 0, 상한 없음, NaN/inf 거부)에 맞는
것이 없으므로 하나 추가한다 — `_require_positive_float`/`_require_fraction`
과 나란히, 같은 스타일로:

```python
def _require_non_negative_finite_float(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigError(f"'{name}' must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise TrainingConfigError(f"'{name}' must be a finite number >= 0.0, got {value!r}")
```

`math.isfinite()`가 `NaN`/`+inf`/`-inf`를 한 번에 걸러낸다(`math.isfinite(float("nan"))`,
`math.isfinite(float("inf"))` 전부 `False`). `config.py`에 `import math`
추가가 필요하다(현재 없음).

`__post_init__()`에는 다른 필드들과 같은 자리(optimizer 관련 검증
블록)에 `_require_non_negative_finite_float("weight_decay", self.weight_decay)`
한 줄을 추가한다. `momentum`과 동일하게 **optimizer 종류와 무관하게
항상 검증**한다(어떤 optimizer를 고르든 `TrainingConfig`가 항상
일관되게 유효한 값을 갖도록 하기 위함 — 기존 momentum 검증의 근거를
그대로 재사용).

---

## 5. Resume 호환성: `weight_decay`에 한정된 마이그레이션 규칙

### 5-1. 정책(확정)

`weight_decay`를 `RESUME_CONFIG_FIELDS`에 추가하되, **오직 이 필드에만**
다음 예외를 둔다: checkpoint의 저장된 `training_config`에 `weight_decay`
키가 없으면(= Phase 4L 이전에 저장된 checkpoint), 그 checkpoint는
`weight_decay=0.0`으로 학습된 것으로 간주한다. 다른 필드가 누락된
경우는 기존과 동일하게 무조건 거부한다(정책 변경 없음).

```text
과거 checkpoint config에 weight_decay 없음 + 현재 config.weight_decay == 0.0
  → resume 허용 (과거 checkpoint의 실제 의미가 0.0이었다고 보는 것이 안전)

과거 checkpoint config에 weight_decay 없음 + 현재 config.weight_decay > 0.0
  → resume compatibility 오류 (값이 실제로 달라지므로 momentum/lr 불일치와 동일하게 거부)
```

### 5-2. 구현 형태 (`require_compatible_resume_config()`)

`checkpoint_config` dict 자체는 **mutate하지 않는다** — 지역 변수로만
치환값을 다룬다:

```python
def require_compatible_resume_config(checkpoint_config: dict, resume_config: TrainingConfig) -> None:
    if not isinstance(checkpoint_config, dict):
        raise ValueError(...)

    # weight_decay는 Phase 4L 이전 checkpoint에 존재하지 않을 수 있으므로
    # "필수 필드 누락" 검사에서 별도로 취급한다(§5) -- 그 외 필드는 기존과
    # 동일하게 반드시 있어야 한다.
    strictly_required_fields = [name for name in RESUME_CONFIG_FIELDS if name != "weight_decay"]
    missing = [name for name in strictly_required_fields if name not in checkpoint_config]
    if missing:
        raise ValueError(f"checkpoint training_config is missing required field(s): {missing}")

    for field_name in RESUME_CONFIG_FIELDS:
        if field_name == "weight_decay" and field_name not in checkpoint_config:
            # Phase 4L 이전 checkpoint 호환: weight_decay 키가 없으면 그
            # checkpoint가 실제로 weight_decay=0.0으로 학습된 것으로
            # 간주한다(checkpoint_config 자체는 건드리지 않음).
            saved_value = 0.0
        else:
            saved_value = checkpoint_config[field_name]
        new_value = getattr(resume_config, field_name)
        if saved_value != new_value:
            raise ValueError(
                f"cannot resume: checkpoint was saved with {field_name}={saved_value!r} "
                f"but resume config uses {field_name}={new_value!r}"
            )
```

`weight_decay`가 checkpoint에 **존재하는 경우**(Phase 4L 이후 저장된
checkpoint)는 이 특수 처리가 전혀 개입하지 않고 다른 필드와 완전히
동일한 일치 검사를 받는다 — 예외는 오직 "키 자체가 없을 때"만
적용된다.

### 5-3. 이 정책이 안전한 이유

`load_training_checkpoint()`(`checkpoint.py`)는 `_REQUIRED_HISTORY_FIELDS`
같은 구조적 필수 필드 목록과 별개로, `training_config` dict 자체의
필드 존재 여부는 검사하지 않는다(순수 dict로 반환) — 따라서 이 정책은
`checkpoint.py`를 전혀 건드리지 않고 `config.py`의 비교 함수 안에서만
완결된다. `RESUME_CONFIG_FIELDS`에 새 필드를 추가하는 기존 패턴
(momentum, lr_scheduler 등)과 달리 `weight_decay`만 "누락 시 기본값
간주"라는 예외를 갖는다는 점을 이 함수 docstring에 명시해, 향후 또
다른 필드를 추가할 때 이 예외가 실수로 일반화되지 않도록 한다.

---

## 6. `_build_optimizer()` 변경 (`loop.py`)

```python
def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """config.optimizer에 따라 Adam, AdamW, 또는 SGD를 생성. TrainingConfig.__post_init__이
    이미 "adam"/"sgd"/"adamw" 외의 값을 거부하므로 그 외 분기는 없다.
    weight_decay는 세 optimizer 모두에 공통으로 전달한다(Phase 4L)."""
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
```

`_build_scheduler()`/epoch 루프/`checkpoint_hook`/`progress_callback`/
`should_stop` 순서는 전혀 건드리지 않았다 — scheduler는 optimizer
인스턴스만 받으므로(`ReduceLROnPlateau(optimizer, ...)`) `AdamW`를
넘겨도 무수정으로 그대로 동작했다.

---

## 7. CLI 변경 (`scripts/train_imagefolder.py`)

```python
parser.add_argument("--optimizer", choices=["adam", "sgd", "adamw"], default="adam")
...
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.0,
    help="L2 정규화 계수 (Adam/SGD/AdamW 공통 적용, 기본값 0.0 = 미적용)",
)
```

`main()`의 `TrainingConfig(...)` 조립부에 `weight_decay=args.weight_decay`
한 줄만 추가했다. `ImageFolderWorkflowRequest`/이후 배선은 계획대로
무수정(§2-5).

---

## 8. 테스트 및 검증 결과

기존 전체 테스트 437개에 Phase 4L 신규 테스트 24개를 더해 `pytest -q`
전체 **461 passed**(신규 실패/skip 없음)를 확인했다.

### 8-1. `tests/training/test_config.py`

- `weight_decay` 기본값 `0.0`(기존 `test_default_config_reproduces_phase_4a_4d_behavior`
  패턴에 단언 추가 또는 신규 테스트).
- 음수 거부, `NaN`/`+inf`/`-inf` 거부, bool 거부.
- `0.0`과 `1.0` 이상의 유한한 양수 모두 허용(상한 없음 확인).
- `optimizer="adamw"` 허용(`test_accepts_adam_optimizer`/`test_accepts_sgd_optimizer`와
  나란히 `test_accepts_adamw_optimizer`).
- **resume 호환성 마이그레이션 규칙**(§5, 신규 핵심 테스트):
  - checkpoint config에 `weight_decay` 키 없음 + `resume_config.weight_decay == 0.0`
    → `require_compatible_resume_config()`가 예외 없이 통과.
  - checkpoint config에 `weight_decay` 키 없음 + `resume_config.weight_decay > 0.0`
    → `ValueError`(다른 필드 불일치와 동일한 메시지 형식).
  - 이 두 테스트는 `checkpoint_config` dict가 함수 호출 후에도 원래
    내용 그대로인지(mutate 안 됨) 함께 확인한다.
- **실제 결과**: 위 항목 전부 테스트로 작성해 통과를 확인했다(adamw
  허용, weight_decay 기본값 0.0, 0 허용, 상한 없는 유한한 양수 허용,
  음수 거부, NaN 거부, +inf/-inf 거부, bool 거부, optimizer 종류와
  무관한 weight_decay 검증, migration 두 시나리오 + `checkpoint_config`
  dict 비변경 확인 포함).

### 8-2. `tests/training/test_loop.py`

- `_build_optimizer()`가 Adam/SGD/AdamW 각각에 `weight_decay`를 실제로
  전달하는지(`optimizer.param_groups[0]["weight_decay"]`).
- `optimizer="adamw"`가 실제 `torch.optim.AdamW` 인스턴스를 반환하는지.
- **exact-resume 회귀**: `weight_decay != 0`(+ 가능하면 `optimizer="adamw"`
  조합)에서도 continuous run과 resume run이 tensor-level exact
  equality를 유지하는지(기존
  `test_run_training_resume_matches_continuous_run_exactly` 패턴 재사용,
  Dropout 포함 모델로 CPU RNG 복원 필요성까지 함께 고정).
- **실제 결과**: Adam/SGD/AdamW 각각에 weight_decay가 param_groups에
  실제로 전달됨을 확인, `optimizer="adamw"`가 실제 `torch.optim.AdamW`를
  생성함을 확인, 기본값 `weight_decay=0.0`이 param_groups에도 그대로
  반영됨을 확인, AdamW + non-zero weight_decay 조합의 continuous run과
  resume run이 model/optimizer/scheduler state 전부 tensor-level exact
  equality를 유지함을 회귀 테스트로 고정했다.

### 8-3. `tests/training/test_checkpoint.py`

- `test_require_compatible_resume_config_rejects_mismatched_fields`의
  parametrize 목록에 `{"weight_decay": 0.5}` 추가(양쪽 다 필드가 있는
  일반 불일치 케이스 — §5의 마이그레이션 특수 케이스와는 별개).
- 기존 9개 이상의 `require_compatible_resume_config`/atomic save 테스트
  전부 무수정 통과 재확인.
- **실제 결과**: `weight_decay` mismatch 케이스를 parametrize 목록에
  추가해 통과를 확인했고, 기존 테스트는 전부 무수정으로 통과했다.

### 8-4. `tests/scripts/test_train_imagefolder_cli.py`

- `--weight-decay` 값이 `TrainingConfig`에 정확히 전달되는지(기존
  `test_checkpoint_every_forwards_exact_value_to_workflow_request`와
  동일한 request 캡처 패턴).
- 플래그 생략 시 기본값 `0.0`.
- 기존 CLI 테스트 전부 무수정 통과.
- **실제 결과**: `--weight-decay`/`--optimizer adamw` 전달을 request
  캡처로 확인했고, 플래그 생략 시 `weight_decay=0.0` 기본값도 별도
  테스트로 확인했다. 기존 CLI 테스트는 전부 무수정으로 통과했다.

### 8-5. E2E

새 E2E 스크립트를 추가하지 않았다. 기존 4개 E2E(`run_training_e2e.py`/
`run_real_training_e2e.py`/`run_resume_training_e2e.py`/
`run_imagefolder_training_e2e.py`)는 전부 `optimizer="adam"`(또는
`TrainingConfig` 기본값) + `weight_decay` 미지정 조합만 쓴다. 실제로
4개 전부 재실행해 **기존 수치 anchor가 완전히 동일하게 유지된 채
PASS**함을 확인했고, TorchScript export/C++(LibTorch) CPU·CUDA parity도
기존과 동일하게 PASS했다.

---

## 9. 호환성과 위험 요소

- **기본 동작 변화**: 없음(`weight_decay=0.0`, `optimizer` 기본값
  `"adam"` 그대로).
- **기존 checkpoint 호환성**: §5의 마이그레이션 규칙으로 해결 —
  Phase 4L 이전 checkpoint를 `weight_decay=0.0`으로 resume하는 것은
  계속 가능하고, `weight_decay>0`으로 resume하려 하면 명확히 거부된다.
- **serialization**: 영향 없음(순수 float 필드, 신규 텐서 없음).
- **deterministic/reproducibility**: `weight_decay`/`AdamW` 선택은
  optimizer의 매 step 계산에 관여하므로 값이 다르면 학습 궤적이
  달라지는 것이 당연하다 — `RESUME_CONFIG_FIELDS` 강제 일치로 resume
  중 값이 조용히 바뀌는 것을 막는다(§5의 예외는 "누락 시 0.0 간주"뿐,
  실제 값 불일치는 여전히 거부).
- **CPU/CUDA 차이**: 없음(이번 Phase는 CPU 경로만 다룸, device 노출은
  범위 밖).
- **TorchScript/C++ parity**: 영향 없음 — weight_decay/optimizer 선택은
  학습 중에만 관여하고 export된 추론 그래프를 바꾸지 않는다.
- **Phase 4K(graceful interruption)와의 상호작용**: 없음 — Ctrl+C로
  중단된 학습의 checkpoint에도 새 `weight_decay` 필드가 동일한 경로로
  자동 포함된다(별도 배선 불필요).
- **기존 CLI 호출과의 backward compatibility**: `--weight-decay`를
  생략하면 기존 명령어가 완전히 동일하게 동작한다.
- **검증 경계 (수동 재현 미실시)**: Phase 4L 이전에 실제로 생성된
  사용자 checkpoint 파일을 이용한 수동 resume 재현은 이번 구현 라운드에서
  수행하지 않았다. 다만 `weight_decay` 키가 없는 checkpoint config를
  흉내낸 dict를 이용해 (a) 현재 `weight_decay=0.0`이면 resume 허용,
  (b) 현재 `weight_decay>0.0`이면 resume 거부, (c) 두 경우 모두
  `checkpoint_config` dict가 mutate되지 않음을 자동화 테스트로 검증했다
  (§8-1). 이것은 현재 blocker로 간주하지 않는다 -- 실제 구버전 checkpoint의
  `training_config` 저장 형식(순수 `asdict()` dict)은 이 테스트가 흉내낸
  구조와 동일하다.

---

## 10. Acceptance Criteria

- [x] `TrainingConfig.weight_decay` 기본값 `0.0`, 음수/`NaN`/`±inf`/bool
      거부, 상한 없이 유한한 양수 허용 -- `test_config.py`에서 검증 완료
- [x] `OPTIMIZER_CHOICES`에 `"adamw"` 추가, 기존 `"adam"`/`"sgd"`와
      동일한 `_require_one_of` 검증 경로를 통과 -- 검증 완료
- [x] `_build_optimizer()`가 Adam/SGD/AdamW 세 경우 모두에 `weight_decay`를
      전달함(단위 테스트로 고정) -- `test_build_optimizer_passes_weight_decay_to_param_groups`로 검증 완료
- [x] `optimizer="adamw"`가 `torch.optim.AdamW` 인스턴스를 생성함 -- 검증 완료
- [x] `RESUME_CONFIG_FIELDS`에 `weight_decay` 추가, §5의 마이그레이션
      규칙(누락 시 0.0 간주, 그 값과 다르면 거부)이 정확히 동작하고
      `checkpoint_config` dict를 mutate하지 않음 -- migration 두 시나리오
      + dict 비변경 확인 전부 검증 완료
- [x] `weight_decay != 0`(+ `optimizer="adamw"` 조합 포함)에서도
      continuous run과 resume run이 tensor-level exact equality를
      유지함(회귀 테스트로 고정) --
      `test_run_training_resume_matches_continuous_run_exactly_with_adamw_weight_decay`로
      검증 완료
- [x] `--weight-decay` CLI 플래그가 `TrainingConfig`에 정확히 배선됨 -- CLI wiring 테스트로 검증 완료
- [x] `checkpoint.py`/`imagefolder_workflow.py` 무수정 확인(git diff로
      검증) -- `git diff --stat`에 두 파일이 나타나지 않음을 확인
- [x] 전체 pytest 통과 -- 기존 437개 + 신규 24개 = **461 passed**
- [x] 4개 기존 E2E anchor 수치 완전 불변 -- 4개 전부 재실행해 PASS,
      TorchScript/C++ CPU·CUDA parity도 기존과 동일하게 PASS
- [x] README "현재 지원 범위"/"아직 미지원" 목록 갱신 -- 반영 완료

---

## 11. 구현 순서 및 완료 결과

1. [x] `config.py`: `import math` 추가, `_require_non_negative_finite_float()`
   추가, `TrainingConfig.weight_decay` 필드 + `__post_init__()` 검증
   추가, `OPTIMIZER_CHOICES`에 `"adamw"` 추가, `RESUME_CONFIG_FIELDS`에
   `weight_decay` 추가, `require_compatible_resume_config()`에 §5의
   마이그레이션 로직 반영.
2. [x] `test_config.py`: §8-1의 단위 테스트 작성/통과.
3. [x] `loop.py`: `_build_optimizer()`에 `weight_decay` 전달 + AdamW 분기
   추가.
4. [x] `test_loop.py`: §8-2의 단위/exact-resume 테스트 작성/통과.
5. [x] `test_checkpoint.py`: §8-3의 회귀 테스트 추가/통과.
6. [x] `scripts/train_imagefolder.py`: `--weight-decay` 플래그 + `--optimizer`
   choices 확장 + `TrainingConfig` 조립부 배선.
7. [x] `test_train_imagefolder_cli.py`: §8-4의 CLI 배선 테스트 작성/통과.
8. [x] 전체 pytest 재실행(461 passed) + 4개 기존 E2E 재실행(anchor 불변 확인, 전부 PASS).
9. [x] README 갱신("현재 지원 범위"에 Phase 4L 항목 추가, "아직 미지원"
   목록에서 weight decay 관련 문구 정리).

위 9단계 전부 이 순서 그대로 완료됐다 -- 계획과 실제 구현 순서 사이에
차이는 없었다.

---

## 12. 향후 확장과의 연결

- **gradient clipping**: `_build_optimizer()`가 아니라 `train_one_epoch()`
  내부(`loss.backward()`와 `optimizer.step()` 사이) 한 줄 추가로 끝나는
  구조라, 이번 Phase가 확립한 "config 필드 → RESUME_CONFIG_FIELDS →
  CLI" 절차를 그대로 재사용할 수 있는 별도 소규모 Phase 후보.
- **loss function 선택(label smoothing)**: `criterion` 생성이
  `train_one_epoch()`/`evaluate()` 두 곳에 중복 하드코딩되어 있어,
  이번 Phase의 `_build_optimizer()`와 유사한 `_build_criterion()` 팩토리
  신설이 자연스러운 다음 단계.
- **epoch-based scheduler 확장(StepLR/CosineAnnealingLR)**: `scheduler.
  step(val_loss)` 고정 호출을 metric 기반/epoch 기반으로 분기해야 해
  `loop.py`의 핵심 제어 흐름을 다시 여는 더 큰 작업 — 이번 Phase처럼
  가볍게 끝나지 않는다.
