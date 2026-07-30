# Phase 4A: Training

이 문서는 Phase 4A(학습 기반 구현)의 설계와 구현 내용을 정리한다. 목표는
범용 학습 프레임워크가 아니라, 다음 경로를 이 프로젝트에서 **처음으로**
끝까지 연결하는 것이다:

```text
ModelSpec JSON -> build_model() -> 실제 학습 -> validation
    -> weight 저장/재로드 -> TorchScript export -> 기존 C++ inference
    -> Python/C++ parity
```

Phase 0~3은 전부 random-init 가중치로 export/parity 파이프라인만
검증했다 (`tools/prepare_test_artifacts.py` 참고 -- 학습 코드 자체가
없었다). Phase 4A가 이 프로젝트 최초로 "실제로 학습된" 모델을 배포하는
경로다.

**중요 -- 실행 환경 제약**: 이번 작업 환경에는 Python이 설치되어 있지
않아 (`python --version` -> Windows Store 스텁, exit 49) `pytest`/E2E
스크립트를 실제로 실행하지 못했다. 아래 결과 항목은 실행 결과가 아니라
코드 재검토로 확인한 정적 검증이며, 실측이 필요한 항목은 명시적으로
"미실행"으로 표시했다.

## 1. 범위

`model_definition/*`는 전혀 수정하지 않는다. `training/`은 `build_model()`이
반환하는 표준 `nn.Module`만 소비하는 완전히 독립적인 새 패키지다.

지원:

* epochs/batch_size/learning_rate만 있는 `TrainingConfig` (optimizer=Adam,
  loss=CrossEntropyLoss 고정)
* 외부 다운로드 없는 합성 이미지 분류 데이터셋 (train/val 분리, 고정 seed)
* `train_one_epoch()` / `evaluate()` / 이 둘을 묶는 `run_training()`
* `state_dict` 저장/재로드

미지원 (의도적, Phase 4B 이후):

* optimizer/loss 선택 registry, scheduler, early stopping
* optimizer state/epoch가 포함된 full checkpoint, resume
* 실제 외부 데이터셋(CIFAR-10 등), torchvision, augmentation
* mixed precision, multi-GPU/distributed
* 일반 DAG, UI, detection/segmentation training

## 2. 디렉터리 구조

```text
src/image_ai_studio/training/
    __init__.py
    config.py          TrainingConfig (+ TrainingConfigError)
    dataset.py           make_class_patterns, SyntheticImageDataset, make_train_val_datasets
    loop.py                train_one_epoch, evaluate, run_training, TrainingHistory
    checkpoint.py            save_state_dict, load_state_dict

tests/training/
    test_config.py
    test_dataset.py
    test_loop.py
    test_model_definition_integration.py   # BatchNorm/Dropout/ResidualBlock/Branch 학습 검증
    test_checkpoint.py
    test_train_export_parity.py            # 학습 -> TorchScript export -> parity

scripts/run_training_e2e.py
examples/models/phase4_training_model.json
```

`model_definition`과의 의존 방향은 한쪽으로만 흐른다: `training/*`는
`model_definition.builder.build_model()`이 반환한 `nn.Module`만 다루고,
`model_definition/*`는 `training`을 전혀 모른다.

## 3. Synthetic Dataset

클래스마다 고정된 랜덤 패턴(`class_patterns`)을 Train/Validation이
**공유**하고, 각자 다른 seed로 뽑은 labels/noise만 따로 갖는 구조다:

```text
seed
 -> 공통 class_patterns 생성

공통 class_patterns
 +-> Train labels/noise      : seed + 1
 +-> Validation labels/noise : seed + 2
```

```python
def make_class_patterns(input_shape, num_classes, seed) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_classes, *input_shape, generator=generator)


class SyntheticImageDataset(Dataset):
    def __init__(self, class_patterns, num_samples, seed, noise_scale=NOISE_SCALE):
        num_classes = class_patterns.shape[0]
        input_shape = tuple(class_patterns.shape[1:])
        generator = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        noise = torch.randn(num_samples, *input_shape, generator=generator) * noise_scale
        self.class_patterns = class_patterns
        self.images = class_patterns[labels] + noise
        self.labels = labels


def make_train_val_datasets(input_shape, num_classes, seed, train_size=64, val_size=32):
    class_patterns = make_class_patterns(input_shape, num_classes, seed=seed)
    train_dataset = SyntheticImageDataset(class_patterns, train_size, seed=seed + 1)
    val_dataset = SyntheticImageDataset(class_patterns, val_size, seed=seed + 2)
    return train_dataset, val_dataset
```

설계 근거:

* **Train/Validation이 같은 class prototype을 공유해야 한다.** class_patterns를
  따로 생성하면 같은 class index("class 2")가 train과 val에서 서로 다른
  특징을 의미하게 되어 검증 자체가 왜곡된다 -- `make_train_val_datasets`가
  `class_patterns`를 한 번만 만들어 두 `SyntheticImageDataset`에 동일한
  텐서를 그대로 전달해 이를 보장한다 (`train.class_patterns`와
  `val.class_patterns`는 값이 같은 게 아니라 같은 텐서 객체).
* **labels/noise는 서로 다른 파생 seed(`seed+1`, `seed+2`)로 뽑아 완전히
  분리된 샘플**을 만든다 (하나의 데이터셋을 슬라이싱하는 방식보다
  통계적 독립성이 명확함). Test set은 이번 Phase 범위에서 제외했다
  (요청 사항).
* **자체 `torch.Generator`만 사용**하고 전역 RNG(`torch.manual_seed`)에는
  손대지 않는다. 그래서 이 데이터셋을 언제, 몇 번, 어떤 순서로 생성하든
  같은 `seed`면 항상 완전히 동일한 텐서가 나온다 (테스트 재현성이 호출
  순서에 의존하지 않음).
* **순수 노이즈가 아니라 클래스별로 실제로 구분되는 패턴**을 사용한다.
  라벨과 무관한 순수 노이즈였다면 "training loss가 감소한다"는 검증
  자체가 성립할 수 없다 -- 이 데이터셋은 CNN이 실제로 풀 수 있는 최소
  분류 문제가 되도록 만들어졌다.
* `torchvision`은 의존성에 추가하지 않았다 (`torch.utils.data.Dataset`만
  사용, `requirements.txt` 변경 없음).

## 4. TrainingConfig

```python
@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
```

`model_definition/specs.py`의 검증 스타일(양의 정수/실수 확인, bool을
int로 오인하지 않도록 별도 차단)을 그대로 따르되, `model_definition`을
import하지 않고 `training/config.py`에 동일한 검증 로직을 자체적으로
작성했다 -- `TrainingConfigError`는 `ModelValidationError`와 별개의
타입이다 ("모델 정의가 잘못됨"과 "학습 설정이 잘못됨"은 다른 개념이라
같은 예외를 재사용하지 않았다). optimizer는 `torch.optim.Adam`,
loss는 `nn.CrossEntropyLoss`로 `loop.py`에 하드코딩했다 -- 선택
registry는 만들지 않았다 (요청 사항).

## 5. Training Loop

```python
def train_one_epoch(model, loader, optimizer, device="cpu") -> float:
    model.train()
    ...  # forward -> CrossEntropyLoss -> zero_grad -> backward -> step
    return epoch_avg_loss

def evaluate(model, loader, device="cpu") -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        ...
    return avg_loss, accuracy

def run_training(model, train_loader, val_loader, config, device="cpu") -> TrainingHistory:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        ...  # train_one_epoch + evaluate 반복
    return history
```

`train_one_epoch`/`evaluate`는 요청대로 최소 단위로 분리했다.
`run_training`은 이 둘을 `config.epochs`만큼 반복하고 Adam optimizer를
1회 생성하는 얇은 조립 함수다 -- E2E 스크립트와 여러 테스트가 "N epoch
학습 + 매 epoch validation"을 반복 구현하지 않도록 추가했을 뿐, 새로운
개념(scheduler, resume 등)은 없다.

`model.to(device)`는 호출자 책임이다 (loop 함수는 배치 텐서만
`device`로 옮긴다) -- Phase 0/1 스크립트들이 이미 이 관례를 쓰고 있어
그대로 따랐다.

## 6. Weight 저장/재로드

```python
def save_state_dict(model, path) -> None: ...
def load_state_dict(model, path, *, map_location="cpu") -> nn.Module: ...
```

`state_dict`만 저장한다 (optimizer state, epoch 번호, resume 기능 없음
-- 요청 사항). `torch.load(..., weights_only=True)`를 사용해
`run_phase1_e2e.py`가 이미 쓰던 안전한 로드 방식과 동일하게 맞췄다.
`tests/training/test_checkpoint.py`에서 "다른 초기값으로 만든 새
모델 -> load 전엔 출력이 다름 -> load 후엔 원본과 동일한 출력" 순서로
대조군을 포함해 검증한다 (load가 실제로 가중치를 바꾼다는 것 자체를
증명하기 위함).

## 7. Phase 1~3 모델의 실제 학습 검증

`tests/training/test_model_definition_integration.py`에서 지금까지
`model.eval()`로만 검증되던 것들을 처음으로 `model.train()` +
`loss.backward()` 경로로 확인한다:

* **BatchNorm running stats 갱신**: `train_one_epoch()` 전후로
  `running_mean`/`running_var`이 실제로 바뀌는지 비교
* **Dropout train/eval 전환** (`p=0.5`): `model.train()`에서 동일한
  seed를 다시 설정하면 동일한 결과가 재현되는지, `model.eval()`에서는
  dropout이 비활성화되어 seed와 무관하게 항상 결정적인 출력이 나오는지
  확인 -- "서로 다른 seed는 서로 다른 출력을 만든다"는 식의 확률적
  가정에 기대지 않고, 재현성/결정성이라는 검증 가능한 사실만으로
  구성했다
* **ResidualBlockSpec backward**: `in_channels != out_channels`로
  projection shortcut 경로를 강제한 모델에서 모든 파라미터에
  `.grad`가 채워지고 실제로 갱신되는지 확인
* **BranchSpec + IdentitySpec backward**: `merge="add"`의 한쪽
  branch가 파라미터 없는 `IdentitySpec()`이어도 다른 branch(Conv+BN)로
  gradient가 정상적으로 흐르는지 확인 -- 이번 검증 중 가장 까다로운
  경계 케이스라고 판단해 별도 테스트로 분리했다

## 8. Training E2E 스크립트

`scripts/run_training_e2e.py`는 `scripts/run_phase1_e2e.py`와 동일한
구조(단계별 PASS/FAIL 출력, `--model-json` 인자, 자동 빌드 폴백)를 따르되
학습 단계를 추가했다:

```text
Model JSON -> load_model_spec/validate_model_spec (변경 없음)
    -> build_model() (변경 없음)
    -> make_train_val_datasets() (신규)
    -> run_training() (신규)
    -> save_state_dict() (신규)
    -> build_model() 새 인스턴스 + load_state_dict() (신규)
    -> .eval()
    -> TorchScriptExporter().export() (Phase 0, 변경 없음)
    -> run_torchscript.exe (Phase 0, 변경 없음)
    -> Python/C++ parity (Phase 0, 변경 없음)
```

기본 예시 모델(`examples/models/phase4_training_model.json`, 4 classes)은
`Conv-BN-ReLU-ResidualBlock-Branch(Conv+BN / Identity)-ReLU-Dropout-
AdaptiveAvgPool-Flatten-Linear` 구조로, Phase 1~3에서 추가된 모든
composite layer(ResidualBlockSpec, BranchSpec, IdentitySpec)와
BatchNorm/Dropout을 전부 한 모델 안에서 실제로 학습시킨다.
`TorchScriptExporter`, `run_and_compare.run_case`/`find_runner_binary`,
C++ `run_torchscript.exe`는 전혀 수정하지 않았다.

## 9. 실제 변경/추가 파일

| 파일 | 변경 내용 |
|---|---|
| `src/image_ai_studio/training/config.py` | 신규 -- `TrainingConfig` |
| `src/image_ai_studio/training/dataset.py` | 신규 -- 합성 데이터셋 |
| `src/image_ai_studio/training/loop.py` | 신규 -- 학습/평가 루프 |
| `src/image_ai_studio/training/checkpoint.py` | 신규 -- state_dict 저장/재로드 |
| `tests/training/*.py` (6개) | 신규 |
| `scripts/run_training_e2e.py` | 신규 |
| `examples/models/phase4_training_model.json` | 신규 |
| `docs/phase4a_training_design.md` | 신규 (이 문서) |

**변경 없음**: `model_definition/*` 전부, `export/*`, `models/*`,
`parity/*`, `tools/run_and_compare.py`, `scripts/run_phase1_e2e.py`,
C++ 코드 전부, `requirements.txt`(신규 의존성 없음).

## 10. 검증 상태

**Python이 없는 환경이라 아래 항목을 실제로 실행하지 못했다.** 코드
재검토(shape 계산 손검산, 기존 패턴과의 일치 여부 대조)로 정적 확인만
했으며, 다음은 Python이 있는 환경에서 실행 확인이 필요하다:

* `pytest` 전체 (신규 `tests/training/*` 포함)
* Phase 0~3 regression (`scripts/run_torchscript_tests.py`,
  `scripts/run_phase1_e2e.py`)
* `python scripts/run_training_e2e.py` (CPU/CUDA)
* training loss 실제 감소 폭, validation accuracy 실측치
* CPU/CUDA C++ parity 실측

미실행 사유는 최종 채팅 보고에도 동일하게 명시한다.
