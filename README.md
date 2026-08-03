# Image AI Studio

## 장기 목표

Image AI Studio는 사용자가 UI에서 이미지 AI 모델을 설계하고, PyTorch로
학습시키고, C++ 추론 환경으로 내보낸 뒤, Python과 C++ 출력 및 성능을
비교할 수 있도록 하는 것을 목표로 합니다.

```text
모델 설계
    -> PyTorch 모델
    -> 학습/체크포인트
    -> C++ 내보내기(export)
    -> Python 추론
    -> C++ 추론
    -> Python/C++ 출력 및 성능 비교
```

## Phase 0

Phase 0은 최종 제품이 아니라 기술 스파이크(technical spike)입니다.

장기 아키텍처에서 가장 위험하고 불확실한 부분을 검증하는 것이 목적입니다.

```text
Python PyTorch 모델
    -> C++ 내보내기
    -> C++ 프로그램에서 로드
    -> 동일한 입력 텐서로 실행
    -> Python과 C++ 출력 비교
```

두 가지 배포 경로를 독립적으로 평가합니다:

1. **TorchScript** (`torch.jit.trace`)

   * 업스트림에서 이미 지원 중단(deprecated)된 상태.
   * 현재 Phase 0에서는 호환성 및 안정적인 C++ 배포 경로로 사용됨.

2. **torch.export + AOTInductor** (`.pt2`)

   * 새로운 C++ 배포 경로로 평가.
   * 가정하지 않고 실제 빌드 및 런타임 동작을 테스트함.

`run_torchscript`와 `run_aoti`는 별도의 CMake 타겟으로 빌드되는
별개의 실행 파일입니다.

AOTInductor 경로의 실패가 TorchScript 경로의 빌드나 테스트를 막아서는
안 되며, 그 반대의 경우도 마찬가지입니다.

---

## Phase 1

Phase 0 결과(`docs/phase0_results.md`)를 바탕으로, Phase 1부터는 C++
배포/추론 경로로 **TorchScript만 사용**합니다. Phase 0에서 확인된
Windows CPU 런타임 종료 시 크래시와 CUDA Compute Capability 제약
때문에 AOTInductor는 신규 Phase 1 작업에서 제외되며, 기존
AOTInductor 코드는 기록용으로만 유지되고 새 코드는 이를 참조하지
않습니다.

Phase 1에서는 앞으로 Image AI Studio의 중심이 될 **Model Definition
Layer**를 구현합니다:

```text
Model Definition
    -> Shape Inference / Validation
    -> PyTorch Model Builder
    -> torch.nn.Module
    -> TorchScript Export
    -> C++ Inference
```

전체 설계(지원 레이어, shape inference, validation, JSON 포맷, 기존
TorchScript exporter와의 연동 방식)는 `docs/phase1_design.md`를
참고하세요. Phase 1에는 PySide6 UI, 학습, IPC, Detection/Segmentation이
포함되지 않습니다.

### Phase 1 E2E 검증

`ModelSpec`으로 정의한 임의의 모델이 실제로 Phase 0의 C++ TorchScript
러너까지 도달하는지 검증합니다 (새 C++ runner나 새 exporter를 만들지
않고 기존 `run_torchscript.exe`/`TorchScriptExporter`를 그대로 재사용):

```text
Model JSON (examples/models/phase1_e2e_model.json)
    -> ModelSpec
    -> TorchScript
    -> C++ Runner (run_torchscript.exe)
    -> Output parity
```

```bash
python scripts/run_phase1_e2e.py
```

`--model-json`으로 다른 `ModelSpec` JSON을 지정하면 같은 검증 흐름을
그대로 다른 모델에 재사용할 수 있습니다 (기본값은
`examples/models/phase1_e2e_model.json`이므로 인자 없이 실행하면
기존과 동일하게 동작합니다):

```bash
python scripts/run_phase1_e2e.py --model-json examples/models/phase1_e2e_alt_model.json
python scripts/run_phase1_e2e.py --model-json path/to/other_model.json
```

C++ 러너가 필요 없는 `pytest`와 달리, `run_phase1_e2e.py`는 빌드된
`run_torchscript` 실행 파일이 필요합니다 (없으면 `scripts/build_torchscript.py`를
자동으로 호출해 빌드합니다).

---

## Phase 2

Phase 1의 Sequential 기반 Model Definition Layer 위에, 내부에 skip
connection이 있는 **`ResidualBlockSpec`**을 추가합니다.
`_SHAPE_HANDLERS`/`_BUILDERS`/`_LAYER_REGISTRY` 레지스트리에 항목만
추가하는 방식으로, 기존 shape inference/builder/serialization 코드는
전혀 바꾸지 않았습니다.

* `ResidualBlockSpec` (Conv-BN-ReLU-Conv-BN + shortcut -> Add -> ReLU)
* identity shortcut(`in_channels == out_channels`, `stride == 1`)과
  projection shortcut(1x1 Conv+BatchNorm) 둘 다 지원
* TorchScript export 및 C++ CPU/CUDA parity로 실제 검증됨

설계와 검증 결과는 `docs/phase2_residual_block_design.md`를 참고하세요.

---

## Phase 3

일반 DAG를 바로 도입하지 않고, 범위를 좁힌 **`BranchSpec`**/**`IdentitySpec`**
으로 사용자가 직접 구성하는 분기/합류를 지원합니다.

* `BranchSpec`: 입력 하나를 N개 병렬 branch로 나눈 뒤 다시 하나로
  합류 (`merge="add"` 또는 `merge="concat"`; concat은 channel 방향만
  지원)
* `IdentitySpec`: skip path를 명시적으로 표현하는 passthrough 레이어
  (빈 branch를 암묵적으로 Identity로 취급하지 않음)
* 즉시 합류하는 분기 구조만 지원 -- ResNet/Inception류 구조는 표현
  가능하지만, 임의 DAG나 long skip, 중첩 `BranchSpec`은 아직 지원하지
  않음
* TorchScript export 및 C++ CPU/CUDA parity로 실제 검증됨

설계와 검증 결과는 `docs/phase3_branch_design.md`를 참고하세요.

---

## Phase 4A / 4B: Training

Phase 0~3이 만든 Model Definition Layer + TorchScript/C++ 배포 경로
위에, 이 프로젝트에서 처음으로 **실제 학습**을 연결합니다:

```text
Model JSON
    -> ModelSpec
    -> build_model()
    -> Train / Validation (synthetic dataset)
    -> best epoch 추적 (validation loss 기준)
    -> TrainingHistory 저장
    -> best epoch의 state_dict 저장
    -> TorchScript export
    -> C++ LibTorch inference
    -> Python/C++ parity
```

**Phase 4A**가 이 흐름을 처음으로 끝까지 연결했고, **Phase 4B**는 "마지막
epoch"이 아니라 "가장 좋았던 epoch"의 가중치를 배포하도록 발전시켰습니다:

* `TrainingConfig`(epochs/batch_size/learning_rate), optimizer=Adam,
  loss=CrossEntropyLoss로 고정 (선택 registry 없음)
* 외부 다운로드 없는 synthetic 이미지 분류 데이터셋 (train/validation
  분리, 고정 seed로 재현 가능, torchvision 미사용)
* `train_one_epoch()` / `evaluate()` / `run_training()`
* BatchNorm running stats 갱신, Dropout train/eval 전환,
  `ResidualBlockSpec`/`BranchSpec`의 backward를 실제 학습 경로로 검증
* validation loss가 strict하게 개선될 때만 **best epoch**로 갱신하고,
  그 시점의 `state_dict`를 메모리에 deep copy로 보존 (`run_training()`은
  파일을 쓰지 않고 `TrainingResult`로 반환 -- 저장은 호출자 책임)
* `TrainingHistory`(epoch별 train/val loss, val accuracy, best epoch)
  JSON 저장/재로드
* **마지막 epoch이 아니라 best epoch의 가중치**로 state_dict 저장,
  TorchScript export, C++ CPU/CUDA parity까지 확인

실행:

```bash
pytest tests/training/
python scripts/run_training_e2e.py
```

실제로 검증됨(이 저장소에서 직접 실행 확인): `tests/training/` 38 passed,
전체 `pytest` 195 passed, Phase 0 regression / Phase 1~3 E2E regression /
Phase 4B Training E2E 전부 PASS, best epoch 모델 기준 C++ CPU/CUDA
parity PASS.

설계 배경은 `docs/phase4a_training_design.md`를 참고하세요.

---

## Phase 4C: 실제 이미지 데이터셋 (torchvision / CIFAR-10)

Phase 4A/4B는 synthetic dataset(외부 다운로드 없는 랜덤 패턴 이미지)만
사용했습니다. Phase 4C는 이 학습 루프를 처음으로 **실제 이미지
dataset**(torchvision `CIFAR10`)에 연결합니다:

```text
Model JSON
    -> ModelSpec
    -> build_model()
    -> torchvision CIFAR-10 (공식 train split -> Train/Validation 결정론적 분리)
    -> 기존 학습 루프(train_one_epoch/evaluate/run_training, 변경 없음)
    -> best epoch 추적 (변경 없음)
    -> 공식 CIFAR-10 test split으로 best model 최종 평가 (신규)
    -> TorchScript export -> C++ LibTorch inference -> Python/C++ parity
```

* `torchvision`을 사용하되 CIFAR-10 전용 구조로 만들지 않았습니다 --
  `build_transform()`/`limit_dataset()`은 다른 torchvision dataset
  (`ImageFolder` 등)에도 그대로 재사용 가능한 형태입니다.
* CIFAR-10 공식 train 50,000을 고정 seed로 Train(45,000)/
  Validation(5,000)으로 결정론적으로 재분리하고, 공식 test(10,000)는
  완전히 별도로 유지해 **best epoch 선택이나 학습 중 어떤 판단에도 test
  split을 사용하지 않습니다**. best epoch 확정 후, 그 model 하나에 대해
  test split으로 딱 한 번만 최종 평가합니다.
* 전처리는 `Resize(ModelSpec.input_shape) -> ToTensor -> Normalize`뿐이며
  augmentation은 포함하지 않습니다 (Train/Validation/Test 전부 동일한
  deterministic transform).
* `ModelSpec.input_shape[0] != 3`(RGB가 아님)이면 명확한 `ValueError`로
  거부합니다 -- grayscale을 억지로 맞추지 않습니다.
* `train_one_epoch`/`evaluate`/`run_training`/`TrainingHistory`/
  `TrainingResult`/state_dict 저장/`TorchScriptExporter`/C++ 러너는
  Phase 4A/4B 코드를 그대로 재사용합니다. `scripts/run_training_e2e.py`
  (synthetic)는 이번 Phase에서 전혀 수정하지 않았습니다.

실행:

```bash
pytest tests/training/test_torchvision_dataset.py
python scripts/run_real_training_e2e.py
```

`run_real_training_e2e.py`는 처음 실행 시 `--data-root`(기본값
`artifacts/datasets/cifar10`)에 CIFAR-10을 다운로드합니다 (네트워크
필요). `--train-limit`/`--val-limit`/`--test-limit`(기본 256/64/128)으로
빠른 검증용 subset 크기를 조절할 수 있고, `0` 이하를 넘기면 전체 공식
split을 사용합니다.

실제로 검증됨(이 저장소에서 직접 실행 확인, subset
train=256/val=64/test=128/epochs=5): training loss 2.3558 -> 2.0817,
best epoch 4(best val loss 2.1933), best model 기준 test_loss=2.1608 /
test_accuracy=0.1953, best model save/reload PASS, TorchScript export
PASS, C++ CPU/CUDA parity PASS. 이 수치는 작은 subset/짧은 학습에서 나온
것이라 벤치마크 성능이 아니라 "실제 이미지 dataset 경로가 끝까지
연결되어 동작한다"는 것을 보여주는 결과입니다.

설계 배경과 상세 검증 결과는 `docs/phase4c_real_dataset_design.md`를
참고하세요.

---

## Phase 4D: 사용자 ImageFolder 데이터셋

Phase 4C는 torchvision에 내장된 특정 dataset(CIFAR-10)만 연결했습니다.
Phase 4D는 **특정 내장 dataset에 의존하지 않고, 사용자가 준비한 일반
이미지 폴더**를 `torchvision.datasets.ImageFolder`로 읽어 같은
파이프라인에 연결합니다:

```text
사용자 이미지 폴더 (train/val/test로 이미 분리됨)
    -> ImageFolder (3 split)
    -> class_to_idx 일치 검증
    -> dataset 클래스 수 vs ModelSpec 최종 출력 shape 검증
    -> 기존 학습 루프(변경 없음)
    -> best epoch 추적 (변경 없음)
    -> class mapping JSON 저장/재로드 (신규)
    -> best model의 test split 최종 평가 (Phase 4C와 동일 패턴)
    -> TorchScript export -> C++ LibTorch inference -> Python/C++ parity
```

요구되는 폴더 구조 (사용자가 이미 train/val/test로 분리해 둔 상태만
지원, **자동 split은 미지원**):

```text
dataset_root/
├─ train/
│  ├─ cat/
│  └─ dog/
├─ val/
│  ├─ cat/
│  └─ dog/
└─ test/
   ├─ cat/
   └─ dog/
```

* 클래스는 `ImageFolder`가 폴더 이름에서 자동으로 찾습니다
  (하위 폴더 하나 = 클래스 하나).
* `train`/`val`/`test`의 `class_to_idx`가 완전히 일치하는지 학습 시작
  전에 검증합니다. 클래스가 하나라도 다르면(빠지거나 더 있으면) 어느
  split에 어떤 클래스가 다른지 보여주는 에러로 즉시 실패합니다.
* dataset이 실제로 찾은 클래스 수와 `ModelSpec` 최종 출력 shape가
  다르면(`dataset has N classes but model output shape is (M,)`) 학습
  시작 전에 실패합니다.
* Phase 4C의 `build_transform()`(Resize/ToTensor/Normalize, augmentation
  없음)을 그대로 재사용해 Train/Validation/Test 전부 동일한
  deterministic transform을 씁니다. `ModelSpec.input_shape[0] != 3`이면
  거부합니다 (RGB 계약, Phase 4C와 동일). 이 계약은 모델 입력 채널
  수에 대한 것이며, 원본 이미지 파일 자체는 `ImageFolder`의 기본
  loader가 항상 `convert("RGB")`로 읽으므로 grayscale/alpha(RGBA) 원본
  이미지도 자동으로 3채널이 됩니다 (이 프로젝트가 구현한 기능이
  아니라 torchvision의 기본 동작입니다).
* class 이름/인덱스 매핑을 `artifacts/training/{model_name}_classes.json`
  으로 저장합니다 (best model과 함께 inference에 필요한 metadata).

실행:

```bash
python scripts/prepare_cifar10_imagefolder_fixture.py
python scripts/run_imagefolder_training_e2e.py --dataset-root path/to/dataset --model-json examples/models/phase4c_cifar10_model.json
```

`prepare_cifar10_imagefolder_fixture.py`는 제품 기능이 아니라, 별도
개인/사내 dataset 없이도 Phase 4D 경로를 검증할 수 있도록 CIFAR-10
일부를 일반 `ImageFolder` 구조로 export하는 테스트 준비 전용
스크립트입니다 (수동 실행 전용, pytest에서 호출되지 않음).

실제로 검증됨(이 저장소에서 직접 실행 확인, CIFAR-10 10 classes,
train=200/val=50/test=50, epochs=5): training loss 2.3903 -> 2.1509,
best epoch 5(best val loss 2.1269), class mapping 저장/재로드 PASS,
best model 기준 test_loss=2.1859 / test_accuracy=0.2600, TorchScript
export PASS, C++ CPU/CUDA parity PASS. 자동 split과 augmentation은
아직 지원하지 않습니다.

설계 배경과 상세 검증 결과는 `docs/phase4d_imagefolder_design.md`를
참고하세요.

---

## Phase 4E: TrainingConfig 확장 (optimizer / scheduler / early stopping)

Phase 4A~4D는 optimizer=Adam, scheduler 없음, early stopping 없음으로
전부 고정되어 있었습니다. Phase 4E는 이 학습 경로에 사용자가 지정할 수
있는 최소한의 학습 설정을 추가합니다:

* `TrainingConfig.optimizer`: `"adam"`(기본값) | `"sgd"`(`momentum` 함께 지정)
* `TrainingConfig.lr_scheduler`: `None`(기본값, scheduler 없음) |
  `"plateau"`(`ReduceLROnPlateau`, `lr_scheduler_factor`/
  `lr_scheduler_patience` 지정)
* `TrainingConfig.early_stopping_patience`: `None`(기본값, 비활성화) |
  양의 정수 -- validation loss가 N회 연속 개선(strict `<`)되지 않으면
  N번째 epoch를 완료한 뒤 학습을 중단합니다
* `TrainingHistory.stopped_early: bool` -- early stopping으로 중단됐는지
  기록 (기본값 `False`, 기존 history JSON에 이 키가 없어도 하위 호환)

새 필드는 전부 기존 동작(Adam, scheduler 없음, early stopping 없음)을
그대로 재현하는 기본값을 가지므로, 기존 `TrainingConfig(epochs=...,
batch_size=..., learning_rate=...)` 호출은 코드 수정 없이 그대로
동작합니다. `run_training()`의 외부 시그니처(인자/반환 타입)도 변경되지
않았습니다.

`scripts/run_imagefolder_training_e2e.py`에 `--optimizer`/`--momentum`/
`--lr-scheduler`/`--lr-scheduler-factor`/`--lr-scheduler-patience`/
`--early-stopping-patience` CLI 플래그를 추가했습니다 (전부 생략하면
기존 동작 재현). 예:

```bash
python scripts/run_imagefolder_training_e2e.py \
    --optimizer sgd --momentum 0.9 \
    --lr-scheduler plateau --lr-scheduler-factor 0.5 --lr-scheduler-patience 1 \
    --early-stopping-patience 3
```

`scripts/run_training_e2e.py`(Phase 4A/4B 회귀 앵커)와
`scripts/run_real_training_e2e.py`(Phase 4C CIFAR-10)는 이번 Phase에서
수정하지 않았습니다.

실제로 검증됨(이 저장소에서 직접 실행 확인): 기본 설정으로 재실행한
Phase 4A/4B synthetic E2E, Phase 4C CIFAR-10 E2E, Phase 4D ImageFolder
E2E 전부 기존과 완전히 동일한 수치 재현. SGD + `ReduceLROnPlateau` +
early stopping(patience=3) 조합으로 Phase 4D ImageFolder E2E를 실행해
TorchScript export/C++ CPU·CUDA parity까지 PASS 확인. Early
stopping의 정확한 중단 시점(off-by-one 경계)과 `ReduceLROnPlateau`가
실제로 LR을 줄이는 호출 순번은 결정론적 unit test로 별도 고정했습니다.

loss function 선택, Adam betas/weight decay, `"plateau"` 외 scheduler,
full checkpoint/resume은 이번 Phase에서도 지원하지 않습니다.

설계 배경과 상세 검증 결과는
`docs/phase4e_training_config_design.md`를 참고하세요.

---

## Phase 4F: Full Checkpoint + Resume

Phase 4A~4E는 `save_state_dict()`/`load_state_dict()`로 모델 가중치만
저장/재로드할 수 있었습니다 -- optimizer momentum, LR scheduler 진행
상태, epoch 카운터, `TrainingHistory`, early stopping 카운터는
`run_training()` 호출이 끝나면 사라졌습니다. Phase 4F는 이 상태를 전부
저장해 **중단된 학습을 이어서(resume) 실행**할 수 있게 합니다.

핵심 계약은 "optimizer state를 저장하는가"가 아니라 "재개한 학습이
중단 없이 연속 실행한 학습과 어디까지 같은가"입니다:

* epoch 경계에서만 checkpoint 저장 가능 (`run_training()` 호출이
  끝난 뒤 caller가 저장 -- 매 epoch 자동 저장/callback 없음)
* `TrainingConfig.epochs`는 resume 여부와 무관하게 "이번 호출에서
  추가로 실행할 epoch 수"를 뜻합니다. completed epoch 수는 별도
  필드가 아니라 `len(history.train_losses)`로 계산합니다
* optimizer/learning_rate/momentum/lr_scheduler/lr_scheduler_factor/
  lr_scheduler_patience/batch_size는 resume 시 checkpoint와 반드시
  일치해야 합니다 (`optimizer.load_state_dict()`가 저장된 param
  group 값을 그대로 복원하므로, 다른 값을 줘도 조용히 무시됩니다).
  `epochs`/`early_stopping_patience`는 자유롭게 바꿀 수 있습니다.
  이 검증은 caller 관례가 아니라 `run_training()`이 `resume_state`를
  받을 때 항상 스스로 강제하는 core API 계약입니다
* checkpoint **파일 자체**는 `stopped_early=True`여도
  `load_training_checkpoint()`로 정상 조회/가중치 추출이 가능합니다 --
  거부되는 것은 **resume 실행**뿐입니다(early stopping이 이미 학습
  종료를 결정한 상태이므로). 그 가중치로 새로 학습하려면:
  ```python
  payload = load_training_checkpoint(path)
  model.load_state_dict(payload["best_state_dict"])  # nn.Module.load_state_dict()
  ```
  로 가중치를 적용한 뒤 새 `TrainingConfig`로 학습을 시작하세요
  (`training.checkpoint.load_state_dict(model, path)` helper는 파일
  경로를 받는 별개의 함수이므로 혼동하지 마세요)
* DataLoader shuffle generator state와 CPU RNG state를 함께 복원해야
  exact resume이 됩니다 -- 이 둘은 서로 다른 상태입니다(전자는
  로컬 `torch.Generator`, 후자는 `nn.Dropout`이 쓰는 전역 RNG)
* CUDA RNG state, batch-level(worker/sampler) resume은 지원하지
  않습니다 -- 학습이 CPU 전용으로 고정되어 있고(`device="cpu"`),
  `num_workers=0`이라 애초에 필요하지 않습니다

실행:

```bash
python scripts/run_resume_training_e2e.py
```

이 스크립트는 synthetic dataset과 Dropout이 포함된 기존 모델
(`examples/models/phase4_training_model.json`)로 "연속 5 epoch 실행"과
"3 epoch 실행 + checkpoint 저장/로드 + 2 epoch resume"을 비교합니다.
TorchScript export/C++ parity는 다시 수행하지 않습니다(다른 E2E가 이미
검증). 기존 3개 E2E 스크립트는 이번 Phase에서도 수정하지 않았습니다.

실제로 검증됨(이 저장소에서 직접 실행 확인): model parameters,
optimizer state, scheduler state, `TrainingHistory`(train_losses/
val_losses/val_accuracies), `best_state_dict`, `best_epoch`,
`best_val_loss`, `epochs_without_improvement` **10개 항목 전부** 연속
실행과 resume 실행이 정확히 일치(`torch.equal`/정확한 `==`)함을
확인했습니다. 기존 Phase 4A/4B/4C/4D E2E도 기본 설정으로 재실행해
기존과 완전히 동일한 수치가 재현됨을 확인했습니다(회귀 없음).

설계 배경과 상세 검증 결과는
`docs/phase4f_checkpoint_resume_design.md`를 참고하세요.

---

## Phase 4G: ImageFolder Resume CLI Integration

Phase 4F는 core 수준(라이브러리 함수)에서만 checkpoint/resume을
제공했습니다. Phase 4G는 이 기능을 실제 ImageFolder 학습 CLI에
연결합니다: `--epochs`(신규 -- 이전에는 CLI로 바꿀 수 없었음),
`--resume-from PATH`, `--checkpoint-out PATH` 세 옵션을 추가했습니다.

> **Phase 4H 갱신**: 이 절이 설명하는 CLI 옵션(`--epochs`/`--resume-from`/
> `--checkpoint-out` 등)은 Phase 4H에서 `scripts/train_imagefolder.py`로
> 옮겨졌습니다. `run_imagefolder_training_e2e.py`는 더 이상 이 옵션들을
> 지원하지 않고 회귀 검증 전용으로 재구성됐습니다 -- 아래 "Phase 4H:
> Production ImageFolder Training CLI Separation" 절을 참고하세요. 이
> 절의 나머지 내용(metadata 검증 범위, checkpoint 저장 시점, resume
> 시작점 등 core 계약)은 옮겨간 뒤에도 동일하게 유지됩니다.

```bash
# 새로 학습 + checkpoint 저장 (Phase 4H부터는 scripts/train_imagefolder.py)
python scripts/train_imagefolder.py --model-json m.json --dataset-root d --output-dir out --epochs 3 --checkpoint-out out/checkpoint.pt

# 이어서 2 epoch 더 (--epochs는 "총 epoch"가 아니라 "이번에 추가로
# 실행할 epoch 수"이다 -- Phase 4F 계약을 CLI에서도 그대로 유지)
python scripts/train_imagefolder.py --model-json m.json --dataset-root d --output-dir out --epochs 2 --resume-from out/checkpoint.pt --checkpoint-out out/checkpoint.pt
```

핵심 설계 사항(옮겨간 뒤에도 유지되는 core 계약):

* checkpoint(.pt)만으로는 resume할 수 없습니다 -- ImageFolder 전용
  metadata(`<checkpoint>.meta.json`, ModelSpec 해시 + class_to_idx +
  split별 크기/파일 목록 해시)가 항상 같이 저장/검증됩니다. 별도
  플래그 없이 checkpoint 경로로부터 자동 유도됩니다
  (`checkpoint.pt` -> `checkpoint.pt.meta.json`). 이 metadata는 Phase 4F의
  checkpoint 포맷(`CHECKPOINT_FORMAT_VERSION`)과 독립적인 별도 JSON
  파일입니다 -- dataset-agnostic한 core checkpoint 포맷을 건드리지
  않습니다
  * 검증 범위는 class_to_idx + train/val/test 크기 + 파일별
    (상대경로, class_index) 목록 해시입니다. **이미지 내용 자체는
    해싱하지 않습니다** -- 같은 경로의 파일 내용만 바뀌는 경우는
    탐지하지 못하는 것이 알려진 한계입니다. dataset 루트의 절대경로
    일치는 요구하지 않으므로, dataset을 다른 머신/디렉터리로 옮겨도
    resume할 수 있습니다
  * `--resume-from`과 `--checkpoint-out`이 같은 경로여도 정상
    동작합니다(반복 resume의 일반적인 사용 패턴). 다만 checkpoint와
    metadata는 독립된 두 파일이라, 저장 도중 프로세스가 중단되면
    한쪽만 갱신된 채로 남을 수 있습니다 -- 두 파일을 하나의 atomic
    연산으로 묶는 것은 이번 Phase 범위 밖입니다
* checkpoint 저장은 `best_model`(별도 인스턴스)이 만들어지기 **전**에
  일어납니다 -- `model`(현재/마지막 epoch 가중치)이 best 가중치로
  덮어써지는 시점이 스크립트 구조상 존재하지 않으므로,
  best_state_dict를 현재 모델로 착각해서 저장하는 버그가 애초에
  발생할 수 없습니다
* resume 시 checkpoint의 `model_state_dict`(현재/마지막 epoch)를 쓰고,
  `best_state_dict`(최고 성능 epoch)는 쓰지 않습니다 -- 후자를 쓰면
  resume 시작점이 어긋납니다
* `stopped_early=True`인 checkpoint는 (Phase 4F와 동일하게) resume
  실행은 거부되지만 조회/가중치 추출은 여전히 가능합니다. 이 CLI가
  스스로 early stopping으로 멈추고 `--checkpoint-out`이 주어진
  경우에도 checkpoint는 그대로 저장됩니다(이후 resume만 거부됨)

`--resume-from` 없이 실행하면 기존 동작과 완전히 동일합니다(하위
호환). 설계 배경과 상세 검증 결과는
`docs/phase4g_imagefolder_resume_design.md`를 참고하세요.

---

## Phase 4H: Production ImageFolder Training CLI Separation

Phase 4E/4G를 거치며 `run_imagefolder_training_e2e.py`는 원래 "E2E 회귀
검증 스크립트"였다가 점점 일반 사용자 학습 CLI(optimizer/scheduler/
resume 옵션까지 갖춘)로 성장했습니다. Phase 4H는 이 두 책임을 분리합니다:

```text
scripts/train_imagefolder.py            실제 사용자용 production 학습 CLI
scripts/run_imagefolder_training_e2e.py 회귀 검증 + TorchScript/C++ parity 전용 E2E
src/image_ai_studio/training/imagefolder_workflow.py
    "학습 본질" 로직(ModelSpec/dataset 검증, model build/resume, 학습 실행,
    checkpoint/history/best model/class mapping/test 결과 저장, TorchScript
    export)을 담은 공통 모듈 -- 위 두 스크립트가 서로 import하지 않고
    이 모듈만 향해 의존합니다.
```

`train_imagefolder.py` 사용법:

```bash
# 새로 학습
python scripts/train_imagefolder.py \
    --model-json my_model.json --dataset-root path/to/dataset \
    --epochs 20 --batch-size 32 --learning-rate 5e-4 \
    --output-dir artifacts/my_run --checkpoint-out artifacts/my_run/checkpoint.pt

# 이어서 학습
python scripts/train_imagefolder.py \
    --model-json my_model.json --dataset-root path/to/dataset \
    --epochs 10 --batch-size 32 --learning-rate 5e-4 \
    --output-dir artifacts/my_run \
    --resume-from artifacts/my_run/checkpoint.pt --checkpoint-out artifacts/my_run/checkpoint.pt

# TorchScript export 생략 (가중치만 필요한 경우)
python scripts/train_imagefolder.py --model-json my_model.json --dataset-root ... \
    --output-dir artifacts/my_run --no-export-torchscript
```

`--model-json`/`--dataset-root`/`--output-dir` 세 개는 필수입니다(기본값
없음 -- CIFAR-10 fixture를 실수로 학습하는 걸 막기 위해 의도적으로
기본값을 두지 않았습니다). `--epochs`/`--batch-size`/`--learning-rate`/
`--optimizer`/`--momentum`/`--lr-scheduler*`/`--early-stopping-patience`/
`--resume-from`/`--checkpoint-out`은 기존과 동일하게 동작합니다.
`--batch-size`/`--learning-rate`는 Phase 4H에서 새로 노출됐습니다(이전에는
하드코딩이었습니다). `--seed`도 새로 노출되지만, **resume 시에는 사실상
무시됩니다** -- model은 곧바로 checkpoint의 가중치로 덮어써지고,
DataLoader shuffle 순서와 CPU RNG는 checkpoint에 저장된 상태로 복원되기
때문입니다.

`train_imagefolder.py`가 의도적으로 하지 않는 것:

* **C++ parity를 실행하지 않습니다** -- 빌드된 러너 바이너리나 CUDA
  가용성에 전혀 의존하지 않습니다. TorchScript export는 순수 Python
  (`torch.jit.trace`)이라 기본 포함되며 `--no-export-torchscript`로 끌 수
  있습니다. 끄면 같은 `--output-dir`에 이전 실행이 남긴 TorchScript
  산출물(`model.ts`/`model_metadata.json`)도 함께 정리해 최신 실행
  상태와 디렉터리 내용이 항상 일치하도록 합니다(다른 파일은 건드리지
  않습니다)
* **loss가 실제로 줄었는지, class mapping/best model이 저장 후 다시
  읽어도 같은 값을 내는지를 자동으로 판정하지 않습니다** -- 이런
  자체 검증은 이미 단위 테스트(`tests/training/`)로 커버된 불변조건을
  매 실행마다 다시 확인하는 것과 같아서 production 경로에서는 제거했고,
  회귀 검증이 실제로 필요한 `run_imagefolder_training_e2e.py`에만
  남아 있습니다
* **artifact 경로를 하드코딩하지 않습니다** -- `--output-dir` 아래
  고정 파일명(`best_model_state_dict.pt`, `training_history.json`,
  `class_mapping.json`, `test_result.json`, 선택적으로 `model.ts`/
  `model_metadata.json`)으로만 저장합니다. 기존 파일은 확인 없이
  덮어씁니다(이 프로젝트의 다른 저장 함수들과 동일한 정책)

재구성된 `run_imagefolder_training_e2e.py`는 CIFAR-10 ImageFolder
fixture로 `run_imagefolder_training_workflow()`를 고정 설정(fresh 3
epoch + checkpoint 저장, 이어서 resume 2 epoch)으로 두 번 호출한 뒤,
loss 감소 게이트 + class mapping 재검증 + TorchScript reload + C++
CPU/CUDA parity를 자체 책임으로 수행합니다 -- 아래 "ImageFolder
Training E2E" 절 참고. best model save/reload 재검증은 두지 않습니다
(같은 파일을 두 번 읽어 비교하는 것은 항상 같은 결과만 나오는 무의미한
검증이라 회귀 가치가 낮다고 판단해 제거했습니다 -- state_dict 저장/
재로드 정확성은 `tests/training/test_imagefolder_workflow.py`/
`test_checkpoint.py`가 단위 테스트로 이미 커버합니다). regression anchor
수치(best_epoch/best_val_loss/test_accuracy 등)는 스크립트가 출력만
하고 자동 실패 조건으로 삼지 않습니다 -- 환경/PyTorch 버전에 따라
소수점 마지막 자리가 흔들릴 수 있는 값을 엄격한 자동 gate로 만들지
않기 위한 의도적 선택입니다. 설계 배경과 상세 검증 결과는
`docs/phase4h_production_training_cli_design.md`를 참고하세요.

---

## 현재 지원 범위

* Sequential 기반 Model Definition (`ModelSpec`/`LayerSpec`, JSON
  직렬화)
* `ResidualBlock` (Phase 2)
* `Branch` Add / channel `Concat` + `Identity` skip path (Phase 3)
* Classification 학습, loss=CrossEntropyLoss 고정, optimizer는 Adam/SGD
  선택 가능 (Phase 4E)
* LR scheduler `ReduceLROnPlateau` 선택, early stopping patience 지정,
  `TrainingHistory.stopped_early` 기록 (Phase 4E)
* Synthetic train/validation 데이터셋 (Phase 4A/4B, 오프라인)
* torchvision CIFAR-10 real-image 데이터셋, 공식 train split의 결정론적
  Train/Validation 재분리, 공식 test split의 최종(1회) 평가 (Phase 4C)
* 사용자가 준비한 `ImageFolder` 폴더(train/val/test로 이미 분리된
  구조) 학습, class_to_idx 일치 검증, dataset 클래스 수와 `ModelSpec`
  출력 shape 일치 검증, class mapping JSON 저장 (Phase 4D)
* Best epoch 모델 추적 + `TrainingHistory` JSON 저장
* Full training checkpoint(model/optimizer/scheduler state, history,
  best model, early stopping 카운터, DataLoader generator/CPU RNG
  state) 저장과 epoch 경계 resume, CPU 학습 경로에서 연속 실행과
  tensor-level exact equality 목표 (Phase 4F)
* `--resume-from`/`--checkpoint-out`으로 Phase 4F checkpoint/resume 실행,
  ImageFolder 전용 metadata(ModelSpec 해시 + class_to_idx + 파일 목록
  해시)로 dataset/model 호환성 자동 검증 (Phase 4G)
* 실제 사용자용 production 학습 CLI(`scripts/train_imagefolder.py`)와
  회귀 검증 전용 E2E(`scripts/run_imagefolder_training_e2e.py`)의 책임
  분리, `--batch-size`/`--learning-rate`/`--output-dir`/`--seed`/
  `--export-torchscript` 신규 노출, production CLI는 C++ parity를
  실행하지 않음 (Phase 4H)
* TorchScript 배포, C++(LibTorch) CPU/CUDA 추론
* Python/C++ parity 검증 (Phase 0~4E 배포 경로; Phase 4F는 export/parity
  코드를 변경하지 않았고, 기존 parity E2E를 재실행해 회귀 없음을 확인함
  -- Phase 4F의 resume 기능 자체는 C++에서 실행/검증되지 않음)

## 아직 미지원 / 향후 계획

다음은 아직 구현되지 않았습니다:

* augmentation (RandomCrop, RandomHorizontalFlip, ColorJitter,
  RandAugment, AutoAugment 등)
* `ImageFolder` 폴더의 자동 Train/Val/Test split (train/val/test로
  이미 분리된 구조만 지원 -- 클래스 폴더만 있는 구조를 자동으로
  나누는 기능은 없음)
* dataset registry/factory를 통한 통합 연동 (CIFAR-10과 `ImageFolder`가
  각각 별도 함수로 연결되어 있고, 둘을 묶는 공통 factory/registry는
  아직 없음), Oxford-IIIT Pet 등 다른 dataset의 실제 연동
* class imbalance 처리, weighted sampler
* loss function 선택 (CrossEntropyLoss 고정), Adam betas/weight decay,
  SGD dampening/nesterov, `"plateau"` 외 LR scheduler(StepLR/
  CosineAnnealingLR 등), scheduler threshold/cooldown/min_lr
* resume 시 config 자유 변경 (optimizer/learning_rate/momentum/
  lr_scheduler/lr_scheduler_factor/lr_scheduler_patience/batch_size는
  checkpoint와 반드시 일치해야 함), CUDA RNG state 저장(학습이 CPU
  전용으로 고정되어 있어 검증 불가), batch-level(worker/sampler
  iterator) resume, epoch 중간/자동 checkpoint, distributed checkpoint
* mixed precision, multi-GPU/distributed training
* 일반 DAG(`GraphSpec`/`NodeSpec`/`EdgeSpec`), long skip connection,
  중첩 `BranchSpec`
* Detection/Segmentation training
* PySide6 UI

이 항목들은 구체적인 필요가 확인되기 전까지 의도적으로 보류하고
있습니다 (과설계 방지). 각 Phase가 무엇을 의도적으로 제외했는지는
해당 `docs/phase*.md` 문서의 "의도적으로 구현하지 않은 것"(또는
"미지원") 절에 정리되어 있습니다.

---

## 크로스 플랫폼 범위

주요 대상 환경은 다음과 같습니다:

```text
Windows 11
Visual Studio 2022
MSVC
x64 Release
NVIDIA CUDA GPU
```

C++ 코드는 MSVC 전용 API에 의도적으로 의존하지 않는 표준 C++17로
작성되어 있어, 동일한 CMake 프로젝트를 macOS와 Linux에서도 빌드할
수 있습니다.

### Windows

주요 검증 환경입니다.

```text
CPU 추론
CUDA 추론
TorchScript
AOTInductor
C++ 패리티(parity) 테스트
```

현재 검증 결과에서는 TorchScript CPU/CUDA가 정상 동작했으며,
AOTInductor는 Windows CPU 런타임 종료(teardown) 시 발생하는 문제와
테스트에 사용된 GPU의 Compute Capability 제약이 확인되었습니다.
자세한 내용은 `docs/phase0_results.md`를 참고하세요.

### Linux

동일한 C++ 구현으로 CPU와 CUDA를 모두 지원할 것으로 예상됩니다.

Linux CUDA 검증은 아직 완료되지 않았습니다.

### macOS

CPU 전용 검증 환경입니다.

Apple Silicon은 NVIDIA CUDA를 지원하지 않으므로:

```text
--device cuda
```

는 `UNSUPPORTED`를 반환해야 합니다.

러너(runner)는 절대 조용히 CPU로 폴백해서는 안 됩니다.

실제로 실행 및 검증된 환경은 `docs/phase0_results.md`를 참고하세요.

---

## Phase 0에 포함된 내용

* `TinyCNN`
* `TinyResidualCNN`
* 잔차 연결(Residual connection) 테스트
* BatchNorm 실행 통계(running statistics)
* 재현 가능한 테스트 입력 텐서
* 재현 가능한 공유 `state_dict`
* 고정된 랜덤 시드
* SHA-256 아티팩트 체크섬
* Python CPU 참조 출력
* 가능한 경우 Python CUDA 참조 출력
* TorchScript trace 내보내기
* AOTInductor 내보내기
* 내보내기 환경 메타데이터
* 독립적인 C++ TorchScript 러너
* 독립적인 C++ AOTInductor 러너
* CPU FP32 추론
* 가능한 경우 CUDA FP32 추론
* Python/C++ 출력 패리티 비교
* 100회 반복 안정성 테스트
* 추론 타이밍 통계
* GPU 메모리 관측

---

## Phase 0에서 제외된 내용

Phase 0은 다음 항목들을 의도적으로 포함하지 않습니다:

* PySide6 UI
* 모델 그래프 편집기
* 학습 루프
* `ImageFolder` 데이터셋 통합
* 공유 메모리 IPC
* 소켓 IPC
* JSON-Lines IPC
* 장기 실행 워커 프로세스
* 동적 shape(Dynamic shapes)
* ONNX Runtime
* TensorRT
* 객체 탐지(Detection)
* 세그멘테이션(Segmentation)
* 실시간 비디오 처리
* 모델 버전 관리 UI
* Git LFS
* 디버그 빌드
* LibTorch 소스 빌드

---

# 설치(Setup)

## 1. Python 환경 생성

Phase 0에서는 현재 Python 3.11을 사용합니다.

conda 사용 시:

```bash
conda create -n ias python=3.11 pip -y
conda activate ias
```

확인:

```bash
python --version
python -m pip --version
```

---

## 2. 공통 Python 의존성 설치

`requirements.txt`에는 플랫폼에 독립적인 Python 의존성만 포함되어
있습니다.

필요한 PyTorch 빌드가 운영체제와 GPU 환경에 따라 달라지므로,
PyTorch는 의도적으로 `requirements.txt`에 **포함되어 있지 않습니다**.

먼저 공통 의존성을 설치하세요:

```bash
python -m pip install -r requirements.txt
```

현재 `requirements.txt`:

```text
filelock==3.32.0
fsspec==2026.6.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
packaging==26.0
sympy==1.14.0
typing_extensions==4.16.0
```

---

## 3. PyTorch 별도 설치

PyTorch는 대상 환경에 맞게 별도로 설치해야 합니다.

이는 의도된 것입니다.

다음 명령이

```bash
pip install torch
```

원하는 CUDA 지원 빌드를 설치해줄 것이라고 가정하지 마세요.

아래 버전 번호는 설치 예시입니다. 실제 검증에 사용된 PyTorch 버전은
`docs/phase0_results.md`를 참고하세요.

### Windows + NVIDIA CUDA

환경에 맞는 CUDA 지원 PyTorch wheel을 설치하세요.

예시:

```bat
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

Phase 4C(실제 이미지 dataset)를 사용하려면 `torchvision`도 **같은 CUDA
index에서, 설치된 `torch` 버전과 짝이 맞는 버전**으로 설치해야 합니다.
최신 `torchvision`을 그냥 설치하면 `torch`를 다른 버전으로 업그레이드할
수 있으므로, 먼저 `--dry-run`으로 실제 설치될 `torch` 버전을 확인하는
것을 권장합니다:

```bat
python -m pip install torchvision==0.27.1+cu126 --index-url https://download.pytorch.org/whl/cu126 --dry-run
python -m pip install torchvision==0.27.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

`torchvision==0.27.1+cu126`은 이 저장소에서 `torch==2.12.1+cu126`을
바꾸지 않고 그대로 설치된다는 것을 실제로 확인한 조합입니다. 설치 후
`torch.__version__`과 `torch.cuda.is_available()`이 이전과 동일한지
다시 확인하세요.

설치 후 확인:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'None')"
```

CUDA 지원 환경이라면 다음 결과가 중요합니다:

```text
CUDA available: True
```

기본적인 CUDA 연산도 테스트할 수 있습니다:

```bat
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print(y.device); print(y.mean())"
```

### macOS

표준 macOS PyTorch 패키지를 설치하세요:

```bash
python -m pip install torch==2.12.1
```

macOS에서는 CUDA를 사용할 수 없을 것으로 예상됩니다.

Phase 4C를 사용하려면 설치된 `torch` 버전과 짝이 맞는 `torchvision`을
같은 방식으로 설치하세요 (예: `python -m pip install torchvision==0.27.1`).

### Linux + NVIDIA CUDA

대상 Linux CUDA 환경에 맞는 PyTorch CUDA wheel을 설치하세요.

정확한 wheel은 `requirements.txt`에 하드코딩하는 대신, 목표로 하는
PyTorch 및 CUDA 구성에 따라 선택해야 합니다.

Phase 4C를 사용하려면 `torchvision`도 동일한 CUDA index에서 설치된
`torch` 버전과 짝이 맞는 버전으로 설치하세요 (Windows 예시와 동일하게
`--dry-run`으로 먼저 확인하는 것을 권장합니다).

---

## 4. Image AI Studio를 editable 모드로 설치

이 프로젝트는 `src/` Python 패키지 레이아웃을 사용합니다.

프로젝트 자체를 현재 Python 환경에 설치하세요:

```bash
python -m pip install -e .
```

`-e`는 editable(수정 가능) 설치를 의미합니다.

Python이 현재 소스 트리를 직접 참조하므로,

```text
src/image_ai_studio/
```

아래의 변경 사항은 패키지를 매번 재설치하지 않아도 반영됩니다.

확인:

```bash
python -c "import image_ai_studio; print(image_ai_studio.__file__)"
```

경로는 이 저장소의

```text
src/image_ai_studio/
```

디렉터리를 가리켜야 합니다.

---

## 5. 환경 점검

실행:

```bash
python scripts/inspect_environment.py
```

환경 점검 결과에는 다음 정보가 포함되어야 합니다:

```text
Python 버전
PyTorch 버전
PyTorch CUDA 버전
CUDA 사용 가능 여부
GPU 이름
GPU compute capability
CUDA Toolkit
NVIDIA 드라이버
CMake
컴파일러
LibTorch 위치
```

---

# C++ 빌드를 위한 LibTorch

기본 Phase 0 워크플로우에서는 별도의 LibTorch 다운로드가 필요하지
않습니다.

C++ 빌드는 설치된 Python `torch` 패키지에 번들로 포함된 LibTorch
파일을 사용합니다.

CMake 경로는 다음으로 확인할 수 있습니다:

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

이를 통해 Python과 C++ 양쪽이 동일한 PyTorch 설치를 사용하게 되어,

```text
Python PyTorch
C++ LibTorch
```

사이의 버전 불일치를 방지할 수 있습니다.

예를 들어 Windows에 CUDA 지원 PyTorch wheel을 설치하면, C++ 빌드도
동일한 PyTorch 설치에 번들된 LibTorch를 사용하게 됩니다.

---

# Windows C++ 빌드 구성

Phase 0의 주요 C++ 타겟은 다음과 같습니다:

```text
플랫폼: x64
구성: Release
```

디버그 빌드는 Phase 0 범위에서 의도적으로 제외됩니다.

Debug와 Release MSVC CRT/ABI 조합을 혼용해서는 안 됩니다.

호환되지 않는 Debug/Release 조합으로 인한 구성 오류는 TorchScript나
AOTInductor 백엔드 실패가 아니라 다음으로 분류됩니다:

```text
INVALID_BUILD_CONFIGURATION
```

자세한 빌드 지침은 다음을 참고하세요:

```text
docs/build.md
```

---

# AOTInductor 지원 여부 프로브(probe)

전체 AOTInductor C++ 러너를 시도하기 전에, 설치된 LibTorch 배포판에
필요한 AOTInductor C++ 런타임이 포함되어 있는지 확인하세요.

실행:

```bash
python scripts/probe_aoti_support.py
```

그런 다음 독립된 C++ 프로브를 빌드하세요:

```bash
python scripts/build_aoti.py --build-dir build-aoti-probe --probe-only
```

패키지 테스트 예시:

```bash
./build-aoti-probe/cpp/aoti_probe/probe_aoti \
  --package artifacts/aoti/tiny_cnn/cpu/model.pt2 \
  --input-bin artifacts/common/input.bin \
  --input-meta artifacts/common/input.json
```

Windows에서는 실행 파일이 해당하는 `.exe` 경로를 사용합니다.
Visual Studio의 멀티 컨피그 generator를 사용하는 경우 다음과 같이
`Release\` 하위 경로가 될 수 있습니다:

```bat
build-aoti-probe\cpp\aoti_probe\Release\probe_aoti.exe ^
  --package artifacts\aoti\tiny_cnn\cpu\model.pt2 ^
  --input-bin artifacts\common\input.bin ^
  --input-meta artifacts\common\input.json
```

가능한 지원 상태(capability states)는 다음과 같습니다:

```text
HEADER_NOT_FOUND
LIBRARY_OR_SYMBOL_NOT_FOUND
COMPILE_FAILED
LINK_FAILED
PACKAGE_LOAD_FAILED
SUPPORTED
```

AOTInductor 지원 여부 프로브가 실패해도 TorchScript의 빌드나 테스트를
막지 않습니다.

실제 결과는 `docs/phase0_results.md`를 참고하세요.

---

# 테스트 아티팩트 생성

결정론적(deterministic) 모델 가중치와 테스트 입력을 생성하세요:

```bash
python -m image_ai_studio.tools.prepare_test_artifacts
```

Python 참조 출력을 생성하세요:

```bash
python -m image_ai_studio.tools.run_python_reference
```

모델을 내보내세요:

```bash
python scripts/export_models.py
```

이 과정을 통해 C++ 러너에 필요한 아티팩트가 생성됩니다.

---

# C++ 러너 빌드

TorchScript와 AOTInductor는 독립적으로 빌드됩니다.

## TorchScript

```bash
python scripts/build_torchscript.py
```

예상 출력:

```text
build-torchscript/.../run_torchscript
```

Windows에서는:

```text
run_torchscript.exe
```

Visual Studio처럼 멀티 컨피그 generator를 사용하는 Windows에서는
실행 파일이 `Release/` 하위(예: `build-torchscript/.../Release/run_torchscript.exe`)에
생성될 수 있습니다.

## AOTInductor

```bash
python scripts/build_aoti.py
```

예상 출력:

```text
build-aoti/.../run_aoti
```

Windows에서는:

```text
run_aoti.exe
```

마찬가지로 `build-aoti/.../Release/run_aoti.exe` 형태가 될 수 있습니다.

각 백엔드는 자체 빌드 디렉터리를 사용합니다.

AOTInductor 빌드가 깨지더라도 다음에는 영향을 주지 않아야 합니다:

```text
build-torchscript/
```

---

# 테스트 실행

## Model Definition Layer / Training (unit test)

```bash
python -m pip install -r requirements-dev.txt
pytest
```

`tests/model_definition/`(Phase 1~3)과 `tests/training/`(Phase
4A/4B/4C/4D/4E/4F)를 포함한 전체 unit test입니다. 전부 CPU에서 동작하며
빌드된 C++ 러너가 필요 없고, 네트워크 접근도 없습니다 (CIFAR-10
다운로드는 `pytest`가 아니라 아래 Real-Image Training E2E에서만
발생하고, `ImageFolder` 관련 테스트는 `tmp_path` + PIL로 직접 만든
픽스처만 사용합니다). Training 테스트만 따로 돌리려면:

```bash
pytest tests/training/
```

자세한 내용은 `docs/phase1_design.md`, `docs/phase4a_training_design.md`,
`docs/phase4c_real_dataset_design.md`, `docs/phase4d_imagefolder_design.md`,
`docs/phase4e_training_config_design.md`, `docs/phase4f_checkpoint_resume_design.md`를
참고하세요.

## TorchScript (Phase 0 C++ 패리티)

```bash
python scripts/run_torchscript_tests.py
```

## AOTInductor

```bash
python scripts/run_aoti_tests.py
```

CUDA를 사용할 수 있는 경우, 테스트에는 CPU와 CUDA 패리티가 포함됩니다.

CUDA를 사용할 수 없는 경우, CUDA 테스트는 조용히 CPU로 폴백되지 않고
건너뜀(skipped) 또는 미지원(unsupported)으로 보고됩니다.

## Training E2E (Phase 4A/4B)

```bash
python scripts/run_training_e2e.py
```

`ModelSpec` -> 학습 -> best epoch 추적 -> TorchScript export -> C++
CPU/CUDA parity까지 전체 흐름을 실행합니다. `--model-json`으로 다른
`ModelSpec` JSON을 지정할 수 있습니다 (기본값:
`examples/models/phase4_training_model.json`).

## Real-Image Training E2E (Phase 4C)

```bash
python scripts/run_real_training_e2e.py
```

torchvision CIFAR-10을 사용하는 실제 이미지 학습 E2E입니다. 처음
실행 시 `--data-root`(기본값 `artifacts/datasets/cifar10`)에 CIFAR-10을
다운로드하므로 네트워크가 필요합니다. `--train-limit`/`--val-limit`/
`--test-limit`(기본 256/64/128, `0` 이하면 전체 split)으로 실행 시간을
조절할 수 있습니다. `scripts/run_training_e2e.py`(synthetic)는 이
스크립트와 별개로 그대로 유지됩니다.

## ImageFolder Training CLI (Phase 4D~4H)

```bash
python scripts/train_imagefolder.py \
    --model-json my_model.json --dataset-root path/to/dataset \
    --output-dir artifacts/my_run
```

사용자가 준비한 `ImageFolder` 폴더(`train`/`val`/`test`로 이미 분리된
구조)를 학습하는 **실제 사용자용 production CLI**입니다(Phase 4D~4G에서
`run_imagefolder_training_e2e.py`가 맡던 학습 CLI 역할을 Phase 4H에서
이 스크립트로 옮겼습니다). `--model-json`/`--dataset-root`/`--output-dir`
세 개가 필수입니다. `--optimizer`/`--momentum`/`--lr-scheduler`/
`--lr-scheduler-factor`/`--lr-scheduler-patience`/
`--early-stopping-patience`/`--epochs`/`--batch-size`/`--learning-rate`/
`--resume-from`/`--checkpoint-out`/`--seed`/
`--export-torchscript`(`--no-export-torchscript`)를 지원합니다. 전부
생략하면 Adam/scheduler 없음/early stopping 없음/TorchScript export
포함으로 동작합니다. 자세한 내용은 위 "Phase 4H: Production ImageFolder
Training CLI Separation" 절과
`docs/phase4h_production_training_cli_design.md`를 참고하세요.

## ImageFolder Training E2E (Phase 4D, Phase 4H에서 재구성)

```bash
python scripts/prepare_cifar10_imagefolder_fixture.py
python scripts/run_imagefolder_training_e2e.py --dataset-root path/to/dataset --model-json examples/models/phase4c_cifar10_model.json
```

`train_imagefolder.py`가 호출하는 것과 같은 workflow 함수
(`run_imagefolder_training_workflow()`)를 고정 설정(fresh 3 epoch +
checkpoint 저장, 이어서 resume 2 epoch)으로 두 번 호출해 Phase 4D~4G의
전체 계약(학습, checkpoint/resume, metadata 검증)과 TorchScript export +
C++ CPU/CUDA parity를 한 번에 회귀 검증하는 **E2E 전용** 스크립트입니다
-- 더 이상 일반 학습 CLI가 아닙니다(옵션은 `--model-json`/`--dataset-root`
뿐). `--dataset-root`를 생략하면 `prepare_cifar10_imagefolder_fixture.py`가
만드는 `artifacts/datasets/cifar10_imagefolder`를 기본값으로 사용합니다.
`prepare_cifar10_imagefolder_fixture.py`는 제품 기능이 아니라, CIFAR-10
일부를 `ImageFolder` 구조로 export해 이 E2E를 별도 dataset 없이 검증할
수 있게 해주는 테스트 준비 전용 스크립트입니다 (pytest에서 호출되지
않음).

자세한 내용은 `docs/phase4d_imagefolder_design.md`(초기 설계),
`docs/phase4e_training_config_design.md`(TrainingConfig 확장),
`docs/phase4g_imagefolder_resume_design.md`(resume 연결),
`docs/phase4h_production_training_cli_design.md`(production CLI 분리)를
참고하세요.

## Resume Training E2E (Phase 4F)

```bash
python scripts/run_resume_training_e2e.py
```

연속 5 epoch 실행과, 3 epoch 실행 후 full checkpoint 저장/로드를 거쳐
2 epoch를 resume한 실행을 비교해 model/optimizer/scheduler state,
history, best model, early stopping 카운터가 전부 정확히 일치하는지
증명하는 전용 E2E입니다. synthetic dataset과 Dropout이 포함된 기존
모델(`examples/models/phase4_training_model.json`)을 사용하며,
TorchScript export/C++ parity는 다시 수행하지 않습니다(다른 E2E가 이미
검증). 자세한 내용은 `docs/phase4f_checkpoint_resume_design.md`를
참고하세요.

---

# 전체 Phase 0 워크플로우 실행

실행:

```bash
python scripts/run_phase0.py
```

이 워크플로우는 다음을 수행합니다:

```text
환경 점검
    -> AOTInductor 지원 여부 프로브
    -> 결정론적 모델/입력 생성
    -> Python 참조 추론
    -> TorchScript 내보내기
    -> AOTInductor 내보내기
    -> C++ 빌드
    -> C++ 추론
    -> Python/C++ 패리티 비교
    -> 반복 안정성 테스트
    -> 결과 생성
```

AOTInductor 단계에서의 실패는 TorchScript 경로의 진행을 막지 않습니다.

---

# 알려진 제한 사항

* TorchScript 검증은 현재 정적(static) `torch.jit.trace` 경로만
  다룹니다.

* `torch.jit.script`는 Phase 0 범위 밖입니다.

* AOTInductor API는 PyTorch 버전에 따라 달라질 수 있습니다.

* 일부 AOTInductor compile/package/load API는 내부/비공개 모듈인
  `torch._inductor` 아래에 있을 수 있으며, 버전에 민감한 것으로
  취급해야 합니다.

* 현재 공유 바이너리 + JSON 텐서 포맷은 float32 텐서만 지원합니다.

* macOS 검증은 CPU 전용입니다.

* Windows x64 Release + NVIDIA CUDA가 주요 타겟 환경이며, 현재
  검증 결과에서는 TorchScript가 Phase 0의 기본 배포 백엔드로
  권장됩니다 (자세한 내용은 `docs/phase0_results.md`의 "권장 백엔드"
  참고).

* macOS CPU에서 동작하는 백엔드가 Windows MSVC/CUDA 경로도
  지원됨을 자동으로 의미하지는 않습니다.

---

# 의존성 정책

`requirements.txt`에는 의도적으로 PyTorch가 포함되어 있지 않습니다.

의존성 전략은 다음과 같습니다:

```text
requirements.txt
    -> 공통 Python 의존성

PyTorch
    -> OS/GPU 구성별로 별도 설치

pip install -e .
    -> Image AI Studio 소스 패키지 자체를 설치
```

이를 통해 저장소가 하나의 PyTorch CPU/CUDA wheel에 종속되는 것을
방지하고, 플랫폼별 환경을 명시적으로 유지합니다.
