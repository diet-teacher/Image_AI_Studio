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

loss function 선택, Adam betas, `"plateau"` 외 scheduler, full
checkpoint/resume은 이번 Phase에서도 지원하지 않습니다(weight decay와
AdamW는 Phase 4L에서 추가됐습니다 -- 아래 "Phase 4L" 절 참고).

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
  않습니다 -- `num_workers=0`이라 worker/sampler RNG는 애초에 필요하지
  않고, CUDA RNG state는 checkpoint에 아직 저장되지 않으므로 CPU→CPU가
  아닌 조합(Phase 4Q의 CPU↔CUDA 등)은 resume이 정상 동작(portable)은
  하지만 bitwise exact-resume 계약의 대상이 아닙니다(위 "Phase 4Q" 절
  참고)

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

## Phase 4I: Training Progress Callback and Safe Stop

핵심 `run_training()`에 epoch 경계 progress callback과 협조적(cooperative)
stop 메커니즘을 추가했습니다. 둘 다 키워드 전용(keyword-only) 파라미터로,
넘기지 않으면(기본값 `None`) Phase 4H까지의 동작과 완전히 동일합니다:

```python
result = run_training(
    model, train_loader, val_loader, config,
    progress_callback=lambda progress: print(progress.global_epoch, progress.val_loss),
    should_stop=lambda: stop_requested,  # 인자 없이 bool을 반환하는 아무 callable이나 가능 (예: threading.Event().is_set)
)
```

* **`progress_callback`**: epoch이 완료될 때마다(early stopping으로 끝난
  epoch 포함) 정확히 한 번, 읽기 전용 `TrainingProgress` 스냅샷과 함께
  호출됩니다. `TrainingProgress`는 관찰/UI 갱신 전용 지표만 담고
  (`run_epoch`/`total_run_epochs`/`global_epoch`/`train_loss`/`val_loss`/
  `val_accuracy`/`learning_rate`/`best_epoch`/`best_val_loss`/
  `epochs_without_improvement`/`stopped_early`), model/state_dict/
  optimizer/scheduler 객체나 시간 정보는 포함하지 않습니다. 콜백이
  예외를 던지면 그대로 전파되고(감싸지 않음) `TrainingResult`는 반환되지
  않습니다.
* **`should_stop`**: `progress_callback` 호출 직후, 같은 epoch 경계에서
  평가됩니다(콜백 안에서 동기적으로 stop 플래그를 세팅하면 지연 없이
  바로 반영됨). 단, **epoch 시작 전에는 절대 평가되지 않고**(최소 1개
  epoch은 항상 실행됨), **이번 호출의 마지막 요청 epoch이거나
  `config.epochs == 1`이면 평가 자체가 없습니다**(더 이상 건너뛸 epoch이
  없으므로). `True`를 반환하면 `TrainingHistory.stopped_by_user = True`를
  설정하고 멈춥니다.
* **`stopped_by_user`**(새 `TrainingHistory` 필드, 기본값 `False`): 사용자
  요청으로 epoch를 남긴 채 멈췄음을 나타냅니다. `stopped_early`와 달리
  resume이 거부되지 않습니다(사용자가 잠시 멈춘 것뿐이므로) -- 기존
  checkpoint/history JSON은 이 필드가 없어도 기본값으로 채워지며,
  `checkpoint.py`/`history.py`는 변경되지 않았습니다(둘 다 `asdict()`/
  `**dict` 기반으로 완전히 범용적이라 새 필드 추가에 코드 변경이
  필요 없었습니다). resume 시에는 이전 checkpoint의 `stopped_by_user`
  값이 그대로 이어받지 않도록 항상 `False`로 리셋됩니다.
* `run_imagefolder_training_workflow()`도 동일한 키워드 전용 파라미터를
  받아 `run_training()`에 그대로 전달합니다(`ImageFolderWorkflowRequest`
  dataclass 필드가 아님 -- Request의 JSON 직렬화 가능성과 데이터/제어
  흐름 관심사 분리를 유지하기 위함). 사용자 중단으로 끝난 결과도 정상
  완료와 동일하게 checkpoint/history/best model/class mapping/test
  결과/TorchScript export 전체 아티팩트 파이프라인을 거칩니다(별도
  분기 없음).
* `scripts/train_imagefolder.py`는 `progress_callback`으로 epoch마다
  실시간으로 한 줄씩 출력합니다(`progress.global_epoch` 기준). **resume
  실행에서는 새로 완료된 epoch만 찍습니다** -- 과거처럼 누적 history
  전체를 사후 재출력하지 않는 의도된 동작 변경입니다. 새 CLI 플래그(중단
  요청용)는 추가하지 않았습니다 -- `should_stop`은 core API 계약이고,
  CLI에서 실제로 중단을 트리거하는 방법(시그널/파일/등)은 이번 Phase
  범위 밖입니다(Ctrl+C를 이 `should_stop`에 연결하는 실제 트리거는
  Phase 4K에서 구현됐습니다 -- 위 "Phase 4K" 절 참고).
  `scripts/run_imagefolder_training_e2e.py`는 변경하지 않았습니다.

설계 배경, 3라운드 리뷰에서 수정된 논리적 충돌, 17개 동작 계약 테스트
목록은 `docs/phase4i_training_progress_and_stop_design.md`를 참고하세요.

---

## Phase 4J: Epoch-end Automatic Checkpointing and Recovery

`run_training()`에 epoch 경계 `checkpoint_hook`을 추가하고,
`scripts/train_imagefolder.py`에 `--checkpoint-every N`을 노출해서
학습 도중 주기적으로 checkpoint를 자동 저장할 수 있게 했습니다.
Phase 4F까지는 학습이 끝난 뒤 딱 한 번만 checkpoint를 저장했는데,
epoch 수가 많은 학습 중 중간에 실패/중단되면 그 지점까지의 진행이
전부 사라졌습니다.

```bash
python scripts/train_imagefolder.py \
    --model-json m.json --dataset-root d --output-dir out \
    --epochs 20 --checkpoint-out out/checkpoint.pt --checkpoint-every 5
```

* **`--checkpoint-every N`**: global epoch(resume을 포함한 절대 epoch
  번호, `len(history.train_losses)`)이 `N`의 배수가 될 때마다
  `--checkpoint-out` 경로를 자동으로 갱신합니다. **기본값은 꺼짐**
  (`None`)이라 생략하면 Phase 4I까지와 완전히 동일하게 학습 종료
  시에만 저장됩니다. `--checkpoint-out` 없이 이 옵션만 주면
  오류입니다. cadence는 이번 호출 안에서의 hook 호출 횟수가 아니라
  **global epoch 기준**이라, 예를 들어 기존 checkpoint가 global epoch
  7까지 진행된 상태에서 `--checkpoint-every 5`로 3 epoch를 추가하면
  다음 자동 저장은 global epoch 10에서 발생합니다(12가 아님).
* **최종 checkpoint는 항상 저장됩니다**: `--checkpoint-every`를 켜지
  않았거나, 켰지만 마지막 epoch가 마침 그 주기에 맞지 않았어도,
  `--checkpoint-out`이 주어지면 학습이 정상 종료된 뒤 항상 한 번 더
  저장합니다. **마지막 epoch가 마침 자동 저장 주기와 겹치면, 같은
  global epoch가 두 번 저장될 수 있습니다** -- 이는 의도된 동작입니다.
  사용자가 학습 도중 멈춘 경우(Phase 4I `should_stop`) 자동 저장은
  항상 `stopped_by_user=False`로 저장되고(`should_stop` 평가 이전에
  실행되므로), 이 최종 저장만이 정확한 `stopped_by_user=True`를
  반영합니다.
* **checkpoint/metadata 저장은 원자적입니다**: 각 파일은 임시 파일에
  완전히 쓴 뒤 `os.replace()`로 교체합니다. 기존 파일이 있는 경우
  저장 실패 시 기존 버전이 보존되며, 새 경로에서는 반쯤 쓰인 파일이
  남지 않습니다. 저장 실패(디스크 가득 참, 권한 문제 등)는 재시도나
  폴백 없이 예외로 그대로 전파됩니다 -- 학습이 실패로 처리됩니다.
* **출력 경로 재사용 정책(breaking change)**: `--resume-from`과
  `--checkpoint-out`이 **정확히 같은 경로**를 가리키는 경우(in-place
  resume)만 기존 checkpoint 파일을 계속 갱신할 수 있습니다. 그 외의
  모든 경우(fresh 학습, 또는 `--resume-from`과 다른 경로로의 resume)는
  `--checkpoint-out`이 checkpoint 파일과 그 metadata sidecar
  (`<checkpoint>.meta.json`) 둘 다 **완전히 존재하지 않는 새 경로**여야
  합니다 -- 있으면 학습을 시작하기도 전에 명확한 오류로 거부되고
  기존 파일은 전혀 바뀌지 않습니다. 기존 checkpoint를 이어서 계속
  갱신하려면 `--resume-from`과 `--checkpoint-out`에 같은 경로를
  넘기세요. **경로를 덮어쓰도록 강제하는 옵션(예: `--overwrite-checkpoint`)은
  아직 없습니다** -- 다른 경로에 저장하고 싶으면 새 파일명을 쓰세요.
* **`--checkpoint-out`은 항상 "최신" 하나만 관리합니다**: epoch별로
  별도 파일을 남기지 않고, 매 저장이 같은 경로를 원자적으로 교체합니다.
* **custom callback을 쓸 때 주의**: `progress_callback`/`should_stop`을
  직접 구현해 넘긴다면, PyTorch RNG를 소비하거나(`torch.rand()` 등)
  model/optimizer/scheduler/DataLoader generator를 변경하면 안 됩니다.
  `checkpoint_hook`(자동 저장)이 이 콜백들보다 먼저 실행되므로, 이
  계약을 어기면 저장된 checkpoint가 실제로 다음 epoch가 시작할 때의
  상태와 달라져 **exact-resume이 깨질 수 있습니다**. `scripts/
  train_imagefolder.py`가 기본 제공하는 `_print_progress()`/자동 저장
  hook은 이 계약을 지킵니다.

`run_training()` 직접 호출자는 새 키워드 전용 파라미터
`checkpoint_hook: Callable[[EpochCheckpointView], None] | None = None`을
쓸 수 있습니다(기본값 `None`이면 기존 동작과 완전히 동일). epoch 처리
순서는 `train → validate → history 기록 → best/카운터 갱신 →
scheduler.step() → early stopping 판정 → checkpoint_hook →
progress_callback → should_stop 평가`입니다. `EpochCheckpointView`는
model/history/optimizer/scheduler/loader_generator의 살아있는
참조만 담는 synchronous ephemeral view라, hook 호출이 반환된 뒤에는
보관하거나 비동기로 넘기면 안 됩니다.

설계 배경, 여러 차례의 설계 리뷰에서 정리된 정책(metadata 준비 시점,
출력 경로 재사용 정책 전환 이유, RNG-purity 계약)은
`docs/phase4j_epoch_checkpoint_design.md`를 참고하세요.

---

## Phase 4K: Graceful SIGINT and Cooperative Training Stop

`scripts/train_imagefolder.py` 실행 중 Ctrl+C(SIGINT)를 즉시 강제 종료로
처리하지 않고, Phase 4I의 `should_stop`/Phase 4J의 checkpoint 저장 경로에
연결된 cooperative stop으로 바꿨습니다. 단일 Ctrl+C graceful stop 경로는
실제 Windows 터미널에서 수동 acceptance까지 완료했습니다(상세 결과는
`docs/phase4k_graceful_interruption_design.md` §14 참고).

**첫 번째 Ctrl+C**:

* stop request를 설정합니다(예외를 던지지 않음).
* 다음으로 실제 평가되는 epoch 경계(`should_stop()` 호출 지점)에서 학습을
  안전하게 중단합니다 -- 현재 실행 중인 epoch는 항상 끝까지 완료됩니다.
* `training_history.json`/`class_mapping.json`/`best_model_state_dict.pt`/
  `test_result.json`(+TorchScript export) 등 final artifact를 정상적으로
  저장합니다.
* `--checkpoint-out`이 있으면 final checkpoint도 저장되고, 여기에
  `stopped_by_user=True`가 기록됩니다.
* 성공적으로 완료되면 **exit code 0**입니다(cooperative stop은 정상 종료
  경로입니다).
* **이번 호출의 마지막 요청 epoch 중에는 `should_stop()` 자체가 평가되지
  않으므로**(Phase 4I의 기존 규칙), 그 시점에 Ctrl+C를 눌러도
  `stopped_by_user=False`로 남을 수 있습니다 -- 어차피 더 이상 건너뛸
  epoch가 없어 학습이 그대로 끝나기 때문입니다.
* 안내 메시지("Interrupt requested. Training will stop at the next safe
  epoch boundary. ...")는 첫 번째 Ctrl+C에서 저수준 stderr 출력을 **한
  번 시도**합니다 -- 매우 드물게(stderr가 닫힌 파이프인 경우 등) 출력이
  실패하거나 아주 긴 C/CUDA 호출 도중이라 지연될 수 있지만, 그런 경우에도
  stop request 자체는 정상적으로 설정됩니다.

**두 번째 Ctrl+C**:

* 그 자리에서 즉시 `KeyboardInterrupt`가 발생해 강제 종료합니다.
* **exit code 130**입니다.
* 남은 artifact 저장이나 final checkpoint 저장은 보장되지 않습니다.
* **마지막으로 원자적 저장이 완료된 유효한 checkpoint는 보존됩니다** --
  기존 checkpoint 파일과 새로 쓰던 파일의 일부가 섞인 상태로 노출되는
  일은 없습니다(checkpoint/metadata 저장은 임시 파일에 다 쓴 뒤
  `os.replace()`로 교체하는 원자적 저장이라, Phase 4J부터 이미 이
  보호를 제공합니다).
* `training_history.json`/`test_result.json`/TorchScript export처럼
  원자적 저장을 쓰지 않는 산출물은 두 번째 Ctrl+C 도중이면 불완전한
  상태로 남을 수 있습니다.

**exit code 요약**:

```text
정상 완료                              0
첫 번째 Ctrl+C cooperative stop 성공   0
검증/저장/일반 오류                    1
두 번째 Ctrl+C / KeyboardInterrupt     130
```

**`--checkpoint-every`와의 관계**: `--checkpoint-every`는 graceful stop의
필수 옵션이 **아닙니다**. 켜져 있지 않아도 첫 번째 Ctrl+C가 정상적으로
처리되면 final checkpoint는 항상 저장됩니다. `--checkpoint-every`가 실제로
값을 더하는 상황은 **두 번째 Ctrl+C, 프로세스 crash처럼 final save
자체에 도달하지 못하는 비정상 종료**뿐입니다 -- 그런 경우에 대비해 학습
도중 가장 최근에 완료된 epoch까지의 상태를 미리 보존해 둡니다.

**Python `signal`의 한계**: 매우 긴 C/CUDA 호출(예: 하나의 큰 저장 작업이나
CUDA 동기화) 도중에는 Python 레벨 SIGINT handler 실행 자체가 그 호출이
끝날 때까지 지연될 수 있습니다 -- 안내 메시지가 바로 뜨지 않는다고 Ctrl+C를
반복해서 누르면, 그 반복 입력이 두 번째 Ctrl+C로 해석되어 의도치 않게
강제 종료될 수 있습니다.

설계 배경과 시나리오별 상세 동작표는
`docs/phase4k_graceful_interruption_design.md`를 참고하세요.

---

## Phase 4L: Optimizer Regularization Extension (weight decay / AdamW)

Phase 4E 이후 미지원으로 남아 있던 weight decay(L2 정규화)와 AdamW
optimizer를 `TrainingConfig`/`_build_optimizer()`의 기존 확장 지점
안에서 추가했습니다.

* `TrainingConfig.optimizer`: `"adam"`(기본값) | `"sgd"` | `"adamw"`(신규)
* `TrainingConfig.weight_decay`: `float`, 기본값 `0.0` -- Adam/SGD/AdamW
  공통 적용. 음수/`NaN`/`±inf`/bool은 거부하지만 **임의의 상한은 두지
  않습니다**(실제 값이 학습에 적절한지는 사용자의 hyperparameter 선택
  책임).
* `scripts/train_imagefolder.py`에 `--weight-decay FLOAT`(기본값 `0.0`)
  플래그를 추가하고, `--optimizer` 선택지에 `adamw`를 추가했습니다.
* **resume 호환성**: `weight_decay`도 다른 optimizer 관련 필드와 동일하게
  resume 시 반드시 일치해야 하지만, **오직 이 필드에 한해** Phase 4L
  이전에 저장된 checkpoint(`weight_decay` 키 자체가 없음)는
  `weight_decay=0.0`으로 학습된 것으로 간주합니다 -- 그래서 새 config도
  `weight_decay=0.0`이면 그대로 resume할 수 있고, `weight_decay>0.0`으로
  resume하려 하면 값이 실제로 달라지므로 거부됩니다. 다른 필드가
  누락된 경우는 기존과 동일하게 항상 거부합니다.
* `src/image_ai_studio/training/imagefolder_workflow.py`는 수정하지
  않았습니다 -- `training_config`를 그대로 저장/전달하는 기존 구조
  덕분에 새 필드가 checkpoint에 자동으로 포함됩니다. `checkpoint.py`의
  `load_training_checkpoint()`는 후속 hotfix에서 구조 검사 한 곳을
  수정해, 위 resume 호환성 규칙이 실제 checkpoint 파일 경로에서도
  정확히 적용되도록 했습니다.

실제로 검증됨: `optimizer="adamw"` + `weight_decay>0` 조합에서도 continuous
run과 resume run이 model parameter/optimizer state/scheduler state 전부
tensor-level로 정확히 일치하는 회귀 테스트로 고정했습니다. 기존
`optimizer="adam"`(기본값, `weight_decay=0.0`) 4개 E2E 앵커 수치는 변경
없이 그대로입니다.

loss function 선택, `"plateau"` 외 scheduler, Adam betas/eps, SGD
dampening/nesterov, GPU/device 노출, mixed precision은 이번 Phase에서도
지원하지 않습니다(gradient norm clipping은 Phase 4M에서 추가됐습니다 --
아래 "Phase 4M" 절 참고).

설계 배경과 상세 계약은 `docs/phase4l_optimizer_regularization_design.md`를
참고하세요.

---

## Phase 4M: Gradient Norm Clipping

`TrainingConfig`에 `gradient_clip_norm`을 추가하고,
`train_one_epoch()`의 `loss.backward()`와 `optimizer.step()` 사이에
`torch.nn.utils.clip_grad_norm_()` 기반 L2 gradient norm clipping을
넣었습니다. 깊거나 learning rate가 큰 모델에서 gradient explosion을
완화할 수단이 없던 공백을 메웁니다.

* `TrainingConfig.gradient_clip_norm: float | None = None` -- `None`(기본값)
  이면 clipping이 전혀 일어나지 않아 Phase 4A~4L의 기존 동작을 완전히
  재현합니다. 설정하면 매 batch, `optimizer.step()` 직전에 L2 norm이
  이 값을 넘지 않도록 scale down합니다.
* 검증: `0`/음수/bool/`NaN`/`+inf`/`-inf`는 거부하고, 유한한 양수는
  **상한 없이** 전부 허용합니다. 기존 `_require_positive_float()`는
  `NaN`/`+inf`를 실수로 통과시키는 것을 코드로 확인했기 때문에(`NaN <= 0.0`
  과 `inf <= 0.0`이 파이썬에서 둘 다 `False`), 재사용하지 않고 별도
  `_require_positive_finite_float()` helper로 검증합니다.
* `scripts/train_imagefolder.py`에 `--gradient-clip-norm FLOAT`(기본값
  생략 시 `None`) 플래그를 추가했습니다.
* **resume 호환성**: `TrainingConfig` 전체가 `asdict()`로 그대로
  저장되므로 `gradient_clip_norm` 값 자체는 checkpoint의
  `training_config`에 저장되지만, `RESUME_CONFIG_FIELDS`에
  **포함되지 않아** resume compatibility 비교 대상은 아닙니다 --
  optimizer의 `param_groups`에 속하지 않는 순수 runtime 파라미터라
  `optimizer.load_state_dict()`가 조용히 덮어쓸 위험이 없기 때문입니다.
  그래서 resume할 때마다 이 값을 자유롭게 바꿀 수 있습니다
  (`epochs`/`early_stopping_patience`와 동일한 이유). `checkpoint.py`/
  `imagefolder_workflow.py`는 이번 Phase에서 수정하지 않았고, checkpoint
  format version도 그대로입니다.

실제로 검증됨: clipping을 켰을 때 실제 gradient의 L2 norm이 지정한
`max_norm` 이하로 줄어드는 것을 직접 계산해 확인했고(대조군으로 clipping
없이 실행하면 norm이 훨씬 큼도 함께 확인), `gradient_clip_norm != None`
조합에서도 continuous run과 resume run이 tensor-level exact equality를
유지하는 회귀 테스트로 고정했습니다. 기존 4개 E2E 앵커 수치는 변경
없이 그대로입니다.

gradient value clipping, custom `norm_type`, `error_if_nonfinite` 노출,
gradient norm history/metric 기록, 추가 scheduler, 평가 metric 확장,
GPU/device 노출은 이번 Phase에서도 지원하지 않습니다(label smoothing은
Phase 4N에서 추가됐습니다 -- 아래 "Phase 4N" 절 참고).

설계 배경과 상세 계약은 `docs/phase4m_gradient_clipping_design.md`를
참고하세요.

---

## Phase 4N: CrossEntropy Label Smoothing

`TrainingConfig`에 `label_smoothing`을 추가하고, `loop.py`에
`_build_optimizer()`/`_build_scheduler()`와 나란한 `_build_criterion()`
을 신설해 `train_one_epoch()`에 주입했습니다. label smoothing은
**training loss에만** 적용됩니다 -- validation/test loss와
`ReduceLROnPlateau`/early stopping/best-model-selection의 의미는
기존 unsmoothed `nn.CrossEntropyLoss()` 그대로 유지합니다
(`evaluate()`는 이번 Phase에서 전혀 수정하지 않았습니다).

* `TrainingConfig.label_smoothing: float = 0.0` -- `[0.0, 1.0]` 양끝
  포함, 기본값 `0.0`이면 기존 `nn.CrossEntropyLoss()`와 bitwise
  동일한 결과를 냅니다(직접 확인).
* 검증: bool/음수/`>1.0`/`NaN`/`+inf`/`-inf`는 거부합니다. 기존
  `_require_fraction()`은 상한이 항상 `<1.0`(배타적)으로 고정돼 있어
  `1.0`을 거부하므로 재사용할 수 없어, 별도
  `_require_closed_unit_interval()` helper로 검증합니다.
* `scripts/train_imagefolder.py`에 `--label-smoothing FLOAT`(기본값
  `0.0`) 플래그를 추가했습니다.
* **train/validation/test loss 의미**: `label_smoothing > 0`일 때
  `train_loss`는 smoothed CrossEntropyLoss, `val_loss`/`test_loss`는
  항상 ordinary(unsmoothed) CrossEntropyLoss입니다 -- 같은 objective의
  숫자가 아니므로 직접 비교(예: overfitting 판단)에 주의해야 합니다.
  이 정책 덕분에 `evaluate()`를 monkeypatch하는 기존 테스트 약 20곳이
  전혀 영향받지 않습니다.
* **resume 호환성**: `TrainingConfig` 전체가 `asdict()`로 그대로
  저장되므로 `label_smoothing` 값 자체는 checkpoint의
  `training_config`에 저장되지만, `RESUME_CONFIG_FIELDS`에
  **포함되지 않아** resume compatibility 비교 대상은 아닙니다 --
  `CrossEntropyLoss`에는 `*.load_state_dict()`로 저장/복원되는 state가
  없어 `gradient_clip_norm`과 동일한 이유로 resume할 때마다 자유롭게
  바꿀 수 있습니다. `checkpoint.py`/`imagefolder_workflow.py`는 이번
  Phase에서 수정하지 않았고, checkpoint format version도 그대로입니다.

실제로 검증됨: `label_smoothing=0.0`이 인자 없는 `nn.CrossEntropyLoss()`
와 `torch.equal()` 수준으로 동일함을 확인했고, `label_smoothing=0.1`이
PyTorch reference 구현(`nn.CrossEntropyLoss(label_smoothing=0.1)`)과
정확히 일치함을 확인했습니다. `label_smoothing != 0`(continuous run과
resume run이 동일 값을 쓸 때) 조합에서도 tensor-level exact equality를
유지하는 회귀 테스트로 고정했습니다. 기존 4개 E2E 앵커 수치는 변경
없이 그대로입니다.

BCE/BCEWithLogitsLoss, multilabel, focal loss, class weight, custom
loss, regression loss, `loss` 이름 선택 필드, reduction/ignore_index
변경, validation/test smoothing은 이번 Phase에서도 지원하지 않습니다.

설계 배경과 상세 계약은 `docs/phase4n_loss_function_extension_design.md`
를 참고하세요.

---

## Phase 4O: Test Classification Metrics

ImageFolder 학습의 **최종 test 평가에서만** confusion matrix + per-class
recall + macro precision/recall/F1을 계산해 `test_result.json`에
추가했습니다(test-only). 학습 루프의 validation 경로가 쓰는 `evaluate()`,
`TrainingHistory`, checkpoint/resume, config는 이번 Phase에서 전혀
수정하지 않았습니다.

* 신규 `src/image_ai_studio/training/metrics.py`: `ClassificationMetrics`
  dataclass(`confusion_matrix`, `per_class_recall`, `macro_precision`,
  `macro_recall`, `macro_f1`)와 confusion matrix tensor로부터 이 값들을
  파생 계산하는 순수 함수 `compute_classification_metrics()`. 모델
  forward/DataLoader 순회는 이 모듈에 없습니다(순수 계산만).
* 신규 `loop.py::evaluate_classification_metrics(model, loader,
  num_classes, device="cpu") -> (loss, accuracy, ClassificationMetrics)`:
  기존 `evaluate()`와 동일한 의미(unsmoothed `CrossEntropyLoss`, argmax
  accuracy, sample-weighted 평균)로 loss/accuracy를 계산하면서, 같은 한
  번의 forward pass 안에서 confusion matrix도 배치 단위로 누적합니다 --
  같은 test 데이터셋을 두 번 순회하지 않습니다. `evaluate()` 자체는
  무수정이며 계속 validation 경로에서만 쓰입니다.
* confusion matrix 컨벤션(고정): `confusion_matrix[true_idx][predicted_idx]`
  -- row=실제 클래스, column=예측 클래스, shape `[num_classes,
  num_classes]`. class 이름은 이 dataclass에 담지 않습니다 -- generic
  training core는 class 이름 개념 자체가 없고(ImageFolder 전용 계층에만
  존재), index 순서는 기존 `class_mapping.json`의 `classes` 배열 순서와
  동일하다는 계약으로 대신합니다(`test_result.json`에 class 이름을
  복제해서 넣지 않습니다).
* `per_class_recall`(이름 주의: "per-class accuracy"라 부르지 않습니다 --
  class별 accuracy는 정의상 recall과 동일합니다), `macro_precision`,
  `macro_recall`, `macro_f1`을 계산합니다. **`macro_f1`은 class별 F1을
  먼저 구한 뒤 평균한 값**입니다 -- `harmonic_mean(macro_precision,
  macro_recall)`이 아닙니다.
* zero-division 정책(고정): 분모(해당 class의 true/predicted sample 수)가
  0이면 그 class의 지표는 `0.0`입니다. **`0.0`이 항상 "모델이 그 class를
  전부 틀렸다"는 뜻은 아닙니다** -- 특히 test set에 해당 class의 true
  sample 자체가 없으면 `recall=0.0`은 "측정할 sample이 없어 정책상
  0.0을 기록했다"는 뜻입니다. 함께 저장되는 confusion matrix의 해당
  row/column 합이 0인지 봐서 이 두 경우를 구별할 수 있습니다(별도
  support/count 필드는 추가하지 않았습니다).
* `test_result.json` 스키마(additive, nested): 기존 `test_loss`/
  `test_accuracy` top-level key는 그대로 유지하고, `classification_metrics`
  키 아래에 위 5개 필드를 nested로 추가했습니다. schema version 필드는
  추가하지 않았습니다.
  ```json
  {
    "test_loss": 0.42,
    "test_accuracy": 0.88,
    "classification_metrics": {
      "confusion_matrix": [[45, 5], [7, 43]],
      "per_class_recall": [0.9, 0.86],
      "macro_precision": 0.87,
      "macro_recall": 0.88,
      "macro_f1": 0.875
    }
  }
  ```
* `ImageFolderWorkflowResult`에 `test_metrics: ClassificationMetrics |
  None = None` 필드를 **기본값과 함께 마지막 필드로** 추가했습니다 --
  기존에 이 dataclass를 `test_metrics` 없이 직접 생성하던 코드(기존
  테스트의 manual/fake constructor 호출)는 그대로 동작합니다.
  `run_imagefolder_training_workflow()`가 정상 완료해 반환하는 production
  결과의 `test_metrics`는 항상 실제 `ClassificationMetrics`입니다 --
  `None`은 constructor 하위호환 전용이지 production에서 test 평가가
  생략될 수 있다는 뜻이 아닙니다. 기존 `result.test_loss`/
  `result.test_accuracy` 필드는 변경 없이 그대로 유지됩니다.
* sklearn 등 신규 dependency를 추가하지 않았습니다 -- confusion
  matrix/파생 지표는 전부 torch/Python으로 직접 계산합니다(기존
  `dependencies = ["torch>=2.4", "numpy"]` 그대로).
* `scripts/train_imagefolder.py`의 stdout은 변경하지 않았습니다(기존
  `Test: loss=... accuracy=...` 한 줄 그대로) -- 상세 지표는
  `test_result.json`에서 확인합니다.
* `TrainingHistory`/`checkpoint.py`/`config.py`는 이번 Phase에서
  전혀 수정하지 않았습니다. `RESUME_CONFIG_FIELDS`,
  `RESUME_CONFIG_LEGACY_DEFAULTS`, checkpoint format version, exact
  resume 전부 무영향입니다(metric 계산은 `evaluate()`와 마찬가지로
  `model.eval()` + `torch.inference_mode()`에서 순수 forward pass만
  수행하므로 gradient/RNG 소비가 없습니다).

이번 Phase에서 지원하지 않는 것: validation epoch별 상세
metric, `TrainingHistory`의 metric 필드, metric 기반 early
stopping/scheduler, class weight, BCE/focal loss, per-class
precision/F1 노출, micro/weighted average, sklearn dependency,
ROC-AUC/PR-AUC, CLI macro F1 출력, GPU/device 노출.

설계 배경과 상세 계약은 `docs/phase4o_evaluation_metrics_design.md`를
참고하세요.

---

## Phase 4P: Explicit Class-weighted CrossEntropy

`TrainingConfig`에 `class_weights`를 추가하고, `_build_criterion()`
(Phase 4N이 만든 확장점)이 `nn.CrossEntropyLoss(weight=..., label_smoothing=...)`
를 생성하도록 확장했습니다. **사용자가 직접 지정하는 explicit weight만
지원합니다** -- 자동(inverse-frequency 등) 계산이나 `WeightedRandomSampler`
는 이번 Phase 범위 밖입니다(§ 아직 미지원 참고). class weighting은
**training loss에만** 적용됩니다 -- validation/test loss와
`ReduceLROnPlateau`/early stopping/best-model-selection/Phase 4O
classification metrics의 의미는 기존 unweighted `nn.CrossEntropyLoss()`
그대로 유지합니다(`evaluate()`, `evaluate_classification_metrics()`
둘 다 이번 Phase에서 전혀 수정하지 않았습니다).

* `TrainingConfig.class_weights: tuple[float, ...] | None = None` --
  공식 representation은 **tuple뿐**입니다(list 등 다른 sequence는
  `TrainingConfigError`로 거부). 각 원소는 finite + strictly positive
  (`> 0`)여야 합니다 -- 0/음수/NaN/`+inf`/`-inf`는 전부 거부됩니다.
  PyTorch의 `CrossEntropyLoss(weight=...)`는 이 값들을 constructor/forward
  어디서도 검증하지 않고(실측 확인: 0/음수/NaN/inf 전부 조용히 통과,
  all-zero 또는 한 배치가 우연히 zero-weight class 샘플로만 구성되면
  `NaN` loss가 실제로 재현됨) 그 방어를 이 프로젝트가 대신합니다.
* `scripts/train_imagefolder.py`에 `--class-weights FLOAT [FLOAT ...]`
  (`nargs="+"`, 기본값 없음 = weighting 비활성) 플래그를 추가했습니다.
  CLI boundary에서만 `tuple(args.class_weights)`로 canonicalize하며,
  `TrainingConfig` 자체는 list를 자동 변환하지 않습니다.
* **class 순서 계약**: `--class-weights`의 순서는 `class_mapping.json`의
  `classes`/`class_to_idx` 순서와 반드시 일치해야 합니다(예:
  `classes=["cat","dog"]`이면 `--class-weights 1.0 3.0`은 `cat=1.0,
  dog=3.0`). `TrainingConfig`/generic training core에는 class 이름을
  전혀 넣지 않습니다 -- Phase 4O가 확립한 generic core/ImageFolder
  presentation 분리를 그대로 유지합니다.
* **class 수 불일치 검증 범위(중요, 과장하지 않음)**: ImageFolder
  workflow는 학습 시작 전에 `len(class_weights)`와 dataset의 실제 class
  수(`len(splits.classes)`)가 일치하는지 명시적으로 검증합니다(기존
  `require_matching_num_classes()`와 대칭적인 위치). 하지만 generic
  `run_training()`/`TrainingConfig` 경로는 ModelSpec도 dataset도 몰라서
  이 길이를 스스로 검증하지 않습니다 -- 그 경로에서 길이가 어긋나면
  PyTorch `CrossEntropyLoss`의 forward-time `RuntimeError`(실측 확인된
  shape 검증)가 최종 backstop입니다. **"class weight 길이는 항상
  사전 검증된다"는 진술은 정확하지 않습니다** -- ImageFolder 경로에
  한해서만 사전 검증됩니다.
* **weight tensor의 dtype/device**: `_build_criterion(config, device)`가
  `device` 위에 `torch.float32` dtype으로 직접 생성합니다(model/입력과
  같은 device에 있어야 forward에서 device mismatch가 나지 않음).
* **label smoothing과 조합 가능**: `class_weights`와 `label_smoothing`을
  동시에 켤 수 있습니다(PyTorch가 이 조합을 제약 없이 지원함을 실측
  확인). 기본 경로(`class_weights=None, label_smoothing=0.0`)의 기존
  numerical anchor는 그대로 유지됩니다.
* **resume 호환성**: `class_weights`는 `RESUME_CONFIG_FIELDS`에
  **포함되지 않습니다** -- checkpoint에는 `training_config`를 통해
  저장되지만 resume compatibility 비교 대상은 아니라서 resume 시
  자유롭게 변경할 수 있습니다. 이 정책의 실제 근거를 정확히 표현하면:
  **weight가 설정된 `CrossEntropyLoss`는 실제로 `weight` buffer를 가져
  `state_dict()`가 비어있지 않습니다** -- "criterion의 state_dict가
  항상 비어서"가 resume-free-change의 근거가 아닙니다. 진짜 근거는
  `training/checkpoint.py`(checkpoint subsystem)가 criterion의 state
  자체를 애초에 저장하지도 복원하지도 않기 때문입니다(`run_training()`
  이 매번 config로 criterion을 새로 생성) -- optimizer/scheduler처럼
  checkpoint에서 `load_state_dict()`로 복원되어 새 config 값을 조용히
  덮어쓸 경로가 criterion에는 없습니다. Phase 4P 이전 checkpoint 파일에는
  `class_weights` 키 자체가 없지만, `RESUME_CONFIG_FIELDS`에 없으므로
  구조 검증/호환성 비교 어느 단계에서도 요구되지 않아 별도 legacy
  migration이 필요 없습니다(Phase 4L의 `weight_decay` 문제와 다른
  구조 -- 실제 checkpoint 파일 기반 회귀 테스트로 확인함).
* **exact-resume**: 동일한 `class_weights`로 resume하면 기존 exact-resume
  계약(model/optimizer/scheduler state, history, best state 등 tensor-level
  일치)이 그대로 유지됩니다(weight tensor 생성은 RNG를 소비하지 않는
  결정론적 연산).
* CLI stdout은 무수정입니다 -- 상세 metric은 기존과 동일하게
  `test_result.json`(Phase 4O)에서 확인합니다.

이번 Phase에서 지원하지 않는 것: automatic(inverse-frequency 등) class
weight 계산, `WeightedRandomSampler`/oversampling/undersampling,
class-name 기반 weight 지정 문법(index/tuple 순서만 지원), zero/negative
weight, validation/test weighting, Phase 4O metric 로직 변경, GPU/device
CLI 노출, AMP, 추가 LR scheduler.

설계 배경과 상세 계약은
`docs/phase4p_class_weighted_cross_entropy_design.md`를 참고하세요.

---

## Phase 4Q: Runtime Training Device Exposure

ImageFolder 학습에 CPU/CUDA/CUDA:N device를 명시적으로 선택할 수 있게
했습니다. `run_training()`/`evaluate()`/`evaluate_classification_metrics()`
/`_build_criterion()` 같은 generic training core(`loop.py`)는 이미
전부 `device` 파라미터를 갖고 있었으므로(Phase 4A~4P가 이미 device-aware
하게 설계해 둠), `loop.py`는 **전혀 수정하지 않았습니다** -- 이번 Phase는
그 device 파라미터를 ImageFolder CLI/workflow에서 실제로 선택 가능하게
연결하는 배선(wiring) 작업입니다.

* `ImageFolderWorkflowRequest.device: str = "cpu"`(기본값 CPU, 기존
  direct constructor 호출과 하위호환) + CLI `--device`. **runtime
  실행 파라미터로 취급**합니다 -- `seed`처럼 `TrainingConfig` 밖에서
  관리되는 run-level 파라미터라는 점에서 같은 계층입니다(`seed`는
  RNG/initialization 결과에 영향을 주고 `device`는 execution backend를
  정한다는 차이는 있지만, 둘 다 학습 objective를 바꾸는 hyperparameter가
  아니므로 `TrainingConfig`에는 두지 않았습니다). checkpoint의
  `training_config`/`RESUME_CONFIG_FIELDS`와 완전히 무관합니다.
* 공식 지원 syntax는 `cpu`/`cuda`/`cuda:N`뿐입니다(대소문자 변형,
  `mps`/`xpu`/`hip` 등 다른 backend, zero-padding/음수 index는 전부
  거부). `--device cuda`/`cuda:N`인데 `torch.cuda.is_available()`이
  `False`이거나 index가 `torch.cuda.device_count()` 범위를 벗어나면
  **CPU로 조용히 대체하지 않고 학습 시작 전에 명확한 오류로 거부**합니다.
* 학습 전 `model.to(device)`를 수행합니다(optimizer 생성보다 반드시
  먼저 -- PyTorch semantics). `_build_criterion(config, device)`(Phase 4P)
  가 그대로 재사용되어, class-weighted CrossEntropy의 weight tensor도
  자동으로 학습 device 위에 생성됩니다.
* **최종 test 평가/TorchScript export/C++ parity는 학습 device와
  무관하게 항상 CPU를 유지합니다** -- `best_model`은 `build_model()`로
  새로 만들어져 항상 CPU이고(GPU에서 학습한 `best_state_dict`를 로드해도
  PyTorch가 cross-device 복사를 안전하게 처리함을 직접 실측 확인),
  기존 `TorchScriptExporter`가 `example_input`을 CPU로 강제하는 암묵적
  계약(model도 CPU여야 함)을 그대로 유지합니다. 이번 Phase는 "training
  device exposure"이지 "evaluation device exposure"가 아닙니다.
* `checkpoint.py`는 **전혀 수정하지 않았습니다** -- `load_training_checkpoint()`
  /`load_state_dict()`가 이미 `map_location="cpu"` 기본값을 가지므로
  GPU에서 저장한 checkpoint도 CPU 전용 환경에서 문제없이 로드됩니다.
  로컬 CUDA로 직접 실측한 결과, `model.load_state_dict()`/
  `optimizer.load_state_dict()` 둘 다 저장 시점 device와 무관하게 현재
  model/optimizer의 device로 자동 이관됨을 확인했습니다(CPU→GPU,
  GPU→CPU 양방향).
* **resume 계약(중요, 정확히 구분)**: CPU→CPU는 기존 exact-resume
  계약(tensor-level exact equality)을 **완전히 그대로 유지**합니다.
  CUDA를 포함한 resume은 model/optimizer **state portability는
  지원**하지만 **bitwise exact-resume은 보장하지 않습니다** -- 현재
  checkpoint는 CPU RNG state와 DataLoader generator state만 저장하고
  **CUDA RNG state(`torch.cuda.get_rng_state_all()`)는 저장하지 않기
  때문**입니다. Phase 4Q에서는 대표 경로인 CPU→CUDA resume을 실제
  CUDA smoke test로 검증했습니다(CUDA→CPU/CUDA→CUDA 전용 workflow
  테스트는 별도로 추가하지 않았습니다 -- model/optimizer state의
  cross-device 이관 자체는 PyTorch 레벨 실측으로 양방향 확인). CUDA를
  포함한 exact-resume은 별도 Phase(CUDA RNG checkpoint, deterministic
  algorithm 설정 필요)로 분리했습니다.
* CLI stdout에 `Device: cpu`처럼 사용자가 지정한 값을 한 줄 echo합니다
  -- `Model JSON`/`Dataset root`/`Resume from`/`Checkpoint out`과 같은
  계층의 입력값 echo이지 새로 계산된 지표가 아니므로, Phase 4O의 "상세
  metric stdout 확대 금지" 원칙과 충돌하지 않습니다.

이번 Phase에서 지원하지 않는 것: CUDA exact-resume(CUDA RNG checkpoint,
deterministic algorithm 설정), AMP/mixed precision, gradient
accumulation, multi-GPU/distributed training, `mps`/`xpu`/`hip` 등 CUDA
외 backend, 학습 device와 다른 별도 evaluation device, artifact에 device
기록, GPU 성능 튜닝(pin_memory/num_workers).

설계 배경과 상세 계약은
`docs/phase4q_runtime_training_device_design.md`를 참고하세요.

---

## Phase 4R: Same-device CUDA Exact Resume

Phase 4Q가 남겨둔 gap을 메웁니다 -- single CUDA device 학습에서
checkpoint에 CUDA RNG state를 추가로 저장/복원하고, CUDA training
구간에만 scoped deterministic execution을 적용해 **같은 물리 CUDA
device / 같은 머신 / 같은 소프트웨어 환경** 기준으로 continuous
training과 split+resume training이 bitwise exact하도록 만듭니다.

* `cuda_rng_state`(checkpoint의 새 optional 필드, `cpu_rng_state`/
  `loader_generator_state`와 대칭적인 순수 실행 state) -- CPU RNG/
  DataLoader RNG와 함께 같은 지점에서 캡처/복원됩니다. CPU checkpoint는
  이 값이 `None`입니다.
* CUDA training 구간(RNG 복원 → `run_training()` → 최종 RNG 캡처)에만
  scoped context로 `torch.use_deterministic_algorithms(True)`/
  `torch.backends.cudnn.deterministic=True`/`torch.backends.cudnn.benchmark=False`
  를 적용합니다 -- workflow 호출이 끝나면(정상 종료든 예외든) caller의
  기존 전역 설정을 정확히 복원합니다(직접 실측 확인). **CPU 경로는 이
  설정을 전혀 읽거나 쓰지 않습니다.**
* deterministic kernel을 강제로 선택하므로 CUDA training 일부가 Phase
  4Q보다 느려질 수 있습니다 -- 이는 버그가 아니라 exact-resume
  guarantee의 tradeoff입니다.
* CPU→CPU exact-resume 계약은 완전히 그대로 유지됩니다(무영향).
* **CPU↔CUDA**는 Phase 4Q의 state portability 계약만 유지합니다 --
  Phase 4R의 bitwise exact 계약 대상이 아닙니다. **cross-GPU
  architecture**(`cuda:0`→다른 물리 GPU 포함)도 same-device exact
  계약 대상이 아니라 bitwise exact를 검증/보장하지 않습니다.
  **multi-GPU/distributed는 exact 여부 이전에 training 자체가
  현재 지원 범위 밖**입니다. fresh-run 전역 reproducibility 보장과
  continuous-vs-resume exactness는 서로 다른 계약입니다. pre-4R CUDA
  checkpoint는 resume은 가능하지만(portable-only) same-device exact
  계약 대상이 아닙니다(별도 warning 없음).
* `CUBLAS_WORKSPACE_CONFIG` 환경변수는 이번 Phase가 요구하지 않습니다 --
  실측(subprocess, 환경변수 unset/설정 양쪽)에서 이 프로젝트가 실제
  쓰는 연산은 결과/에러 여부가 동일했습니다. 다른 CUDA 환경에서 향후
  deterministic 관련 오류가 발생한다면 process 시작 전 환경변수 설정이
  필요할 수 있으나, workflow가 실행 중 이 환경변수를 조용히 바꾸지는
  않습니다.
* AMP/Mixed Precision은 이 Phase에서는 미지원입니다(CUDA FP16 AMP는
  Phase 4S에서 지원 -- 아래 "Phase 4S" 절 참고).

설계 배경과 상세 계약(RNG inventory, positive/negative control 실측,
Conv2d/BatchNorm 비결정성 실측, deterministic context 설계, 성능
tradeoff)은 `docs/phase4r_cuda_exact_resume_design.md`를 참고하세요.

---

## Phase 4S: Same-device CUDA AMP (FP16) Training

single CUDA device ImageFolder training에 `torch.amp.autocast`(FP16)+
`torch.amp.GradScaler`를 도입하고, checkpoint에 `scaler_state_dict`를
추가해 Phase 4R의 same-device exact-resume 계약을 AMP-enabled training
까지 확장합니다. `precision="fp32"`(기본값)에서는 CPU/CUDA 기존 FP32
학습 semantics와 numerical behavior를 그대로 유지합니다(기존 5개 E2E
numerical anchor 무변경, 기존 Phase 4R CUDA FP32 exact-resume test
PASS로 확인).

* `TrainingConfig.precision`(`"fp32"`(기본값) | `"fp16"`) -- CLI
  `--precision {fp32,fp16}`으로도 선택 가능합니다. `RESUME_CONFIG_FIELDS`
  에는 포함되지 않습니다 -- `gradient_clip_norm`/`label_smoothing`/
  `class_weights`와 같은 범주(자유롭게 바뀔 수 있는 training semantics)로,
  optimizer/scheduler 구조를 바꾸지 않기 때문입니다(현재 지원 optimizer와
  실측한 AMP 경로에서는 optimizer momentum buffer 등이 precision과
  무관하게 float32로 유지됨을 확인했습니다).
* `precision="fp16"`은 CUDA에서만 허용됩니다 -- `device="cpu"`와 조합하면
  (`ImageFolderWorkflowRequest`를 거치는 workflow 경로든, `TrainingConfig`
  +`run_training()`을 직접 쓰는 generic 경로든) silent fallback 없이
  명확히 거부됩니다(CPU AMP는 이번 Phase 범위 밖).
* CUDA FP16 AMP training은 `torch.amp.autocast(device_type="cuda",
  dtype=torch.float16)`로 forward+loss 계산을 감싸고,
  `torch.amp.GradScaler("cuda")`로 backward/step을 수행합니다(현재
  PyTorch가 권장하는 API -- `torch.cuda.amp.autocast`/`GradScaler`는
  FutureWarning으로 deprecated임을 직접 실측 확인했습니다).
* gradient clipping(Phase 4M)과 AMP를 함께 쓰면 `scaler.unscale_(optimizer)`
  를 clipping 직전에 정확히 한 번 호출합니다 -- 생략하면 grad norm이
  `scale`배로 부풀어 clipping 임계값이 무의미해짐을 실측으로 확인했습니다.
* validation/test 평가와 TorchScript export/C++ parity는 이 값과 무관하게
  항상 FP32(train에만 AMP 적용)로 수행됩니다 -- Phase 4Q/4R의 "최종 평가는
  항상 CPU" 원칙과 같은 이유로, scheduler/early stopping/best model
  selection의 기존 수치적 의미를 그대로 지킵니다.
* checkpoint의 새 optional 필드 `scaler_state_dict`(`torch.amp.GradScaler
  .state_dict()`, FP32/CPU checkpoint에서는 `None`) -- `cuda_rng_state`와
  달리 caller가 별도로 채취하는 게 아니라 `TrainingResult.scaler_state_dict`
  (loop.py의 `run_training()`이 이미 채워서 반환)에서 그대로 읽습니다.
  positive/negative control 실측으로 GradScaler state 복원이 same-device
  AMP exact-resume에 **필수**임을 직접 증명했습니다(state를 복원하지
  않으면 이어지는 학습 결과가 실제로 갈라짐).
* same physical CUDA device / 같은 머신 / 같은 소프트웨어 환경 기준의 AMP
  exact-resume은 **scaler_state_dict가 있는 새 checkpoint에서만** 보장됩니다.
  precision을 바꿔 resume(FP32↔AMP)하거나 legacy(pre-4S) checkpoint를
  AMP로 resume하는 것은 여전히 허용되지만(portable, silent fallback
  없음), exact 계약 대상은 아닙니다 -- 실측으로 양방향 모두 model/optimizer
  state가 깨지지 않고 정상 동작함을 확인했으며, bitwise exact는 same
  precision끼리만 보장합니다. **이 precision 변경 resume 정책은
  "resume에 쓸 새 precision이 무엇이든 허용"이라는 뜻일 뿐, `cpu`+`fp16`
  처럼 애초에 유효하지 않은 device/precision 조합까지 허용한다는 뜻은
  아닙니다** -- resume 시 새로 지정한 device/precision도 위 조합 규칙을
  그대로 따라야 합니다.
* Phase 4R의 scoped deterministic context(`use_deterministic_algorithms`/
  `cudnn.deterministic`/`cudnn.benchmark`)를 그대로 재사용합니다 -- AMP
  전용 별도 deterministic 설정은 추가하지 않았습니다(실측: FP16 autocast+
  GradScaler가 기존 context 안에서 RuntimeError 없이 정상 동작).
* deterministic FP16 kernel을 쓰므로 CUDA training 속도/메모리 사용량이
  하드웨어에 따라 달라질 수 있습니다 -- 특정 GPU에서의 speedup을
  보장하지 않습니다(Tensor Core가 없는 GPU에서는 이득이 작거나 없을 수
  있습니다).
* CPU AMP, multi-GPU/distributed, gradient accumulation, GradScaler
  tuning parameter(`init_scale`/`growth_interval` 등) 노출, AMP
  inference/export는 이 Phase에서는 미지원입니다(CUDA BF16은 Phase 4T
  에서 지원 -- 아래 "Phase 4T" 절 참고).

설계 배경과 상세 계약(AMP API 조사, GradScaler lifecycle 실측,
positive/negative control, gradient clipping 통합, exact-resume 계약)은
`docs/phase4s_amp_mixed_precision_design.md`를 참고하세요.

---

## Phase 4T: CUDA BF16 Mixed Precision Training

Phase 4S의 CUDA FP16 AMP에 이어 CUDA BF16 mixed precision을 지원합니다.
BF16은 FP16과 달리 `torch.amp.GradScaler`를 쓰지 않습니다 -- BF16은
FP32와 동일한 8-bit exponent range를 가져(FP16의 5-bit보다 넓음) FP16
에서 loss scaling이 특히 필요했던 좁은 dynamic range 문제를 크게
완화합니다. 이 프로젝트의 실제 BF16 학습 경로에서 GradScaler 없이도
정상 학습과 same-device exact-resume이 성립함을 실측으로 확인했으며,
그에 따라 **production contract로 BF16에는 GradScaler를 사용하지
않습니다**(BF16에 이론적으로 절대 underflow가 없다는 뜻은 아닙니다).

* `TrainingConfig.precision`에 `"bf16"`이 추가됐습니다(`"fp32"`(기본값)
  | `"fp16"` | `"bf16"`) -- CLI `--precision {fp32,fp16,bf16}`으로도
  선택 가능합니다. `RESUME_CONFIG_FIELDS`에는 여전히 포함되지 않습니다.
* CUDA BF16 training은 `torch.amp.autocast(device_type="cuda",
  dtype=torch.bfloat16)`로 forward+loss 계산만 감싸고, backward/
  clipping/`optimizer.step()`은 GradScaler 없이 FP32와 동일한 순서
  (`loss.backward()` → [clip] → `optimizer.step()`)를 그대로 씁니다.
* `precision="fp16"`/`"bf16"` 둘 다 CUDA에서만 허용됩니다 -- `cpu`나
  이 프로젝트가 인식하지 않는 다른 backend(`mps`/`xpu` 등)와 조합하면
  silent fallback 없이 명확히 거부됩니다(workflow 레벨과, workflow를
  거치지 않는 generic `run_training()` 호출 레벨 양쪽에서).
* checkpoint의 `scaler_state_dict`는 BF16 checkpoint에서 항상 `None`
  입니다 -- BF16이 GradScaler를 쓰지 않으므로 새 checkpoint 필드/
  `CHECKPOINT_FORMAT_VERSION` 변경이 전혀 필요 없습니다(`checkpoint.py`
  는 이번 Phase에서 무수정입니다).
* same physical CUDA device / 같은 머신 / 같은 소프트웨어 환경 기준의
  BF16 same-device exact-resume을 지원합니다 -- GradScaler state가
  없는데도(즉 FP32와 동일한 state 집합만으로) continuous training과
  split+resume training이 bitwise exact함을 production workflow
  경로에서 직접 실측 확인했습니다.
* Phase 4R의 scoped deterministic context를 FP16과 동일하게 그대로
  재사용합니다 -- BF16 전용 별도 deterministic 설정은 추가하지
  않았습니다.
* **BF16 실행을 지원한다는 것이지 "더 빠르다"는 계약이 아닙니다.**
  Tensor Core 기반 네이티브 BF16 하드웨어가 없는 GPU에서는
  `torch.amp.autocast(dtype=torch.bfloat16)`가 emulation으로 동작할
  수 있고, 이 경우 FP32 대비 속도 이득이 없거나 더 느릴 수 있습니다.
  production code는 hardware capability를 강제로 검증하지 않습니다
  (기능적으로 지원되면 하드웨어 세대와 무관하게 실행을 허용합니다).
* validation/test 평가, TorchScript export, C++ parity는 이 값과
  무관하게 항상 CPU+FP32로 수행됩니다(Phase 4Q/4R/4S와 동일한 정책).
* CPU BF16, multi-GPU/distributed, gradient accumulation, AMP
  inference/export, FP8은 아직 미지원입니다.

설계 배경과 상세 계약(BF16 API/hardware 실측, GradScaler 불필요성 근거,
exact-resume 실측, cross-precision resume matrix, compatibility matrix)
은 `docs/phase4t_cuda_bf16_mixed_precision_design.md`를 참고하세요.

---

## Phase 4U: CUDA H2D Transfer Optimization

CUDA ImageFolder training의 host(CPU) → device(CUDA) batch 전송 경로에
`pin_memory`/`non_blocking` 두 가지 순수 runtime 최적화 옵션을
지원합니다. `device`(Phase 4Q)와 같은 계층의 실행 파라미터라
`TrainingConfig`/`RESUME_CONFIG_FIELDS`와 무관합니다.

* `ImageFolderWorkflowRequest.pin_memory`/`.non_blocking`(기본값 둘 다
  `False`) -- CLI `--pin-memory`/`--non-blocking`으로도 선택 가능하며,
  서로 독립적으로 설정할 수 있습니다(한쪽만 켜는 조합을 거부하지
  않습니다). `--device cpu`면 두 값 모두 항상 무시되고 effective 값은
  강제로 `False`입니다 -- CPU에는 host→device 전송 자체가 없어
  optimization hint가 의미를 갖지 못하며, 이 강제 처리 덕분에 PyTorch의
  "no accelerator found" 경고도 애초에 발생하지 않습니다.
* train/val DataLoader 둘 다 같은 effective `pin_memory`를 받습니다.
  최종 test 평가는 Phase 4Q부터 항상 CPU 고정이라 이 optimization을
  적용하지 않습니다.
* `pin_memory`/`non_blocking`은 서로 독립적인 runtime optimization
  hint입니다. `non_blocking=True`는 training/validation의
  `images.to(device, non_blocking=...)`/`labels.to(...)` 호출에 그대로
  전달되며, host-side synchronization을 줄일 수 있어 pageable
  (unpinned) source에서도 의미가 있을 수 있습니다("unpinned면 항상
  blocking으로 강등된다"고 일반화하지 않습니다). `pin_memory=True`는
  DataLoader가 반환하는 host tensor를 page-locked memory에 배치해
  CUDA H2D transfer를 더 효율적으로 만들 수 있습니다. 다만 이
  프로젝트는 항상 default CUDA stream만 쓰므로(별도 stream 없음), 두
  값을 함께 켜더라도 H2D copy와 model kernel execution의 GPU-side
  overlap을 보장하지는 않습니다 -- **정확성**은 동일 stream 내 커널
  실행 순서 보장으로 실측 확인했지만(blocking과 non_blocking(pinned)의
  forward 결과가 bit-identical), 이는 "정확하다"는 보장이지 "겹쳐
  실행되어 더 빠르다"는 보장이 아닙니다. 별도 CUDA stream을 도입하면
  이 전제가 깨지므로 그 시점에 재검증이 필요합니다.
* `pin_memory`/`non_blocking`은 loader generator를 전혀 소비하지 않는
  순수 host-memory 전송 최적화라, **resume 경계에서 값이 checkpoint
  저장 당시와 달라져도** bitwise exact-resume에 영향이 없습니다
  (`RESUME_CONFIG_FIELDS`와도 무관) -- 기존 checkpoint 필드
  (`loader_generator_state` 등)만으로 이미 충분함을, continuous 5 epoch
  (`pin_memory=False`/`non_blocking=False`) vs split(3+2, resume 2 epoch만
  `True`/`True`로 전환) 조합으로 production 경로에서 직접 실측
  확인했고, 새 checkpoint field/`CHECKPOINT_FORMAT_VERSION` 변경은
  없습니다.
* `num_workers`/`persistent_workers`/`prefetch_factor`는 이번 Phase에서
  의도적으로 미노출입니다 -- `persistent_workers=True`는 현재
  checkpoint/resume 설계와 구조적으로 충돌해(continuous run과 resume
  사이 DataLoader worker seed 소비 횟수가 어긋남) exact-resume이
  깨지고, `num_workers>0` 자체는 exact-resume은 안전하지만 이
  프로젝트의 전형적인 작은 데이터셋 규모 + Windows spawn 환경에서
  오히려 뚜렷하게 느려짐을 실측했습니다.
* **성능 향상을 보장하지 않습니다.** 실제 효과는 dataset 크기/storage
  I/O/GPU/batch size에 따라 다르며, 로컬 GTX 1080 + 이 프로젝트의
  작은 CIFAR-10 ImageFolder 규모에서는 뚜렷한 speedup을 보이지
  않았습니다.

설계 배경과 상세 계약(PyTorch DataLoader base_seed 소스 분석,
persistent_workers 비호환 메커니즘, exact-resume 실측, 성능 측정
caveat)은 `docs/phase4u_cuda_h2d_transfer_optimization_design.md`를
참고하세요.

---

## Phase 4V: Progress / Runtime Observability

향후 GUI를 포함한 caller가 training engine 내부 로직을 재현하지
않고도 학습 진행 상황과 종료 상태를 알 수 있도록, 기존
`TrainingProgress`/`TrainingResult`에 관찰값 두 가지만 최소로
추가했습니다. `TrainingProgress`는 이미 Phase 4I부터 `run_epoch`/
`total_run_epochs`/`global_epoch`/`train_loss`/`val_loss`/
`val_accuracy`/`learning_rate`/`best_epoch`/`best_val_loss`/
`epochs_without_improvement`/`stopped_early`를 매 epoch 제공하고
있었습니다.

* `TrainingProgress.epoch_duration_seconds`(신규) -- 그 epoch의
  engine wall-clock duration(초, `time.perf_counter()` 기준
  monotonic 측정). `train_one_epoch()` 시작부터 `checkpoint_hook`
  호출 완료까지를 포함하고, `progress_callback` 자신과 그 이후의
  `should_stop()` 실행 시간은 포함하지 않습니다. session-local
  값입니다 -- resume해도 이전 호출의 duration을 복원/누적하지
  않고, 이번 호출에서 새로 측정한 값만 담습니다. deterministic한
  학습 state가 아니므로 checkpoint에 저장되지 않고 exact-resume
  비교 대상도 아닙니다.
* `TrainingResult.stop_reason`(신규, 기본값 `"completed"`) --
  `"completed"`/`"early_stopped"`/`"user_stopped"` 중 하나인
  authoritative 최종 종료 사유입니다. caller가 매번
  `history.stopped_early`/`stopped_by_user`를 직접 조합해
  재추론할 필요가 없습니다. **`TrainingProgress`에는 대응하는
  필드를 추가하지 않았습니다** -- `should_stop()`이 항상 마지막
  `progress_callback` 호출 이후에만 평가되는 기존 ordering(Phase
  4I) 때문에, 어떤 epoch의 progress event도 "user가 멈췄다"는
  사실을 알 수 있는 시점에 존재하지 않습니다(구조적으로 불가능 --
  이 ordering 자체를 바꾸지 않았습니다).
* `run_imagefolder_training_workflow()`가 이 프로젝트의 GUI-facing
  public entrypoint이므로, `ImageFolderWorkflowResult.stop_reason`
  (신규, 기본값 `"completed"`)도 `TrainingResult.stop_reason`을
  그대로 forwarding합니다(재계산하지 않음, single source of truth).
* checkpoint/`CHECKPOINT_FORMAT_VERSION`/`RESUME_CONFIG_FIELDS`/
  `TrainingHistory`(→ `training_history.json` 및 checkpoint payload의
  `history` 서브딕트) 스키마는 전부 무수정입니다 -- 두 값 모두 어디에도
  직렬화되지 않는 순수 runtime 값입니다.
* 기존 `learning_rate`(scheduler.step() 이전 값), resume epoch
  numbering(`global_epoch`은 절대, `run_epoch`/`total_run_epochs`는
  호출-local), callback ordering(checkpoint_hook → progress_callback
  → should_stop), callback contract(동기, 예외 그대로 propagate,
  frozen snapshot) 전부 무변경입니다.
* batch-level progress, stage/phase event, completion event,
  device/precision을 progress에 포함, resume 간 누적(cumulative)
  elapsed time, ETA는 이번 Phase의 범위 밖입니다.

설계 배경과 상세 계약(callback ordering 재구성, early stopping vs
user stop ordering의 구조적 차이, duration 측정 경계, checkpoint/
artifact schema 무영향 근거, GUI worker-thread 예상 사용 방식)은
`docs/phase4v_progress_runtime_observability_design.md`를
참고하세요.

---

## Phase 4W: Final Training Integration / Graduation

Phase 4A~4V에서 구현한 기능을 하나의 production pipeline으로 통합
검증하고 Phase 4를 공식 종료했습니다. **새 기능을 추가하지 않았습니다**
-- 기존 production API를 실제 사용자 흐름으로 조합해 재검증하는 것이
유일한 목적이었고, production code 변경 없이 검증만으로 완료됐습니다.

* full pytest 719/719 PASS(CUDA 포함, skip 없음).
* CPU exact-resume(기본 + AdamW/gradient-clip/label-smoothing/
  class-weights/user-stop 변형 6종), CUDA FP32(Phase 4R)/FP16(Phase 4S)/
  BF16(Phase 4T)/H2D option-change(Phase 4U) exact-resume 전부 실제
  GPU에서 PASS.
* 기존 5개 E2E(`run_phase1_e2e.py`/`run_training_e2e.py`/
  `run_real_training_e2e.py`/`run_resume_training_e2e.py`/
  `run_imagefolder_training_e2e.py`) 전부 PASS, numerical anchor
  (`1.3386→0.2867`, `2.3558→2.0817`, resume epoch5
  `train_loss=1.017424`, ImageFolder `2.3903→2.1509`) 정확히 일치.
* legacy checkpoint 하위호환(`weight_decay`/`class_weights`/
  `cuda_rng_state`/`scaler_state_dict` 키 부재) 전부 정상.
* 대표 ImageFolder production run의 artifact(checkpoint/metadata/
  best model/class mapping/TorchScript/test result/training history)
  전부 정상 생성, `training_history.json`/checkpoint `history`
  서브딕트 schema가 Phase 4V 계약 그대로(`stop_reason`/
  `epoch_duration_seconds` 누출 없음) 유지됨을 직접 확인.
* TorchScript export, Python/C++ parity 전부 PASS.

**PHASE 4 GRADUATED.** 상세 결과와 known limitations/deferred
항목(Phase 5 backlog), Phase 5 handoff public API 목록은
`docs/phase4w_final_training_integration_graduation.md`를
참고하세요.

---

## Phase 5B: Application + Qt Worker Integration

Phase 5A(architecture investigation)에서 결정한 대로, Phase 4
training workflow를 PySide6 application에서 안전하게 실행할 수 있는
application/controller + Qt worker 계층을 추가했습니다. **실제 GUI
화면(Training Page/MainWindow)은 아직 없습니다** -- 그건 Phase 5C입니다.

* `src/image_ai_studio/application/training_controller.py` --
  PySide6를 전혀 모르는 framework-agnostic 계층. `TrainingController`
  가 `idle`/`running`/`stopping`/`finished`/`failed` state와 single
  active run, cooperative stop(`threading.Event`)을 관리하고,
  backend(기본값 `run_imagefolder_training_workflow`)를 주입 가능하게
  감쌉니다. `build_training_request()`는 UI 입력값을
  `ImageFolderWorkflowRequest`로 조립만 할 뿐 검증은 하지 않습니다
  (검증은 여전히 `TrainingConfig`/workflow 자신의 책임).
* `src/image_ai_studio/gui/qt_training_worker.py` -- `QtTrainingWorker`
  (`QObject`)가 `QThread`에서 실제 학습 전체(model 생성부터 완료까지)를
  실행하고, `progress`/`finished`/`failed` Qt signal로 결과를
  전달합니다. `QThread.terminate()`나 강제 kill은 쓰지 않습니다 --
  Phase 4의 기존 epoch 경계 cooperative stop 그대로입니다.
* PySide6 + `QThread` + CUDA 조합을 실제 로컬 GPU로 검증했습니다 --
  model 생성/`.to("cuda")`/forward·backward가 worker thread 안에서
  정상 동작하고 크래시/hang 없이 종료됩니다.
* **중요 발견**: `Signal`을 QObject가 아닌 평범한 함수/lambda에
  connect하면 GUI thread로 자동 queue되지 않고 emit이 일어난 worker
  thread에서 직접 실행됩니다(empirical 확인). Phase 5C는 반드시 실제
  QObject 메서드에 connect하거나 `QueuedConnection`을 명시해야
  합니다.
* training core(`src/image_ai_studio/training/*.py`)는 이번 Phase에서
  전혀 수정하지 않았습니다 -- 기존 public API만으로 충분했습니다.

설계 배경과 상세 계약(state model, single-active-run contract,
signal thread-affinity 경고, Phase 5C handoff contract)은
`docs/phase5b_application_qt_worker_integration_design.md`를
참고하세요.

---

## Phase 5C: Training GUI

Phase 5B의 `TrainingController`/`QtTrainingWorker` 위에 실제 사용
가능한 Training 화면을 올렸습니다. `python scripts/run_gui.py`로
실행하면 Model JSON/Dataset root/Output directory를 선택하고, Basic/
Advanced `TrainingConfig` 옵션과 device/precision을 설정하고, 학습을
시작·관찰·cooperative stop하고, 결과(완료/조기종료/사용자중단)와
artifact 경로를 확인할 수 있습니다.

* `src/image_ai_studio/gui/training_page.py` -- `TrainingPage`
  (`QWidget`)가 이 Phase의 핵심입니다. widget 값의 snapshot에서 Phase
  5B의 `build_training_request()`를 그대로 호출해 request를 조립하고
  (semantic validation은 여전히 `TrainingConfig`의 책임), `Start`를
  누를 때마다 새 `QThread`+`QtTrainingWorker`를 만들어 실행합니다
  (같은 `TrainingController`는 재사용) -- 여러 번 연속 학습을
  지원합니다.
* `src/image_ai_studio/gui/main_window.py` -- `MainWindow`
  (`QMainWindow`)는 `TrainingPage` 하나만 담는 얇은 창입니다. 학습
  도중 창을 닫으려 하면 확인 다이얼로그를 띄우고, 동의하면
  cooperative stop을 요청한 뒤 학습이 안전하게 끝난 뒤에만 실제로
  닫습니다(`QThread.terminate()`나 GUI thread를 막는 blocking wait는
  쓰지 않습니다).
* progress bar는 **`run_epoch`/`total_run_epochs`**를 씁니다(`
  global_epoch`을 분모/분자로 쓰면 resume 이후 잘못된 비율이 나오는
  버그가 있었음 -- Phase 4V/5B에서 이미 고친 계약을 여기서도 지킵니다).
* 결과 화면은 `result.stop_reason`을 그대로 읽어 상태 문구로
  매핑합니다(`history.stopped_early`/`stopped_by_user`를 재계산하지
  않음, single source of truth).
* Phase 5B의 signal thread-affinity 경고를 그대로 지켜 `worker.
  progress`/`finished`/`failed`를 전부 `TrainingPage`의 실제 QObject
  bound method에 connect했습니다(plain 함수/lambda 사용 없음).
* `scripts/run_gui.py`는 얇은 launcher입니다(`QApplication`+
  `MainWindow`+`show()`+`app.exec()`) -- import 자체로는 어떤 side
  effect도 없습니다.
* `src/image_ai_studio/training/*.py`, `TrainingController`/
  `QtTrainingWorker` 아키텍처는 이번 Phase에서 전혀 수정하지
  않았습니다.
* stabilization 라운드에서 드물게 재현되던 native abort(`Fatal
  Python error: Aborted`)의 원인을 `worker.deleteLater()`가
  `thread.finished`(worker thread의 event loop가 이미 멈춘 뒤)에
  연결돼 있던 것으로 좁혀 Qt canonical `moveToThread` 패턴(worker
  자신의 `finished`/`failed`에 연결)으로 고쳤습니다 -- 반복 실행
  regression test로 고정.

설계 배경과 상세 계약(field 매핑, QThread lifecycle, close 처리,
테스트 구성, QThread lifecycle stabilization 내역)은
`docs/phase5c_training_gui_design.md`를 참고하세요.

---

## Phase 5D: Final Integration Validation & Graduation — **PHASE 5 COMPLETE**

Phase 5A(조사)→5B(application/Qt worker)→5C(Training GUI) 위에서
새 기능을 추가하지 않고, 실제 사용자 관점에서 하나의 완성된 학습
애플리케이션으로 정상 동작하는지 최종 통합 검증했습니다. **Phase 5는
이 라운드로 종료됩니다** -- 이후 신규 기능은 별도 Phase에서 다룹니다.

**GUI 실행 방법**:

```bash
python scripts/run_gui.py
```

**GUI에서 가능한 것**: Model JSON/Dataset root/Output directory 선택,
Basic(epochs/batch size/learning rate/optimizer/device/precision) +
Advanced(momentum/weight decay/gradient clip/label smoothing/class
weights/LR scheduler/early stopping/checkpoint/pin memory/non
blocking/TorchScript export/seed) 설정, CPU/CUDA(`cuda`/`cuda:N`)
device 선택, fp32/fp16/bf16 precision 선택, resume checkpoint 선택
(optional), 학습 시작·진행률 관찰·cooperative stop, 완료/조기종료/
사용자중단 결과와 실패 메시지·artifact 경로 확인, 완료 후 같은 창에서
새 학습 반복 시작, 학습 중 창 닫기 시 확인 후 안전한 지연 종료.

**이번 라운드에서 확인한 것**(전부 실제 코드/실제 실행 기준):

* 실제 `python scripts/run_gui.py` 실행과 실제 `TrainingPage.
  _start_button`에 대한 실제 mouse click(`QTest.mouseClick`)으로 GUI
  전체 흐름(Model/Dataset/Output 지정 → Basic/Advanced 설정 → Start →
  Running 중 모든 configuration control과 Browse/Clear 버튼 6개 비활성
  → 완료 → 재활성)을 end-to-end로 확인 -- traceback/Qt warning 없음.
* 실제 tiny ImageFolder + 실제 `ModelSpec`으로 CPU 학습을 실제 GUI
  경로(`TrainingPage`→`TrainingController`→`QtTrainingWorker`→
  `run_imagefolder_training_workflow()`)로 완주, test loss/accuracy
  표시와 best model/training history/class mapping/test result
  artifact 생성을 확인.
* resume checkpoint를 만든 뒤 **새 `TrainingPage`**에서 `resume_from`
  필드로 이어받아 추가 epoch를 실행 -- `global_epoch`은 누적(2)되고
  progress bar는 여전히 `run_epoch`/`total_run_epochs`(1/1)만 쓰는
  계약을 실측으로 재확인.
* 이 머신의 실제 GPU(`torch.cuda.is_available() == True`, 1 device)로
  `device="cuda"` 선택 → 실제 QThread → 실제 워커 → 실제 학습
  완료까지 GUI thread를 막지 않고 정상 동작, progress handler가
  main thread에서 실행됨을 재확인.
* 실제(fake가 아닌) CPU backend로 cooperative stop
  (Start → Stop → "Stopping after current epoch..." → 실제 학습
  중단 → "Training stopped by user")과, 실제 backend가 활성 상태일 때
  `MainWindow.close()`(확인 다이얼로그 → 지연 close → 학습 종료 후
  실제 종료)를 각각 재확인.
* repeated-run/QThread cleanup stabilization regression
  (`test_repeated_run_thread_lifecycle_stress`, `test_main_window.py`)
  을 반복 실행해 native abort 재발이 없음을 재확인.
* 전체 `pytest -q` 764/764 PASS를 여러 차례 반복 실행, native abort
  0회.

**이번 라운드에서 발견했지만 수정하지 않은 것**: Start 클릭 직후
worker thread가 아직 `TrainingController.begin_run()`을 호출하기 전
(실측 1ms 미만의 창)에 곧바로 Stop을 호출하면 `request_stop()`이
조용히 no-op되는 race를 실측으로 확인했습니다. 이 창은 사람이 마우스로
두 버튼을 연달아 클릭하는 데 걸리는 시간(통상 100ms 이상)보다 훨씬
짧아 현재 일반적인 GUI 마우스 조작으로 재현될 가능성은 매우 낮다고
판단합니다. 올바른 수정(`begin_run()`을 GUI thread로 옮기는 것)은
`QtTrainingWorker`의 기존 공개 계약을 바꾸는 architecture 변경이라
이번 verification-first 라운드의 범위 밖으로 판단해 코드는 건드리지
않고 `docs/phase5_final_integration.md`의 known limitations에
기록만 했습니다.

**non-goals(Phase 5 전체, 계속 유효)**: 그래프/차트, 실험 이력 DB,
multi-run, inference GUI, packaging/installer, custom theme/style,
새 optimizer/scheduler/dataset/model layer, training core나
`TrainingController`/`QtTrainingWorker` architecture 변경.

Phase 5 전체 architecture, runtime flow, QThread ownership/deleteLater
계약, 최종 test coverage, known limitations, graduation 판정 근거는
`docs/phase5_final_integration.md`를 참고하세요.

---

## 현재 지원 범위

* Sequential 기반 Model Definition (`ModelSpec`/`LayerSpec`, JSON
  직렬화)
* `ResidualBlock` (Phase 2)
* `Branch` Add / channel `Concat` + `Identity` skip path (Phase 3)
* Classification 학습, loss=CrossEntropyLoss 고정(label smoothing 계수는
  선택 가능, training loss에만 적용), optimizer는 Adam/SGD/AdamW 선택
  가능, weight decay(L2 정규화) 공통 적용 (Phase 4E, Phase 4L, Phase 4N)
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
* `run_training()`의 epoch 경계 progress callback + 협조적(cooperative)
  stop(`should_stop`), `TrainingHistory.stopped_by_user` 기록, resume
  계속 가능 (Phase 4I)
* `run_training()`의 epoch 경계 `checkpoint_hook` + `--checkpoint-every N`
  (global epoch 기준 자동 저장), checkpoint/metadata 원자적 저장,
  in-place resume이 아닌 경로의 기존 checkpoint 재사용 거부 (Phase 4J)
* `scripts/train_imagefolder.py`의 Ctrl+C(SIGINT) graceful cooperative
  stop -- 첫 번째 Ctrl+C는 다음 epoch 경계에서 안전하게 중단(exit 0),
  두 번째 Ctrl+C는 즉시 강제 종료(exit 130) (Phase 4K)
* `--weight-decay`(상한 없이 0 이상, 기본값 0.0)와 `--optimizer adamw`,
  Phase 4L 이전 checkpoint에 대한 `weight_decay` 전용 하위 호환 resume
  규칙 (Phase 4L)
* `--gradient-clip-norm`(L2 norm clipping, 상한 없는 양수, 기본값 없음
  = 비활성화) -- checkpoint의 `training_config`에는 저장되지만 resume
  compatibility 비교 대상은 아니라서 resume 시 자유롭게 변경 가능
  (Phase 4M)
* `--label-smoothing`(`[0.0, 1.0]`, 기본값 0.0), training loss에만
  적용 -- 마찬가지로 checkpoint에는 저장되지만 resume compatibility
  비교 대상은 아니라서 resume 시 자유롭게 변경 가능 (Phase 4N)
* ImageFolder 최종 test 평가에서 confusion matrix + per-class recall +
  macro precision/recall/F1을 계산해 `test_result.json`에 저장(test-only,
  validation/`TrainingHistory`/checkpoint 무영향) (Phase 4O)
* `--class-weights`(class별 명시적 explicit weight, tuple, 0보다 큰 유한한
  값만 허용, training loss에만 적용, label smoothing과 조합 가능) --
  마찬가지로 checkpoint에는 저장되지만 resume compatibility 비교 대상은
  아니라서 resume 시 자유롭게 변경 가능 (Phase 4P)
* `--device`(`cpu`/`cuda`/`cuda:N`, 기본값 `cpu`)로 ImageFolder 학습
  device 선택 -- runtime 실행 파라미터로 `TrainingConfig`/
  `RESUME_CONFIG_FIELDS`와 무관, CUDA 미가용/index 범위 초과 시 명확히
  거부(silent CPU fallback 없음), 최종 test/TorchScript export는 항상
  CPU 유지, CPU→CPU exact-resume 유지 (Phase 4Q)
* same physical CUDA device / 같은 머신 / 같은 소프트웨어 환경 기준의
  CUDA exact-resume(`cuda_rng_state` checkpoint + CUDA training 구간에만
  scoped deterministic execution 자동 적용, workflow 종료 시 caller의
  전역 설정 원복) -- CPU↔CUDA는 Phase 4Q state portability 계약만 유지
  (exact 아님), cross-GPU architecture는 bitwise exact 미검증/미보장,
  multi-GPU/distributed는 training 자체가 미지원 (Phase 4R)
* `--precision {fp32,fp16}`(기본값 `fp32`)로 CUDA FP16 AMP training
  (`torch.amp.autocast`+`torch.amp.GradScaler`) 선택 -- `RESUME_CONFIG_FIELDS`
  와 무관(gradient_clip_norm/label_smoothing/class_weights와 같은 범주),
  `cpu`+`fp16` 조합은 명확히 거부(silent CPU fallback 없음), AMP+gradient
  clipping 통합(`scaler.unscale_()` 순서 보장), validation/test/export는
  항상 FP32 유지, checkpoint의 `scaler_state_dict`로 same-device AMP
  exact-resume 지원(scaler state가 있는 신규 checkpoint만 exact 보장,
  precision을 바꾼 resume은 portable-only) (Phase 4S)
* `--precision bf16`으로 CUDA BF16 mixed precision training 선택 --
  GradScaler를 쓰지 않고(`torch.amp.autocast(dtype=torch.bfloat16)`만),
  `cpu`/기타 non-CUDA backend와 조합하면 fp16과 동일하게 명확히 거부,
  checkpoint `scaler_state_dict`는 항상 `None`(새 checkpoint field
  없음), same-device BF16 exact-resume 지원, hardware capability를
  강제 검증하지 않음(기능 지원과 속도 보장은 별개) (Phase 4T)
* `--pin-memory`/`--non-blocking`(둘 다 기본값 `False`, 서로 독립적으로
  설정 가능)으로 CUDA ImageFolder training의 host→device batch 전송을
  선택적으로 최적화 -- `--device cpu`면 항상 무시(effective 값 강제
  `False`), `RESUME_CONFIG_FIELDS`와 무관하며 resume 전후로 값이
  달라져도 exact-resume 유지(새 checkpoint field 없음), `num_workers`/
  `persistent_workers`는 이번 Phase에서 의도적으로 미노출, 성능 향상
  비보장 (Phase 4U)
* `TrainingProgress.epoch_duration_seconds`(session-local wall-clock,
  checkpoint_hook까지 포함/progress_callback 자신은 제외)와
  `TrainingResult`/`ImageFolderWorkflowResult`의 `stop_reason`
  (`"completed"`/`"early_stopped"`/`"user_stopped"`) -- 기존 callback
  ordering/checkpoint/`TrainingHistory` JSON schema 전부 무변경
  (Phase 4V)
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
* class imbalance 자동 처리 -- explicit class weight(사용자가 직접 지정)는
  Phase 4P에서 지원하지만, automatic(inverse-frequency 등) 계산과
  `WeightedRandomSampler`/oversampling/undersampling은 미지원(위 "Phase 4P"
  절 참고), class-name 기반 weight 지정 문법도 미지원(index/tuple 순서만)
* validation epoch별 상세 classification metric(confusion matrix 등 --
  test 평가에서만 지원, 위 "Phase 4O" 절 참고), metric 기반 early
  stopping/scheduler, per-class precision/F1 노출, micro/weighted
  average, ROC-AUC/PR-AUC, top-k accuracy, specificity, CLI에 macro
  F1 등 상세 지표 출력(상세 지표는 `test_result.json` 파일로만 제공)
* loss function 종류 선택(CrossEntropyLoss 고정 -- BCE/multilabel/
  focal loss/custom loss/regression loss는 미지원, label smoothing
  계수는 Phase 4N에서, class별 explicit weight는 Phase 4P에서 지원),
  validation/test 시의 label smoothing/class weighting(항상
  unsmoothed/unweighted), Adam betas/eps, SGD dampening/nesterov,
  `"plateau"` 외 LR scheduler(StepLR/CosineAnnealingLR 등), scheduler
  threshold/cooldown/min_lr, gradient value clipping, custom gradient
  `norm_type`, `error_if_nonfinite` 노출, gradient norm history/metric
  기록 (L2 norm clipping 자체는 Phase 4M에서 지원 -- 위 "Phase 4M" 절 참고)
* resume 시 config 자유 변경 (optimizer/learning_rate/momentum/
  weight_decay/lr_scheduler/lr_scheduler_factor/lr_scheduler_patience/
  batch_size는 checkpoint와 반드시 일치해야 함 -- weight_decay만 Phase 4L
  이전 checkpoint에 한해 누락 시 0.0으로 간주하는 좁은 예외가 있음),
  CPU↔CUDA/cross-GPU bitwise exact-resume(same-device CUDA exact-resume
  자체는 Phase 4R에서 지원 -- 위 "Phase 4R" 절 참고), multi-GPU/distributed
  training(exact-resume 이전에 training 기능 자체가 미지원), batch-level
  (worker/sampler iterator) resume, distributed checkpoint
* DataLoader `num_workers`/`persistent_workers`/`prefetch_factor` 노출
  (`num_workers>0` 자체는 exact-resume이 안전함을 실측 확인했지만, 이
  프로젝트의 전형적인 작은 데이터셋 규모 + Windows spawn 환경에서
  오히려 뚜렷하게 느려져 의도적으로 미노출, `persistent_workers=True`는
  현재 checkpoint/resume 설계와 구조적으로 충돌해 exact-resume이 깨짐
  -- host→device 전송 최적화(`pin_memory`/`non_blocking`) 자체는 Phase
  4U에서 지원, 위 "Phase 4U" 절 참고), worker RNG state/prefetch queue
  checkpoint
* batch-level progress streaming, stage/phase event(training/
  validation/checkpoint saving 등 epoch 중간 상태), 별도 completion
  event, resume 간 누적(cumulative) elapsed time, ETA 추정(epoch 단위
  `epoch_duration_seconds`/`stop_reason` 자체는 Phase 4V에서 지원 --
  위 "Phase 4V" 절 참고)
* 기존 `--checkpoint-out` 경로를 명시적으로 덮어쓰도록 강제하는 옵션
  (예: `--overwrite-checkpoint`) -- in-place resume(`--resume-from`과
  `--checkpoint-out`이 같은 경로) 외에는 항상 새 경로가 필요함 (Phase 4J)
* `SIGTERM`/`SIGHUP` graceful shutdown, batch 중간 cancellation, GUI stop
  button(Ctrl+C cooperative stop 자체는 Phase 4K에서 지원 -- 위 "Phase 4K"
  절 참고)
* CPU AMP(CUDA FP16/BF16 AMP 자체는 각각 Phase 4S/4T에서 지원 -- 위
  "Phase 4S"/"Phase 4T" 절 참고), multi-GPU/distributed training,
  gradient accumulation, GradScaler tuning parameter CLI/config 노출,
  AMP inference/export, FP8
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

현재 `requirements.txt`(Phase 5B부터 PySide6/GUI 관련 패키지 포함 --
PyTorch와 달리 GPU 환경에 따라 빌드가 달라지지 않으므로 여기 포함됩니다):

```text
filelock==3.32.0
fsspec==2026.6.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
packaging==26.0
PySide6==6.11.1
PySide6_Addons==6.11.1
PySide6_Essentials==6.11.1
shiboken6==6.11.1
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
