# Phase 4C: 실제 이미지 데이터셋 (torchvision / CIFAR-10)

이 문서는 설계 검토와 실제 구현/실행 결과를 함께 정리한다. Phase 4A/4B는
synthetic dataset(외부 다운로드 없는 랜덤 패턴 이미지)만으로 학습 루프
자체를 검증했다. Phase 4C의 목표는 **높은 CIFAR-10 정확도를 만드는 것이
아니라**, 그 학습 루프를 실제 이미지 dataset(torchvision `CIFAR10`)으로
처음부터 끝까지 연결하는 것이다:

```text
ModelSpec JSON -> build_model()
    -> torchvision CIFAR-10 (공식 train split -> Train/Validation 결정론적 분리)
    -> 기존 학습 루프(train_one_epoch/evaluate/run_training, 변경 없음)
    -> best epoch 추적 (변경 없음)
    -> 공식 CIFAR-10 test split으로 best model 최종 평가 (신규)
    -> TorchScript export (변경 없음)
    -> C++ LibTorch inference (변경 없음)
    -> Python/C++ parity (변경 없음)
```

## 1. 범위

`training/loop.py`(`train_one_epoch`/`evaluate`/`run_training`/
`TrainingHistory`/`TrainingResult`), `training/config.py`,
`training/checkpoint.py`, `training/history.py`, `training/dataset.py`
(synthetic), `model_definition/*`, `export/*`, C++ 러너는 전혀 수정하지
않는다. Phase 4C가 새로 하는 일은 정확히 두 가지뿐이다:

1. 실제 이미지(torchvision `Dataset`)를 만들어 기존 `DataLoader`/학습
   루프에 넘겨주는 것
2. best epoch 확정 후, 공식 test split에 대해 기존 `evaluate()`를
   딱 한 번 호출해 최종 성능을 기록하는 것

## 2. torchvision 의존성

`torch==2.12.1+cu126`을 그대로 유지해야 했으므로, 최신 `torchvision`을
그냥 설치하지 않고 먼저 호환 조합을 확인했다. `pip install
torchvision==X+cu126 --dry-run`을 여러 버전에 대해 시도한 결과:

* `torchvision`(최신, 0.28.0)은 `torch==2.13.0`을 요구 (업그레이드 필요 -- 거부)
* `torchvision==0.27.1+cu126`은 dry-run에서 `torch==2.12.1`(이미 설치된
  버전)을 그대로 요구 -- **정확히 일치**

`torchvision==0.27.1+cu126`을 설치했고, 설치 후 `torch.__version__`이
여전히 `2.12.1+cu126`, `torch.cuda.is_available()`이 여전히 `True`임을
확인했다. Pillow가 새 transitive 의존성으로 추가된 것 외에 다른 부작용은
없었다.

`requirements.txt`/`requirements-dev.txt`는 수정하지 않았다 -- 기존에도
PyTorch 계열 패키지(GPU/OS별로 wheel이 다름)는 의도적으로
`requirements.txt`에 포함하지 않고 README에 설치 명령으로만 안내하는
정책이었고, `torchvision`도 `torch`와 동일한 CUDA index에서 짝을 맞춰
설치해야 하므로 같은 정책을 따랐다 (README의 "PyTorch 별도 설치" 절에
`torchvision` 설치 명령을 추가).

## 3. 디렉터리 구조

```text
src/image_ai_studio/training/
    dataset.py              (변경 없음) synthetic dataset
    torchvision_dataset.py  (신규) torchvision 기반 real dataset

tests/training/
    test_dataset.py              (변경 없음)
    test_torchvision_dataset.py  (신규, 오프라인)

scripts/
    run_training_e2e.py       (변경 없음) synthetic E2E
    run_real_training_e2e.py  (신규) real-image E2E

examples/models/
    phase4_training_model.json  (변경 없음)
    phase4c_cifar10_model.json  (신규)
```

`dataset.py`(synthetic)를 CIFAR-10으로 바꾸거나 지우지 않고 별도 모듈로
분리했다 -- 둘은 역할이 다르다:

* `dataset.py` -> offline/deterministic pytest, 학습 루프 자체의 회귀
  검증 (네트워크 없음)
* `torchvision_dataset.py` -> 실제 이미지 학습 E2E (첫 실행 시 네트워크
  다운로드 발생)

`training/` 패키지 전체를 `datasets/` 서브패키지로 재구성하는 것은
검토했으나 채택하지 않았다 -- 파일 2개짜리 이번 변경에 비해 과한
리팩터링이고, 향후 데이터셋이 늘어나 실제로 필요해지면 그때 옮겨도
`import` 경로 변경 외의 비용이 없다.

## 4. Train / Validation / Test 분리

CIFAR-10은 공식적으로 train 50,000 / test 10,000 두 split만 제공한다.
Validation은 이 프로젝트가 공식 train split을 다시 나눠서 만든다:

```python
DEFAULT_VAL_FRACTION = 0.1  # 공식 train 50,000 -> train 45,000 / val 5,000

def make_cifar10_train_val_datasets(input_shape, root, seed, val_fraction=DEFAULT_VAL_FRACTION, download=True):
    full_train = CIFAR10(root=root, train=True, download=download, transform=transform)
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(full_train, [train_size, val_size], generator=generator)

def make_cifar10_test_dataset(input_shape, root, download=True):
    return CIFAR10(root=root, train=False, download=download, transform=transform)
```

세 역할은 함수 수준에서 구조적으로 분리된다:

* **Train** -- `make_cifar10_train_val_datasets()`의 첫 번째 반환값만
  optimizer/backward에 사용 (`run_training()`, 변경 없음)
* **Validation** -- 같은 함수의 두 번째 반환값만 매 epoch
  `evaluate()`에 사용, best epoch 선택 기준 (`run_training()`, 변경
  없음)
* **Test** -- `make_cifar10_test_dataset()`은 완전히 별도 함수이자 별도
  CIFAR-10 split(공식 test)에서 만들어진다. `run_real_training_e2e.py`는
  `test_dataset`/`test_loader`를 다른 두 DataLoader와 함께 학습 전에 미리
  생성해 두지만, **`run_training()`에는 전달하지 않는다** -- best epoch
  선택 루프(`run_training()`)는 `train_loader`/`val_loader`만 인자로
  받으므로 test dataset을 참조할 방법 자체가 없다. `test_loader`가 실제로
  **사용되는** 시점은 `training_result.best_state_dict`가 이미 확정된
  이후, `evaluate(best_model, test_loader, ...)`가 한 번 호출될 때뿐이다.
  즉 test dataset/loader의 **생성 시점**이 아니라 **사용 시점과 용도**가
  leakage를 막는다.

Train/Validation 분리는 `torch.Generator().manual_seed(seed)`(전용
generator, 전역 RNG 아님)로 결정론적이다 -- 같은 `seed`면 항상 같은
분할이 나온다.

## 5. Transform / 전처리

```python
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

def build_transform(input_shape):
    _, height, width = input_shape
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
```

* Resize를 `ModelSpec.input_shape`에서 직접 읽어온다 -- CIFAR-10 고유의
  32x32를 하드코딩하지 않았으므로, 다른 해상도의 `ModelSpec`이나 다른
  dataset에도 그대로 재사용 가능하다.
* CIFAR-10 채널별 평균/표준편차로 튜닝한 값이 아니라 `[0,1] -> [-1,1]`로
  옮기는 범용 정규화를 의도적으로 사용했다 -- 이번 Phase 목표가
  벤치마크 정확도가 아니라 경로 검증이라는 점과 일치시켰다.
* augmentation(RandomCrop/RandomHorizontalFlip/ColorJitter/RandAugment/
  AutoAugment 등)은 포함하지 않는다. Train/Validation/Test 전부 완전히
  동일한 deterministic transform을 쓴다.
* `tests/training/test_torchvision_dataset.py`의
  `test_build_transform_generalizes_to_imagefolder_dataset`에서, 같은
  `build_transform()`을 `torchvision.datasets.ImageFolder`(CIFAR-10이
  아닌 다른 dataset)에 적용해도 정상 동작함을 오프라인으로 증명한다 --
  CIFAR-10 전용 코드가 아니라는 근거.

## 6. RGB 계약

```python
def _require_rgb_input_shape(input_shape):
    channels = input_shape[0]
    if channels != 3:
        raise ValueError(
            "real-image classification requires a 3-channel (RGB) "
            f"input_shape, got input_shape={tuple(input_shape)} (channels={channels})"
        )
```

`make_cifar10_train_val_datasets()`/`make_cifar10_test_dataset()` 둘 다
`CIFAR10(...)`을 인스턴스화하기 **전에** 이 검증을 통과해야 한다 --
grayscale(`input_shape[0] == 1`) 등으로 억지로 맞추지 않고, 명확한
`ValueError`로 즉시 거부한다. 이 순서 덕분에 `download=False`로도
안전하게 오프라인 pytest를 작성할 수 있었다 (RGB 검증이 실패하면
`CIFAR10.__init__`까지 도달하지 않으므로 네트워크 접근이 아예 발생하지
않는다).

## 7. Subset 제한 (`limit_dataset`)

```python
def limit_dataset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))
```

`torch.utils.data.Subset`으로 앞부분 `limit`개만 취한다 (무작위 샘플링이
아니라 결정론적으로 첫 N개 -- E2E 재현성을 위해 의도적으로 무작위화하지
않았다). `run_real_training_e2e.py`는 `--train-limit`/`--val-limit`/
`--test-limit` CLI 인자로 이 값을 받고, 기본값은 빠른 E2E 검증을 위한
작은 subset(256/64/128)이다. `0` 이하를 넘기면 "제한 없음"으로 해석되어
전체 split(45,000/5,000/10,000)을 그대로 쓸 수 있다 -- 코드 경로 자체는
전체 데이터셋도 다룰 수 있게 설계되어 있고, 기본 CLI 값만 실행 시간을
줄이기 위해 작게 잡았다.

## 8. DataLoader

새 `DataLoaderConfig` 추상화는 만들지 않았다 -- `batch_size`는 기존
`TrainingConfig`에서 그대로 가져오고, 나머지(`shuffle`, `num_workers`
등)는 스크립트 안에서 표준 `torch.utils.data.DataLoader` 인자로 직접
지정하는 것으로 충분했다 (설정할 축이 3~4개뿐이라 별도 dataclass를
두면 오히려 간접 계층만 늘어남).

```python
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True,
    generator=loader_generator, drop_last=True, num_workers=0,
)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
```

* `train_loader`만 `shuffle=True` (전용 `torch.Generator`로 고정,
  재현 가능), `val_loader`/`test_loader`는 `shuffle=False` -- best epoch
  판단과 최종 평가가 배치 순서에 영향받지 않도록 한다.
* `drop_last=True`는 train에만 적용한다 -- 작은 마지막 batch가
  training-time `BatchNorm2d` 동작에 영향을 주는 것을 피하고 E2E의
  train batch 구성을 일정하게 유지하기 위함이다 (val/test는
  `evaluate()`가 `model.eval()`이라 BatchNorm이 배치 통계를 쓰지
  않으므로 마지막 배치 크기와 무관하다).
* `num_workers=0`을 선택했다 -- Windows에서 멀티프로세스 DataLoader
  워커는 `if __name__ == "__main__":` 가드, spawn 방식의 pickling 제약
  등 별도 처리가 필요한데, 이번 Phase의 CIFAR-10 이미지 크기와 subset
  규모에서는 단일 프로세스로도 충분히 빠르고, 플랫폼에 안전한 값을
  우선했다 (요청 사항).

## 9. Test 평가

```python
test_loss, test_accuracy = evaluate(best_model, test_loader, device="cpu")
```

새 dataclass를 만들지 않고 `evaluate()`(Phase 4A, 변경 없음)를 그대로
재사용해 `(loss, accuracy)` 튜플을 받는다. `TrainingHistory`는 요청대로
train/val만 다루도록 그대로 두고, test 결과는 별도의 작은 JSON
(`{"test_loss": ..., "test_accuracy": ...}`)으로
`artifacts/training/{model_name}_test_result.json`에 저장한다 -- epoch
history에 억지로 끼워 넣지 않는다.

**Test는 best epoch 선택에 전혀 관여하지 않는다.** `test_dataset`/
`test_loader` 자체는 다른 DataLoader들과 함께 학습 전에 미리 만들어져
있지만, `run_training()`은 `train_loader`/`val_loader`만 인자로 받으므로
그 안에서 test dataset을 참조할 수 없다. `test_loader`가 실제로
**사용되는** 지점은 `run_training()`(best epoch 확정) -> `best_model`
빌드/저장 이후, `evaluate(best_model, test_loader, ...)`가 호출되는
한 곳뿐이다 -- leakage를 막는 것은 변수의 생성 시점이 아니라, 학습/best
epoch 선택 경로에 test 결과가 전달되지 않는다는 사용 시점과 용도다.

## 10. `phase4c_cifar10_model.json`

```json
{
  "name": "phase4c_cifar10_model",
  "input_shape": [3, 32, 32],
  "layers": [
    {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "batch_norm2d"},
    {"type": "relu"},
    {"type": "max_pool2d", "kernel_size": 2, "stride": 2, "padding": 0},
    {"type": "conv2d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "batch_norm2d"},
    {"type": "relu"},
    {"type": "max_pool2d", "kernel_size": 2, "stride": 2, "padding": 0},
    {"type": "adaptive_avg_pool2d", "output_size": 1},
    {"type": "flatten"},
    {"type": "linear", "out_features": 10}
  ]
}
```

CIFAR-10의 `input_shape=[3,32,32]`, `final output=[10]`(10 classes)에
맞춘 작은 CNN이다. Phase 1~3에서 이미 검증된 layer 타입(`conv2d`/
`batch_norm2d`/`relu`/`max_pool2d`/`adaptive_avg_pool2d`/`flatten`/
`linear`)만 재사용했다 -- `ResidualBlockSpec`/`BranchSpec`은 이미
`phase4_training_model.json`(synthetic E2E)에서 학습 경로까지 검증됐으므로,
이번 모델에서는 목적(실제 이미지 경로 연결 검증)에 맞게 의도적으로
가볍게 유지해 E2E 실행 시간을 줄였다.

## 11. `run_real_training_e2e.py`

`scripts/run_training_e2e.py`(synthetic)와 동일한 단계별 PASS/FAIL 출력
구조를 따르되, 두 지점만 다르다:

```text
ModelSpec 로드/검증 (변경 없음)
    -> classification 출력 shape 확인 (final_shape == (10,) 정확히 일치 요구 -- CIFAR-10 전용)
    -> build_model() (변경 없음)
    -> make_cifar10_train_val_datasets() + make_cifar10_test_dataset() (신규)
    -> limit_dataset() 3곳 적용 (신규)
    -> DataLoader 3개 구성 (신규)
    -> run_training() (변경 없음)
    -> best epoch 저장/재로드 (변경 없음)
    -> evaluate(best_model, test_loader) (신규, best epoch 확정 이후 1회만)
    -> TorchScript export (변경 없음)
    -> C++ CPU/CUDA parity (변경 없음)
```

`scripts/run_training_e2e.py` 자체는 이번 Phase에서 한 줄도 수정하지
않았다 -- Phase 4A/4B 회귀 검증용 synthetic 경로로 그대로 남아 있다.

## 12. 신규 오프라인 unit test

`tests/training/test_torchvision_dataset.py` (8개, 전부 네트워크 접근
없이 통과):

* RGB 계약 위반 시 `CIFAR10` 인스턴스화 전에 `ValueError`가 발생함을
  `download=False`로 안전하게 검증 (2개)
* `build_transform()`이 `input_shape`에 맞게 resize하고 `float32`
  텐서를 반환하는지 (1개)
* `build_transform()`의 정규화 결과가 흰 픽셀에 대해 기대값과 일치하는지
  (1개)
* `build_transform()`을 `ImageFolder`(다른 torchvision dataset)에
  적용해도 동작함을 증명 -- CIFAR-10 전용이 아님을 실제 코드로 입증
  (1개)
* `limit_dataset()`이 `None`/초과 limit에서 원본을 그대로 반환하고,
  일반 limit에서는 결정론적으로 앞 N개만 취하는지 (3개)

CIFAR-10 자체(다운로드가 필요한 부분)는 pytest에서 다루지 않는다 --
오프라인/결정론적 정책을 유지하기 위해, 실제 다운로드/학습은
`scripts/run_real_training_e2e.py`(수동/E2E 전용)에서만 일어난다.

## 13. 실제 변경/추가 파일

| 파일 | 변경 내용 |
|---|---|
| `src/image_ai_studio/training/torchvision_dataset.py` | 신규 -- CIFAR-10 기반 real dataset 로더 |
| `tests/training/test_torchvision_dataset.py` | 신규 -- 오프라인 unit test 8개 |
| `scripts/run_real_training_e2e.py` | 신규 -- real-image E2E |
| `examples/models/phase4c_cifar10_model.json` | 신규 |
| `docs/phase4c_real_dataset_design.md` | 신규 (이 문서) |
| `README.md` | torchvision 설치 안내, Phase 4C 절, 현재 지원 범위 갱신 |

**변경 없음**: `training/config.py`, `training/dataset.py`(synthetic),
`training/loop.py`, `training/checkpoint.py`, `training/history.py`,
`model_definition/*`, `export/*`, `parity/*`, C++ 코드 전부,
`scripts/run_training_e2e.py`, `scripts/run_phase1_e2e.py`,
`requirements.txt`, `requirements-dev.txt`.

## 14. 실제 실행 검증 결과

Windows 11, PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, GTX 1080에서
전부 실제로 실행하여 확인했다 (추정치 없음):

* **신규 Phase 4C unit test**: `tests/training/test_torchvision_dataset.py`
  8 passed
* **`tests/training/` 전체**: 46 passed
* **전체 `pytest`**: 203 passed
* **Phase 0 regression** (`scripts/run_torchscript_tests.py`):
  `tiny_cnn`/`tiny_residual_cnn` CPU/CUDA 전부 PASS
* **Phase 1~3 E2E regression** (`scripts/run_phase1_e2e.py`, 4개 예시
  JSON): `phase1_e2e_model`/`phase1_e2e_alt_model`/
  `phase2_residual_model`/`phase3_branch_model` 전부 ModelSpec/build/
  TorchScript export/C++ CPU/CUDA parity PASS
* **기존 synthetic `run_training_e2e.py`**: 기존과 동일하게 재실행,
  training loss 1.3386 -> 0.2867, best epoch 10, best val loss 0.1687,
  best model save/reload PASS, TorchScript export PASS, C++ CPU/CUDA
  parity PASS -- Phase 4C 추가로 인한 회귀 없음 확인
* **신규 CIFAR-10 real-image E2E**
  (`python scripts/run_real_training_e2e.py`, 기본 subset
  train=256/val=64/test=128, epochs=5, batch_size=8):

  ```text
  ModelSpec: PASS (11 layers, final shape (10,))
  Classification output check: PASS (num_classes=10)
  PyTorch model build: PASS
  CIFAR-10 dataset: train=256 val=64 test=128
  Training:
    epoch 1: train_loss=2.3558 val_loss=2.2770 val_acc=0.1719
    epoch 2: train_loss=2.2214 val_loss=2.2569 val_acc=0.1875
    epoch 3: train_loss=2.1500 val_loss=2.2324 val_acc=0.1562
    epoch 4: train_loss=2.1150 val_loss=2.1933 val_acc=0.2656
    epoch 5: train_loss=2.0817 val_loss=2.2238 val_acc=0.2500
    training loss decreased: True (2.3558 -> 2.0817)
  Best epoch: 4
  Best validation loss: 2.1933
  Test evaluation (best model, official CIFAR-10 test split):
    test_loss=2.1608 test_accuracy=0.1953
  Best model save/reload: PASS
  TorchScript export: PASS
  C++ TorchScript runner: CPU PASS, CUDA PASS
  Parity: PASS

  PHASE 4C E2E: PASS
  ```

  이 실행은 작은 subset(256 train 이미지, 5 epoch)만 사용했으므로
  test accuracy(19.53%, 10-class random chance인 10%보다는 높음)는
  벤치마크 성능이 아니라 "경로가 실제로 끝까지 연결되어 동작한다"는
  것을 보여주는 결과로 해석해야 한다 -- 애초에 이 Phase의 목표가 그
  경로 연결이었다.

## 15. 이번 Phase 4C에서 의도적으로 구현하지 않은 것

* augmentation (RandomCrop, RandomHorizontalFlip, ColorJitter,
  RandAugment, AutoAugment 등)
* optimizer/loss 선택, LR scheduler, early stopping
* optimizer state/epoch가 포함된 full checkpoint, resume
* mixed precision, multi-GPU/distributed training
* Detection/Segmentation training
* 일반 DAG, dataset registry/factory (`ImageFolder`, Oxford-IIIT Pet 등
  다른 dataset은 `build_transform()`/`limit_dataset()`을 그대로 재사용할
  수 있는 구조로 만들었지만, 이를 위한 통합 factory/registry는 아직
  만들지 않았다)
* PySide6 UI

이 목록은 필요성이 구체적으로 확인되기 전까지 보류하며, `ModelSpec`
구조나 `training/loop.py`를 바꾸지 않고 확장할 수 있는 지점(새 dataset
모듈 추가, DataLoader 설정 조정)에 위치시켰다.
