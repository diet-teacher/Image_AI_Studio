# Phase 4D: 사용자 ImageFolder 데이터셋

이 문서는 설계 검토와 실제 구현/실행 결과를 함께 정리한다. Phase 4C는
torchvision에 내장된 특정 dataset(CIFAR-10)을 학습 파이프라인에
연결했다. Phase 4D의 목표는 **특정 내장 dataset에 의존하지 않고,
사용자가 준비한 일반 이미지 폴더를 `torchvision.datasets.ImageFolder`로
읽어 같은 파이프라인에 연결하는 것**이다:

```text
사용자 이미지 폴더 (train/val/test로 이미 분리됨)
    -> ImageFolder (3 split)
    -> class_to_idx 일치 검증
    -> dataset 클래스 수 vs ModelSpec 최종 출력 shape 검증
    -> 기존 학습 루프(train_one_epoch/evaluate/run_training, 변경 없음)
    -> best epoch 추적 (변경 없음)
    -> class mapping JSON 저장/재로드 (신규)
    -> best model의 test split 최종 평가 (Phase 4C와 동일 패턴)
    -> TorchScript export -> C++ LibTorch inference -> Python/C++ parity
```

## 1. 범위

Phase 4C와 마찬가지로 `training/loop.py`, `training/config.py`,
`training/checkpoint.py`, `training/history.py`, `training/dataset.py`
(synthetic), `model_definition/*`, `export/*`, C++ 러너는 전혀 수정하지
않는다. `build_transform()`(Phase 4C)도 그대로 재사용한다.

Phase 4D가 새로 하는 일은 세 가지뿐이다:

1. 사용자가 미리 train/val/test로 분리해 둔 `ImageFolder` 폴더를 읽어
   기존 `DataLoader`/학습 루프에 넘겨주는 것
2. 세 split의 `class_to_idx`가 완전히 일치하는지, dataset의 실제
   클래스 수가 `ModelSpec` 최종 출력 shape와 일치하는지 학습 시작
   전에 검증하는 것
3. class 이름/인덱스 매핑을 JSON artifact로 저장/재로드하는 것

**자동 Train/Val/Test split은 이번 Phase 범위 밖이다** -- 사용자가
`dataset_root/cat/`, `dataset_root/dog/`처럼 클래스별 폴더만 준비한
경우를 프로그램이 알아서 나누는 기능은 만들지 않았고, 반드시
`train/`, `val/`, `test/`가 이미 분리되어 있어야 한다.

## 2. 디렉터리 구조

```text
src/image_ai_studio/training/
    torchvision_dataset.py   Phase 4C(CIFAR-10) 함수 + Phase 4D(ImageFolder) 함수 (동일 파일 확장)

tests/training/
    test_torchvision_dataset.py   (변경 없음) CIFAR-10 부분 테스트
    test_imagefolder_dataset.py   (신규) ImageFolder 부분 테스트, 완전 오프라인

scripts/
    run_real_training_e2e.py           (변경 없음) CIFAR-10 직접 로딩 E2E
    run_imagefolder_training_e2e.py     (신규) 사용자 ImageFolder E2E
    prepare_cifar10_imagefolder_fixture.py  (신규) 테스트용 fixture 준비 전용, 제품 기능 아님

examples/models/
    phase4c_cifar10_model.json  (변경 없음, Phase 4D에서 재사용)
```

`training/torchvision_dataset.py`를 그대로 확장하는 쪽을 선택했다 --
CIFAR-10과 ImageFolder 둘 다 "torchvision Dataset + `ModelSpec.input_shape`
기반 transform"이라는 같은 역할을 공유하고(`build_transform()`을 두
경로 모두 재사용), 파일을 나눴을 때 얻는 이득(모듈 크기)보다 공유
로직이 두 파일에 흩어지는 비용이 더 크다고 판단했다. 확장 후에도
CIFAR-10 관련 함수/ImageFolder 관련 함수는 파일 안에서 주석으로 섹션이
분리되어 있고, 새 파일/패키지(`datasets/` 서브패키지 등)를 만들지는
않았다.

## 3. 지원하는 폴더 구조

```text
dataset_root/
├─ train/
│  ├─ cat/
│  │  ├─ 001.jpg
│  │  └─ 002.jpg
│  └─ dog/
│     ├─ 001.jpg
│     └─ 002.jpg
├─ val/
│  ├─ cat/
│  └─ dog/
└─ test/
   ├─ cat/
   └─ dog/
```

`dataset_root/cat/`, `dataset_root/dog/`처럼 클래스 폴더만 있는 구조를
프로그램이 자동으로 train/val/test로 나누는 기능은 지원하지 않는다
(향후 Phase로 보류).

## 4. 로더 API

```python
class ImageFolderSplits(NamedTuple):
    train: ImageFolder
    val: ImageFolder
    test: ImageFolder
    classes: list[str]
    class_to_idx: dict[str, int]


def make_imagefolder_datasets(input_shape, root) -> ImageFolderSplits:
    _require_rgb_input_shape(input_shape)
    split_dirs = _require_split_directories(root)     # train/val/test 폴더 존재 확인
    transform = build_transform(input_shape)           # Phase 4C 재사용

    train_dataset = ImageFolder(str(split_dirs["train"]), transform=transform)
    val_dataset = ImageFolder(str(split_dirs["val"]), transform=transform)
    test_dataset = ImageFolder(str(split_dirs["test"]), transform=transform)

    _require_matching_classes(train_dataset, val_dataset, test_dataset)

    return ImageFolderSplits(train_dataset, val_dataset, test_dataset,
                              list(train_dataset.classes), dict(train_dataset.class_to_idx))
```

`train`/`val`/`test` 필드 타입은 원래 `Dataset`(더 넓은 타입)이었으나,
실제로 항상 `ImageFolder` 인스턴스를 그대로 반환하므로 `ImageFolder`로
좁혔다(리뷰 과정에서 발견 -- 동작 변경 없음, 타입 정확도만 개선).
`classes`/`class_to_idx`도 `train_dataset.classes`/`.class_to_idx`를
그대로 aliasing하지 않고 `list()`/`dict()`로 복사해서 반환한다 --
호출자가 `splits.classes`를 그 자리에서 수정해도(정렬 등)
`splits.train`(=`train_dataset`) 내부 metadata가 함께 바뀌지 않는다
(리뷰 과정에서 발견, 방어적 복사 추가).

요청된 예상 API(`make_imagefolder_datasets(input_shape, root)`)를 그대로
따르되, 반환값은 3-tuple 대신 `NamedTuple`(`ImageFolderSplits`)로
만들었다. 필드가 5개(`train`/`val`/`test`/`classes`/`class_to_idx`)라
`train, val, test = splits`처럼 3개로 바로 언패킹하면
`ValueError: too many values to unpack`가 난다 -- 필요하면
`train, val, test, *_ = splits`(나머지 2개를 한 변수로 묶음)를 쓰거나,
`splits.train`/`splits.classes`처럼 속성으로 접근한다.
`splits.classes`/`splits.class_to_idx`로 class mapping도 같은
반환값에서 바로 꺼낼 수 있어 별도 helper 호출이 필요 없다는 점은
그대로 유지된다.

## 5. Transform 재사용

CIFAR-10과 완전히 동일한 `build_transform(input_shape)`(Phase 4C,
변경 없음)를 그대로 호출한다:

```text
PIL image -> Resize(ModelSpec.input_shape) -> ToTensor -> Normalize
```

augmentation은 이번 Phase에서도 추가하지 않았다. Train/Val/Test 세
`ImageFolder` 인스턴스 모두 같은 `transform` 객체를 공유한다.

## 6. RGB 계약

Phase 4C의 `_require_rgb_input_shape()`를 그대로 재사용한다 --
`ModelSpec.input_shape[0] != 3`이면 `ImageFolder`를 인스턴스화하기 전에
`ValueError`로 거부한다. 이건 **모델의 `input_shape` 계약**(3채널만
허용)에 대한 검증이고, **원본 이미지 파일 자체의 PIL 모드**는 이
프로젝트 코드가 별도로 검사하지 않는다.

리뷰 과정에서 실제 동작을 재확인한 결과: `ImageFolder`가 별도
`loader=`를 지정하지 않으면 기본값인
`torchvision.datasets.folder.pil_loader`를 쓰는데, 이 함수는
`Image.open(f).convert("RGB")`를 무조건 호출한다. 즉 grayscale(`"L"`),
팔레트(`"P"`), 알파 채널 포함(`"RGBA"`) 등 어떤 PIL 모드의 원본
이미지를 넣어도 `ImageFolder.__getitem__`이 반환하는 이미지는 항상
이미 3채널 RGB로 변환되어 있다 -- 실제로 `L`/`P`/`RGBA` PNG를 만들어
`make_imagefolder_datasets((3, H, W), ...)`에 넣고 확인한 결과 텐서는
항상 `(3, H, W)`이었다(`tests/training/test_imagefolder_dataset.py::
test_make_imagefolder_datasets_converts_non_rgb_source_images_to_3_channels`).

이 변환은 **이 프로젝트가 구현한 기능이 아니라 torchvision의 기본
동작**이므로, README의 "아직 미지원" 목록에 "grayscale/alpha 이미지
자동 변환"을 올리는 것은 부정확하다(리뷰에서 발견해 README에서
제거함) -- 실제로는 이미 항상 자동 변환되고 있고, 이 프로젝트가
추가로 구현하거나 끌 수 있는 옵션도 아니다. `input_shape` 계약
자체(모델이 3채널 이외를 요구하면 거부)는 그대로 유지된다.

## 7. `class_to_idx` 검증

```python
def _require_matching_classes(train_dataset, val_dataset, test_dataset) -> None:
    reference = train_dataset.class_to_idx
    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        other = dataset.class_to_idx
        if other == reference:
            continue
        missing = sorted(set(reference) - set(other))
        extra = sorted(set(other) - set(reference))
        ...
        raise ValueError(
            "ImageFolder class mismatch between splits -- "
            f"'train' classes={sorted(reference)} vs '{split_name}' classes={sorted(other)} "
            f"({'; '.join(detail_parts)})"
        )
```

`train`을 기준(reference)으로 `val`/`test`의 `class_to_idx`를 각각
비교한다. 완전히 같으면 통과, 다르면 어느 split에 어떤 클래스가
빠졌는지(`missing`)/더 있는지(`extra`)를 구체적으로 담은 `ValueError`를
즉시 발생시킨다. 예:

```text
ImageFolder class mismatch between splits -- 'train' classes=['cat', 'dog'] vs 'val' classes=['bird', 'cat', 'dog'] (extra in 'val': ['bird'])
```

클래스 이름 집합은 같지만 `class_to_idx` 값 자체가 다른 극단적인
경우(정상적인 `ImageFolder` 사용에서는 발생하지 않지만 방어적으로
포함)에도 `class_to_idx differs: ...` 형태로 두 딕셔너리를 그대로
보여준다. 이 검증은 세 `ImageFolder`가 전부 인스턴스화된 직후, 학습
루프(`run_training()`) 호출 전에 실행되므로 항상 학습 시작 전에
실패한다.

## 8. `ModelSpec` 출력 클래스 수 검증

```python
def require_matching_num_classes(num_classes: int, final_shape: tuple[int, ...]) -> None:
    if len(final_shape) != 1 or final_shape[0] != num_classes:
        raise ValueError(f"dataset has {num_classes} classes but model output shape is {tuple(final_shape)}")
```

Phase 4C의 `run_real_training_e2e.py`는 CIFAR-10 전용이라 `NUM_CLASSES
= 10`을 고정값으로 검증했지만, Phase 4D는 dataset마다 클래스 수가
달라지므로 `len(splits.classes)`(실제 `ImageFolder`가 발견한 클래스
개수)를 기준으로 일반화했다. `run_imagefolder_training_e2e.py`에서
`validate_model_spec()`이 반환한 최종 shape와 함께 학습 시작 전에
호출된다.

## 9. Train / Validation / Test 역할 분리

Phase 4C와 동일한 정책을 그대로 따른다:

* **Train** -- `splits.train`만 `train_loader`로 만들어 optimizer/backward에 사용
* **Validation** -- `splits.val`만 `val_loader`로 만들어 매 epoch
  `evaluate()`에 사용, best epoch 선택 기준
* **Test** -- `splits.test`로 `test_loader`도 다른 두 DataLoader와 함께
  학습 전에 미리 생성되지만, **`run_training()`에는 전달하지 않는다**
  (`run_training()`은 `train_loader`/`val_loader`만 인자로 받으므로
  test dataset을 참조할 방법 자체가 없다). `test_loader`가 실제로
  사용되는 시점은 `training_result.best_state_dict`가 확정된 이후,
  `evaluate(best_model, test_loader, ...)`가 한 번 호출될 때뿐이다.
  leakage를 막는 것은 test dataset/loader의 **생성 시점**이 아니라
  **사용 시점과 용도**다.

## 10. DataLoader 구성

Phase 4C 정책을 그대로 재사용했다. 새 `DataLoaderConfig` 추상화는
만들지 않았다:

```python
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True,
    generator=loader_generator,
    # 작은 마지막 batch가 training-time BatchNorm 동작에 영향을 주는 것을 피하고
    # E2E의 train batch 구성을 일정하게 유지
    drop_last=True, num_workers=0,
)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
```

`batch_size`는 기존 `TrainingConfig`에서 그대로 가져오고, `num_workers=0`
(Windows 안전성 우선)도 Phase 4C와 동일하다.

## 11. Class Mapping 저장/재로드

```python
def save_class_mapping(classes, class_to_idx, path) -> None:
    payload = {"classes": list(classes), "class_to_idx": dict(class_to_idx)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_class_mapping(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
```

숫자 class index만으로는 향후 inference에서 실제 class 이름을 알 수
없으므로, best model과 함께 사용할 metadata로 저장한다. 저장 경로는
`artifacts/training/{model_name}_classes.json` 패턴(기존 history/
state_dict/test_result와 동일한 naming 정책)이다. `history.py`의
`save_training_history()`와 같은 표준 `json` 저장 패턴을 따랐다.
`run_imagefolder_training_e2e.py`는 저장 직후 다시 읽어 원본과 완전히
같은지(`classes`, `class_to_idx` 둘 다) 확인한다.

## 12. `phase4c_cifar10_model.json` 재사용

Phase 4D 전용 새 `ModelSpec` JSON은 만들지 않았다. E2E 검증에 사용하는
CIFAR-10 ImageFolder fixture(10 classes)가 Phase 4C 모델의 최종 출력
`(10,)`과 정확히 일치하므로, `examples/models/phase4c_cifar10_model.json`을
그대로 재사용했다.

다만 `run_imagefolder_training_e2e.py`는 `run_real_training_e2e.py`(같은
모델을 CIFAR-10에서 직접 로딩)와 artifact 경로가 겹치지 않도록,
저장 파일명에 `model_spec.name` 뒤에 `_imagefolder` 접미사를 붙인다
(`phase4c_cifar10_model_imagefolder_history.json` 등). `ModelSpec`
자체(`model_spec.name`, `build_model()` 입력)는 바꾸지 않고, 파일
이름 충돌을 막기 위한 스크립트 레벨의 조치일 뿐이다 -- 두 E2E를 각각
실행해도 서로의 결과물을 덮어쓰지 않아, 둘 다 독립적으로 회귀
검증할 수 있다.

## 13. CIFAR-10 -> ImageFolder fixture 준비 (테스트 준비 전용)

`scripts/prepare_cifar10_imagefolder_fixture.py`는 제품 기능이 아니라
Phase 4D E2E 검증을 위한 fixture 준비 스크립트다:

```text
torchvision CIFAR10 (artifacts/datasets/cifar10, Phase 4C와 데이터 공유)
    -> 클래스별로 정해진 개수만큼 결정론적으로 샘플 선택
    -> PNG로 저장 -> train/val/test ImageFolder 구조 생성
```

* 클래스별 index를 고정 seed로 한 번만 섞은 뒤, 앞부분은 train, 그
  다음 부분은 val로 슬라이스한다 -- 같은 permutation을 나눠 쓰므로
  train/val 이미지가 절대 겹치지 않는다. test는 공식 CIFAR-10 test
  split에서 별도 seed로 뽑는다.
* 기본값은 클래스당 train 20장/val 5장/test 5장(10 classes 기준
  200/50/50장) -- CIFAR-10 전체를 export하지 않고, Phase 4D 경로
  검증에 필요한 만큼만 작게 만들었다.
* `--overwrite`를 주지 않으면 이미 만들어진 fixture를 다시 만들지
  않고 조용히 종료한다 (반복 실행 안전).
* 출력 경로(`artifacts/datasets/cifar10_imagefolder/`)는 `artifacts/`
  전체가 `.gitignore`로 이미 제외되어 있어 Git에 커밋되지 않는다.
* pytest에서 호출되지 않는다 (오프라인 정책과 무관 -- 애초에
  `tests/`가 이 스크립트를 import하거나 실행하지 않는다).

## 14. 신규 오프라인 unit test

`tests/training/test_imagefolder_dataset.py` (총 15개):

1. 정상 ImageFolder 3 split 로딩
2. `class_to_idx` 세 split 동일 확인
3. `classes`/`class_to_idx`가 `splits.train`과 독립된 복사본인지 확인
   (리뷰에서 발견된 aliasing 문제에 대한 회귀 테스트, 2번 검토 참고)
4. 클래스 불일치(적은 클래스) 시 실패
5. val에 클래스 누락 시 실패 (에러 메시지에 `missing in 'val'` + 클래스명 포함 확인)
6. test에 클래스 추가 시 실패 (에러 메시지에 `extra in 'test'` + 클래스명 포함 확인)
7. `require_matching_num_classes` 정상/불일치/비-1D shape (3개 테스트)
8. RGB `input_shape` 계약 위반 시 `ValueError`
9. grayscale("L")/팔레트("P")/알파("RGBA") 원본 이미지가 실제로 3채널로
   변환되는지 확인 (리뷰에서 확인된 torchvision 기본 loader 동작에 대한
   회귀 테스트, 7번 검토 참고)
10. transform 결과 shape/dtype 확인
11. `train`/`val`/`test` 폴더 자체가 없을 때 명확한 에러 (2개 테스트:
    일부 누락, root 자체가 없음)
12. class mapping JSON 저장/재로드 round-trip

위 테스트는 총 15개이며, 전부 `tmp_path` + PIL로 생성한 로컬 PNG
fixture만 사용하고 CIFAR-10 다운로드나 네트워크 접근은 없다.

## 15. 실제 변경/추가 파일

| 파일 | 변경 내용 |
|---|---|
| `src/image_ai_studio/training/torchvision_dataset.py` | 확장 -- `ImageFolderSplits`, `make_imagefolder_datasets`, `require_matching_num_classes`, `save_class_mapping`, `load_class_mapping` 및 내부 헬퍼 추가 (기존 CIFAR-10 함수는 변경 없음) |
| `tests/training/test_imagefolder_dataset.py` | 신규 -- 오프라인 unit test 15개 |
| `scripts/run_imagefolder_training_e2e.py` | 신규 -- 사용자 ImageFolder E2E |
| `scripts/prepare_cifar10_imagefolder_fixture.py` | 신규 -- 테스트용 fixture 준비 전용 (제품 기능 아님) |
| `docs/phase4d_imagefolder_design.md` | 신규 (이 문서) |
| `README.md` | ImageFolder 지원 절 추가, 현재 지원 범위 갱신 |

**변경 없음**: `training/config.py`, `training/dataset.py`(synthetic),
`training/loop.py`, `training/checkpoint.py`, `training/history.py`,
Phase 4C의 CIFAR-10 관련 함수(`make_cifar10_train_val_datasets`,
`make_cifar10_test_dataset`, `build_transform`, `limit_dataset`),
`model_definition/*`, `export/*`, `parity/*`, C++ 코드 전부,
`scripts/run_real_training_e2e.py`, `scripts/run_training_e2e.py`,
`scripts/run_phase1_e2e.py`, `examples/models/phase4c_cifar10_model.json`,
`requirements.txt`, `requirements-dev.txt`.

## 16. 실제 실행 검증 결과

Windows 11, PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, GTX 1080에서
전부 실제로 실행하여 확인했다 (추정치 없음):

* **신규 Phase 4D unit test**: `tests/training/test_imagefolder_dataset.py`
  15 passed (최초 구현 13개 + 코드 리뷰에서 추가된 회귀 테스트 2개:
  aliasing 방지 확인, RGB 자동 변환 확인)
* **`tests/training/` 전체**: 61 passed (Phase 4C까지의 46 + 신규 15)
* **전체 `pytest`**: 218 passed (Phase 4C까지의 203 + 신규 15)
* **Phase 0 regression** (`scripts/run_torchscript_tests.py`):
  `tiny_cnn`/`tiny_residual_cnn` CPU/CUDA 전부 PASS
* **Phase 1~3 E2E regression** (`scripts/run_phase1_e2e.py`, 4개 예시
  JSON): 전부 ModelSpec/build/TorchScript export/C++ CPU/CUDA parity PASS
* **Phase 4A/4B synthetic E2E** (`scripts/run_training_e2e.py`): 기존과
  완전히 동일한 수치(train loss 1.3386 -> 0.2867, best epoch 10) 재현,
  C++ CPU/CUDA parity PASS -- 회귀 없음
* **Phase 4C CIFAR-10 E2E** (`scripts/run_real_training_e2e.py`): 기존과
  완전히 동일한 수치(best epoch 4, test_accuracy=0.1953) 재현, C++
  CPU/CUDA parity PASS -- 회귀 없음
* **CIFAR-10 -> ImageFolder fixture 준비**
  (`scripts/prepare_cifar10_imagefolder_fixture.py`, 클래스당
  train 20/val 5/test 5): `artifacts/datasets/cifar10_imagefolder/`
  아래 10 classes x (train 200 / val 50 / test 50)장 생성 확인
* **신규 Phase 4D ImageFolder E2E**
  (`python scripts/run_imagefolder_training_e2e.py`, 위 fixture 사용,
  `phase4c_cifar10_model.json` 재사용, epochs=5, batch_size=8):

  ```text
  ModelSpec: PASS (11 layers, final shape (10,))
  ImageFolder dataset (train/val/test):
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train=200 val=50 test=50
  Dataset class count vs ModelSpec output check: PASS (num_classes=10)
  Training:
    epoch 1: train_loss=2.3903 val_loss=2.2666 val_acc=0.1000
    epoch 2: train_loss=2.2862 val_loss=2.2082 val_acc=0.1600
    epoch 3: train_loss=2.2040 val_loss=2.1678 val_acc=0.1800
    epoch 4: train_loss=2.1784 val_loss=2.1553 val_acc=0.2200
    epoch 5: train_loss=2.1509 val_loss=2.1269 val_acc=0.3000
    training loss decreased: True (2.3903 -> 2.1509)
  Best epoch: 5
  Best validation loss: 2.1269
  Class mapping save/reload: PASS
  Test evaluation (best model, user-provided test split):
    test_loss=2.1859 test_accuracy=0.2600
  Best model save/reload: PASS
  TorchScript export: PASS
  C++ TorchScript runner: CPU PASS, CUDA PASS
  Parity: PASS

  PHASE 4D E2E: PASS
  ```

  train 200장/5 epoch의 작은 fixture 결과이므로 test accuracy(26%)는
  벤치마크 성능이 아니라 "사용자가 준비한 일반 이미지 폴더가 실제로
  끝까지 연결되어 동작한다"는 경로 검증 결과로 해석해야 한다.

## 17. 이번 Phase 4D에서 의도적으로 구현하지 않은 것

* 자동 Train/Val/Test split (클래스 폴더만 있는 구조를 프로그램이
  알아서 나누는 기능)
* augmentation
* class imbalance 처리, weighted sampler
* optimizer/loss 선택, LR scheduler, early stopping
* optimizer state/epoch가 포함된 full checkpoint, resume
* mixed precision, multi-GPU/distributed training
* Detection/Segmentation training
* dataset registry/factory (CIFAR-10, ImageFolder 각각 별도 함수로
  존재하며, 공통 인터페이스로 묶는 통합 계층은 아직 없음)
* PySide6 UI

이 목록은 필요성이 구체적으로 확인되기 전까지 보류하며, `ModelSpec`
구조나 `training/loop.py`를 바꾸지 않고 확장할 수 있는 지점(새 dataset
함수 추가)에 위치시켰다.
