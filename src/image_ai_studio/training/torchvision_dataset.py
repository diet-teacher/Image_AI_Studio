"""torchvision 데이터셋을 학습 파이프라인에 연결. Phase 4C(CIFAR-10)에서는
"torchvision Dataset 클래스 + root + ModelSpec.input_shape -> resize/
normalize된 Dataset"이라는 좁은 역할로 한정했고, Phase 4D는 그 역할을
그대로 유지한 채 두 번째 dataset(사용자가 준비한 `ImageFolder` 폴더)을
같은 모듈에 이어붙였다. dataset registry/factory는 이번 Phase에서도
만들지 않는다.

`training/dataset.py`(synthetic)와 역할이 다르다:
- dataset.py            -> offline/deterministic pytest, 학습 루프 자체 검증
- torchvision_dataset.py -> 실제 이미지 학습 E2E (네트워크 다운로드 발생 가능)

pytest는 이 모듈의 CIFAR10 관련 함수를 직접 호출하지 않는다 (오프라인
정책 유지) -- tests/training/test_torchvision_dataset.py는 network 없이
검증 가능한 부분(입력 검증, transform, limit_dataset, ImageFolder는
tmp_path 픽스처)만 다루고, CIFAR10 다운로드는
scripts/run_real_training_e2e.py에서만 일어난다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

NUM_CLASSES = 10
DEFAULT_VAL_FRACTION = 0.1  # 공식 train 50,000 -> train 45,000 / val 5,000

IMAGEFOLDER_SPLIT_NAMES = ("train", "val", "test")

# CIFAR-10 자체 통계로 튜닝한 값이 아니라, [0,1] 픽셀을 [-1,1]로 옮기는
# 범용 정규화다 -- 이 모듈이 특정 dataset 전용이 되지 않도록 의도적으로
# 단순하게 유지했다 (벤치마크 정확도가 아니라 경로 검증이 목적).
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def _require_rgb_input_shape(input_shape: tuple[int, int, int]) -> None:
    channels = input_shape[0]
    if channels != 3:
        raise ValueError(
            "real-image classification requires a 3-channel (RGB) "
            f"input_shape, got input_shape={tuple(input_shape)} (channels={channels})"
        )


def build_transform(input_shape: tuple[int, int, int]) -> transforms.Compose:
    """PIL Image -> ModelSpec.input_shape에 맞게 resize된 float32 Tensor ->
    정규화. augmentation은 포함하지 않는다 (Train/Validation/Test 전부
    동일한 deterministic transform)."""
    _, height, width = input_shape
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    """dataset 앞부분 limit개만 사용 (E2E를 빠르게 돌리기 위함). limit이
    None이거나 dataset 크기 이상이면 원본을 그대로 반환. 어떤 torchvision
    Dataset/Subset에도 동일하게 적용 가능해 CIFAR-10 전용이 아니다."""
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def make_cifar10_train_val_datasets(
    input_shape: tuple[int, int, int],
    root: str | Path,
    seed: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    download: bool = True,
) -> tuple[Dataset, Dataset]:
    """CIFAR-10 공식 train split(50,000)을 고정 seed로 Train/Validation으로
    결정론적으로 분리한다. 공식 test split은 다루지 않는다 --
    make_cifar10_test_dataset()으로 명확히 분리해, Test가 실수로 학습/모델
    선택에 섞여 들어가지 않도록 한다."""
    _require_rgb_input_shape(input_shape)
    transform = build_transform(input_shape)

    full_train = CIFAR10(root=str(root), train=True, download=download, transform=transform)

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


def make_cifar10_test_dataset(
    input_shape: tuple[int, int, int],
    root: str | Path,
    download: bool = True,
) -> Dataset:
    """CIFAR-10 공식 test split. best epoch 확정 후 딱 한 번, 최종 성능
    평가에만 사용해야 한다 -- best epoch 선택이나 학습 중 어떤 판단에도
    이 함수의 결과를 사용하면 안 된다."""
    _require_rgb_input_shape(input_shape)
    transform = build_transform(input_shape)
    return CIFAR10(root=str(root), train=False, download=download, transform=transform)


# -- Phase 4D: 사용자 제공 ImageFolder 폴더 -----------------------------------
#
# CIFAR-10과 달리 다운로드가 없고, "이미 train/val/test로 분리된 폴더
# 구조"만 지원한다 (자동 split은 이번 Phase 범위 밖). transform은 위
# build_transform()을 그대로 재사용해 Train/Validation/Test가 완전히
# 동일한 deterministic 전처리를 쓰도록 한다.


class ImageFolderSplits(NamedTuple):
    """make_imagefolder_datasets()의 반환값. 필드가 5개라 `train, val,
    test = splits`처럼 3개로 바로 언패킹하면 ValueError가 난다 -- 필요하면
    `train, val, test, *_ = splits`(나머지를 한 변수로 묶음)를 쓰거나,
    `splits.train`/`splits.classes`처럼 속성으로 접근한다. classes/
    class_to_idx는 저장(save_class_mapping)이나 ModelSpec 출력 shape
    검증(require_matching_num_classes)에 바로 쓸 수 있게 함께 담는다."""

    train: ImageFolder
    val: ImageFolder
    test: ImageFolder
    classes: list[str]
    class_to_idx: dict[str, int]


def _require_split_directories(root: str | Path) -> dict[str, Path]:
    """`root/train`, `root/val`, `root/test`가 모두 존재하는지 확인하고
    경로 dict를 반환. 하나라도 없으면 어떤 폴더가 없는지 명시한
    ValueError를 즉시 발생시킨다 (ImageFolder 자체의 불명확한 에러 대신)."""
    root = Path(root)
    split_dirs = {name: root / name for name in IMAGEFOLDER_SPLIT_NAMES}
    missing = [name for name, path in split_dirs.items() if not path.is_dir()]
    if missing:
        raise ValueError(
            f"ImageFolder dataset root is missing required split directories {missing} "
            f"under root={root} -- expected {root}/train, {root}/val, {root}/test "
            "(each containing one subdirectory per class)"
        )
    return split_dirs


def _require_matching_classes(
    train_dataset: ImageFolder, val_dataset: ImageFolder, test_dataset: ImageFolder
) -> None:
    """세 split의 class_to_idx가 완전히 동일한지 검증. 다르면 어느
    split에 어떤 클래스가 더 있거나 없는지 구체적으로 보여주는
    ValueError를 발생시킨다 -- 학습 시작 전에 실패해야 한다."""
    reference_name, reference = "train", train_dataset.class_to_idx
    for split_name, dataset in (("val", val_dataset), ("test", test_dataset)):
        other = dataset.class_to_idx
        if other == reference:
            continue

        missing = sorted(set(reference) - set(other))
        extra = sorted(set(other) - set(reference))
        detail_parts = []
        if missing:
            detail_parts.append(f"missing in '{split_name}': {missing}")
        if extra:
            detail_parts.append(f"extra in '{split_name}': {extra}")
        if not detail_parts:
            # 클래스 이름 집합은 같지만 index 배정이 다른 경우 (예: 커스텀 정렬)
            detail_parts.append(f"class_to_idx differs: '{reference_name}'={reference} vs '{split_name}'={other}")

        raise ValueError(
            "ImageFolder class mismatch between splits -- "
            f"'{reference_name}' classes={sorted(reference)} vs '{split_name}' classes={sorted(other)} "
            f"({'; '.join(detail_parts)})"
        )


def make_imagefolder_datasets(
    input_shape: tuple[int, int, int],
    root: str | Path,
) -> ImageFolderSplits:
    """사용자가 미리 Train/Validation/Test로 분리해 둔 `ImageFolder` 구조를
    읽어 기존 학습 파이프라인에 연결한다.

    요구되는 폴더 구조 (자동 split 없음, 각 split에 이미 클래스별
    하위 폴더가 있어야 함)::

        root/
            train/<class_name>/*.jpg
            val/<class_name>/*.jpg
            test/<class_name>/*.jpg

    세 split은 완전히 독립된 `ImageFolder` 인스턴스이므로, 클래스
    이름/개수가 하나라도 다르면(`_require_matching_classes`) 학습을
    시작하기 전에 ValueError로 실패한다.
    """
    _require_rgb_input_shape(input_shape)
    split_dirs = _require_split_directories(root)
    transform = build_transform(input_shape)

    train_dataset = ImageFolder(str(split_dirs["train"]), transform=transform)
    val_dataset = ImageFolder(str(split_dirs["val"]), transform=transform)
    test_dataset = ImageFolder(str(split_dirs["test"]), transform=transform)

    _require_matching_classes(train_dataset, val_dataset, test_dataset)

    return ImageFolderSplits(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        # train_dataset.classes/class_to_idx를 그대로 aliasing하지 않고 복사한다
        # -- 반환된 리스트/딕셔너리를 호출자가 그 자리에서 수정해도(정렬 등)
        # train_dataset(=splits.train)의 내부 metadata가 함께 바뀌지 않도록 한다.
        classes=list(train_dataset.classes),
        class_to_idx=dict(train_dataset.class_to_idx),
    )


def require_matching_num_classes(num_classes: int, final_shape: tuple[int, ...]) -> None:
    """dataset의 실제 클래스 수와 ModelSpec 최종 출력 shape가 일치하는지
    확인. CIFAR-10 E2E(run_real_training_e2e.py)처럼 클래스 수를 고정값
    (10)으로 검증하는 대신, ImageFolder처럼 클래스 수가 dataset마다
    달라지는 경우를 위한 일반화된 버전이다."""
    if len(final_shape) != 1 or final_shape[0] != num_classes:
        raise ValueError(f"dataset has {num_classes} classes but model output shape is {tuple(final_shape)}")


def save_class_mapping(classes: list[str], class_to_idx: dict[str, int], path: str | Path) -> None:
    """class 이름 목록/매핑을 JSON으로 저장 (best model과 함께 inference에
    필요한 metadata -- 숫자 class index만으로는 실제 class 이름을 알 수
    없기 때문). 저장 형식은 history.py의 save_training_history()와 동일한
    표준 json 패턴을 따른다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"classes": list(classes), "class_to_idx": dict(class_to_idx)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_class_mapping(path: str | Path) -> dict[str, object]:
    """save_class_mapping()으로 저장한 JSON을 읽어
    `{"classes": [...], "class_to_idx": {...}}` 형태로 반환."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
