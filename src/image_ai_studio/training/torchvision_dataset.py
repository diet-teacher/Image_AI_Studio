"""torchvision 데이터셋을 학습 파이프라인에 연결. 이번 Phase의 첫 구현은
CIFAR-10이지만, "torchvision Dataset 클래스 + root + ModelSpec.input_shape
-> resize/normalize된 Dataset"이라는 좁은 역할로 한정해 다른 torchvision
dataset(ImageFolder, Oxford-IIIT Pet 등)에도 큰 구조 변경 없이 이어붙일
수 있게 했다. dataset registry/factory는 이번 Phase에서 만들지 않는다.

`training/dataset.py`(synthetic)와 역할이 다르다:
- dataset.py            -> offline/deterministic pytest, 학습 루프 자체 검증
- torchvision_dataset.py -> 실제 이미지 학습 E2E (네트워크 다운로드 발생 가능)

pytest는 이 모듈의 CIFAR10 관련 함수를 직접 호출하지 않는다 (오프라인
정책 유지) -- tests/training/test_torchvision_dataset.py는 network 없이
검증 가능한 부분(입력 검증, transform, limit_dataset)만 다루고, CIFAR10
다운로드는 scripts/run_real_training_e2e.py에서만 일어난다.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

NUM_CLASSES = 10
DEFAULT_VAL_FRACTION = 0.1  # 공식 train 50,000 -> train 45,000 / val 5,000

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
