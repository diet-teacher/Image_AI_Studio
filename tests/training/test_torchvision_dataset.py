"""training/torchvision_dataset.py 테스트.

CIFAR-10 자체(네트워크 다운로드)는 여기서 다루지 않는다 -- 오프라인/
결정론적 pytest 정책 유지. RGB 검증은 실제 CIFAR10 인스턴스화 전에
일어나므로 download=False로도 안전하게 테스트할 수 있고, transform/
limit_dataset은 PIL로 직접 만든 이미지와 ImageFolder(다른 torchvision
dataset의 예시)로 검증한다 -- CIFAR-10 전용이 아님을 실제로 증명한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

from image_ai_studio.training.torchvision_dataset import (
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    build_transform,
    limit_dataset,
    make_cifar10_test_dataset,
    make_cifar10_train_val_datasets,
)


# -- RGB 계약 (CIFAR10을 실제로 건드리기 전에 raise 되므로 download=False로 안전) --


def test_train_val_datasets_reject_non_rgb_input_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="3-channel"):
        make_cifar10_train_val_datasets((1, 32, 32), root=tmp_path, seed=0, download=False)


def test_test_dataset_rejects_non_rgb_input_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="3-channel"):
        make_cifar10_test_dataset((4, 32, 32), root=tmp_path, download=False)


# -- transform (오프라인, CIFAR10 불필요) -----------------------------------------


def test_build_transform_resizes_to_input_shape_and_returns_float32() -> None:
    transform = build_transform((3, 16, 16))
    image = Image.new("RGB", (32, 32))

    tensor = transform(image)

    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32


def test_build_transform_normalizes_white_pixel_to_expected_value() -> None:
    transform = build_transform((3, 4, 4))
    white_image = Image.new("RGB", (4, 4), color=(255, 255, 255))

    tensor = transform(white_image)

    # ToTensor()는 255 -> 1.0으로 스케일링하고, Normalize((0.5,)*3,(0.5,)*3)는
    # (1.0 - 0.5) / 0.5 = 1.0으로 옮긴다.
    expected = (1.0 - NORMALIZE_MEAN[0]) / NORMALIZE_STD[0]
    assert torch.allclose(tensor, torch.full_like(tensor, expected), atol=1e-5)


def test_build_transform_generalizes_to_imagefolder_dataset(tmp_path: Path) -> None:
    """CIFAR-10 전용이 아니라 다른 torchvision dataset(ImageFolder)에도
    그대로 재사용 가능함을 오프라인으로 증명 (네트워크 없음, 저장소에
    생성한 작은 fixture 이미지만 사용)."""
    class_dir = tmp_path / "class_a"
    class_dir.mkdir()
    Image.new("RGB", (20, 20), color=(10, 20, 30)).save(class_dir / "sample.png")

    dataset = ImageFolder(str(tmp_path), transform=build_transform((3, 8, 8)))

    image, label = dataset[0]
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label == 0


# -- limit_dataset (오프라인, CIFAR10 불필요) --------------------------------------


def test_limit_dataset_returns_original_when_limit_is_none() -> None:
    dataset = TensorDataset(torch.arange(10))
    assert limit_dataset(dataset, None) is dataset


def test_limit_dataset_returns_original_when_limit_exceeds_length() -> None:
    dataset = TensorDataset(torch.arange(10))
    assert limit_dataset(dataset, 100) is dataset


def test_limit_dataset_takes_first_n_elements_not_random() -> None:
    dataset = TensorDataset(torch.arange(10))
    limited = limit_dataset(dataset, 3)

    assert len(limited) == 3
    assert [limited[i][0].item() for i in range(3)] == [0, 1, 2]
