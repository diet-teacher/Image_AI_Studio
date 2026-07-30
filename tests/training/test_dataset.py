"""합성 데이터셋 재현성 / train-val 분리 / class_patterns 공유 테스트."""
from __future__ import annotations

import torch

from image_ai_studio.training.dataset import (
    SyntheticImageDataset,
    make_class_patterns,
    make_train_val_datasets,
)


def test_same_class_patterns_and_seed_produce_identical_dataset() -> None:
    class_patterns = make_class_patterns((3, 8, 8), num_classes=4, seed=1)
    a = SyntheticImageDataset(class_patterns, num_samples=16, seed=123)
    b = SyntheticImageDataset(class_patterns, num_samples=16, seed=123)
    assert torch.equal(a.images, b.images)
    assert torch.equal(a.labels, b.labels)


def test_different_seed_produces_different_samples() -> None:
    class_patterns = make_class_patterns((3, 8, 8), num_classes=4, seed=1)
    a = SyntheticImageDataset(class_patterns, num_samples=16, seed=123)
    b = SyntheticImageDataset(class_patterns, num_samples=16, seed=456)
    assert not torch.equal(a.images, b.images)


def test_dataset_shapes_and_dtypes() -> None:
    class_patterns = make_class_patterns((3, 8, 8), num_classes=4, seed=1)
    dataset = SyntheticImageDataset(class_patterns, num_samples=10, seed=1)
    assert len(dataset) == 10

    image, label = dataset[0]
    assert tuple(image.shape) == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label.dtype == torch.long
    assert 0 <= int(label) < 4


def test_labels_cover_multiple_classes() -> None:
    class_patterns = make_class_patterns((3, 8, 8), num_classes=4, seed=1)
    dataset = SyntheticImageDataset(class_patterns, num_samples=64, seed=1)
    assert len(set(dataset.labels.tolist())) > 1


def test_make_train_val_datasets_is_reproducible() -> None:
    train_a, val_a = make_train_val_datasets((3, 8, 8), num_classes=4, seed=7, train_size=16, val_size=8)
    train_b, val_b = make_train_val_datasets((3, 8, 8), num_classes=4, seed=7, train_size=16, val_size=8)

    assert torch.equal(train_a.images, train_b.images)
    assert torch.equal(val_a.images, val_b.images)
    assert len(train_a) == 16
    assert len(val_a) == 8


def test_make_train_val_datasets_share_class_patterns() -> None:
    """Train/Validation은 같은 class prototype을 공유해야 같은 class index가
    양쪽에서 같은 특징을 의미한다."""
    train, val = make_train_val_datasets((3, 8, 8), num_classes=4, seed=7, train_size=16, val_size=8)
    assert torch.equal(train.class_patterns, val.class_patterns)


def test_make_train_val_datasets_produce_disjoint_samples() -> None:
    """class_patterns는 같아도 labels/noise는 서로 다른 파생 seed(seed+1, seed+2)에서
    나오므로 실제 샘플(이미지)은 train/val이 달라야 한다."""
    train, val = make_train_val_datasets((3, 8, 8), num_classes=4, seed=7, train_size=16, val_size=8)
    assert not torch.equal(train.images[: len(val.images)], val.images)
