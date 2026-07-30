"""외부 다운로드 없는 합성 이미지 분류 데이터셋. 고정 seed로 재현 가능 (torchvision 미사용)."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

NOISE_SCALE = 0.3


def make_class_patterns(input_shape: tuple[int, int, int], num_classes: int, seed: int) -> torch.Tensor:
    """클래스마다 고정된 랜덤 패턴 생성. Train/Validation이 이 결과를 공유해야
    같은 class index가 양쪽에서 같은 특징을 의미한다."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_classes, *input_shape, generator=generator)


class SyntheticImageDataset(Dataset):
    """공유 class_patterns + 자체 labels/noise로 만든 합성 분류 데이터셋.

    자체 torch.Generator만 사용해 전역 RNG 상태와 무관하게 동작 -- 같은
    class_patterns/seed 조합이면 호출 순서와 상관없이 항상 동일한 이미지/
    라벨을 생성한다.
    """

    def __init__(
        self,
        class_patterns: torch.Tensor,
        num_samples: int,
        seed: int,
        noise_scale: float = NOISE_SCALE,
    ) -> None:
        num_classes = class_patterns.shape[0]
        input_shape = tuple(class_patterns.shape[1:])

        generator = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        noise = torch.randn(num_samples, *input_shape, generator=generator) * noise_scale

        self.class_patterns = class_patterns
        self.images = (class_patterns[labels] + noise).to(torch.float32)
        self.labels = labels.to(torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def make_train_val_datasets(
    input_shape: tuple[int, int, int],
    num_classes: int,
    seed: int,
    train_size: int = 64,
    val_size: int = 32,
) -> tuple[SyntheticImageDataset, SyntheticImageDataset]:
    """Train/Validation이 같은 class_patterns(seed)를 공유하고, 서로 다른
    파생 seed(seed+1, seed+2)로 각자의 labels/noise만 뽑아 완전히 분리된
    샘플을 만든다. Test set은 Phase 4A 범위 밖."""
    class_patterns = make_class_patterns(input_shape, num_classes, seed=seed)
    train_dataset = SyntheticImageDataset(class_patterns, train_size, seed=seed + 1)
    val_dataset = SyntheticImageDataset(class_patterns, val_size, seed=seed + 2)
    return train_dataset, val_dataset
