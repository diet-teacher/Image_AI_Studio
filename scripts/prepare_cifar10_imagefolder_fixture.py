#!/usr/bin/env python
"""CIFAR-10(torchvision) 일부를 일반 `ImageFolder` 구조로 export하는 테스트
픽스처 준비 스크립트.

Phase 4D(사용자 `ImageFolder` dataset 지원) E2E 검증을 위한 것으로,
제품 기능이 아니라 **테스트 준비 전용**이다. pytest에서 호출되지 않고
(오프라인 정책과 무관), 실행 시 필요하면 CIFAR-10을 다운로드하므로
네트워크가 필요하다.

역할:

    torchvision CIFAR10 (root: artifacts/datasets/cifar10, Phase 4C와 공유)
        -> 클래스별로 지정된 개수만큼 결정론적으로 샘플 선택
        -> PNG로 저장, train/<class_name>/*.png 형태의 ImageFolder 구조 생성

CIFAR-10 전체를 export하지 않는다 -- Phase 4D E2E 경로(사용자 이미지
폴더 -> ImageFolder -> 학습 -> export) 검증에 필요한 만큼의 작은
subset만 만든다. 생성된 이미지는 artifacts/ 아래에 저장되며 Git에
커밋하지 않는다 (.gitignore로 제외).

사용법:

    python scripts/prepare_cifar10_imagefolder_fixture.py
    python scripts/prepare_cifar10_imagefolder_fixture.py --train-per-class 20 --val-per-class 5 --test-per-class 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torchvision.datasets import CIFAR10

CIFAR10_DATA_ROOT = REPO_ROOT / "artifacts" / "datasets" / "cifar10"  # Phase 4C와 공유 (재다운로드 방지)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "datasets" / "cifar10_imagefolder"

DEFAULT_TRAIN_PER_CLASS = 20
DEFAULT_VAL_PER_CLASS = 5
DEFAULT_TEST_PER_CLASS = 5

SEED = 20260730  # Phase 4C E2E와 동일한 값 재사용 (일관성 목적, 특별한 의미 없음)


def _permuted_indices_by_class(dataset: CIFAR10, seed: int) -> dict[int, list[int]]:
    """클래스별로 (전체 인덱스를) 고정 seed로 섞은 리스트를 반환.

    train/val을 같은 공식 split에서 뽑을 때, 이 permutation을 공유하고
    슬라이스만 다르게 하면(첫 N개 vs 그다음 M개) 겹치지 않는 두 부분집합을
    결정론적으로 얻을 수 있다."""
    targets = torch.tensor(dataset.targets)
    generator = torch.Generator().manual_seed(seed)
    result: dict[int, list[int]] = {}
    for class_idx in range(len(dataset.classes)):
        class_indices = (targets == class_idx).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_indices), generator=generator)
        result[class_idx] = class_indices[perm].tolist()
    return result


def _export_indices(dataset: CIFAR10, indices: list[int], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        image, _ = dataset[idx]  # transform 없이 raw PIL Image
        image.save(output_dir / f"{idx:05d}.png")
    return len(indices)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=CIFAR10_DATA_ROOT)
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_TRAIN_PER_CLASS)
    parser.add_argument("--val-per-class", type=int, default=DEFAULT_VAL_PER_CLASS)
    parser.add_argument("--test-per-class", type=int, default=DEFAULT_TEST_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true", help="output-root가 이미 있으면 지우고 다시 생성")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.output_root.exists():
        if not args.overwrite:
            print(f"output root already exists: {args.output_root} (use --overwrite to regenerate)")
            return 0
        shutil.rmtree(args.output_root)

    print("CIFAR-10 -> ImageFolder fixture 준비")
    print(f"  data root: {args.data_root}")
    print(f"  output root: {args.output_root}")

    official_train = CIFAR10(root=str(args.data_root), train=True, download=True)
    official_test = CIFAR10(root=str(args.data_root), train=False, download=True)

    # train/val: 공식 train split을 클래스별로 한 번만 섞고, 앞부분은 train,
    # 그다음 부분은 val로 슬라이스 -- 같은 permutation을 나눠 쓰므로 두 split은
    # 절대 겹치지 않는다.
    train_val_indices = _permuted_indices_by_class(official_train, seed=args.seed)
    test_indices = _permuted_indices_by_class(official_test, seed=args.seed + 1)

    for split_name, per_class in (("train", args.train_per_class), ("val", args.val_per_class)):
        print(f"  {split_name}: {per_class} images/class")

    print(f"  test: {args.test_per_class} images/class (공식 test split)")

    for class_idx, class_name in enumerate(official_train.classes):
        indices = train_val_indices[class_idx]
        train_indices = indices[: args.train_per_class]
        val_indices = indices[args.train_per_class : args.train_per_class + args.val_per_class]
        _export_indices(official_train, train_indices, args.output_root / "train" / class_name)
        _export_indices(official_train, val_indices, args.output_root / "val" / class_name)

        test_class_indices = test_indices[class_idx][: args.test_per_class]
        _export_indices(official_test, test_class_indices, args.output_root / "test" / class_name)

    print(f"done: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
