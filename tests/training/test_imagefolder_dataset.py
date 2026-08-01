"""training/torchvision_dataset.py의 ImageFolder(Phase 4D) 부분 테스트.

CIFAR-10과 달리 다운로드가 필요 없으므로, tmp_path에 PIL로 작은
train/val/test ImageFolder 구조를 직접 만들어 완전히 오프라인으로
검증한다 (네트워크 접근 없음).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from image_ai_studio.training.torchvision_dataset import (
    load_class_mapping,
    make_imagefolder_datasets,
    require_matching_num_classes,
    save_class_mapping,
)


def _make_split(root: Path, split: str, class_to_images: dict[str, int]) -> None:
    """root/split/<class>/*.png 구조를 만든다. class_to_images는
    클래스별로 만들 이미지 개수."""
    for class_name, count in class_to_images.items():
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (20, 20), color=(10 * i, 20, 30)).save(class_dir / f"{i}.png")


def _make_standard_dataset(root: Path) -> None:
    """train/val/test 전부 cat/dog 두 클래스로 동일하게 구성."""
    for split in ("train", "val", "test"):
        _make_split(root, split, {"cat": 2, "dog": 2})


# -- 1. 정상 ImageFolder 3 split 로딩 -----------------------------------------


def test_make_imagefolder_datasets_loads_all_three_splits(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)

    splits = make_imagefolder_datasets((3, 8, 8), tmp_path)

    assert len(splits.train) == 4
    assert len(splits.val) == 4
    assert len(splits.test) == 4


# -- 2. class_to_idx 동일 -----------------------------------------------------


def test_make_imagefolder_datasets_returns_consistent_class_mapping(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)

    splits = make_imagefolder_datasets((3, 8, 8), tmp_path)

    assert splits.classes == ["cat", "dog"]
    assert splits.class_to_idx == {"cat": 0, "dog": 1}


def test_make_imagefolder_datasets_returns_independent_copies_of_class_metadata(tmp_path: Path) -> None:
    """splits.classes/class_to_idx는 splits.train.classes/class_to_idx와
    값은 같지만 별개 객체여야 한다 -- 호출자가 splits.classes를 그
    자리에서 수정해도 splits.train(ImageFolder 인스턴스)의 내부
    metadata가 함께 바뀌면 안 된다."""
    _make_standard_dataset(tmp_path)

    splits = make_imagefolder_datasets((3, 8, 8), tmp_path)
    splits.classes.append("mutated")
    splits.class_to_idx["mutated"] = 99

    assert splits.train.classes == ["cat", "dog"]
    assert splits.train.class_to_idx == {"cat": 0, "dog": 1}


# -- 3. 클래스 불일치 시 실패 (val에 없는 클래스가 test에 있는 경우) -----------


def test_make_imagefolder_datasets_rejects_class_set_mismatch(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", {"cat": 2, "dog": 2})
    _make_split(tmp_path, "val", {"cat": 2, "dog": 2})
    _make_split(tmp_path, "test", {"cat": 2})  # dog 누락

    with pytest.raises(ValueError, match="class mismatch"):
        make_imagefolder_datasets((3, 8, 8), tmp_path)


# -- 4. val 클래스 누락 시 실패 ------------------------------------------------


def test_make_imagefolder_datasets_rejects_missing_class_in_val(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", {"cat": 2, "dog": 2})
    _make_split(tmp_path, "val", {"cat": 2})  # dog 누락
    _make_split(tmp_path, "test", {"cat": 2, "dog": 2})

    with pytest.raises(ValueError, match=r"missing in 'val'.*dog"):
        make_imagefolder_datasets((3, 8, 8), tmp_path)


# -- 5. test 클래스 추가 시 실패 -----------------------------------------------


def test_make_imagefolder_datasets_rejects_extra_class_in_test(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", {"cat": 2, "dog": 2})
    _make_split(tmp_path, "val", {"cat": 2, "dog": 2})
    _make_split(tmp_path, "test", {"cat": 2, "dog": 2, "bird": 2})  # bird 추가

    with pytest.raises(ValueError, match=r"extra in 'test'.*bird"):
        make_imagefolder_datasets((3, 8, 8), tmp_path)


# -- 6. ModelSpec output class 수 mismatch 검증 --------------------------------


def test_require_matching_num_classes_passes_when_equal() -> None:
    require_matching_num_classes(2, (2,))  # raise 없이 통과해야 함


def test_require_matching_num_classes_rejects_mismatched_count() -> None:
    with pytest.raises(ValueError, match=r"dataset has 2 classes but model output shape is \(10,\)"):
        require_matching_num_classes(2, (10,))


def test_require_matching_num_classes_rejects_non_1d_shape() -> None:
    with pytest.raises(ValueError, match="model output shape"):
        require_matching_num_classes(2, (2, 1, 1))


# -- 7. RGB input_shape 계약 --------------------------------------------------


def test_make_imagefolder_datasets_rejects_non_rgb_input_shape(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)

    with pytest.raises(ValueError, match="3-channel"):
        make_imagefolder_datasets((1, 8, 8), tmp_path)


def test_make_imagefolder_datasets_converts_non_rgb_source_images_to_3_channels(tmp_path: Path) -> None:
    """`_require_rgb_input_shape`는 ModelSpec.input_shape 계약만 검사한다
    (3채널이 아니면 거부) -- 원본 이미지 파일 자체의 PIL 모드는 검사하지
    않는다. 대신 torchvision.datasets.ImageFolder의 기본 loader(pil_loader)가
    모든 이미지를 `img.convert("RGB")`로 읽으므로, grayscale("L")/
    palette("P")/alpha("RGBA") 이미지를 넣어도 실제로는 항상 3채널
    텐서가 나온다. 이 동작은 이 프로젝트가 구현한 것이 아니라
    torchvision의 기본 동작이며, 이 테스트는 그 실제 계약을 고정한다."""
    for split in ("train", "val", "test"):
        class_dir = tmp_path / split / "cat"
        class_dir.mkdir(parents=True)
        Image.new("L", (20, 20), color=128).save(class_dir / "gray.png")
        Image.new("RGBA", (20, 20), color=(10, 20, 30, 255)).save(class_dir / "rgba.png")
        Image.new("P", (20, 20), color=1).save(class_dir / "palette.png")

    splits = make_imagefolder_datasets((3, 8, 8), tmp_path)

    for i in range(len(splits.train)):
        image, _ = splits.train[i]
        assert image.shape == (3, 8, 8)


# -- 8. transform 결과 shape/dtype --------------------------------------------


def test_make_imagefolder_datasets_applies_transform_shape_and_dtype(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)

    splits = make_imagefolder_datasets((3, 8, 8), tmp_path)
    image, label = splits.train[0]

    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label in (0, 1)


# -- 9. dataset root/train/val/test 폴더 누락 시 명확한 에러 -------------------


def test_make_imagefolder_datasets_rejects_missing_split_directory(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", {"cat": 2, "dog": 2})
    _make_split(tmp_path, "val", {"cat": 2, "dog": 2})
    # test/ 폴더 자체가 없음

    with pytest.raises(ValueError, match=r"missing required split directories.*test"):
        make_imagefolder_datasets((3, 8, 8), tmp_path)


def test_make_imagefolder_datasets_rejects_completely_missing_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "does_not_exist"

    with pytest.raises(ValueError, match="missing required split directories"):
        make_imagefolder_datasets((3, 8, 8), missing_root)


# -- class mapping JSON 저장/로드 ----------------------------------------------


def test_save_and_load_class_mapping_round_trip(tmp_path: Path) -> None:
    _make_standard_dataset(tmp_path)
    splits = make_imagefolder_datasets((3, 8, 8), tmp_path)

    mapping_path = tmp_path / "classes.json"
    save_class_mapping(splits.classes, splits.class_to_idx, mapping_path)
    loaded = load_class_mapping(mapping_path)

    assert loaded["classes"] == splits.classes
    assert loaded["class_to_idx"] == splits.class_to_idx


# -- 10. 네트워크 접근 없음 ----------------------------------------------------
#
# 위 테스트는 전부 tmp_path에 직접 만든 PNG 픽스처만 사용하고
# torchvision.datasets.CIFAR10을 전혀 참조하지 않는다 -- 이 파일이
# import하는 것은 make_imagefolder_datasets/require_matching_num_classes/
# save_class_mapping/load_class_mapping뿐이며, 이 중 어느 것도 네트워크
# 접근이 필요한 코드 경로를 갖지 않는다.
