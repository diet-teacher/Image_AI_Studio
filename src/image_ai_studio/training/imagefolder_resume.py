"""ImageFolder 학습 CLI(scripts/run_imagefolder_training_e2e.py)의 resume
전용 metadata. Phase 4F의 core checkpoint(training/checkpoint.py)는
dataset-agnostic하게 유지하고, "이 checkpoint를 지금의 ModelSpec/ImageFolder
dataset으로 이어서 학습해도 되는가"라는 ImageFolder 전용 질문은 이 모듈이
별도 JSON 사이드카(<checkpoint>.meta.json)로 답한다 (docs/
phase4g_imagefolder_resume_design.md 참고).

checkpoint.py의 CHECKPOINT_FORMAT_VERSION과는 독립적인
METADATA_FORMAT_VERSION을 쓴다 -- 두 포맷은 서로 다른 이유로 바뀔 수 있는
독립된 개념이라 하나로 묶지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from torchvision.datasets import ImageFolder

from image_ai_studio.model_definition.serialization import model_spec_to_dict
from image_ai_studio.model_definition.specs import ModelSpec
from image_ai_studio.training.torchvision_dataset import ImageFolderSplits

METADATA_FORMAT_VERSION = 1

_REQUIRED_METADATA_FIELDS = (
    "metadata_version",
    "model_spec_hash",
    "class_to_idx",
    "train_size",
    "val_size",
    "test_size",
    "train_files_hash",
    "val_files_hash",
    "test_files_hash",
)


@dataclass
class ImageFolderResumeMetadata:
    """checkpoint 저장 시점의 ModelSpec/ImageFolder dataset 신원 정보.
    require_compatible_imagefolder_resume_metadata()가 resume 시점의
    현재 metadata와 이 값을 비교한다."""

    metadata_version: int
    model_spec_hash: str
    class_to_idx: dict[str, int]
    train_size: int
    val_size: int
    test_size: int
    train_files_hash: str
    val_files_hash: str
    test_files_hash: str


def hash_model_spec(model_spec: ModelSpec) -> str:
    """ModelSpec의 canonical JSON(model_spec_to_dict() 재사용, 키 정렬,
    공백 없음)을 SHA-256 해시한다. model_spec_to_dict()는 layer 구조/파라미터만
    담고 파일 경로 정보를 포함하지 않으므로, model-json 파일을 옮기거나
    이름을 바꿔도 내용이 같으면 같은 해시가 나온다."""
    canonical = json.dumps(
        model_spec_to_dict(model_spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_imagefolder_files(dataset: ImageFolder) -> str:
    """dataset.samples(정렬된 (절대경로, class_idx) 목록)로부터
    {path, class_index} canonical JSON을 만들어 SHA-256 해시한다. 경로는
    dataset.root 기준 상대경로 + '/' 정규화(as_posix) -- 데이터셋을 다른
    머신/디렉터리로 옮겨도 절대경로 차이로 오탐하지 않기 위함. 이미지
    내용 자체는 해싱하지 않는다 (비용 문제로 의도적으로 제외, 같은
    경로에서 파일 내용만 바뀌는 경우는 탐지하지 못하는 것이 알려진 한계)."""
    root = Path(dataset.root)
    entries = [
        {"path": Path(path).relative_to(root).as_posix(), "class_index": class_index}
        for path, class_index in dataset.samples
    ]
    entries.sort(key=lambda item: (item["path"], item["class_index"]))

    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_imagefolder_resume_metadata(model_spec: ModelSpec, splits: ImageFolderSplits) -> ImageFolderResumeMetadata:
    """현재 ModelSpec/ImageFolder splits로부터 metadata를 만든다. 신규
    학습(checkpoint 저장 시점)과 resume(호환성 검증 시점) 양쪽에서 같은
    함수로 "현재 상태"를 계산해 대칭성을 유지한다."""
    return ImageFolderResumeMetadata(
        metadata_version=METADATA_FORMAT_VERSION,
        model_spec_hash=hash_model_spec(model_spec),
        class_to_idx=dict(splits.class_to_idx),
        train_size=len(splits.train),
        val_size=len(splits.val),
        test_size=len(splits.test),
        train_files_hash=_hash_imagefolder_files(splits.train),
        val_files_hash=_hash_imagefolder_files(splits.val),
        test_files_hash=_hash_imagefolder_files(splits.test),
    )


def _atomic_write_text(text: str, path: Path) -> None:
    """text를 path에 원자적으로 쓴다(Phase 4J, docs/
    phase4j_epoch_checkpoint_design.md §7-2) -- checkpoint.py의
    _atomic_torch_save()와 같은 계약(목적지와 같은 디렉터리에 임시
    파일 생성 -> flush()/os.fsync() -> os.replace(), 실패 시 목적지
    보존, 정리 실패가 원래 예외를 가리지 않음)을 텍스트 파일에 대해
    수행한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def save_imagefolder_resume_metadata(metadata: ImageFolderResumeMetadata, path: str | Path) -> None:
    """metadata를 JSON 파일로 저장. 상위 디렉터리 자동 생성. 저장은
    원자적이다(§7-2) -- 예외가 나면 기존 파일은 전혀 바뀌지 않는다."""
    path = Path(path)
    _atomic_write_text(json.dumps(asdict(metadata), indent=2), path)


def load_imagefolder_resume_metadata(path: str | Path) -> ImageFolderResumeMetadata:
    """save_imagefolder_resume_metadata()로 저장한 JSON을 로드한다. 필수
    필드가 없거나 metadata_version이 지원 범위 밖이면 명확한 ValueError를
    낸다."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"{path} does not exist -- ImageFolder resume requires a metadata sidecar file "
            "next to the checkpoint (see metadata_path_for_checkpoint())"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: metadata JSON must be an object, got {type(data).__name__}")

    missing = [name for name in _REQUIRED_METADATA_FIELDS if name not in data]
    if missing:
        raise ValueError(f"{path}: metadata is missing required field(s): {missing}")

    if data["metadata_version"] != METADATA_FORMAT_VERSION:
        raise ValueError(
            f"{path} has unsupported metadata_version={data['metadata_version']!r} "
            f"(expected {METADATA_FORMAT_VERSION})"
        )

    return ImageFolderResumeMetadata(**{name: data[name] for name in _REQUIRED_METADATA_FIELDS})


def require_compatible_imagefolder_resume_metadata(
    saved: ImageFolderResumeMetadata, current: ImageFolderResumeMetadata
) -> None:
    """checkpoint 저장 시점의 metadata(saved)와 지금 resume하려는 시점의
    metadata(current)가 ModelSpec/dataset 관점에서 호환되는지 확인한다.
    metadata_version은 load_imagefolder_resume_metadata()가 이미 검사하므로
    여기서는 다시 보지 않는다. 불일치하는 첫 필드에서 구체적인 값을 보여주는
    ValueError를 낸다."""
    for field_name in (
        "model_spec_hash",
        "class_to_idx",
        "train_size",
        "val_size",
        "test_size",
        "train_files_hash",
        "val_files_hash",
        "test_files_hash",
    ):
        saved_value = getattr(saved, field_name)
        current_value = getattr(current, field_name)
        if saved_value != current_value:
            raise ValueError(
                f"cannot resume: checkpoint metadata field '{field_name}' does not match the current "
                f"ModelSpec/dataset -- saved={saved_value!r}, current={current_value!r}"
            )


def metadata_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """checkpoint 파일 경로로부터 metadata sidecar 경로를 유도한다 (예:
    checkpoint.pt -> checkpoint.pt.meta.json). 기존 파일명을 자르거나
    바꾸지 않고 항상 뒤에 ".meta.json"만 붙인다 (Path.with_suffix()는
    마지막 확장자를 교체해버려 점이 여러 개인 파일명에서 의도와 다르게
    동작할 수 있어 쓰지 않는다). 저장/로드 양쪽이 반드시 이 함수 하나만
    거치도록 해서 경로 유도 규칙이 두 곳에서 따로 구현되어 어긋나는 것을
    막는다."""
    path = Path(checkpoint_path)
    return path.parent / f"{path.name}.meta.json"
