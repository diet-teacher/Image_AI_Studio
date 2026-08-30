"""Phase 10: 폴더 단위 inference contract. GUI/Qt/application controller를
전혀 모른다 -- 디렉터리 하나에서 지원 이미지 목록을 결정론적으로
찾아내고, 각 이미지를 기존 `InferenceRequest`로 변환해 주입된 단일
이미지 backend(`run_single_image_inference`와 호환)에 순차적으로 넘긴다.

한 이미지의 실패는 bounded per-image error로 격리되며 이후 이미지
처리를 막지 않는다. Phase 6B의 single-image public API
(`InferenceRequest`/`InferenceResult`/`run_single_image_inference`)와
Phase 7 portable artifact 포맷은 전혀 건드리지 않는다 -- 이 모듈은
그 위에 얇게 얹히는 조립 계층일 뿐이다."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from image_ai_studio.inference.single_image_inference import (
    InferenceRequest,
    InferenceResult,
    run_single_image_inference,
)

__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "FolderInferenceError",
    "FolderInferenceRequest",
    "ImageOutcome",
    "FolderInferenceResult",
    "SingleImageBackend",
    "discover_supported_images",
    "run_folder_inference",
]

# 문서화된 지원 이미지 확장자(소문자, 점 포함). 확장자 비교는 대소문자를
# 구분하지 않으므로 ".JPG" 같은 파일도 동일하게 포함된다. 이 목록에
# 없는 확장자와 확장자가 없는 파일은 discovery에서 제외된다.
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)
_SUPPORTED_EXTENSION_SET = frozenset(SUPPORTED_IMAGE_EXTENSIONS)

# per-image 오류 메시지 상한(문자 수). backend가 아무리 긴 예외
# 문자열을 던져도 aggregate 결과가 무한정 커지지 않도록 자른다.
_MAX_ERROR_CHARS = 500

# `run_single_image_inference`와 호출 호환인 단일 이미지 backend.
SingleImageBackend = Callable[[InferenceRequest], InferenceResult]


class FolderInferenceError(ValueError):
    """폴더 수준 precondition 위반 -- 존재하지 않는 경로, 디렉터리가
    아닌 경로, 지원 이미지가 하나도 없는 폴더. 항상 backend 호출
    전에, 같은 입력에 대해 같은 메시지로 발생한다."""


@dataclass(frozen=True)
class FolderInferenceRequest:
    """폴더 단위 inference에 필요한 전부. artifact 값(model JSON /
    state_dict / class mapping)과 device/precision은 폴더 안 모든
    이미지에 그대로 공유된다 -- 이미지마다 달라지는 `image_path`만
    여기 담지 않고 discovery가 채운다."""

    model_json_path: Path
    state_dict_path: Path
    class_mapping_path: Path
    folder_path: Path
    device: str
    precision: str


@dataclass(frozen=True)
class ImageOutcome:
    """이미지 한 장의 처리 결과. 성공이면 `result`는 backend가 돌려준
    `InferenceResult` 객체 그대로이고 `error`는 None이다. 실패면
    `result`가 None이고 `error`는 bounded 오류 문자열이다 -- 정확히
    둘 중 하나만 설정된다."""

    image_path: Path
    result: InferenceResult | None
    error: str | None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("ImageOutcome requires exactly one of result or error")

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class FolderInferenceResult:
    """폴더 전체 결과. `items`의 순서는 discovery가 정한 결정론적
    순서를 그대로 보존한다. total/succeeded/failed는 items에서 파생된
    관측값으로, 따로 저장하지 않는다."""

    items: tuple[ImageOutcome, ...]

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.items if item.succeeded)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.items if not item.succeeded)


def discover_supported_images(folder_path: Path) -> list[Path]:
    """`folder_path`(반드시 존재하는 디렉터리) 바로 아래에서 지원
    확장자를 가진 파일만 골라 이름 기준 오름차순으로 돌려준다.
    디렉터리와 지원하지 않는 파일은 제외한다. 반환 순서는 파일시스템
    열거 순서와 무관하다. 하위 폴더로 재귀하지 않는다.

    존재하지 않거나 디렉터리가 아닌 경로는 `FolderInferenceError`.
    지원 이미지가 없으면 빈 리스트를 돌려준다(그 자체는 오류가
    아니다 -- fatal 판정은 `run_folder_inference`가 한다)."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FolderInferenceError(f"folder does not exist: {folder}")
    if not folder.is_dir():
        raise FolderInferenceError(f"folder path is not a directory: {folder}")

    images = [
        entry
        for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in _SUPPORTED_EXTENSION_SET
    ]
    images.sort(key=lambda path: path.name)
    return images


def _build_single_image_request(
    request: FolderInferenceRequest, image_path: Path
) -> InferenceRequest:
    """공유 artifact/device/precision 값에 이미지 경로 하나만 끼워
    기존 `InferenceRequest`를 만든다(새 필드/포맷 없음)."""
    return InferenceRequest(
        model_json_path=request.model_json_path,
        state_dict_path=request.state_dict_path,
        class_mapping_path=request.class_mapping_path,
        image_path=image_path,
        device=request.device,
        precision=request.precision,
    )


def _bounded_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) > _MAX_ERROR_CHARS:
        text = text[: _MAX_ERROR_CHARS - 3] + "..."
    return text


def run_folder_inference(
    request: FolderInferenceRequest,
    backend: SingleImageBackend = run_single_image_inference,
) -> FolderInferenceResult:
    """`request.folder_path`에서 지원 이미지를 discovery한 뒤 각각을
    기존 `InferenceRequest`로 바꿔 `backend`(기본값
    `run_single_image_inference`)에 하나씩 순차적으로 넘긴다.

    이미지가 하나도 없으면 backend를 부르기 전에
    `FolderInferenceError`를 던진다. 한 이미지에서 backend가 예외를
    던지면 그 이미지는 bounded 오류를 담은 실패 `ImageOutcome`으로
    기록하고 다음 이미지로 계속 진행한다. 성공한 이미지는 backend가
    돌려준 `InferenceResult`를 그대로 담는다 -- 예측값을 다시
    계산하지 않는다. 결과 항목 순서는 discovery 순서와 동일하다."""
    image_paths = discover_supported_images(request.folder_path)
    if not image_paths:
        raise FolderInferenceError(
            f"no supported images in folder: {Path(request.folder_path)}"
        )

    outcomes: list[ImageOutcome] = []
    for image_path in image_paths:
        single_request = _build_single_image_request(request, image_path)
        try:
            result = backend(single_request)
        except Exception as exc:  # noqa: BLE001 -- per-image 격리가 이 계층의 계약이다
            outcomes.append(
                ImageOutcome(image_path=image_path, result=None, error=_bounded_error(exc))
            )
        else:
            outcomes.append(
                ImageOutcome(image_path=image_path, result=result, error=None)
            )
    return FolderInferenceResult(items=tuple(outcomes))
