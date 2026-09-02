"""Phase 10: 폴더 단위 inference contract. GUI/Qt/application controller를
전혀 모른다 -- 디렉터리 하나에서 지원 이미지 목록을 결정론적으로
찾아내고, 각 이미지를 기존 `InferenceRequest`로 변환해 주입된 단일
이미지 backend(`run_single_image_inference`와 호환)에 순차적으로 넘긴다.

한 이미지의 실패는 bounded per-image error로 격리되며 이후 이미지
처리를 막지 않는다. Phase 6B의 single-image public API
(`InferenceRequest`/`InferenceResult`/`run_single_image_inference`)와
Phase 7 portable artifact 포맷은 전혀 건드리지 않는다 -- 이 모듈은
그 위에 얇게 얹히는 조립 계층일 뿐이다.

Phase 12 CP1: `run_folder_inference`에 framework-independent한 진행률
관측과 협조적(cooperative) 취소 계약을 **하위 호환**으로 얹는다.

* `progress_callback` -- discovery 직후 `0-of-total` 스냅샷 하나, 그 뒤
  backend 호출이 끝난 이미지마다 정확히 한 번, item 순서대로 호출된다.
  스냅샷은 frozen `FolderInferenceProgress` 값(Qt/가변 inference 상태
  없음)이고 `completed`는 단조 증가한다.
* `should_cancel` -- 첫 이미지 전, 그리고 "한 이미지가 끝난 뒤 다음
  이미지가 시작되기 전"에만 관측되는 thread-safe 호환 콜백이다. 이미
  진행 중인 단일 이미지 backend 호출은 절대 중단/종료/폐기하지 않는다.
  취소가 관측되면 지금까지 완료된 `ImageOutcome`을 그대로 담은 평범한
  `FolderInferenceResult`와 discovered-total을 실은 `FolderInferenceCancelled`
  (fatal `FolderInferenceError`와 구분되는 별도 terminal 값)를 던진다.

두 hook 모두 생략하면(기본값 `None`) 기존 호출자·결과·backend 호출
동작은 하나도 바뀌지 않는다. `FolderInferenceResult`와 Phase 11 export
`format_version` 1은 이 checkpoint에서 **바뀌지 않는다**: 부분(partial)
결과를 export하면 total/succeeded/failed는 "실제로 처리된 항목"만
기술하고, 취소 여부와 unprocessed 개수 같은 메타데이터는 export 스키마
밖(`FolderInferenceCancelled` 위)에 남는다 -- 이 분리가 하위 호환의
근거다."""
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
    "FolderInferenceCancelled",
    "FolderInferenceRequest",
    "ImageOutcome",
    "FolderInferenceResult",
    "FolderInferenceProgress",
    "SingleImageBackend",
    "ProgressCallback",
    "ShouldCancel",
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

# Phase 12 CP1 optional hook 타입. 둘 다 순수 함수 계약이며 이 모듈은
# Qt/이벤트 루프/스레딩 프리미티브를 import하지 않는다 -- `should_cancel`은
# 그저 매 호출마다 현재 취소 여부를 bool로 돌려주면 되므로 호출자가
# 원자적 플래그/락 뒤에 두면 thread-safe 호환이다.
ProgressCallback = Callable[["FolderInferenceProgress"], None]
ShouldCancel = Callable[[], bool]


class FolderInferenceError(ValueError):
    """폴더 수준 precondition 위반 -- 존재하지 않는 경로, 디렉터리가
    아닌 경로, 지원 이미지가 하나도 없는 폴더. 항상 backend 호출
    전에, 같은 입력에 대해 같은 메시지로 발생한다."""


class FolderInferenceCancelled(Exception):
    """협조적 폴더 취소의 명시적 terminal 값. `should_cancel`이 첫 이미지
    전이나 이미지 경계에서 참을 돌려줬을 때 `run_folder_inference`가
    던진다.

    fatal한 `FolderInferenceError`와 **구분되는** 별도 예외 계층이다 --
    취소는 오류가 아니라 "지금까지 완료된 것은 온전히 보존한 채" 일찍
    멈추는 정상 흐름이다. 그래서 이미 backend 호출이 끝난 이미지들의
    `ImageOutcome`을 순서/성공·오류 값 그대로 담은 평범한
    `FolderInferenceResult`(`result`)와, discovery가 찾아낸 총 이미지
    수(`discovered_total`)를 실어 나른다. `unprocessed`는 그 둘에서
    파생되는 관측값(backend 호출이 아직 시작조차 되지 않은 이미지 수)일
    뿐 따로 저장하지 않는다.

    `result`는 Phase 11 CSV/JSON exporter와 그대로 호환된다. 취소
    여부와 `discovered_total`/`unprocessed`는 export `format_version` 1
    스키마 **밖**의 메타데이터로, 이 예외 위에만 존재한다."""

    def __init__(
        self, result: FolderInferenceResult, discovered_total: int
    ) -> None:
        self.result = result
        self.discovered_total = discovered_total
        super().__init__(
            f"folder inference cancelled after "
            f"{result.total} of {discovered_total} image(s)"
        )

    @property
    def unprocessed(self) -> int:
        """backend 호출이 아직 시작되지 않은 이미지 수 -- discovery 총
        개수에서 완료된(=결과에 담긴) 이미지 수를 뺀 파생값. 항상
        `0 <= unprocessed <= discovered_total`."""
        return self.discovered_total - self.result.total


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


@dataclass(frozen=True)
class FolderInferenceProgress:
    """폴더 inference 진행률의 불변(immutable) 스냅샷. Qt 타입도, 가변
    inference 상태(`InferenceResult`/`ImageOutcome` 참조 등)도 담지 않고
    오직 정수 카운트 넷만 담는다 -- framework에 독립적이다.

    `total`     -- discovery가 찾아낸 지원 이미지 총 개수(고정).
    `completed` -- backend 호출이 끝난(성공+실패) 이미지 수.
    `succeeded` -- 그중 성공한 이미지 수.
    `failed`    -- 그중 격리된 per-image 실패 수.

    불변식(생성 시 강제): ``0 <= succeeded``, ``0 <= failed``,
    ``succeeded + failed == completed`` 그리고 ``completed <= total``."""

    total: int
    completed: int
    succeeded: int
    failed: int

    def __post_init__(self) -> None:
        if self.succeeded < 0 or self.failed < 0:
            raise ValueError("FolderInferenceProgress counts must be non-negative")
        if self.succeeded + self.failed != self.completed:
            raise ValueError(
                "FolderInferenceProgress requires succeeded + failed == completed"
            )
        if not 0 <= self.completed <= self.total:
            raise ValueError(
                "FolderInferenceProgress requires 0 <= completed <= total"
            )


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
    *,
    progress_callback: ProgressCallback | None = None,
    should_cancel: ShouldCancel | None = None,
) -> FolderInferenceResult:
    """`request.folder_path`에서 지원 이미지를 discovery한 뒤 각각을
    기존 `InferenceRequest`로 바꿔 `backend`(기본값
    `run_single_image_inference`)에 하나씩 순차적으로 넘긴다.

    이미지가 하나도 없으면 backend를 부르기 전에
    `FolderInferenceError`를 던진다. 한 이미지에서 backend가 예외를
    던지면 그 이미지는 bounded 오류를 담은 실패 `ImageOutcome`으로
    기록하고 다음 이미지로 계속 진행한다. 성공한 이미지는 backend가
    돌려준 `InferenceResult`를 그대로 담는다 -- 예측값을 다시
    계산하지 않는다. 결과 항목 순서는 discovery 순서와 동일하다.

    Phase 12 CP1 optional hook (**둘 다 생략하면 위 동작은 하나도
    바뀌지 않는다**):

    * ``progress_callback`` -- discovery 성공 직후 `0-of-total` 스냅샷
      하나로 호출되고, 그 뒤 backend 호출이 끝난 이미지마다(성공이든
      격리된 실패든) 정확히 한 번, discovery 순서대로 호출된다. 인자는
      frozen `FolderInferenceProgress`이며 `completed`는 1씩 단조
      증가한다. 콜백이 던진 예외는 격리하지 않고 그대로 전파한다
      (discovery 예외와 같은 fatal 취급).
    * ``should_cancel`` -- 첫 이미지의 `_build_single_image_request`/
      `backend(...)` 전에 한 번, 그리고 "직전 이미지의 backend 호출이
      끝난 뒤 다음 이미지가 시작되기 전"에만 관측된다. 이미 진행 중인
      backend 호출은 절대 중단하지 않는다. 참을 돌려주면 지금까지
      완료된 `ImageOutcome`을 담은 평범한 `FolderInferenceResult`와
      discovered-total을 실은 `FolderInferenceCancelled`를 던진다.
      마지막 이미지가 끝난 뒤에는 더 이상 관측하지 않으므로, 완전히
      끝난 실행이 취소로 뒤집히지 않는다 -- 정상 `FolderInferenceResult`가
      반환된다."""
    image_paths = discover_supported_images(request.folder_path)
    if not image_paths:
        raise FolderInferenceError(
            f"no supported images in folder: {Path(request.folder_path)}"
        )

    discovered_total = len(image_paths)

    def emit(completed: int, succeeded: int, failed: int) -> None:
        if progress_callback is not None:
            progress_callback(
                FolderInferenceProgress(
                    total=discovered_total,
                    completed=completed,
                    succeeded=succeeded,
                    failed=failed,
                )
            )

    # discovery 직후의 초기 0-of-total 스냅샷. 이후 취소가 첫 이미지
    # 전에 관측되더라도 이 스냅샷은 이미 나갔다.
    emit(0, 0, 0)

    outcomes: list[ImageOutcome] = []
    succeeded = 0
    failed = 0
    for image_path in image_paths:
        # 이미지 경계(첫 이미지 전 포함)에서만 취소를 관측한다. 여기
        # 도달했다는 것은 직전 이미지가 있었다면 그 backend 호출이
        # 이미 끝났다는 뜻이다 -- 진행 중 호출을 자르지 않는다.
        if should_cancel is not None and should_cancel():
            raise FolderInferenceCancelled(
                FolderInferenceResult(items=tuple(outcomes)), discovered_total
            )
        single_request = _build_single_image_request(request, image_path)
        try:
            result = backend(single_request)
        except Exception as exc:  # noqa: BLE001 -- per-image 격리가 이 계층의 계약이다
            outcomes.append(
                ImageOutcome(image_path=image_path, result=None, error=_bounded_error(exc))
            )
            failed += 1
        else:
            outcomes.append(
                ImageOutcome(image_path=image_path, result=result, error=None)
            )
            succeeded += 1
        emit(len(outcomes), succeeded, failed)
    return FolderInferenceResult(items=tuple(outcomes))
