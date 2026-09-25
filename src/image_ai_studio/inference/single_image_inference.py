"""Phase 6B: canonical single-image inference path(docs/
phase6a_inference_architecture.md §1-C/§3). `state_dict + ModelSpec`
을 canonical model loading으로 쓴다(TorchScript는 이번 Phase의 범위
밖) -- 아래 함수들은 training이 이미 갖고 있는 public API만 재사용
한다. inference 전용 model builder/transform을 새로 만들지 않는다:

    load_model_spec()      -- model_definition/serialization.py
    validate_model_spec()  -- model_definition/validation.py
    build_model()           -- model_definition/builder.py
    load_state_dict()       -- training/checkpoint.py
    build_transform()       -- training/torchvision_dataset.py
    load_class_mapping()    -- training/torchvision_dataset.py
    require_matching_num_classes() -- training/torchvision_dataset.py
    _validate_device()/_validate_precision_device_compatibility()/
    _is_cuda_device() -- training/device.py(Phase 6B가 새로 추출한
    공용 모듈, imagefolder_workflow.py와 완전히 동일한 검증 로직)

이 모듈은 GUI/Qt/application controller state를 전혀 모른다 --
순수하게 artifact + 이미지 경로 하나를 받아 `InferenceResult`를
반환하는 함수 하나가 public surface의 전부다."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from image_ai_studio.model_definition.builder import build_model
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.model_definition.validation import validate_model_spec
from image_ai_studio.training.artifact_manifest import (
    ARTIFACT_MANIFEST_FILENAME,
    ManifestError,
    load_manifest,
    verify_manifest,
)
from image_ai_studio.training.checkpoint import load_state_dict
from image_ai_studio.training.config import PRECISION_CHOICES
from image_ai_studio.training.device import (
    _is_cuda_device,
    _validate_device,
    _validate_precision_device_compatibility,
)
from image_ai_studio.training.torchvision_dataset import (
    build_transform,
    load_class_mapping,
    require_matching_num_classes,
)

_CUDA_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

_CANONICAL_BUNDLE_FILENAMES = (
    "model_definition.json",
    "best_model_state_dict.pt",
    "class_mapping.json",
)


@dataclass(frozen=True)
class InferenceRequest:
    """단일 이미지 inference에 필요한 전부(Phase 6A §7 -- single image만,
    폴더/배치는 범위 밖). GUI 입력 문자열을 `Path`로 바꾸는 것 같은
    조립은 `application.inference_controller.build_inference_request()`
    의 책임이다 -- 이 dataclass 자신은 이미 정리된 값만 받는다."""

    model_json_path: Path
    state_dict_path: Path
    class_mapping_path: Path
    image_path: Path
    device: str
    precision: str


@dataclass(frozen=True)
class InferenceResult:
    """GUI가 표시할 최종 결과. raw logits은 담지 않는다(Phase 6A §4 --
    GUI가 요구하지 않는 값은 넣지 않는다)."""

    predicted_index: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    inference_duration_seconds: float


def _require_valid_precision(precision: str) -> None:
    if precision not in PRECISION_CHOICES:
        raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {precision!r}")


def _physical_parent_identity(path: Path) -> str:
    """Return a platform-normalized identity for ``path``'s parent directory.

    ``Path.resolve(strict=True)`` normalizes relative components and resolves
    directory symlinks (and Windows junctions via the platform realpath
    implementation).  Only the parent is resolved: the artifact path itself
    remains untouched so ``verify_manifest()`` can still reject an individual
    canonical artifact that is a symlink.  A canonical parent whose identity
    cannot be established fails closed instead of silently bypassing integrity
    verification.
    """
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManifestError(
            f"cannot resolve parent directory for canonical artifact {path.name!r} "
            f"({type(exc).__name__})"
        ) from exc
    return os.path.normcase(os.fspath(resolved_parent))


def _verify_canonical_bundle_manifest(request: InferenceRequest) -> None:
    """Verify an optional Phase 15 manifest before any artifact is decoded.

    Automatic binding is deliberately narrow: all three request paths must use
    the exact canonical filenames and normalize to one common directory.
    Explicit noncanonical paths retain the pre-Phase-15 behavior even if an
    unrelated sibling manifest exists.  A completely absent manifest likewise
    preserves legacy bundle compatibility, while a present invalid manifest
    fails closed through the existing ``ValueError``-compatible boundary.
    """
    artifact_paths = (
        Path(request.model_json_path),
        Path(request.state_dict_path),
        Path(request.class_mapping_path),
    )
    if tuple(path.name for path in artifact_paths) != _CANONICAL_BUNDLE_FILENAMES:
        return

    try:
        physical_parents = {_physical_parent_identity(path) for path in artifact_paths}
        if len(physical_parents) != 1:
            return

        # Keep an original request parent as the access path.  Resolving only
        # for identity avoids erasing evidence that an artifact itself is a
        # symlink, which verify_manifest() must continue to reject.
        bundle_dir = artifact_paths[0].parent
        manifest = load_manifest(bundle_dir)
        if manifest is not None:
            verify_manifest(bundle_dir)
    except ManifestError as exc:
        # ManifestError is already bounded and never embeds artifact contents.
        # Add a stable high-level prefix for application/GUI error surfaces.
        raise ManifestError(f"artifact bundle integrity verification failed: {exc}") from exc


def run_single_image_inference(request: InferenceRequest) -> InferenceResult:
    """canonical inference 경로(Phase 6A §1-C):

        ModelSpec JSON -> load_model_spec() -> validate_model_spec()
        -> build_model() -> load_state_dict() -> model.to(device)
        -> model.eval()

    이미지는 `build_transform(model_spec.input_shape)`로 training과
    완전히 동일하게 전처리한다(preprocessing parity, Phase 6A §2 --
    tests/inference/test_preprocessing_parity.py가 이를 고정한다).
    `inference_duration_seconds`의 측정 구간은 "model/입력 준비 완료
    -> forward 시작"부터 "softmax/argmax/confidence/probabilities
    추출 완료"까지다(model/입력 준비, 즉 파일 IO·모델 재구성·이미지
    전처리는 측정에 포함하지 않는다 -- 이 함수가 매번 새로
    model/artifact를 로드하는 것은 "이번 호출의 준비 비용"이지
    사용자가 이해하는 "추론 시간"이 아니기 때문이다). CUDA device면
    `.item()` 호출들이 이미 암묵적으로 동기화를 강제하지만, 의도를
    명확히 하고 향후 코드 변경에 안전하도록 `torch.cuda.synchronize()`
    를 측정 구간 끝에 명시적으로 한 번 추가한다(host-side enqueue
    latency가 아니라 실제 GPU 완료 시간에 최대한 가깝게 측정하기
    위함, 이 synchronize 호출도 측정 구간 *안*에 포함된다)."""
    _validate_device(request.device)
    _require_valid_precision(request.precision)
    _validate_precision_device_compatibility(request.precision, request.device)

    # This must precede model JSON, mapping, state-dict, and image decoding so
    # corrupt/mixed canonical bundles are rejected without deserialization.
    _verify_canonical_bundle_manifest(request)

    model_spec = load_model_spec(request.model_json_path)
    shape_trace = validate_model_spec(model_spec)
    final_shape = shape_trace[-1].output_shape

    class_mapping = load_class_mapping(request.class_mapping_path)
    classes: list[str] = class_mapping["classes"]
    require_matching_num_classes(len(classes), final_shape)

    model = build_model(model_spec)
    load_state_dict(model, request.state_dict_path, map_location="cpu")
    model = model.to(request.device)
    model = model.eval()

    image = Image.open(request.image_path).convert("RGB")
    transform = build_transform(model_spec.input_shape)
    inputs = transform(image).unsqueeze(0).to(request.device)

    with torch.inference_mode():
        started_at = time.perf_counter()
        if request.precision == "fp32":
            logits = model(inputs)
        else:
            autocast_dtype = _CUDA_AUTOCAST_DTYPES[request.precision]
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(inputs)

        probabilities_tensor = torch.softmax(logits, dim=1)
        predicted_index = int(probabilities_tensor.argmax(dim=1).item())
        confidence = float(probabilities_tensor[0, predicted_index].item())
        probabilities = {
            class_name: float(probabilities_tensor[0, index].item())
            for index, class_name in enumerate(classes)
        }
        if _is_cuda_device(request.device):
            torch.cuda.synchronize(request.device)
        inference_duration_seconds = time.perf_counter() - started_at

    return InferenceResult(
        predicted_index=predicted_index,
        predicted_class=classes[predicted_index],
        confidence=confidence,
        probabilities=probabilities,
        inference_duration_seconds=inference_duration_seconds,
    )
