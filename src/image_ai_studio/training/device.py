"""device/precision 검증(Phase 4Q/4S/4T)의 단일 구현. `training/
imagefolder_workflow.py`에 private으로 갇혀 있던 로직을 Phase 6B에서
그대로(로직/메시지 무변경) 이 모듈로 옮겼다 -- `run_imagefolder_
training_workflow()`가 이 모듈에서 import해서 쓰고,
`image_ai_studio.inference`도 같은 함수를 import해서 쓴다(중복 구현
없음). 이름은 기존 private 이름(`_` 접두)을 그대로 유지한다 -- 이
모듈 밖에서 이 함수들을 쓰는 다른 코드(inference 포함)가 이미 같은
패키지 내부이므로, public API로 승격할 필요 없이 그대로 import해서
쓸 수 있다(최소 변경 원칙)."""
from __future__ import annotations

import re

import torch

# Phase 4Q: 이 프로젝트가 명시적으로 지원하는 문법은 "cpu"/"cuda"/"cuda:N"
# 뿐이다(N은 leading zero 없는 음이 아닌 정수). torch.device()도 대소문자
# 구분, zero-padding, 그 외 backend(mps/xpu/hip 등)를 이미 엄격하게
# 거부함을 실측했지만, torch.device() 혼자서는 이 프로젝트가 검증하지
# 않은 backend까지 그대로 통과시키므로(mps/xpu/hip...), 이 프로젝트는
# 자체 허용 목록을 별도로 관리한다 -- 과도한 자체 parser 대신 최소
# 정규식 하나로 충분하다.
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda|cuda:(0|[1-9][0-9]*))$")


def _validate_device(value: str) -> None:
    """device 문자열을 검증한다(Phase 4Q). CUDA 계열이면 추가로
    `torch.cuda.is_available()`/`torch.cuda.device_count()`로 조기
    검증한다 -- 그러지 않으면 `.to(device)`까지 내려가서야 저수준
    `AcceleratorError`("CUDA error: invalid device ordinal")가 나는데,
    이 경우가 실제로 재현됨을 로컬에서 직접 확인했다(require_matching_num_classes()
    와 동일한 "깊은 곳 대신 조기에 명확한 에러" 패턴). CUDA 미가용 시
    CPU로 조용히 대체하지 않는다 -- 사용자가 명시한 실행 의도를 그대로
    존중한다."""
    if not isinstance(value, str) or not _DEVICE_PATTERN.fullmatch(value):
        raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:N', got {value!r}")
    if value == "cpu":
        return
    if not torch.cuda.is_available():
        raise ValueError(f"device={value!r} requires CUDA, but torch.cuda.is_available() is False")
    if ":" in value:
        index = int(value.split(":", 1)[1])
        device_count = torch.cuda.device_count()
        if index >= device_count:
            raise ValueError(
                f"device={value!r} is out of range -- torch.cuda.device_count()=={device_count}"
            )


def _is_cuda_device(device: str) -> bool:
    """`device`가 CUDA 계열("cuda"/"cuda:N")인지 판별한다(Phase 4U). 이미
    `_validate_device()`가 통과시킨 값에만 적용되므로 별도 형식 검증은
    하지 않는다. `loop.py`의 `_build_precision_execution()`이 쓰는 것과
    동일한 판별 predicate다(그쪽은 fp16/bf16 CUDA 전용 여부, 이쪽은
    pin_memory/non_blocking effective 여부 -- 판별 기준 자체는 같다)."""
    return device == "cuda" or device.startswith("cuda:")


_CUDA_ONLY_PRECISIONS = ("fp16", "bf16")


def _validate_precision_device_compatibility(precision: str, device: str) -> None:
    """`precision="fp16"`(Phase 4S)/`"bf16"`(Phase 4T)은 CUDA에서만
    허용한다. `TrainingConfig` 자신은 device를 모르므로(`_require_one_of()`
    가 "fp32"/"fp16"/"bf16" 값 자체만 검증) 이 cross-field 검증은 device를
    이미 아는 workflow 레벨에서 한다 -- `class_weights` 길이를 dataset
    크기와 여기서 검증하는 것과 같은 이유(`require_matching_num_classes()`
    근처 참고). CPU AMP(`torch.amp.autocast(device_type="cpu", ...)`)는
    이번 Phase의 범위 밖이라 silent CPU fallback 없이 명확히 거부한다.

    이 함수는 `run_training()`이 내부적으로 강제하는 `loop.py`의
    `_build_precision_execution()`(같은 조합을 `ValueError`로 거부)와
    같은 검증을 중복하는 것이 아니라, 서로 다른 경계를 보호하는
    defense-in-depth다 -- 이 함수는 dataset/model 준비 전 user-facing
    fail-fast(workflow 진입 즉시 거부)이고, `_build_precision_execution()`
    는 이 workflow를 거치지 않고 `TrainingConfig`+`run_training()`을
    직접 호출하는 generic caller까지 보호한다. `fp16`/`bf16` 둘 다
    `_CUDA_ONLY_PRECISIONS`에 속하는 값이면 device="cpu"일 때 동일하게
    거부한다 -- fp16만 검사하고 bf16을 빠뜨리면 Phase 4S stabilization
    에서 발견한 "특정 precision 값만 하드코딩 검사"와 같은 종류의 누락이
    반복된다.

    Phase 6B부터는 `image_ai_studio.inference`도 이 함수를 그대로
    재사용한다(inference 전용 precision/device 검증을 새로 만들지
    않는다) -- inference도 training과 동일하게 CPU + fp16/bf16 조합을
    거부해야 하는 이유가 같기 때문이다(CPU AMP는 이 프로젝트의 범위
    밖)."""
    if precision in _CUDA_ONLY_PRECISIONS and device == "cpu":
        raise ValueError(
            f"precision={precision!r} requires a CUDA device, but device={device!r} -- "
            "CPU AMP is not supported in this phase"
        )
