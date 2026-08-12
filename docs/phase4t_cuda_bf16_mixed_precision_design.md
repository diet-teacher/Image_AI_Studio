# Phase 4T: CUDA BF16 Mixed Precision Training — 설계안

## 1. 목적

Phase 4S가 CUDA FP16 AMP(autocast+GradScaler)를 지원했다. Phase 4T는
같은 CUDA training에 BF16 mixed precision을 추가한다. BF16은 FP16과
달리 GradScaler가 필요 없어("autocast를 쓰는가"와 "AMP dtype이
무엇인가"라는 두 축이 FP16에서는 우연히 "scaler 존재 여부" 하나로
결합돼 있었음), 이 결합을 해소하는 `train_one_epoch()` 리팩터링이
이번 Phase의 핵심이다. `precision="fp32"`(기본값)에서는 CPU/CUDA
기존 FP32 학습 semantics와 numerical behavior를 그대로 유지하고,
Phase 4S의 FP16 contract(GradScaler lifecycle/clipping ordering/
checkpoint/exact-resume/legacy 정책)도 전부 그대로 유지한다.

## 2. Phase 4S baseline 재조사에서 발견한 구조적 문제

Phase 4S의 `train_one_epoch()`은 `if scaler is not None: <FP16
autocast+scaled backward> else: <FP32>`로 분기했다. `_build_grad_scaler()`
는 `precision != "fp16"`이면 `None`을 반환했다. 즉 "scaler 존재 여부"가
"AMP를 쓰는가"와 "AMP dtype이 무엇인가"를 동시에 판별하는 단일 신호였다.
BF16은 autocast는 필요하지만 GradScaler는 필요 없는 조합이라, 이 구조에
`"bf16"`을 단순 추가만 하면(scaler는 여전히 `None`) BF16 요청이
`scaler is None` 때문에 FP32 branch로 조용히 빠지는 **새로운 silent
fallback**이 생긴다는 것을 코드 재조사로 확인했다 -- Phase 4S
stabilization에서 발견한 "non-CUDA+fp16 silent fallback"과 같은
계열의 문제다. 이 문제를 근본적으로 해소하기 위해 `autocast_dtype`
(어떤 dtype으로 autocast할지, `None`이면 autocast 자체를 안 씀)과
`scaler`(scaled backward를 쓸지)를 독립된 두 축으로 분리했다(Design B,
조사 라운드에서 이미 승인됨).

## 3. BF16 API 및 hardware 실측

로컬 환경: `torch==2.12.1+cu126`, `cuda==12.6`, `cudnn==91002`,
GPU `NVIDIA GeForce GTX 1080`(compute capability `(6, 1)`, Pascal,
Tensor Core 없음). `torch.cuda.is_bf16_supported()`는 기본값
(`including_emulation=True`)으로 `True`를 반환하지만,
`torch.cuda.is_bf16_supported(including_emulation=False)`는 `False`
를 반환한다 -- 즉 이 GPU는 **네이티브 BF16 하드웨어가 없고 API
기본값만 보면 이 사실을 놓친다.** 실제
`torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`로
project의 실제 `build_model()` fixture(Conv2d+BatchNorm2d+ReLU+
MaxPool2d+AdaptiveAvgPool2d+Flatten+Dropout+Linear)에서 forward/
backward/optimizer.step이 전부 정상 동작함을 실측 확인했다(loss
finite, Conv/Linear 출력 dtype `bfloat16`, `CrossEntropyLoss` loss
dtype은 자동으로 `float32`로 승격, BatchNorm buffer/optimizer momentum
buffer는 `float32` 유지, MaxPool2d/AdaptiveAvgPool2d 출력도 `bfloat16`
정상 통과). `ResidualBlock`/`BranchBlock`은 코드를 다시 읽어 자체
kernel 없이 Conv2d/BatchNorm2d/ReLU/elementwise add/`torch.cat`의
순수 조합임을 재확인했다(Phase 4R과 동일한 논리로 별도 kernel-level
실측 불필요).

## 4. GradScaler 불필요성 근거

**주장의 범위를 명확히 한다.** BF16은 FP16보다 exponent 폭이 넓고
(8-bit, FP16은 5-bit) FP32와 같은 exponent-bit 폭을 가진다 -- 따라서
FP16에서 loss scaling이 특히 필요했던 좁은 dynamic/exponent range
문제를 크게 완화한다. 이것이 "BF16에는 underflow가 전혀 없다"거나
"GradScaler가 원리적으로 절대 필요할 수 없다"는 일반론은 아니다 --
BF16도 유한한 exponent 범위를 갖는 부동소수점 형식이라 그 범위 밖의
매우 작은 값에서는 여전히 underflow가 발생할 수 있다. Phase 4T가
실제로 근거로 삼는 것은 다음 좁은 실측 사실이다: 이 프로젝트의 실제
BF16 학습 경로(Conv2d+BatchNorm2d+Dropout fixture, `gradient_clip_norm`
포함)에서 GradScaler 없이 정상 학습(loss 감소)이 되고, exact-resume
실측(§8)에서도 GradScaler 없이(즉 어떤 새 checkpoint state도 없이)
continuous==split+resume이 bitwise exact함을 직접 증명했다. API
레벨에서는 BF16+GradScaler 조합도 에러 없이 동작하지만
(`scaler.scale/step/update`가 bf16 loss에도 그냥 실행됨), 위 실측
근거로 굳이 쓸 이유가 없다고 판단했다. **production contract: BF16
에는 GradScaler를 사용하지 않는다** -- 이는 "이 프로젝트의 지원
범위(현재 ModelSpec/optimizer 조합)에서 검증된 선택"이지 BF16 일반에
대한 수학적 증명이 아니다.

## 5. `_build_precision_execution()` 설계

`loop.py`의 `_build_grad_scaler(config, device) -> GradScaler | None`
를 다음으로 대체했다:

```python
def _build_precision_execution(
    config: TrainingConfig, device: str
) -> tuple[torch.dtype | None, torch.amp.GradScaler | None]:
    if config.precision == "fp32":
        return None, None
    if config.precision not in ("fp16", "bf16"):
        raise ValueError(f"unsupported precision: {config.precision!r}")
    if not (device == "cuda" or device.startswith("cuda:")):
        raise ValueError(f"precision={config.precision!r} requires a CUDA device, but device={device!r}")
    if config.precision == "fp16":
        return torch.float16, torch.amp.GradScaler("cuda")
    return torch.bfloat16, None  # config.precision == "bf16"
```

반환값 매트릭스:

```
fp32          -> (None, None)
fp16 + CUDA   -> (torch.float16, GradScaler)
bf16 + CUDA   -> (torch.bfloat16, None)
fp16/bf16 + non-CUDA -> ValueError
그 외 precision 값(향후 PRECISION_CHOICES 확장 시) -> ValueError
```

CUDA 판별(`device == "cuda" or device.startswith("cuda:")`)은 fp16/bf16
둘 다 이 함수 하나를 공유하므로 중복이 없다 -- Phase 4S stabilization의
"device=='cpu'만 검사해서 다른 non-CUDA backend가 통과하는 버그"가
반복되지 않는다.

**explicit precision dispatch(stabilization 라운드에서 수정).** 최초
구현은 `fp32`와 non-CUDA를 걸러낸 뒤 `if precision == "fp16": ...
else: return torch.bfloat16, None`처럼 "fp16이 아니면 무조건 bf16"이라는
implicit dispatch였다. 이는 논리적으로 현재 `TrainingConfig.PRECISION_CHOICES`
(`"fp32"|"fp16"|"bf16"`) 범위 안에서는 항상 맞지만, Phase 4T 자체가
"scaler 존재 여부로 precision을 암묵적으로 추론하던 구조가 BF16
silent fallback을 만들었다"는 문제를 고치는 Phase이므로, 이 helper의
precision dispatch도 같은 원칙(implicit inference 대신 explicit
matching)을 따라야 한다고 판단했다. `config.precision not in
("fp16", "bf16")`이면 정상 경로에서는 도달하지 않지만 명시적으로
`ValueError`를 낸다 -- 향후 `PRECISION_CHOICES`에 새 값(예: `"fp8"`)이
추가되는데 이 함수 수정이 누락되면, 새 값이 조용히 `bf16`으로
처리되는 대신 fail-fast로 즉시 드러난다.

**GradScaler에는 device type만 전달(stabilization 라운드에서 수정).**
최초 구현은 `torch.amp.GradScaler(device)`로 `device`(예: `"cuda:0"`)
를 그대로 전달했다. 설치된 PyTorch의 `GradScaler.__init__` docstring
을 직접 확인한 결과 "Possible values are: 'cuda' and 'cpu'"라고
명시돼 있어 **공식 contract는 device type만**이다. 소스코드도
`self._device == "cuda"`라는 정확한 문자열 비교로 CUDA-availability
경고 분기를 타는데, `"cuda:0"`을 넘기면 이 비교가 항상 거짓이 되어
그 분기가 조용히 스킵된다(`update(new_scale=...)` 내부의
`new_scale.device.type != self._device` assertion도 `tensor.device.type`
이 항상 ordinal 없는 `"cuda"`이므로 마찬가지로 어긋난다 -- 이 프로젝트는
`update()`를 인자 없이만 호출해 이 경로를 실제로 타지는 않지만
`self._device`가 device type만 담아야 한다는 내부 불변식을 보여준다).
scale tensor의 실제 device/ordinal 배치는 `scale()`이 `outputs.device`
(loss tensor가 이미 올라가 있는 실제 device)로 lazy 초기화하므로
`self._device`와 무관하다 -- 즉 ordinal을 전달해도 기능적 이득이
없다. 그래서 `torch.amp.GradScaler("cuda")`로 고정했다 -- model/tensor
의 실제 ordinal 배치는 기존과 동일하게 `model.to(device)`/
`images.to(device)`가 전담하므로 ordinal별 별도 scaler 설계는 여전히
필요 없지만, 그 이유는 "ordinal을 그대로 전달해도 무방하다"가 아니라
"애초에 전달할 필요가 없다"로 정정됐다.

## 6. `train_one_epoch()` 책임 분리

```python
def train_one_epoch(
    model, loader, optimizer, device="cpu", gradient_clip_norm=None,
    criterion=None,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    ...
    for images, labels in loader:
        ...
        optimizer.zero_grad()
        if autocast_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
```

세 조합이 전부 자연스럽게 표현된다:

```
FP32: autocast_dtype=None,           scaler=None            (기존 코드와 실행 semantics 동일)
FP16: autocast_dtype=torch.float16,  scaler=GradScaler       (Phase 4S와 실행 semantics 동일)
BF16: autocast_dtype=torch.bfloat16, scaler=None             (신규 -- FP32와 같은 backward/step, forward만 autocast)
```

`scaler is None`은 더 이상 "AMP를 쓰지 않는다"는 뜻이 아니라 "scaled
backward를 쓰지 않는다"는 뜻일 뿐이다. `autocast_dtype`이 `None`이면
(FP32 경로) 기존과 마찬가지로 `torch.amp.autocast`를 전혀 호출하지
않는다 -- 함수에 새 branch/파라미터가 추가됐지만, 이 branch를 타지
않는 FP32 기본 경로는 원래 코드와 동일한 PyTorch 호출 시퀀스를
실행하므로 numerical anchor가 바뀌지 않는다(실측: 5개 E2E, 아래 §12).

## 7. `run_training()` 배선

```python
autocast_dtype, scaler = _build_precision_execution(config, device)
...
train_loss = train_one_epoch(
    model, train_loader, optimizer, device=device,
    gradient_clip_norm=config.gradient_clip_norm, criterion=criterion,
    autocast_dtype=autocast_dtype, scaler=scaler,
)
```
scaler resume 로드(`resume_state.scaler_state_dict` 존재 시에만
비대칭 로드) 로직은 Phase 4S 그대로 무수정 -- `scaler`가 여전히
"scaled backward를 쓰는지"만 의미하므로 BF16(`scaler=None`)에서는
이 블록이 아무 일도 하지 않는다(기존 조건문 `if scaler is not None
and resume_state.scaler_state_dict is not None`이 자연스럽게 False).

## 8. exact-resume 실측(production 경로)

Phase 4R/4S와 동일한 Conv2d+BatchNorm2d+Dropout fixture로
`run_imagefolder_training_workflow()` + `checkpoint_out`을 그대로
사용해 `precision="bf16"` + `gradient_clip_norm=1.0`으로 continuous
5epoch vs split(3+2) CUDA resume을 비교한 결과, 다음이 전부 exact
PASS(실제 GPU에서 검증됨):

```
history, model_state_dict(BatchNorm running_mean/running_var/
num_batches_tracked 포함), best_state_dict, optimizer_state_dict,
scheduler_state_dict, cuda_rng_state
```

**scaler_state_dict는 두 checkpoint 모두 `None`이다** -- GradScaler
관련 state가 전혀 없는 상태에서도 exact-resume이 완전히 성립함을
production 경로에서 실측으로 재확인했다(조사 라운드의 scratch 실험과
동일한 결론).

## 9. checkpoint 영향 — `checkpoint.py`는 무수정

`scaler_state_dict: dict | None`의 기존 의미("이 필드가 필요한
precision일 때만 dict, 그 외는 None")가 BF16에도 그대로 자연스럽게
적용된다 -- `TrainingResult.scaler_state_dict`가 `_build_precision_execution()`
이 반환한 `scaler`(BF16이면 항상 `None`)로부터 채워지므로,
`checkpoint.py`의 payload 구성/loader 검증/`required_keys`/
`CHECKPOINT_FORMAT_VERSION`(=1) 어느 것도 변경할 필요가 없었다. 새
`precision_state`나 BF16 전용 checkpoint field는 추가하지 않았다 --
근거 없는 field라 조사 라운드 결론을 그대로 따랐다. 실제 구현
과정에서도 checkpoint 관련 변경이 필요하다는 새 증거는 발견되지
않았다(`git status --short`로 `checkpoint.py` 무수정 확인).

## 10. resume precision matrix

조사 라운드에서 FP32/FP16/BF16 3×3 = 9개 조합 전부(자기 자신 포함)
resume이 에러 없이 성공하고 optimizer momentum buffer가 항상
`float32`로 유지됨을 scratch 실측으로 확인했다. Phase 4T 구현에서는
이 정책을 그대로 코드에 반영했다: `precision`은 여전히
`RESUME_CONFIG_FIELDS`/`RESUME_CONFIG_LEGACY_DEFAULTS`에 포함되지
않으므로 `require_compatible_resume_config()`가 이 필드를 전혀
비교하지 않는다 -- 즉 3개 precision 값 중 무엇으로 바뀌어도(신규 값인
`"bf16"` 포함) 동일하게 자유로운 resume이 허용된다("resume 가능"과
"bitwise exact"는 별개 계약 -- exact는 same precision끼리, 그리고
BF16/FP16은 same-device일 때만).

## 11. backend compatibility matrix

```
cpu + fp32        허용
cpu + fp16        거부(Phase 4S)
cpu + bf16        거부(Phase 4T, 신규)

cuda/cuda:N + fp32   허용
cuda/cuda:N + fp16   허용(Phase 4S)
cuda/cuda:N + bf16   허용(Phase 4T, 신규)

기타 backend(mps/xpu 등) + fp16/bf16   거부(둘 다)
```

`_validate_precision_device_compatibility()`(workflow 레벨)의 조건을
`precision == "fp16"`에서 `precision in _CUDA_ONLY_PRECISIONS`
(`("fp16", "bf16")`)로 넓혔다. `_build_precision_execution()`(generic
`run_training()` 레벨)도 fp16/bf16 둘 다 같은 CUDA 판별 로직을 거친다
(§5). 이 두 layer는 서로 다른 경계를 보호하는 defense-in-depth로,
Phase 4S에서 확립한 3-layer 구조(`TrainingConfig`: 값 자체 검증 /
workflow: user-facing 조기 거부 / `run_training()`: generic caller
보호)를 그대로 유지했다. CUDA 판별을 위한 별도 공용 module-level
utility는 만들지 않았다 -- 한 줄짜리 boolean 체크(`device == "cuda"
or device.startswith("cuda:")`)를 위해 새 shared abstraction을 만드는
것은 과설계라고 판단했다(기존에도 이 판별은 `_build_precision_execution()`
안에서만 쓰였고, 지금도 마찬가지다).

## 12. FP32/FP16 regression

`precision="fp32"`(기본값): 기존 5개 E2E 스크립트를 전부 재실행해
numerical anchor가 정확히 동일함을 확인했다(`1.3386→0.2867`,
`2.3558→2.0817`, resume epoch5 `train_loss=1.017424`, ImageFolder
`2.3903→2.1509`). `test_train_one_epoch_fp32_path_never_calls_amp_api`
등 기존 monkeypatch 기반 "AMP API 미호출" 테스트도 무수정으로 계속
PASS한다.

`precision="fp16"`: **FP16 exact-resume regression의 핵심 fixture/
비교 항목/assertion/contract는 변경하지 않았다** --
`test_workflow_cuda_amp_fp16_same_device_exact_resume`이 검증하는
history/model_state_dict/BatchNorm buffers/best_state_dict/
optimizer_state_dict/scheduler_state_dict/cuda_rng_state/
scaler_state_dict exact match 조건은 전부 그대로다. 다만 production
helper가 `_build_grad_scaler()`(단일 GradScaler 반환)에서
`_build_precision_execution()`(tuple 반환)으로 리팩터링됐기 때문에,
이 테스트가 쓰는 test-only monkeypatch helper(`_fast_growth_grad_scaler`
→ `_fast_growth_precision_execution`)의 **이름/반환 형태와
monkeypatch target 문자열만** 그 리팩터링에 맞춰 갱신했다(그대로
뒀다면 monkeypatch target이 더 이상 존재하지 않아 `AttributeError`
로 이 테스트가 깨졌을 것 -- 실제로 재현 후 수정함). 이 wiring 갱신은
회귀 계약 자체를 바꾼 것이 아니라, 계약을 검증하는 test-only 배선을
production 리팩터링에 맞춰 따라간 것이다.

## 13. non-goals(재확인)

```
CPU BF16 AMP, AMP inference, AMP export, gradient accumulation,
DataLoader 최적화(pin_memory/non_blocking 등), torch.compile, multi-GPU,
distributed training, FP8, automatic precision selection,
native BF16 hardware requirement(강제 검증 없음), 성능 benchmark,
GradScaler tuning parameter 노출
```

## 14. 성능 caveat

Phase 4T가 보장하는 것은 "CUDA BF16 training을 기능적으로 지원한다"
이지 "BF16이 FP32보다 빠르다"가 아니다. Tensor Core 기반 네이티브
BF16 하드웨어가 없는 GPU에서는 `torch.amp.autocast(dtype=torch.bfloat16)`
가 emulation으로 동작할 수 있고(§3, 로컬 GTX 1080/compute capability
6.1이 정확히 이 경우), 이 경우 FP32 대비 속도 이득이 없거나 더 느릴
수 있다. production code는 `torch.cuda.is_bf16_supported(including_emulation=False)`
같은 hardware capability 검증을 강제하지 않는다 -- 기능적으로
정상 동작하면 하드웨어 세대와 무관하게 실행을 허용한다(자동
precision 선택/hardware 기반 fallback도 만들지 않았다).
