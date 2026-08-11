# Phase 4S: Same-device CUDA AMP (FP16) Training — 설계안

## 1. 목적

Phase 4R이 CUDA training의 same-device exact-resume 계약을 만들었지만
AMP(Automatic Mixed Precision)는 다루지 않았다. Phase 4S는 CUDA FP16
autocast + GradScaler를 도입하고, GradScaler state를 checkpoint에
추가해 Phase 4R의 exact-resume 계약을 AMP-enabled training까지
확장한다. `precision="fp32"`(기본값)에서는 CPU/CUDA 기존 동작을 한
글자도 바꾸지 않는다.

## 2. Phase 4R baseline

Phase 4R까지의 CUDA training lifecycle:

```
model build(CPU) -> model.to(device) -> DataLoader 생성
-> checkpoint_hook 생성
-> _cuda_deterministic_context(enabled=device!="cpu"):
     CPU/CUDA RNG 복원 -> run_training() -> 최종 RNG 캡처
-> loader_generator state 캡처
-> checkpoint 저장 -> best_model CPU 최종 test/export
```

`run_training()` 내부는 `_build_optimizer`/`_build_scheduler`/
`_build_criterion`을 config로 생성하고, `train_one_epoch()`이 매 배치
`zero_grad -> forward -> loss -> backward -> [clip] -> step`을 수행한다.
AMP는 정확히 이 `train_one_epoch()`의 forward/backward/step 구간에만
필요하다는 것이 Phase 4R 코드 재조사로 확인됐다.

## 3. AMP API 조사

설치된 PyTorch(`2.12.1+cu126`)에서 실측: `torch.cuda.amp.autocast`/
`torch.cuda.amp.GradScaler`는 `FutureWarning`으로 deprecated이며
`torch.amp.autocast(device_type="cuda", dtype=torch.float16)`/
`torch.amp.GradScaler("cuda", ...)`가 권장 API임을 직접 확인했다(경고
메시지: `torch.cuda.amp.autocast(args...) is deprecated. Please use
torch.amp.autocast('cuda', args...) instead.` 등). production code는
후자만 사용한다.

## 4. precision scope

최소 후보(S-A: CUDA FP16 autocast+GradScaler만)를 채택했다. BF16은
GradScaler가 필요 없어 lifecycle이 근본적으로 다르고(scaler 있는
경로/없는 경로를 동시에 검증해야 함), 로컬 GPU(GTX 1080, Pascal)에서
`torch.cuda.is_bf16_supported()`가 `True`를 반환하지만 이는 emulation
포함 여부를 구분하지 않는 API라 네이티브 지원 여부가 불확실하다.
CPU AMP도 비포함(non-goal). `TrainingConfig.PRECISION_CHOICES = ("fp32",
"fp16")`으로 코드 레벨에서도 확정했다. `precision="fp16"`은 CUDA device
에서만 허용하며, CPU와 조합하면 silent FP32 fallback 없이 명확히
거부한다 -- 이 규칙은 ImageFolder workflow 경계뿐 아니라
`TrainingConfig`+`run_training()`을 직접 쓰는 generic 호출 경로에서도
강제된다(아래 §9 참고). 최초 구현에서는 `_build_grad_scaler()`가
`cpu`+`fp16`을 `None`(=조용한 FP32 fallback)으로 잘못 처리하던 버그가
있었고, 이를 실측으로 재현한 뒤 `ValueError`로 명확히 거부하도록
수정했다.

## 5. GradScaler lifecycle 실측

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # clipping 있을 때만
clip_grad_norm_(...)
scaler.step(optimizer)              # inf/nan 있으면 optimizer.step() 자체를 skip
scaler.update()                     # scale 조정
```

`unscale_()`을 생략하면 grad norm이 `scale`배(실측: 1024배)로 부풀어
clipping 임계값이 무의미해짐을 직접 확인했다. 같은 step에서 `unscale_()`
을 두 번 호출하면 `RuntimeError`(`unscale_() has already been called on
this optimizer since the last update()`)가 남을 확인했다. 큰
`init_scale`로 강제 overflow를 유도하면 `scaler.step()`이 실제로
`optimizer.step()`을 skip하고(params 불변), `scaler.update()`가
`backoff_factor`만큼 scale을 낮추고 `_growth_tracker`를 리셋함을 확인했다.

## 6. scaler state 실측

`scaler.state_dict()` 기본값: `{'scale': 65536.0, 'growth_factor': 2.0,
'backoff_factor': 0.5, 'growth_interval': 2000, '_growth_tracker': 0}`
-- 전부 순수 Python `float`/`int`(텐서 아님). `torch.save({...,
"scaler_state_dict": ...}, path)` 후 `torch.load(path,
weights_only=True)`가 문제없이 성공함을 실측 확인했다(checkpoint.py의
`load_training_checkpoint()`가 이미 `weights_only=True`를 쓰므로 완전히
호환). `growth_interval`을 작게(예: 2) 주면 몇 step 안에 `scale`/
`_growth_tracker`가 실제로 변함을 확인했다 -- 이 값은 production
config/CLI에 노출하지 않고 optional CUDA test 내부의 `_build_grad_scaler`
monkeypatch에서만 사용한다(T1 방식, `tests/training/
test_imagefolder_workflow.py`의 `_fast_growth_grad_scaler`).

## 7. positive/negative resume control

Conv2d+BatchNorm2d+Dropout fixture + `growth_interval=2` GradScaler +
`gradient_clip_norm=1.0`으로 continuous 5epoch vs split(3+2, model/
optimizer/scaler/CPU RNG/CUDA RNG/loader_generator 전부 복원) 비교 --
**model/optimizer/scaler state_dict/CUDA RNG state 전부 exact
match: True**. scaler state만 의도적으로 복원하지 않고(fresh scaler로
resume) 나머지는 전부 정확히 복원한 negative control에서는 결과가
명확히 갈라짐(예: 첫 resume epoch loss가 서로 다름) -- GradScaler
state가 same-device AMP exact-resume에 **필수**임을 직접 증명했다. 이
실험은 `tests/training/
test_imagefolder_workflow.py::test_workflow_cuda_amp_fp16_same_device_exact_resume`
로 production 경로에 그대로 고정했다(negative control은 production
suite에 별도로 넣지 않음 -- 이 설계 조사에서 이미 증명됨).

## 8. gradient clipping 통합

`loop.py`의 `train_one_epoch()`에 `scaler: torch.amp.GradScaler | None
= None` 파라미터를 추가했다. 함수 자체에는 `if scaler is not None: ...
else: ...`라는 새 branch가 추가됐지만, `scaler is None`이면(기본값)
`else` branch가 AMP API를 전혀 호출하지 않고 Phase 4A~4R의 기존 FP32
forward/loss/backward/[clip]/optimizer.step 계산 semantics를 그대로
실행한다 -- 즉 "코드가 문자 그대로 동일"한 것이 아니라 "FP32 경로의
numerical/execution semantics가 기존과 동일"하다는 뜻이다("통일된
disabled-scaler wrapper" 방식은 채택하지 않음 -- 기존 FP32 경로가 새
AMP 코드를 전혀 거치지 않는 것이 회귀 위험을 구조적으로 없애는 가장
안전한 선택이라고 판단했다). `scaler`가 주어지면:

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
if gradient_clip_norm is not None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
scaler.step(optimizer)
scaler.update()
```

`autocast` context 안에는 forward+loss 계산만 두고, backward/clip/step/
update는 그 밖에서 호출한다.

## 9. config/CLI

`precision`은 **`TrainingConfig`** 필드로 귀속했다(`ImageFolderWorkflowRequest`
가 아님) -- `device`(Phase 4Q, "어디서 계산하는가")와 달리 precision은
"무엇을 계산하는가"(dtype)를 바꾸는 training 동역학 hyperparameter이며,
`gradient_clip_norm`/`label_smoothing`/`class_weights`와 정확히 같은
범주다. `PRECISION_CHOICES = ("fp32", "fp16")` + `_require_one_of()`로
검증한다(`optimizer`/`lr_scheduler`와 동일한 패턴) -- `TrainingConfig`는
device를 모르므로 이 값 자체(fp32/fp16 중 하나인가)만 검증한다.
**RESUME_CONFIG_FIELDS에는 포함하지 않는다** -- 현재 지원 optimizer와
실측한 AMP 경로에서는 optimizer state(momentum buffer 등)가 precision과
무관하게 float32로 유지됨을 확인했으므로(§11) optimizer/scheduler
구조를 바꾸지 않는다.

precision-device 조합 검증은 3개 layer로 나뉜다:

```
TrainingConfig.__post_init__()
  -> precision 값 자체가 PRECISION_CHOICES에 속하는지만 검증

ImageFolder workflow(_validate_precision_device_compatibility())
  -> device를 아는 시점에 cpu+fp16을 user-facing fail-fast로 조기 거부
     (dataset/model 준비 전)

run_training()/_build_grad_scaler()
  -> workflow를 거치지 않고 TrainingConfig+run_training()을 직접 쓰는
     generic caller까지 보호하는 lower-level invariant(cpu+fp16이면
     ValueError, None을 반환해 조용히 FP32로 대체하지 않음)
```

뒤 두 layer는 중복 검증이 아니라 서로 다른 경계를 보호하는
defense-in-depth다. CLI는 `--precision {fp32,fp16}`(기본값 `fp32`)를
`scripts/train_imagefolder.py`에 추가했다 -- 새 stdout echo는 추가하지
않았다(최소 scope 원칙).

## 10. checkpoint schema

payload key: **`scaler_state_dict`**(`optimizer_state_dict`/
`scheduler_state_dict`와 동일한 네이밍 패턴). `cuda_rng_state`와 달리
`save_training_checkpoint()`의 별도 키워드 인자가 아니다 -- **GradScaler는
`run_training()` 내부(loop.py)에 있는 객체이므로, 그 최종 state는
이미 `TrainingResult.scaler_state_dict`에 담겨 반환된다**. 따라서
`save_training_checkpoint()`는 payload에 `"scaler_state_dict":
training_result.scaler_state_dict`만 추가하면 되고, 새 함수 파라미터가
필요 없다. `CHECKPOINT_FORMAT_VERSION`은 `1` 유지(순수 additive optional
field, Phase 4R의 `cuda_rng_state`와 동일한 근거). loader는
`"scaler_state_dict" in loaded`일 때만 `None`/`dict`인지 최소 검증하고,
내부 key(`scale`/`growth_factor`/...)는 검증하지 않는다(PyTorch 버전
의존 implementation detail).

## 11. legacy/cross-precision policy

pre-4S checkpoint에는 `scaler_state_dict` 키 자체가 없다 -- structural
필수 key가 아니므로 정상 로드된다. resume 시 처리는 **비대칭**이다
(scheduler의 엄격한 양방향 mismatch 검증과 다름): `scaler is not None
and resume_state.scaler_state_dict is not None`일 때만 로드하고, 그
외 조합(precision="fp16"인데 state 없음 / precision="fp32"인데 state
있음)은 에러 없이 자연스럽게 처리된다(전자는 fresh scaler, 후자는 그
값을 그냥 쓰지 않음). `require_compatible_resume_config()`는 `precision`
을 전혀 비교하지 않는다(RESUME_CONFIG_FIELDS 미포함이므로). 실측으로
FP32→AMP, AMP→FP32 양방향 resume이(현재 지원 optimizer 기준)
optimizer state 손상 없이 정상 동작함을 확인했다.

**"precision 변경 resume을 허용한다"와 "지원되지 않는 device+precision
조합을 허용한다"는 서로 다른 계약이다.** 예를 들어 FP32 checkpoint를
`precision="fp16"`으로 resume하는 것은 허용되지만(portable, exact
미보장), 그 resume이 실제로 실행되려면 새 training device는 여전히
CUDA여야 한다 -- `device="cpu"`+`precision="fp16"`으로 resume을
시도하면 §9의 두 layer(workflow 조기 검증, `_build_grad_scaler()`
invariant)에 의해 여전히 명확히 거부된다. precision 변경 resume
정책은 "새 precision이 무엇이든" 허용한다는 뜻이 아니라 "새 precision이
유효한 조합일 때 자유롭게 바꿀 수 있다"는 뜻이다.

## 12. same-device exact contract

공식 계약은 Phase 4R의 R1 범위(same physical CUDA device / 같은 머신 /
같은 소프트웨어 환경)를 그대로 상속하되, **scaler_state_dict가 있는
새 checkpoint에서만** 보장된다. production workflow
(`run_imagefolder_training_workflow()` + `checkpoint_out`)로
Conv2d+BatchNorm2d+Dropout fixture + `precision="fp16"` +
`gradient_clip_norm=1.0`을 continuous 5epoch vs split(3+2)로 비교한
결과, 다음이 전부 exact PASS(실제 GPU에서 검증됨):

```
history, model_state_dict(BatchNorm buffers 포함), best_state_dict,
optimizer_state_dict, scheduler_state_dict, cuda_rng_state, scaler_state_dict
```

## 13. FP32 regression

`precision="fp32"`(기본값)에서 `train_one_epoch()`은 `torch.amp.autocast`/
`GradScaler` API를 전혀 호출하지 않는다 -- monkeypatch("호출되면 fail")
로 loop.py 단위 테스트와 production CPU workflow 양쪽에서 직접 고정했다.
`device="cuda"`+`precision="fp32"`(기본값)에서도 AMP 코드는 실행되지
않고 Phase 4R의 CUDA RNG exact-resume/deterministic context만 그대로
적용된다 -- 기존 `test_workflow_cuda_same_device_exact_resume_conv_bn_dropout`
은 무수정으로 유지했고 실제 GPU에서 계속 PASS함을 확인했다. 5개 기존
CPU E2E 스크립트도 재실행해 numerical anchor가 전부 동일함을 확인했다
(`1.3386→0.2867`, `2.3558→2.0817`, epoch5 `train_loss=1.017424`,
`2.3903→2.1509` 등). 여기서 "기존과 동일"은 코드가 문자 그대로
바뀌지 않았다는 뜻이 아니라 -- `train_one_epoch()`/`run_training()`
에는 실제로 새 파라미터/branch/dataclass 필드가 추가됐다 -- 그 새
코드 경로를 타지 않는 FP32 기본 설정에서는 계산 semantics와 수치
결과가 기존과 동일하다는 뜻이다.

## 14. validation/test strategy

**train only AMP, validation/test는 항상 FP32**(V2). 근거: (a)
label_smoothing/class_weights가 이미 "training loss에만 적용, evaluate()
는 항상 unsmoothed/unweighted"로 분리돼 있어(Phase 4N/4P) 같은 원칙을
따르는 것이 아키텍처 일관성이 높다. (b) val_loss가 scheduler/early
stopping/best model selection에 직접 쓰이므로 그 수치적 의미를 바꾸지
않는 것이 안전하다. `evaluate()`/`evaluate_classification_metrics()`는
Phase 4S에서 전혀 수정하지 않았다. 최종 test 평가/TorchScript export/
C++ parity도 Phase 4Q/4R과 동일하게 항상 CPU+FP32로 고정 유지했다.

**CPU+FP16 silent fallback 회귀 방지도 두 경계 모두에서 test로
고정했다:**

```
ImageFolder workflow: cpu+fp16 조기 거부
  test_workflow_rejects_cpu_fp16_before_training_starts

generic run_training(): cpu+fp16 silent fallback 없이 거부
  test_run_training_rejects_cpu_device_with_fp16_precision_without_workflow
  (workflow를 거치지 않고 TrainingConfig+run_training()을 직접 호출해도
  거부됨을 monkeypatch로 증명 -- train_one_epoch()가 호출되지 않음을
  확인해 실제 batch forward 전에 거부됨을 보장)
```

## 15. performance tradeoff

deterministic FP16 kernel을 쓰므로 CUDA training 속도/메모리 사용량이
하드웨어에 따라 달라질 수 있다. GTX 1080(Pascal, Tensor Core 없음)은
FP16 처리량이 FP32 대비 우월하지 않을 수 있어 speedup을 보장하지
않는다 -- 이번 Phase에서 별도 성능 벤치마크는 수행하지 않았다.

## 16. non-goals

```
BF16, CPU AMP, multi-GPU/distributed training, gradient accumulation,
GradScaler tuning parameter(init_scale/growth_interval/backoff_factor 등)
의 config/CLI 노출, dynamic loss scaling tuning API, 성능 benchmark,
TorchScript AMP inference, C++ AMP inference, torch.compile,
CUBLAS_WORKSPACE_CONFIG 자동 설정(Phase 4R 정책 그대로 유지)
```
