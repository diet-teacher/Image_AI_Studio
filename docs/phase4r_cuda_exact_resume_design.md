# Phase 4R: Same-device CUDA Exact Resume — 설계안

## 1. 목적

Phase 4Q에서는 CUDA가 포함된 resume을 state portability 수준으로만
지원했고(model/optimizer state는 정상 복원되지만) bitwise exact는
보장하지 않았다(CUDA RNG state를 checkpoint하지 않았기 때문). Phase 4R은
그중 **same physical CUDA device / 같은 머신 / 같은 소프트웨어 환경**
이라는 좁은 조건만 exact contract로 승격해, 그 조건에서는 continuous
training과 split+resume training이 실제로 bitwise exact함을 만든다
(`cuda:0`→다른 물리 GPU로의 이동이나 multi-GPU는 이 승격 대상이 아니며
§13 참고).

## 2. Phase 4Q gap (재확인)

Phase 4Q 조사에서 확인한 사실: `imagefolder_workflow.py`의 `_set_seed()`
가 `torch.cuda.manual_seed_all(seed)`를 fresh 학습 시작 시 1회만
호출할 뿐, `torch.cuda.get_rng_state()`/`set_rng_state()`는 코드베이스
어디에도 없었다. checkpoint가 저장/복원하는 RNG는 `cpu_rng_state`/
`loader_generator_state` 둘뿐이었다.

## 3. RNG inventory (Phase 4Q~4R 조사 종합)

* CPU 전역 RNG: model weight 초기화, `nn.Dropout`(CPU 학습 시),
  `cpu_rng_state`로 checkpoint에 이미 포함.
* 독립 `torch.Generator`: synthetic dataset 생성, DataLoader shuffle
  (`loader_generator_state`로 이미 checkpoint에 포함) — 전역 RNG와 무관.
* **CUDA RNG**: `nn.Dropout`이 CUDA에서 실행되면 CPU RNG가 아니라 이
  독립 스트림을 소비함을 직접 실측 확인(§4) — Phase 4R 이전에는 어디에도
  캡처/복원되지 않았다.
* Python `random`/NumPy random: production training 경로에 사용처 없음.

## 4. CUDA RNG positive/negative control 실측

실제 project 함수(`build_model`, `run_training`, `TrainingConfig`,
`TrainingResumeState`)로 continuous 5epoch vs split 3+2epoch(CUDA)를
비교한 결과:

* **positive control**(3epoch 시점 `cpu_rng_state`+CUDA RNG state를
  캡처해 resume 직전 둘 다 복원) → `train_losses`/`val_losses`/model
  state_dict/optimizer state **exact match: True**.
* **negative control**(CUDA RNG를 의도적으로 다른 값으로 교란한 뒤
  resume) → **exact_match: False**(전부 명확히 어긋남).

CUDA Dropout 자체도 직접 실측: 같은 CUDA RNG snapshot을 두 번 복원하고
같은 Dropout을 두 번 호출하면 출력이 bitwise 동일함을 확인, CPU RNG는
CUDA Dropout 호출로 전혀 변하지 않음을 확인(두 스트림이 독립적).

## 5. Conv2d/BatchNorm 비결정성 실측

Linear/ReLU/Dropout만 있는 모델은 기본 설정(`cudnn.deterministic=False`)
에서도 fresh run A == fresh run B였지만, **Conv2d+BatchNorm2d+Dropout
모델은 기본 설정에서 fresh run A != fresh run B**(losses가 7번째
유효숫자에서 갈라짐 — cuDNN convolution backward의 잘 알려진
비결정성이 실제로 재현됨)임을 실측 확인했다. 같은 모델에
`torch.backends.cudnn.deterministic=True` + `torch.use_deterministic_algorithms(True)`
를 켜면 fresh run A == fresh run B로 복구됨(RuntimeError 없음).

추가 실측: `use_deterministic_algorithms(True)`만, `cudnn.deterministic=True`
만, 둘 다 — 세 조합 전부 이 모델에서는 개별로도 충분했지만, 더 넓은
공식 API인 전자를 포함해 **둘 다 명시하는 조합을 최종 채택**했다(비용
없음, 향후 layer 확장의 안전판).

MaxPool2d/AdaptiveAvgPool2d도 `use_deterministic_algorithms(True)` 하에서
RuntimeError 없이 반복 호출 bitwise exact함을 개별 실측했다.
ResidualBlock/BranchBlock은 코드를 직접 읽어 자체 CUDA kernel이 아니라
이미 검증된 layer(Conv2d/BatchNorm2d/ReLU + elementwise add/`torch.cat`)
의 순수 조합(wrapper)임을 확인했다 — 별도 kernel 실측이 불필요하다.

## 6. deterministic context 설계

`imagefolder_workflow.py`에 private context manager
`_cuda_deterministic_context(enabled: bool)`를 추가했다:

```python
@contextmanager
def _cuda_deterministic_context(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    previous_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms_enabled,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
```

`enabled=(request.device != "cpu")`로 호출해 CPU training에서는 이
전역 설정을 전혀 읽지도 쓰지도 않는다.

`torch.use_deterministic_algorithms()`에는 `enabled`와 별개로 process-global
`warn_only` 상태가 있다(`torch.is_deterministic_algorithms_warn_only_enabled()`
로 조회). 초기 구현은 이 상태를 snapshot/restore하지 않아, context 종료
후 caller가 원래 `warn_only=True`로 쓰고 있었더라도 기본값 `False`로
덮어써지는 버그가 있었다(로컬 실측으로 재현: `use_deterministic_algorithms(True,
warn_only=True)` 이후 `enabled`만 복원하면 `warn_only`가 `True`에서
`False`로 조용히 바뀜). 지금은 `previous_warn_only`를 진입 시 함께
캡처해 `finally`에서 `enabled`와 함께 정확히 복원한다.

context **내부** 정책은 `warn_only=False`(strict fail-fast)이다 --
지원 ModelSpec에 향후 deterministic 구현이 없는 CUDA 연산이 추가되면,
`warn_only=True`로 경고만 내고 계속 실행해 exact-resume 계약을 조용히
깨뜨리는 대신 명확한 RuntimeError로 즉시 실패해야 한다(의도된 strict
behavior). 이는 context 내부에서만 강제되는 정책이며, caller의
진입 전 `warn_only` 값 자체를 영구히 바꾸는 것은 아니다 --
"context 내부 정책(`warn_only=False`)"과 "caller 상태 복원(원래
`warn_only` 값)"은 서로 다른 개념이다.

## 7. global state restore 실측

복원 대상은 다음 **4개 process-global state**다:

```python
torch.are_deterministic_algorithms_enabled()
torch.is_deterministic_algorithms_warn_only_enabled()
torch.backends.cudnn.deterministic
torch.backends.cudnn.benchmark
```

정상 종료와 예외 종료 양쪽에서 sentinel 값(현재 값과 다른 조합, 예:
진입 전 `warn_only=True`/`benchmark=True`)이 context 종료 후 이 4개
모두 정확히 복원됨을 로컬에서 직접 실행/확인했다 — `try/finally`가 예외 발생 여부와 무관하게 항상
실행되므로, production에서 OOM/unsupported-op 등 어떤 예외가 나도
caller의 전역 상태가 오염되지 않는다. 이 실측을 그대로
`tests/training/test_imagefolder_workflow.py`의
`test_cuda_deterministic_context_restores_state_on_normal_exit`/
`test_cuda_deterministic_context_restores_state_on_exception`로
고정했다(GPU 없이도 실행 가능 -- 이 flag들은 CUDA 하드웨어 없이도
설정/조회 가능하다).

## 8. CUBLAS_WORKSPACE_CONFIG 조사

subprocess 두 개(`CUBLAS_WORKSPACE_CONFIG` unset / `:4096:8`로 설정)로
`use_deterministic_algorithms(True)` 하의 `nn.Linear` forward/backward를
비교한 결과, **RuntimeError 없음, 두 경우의 loss 값이 완전히 동일**
(이 로컬 환경 — CUDA 12.6/cuDNN 91002 — 에서의 실측 결과이며, 다른
PyTorch/CUDA/cuDNN 조합에서도 동일하다는 일반적 보장은 아니다). 이
프로젝트의 실제 연산/모델 크기 범위, 이 로컬 환경에서는 이 환경변수
설정 유무가 아무 차이를 만들지 않았다는 뜻이다. 따라서 Phase 4R
production code는 이 환경변수를 **전혀 건드리지 않는다** -- workflow
실행 중 `os.environ[...]`을 조용히 바꾸는 설계는 CUDA/cuBLAS 초기화
시점 문제 때문에 채택하지 않았다(늦은 env var 변경은 반영이 보장되지
않음). 다른 CUDA 환경에서 향후 필요해지면 process 시작 전 설정이 필요할
수 있다는 점만 문서화한다.

## 9. checkpoint schema

`save_training_checkpoint()`에 `cuda_rng_state: torch.Tensor | None = None`
을 마지막 keyword-only 파라미터로 추가했다(기존 모든 call site가
무수정으로 계속 동작). payload에 `"cuda_rng_state": cuda_rng_state`를
**항상 명시적으로 쓴다**(CPU checkpoint에도 key가 존재하고 값이
`None`) -- "legacy라 키가 아예 없음"과 "새 checkpoint인데 CPU라서
값이 None임"을 명확히 구분할 수 있다.

`CHECKPOINT_FORMAT_VERSION`은 `1`을 그대로 유지한다 -- 순수 additive
optional execution-state 필드이기 때문이다. `RESUME_CONFIG_FIELDS`/
`RESUME_CONFIG_LEGACY_DEFAULTS`/`TrainingConfig`는 무수정이다 --
`cuda_rng_state`는 training config가 아니다(Phase 4L의 config
migration 패턴을 재사용하지 않는다).

## 10. legacy compatibility

`load_training_checkpoint()`의 structural `required_keys`에
`cuda_rng_state`를 넣지 않았다 -- pre-4R checkpoint에는 이 key 자체가
없다. `load_training_checkpoint()` 자신은 key가 없어도 반환하는
payload에 `"cuda_rng_state": None`을 새로 삽입하지 않는다(구조적
validation만 통과시키고 파일에 저장된 key 구성을 그대로 반환) --
`assert "cuda_rng_state" not in payload`가 legacy checkpoint 테스트의
실제 계약이다. 부재를 `None`으로 취급하는 것은 loader가 아니라 호출
측(`imagefolder_workflow.py`의 `payload.get("cuda_rng_state")`)의
책임이다. 존재하면 `None` 또는 `torch.Tensor`인지만 검증한다(shape
`==16` 같은 PyTorch 버전에 의존적인 값은 검증하지 않는다 -- 기존
`cpu_rng_state`/`loader_generator_state`도 shape을 검증하지 않는 최소
검증 철학과 동일). Warning은 추가하지 않는다 -- pre-4R CUDA checkpoint는
resume 자체는 허용(portable-only)하되 same-device exact 계약 대상이
아니라는 사실을 README/design doc으로만 문서화한다.

## 11. capture/restore lifecycle

```
model 준비 / model.to(device)
DataLoader 준비
checkpoint_hook 생성(아직 호출 안 됨)

with _cuda_deterministic_context(enabled=device != "cpu"):
    if cpu_rng_state is not None: torch.set_rng_state(cpu_rng_state)
    _restore_cuda_rng_state(cuda_rng_state, device)   # device="cpu"거나 state=None이면 아무 것도 안 함

    run_training(...)   # checkpoint_hook이 매 scheduled epoch마다
                         # _capture_cuda_rng_state(device)를 직접 호출(읽기 전용)

    cpu_rng_state_after = torch.get_rng_state().clone()
    cuda_rng_state_after = _capture_cuda_rng_state(device)
# context 종료 -- 전역 deterministic 설정 원복

loader_generator_state_after = ...
최종 checkpoint 저장(cuda_rng_state=cuda_rng_state_after 포함)
best_model CPU 최종 test/export (context 밖, CUDA RNG/deterministic과 무관)
```

`_capture_cuda_rng_state(device)`/`_restore_cuda_rng_state(state,
device)` 둘 다 `device == "cpu"`면 `torch.cuda.*` API를 전혀 호출하지
않는다(CPU 경로가 CUDA API를 절대 건드리지 않는다는 계약을 코드로
강제). `model.to(cuda)`/`_build_optimizer()`/`_build_criterion()`/
`evaluate()`(eval mode) 전부 CUDA RNG를 소비하지 않음을 개별 실측
확인했으므로, 이 순서 안에서 restore와 `run_training()` 사이에 다른
RNG 소비 연산이 끼어들 걱정이 없다. `EpochCheckpointView`/
`run_training()`/`loop.py`는 **전혀 수정하지 않았다** -- checkpoint hook
이 이미 `torch.get_rng_state()`를 직접 호출하는 기존 패턴 그대로,
`torch.cuda.get_rng_state()`도 hook 내부에서 직접(읽기 전용) 호출한다.

## 12. same-device exact contract

공식 보장 범위는 **R1(same physical CUDA device / 같은 머신 / 같은
소프트웨어 환경, continuous == split+resume, bitwise exact)뿐**이다.
production workflow(`run_imagefolder_training_workflow()` +
`checkpoint_out`)를 그대로 사용해 Conv2d+BatchNorm2d+Dropout fixture로
continuous 5epoch vs split 3+2epoch CUDA resume을 비교한 결과, 다음이
전부 exact PASS:

```
history(train_losses/val_losses/val_accuracies/best_epoch/best_val_loss/stopped_early)
model_state_dict 전체(Conv2d/BatchNorm2d weight, running_mean, running_var, num_batches_tracked 포함)
best_state_dict 전체
optimizer_state_dict(deep, momentum buffer 포함)
scheduler_state_dict(deep)
두 checkpoint의 최종 cuda_rng_state 자체
```

## 13. CPU/cross-device boundaries

* **CPU→CPU**: 기존 exact-resume 계약 완전히 그대로 유지(5개 E2E
  numerical anchor 무변경으로 확인).
* **CPU↔CUDA**: Phase 4Q의 state portability만 적용하며, Phase 4R의
  bitwise exact-resume 계약 대상이 아니다. CPU와 CUDA는 서로 다른
  execution backend이므로 이 Phase에서는 cross-backend bitwise
  equality를 검증하거나 보장하지 않는다.
* **cuda:0→다른 physical CUDA device / cross-GPU architecture**:
  Phase 4R의 bitwise exact-resume 계약 대상이 아니다. 로컬 환경에 GPU가
  1개뿐이라 실제 cross-device equality를 검증하지 않았으며, bitwise
  exact를 보장하지 않는다.
* **multi-GPU/distributed**: exact 여부 이전에 training 자체가 현재
  프로젝트의 지원 범위 밖이다.

## 14. test strategy

* **CPU-only**: checkpoint 직렬화 round-trip(None/Tensor/legacy-missing-key/
  invalid-type), 새 writer가 CPU checkpoint에도 key를 명시적으로 쓰는지,
  deterministic context의 `enabled=False`/정상 종료/예외 종료 3가지
  경로 — 각각 algorithms enabled/`warn_only`/`cudnn.deterministic`/
  `cudnn.benchmark` **4개 process-global state 전부**를 진입 전/중/후
  기준으로 검증(CUDA 하드웨어 없이도 flag 설정/조회가 가능하므로 GPU
  없이 전부 실행 가능), `_capture_cuda_rng_state`/`_restore_cuda_rng_state`가
  `device="cpu"`에서 CUDA API를 호출하지 않음(monkeypatch로 "호출되면
  fail"), production CPU workflow(fresh+resume) 전체 경로에서도 동일하게
  CUDA API 미호출, production CPU workflow 실행 전후 이 4개
  process-global state가 sentinel 값 그대로 유지됨(테스트 자신이
  sentinel로 바꾼 값은 각 테스트의 `finally`에서 원래 pytest 프로세스
  상태로 복원 — `warn_only`도 포함).
* **optional CUDA**(`skipif`): production workflow의 실제 checkpoint
  경로를 그대로 사용하는 단일 핵심 regression 테스트(§12) -- Dropout-only
  fixture 대신 Conv2d+BatchNorm2d+Dropout fixture를 써서 CUDA RNG 소비
  경로와 deterministic algorithm이 필요한 경로를 동시에 커버한다.
  legacy CUDA checkpoint 전용 dedicated 테스트, fresh-run reproducibility
  production 테스트, graceful stop 전용 CUDA 테스트는 추가하지 않았다
  -- 각각 CPU-only loader test/조사 단계의 실측 근거/Phase 4K 기존
  regression으로 충분히 커버된다고 판단했다(과도한 GPU 테스트 증식을
  피함).
* 테스트 자신이 sentinel 값으로 바꾼 전역 설정은 각 테스트의 `finally`
  에서 원래 pytest 프로세스 상태로 복원해 테스트 간 오염을 방지했다
  (production contract가 아니라 순수 테스트 isolation 책임).

## 15. performance tradeoff

deterministic kernel을 자동 선택하므로 CUDA training 일부가 Phase 4Q
대비 느려질 수 있다. 이는 버그가 아니라 exact-resume guarantee의
tradeoff이며, 이번 Phase에서 별도 성능 벤치마크는 추가하지 않았다.

## 16. non-goals (재확인)

CPU↔CUDA exact, cross-GPU exact, multi-GPU/distributed training 및
그 RNG checkpoint(multi-GPU는 exact-resume 여부 이전에 training 자체가
지원 범위 밖), AMP/GradScaler, `--deterministic`/`--exact-resume`/`--cuda-rng` 같은 새
CLI 옵션, 새 stdout/stderr 출력, `CUBLAS_WORKSPACE_CONFIG` 자동 설정,
metadata/result schema에 exact-capability flag 추가, fresh-run 전역
reproducibility를 별도로 강제하는 기능, legacy checkpoint에 대한
warning, 새 GPU E2E 스크립트.
