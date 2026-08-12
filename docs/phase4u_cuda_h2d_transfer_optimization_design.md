# Phase 4U: CUDA H2D Transfer Optimization — 설계안

## 1. 목적

CUDA ImageFolder training의 host(CPU) → device(CUDA) batch 전송 경로에
`pin_memory`/`non_blocking` 두 가지 순수 runtime 최적화 옵션을 추가한다.
사전 투자 조사 라운드에서 `num_workers`/`persistent_workers`/
`prefetch_factor`까지 함께 조사했으나, 이번 Phase는 명시적으로
`pin_memory`/`non_blocking`만 구현 범위로 좁힌다(§9 non-goals).

## 2. 조사 라운드 핵심 발견 요약

- 설치된 PyTorch(`2.12.1`) 소스를 직접 확인한 결과, `DataLoader`의
  `base_seed`는 `num_workers` 값과 무관하게 새 iterator가 생성될 때마다
  (즉 `persistent_workers=False`라면 매 epoch) 항상 draw된다 --
  `num_workers` 자체는 loader generator 소비 패턴에 영향이 없다(이는
  `num_workers`의 exact-resume 안전성 근거일 뿐, 이번 Phase가
  `num_workers`를 지원한다는 뜻은 아니다 -- §9).
- `persistent_workers=True`는 DataLoader 객체 생애주기당 단 한 번만
  `base_seed`를 draw한다. continuous run(하나의 DataLoader 객체)과
  resume 후 새로 만들어진 DataLoader 사이에 이 draw 횟수가 어긋나
  exact-resume이 깨짐을 실측으로 재현했다 -- 이 프로젝트는
  `persistent_workers`를 지원하지 않는다(production에 이 옵션 자체를
  노출하지 않는다).
- `pin_memory`/`non_blocking`은 순수 host-memory 전송 최적화로 loader
  generator를 전혀 소비하지 않는다 -- resume 전후로 값이 달라져도
  exact-resume에 영향이 없음을 실측 확인했다(§6, §8).
- `num_workers>0`은 이 프로젝트의 전형적인(작은) 데이터셋 규모 +
  Windows spawn 방식에서, exact-resume 안전을 위해 `persistent_workers=False`
  가 강제되어 매 epoch 워커 재생성 비용이 반복되므로 오히려
  `num_workers=0`보다 뚜렷이 느림을 실측했다 -- 이 성능 역효과가
  이번 Phase에서 `num_workers`를 비노출로 결정한 핵심 근거다
  (exact-resume 자체는 안전함에도 불구하고, §9).

## 3. 옵션 배치 -- `ImageFolderWorkflowRequest`, `TrainingConfig` 아님

Phase 4Q의 `device` 필드 전례를 그대로 따른다: `pin_memory`/
`non_blocking`은 학습 objective/hyperparameter가 아니라 순수 runtime
실행 파라미터이므로, `TrainingConfig`/`RESUME_CONFIG_FIELDS`와 무관하게
`ImageFolderWorkflowRequest`에 둔다.

```python
pin_memory: bool = False
non_blocking: bool = False
```

## 4. CPU/CUDA effective option contract

```python
is_cuda = _is_cuda_device(request.device)  # device == "cuda" or device.startswith("cuda:")
effective_pin_memory = request.pin_memory if is_cuda else False
effective_non_blocking = request.non_blocking if is_cuda else False
```

`device="cpu"`면 request 값과 무관하게 항상 `False`로 강제된다 -- CPU
DataLoader에 `pin_memory=True`를 그대로 넘기면 PyTorch가 "no accelerator
found" 경고를 내므로(실측 확인), 이 강제 처리로 그 경고 자체가 애초에
발생하지 않는다(테스트로 직접 확인:
`test_workflow_cpu_device_forces_effective_pin_memory_and_non_blocking_false`).

## 5. DataLoader / `.to()` wiring

- train loader: `pin_memory=effective_pin_memory`.
- val loader: train과 동일한 `effective_pin_memory` -- validation도
  training epoch 중 같은 `request.device`에서 평가되기 때문이다.
- test loader: Phase 4Q부터 항상 CPU 고정 평가이므로 이번 Phase의
  optimization을 적용하지 않는다(`pin_memory` 인자 자체를 넘기지 않고
  기본값 `False`를 유지).
- `train_one_epoch()`/`evaluate()`: `images.to(device,
  non_blocking=non_blocking)`/`labels.to(...)` -- 둘 다 `run_training()`
  이 자신의 `non_blocking` 인자를 그대로 넘겨 동일한 값을 받는다.
- `evaluate_classification_metrics()`: **무수정(Choice B)**. 이 함수는
  항상 CPU 최종 test 평가에만 쓰이고 Phase 4U optimization 값을 애초에
  전달받지 않으므로, API 일관성만을 위해 쓰이지 않을 인자를 추가하지
  않는다 -- "이 값들이 CPU 최종 test 경로까지 전달되지 않는다"가
  핵심이지 "모든 `.to()` 호출부가 같은 시그니처를 가져야 한다"가
  핵심이 아니다.
- `run_training()`: `non_blocking: bool = False` keyword-only 인자를
  추가하고, 매 epoch의 `train_one_epoch()`/`evaluate()` 호출 둘 다에
  그대로 전달한다.

## 6. exact-resume 실측(production 경로)

**production regression이 직접 증명하는 것**(사전 scratch 조사와는
구분): Phase 4R과 동일한 Conv2d+BatchNorm2d+Dropout fixture로
precision=`fp32`(기본값)를 쓰는 실제
`run_imagefolder_training_workflow()` + `checkpoint_out` 경로에서,
`test_workflow_cuda_pin_memory_non_blocking_resume_boundary_option_change_exact_resume`
가 다음을 실측 확인한다: continuous 5epoch와 split(3+2)의 첫 3epoch는
둘 다 `pin_memory=False`/`non_blocking=False`(Phase 4T까지의 baseline
경로)로 실행하고, **resume 2epoch만 `pin_memory=True`/
`non_blocking=True`로 전환**한다 -- 즉 checkpoint 저장 당시와 다른
옵션 값으로 resume해도, 다음이 전부 exact PASS(실제 GPU에서 검증됨):

```
history, model_state_dict(BatchNorm running_mean/running_var/
num_batches_tracked 포함), best_state_dict, optimizer_state_dict,
scheduler_state_dict, cuda_rng_state, loader_generator_state
```

이는 §8의 "resume 전후 값이 달라져도 exact-resume에 영향이 없다"는
공식 contract를 "같은 값을 유지하는 대표 케이스"보다 강하게, 실제
resume 경계에서 값이 바뀌는 시나리오로 production 경로에서 직접
고정한 것이다(사전 investigation 라운드의 scratch matrix 실험이
먼저 이 결론을 보였고, 이번 stabilization 라운드에서 production
regression을 그 결론에 맞춰 강화했다).

`scaler_state_dict`는 FP32 기본값이므로 두 checkpoint 모두 `None`이다
(`pin_memory`/`non_blocking`과 무관 -- 이 프로젝트는 이 조합이
precision과 독립적으로 안전함을 사전 조사에서 실측했지만, production
regression 자체는 fp32 대표 케이스만 고정한다는 점에 유의).

## 7. checkpoint 영향 -- `checkpoint.py`는 무수정

`pin_memory`/`non_blocking`은 loader generator를 전혀 소비하지 않는
순수 host-memory 전송 최적화이므로, 기존 checkpoint가 이미 저장하는
`loader_generator_state`만으로 exact-resume이 완전히 재현된다(§6). 새
checkpoint field나 `CHECKPOINT_FORMAT_VERSION` 변경이 전혀 필요하지
않았다(실제 구현 과정에서도 `checkpoint.py`를 수정할 필요가 있다는
새 증거는 발견되지 않았다).

## 8. `RESUME_CONFIG_FIELDS` 정책

`pin_memory`/`non_blocking` 둘 다 `RESUME_CONFIG_FIELDS`에 포함되지
않는다. "optimizer/scheduler 구조와 무관하다"는 이유만으로 결론 내리지
않고, **portable resume**과 **bitwise exact resume**을 별도로
검증했다: 두 값 모두 resume 전후로 값이 checkpoint 저장 당시와 달라져도
exact-resume에 영향이 없다 -- `device`처럼 완전히 자유로운 runtime
파라미터로 취급한다.

## 9. non-goals

```
num_workers 노출, persistent_workers, prefetch_factor, worker_init_fn,
worker RNG checkpoint, prefetch queue checkpoint, random augmentation,
별도 CUDA stream, automatic optimization selection,
performance benchmark framework, GUI, logging/observability 확장
```

`num_workers>0` 자체는 exact-resume이 안전함이 실측으로 확인됐지만
(`persistent_workers=False` 전제), 이 프로젝트의 전형적인 작은
데이터셋 규모 + Windows spawn 환경에서는 뚜렷한 성능 역효과가 있어
(§10) 이번 Phase에서 의도적으로 제외했다 -- 안전성과 유용성은 별개
질문이다.

## 10. 성능 caveat

Phase 4U가 보장하는 것은 "CUDA H2D transfer optimization을 선택적으로
활성화할 수 있다"이지 "학습이 빨라진다"가 아니다. 실제 성능 효과는
dataset 크기, storage/I/O, GPU, batch size 등에 따라 달라진다. 로컬
GTX 1080 + 이 프로젝트의 작은 CIFAR-10 ImageFolder 규모에서는
`pin_memory`/`non_blocking` 자체가 뚜렷한 speedup을 보여주지 않았다
(오히려 약간의 고정 오버헤드가 관찰됨) -- 더 크고 I/O가 느린
데이터셋에서는 결과가 다를 수 있으며, 이 caveat을 production 문서/CLI
help 어디에서도 성능 보장으로 표현하지 않는다.

## 11. CUDA stream 범위 -- 정확성 보장과 성능(overlap) 보장을 구분

이 프로젝트는 항상 default CUDA stream만 쓴다(별도 stream 없음). 이
project-local 실행 경로에서는 batch H2D copy와 이후 forward가 같은
normal PyTorch CUDA stream ordering 아래 실행되며, 정확성이 이 전제
위에서 실측으로 확인됐다 -- 동일 stream 내 커널 실행 순서 보장 덕분에,
pinned memory 비동기 복사 직후 곧바로 forward를 호출해도 결과가
안전함을 확인했다(blocking과 non_blocking(pinned)의 forward 결과가
bit-identical, 실제 local production/scratch test 양쪽에서 재현).

**이 정확성 보장은 H2D copy와 model kernel execution이 GPU 상에서
실제로 겹쳐(overlap) 실행된다는 성능 보장이 아니다.** PyTorch 공식
문서/tutorial은 대표적으로 pinned source memory + 별도의 non-default
CUDA stream + hardware capability를 함께 갖춰야 GPU-side overlap이
발생한다고 설명한다 -- 이 프로젝트는 별도 stream을 쓰지 않으므로,
`pin_memory=True`+`non_blocking=True`를 함께 켜더라도 그런 overlap을
보장하지 않는다. "별도 CUDA stream을 쓰면 반드시 틀린 결과가 난다"거나
"default stream이 아니면 non_blocking이 unsafe하다"는 절대적 주장은
하지 않는다 -- 다만 향후 별도 CUDA stream을 도입하면 explicit
synchronization/stream dependency 설계가 새로 필요해지므로, 그 시점에
정확성을 다시 검증해야 한다는 것이 이 project의 현재 contract다.

## 12. 두 옵션의 독립성과 실제 semantics

`pin_memory`/`non_blocking`은 서로 독립적으로 설정 가능하다 -- 한쪽만
켜는 조합을 거부하지 않는다(`if non_blocking and not pin_memory: raise
...` 같은 validation을 추가하지 않았다). 정확성 관점에서는 어떤 조합도
안전하다.

이전 초안은 "unpinned tensor에 `non_blocking=True`를 쓰면 PyTorch가
조용히 blocking으로 처리하므로, 실제 비동기 이득을 내려면
`pin_memory=True`가 사실상 전제"라고 서술했으나, 이는 지나친
단순화라 이번 stabilization 라운드에서 정정한다. 현재 PyTorch
문서/tutorial 기준 CPU→CUDA `.to()`는 내부적으로 asynchronous CUDA
copy를 쓰며, `non_blocking=False`는 host-side synchronization을
추가하고 `non_blocking=True`는 그 synchronization을 생략할 수 있다 --
즉 pageable(unpinned) source에서도 `non_blocking=True`가 완전히
무의미하거나 항상 host-blocking으로 강등된다고 일반화할 수 없다.
반면 GPU-side에서 H2D copy와 kernel execution이 실제로 겹쳐 실행되는
성능 이득에는 §11이 설명하는 더 강한 조건(별도 stream 등)이 필요하며,
이 프로젝트는 그 조건을 충족하지 않는다.

정정된 최종 contract:

```text
pin_memory와 non_blocking은 서로 독립적인 runtime optimization hint다.

non_blocking=True는 host-side synchronization을 줄일 수 있고,
pageable source에서도 의미가 있을 수 있으므로
pin_memory=False + non_blocking=True 조합을 허용한다.

pin_memory=True는 DataLoader가 반환하는 host tensor를 page-locked
memory에 배치해 CUDA H2D transfer를 더 효율적으로 만들 수 있다.

다만 이 프로젝트는 default CUDA stream만 사용하므로,
pin_memory=True + non_blocking=True라고 해서 H2D copy와
model kernel execution의 GPU-side overlap을 보장하지 않는다.

실제 성능 효과는 workload/hardware에 따라 달라지며 speedup을
보장하지 않는다(§10).
```

## 13. 테스트

**CPU-only(CI에서 항상 실행)**: request 기본값(`False`/`False`),
`_is_cuda_device()` predicate, CPU effective 강제(`False`/`False`로
DataLoader/`run_training()`에 전달되는지 + accelerator 경고 미발생),
train/val loader가 동일한 effective 값을 받는지, `train_one_epoch()`/
`evaluate()`의 `non_blocking` wiring(`torch.Tensor.to` spy),
`run_training()`의 배선 연결(criterion 배선 테스트와 동일한 근거 --
"자기 자신의 인자를 올바르게 처리하는지"와 "호출자가 실제로 그 값을
넘기는지"는 서로 다른 실패 지점), CLI `--pin-memory`/`--non-blocking`
4가지 조합.

**CUDA-only(로컬 GPU 필요, optional)**: `pin_memory=True`일 때 train/val
배치가 실제로 `is_pinned()==True`인지, `pin_memory`+`non_blocking` 조합
1 epoch 정상 완료, `pin_memory`+`non_blocking` exact-resume(§6), generic
`run_training()`의 `non_blocking` CUDA smoke.

기존 FP32/FP16/BF16 exact-resume regression과 5개 E2E numerical anchor는
전부 무수정으로 재확인했다(회귀 없음).
