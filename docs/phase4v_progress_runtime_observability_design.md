# Phase 4V: Progress / Runtime Observability — 설계안

## 1. 목적

향후 GUI를 포함한 caller가 training engine 내부 로직을 재현/재추론하지
않고도 학습 진행 상황과 종료 상태를 정확히 표현할 수 있도록, 이미 잘
만들어진 epoch-level `TrainingProgress`/`TrainingResult`에 관찰값 두
가지만 최소로 추가한다: `TrainingProgress.epoch_duration_seconds`와
`TrainingResult.stop_reason`(및 이를 그대로 forwarding하는
`ImageFolderWorkflowResult.stop_reason`). 사전 조사(chat 기록,
Phase 4V investigation round)에서 `best_epoch`/`best_val_loss`/
`epochs_without_improvement` 등 후보 목록의 상당수가 이미 구현돼
있음을 확인했으므로, 이번 Phase는 실제로 좁다.

## 2. 기존 `TrainingProgress` contract

```python
@dataclass(frozen=True)
class TrainingProgress:
    run_epoch: int
    total_run_epochs: int
    global_epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    best_epoch: int
    best_val_loss: float
    epochs_without_improvement: int
    stopped_early: bool
```

frozen dataclass, 완료된 epoch마다 정확히 1회 생성되는 독립 snapshot.
Phase 4V는 이 필드들의 의미를 하나도 바꾸지 않는다.

## 3. callback ordering(무변경)

```text
train_one_epoch() -> evaluate() -> history 기록 -> learning_rate 캡처
-> best 갱신 -> scheduler.step(val_loss) -> early stopping 판정
-> checkpoint_hook -> [epoch_duration_seconds 캡처, Phase 4V 신규]
-> progress_callback -> should_stop() 평가
-> (stopped_early 또는 stopped_by_user면 break)
```

`checkpoint_hook`이 `progress_callback`보다 항상 먼저, `progress_callback`
이 `should_stop()`보다 항상 먼저 실행되는 기존 Phase 4I/4J ordering을
그대로 유지한다. Phase 4V가 추가하는 유일한 삽입 지점은 `checkpoint_hook`
직후·`progress_callback` 직전의 `epoch_duration_seconds` 캡처뿐이다.

## 4. early stopping vs user stop ordering의 차이 (핵심 구조적 근거)

`should_stop()`은 **항상 그 epoch의 `progress_callback` 호출 이후에만**
평가된다(Phase 4I 기존 계약). 따라서:

- `stopped_early`는 `progress_callback` 호출 **이전에** 이미 결정되므로
  `TrainingProgress.stopped_early`에 정확히 반영될 수 있다.
- `stopped_by_user`는 `progress_callback` 호출 **이후에만** 결정되므로,
  **어떤 `TrainingProgress`도 이 사실을 담을 수 없다** -- 구조적으로
  불가능하다(이 ordering을 바꾸지 않는 한).

이것이 Phase 4V가 `stop_reason`/`stopped_by_user`를 `TrainingProgress`
가 아니라 `TrainingResult`(호출 종료 후에만 존재)에 두는 유일하고
근본적인 이유다.

## 5. `epoch_duration_seconds` 정의/경계

```python
epoch_started_at = time.perf_counter()   # for 루프 최상단, run_epoch 계산 직후
...                                        # train -> validation -> history/best
...                                        # -> scheduler.step -> early stopping
...                                        # -> checkpoint_hook
epoch_duration_seconds = time.perf_counter() - epoch_started_at
                                            # progress_callback 호출 직전
```

- **포함**: `train_one_epoch()`, `evaluate()`, history/best 갱신,
  `scheduler.step()`, early stopping 판정, `checkpoint_hook` 실행.
- **제외**: `progress_callback` 자신의 실행 시간, 그 이후의 `should_stop()`.
- `time.perf_counter()`(monotonic, high-resolution)만 사용 -- `time.time()`
  은 시스템 시각 조정에 취약해 duration 측정에 부적합하므로 쓰지 않는다.
- 새 timing helper/class는 만들지 않았다 -- 두 지점의 뺄셈으로 충분하다.

## 6. session-local semantics

`epoch_duration_seconds`는 **이번 `run_training()` 호출(session) 동안만
측정한 값**이다. resume해도 이전 호출에서 걸린 시간을 복원/합산하지
않는다 -- 매 호출은 자기 자신의 epoch들만 새로 측정한다. 누적
(cumulative, 여러 run 합산) elapsed time은 이번 Phase의 범위 밖이다
(checkpoint에 값을 저장해야 하므로 schema 변경이 필요해짐 -- §14 참고).

## 7. `stop_reason` 설계

```python
TrainingStopReason = Literal["completed", "early_stopped", "user_stopped"]
```

## 8. `Literal`을 선택한 이유

값이 정확히 3개뿐이고 runtime-only(직렬화/checkpoint 대상 아님)라 별도
`Enum` class를 두면 과설계다 -- 이 프로젝트가 이미 `TrainingConfig.
PRECISION_CHOICES` 등 "선택지가 적으면 tuple/Literal로 충분"이라는
관례를 갖고 있으므로 그 관례를 그대로 따랐다. exception/failure는 이
값에 포함하지 않는다 -- 예외가 나면 `TrainingResult` 자체가 반환되지
않으므로(`progress_callback`/`checkpoint_hook` 예외는 그대로 propagate하는
기존 정책) `"failed"`류 값이 필요 없다.

## 9. `TrainingResult`의 책임

```python
@dataclass
class TrainingResult:
    ...
    scaler_state_dict: dict | None = None
    stop_reason: TrainingStopReason = "completed"
```

루프 종료 직후 **정확히 한 번만** 계산한다(single source of truth):

```python
if history.stopped_early:
    stop_reason = "early_stopped"
elif history.stopped_by_user:
    stop_reason = "user_stopped"
else:
    stop_reason = "completed"
```

우선순위는 `early_stopped`를 먼저 확인한다 -- 현재 control flow에서
두 flag가 동시에 True가 되는 경우는 없지만(§4의 ordering상 `should_stop()`
이 `stopped_early=True`인 epoch에서는 애초에 평가되지 않음), 향후 변경에도
안전하도록 우선순위를 코드에 명시적으로 고정해 뒀다.

기본값 `"completed"`는 이 필드를 모르는 기존 manual `TrainingResult(...)`
생성 코드(예: `imagefolder_workflow.py`의 scheduled checkpoint 저장용
임시 결과 조립, 일부 테스트 fixture)와의 생성자 하위 호환을 위한 것이다.
`run_training()`이 실제로 반환하는 production 결과는 이 값을 항상
명시적으로 채운다.

## 10. `ImageFolderWorkflowResult.stop_reason` forwarding

`run_imagefolder_training_workflow()`가 이 프로젝트의 GUI-facing public
entrypoint이므로, `ImageFolderWorkflowResult`에도 `training_result.
stop_reason`을 그대로 forwarding하는 필드를 추가했다 -- caller(향후 GUI
포함)가 최종 종료 사유를 알기 위해 내부 `TrainingResult`까지 직접
들여다볼 필요가 없다. **재계산하지 않는다** -- `history.stopped_early`/
`stopped_by_user`로부터 다시 유도하지 않고 `run_training()`이 이미
계산한 값을 그대로 옮긴다(single source of truth 유지). 기본값은
`test_metrics`와 동일한 이유(기존 manual/fake constructor 호환)로
`"completed"`.

## 11. resume epoch semantics(무변경, 재확인)

```text
fresh 3 epochs:   global_epoch=[1,2,3]  run_epoch=[1,2,3]  total_run_epochs=[3,3,3]
resume 2 epochs:  global_epoch=[4,5]    run_epoch=[1,2]    total_run_epochs=[2,2]
```

`global_epoch`은 절대(resume 포함 전체 이력 기준), `run_epoch`/
`total_run_epochs`는 이번 호출-local이다. Phase 4V는 이 계약을 전혀
바꾸지 않았다 -- production regression
(`test_resume_progress_epoch_semantics_unchanged_by_phase_4v`)으로
명문화만 했다.

## 12. `learning_rate` pre-scheduler semantics(무변경, 재확인)

`progress.learning_rate`는 그 epoch의 `train_one_epoch()`가 실제로 쓴
값(= `scheduler.step()` 호출 **이전**에 캡처)이다. 기존 테스트
(`test_run_training_progress_callback_learning_rate_captured_before_scheduler_step`)
가 이미 이 계약을 고정하고 있으며, Phase 4V는 `scheduler.step()` 호출
위치를 전혀 건드리지 않았다.

## 13. callback synchronous/snapshot contract(무변경, 재확인)

`TrainingProgressCallback = Callable[[TrainingProgress], None]` 그대로다.
동기(synchronous) 호출, 반환값 무시, 매 완료 epoch 1회, 예외는 그대로
propagate(`TrainingResult` 미반환), `frozen=True`라 매 epoch 새 인스턴스가
이후에도 절대 변형되지 않는다. Phase 4V는 새 callback type이나 event
union을 추가하지 않았다.

## 14. checkpoint schema 영향 -- 없음

`checkpoint.py`/`CHECKPOINT_FORMAT_VERSION`/`RESUME_CONFIG_FIELDS` 전부
무수정이다. `epoch_duration_seconds`는 `TrainingProgress`(어디에도
직렬화되지 않는 순수 runtime 객체)에만 존재한다. `stop_reason`은
`TrainingResult`에 존재하지만, `save_training_checkpoint()`가
`training_result`를 통째로 `asdict()`하지 않고 필요한 필드만 개별적으로
골라 payload dict에 담는 기존 구조이므로(`checkpoint.py:172-186` 근처),
새 필드를 추가해도 명시적으로 그 payload 구성에 추가하지 않는 한
checkpoint에 전혀 반영되지 않는다 -- 이번 구현은 의도적으로 추가하지
않았다.

## 15. artifact JSON schema 영향 -- 없음

`TrainingHistory`(→ `training_history.json` 및 checkpoint payload의
`"history"` 서브딕트 양쪽)에는 아무 필드도 추가하지 않았다. Phase 4V가
추가한 두 값은 `TrainingHistory`가 아니라 `TrainingProgress`/
`TrainingResult`에만 존재하므로 JSON schema에 영향이 없다. production
regression(`test_phase_4v_observability_fields_do_not_leak_into_artifact_schema`)
으로 `training_history.json`과 checkpoint payload 양쪽의 key 집합이
Phase 4U와 완전히 동일함을 직접 확인했다.

## 16. exact-resume 영향 -- 없음

`time.perf_counter()`는 어떤 RNG state도 소비하지 않는다. `stop_reason`
계산도 이미 존재하는 `history.stopped_early`/`stopped_by_user`를 읽기만
할 뿐 새로운 계산 경로를 만들지 않는다. 실제 GPU에서 기존 4종 exact-resume
regression(FP32/Phase 4R, FP16/Phase 4S, BF16/Phase 4T, H2D option-change/
Phase 4U)을 전부 재실행해 회귀 없음을 확인했다.

## 17. GUI worker-thread/message-queue 예상 사용 방식

```text
GUI
  ↓
worker thread
  ↓
run_imagefolder_training_workflow(
    progress_callback=lambda p: queue.put(("progress", p)),
    should_stop=stop_event.is_set,
  )
  ↓ (동기 callback, worker thread 안에서 실행됨 -- GUI thread 직접 호출 금지)
thread-safe queue로 즉시 전달
  ↓
GUI thread가 queue를 polling/소비
  ↓
progress bar(run_epoch/total_run_epochs 조합 -- 이번 run_training()
호출 기준 진행률, §11 참고), Train/Val Loss, Val Accuracy, Learning
Rate, Best Epoch/Val Loss, Epoch Duration 표시
  ↓
worker thread 함수 반환(ImageFolderWorkflowResult)
  ↓
result.stop_reason으로 "Completed"/"Early stopped"/"Stopped by user" 표시
```

`progress_callback`은 완전히 동기이므로, GUI가 그 안에서 무거운 렌더링을
직접 하면 학습 루프 자체가 지연된다 -- callback 안에서는 queue에 값만
넣고 즉시 반환하는 패턴을 권장한다(이번 Phase는 이 인프라를 구현하지
않는다, 설계 관점 권장 사항일 뿐).

progress bar에 `run_epoch/total_run_epochs`를 쓰는 이유(§11과 동일한
resume semantics): `global_epoch`은 resume 포함 절대 epoch 번호라
progress bar 분모로 그대로 쓰면 resume 시 잘못된 비율이 된다(예:
`global_epoch=4`인데 이번 호출은 2 epoch만 요청했다면 `4/2`는 의미가
없다). `run_epoch/total_run_epochs`가 "이번 호출의 진행률"을 정확히
나타낸다. 전체 absolute target을 표시하고 싶다면 GUI가 직접 유도할 수
있다:

```text
completed_before_run = global_epoch - run_epoch
absolute_target = completed_before_run + total_run_epochs
```

예: resume 후 `global_epoch=4, run_epoch=1, total_run_epochs=2`이면
이번 호출 진행률은 `1/2`이고, `completed_before_run=3`이므로
`absolute_target=5` -- 필요하면 `4/5`로도 표시할 수 있다. 이 유도값을
위한 별도 필드는 추가하지 않는다(§18 non-goals).

## 18. non-goals

```text
batch-level progress, per-batch loss streaming,
stage/phase event(TrainingStarted/EpochStarted 등), completion event,
device/precision을 TrainingProgress에 포함,
cumulative elapsed(resume 합산), ETA 계산,
TensorBoard, Weights & Biases, GUI 구현,
thread/threadpool 생성, asyncio, logging framework, profiling,
GPU utilization/VRAM/system 모니터링, experiment run 관리,
checkpoint schema 변경, training_history.json schema 변경,
CLI presentation 확장, next_learning_rate 필드,
TrainingStopReason을 Enum으로 재설계
```

## 19. Phase 4W와의 경계

Phase 4V는 runtime progress/result observability contract 완성까지다.
Phase 4A~4V 전체(CPU FP32/CUDA FP32/FP16/BF16, fresh/resume/user stop,
checkpoint, best model, export, parity 등)의 production
integration/graduation 검증은 Phase 4W의 몫이며, 이번 Phase에서
그 통합 테스트 범위를 끌어오지 않았다.
