# Phase 4I: Training Progress Callback and Safe Stop — 설계안

**상태: 구현 및 검증 완료.** core training API(`run_training()`)에 epoch
단위 진행 상황 callback과 epoch 경계 cooperative stop을 추가하기 위한
상세 설계와, 그 설계를 그대로 구현한 결과를 담는다.

**구현 결과 요약** (아래 §14의 테스트 계획을 실제로 실행한 뒤 기록):

* `src/image_ai_studio/training/loop.py` — `TrainingProgress`(frozen
  dataclass) + `TrainingProgressCallback`/`ShouldStopCallback` 타입 별칭
  신설, `TrainingHistory.stopped_by_user: bool = False` 필드 추가,
  `run_training()`에 키워드 전용 `progress_callback`/`should_stop`
  파라미터 추가. `TrainingResult`에는 필드를 추가하지 않았다(설계 그대로).
* `src/image_ai_studio/training/imagefolder_workflow.py` —
  `run_imagefolder_training_workflow()`에 동일한 키워드 전용 파라미터를
  추가해 `run_training()` 호출에 그대로 전달(다른 로직 변경 없음).
  `training/checkpoint.py`/`training/history.py`는 설계대로 **전혀
  수정하지 않았다** — 둘 다 `asdict()`/`**dict` 기반 범용 코드라 새
  필드가 기본값으로 자동 처리된다.
* `scripts/train_imagefolder.py` — 사후 일괄 출력 루프를
  `progress_callback`(`_print_progress`, `progress.global_epoch` 사용)으로
  교체, 요약에 `stopped_by_user` 추가. `scripts/run_imagefolder_training_e2e.py`는
  설계대로 수정하지 않았다.
* 테스트: `tests/training/test_loop.py`에 Phase 4I 테스트 함수 14개
  추가(52 passed, 기존 38개 포함),
  `tests/training/test_imagefolder_workflow.py`에 4개 추가(13 passed),
  `tests/scripts/test_train_imagefolder_cli.py`에 2개 추가(8 passed,
  §11에서 정한 6단계 resume 출력 절차 그대로 구현),
  `tests/training/test_history.py`에 하위 호환 테스트 2개 추가 및 기존
  키 목록 검증 테스트 1개 갱신(`stopped_by_user` 키 반영),
  `tests/training/test_checkpoint.py`에
  `test_load_training_checkpoint_accepts_legacy_history_without_stopped_by_user`
  1개 추가(history payload에 `stopped_by_user` 키가 없는 Phase 4H까지의
  checkpoint 형식도 `load_training_checkpoint()`의 구조 검증을 통과하고,
  `TrainingHistory(**payload["history"])` 복원 시 `stopped_by_user=False`
  기본값이 적용되는지 확인 — `_REQUIRED_HISTORY_FIELDS`는 수정하지
  않았다). **최종 결과: `tests/training/` + `tests/scripts/` 205
  passed, 전체 `pytest` 362 passed, 0 failed.** 회귀 검증
  E2E(`run_training_e2e.py`/`run_real_training_e2e.py`/
  `run_resume_training_e2e.py`/`run_imagefolder_training_e2e.py`, 후자는
  TorchScript export + C++ CPU/CUDA parity 포함)도 전부 재실행해 전부
  PASS했고, `progress_callback=None`/`should_stop=None` 기본 경로가
  Phase 4H까지의 동작과 동일함을 확인했다.
* 설계와 다르게 구현한 부분: 없음(구현 중 실제 코드 제약과의 충돌도
  발견되지 않았다).

**파일명 정정**: 사용자가 조사 대상으로 제시한
`tests/training/test_imagefolder_checkpoint_resume.py`는 실제 저장소에
존재하지 않는다 — 해당 역할을 하는 실제 파일은
`tests/training/test_imagefolder_resume.py`(Phase 4G에서 신설)다. 이
문서 전체에서 후자를 기준으로 서술한다.

---

## 1. Phase 4I 구현 전 구조 분석

> **이 절 전체(§1-1~§1-3)는 Phase 4I 설계 당시 분석한 구현 전 기준
> 코드다.** 현재 구현에서는 early stopping의 즉시 `break`가
> `progress_callback` 호출 이후의 공통 종료 판정으로 이동했으며,
> `train_imagefolder.py`의 사후 출력 루프도 실시간 콜백으로 교체됐다
> — 최종 구조는 §7/§11과 실제 `src/image_ai_studio/training/loop.py`/
> `scripts/train_imagefolder.py`에 반영돼 있다. 구현 진행 중 코드가
> 재구성되며 원래 인용했던 줄 번호(`loop.py:320-342` 등)가 더 이상
> 그 시점의 코드를 가리키지 않게 됐으므로, 이 절에서는 줄 번호를 모두
> 제거하고 함수명/처리 순서로만 서술한다 — 아래 내용을 현재 코드의
> 위치로 착각하지 말 것.

### 1-1. 구현 전 run_training()의 epoch 처리 순서

```python
for epoch in range(completed_epochs + 1, completed_epochs + config.epochs + 1):
    history.train_losses.append(train_one_epoch(model, train_loader, optimizer, device=device))  # 1
    val_loss, val_accuracy = evaluate(model, val_loader, device=device)                            # 2
    history.val_losses.append(val_loss)                                                            # 3
    history.val_accuracies.append(val_accuracy)                                                    # 3

    if history.best_val_loss is None or val_loss < history.best_val_loss:                          # 4
        history.best_epoch = epoch
        history.best_val_loss = val_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if scheduler is not None:
        scheduler.step(val_loss)                                                                    # 5

    if (config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience):                      # 6
        history.stopped_early = True
        break
```

**위 코드는 Phase 4I 설계 당시 분석한 구현 전 기준 코드다.** early
stopping 조건이 만족되면 그 자리에서 곧바로 `break`했다. 최종 구현에서는
이 `break`가 제거되고, `progress_callback` 호출과 `should_stop` 판정을
거친 뒤의 공통 종료 판정으로 이동했다 — 최종 순서는 §7의 pseudocode와
실제 `loop.py`의 `run_training()`을 참고할 것.

**중요한 정정(구현 전 코드 기준)**: 사용자가 §4에서 제시한 예상 순서
("train → validate → scheduler 반영 → history 추가 → best state 갱신
→ …")는 이 구현 전 코드와 다르다. 실제 순서는 **train → validate →
history 추가(val) → best state/카운터 갱신 → `scheduler.step()` →
early stopping 판정**이다(`optimizer.step()`은 `train_one_epoch()`
내부, 매 batch. `scheduler.step()`은 epoch당 1번, **best_state_dict
계산 이후**). 이 순서는 구현 전 `run_training()` 자체의 docstring에도
이미 명시돼 있었다:

> "매 epoch의 순서는 train -> validation -> history 기록 -> best
> model/개선 카운터 갱신 -> scheduler.step(val_loss) -> early stopping
> 조건 확인이다."

Phase 4I의 progress callback/stop 확인 지점은 이 **기존에 이미 문서화된
순서의 맨 끝**(early stopping 판정 직후, `break` 이전)에 끼워 넣는 것이
가장 자연스럽다 — 새 순서를 발명하지 않고 기존 문서화된 계약을 그대로
연장한다. (이 방향은 최종 구현에서 그대로 채택됐다 — §7 참고.)

### 1-2. 관련 사실 확인 (구현 전 기준, 현재와의 차이는 각 항목에 표시)

- **`optimizer.step()` 위치**: `train_one_epoch()` 내부 batch 루프 안,
  `loss.backward()` 직후. epoch당 여러 번(batch 수만큼) 호출됨 — Phase
  4I는 batch 단위 개입을 하지 않으므로 이 함수는 전혀 건드리지 않았다
  (현재도 동일).
- **`scheduler.step()` 위치**: best_state_dict 갱신 **이후**, early
  stopping 판정 **이전**. 즉 scheduler가 이번 epoch에 LR을 낮췄다면,
  그 낮아진 LR은 **다음 epoch**부터 적용된다 — 이번 epoch의
  `train_one_epoch()` 호출에는 전혀 영향을 주지 않았다(이미 끝난
  뒤이므로). (현재도 이 상대적 순서는 동일 — §7)
- **`best_state_dict` 갱신 위치**: validation 직후, `scheduler.step()`
  **이전**. `val_loss < best_val_loss`일 때만
  `copy.deepcopy(model.state_dict())`. (현재도 동일)
- **`epochs_without_improvement` 갱신 위치**: `best_state_dict` 갱신과
  같은 `if/else` 블록 — 개선 시 0, 아니면 +1. (현재도 동일)
- **early stopping 판정 위치**: `scheduler.step()` **이후**, loop 맨
  끝에서 조건이 만족되면 `history.stopped_early = True`를 설정하고
  그 자리에서 즉시 `break`했다. **현재 구현에서는 조건 판정과
  `history.stopped_early = True` 설정 자체는 그대로지만, `break`는
  제거되고 `progress_callback` 호출 이후의 공통 종료 판정으로
  옮겨졌다(§7).**
- **`TrainingHistory`에 epoch 결과가 추가되는 시점**: train_loss는
  `train_one_epoch()` 반환 즉시, val_loss/val_accuracy는 `evaluate()`
  반환 즉시 — 둘 다 best_state_dict 갱신보다 먼저다. (현재도 동일)
- **checkpoint에 저장되는 current model/RNG/generator 상태**
  (`save_training_checkpoint()`): `model_state_dict =
  model.state_dict()`(호출 시점의 현재 가중치, best 아님),
  `loader_generator_state`/`cpu_rng_state`는 호출자가 직접 채취해
  넘긴다(`run_training()` 자체는 RNG를 캡처하지 않음 — 캡처는 항상
  호출자 책임, `run_imagefolder_training_workflow()`가 실제 예시).
  (`checkpoint.py`는 이번 Phase에서 수정하지 않았으므로 현재도 동일)
- **resume 시 기존 history와 추가 epoch 번호 연결**: resume 분기와
  epoch loop의 `range(completed_epochs + 1, completed_epochs +
  config.epochs + 1)` — `completed_epochs = len(history.train_losses)`
  가 유일한 출처(`TrainingResumeState` docstring에서 이미 확립된 설계
  원칙). epoch 번호는 항상 **절대(전체 누적) 번호**이고,
  `config.epochs`는 resume 여부와 무관하게 항상 "이번 호출에서
  추가로 실행할 epoch 수"다. (현재도 동일)

### 1-3. 구현 전 imagefolder_workflow.py/CLI/E2E 상태 (Phase 4H 기준)

- `run_imagefolder_training_workflow()`는 `_prepare_resume()`로
  model/generator/resume_state/cpu_rng_state를 얻은 뒤, DataLoader를
  전부 만들고, `cpu_rng_state`가 있으면 `torch.set_rng_state()`,
  **바로 다음 줄에서** `run_training(...)` 호출했다. 이 두 줄 사이에는
  아무 코드도 없었다 — Phase 4I도 이 간격을 벌리지 않는다(§12). **현재
  구현에서도 이 두 줄은 여전히 인접해 있다 — `progress_callback`/
  `should_stop`은 그 `run_training(...)` 호출의 인자로만 추가됐다.**
- `train_imagefolder.py`(production CLI, 구현 전)는
  `run_imagefolder_training_workflow()`가 반환한 뒤 `result.history`의
  리스트들을 **사후에** `zip()`으로 순회하며 epoch별 줄을 출력했다 —
  즉 "실시간 로그"가 아니라 학습이 전부 끝난 뒤 재구성한 요약이었다.
  `stopped_early`만 출력하고 사용자 중단 개념은 아직 없었다. **현재
  구현에서는 이 사후 순회 루프가 `progress_callback` 기반 실시간
  출력(`_print_progress`)으로 교체됐다(§11).**
- `run_imagefolder_training_e2e.py`의 `_run_workflow_stage()`도 동일한
  사후 순회 패턴이었다. **이 스크립트는 이번 Phase에서 수정하지
  않았으므로 현재도 그대로 사후 순회 패턴이다.**
- 둘 다 `run_imagefolder_training_workflow()`를 **위치 인자 하나
  (`request`)만으로** 호출했다 — 키워드 전용 인자를 추가할 여지가
  API상 이미 열려 있었다. **현재는 실제로 `progress_callback`/
  `should_stop` 두 키워드 전용 인자가 추가됐다(§12).**

---

## 2. Phase 4I 목표와 비목표

### 목표

1. `run_training()`에 epoch 완료 시점 progress callback 추가.
2. `run_training()`에 epoch 경계 cooperative stop 추가.
3. `imagefolder_workflow.py`가 이 두 기능을 keyword-only 인자로 그대로
   전달.
4. `train_imagefolder.py`가 progress callback으로 기존 사후 출력을
   실시간 출력으로 전환 — fresh 학습의 출력 형식은 유지하되, resume의
   출력은 "누적 history를 사후에 다시 전부 보여주는" 기존 방식에서
   "이번 호출에서 새로 완료된 epoch만 실시간으로 보여주는" 방식으로
   의도적으로 바꾼다(§11).
5. 기존 checkpoint/JSON history와의 하위 호환성 유지.

### 비목표 (§13 사용자 목록 그대로 확인)

GUI 구현, Qt/PySide/Tkinter 선택, background worker/thread 구현, batch
중간 resume, batch 단위 progress callback, ETA/남은 시간 예측,
TensorBoard/W&B 연동, epoch별 자동 checkpoint, latest/best checkpoint
rotation, SIGINT/KeyboardInterrupt graceful resume, CUDA 학습, AMP,
distributed training, multi-worker DataLoader resume, run directory
자동 생성, run_config.json. 이 항목들은 모두 "향후 확장" 절(§19)에서
Phase 4I API와 어떻게 연결될 수 있는지만 짧게 언급한다.

---

## 3. `TrainingProgress` dataclass

### 최종 필드

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

**(리뷰 반영, §4/§7과 함께 수정됨) `stop_requested` 필드를 제거했다.**
초안에는 있었지만, §4/§7에서 callback 호출 시점을 `should_stop()` 평가
**이전**으로 옮기면서(사용자가 콜백 안에서 stop 플래그를 세팅하고 그걸
같은 epoch 경계에서 즉시 반영하기 위함) 콜백이 호출되는 시점에는 아직
`should_stop()`을 평가하지 않은 상태가 된다 — 즉 콜백에 넘길 정확한
`stop_requested` 값 자체가 존재하지 않는다. 이 필드를 억지로 채우려면
콜백 전에 `should_stop()`을 먼저 평가해야 하는데, 그러면 정확히
사용자가 지적한 순서 문제(콜백이 플래그를 세팅해도 이미 평가가 끝난
뒤라 반영이 한 epoch 늦어짐)가 되돌아온다. 그래서 "진행 중 신호"라는
개념 자체를 이번 Phase에서는 없애고, 사용자 중단 여부는 **학습이 끝난
뒤의 `TrainingHistory.stopped_by_user`로만** 확인하는 것으로 계약을
단순화했다(§9는 그대로 유지).

`frozen=True`로 둔다 — `TrainingHistory`/`TrainingResult`는 현재
`frozen`이 아니지만(`loop.py:92`, `:113`), 그 둘은 `run_training()`
**내부에서 계속 변형되는 작업용 상태**인 반면 `TrainingProgress`는
**외부(callback)로 한 번 건네지고 끝나는 알림 객체**라 성격이 다르다.
frozen으로 두면 callback 구현자가 실수로 이 스냅샷을 수정해 학습
내부 상태와 착각하는 것을 원천 차단한다 — 새로 도입하는 클래스이므로
기존 스타일을 깨는 게 아니라 이 클래스에만 맞는 계약을 추가하는 것이다.

### epoch 번호 필드 이름 비교와 최종 선택

| 후보 | 의미 | 채택 여부 |
|---|---|---|
| `run_epoch` vs `current_run_epoch` | 이번 `run_training()` 호출 안에서 몇 번째 epoch인지(1부터) | **`run_epoch` 채택** — `TrainingProgress`는 애초에 "지금 이 순간"의 스냅샷이므로 `current_` 접두사가 중복 정보다 |
| `completed_epochs` vs `global_epoch` | 이 학습의 전체 이력(모든 resume 포함)에서 절대 epoch 번호 | **`global_epoch` 채택** — `completed_epochs`라는 이름은 "완료된 개수(cardinal)"로도, "방금 완료된 epoch의 절대 번호(ordinal)"로도 읽힐 수 있어 모호하다. `global_epoch`는 "이 값이 하나의 epoch를 가리키는 절대 라벨"이라는 뜻을 이름만으로 더 명확히 전달한다. 또한 `loop.py`의 실제 loop 변수 이름이 이미 `epoch`이므로(`:320`), `global_epoch`가 그 값의 정체를 "resume을 포함한 전역 번호"라고 더 정확히 설명한다 |
| `total_run_epochs` vs `requested_epochs` | 이번 호출에서 실행 예정인 전체 epoch 수(= `config.epochs`) | **`total_run_epochs` 채택** — `run_epoch`와 이름 계열을 맞춰 "run_epoch / total_run_epochs"가 "이번 실행의 X번째 / 총 Y개"로 자연스럽게 읽힌다. `requested_epochs`는 틀린 이름은 아니지만 `run_epoch`와 어근이 안 맞아 나란히 봤을 때 같은 개념 축임이 덜 드러난다 |

fresh 5 epoch: `run_epoch`=1..5, `total_run_epochs`=5, `global_epoch`=1..5
(전부 동일). 2 epoch 완료 checkpoint에서 3 epoch resume:
`run_epoch`=1..3, `total_run_epochs`=3, `global_epoch`=3..5.

### 넣지 않기로 한 필드와 근거

- **optimizer/scheduler 이름, device**: 전부 `TrainingConfig`(그리고
  `device` 문자열)에 이미 있고, 호출자가 `run_training()`에 넘긴 바로
  그 값이다 — 매 epoch마다 똑같은 정적 값을 반복해서 넣는 건 중복이다.
- **model/state_dict**: 사용자 권장안 그대로 제외. Phase 4H가
  `ImageFolderWorkflowResult`를 설계할 때 이미 "경로/지표만 반환, 살아있는
  객체는 담지 않는다"는 원칙을 세웠고(`docs/phase4h_production_training_cli_design.md`
  §6), `TrainingProgress`도 같은 원칙을 따른다. **model 접근 가능
  여부는 호출 계층에 따라 다르다는 점을 정확히 구분한다(리뷰 반영)**:
  - `run_training()`을 **직접** 호출하는 코드(예: `test_loop.py`의
    테스트, 또는 향후 다른 core-level 호출자)는 자신이 `run_training()`
    에 넘긴 바로 그 `model` 객체(`nn.Module`은 참조로 전달됨)를 계속
    들고 있으므로, `progress_callback` 밖에서도(또는 그 안에서도)
    언제든 `model.state_dict()`를 직접 들여다볼 수 있다.
  - 반면 `run_imagefolder_training_workflow()`를 호출하는 코드
    (`train_imagefolder.py`/`run_imagefolder_training_e2e.py`)는 **이
    경로로는 model에 전혀 접근할 수 없다** — `model`은
    `run_imagefolder_training_workflow()` 내부(`_prepare_resume()`이
    만들고 그 함수의 지역 변수로만 존재)에서 생성되므로, 이 workflow에
    넘기는 `progress_callback`은 `TrainingProgress`(지표만)만 받고
    `model` 참조는 받지 못한다.
  - 즉 `TrainingProgress`는 **두 계층 모두에서** "지표 관찰/UI 갱신
    전용" 객체이고, `run_training()` 직접 호출자만 별도 경로(자신이
    보유한 `model` 변수)로 추가 정보를 얻을 수 있을 뿐이다 —
    `TrainingProgress` 자체에 model/state_dict를 담지 않는다는 결정은
    두 계층 모두에서 동일하게 유지된다.
- **elapsed time / epoch duration / 전체 예상 시간(ETA)**: 이번 Phase
  범위에서 전부 제외한다. 나중에 필요해지면 `TrainingProgress`에 기본값
  있는 필드를 추가하는 것만으로 하위 호환을 깨지 않고 확장 가능하다
  (dataclass에 기본값 있는 새 필드를 추가하는 건 항상 안전한 변경) —
  지금 구체적인 요구가 없는 채로 타이머 계측 코드를 넣는 건 과설계다.
- **`stop_requested`(실시간 정지 감지 신호)**: 넣지 않는다(리뷰 반영,
  최초 설계에서는 포함했다가 제거함). §4/§7에서 확정한 새 순서상
  `progress_callback`은 `should_stop()`을 평가하기 **전**에 호출되므로,
  콜백 시점에는 "이번 epoch에 정지 요청이 감지됐는가"라는 값이 아직
  존재하지 않는다 — 존재하지 않는 값을 필드로 만들 수 없다. 사용자
  중단 여부는 `TrainingProgress`가 아니라 학습이 끝난 뒤
  `TrainingResult.history.stopped_by_user`로만 확인한다(§9).

---

## 4. Callback 타입과 호출 시점

```python
TrainingProgressCallback = Callable[[TrainingProgress], None]
```

Protocol이 아니라 plain `Callable` — §7에서 `ShouldStopCallback`과
같은 근거로 다룬다(과설계 방지, 프로젝트 어디에도 이런 자리에 Protocol을
쓴 선례가 없음).

### 호출 시점 (리뷰 반영으로 최종 확정, §7과 함께 갱신)

```text
train_one_epoch 완료 (이미 history.train_losses에 반영됨)
→ validation 완료 (이미 history.val_losses/val_accuracies에 반영됨)
→ best_state_dict / epochs_without_improvement 갱신
→ scheduler.step(val_loss)  [scheduler 있으면]
→ early stopping 판정 (history.stopped_early 설정 포함, 조건 만족 시)
→ TrainingProgress 조립
→ progress_callback 호출
→ (early stopping이 이번 epoch에 발동하지 않았고, 다음에 실행할
   epoch가 남아 있을 때만) should_stop() 평가
→ (stopped_early 또는 방금 평가된 stop 요청이면 break, 아니면 다음 epoch)
```

**최초 설계에서 순서를 바꾼 이유(리뷰 반영)**: 처음에는 `should_stop()`
평가를 콜백보다 먼저 뒀는데, 이러면 "콜백 안에서 사용자가 stop
플래그를 세팅한다"는 가장 자연스러운 UI 사용 패턴이 **그 epoch에서는
전혀 반영되지 못하고 다음 epoch가 최소 1번 더 실행된 뒤에야** 반영되는
문제가 있었다(콜백이 플래그를 세팅하는 시점에는 이미 이번 epoch의
`should_stop()` 평가가 끝난 뒤이므로). **콜백을 `should_stop()` 평가보다
먼저 호출**하도록 바꾸면, 콜백이 동기적으로 실행되는 동안 세팅한
플래그를 **바로 다음 줄의 `should_stop()` 평가가 즉시 볼 수 있어**
같은 epoch 경계에서 다음 epoch 진입을 막을 수 있다(§7에서 상세히
설명).

이 순서 변경의 직접적 결과로, 콜백 호출 시점에는 아직 `should_stop()`을
평가하지 않았으므로 "이번 epoch에 정지 요청이 감지됐는가"라는 값을
`TrainingProgress`에 담을 수 없다 — 그래서 `stop_requested` 필드를
제거했다(§3).

### `learning_rate` 필드의 정확한 의미

`optimizer.param_groups[0]["lr"]`을 **`scheduler.step()` 호출 전에**
읽어서 담는다 — 이 프로젝트의 optimizer 생성(`_build_optimizer()`,
`loop.py:17-22`)이 항상 단일 param group이므로(`Adam(model.parameters(),
lr=...)` / `SGD(model.parameters(), lr=..., momentum=...)`)
`param_groups[0]`만으로 충분하다.

`learning_rate_used` vs `learning_rate_next` 비교: **하나만 두고
`learning_rate`로 명명, "이번 epoch의 `train_one_epoch()`가 실제로 쓴
값"을 의미하도록 확정**한다. 근거:
- `learning_rate_next`(scheduler.step() 이후 값)는 "다음 epoch에 어떤
  일이 있을지"를 예고하는 예측성 정보다 — `TrainingProgress`는 "방금
  끝난 epoch의 결과를 보고하는" 성격(사용자 권장 방향, §3)이므로 예측
  값을 넣는 건 그 성격과 어긋난다.
- 필드 하나로 충분한 이유: `learning_rate_next`가 정말 필요해지는
  경우는 "다음 epoch에 LR이 바뀔지 미리 보여주고 싶다"는 UI 요구뿐인데,
  이는 현재 어떤 사용처도 요구하지 않는다(§14 anti-overengineering 원칙:
  "미래 확장만을 위한 필드 금지"). 필요해지면 나중에 필드를 추가해도
  하위 호환이 깨지지 않는다(§3의 동일 논리).
- docstring에 "이 값은 이번 epoch의 학습에 실제로 쓰인 LR이며,
  scheduler가 이번 epoch에 LR을 바꿨다면 그 바뀐 값은 **다음** 콜백에서
  보임" 문구를 명시해 모호성을 코드로 남기지 않는다.

---

## 5. Callback 예외 정책

**정책 A(전파, 사용자 권장과 동일) 채택.** `progress_callback`이 예외를
던지면 `run_training()`은 그 예외를 그대로 전파한다 — try/except로
감싸지 않는다.

### 예외 발생 시점의 정확한 상태 (문서화 필수 항목)

콜백은 §4의 순서상 "이번 epoch에 대해 결정할 수 있는 모든 것이 이미
결정된 뒤" 호출되므로, 콜백이 예외를 던진 시점에는:

- `history.train_losses`/`val_losses`/`val_accuracies`: 이번 epoch까지
  **전부 반영된 상태**.
- `best_state_dict`/`history.best_epoch`/`history.best_val_loss`/
  `epochs_without_improvement`: 이번 epoch 기준으로 **전부 최신 상태**.
- `optimizer`/`scheduler` 내부 상태(`optimizer.state_dict()`로 꺼낼 수
  있는 것): 이번 epoch까지 반영된 상태(`scheduler.step()`도 이미 실행됨).
- `model`(호출자가 넘긴 바로 그 객체, 참조로 전달됨): 이번 epoch
  학습이 끝난 시점의 가중치 — **호출자가 원래 갖고 있던 `model` 변수를
  통해 예외 발생 후에도 그대로 접근 가능**(같은 객체이므로). 여기서
  "호출자"는 `run_training()`을 **직접** 부른 코드를 뜻한다 —
  `run_imagefolder_training_workflow()` 경로에서는 이 "호출자"가
  workflow 내부 코드 자신이지 그 workflow를 부른 CLI/E2E가 아니므로,
  workflow의 `progress_callback`에는 이 접근성이 그대로 이어지지
  않는다(§3에서 계층별로 정확히 구분).
- **하지만 `run_training()`은 `TrainingResult`를 반환하지 않고 예외로
  끝난다** — 즉 `optimizer_state_dict`/`scheduler_state_dict`/
  `best_state_dict`/`history`를 하나의 `TrainingResult`로 묶어서 받을
  방법이 없다. `optimizer`/`scheduler`는 `run_training()` 지역 변수라
  호출자가 아예 참조를 가진 적이 없다(`model`만 예외).
- **실무적 함의**: callback이 예외로 학습을 실패시키는 것은 "복구
  불가능한 오류"를 위한 것이어야 한다. callback이 어떤 조건에서
  의도적으로 학습을 멈추고 싶다면(예: UI가 이 예외를 이용해 흐름
  제어를 하고 싶다면) **예외가 아니라 `should_stop`을 쓸 것을
  권장한다.** 단, **`ShouldStopCallback` 자체가 checkpoint를 저장하는
  것은 아니다** — `should_stop`을 쓰면 `run_training()`이 예외 없이
  정상적인 `TrainingResult`를 반환하고, 그 결과로 상위 workflow(예:
  `run_imagefolder_training_workflow()`)가 `checkpoint_out` 등 기존
  artifact 후처리를 계속 수행할 수 있게 될 뿐이다. callback 예외를
  쓰면 `TrainingResult` 자체가 없으므로 그 경로 전체를 잃는다.

이 내용을 `run_training()`의 docstring과 `TrainingProgressCallback`
타입 별칭 옆 주석에 그대로 남긴다.

---

## 6. Cooperative stop API 후보 비교와 최종 선택

```python
ShouldStopCallback = Callable[[], bool]
```

| 후보 | 설명 | 판정 |
|---|---|---|
| A. plain callable(`Callable[[], bool]`) | 인자 없이 bool 반환하는 아무 callable | **채택** |
| B. Protocol(`class StopToken(Protocol): def is_stop_requested(self) -> bool`) | 구조적 타입의 프로토콜 클래스 | 기각 |
| C. `threading.Event` 직접 타입으로 받음 | `stop_event: threading.Event \| None` | 기각(타입으로는) |

- **B 기각 근거**: `Callable[[], bool]`이 이미 구조적 타이핑으로
  함수/람다/바운드 메서드/`__call__`을 가진 객체를 전부 받아들인다.
  Protocol을 새로 정의해도 "인자 없이 bool을 반환한다"는 계약 이상의
  것을 표현하지 못한다 — 클래스 정의, 추상 메서드 이름 약속만 늘어나고
  실질적 이득이 없다. 이 프로젝트에 Protocol을 쓴 선례가 없다는 것도
  근거다(§14 anti-overengineering: "불필요한 protocol/interface 금지").
- **C 기각 근거(타입으로 강제하는 것만)**: `threading.Event`를 core
  API의 타입으로 못박으면 `training/loop.py`가 `threading` 모듈에
  의존하게 된다. 지금 실제 사용처(CLI/E2E)는 전부 단일 스레드
  동기 실행이라 threading이 전혀 필요 없다 — 미래의 GUI가 정말
  `threading.Event`를 쓸 수도 있지만, 그때도 `stop_event.is_set`이라는
  **바운드 메서드**를 넘기면 candidate A 타입에 자연스럽게 맞는다(아래
  예시). 즉 C가 주려는 실사용 편의는 A로도 100% 커버되고, A는 추가로
  asyncio/multiprocessing.Event/단순 플래그 객체 등 다른 실행 모델도
  전부 지원한다.

```python
stop_event = threading.Event()
...
run_training(..., should_stop=stop_event.is_set)   # A 타입에 그대로 맞음
```

**최종 선택: A.** `training/loop.py`는 `threading`을 import하지 않는다.

---

## 7. Stop 확인 시점과 시나리오별 동작

### 확인 시점: **callback 호출 이후, 다음 epoch 실행 여부를 결정하기
직전**(사용자 제시 D를 기반으로 하되, 리뷰로 정밀화)

- A(각 batch 전)/B(각 batch 후): `train_one_epoch()`은 이 Phase에서
  전혀 건드리지 않는다(§2 비목표: "batch 단위 progress callback"도
  명시적으로 제외) — batch 중간에 멈추면 optimizer의 부분 진행 상태와
  DataLoader sampler 위치를 복구해야 하는데, 이는 Phase 4F/4G가
  epoch 경계로 못박은 exact-resume 계약과 정면으로 배치된다(사용자
  근거 그대로 채택).
- E(둘 다): 시작 전 체크를 추가하면 아래에서 설명할 "항상 최소 1
  epoch 보장"이라는 단순하고 강력한 불변조건이 깨질 수 있어 기각.
- **D를 그대로 쓰지 않고 정밀화한 이유(리뷰 반영)**: "epoch 완료 후"라는
  표현만으로는 "콜백 전이냐 후냐"가 정해지지 않는다. 최초 설계는
  `should_stop()`을 콜백보다 먼저 평가했는데, 이러면 "콜백 안에서
  사용자가 stop 플래그를 세팅"하는 가장 자연스러운 UI 패턴이 **그
  epoch에서는 반영되지 못하고 다음 epoch가 한 번 더 온전히 실행된
  뒤에야** 반영되는 문제가 있었다. **`should_stop()` 평가를 콜백
  다음으로 옮기면**, 콜백이 동기적으로 실행되며 세팅한 플래그를 바로
  다음 줄의 `should_stop()`이 즉시 관측할 수 있다.

**핵심 설계**: `should_stop()`은 (1) 이번 epoch이 early stopping으로
끝난 게 아니고, (2) `progress_callback` 호출이 끝났고, (3) 이번
호출에서 아직 실행할 epoch가 하나라도 남아 있을 때만 평가한다. 세
조건 중 하나라도 거짓이면 아예 평가하지 않는다.

```python
for epoch in range(completed_epochs + 1, completed_epochs + config.epochs + 1):
    run_epoch = epoch - completed_epochs
    train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
    history.train_losses.append(train_loss)
    val_loss, val_accuracy = evaluate(model, val_loader, device=device)
    history.val_losses.append(val_loss)
    history.val_accuracies.append(val_accuracy)
    learning_rate = optimizer.param_groups[0]["lr"]

    if history.best_val_loss is None or val_loss < history.best_val_loss:
        history.best_epoch = epoch
        history.best_val_loss = val_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if scheduler is not None:
        scheduler.step(val_loss)

    if (config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience):
        history.stopped_early = True

    if progress_callback is not None:
        progress_callback(TrainingProgress(
            run_epoch=run_epoch, total_run_epochs=config.epochs, global_epoch=epoch,
            train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            best_epoch=history.best_epoch, best_val_loss=history.best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            stopped_early=history.stopped_early,
        ))

    # should_stop()은 "이 요청이 실제로 뭔가를 단축시킬 수 있을 때"만
    # 평가한다 -- early stopping으로 이미 끝났거나(더 평가할 이유 없음),
    # 이번이 이번 호출의 마지막 요청 epoch라면(더 이상 건너뛸 epoch가
    # 없음) should_stop()을 평가해도 의미 있는 조기 종료가 아니다(§8 참고).
    has_next_epoch = run_epoch < config.epochs
    if (
        not history.stopped_early
        and has_next_epoch
        and should_stop is not None
        and should_stop()
    ):
        history.stopped_by_user = True

    if history.stopped_early or history.stopped_by_user:
        break
```

(`train_loss`를 지역 변수로 뽑아내는 것 외에는 기존 코드 순서를 그대로
유지한 최소 diff이지만, **하나 정확히 짚어야 할 변경이 있다(리뷰
반영)**: early-stopping 조건 판정 자체(`config.early_stopping_patience`
비교식)와 그 조건이 참일 때 `history.stopped_early = True`를 설정하는
의미는 기존과 완전히 동일하게 유지했지만, **기존 코드가 그 자리에서
바로 실행하던 `break`는 이 자리에서 제거하고 콜백 호출 이후의 공통
종료 판정(`if history.stopped_early or history.stopped_by_user: break`)
으로 옮겼다.** 그래야 early stopping이 발동한 epoch에서도
`progress_callback`이 (그 epoch을 건너뛰지 않고) 정확히 한 번 호출된
뒤에 루프가 끝난다 — 콜백을 항상 부르고 싶다는 §2 목표를 만족하려면
`break` 위치를 반드시 옮겨야 했다. `not history.stopped_early` 가드는
여전히 두 플래그의 상호 배타성을 보장한다(§8).)

### 시나리오 1: 시작 전에 이미 stop 요청이 있는 경우

`should_stop()`은 매 epoch의 콜백 호출 **이후**에만 평가되므로,
`run_training()`이 호출되기 전부터 `should_stop()`이 이미 `True`를
반환하고 있어도 **첫 epoch은 항상 정상적으로 전부 실행되고 콜백도
호출된다** — 그 콜백이 반환된 뒤에야 비로소 `should_stop()`이 평가된다
(단, `config.epochs == 1`이면 `has_next_epoch`이 처음부터 거짓이라
`should_stop()`은 아예 평가되지 않는다 — 아래 "특수 사례" 참고). 즉
**"0 epoch 결과"는 이 설계에서 구조적으로 불가능하다** — 별도의 사전
체크나 특수 분기를 추가하지 않아도 자동으로 보장된다.

### 시나리오 2: epoch 1 callback 이후 stop 요청

**콜백 안에서 동기적으로 stop 플래그를 세팅하는 경우**(가장 흔한 UI
패턴 — 예: 콜백이 progress를 UI에 반영하고, 그 사이 사용자가 이미
누른 "정지" 버튼 상태를 콜백이 확인해 공유 플래그를 세팅): epoch
1의 콜백이 반환되자마자 **바로 다음 줄**에서 `should_stop()`이
평가되므로, **지연 없이 그 자리에서** epoch 2 진입이 막힌다 — 이것이
이번 리뷰로 순서를 바꾼 핵심 이유다.

**콜백과 무관하게 완전히 비동기로(다른 스레드에서 임의 시점에) stop이
요청되는 경우**: 여전히 epoch 경계에서만 관측되므로, 이미 진행 중이던
epoch은 항상 끝까지 완료된다 — 요청 시점이 "epoch N의 `should_stop()`
평가 직후"였다면 epoch N+1이 통째로 한 번 더 실행된 뒤에야 반영된다.
즉 **동기적 콜백 기반 stop은 지연이 사실상 0이고, 완전 비동기 stop은
여전히 최대 1 epoch 지연이 있을 수 있다** — 이 구분을 문서/CLI
출력에 명시한다.

### 시나리오 3: 마지막 epoch가 끝난 뒤 stop 요청 (정책 변경, 리뷰 반영)

**`should_stop()`은 이번 호출의 마지막 요청 epoch에서는 평가되지
않는다** (`has_next_epoch`이 그 epoch에서 이미 거짓이므로). 따라서
모든 요청 epoch를 이미 다 완료한 뒤에 stop 요청이 있었다는 사실은
`history.stopped_by_user`에 **기록되지 않는다** — 이 학습은 그냥
정상 완료로 취급된다.

근거: `stopped_by_user=True`는 "사용자 요청으로 예정된 epoch 중
일부가 실제로 생략됐다"는 의미로만 쓴다. 마지막 요청 epoch까지 전부
실행됐다면 그 stop 요청은 무엇도 단축시키지 못했으므로, 이를
"사용자가 멈췄다"고 기록하면 오해를 준다(예: UI가 "N epoch 중
M에서 중단됨"처럼 보여줄 때, M이 사실 마지막 epoch라면 애초에 표시할
"중단" 자체가 없어야 함). 이 정책은 다음 특수 사례로 자연스럽게
이어진다:

**특수 사례 — `config.epochs == 1`**: `has_next_epoch`이 이 호출의
유일한 epoch에서부터 이미 거짓이므로, `should_stop`에 무엇을
넘기든(심지어 항상 `True`를 반환해도) `should_stop()`은 **절대
평가되지 않는다**. 1 epoch짜리 호출에서 "정지"는 의미론적으로 아무
것도 단축할 게 없으므로(이미 그 1 epoch가 이번 호출의 전부), 이는
버그가 아니라 위 정책의 자연스러운 귀결이다 — §14-1에 이를 확인하는
전용 테스트를 둔다.

---

## 8. Early stopping과 user stop의 우선순위

**early stopping이 항상 우선한다 — 그리고 마지막 요청 epoch에서는
user stop 자체가 성립하지 않는다(§7 시나리오 3, 리뷰로 추가된 조건).**
§7 pseudocode의 `not history.stopped_early and has_next_epoch and ...`
가드가 이를 강제한다 — 이번 epoch에서 early-stopping 조건이 만족되면
`should_stop()`은 아예 평가되지 않고, `history.stopped_early=True`,
`history.stopped_by_user`는 그대로 `False`(기본값)로 남는다. early
stopping이 발동하지 않았더라도, 이번 epoch가 이번 호출의 마지막 요청
epoch였다면(`has_next_epoch`이 거짓) 역시 `should_stop()`이 평가되지
않아 `stopped_by_user`는 `False`로 남는다.

근거:
- 기존 `loop.py`의 early-stopping **조건 판정과
  `history.stopped_early = True` 설정 의미는 그대로 유지**하고, 그
  자리에 있던 즉시 `break`만 콜백 호출 이후의 공통 종료 판정으로
  옮긴 뒤(§7에서 정확히 설명) 그 뒤에 새 stop 판정을 순서대로
  추가하는 것이 최소 diff다.
- 두 조건이 동시에 `True`인 epoch에서 "어느 쪽 표시가 맞는가"를 별도로
  결정해야 하는 모호성을 애초에 만들지 않는다 — 항상 정확히 하나의
  종료 사유만 기록된다(`stopped_early`/`stopped_by_user`가 동시에
  `True`인 경우는 존재하지 않음, §14-1에서 다시 확인).
- early stopping은 "데이터가 보여준 수렴 판단"이고 이미 patience
  epoch만큼 학습이 정체됐다는 강한 신호다 — 같은 순간에 사용자가 stop을
  요청했다면 그 결정과 방향이 같으므로(둘 다 "그만"), 우선순위 자체가
  실질적 손해로 이어지지 않는다.
- "마지막 요청 epoch에서는 stop이 성립하지 않는다"는 조건은 §7
  시나리오 3에서 설명한 대로, `stopped_by_user=True`가 "실제로 뭔가를
  단축시켰다"는 뜻만 갖도록 의미를 명확히 하기 위한 것이다 — 우선순위
  문제라기보다는 애초에 "이 epoch에서는 stop이 의미가 없다"는
  판단이다.

---

## 9. `TrainingHistory`/`TrainingResult` 변경안

### 채택: 후보 A (`TrainingHistory`에 bool 필드 추가), enum 도입 안 함

```python
@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_loss: float | None = None
    stopped_early: bool = False
    stopped_by_user: bool = False   # 신규
```

**enum(`TrainingStopReason`) 기각 근거**: 저장소 전체를 확인한 결과
(`config.py`의 `OPTIMIZER_CHOICES`/`LR_SCHEDULER_CHOICES`는 `Enum`이
아니라 문자열 튜플, `TrainingHistory`/`TrainingResult`/
`ImageFolderResumeMetadata`/`ImageFolderWorkflowRequest`/`Result` 전부
plain dataclass + bool/str/int/float/Path 필드만 사용) **이 프로젝트
어디에도 `Enum`을 쓰는 곳이 없다.** 지금 도입하면 이 파일 하나만
스타일이 달라진다. `stopped_early`가 이미 bool로 존재하는데
`stopped_by_user`만 다른 표현 방식(`TrainingResult.stop_reason:
Literal[...]`, 후보 B/C)을 쓰면 "왜 이 필드만 다른 방식인가"라는
비대칭이 생긴다. 가장 단순하고 기존 계약을 가장 덜 깨는 답은 **같은
자리(`TrainingHistory`)에 같은 방식(bool)으로 나란히 추가**하는 것이다.

`TrainingResult`(`loop.py:113-132`)는 **변경하지 않는다** —
`stopped_early`가 지금도 `TrainingResult`가 아니라 `TrainingResult.history.stopped_early`로
접근되므로(모든 기존 E2E/CLI가 이렇게 사용, §1-3 확인), `stopped_by_user`도
같은 경로(`result.history.stopped_by_user`)로 노출하는 것이 기존
사용 패턴과 완전히 일치한다.

### `stopped_by_user`는 `stopped_early`와 달리 resume을 막지 않는다 (핵심 결정)

`TrainingResumeState.__post_init__`(`loop.py:192-198`)의 기존 거부
로직은 **`stopped_early`만** 검사한다. **이 로직은 절대 확장하지
않는다** — `stopped_by_user=True`인 checkpoint는 그대로 resume 가능해야
한다. 이것이 사실 Phase 4I 전체의 존재 이유다: cooperative stop을
프로세스 강제 종료 대신 만든 이유가 "나중에 정확히 이어서 재개하기
위함"이므로, 재개를 막으면 기능의 의미가 없어진다. `stopped_early`가
resume을 막는 이유(모델이 이미 수렴했다고 판단된 상태라 더 학습해도
의미가 약함)와 `stopped_by_user`가 resume을 막지 않아야 하는 이유(사람이
그냥 잠시 멈췄을 뿐, 모델 상태는 여전히 유효한 학습 중간 지점)는
근본적으로 다른 상황이라는 점을 코드에도 주석으로 남긴다.

### resume 시 `stopped_by_user` 리셋 (놓치기 쉬운 세부사항)

`run_training()`의 resume 분기(`loop.py:310`, `history =
copy.deepcopy(resume_state.history)`)에서 `stopped_by_user`를 복사해
오면, 만약 이전 checkpoint가 사용자 중단으로 저장된 것이었다면(즉
`resume_state.history.stopped_by_user == True`) 그 값이 그대로
새 호출의 history에 남는다. **이번 호출은 아직 중단된 적이 없으므로**,
복사 직후 명시적으로 `False`로 되돌려야 한다:

```python
history = copy.deepcopy(resume_state.history)
history.stopped_by_user = False   # 이번 호출은 아직 stop되지 않았음
```

(`stopped_early`는 이미 `TrainingResumeState.__post_init__`이
`True`인 경우를 아예 거부하므로 이 리셋이 필요 없다 — resume 시점에
`stopped_early`는 항상 `False`로 들어온다는 것이 이미 보장돼 있다.
`stopped_by_user`는 그런 진입 차단이 없으므로 명시적 리셋이 필요하다.)

### 기존 checkpoint/JSON history와의 하위 호환성

- **`history.py`**(`save_training_history()`/`load_training_history()`,
  `history.py:15-26`)는 `asdict(history)`로 저장하고
  `TrainingHistory(**data)`로 복원하는 완전히 제네릭한 코드다 — 필드
  목록을 하드코딩하지 않는다. **코드 변경이 전혀 필요 없다.** 새 필드가
  기본값(`False`)을 가지므로, 옛 JSON(`stopped_by_user` 키 없음)을
  `TrainingHistory(**data)`로 복원하면 Python dataclass가 자동으로
  기본값을 채운다(누락된 키 + 기본값 존재 = 정상 동작, 이건 dataclass의
  표준 동작이지 새로 구현할 게 없다). 새로 저장하는 JSON은
  `asdict()`가 모든 필드를 자동으로 포함하므로 `stopped_by_user`가
  자동으로 들어간다.
- **`checkpoint.py`의 `_REQUIRED_HISTORY_FIELDS`**(`:54-61`)는
  `("train_losses", "val_losses", "val_accuracies", "best_epoch",
  "best_val_loss", "stopped_early")`로 고정돼 있고, **여기에
  `stopped_by_user`를 추가하지 않는다** — 추가하면 옛 checkpoint(이
  키가 없음)가 `load_training_checkpoint()`의 구조 검증(`:190-192`)에서
  거부되어 하위 호환이 깨진다. `stopped_by_user`는 선택적(옵션) 필드로
  남기고, `TrainingHistory(**payload["history"])`가 기본값으로 채운다
  (`imagefolder_workflow.py:150`이 이미 이 패턴을 쓰고 있음 — 코드
  변경 없이 그대로 재사용됨).
- **결론: `checkpoint.py`는 이 Phase에서 전혀 수정하지 않는다.** 이미
  완전히 제네릭하게 설계돼 있어(`asdict()`/`**dict` 패턴), 새 필드
  추가가 요구하는 유일한 작업은 `_REQUIRED_HISTORY_FIELDS`에 **추가하지
  않는 것**(즉 아무것도 안 하는 것)이다.

---

## 10. 중단 후 workflow artifact 저장 정책

**후보 A(정상 종료와 동일하게 모든 artifact 생성) 채택 — 그리고 이
Phase에서는 `imagefolder_workflow.py`의 로직을 사실상 전혀 바꾸지
않아도 이 정책이 자동으로 성립한다.**

근거: §7에서 확정한 대로 `run_training()`은 **항상 최소 1개의 새
epoch을 완료한 뒤에만 멈출 수 있다**(0 epoch 결과가 구조적으로
불가능). 따라서 `stopped_by_user=True`로 끝난 `TrainingResult`도
`stopped_early=True`로 끝난 것과 마찬가지로 **완전히 유효한
결과**다 — `best_state_dict`는 항상 채워져 있고(§1-1의 기존 주석,
`loop.py:344-347` 근거 그대로 유지됨), `optimizer_state_dict`/
`scheduler_state_dict`/`history`도 전부 정상. `imagefolder_workflow.py`의
나머지 코드(`:196-284`: history/class mapping 저장, checkpoint 저장,
best model 저장, test 평가, TorchScript export)는 `training_result`가
어떤 사유로 멈췄는지 **전혀 검사하지 않는다** — 그래서 이 부분은 Phase
4I에서 **한 줄도 바꿀 필요가 없다**.

- `checkpoint_out`이 지정됐다면 중단 지점 checkpoint도 저장된다 —
  이미 그렇게 동작한다(§9 결정 덕분에 이 checkpoint는 `stopped_early`
  checkpoint와 달리 resume도 가능).
- best model/test 평가/TorchScript export도 계속 수행된다 — 이미
  그렇게 동작한다.
- 후보 B(최소 artifact만)/C(Request 옵션화)는 검토했지만 채택하지
  않는다(리뷰 반영 — 실행 시간 추정에 근거하지 않고 다음 이유로
  기각한다): epoch 경계에서 사용자 요청으로 멈춘 결과도 §7/§10 앞부분의
  근거대로 **완전하고 유효한 `TrainingResult`**다 — `stopped_early`로
  끝난 것과 구분할 근거가 없다. 따라서 기존 history/checkpoint/best
  model/test 평가/TorchScript export 후처리 경로를 사유와 무관하게
  그대로 재사용하는 것이 가장 단순하다. B처럼 "사용자 중단일 때만 test
  평가/export를 건너뛴다"는 별도 분기를 두면, 실제로 얼마나 시간이
  걸리는지와 무관하게 "왜 이 종료 사유만 다르게 취급하는가"라는 새
  정책 질문과 그걸 구현/테스트할 코드 경로가 늘어난다 — 지금 요구
  대비 복잡성만 커지므로 채택하지 않는다. C(Request 옵션화)도 같은
  이유로 사용처가 아직 없는 정책 분기라 과설계다(§14 원칙).
- **0 epoch 실행 상태 관련 우려**(사용자가 §8에서 명시적으로 검토를
  요청한 부분)는 §7의 설계로 이미 원천 차단됐으므로 여기서 추가로
  검토할 문제가 남아있지 않다 — `best_state_dict`가 없거나 test 평가가
  불가능한 상황 자체가 발생하지 않는다.

---

## 11. CLI 연결 범위

### 채택: 후보 A + B의 부분집합

- **A(core callback/stop API + 테스트만 추가)**: 그대로 채택.
- **B(CLI에서 progress callback으로 실시간 출력)**: 채택하되, **fresh
  학습의 출력 형식은 유지하고, resume의 출력 "정책"은 의도적으로
  바꾼다(리뷰 반영 — 최초 설계의 "출력 내용을 그대로 유지"라는 설명은
  부정확했다).** `train_imagefolder.py`의 `main()`(`:160-167`)에서
  학습 완료 후 `zip()`으로 **누적** `result.history`를 순회하며
  출력하던 루프(`:161-164`)를 제거하고, 대신
  `run_imagefolder_training_workflow()` 호출 시 `progress_callback=`
  으로 다음과 같은 함수를 넘긴다:

  ```python
  def _print_progress(progress: TrainingProgress) -> None:
      print(
          f"  epoch {progress.global_epoch}: "
          f"train_loss={progress.train_loss:.4f} "
          f"val_loss={progress.val_loss:.4f} "
          f"val_acc={progress.val_accuracy:.4f}"
      )
  ```

  **왜 `run_epoch`이 아니라 `global_epoch`인지(정정)**: 기존
  `train_imagefolder.py`는 resume 시에도 `result.history`(누적 전체)를
  `enumerate(..., start=1)`로 순회하므로, 2 epoch 완료 checkpoint에서
  3 epoch를 resume하면 화면에는 **epoch 1부터 5까지 전부** 다시
  출력됐다(이전 epoch도 사후에 다시 그려짐). `progress_callback`은
  **이번 호출에서 새로 완료된 epoch에 대해서만** 호출되므로(`run_epoch`은
  이번 resume 안에서만 1부터 다시 시작함), 콜백에 `progress.run_epoch`을
  그대로 쓰면 화면에 "epoch 1, 2, 3"이 다시 나타나 실제 누적 epoch
  번호(4, 5)와 완전히 어긋난다. **최종 정책은 다음과 같다**:
  - fresh 학습: `global_epoch`이 1부터 순서대로 출력된다(fresh에서는
    `run_epoch == global_epoch`이므로 출력 자체는 기존과 동일).
  - resume 학습: **이전에 이미 완료된 epoch은 다시 출력하지 않는다.**
    이번 호출에서 **새로 완료된 epoch만**, `progress.global_epoch`
    번호로 실시간 출력한다. 예를 들어 2 epoch checkpoint에서 3 epoch를
    resume하면 화면에는 `epoch 3` → `epoch 4` → `epoch 5`만 순서대로
    출력되고, `epoch 1`/`epoch 2`는(이전 실행에서 이미 출력됐던 것이므로)
    이번 실행에서 다시 그려지지 않는다.
  - 이는 "출력 내용을 그대로 유지"가 아니라 **"fresh 학습의 출력
    형식은 유지하되, resume의 출력 방식 자체는 사후 누적 재출력에서
    실시간 신규분만 출력하는 방식으로 의도적으로 바꾼다"**는 뜻이다 —
    실시간 로그로 전환하는 이상, resume마다 과거 epoch를 화면에 다시
    쏟아내는 것은 실시간 로그의 의미와 맞지 않기 때문이다.

  `main()` 끝부분의 요약 출력(`:165-181`)은 그대로 유지하되,
  `stopped_early` 줄 옆에 `stopped_by_user`도 출력하도록 한 줄
  추가한다(이 요약은 여전히 최종 `result.history`(누적 전체)를 보고
  만드는 것이라 위 정책과 무관 — `best_epoch`/`best_val_loss` 등은
  원래도 누적 기준 값이다).
- **C(Ctrl+C를 cooperative stop으로 변환)**: 기각. `signal` 핸들링과
  `train_one_epoch()`의 batch 루프 도중 `KeyboardInterrupt`가 그대로
  전파되는 문제(정확히는 batch 중간에 예외가 나서 optimizer 상태가
  일관되지 않게 끊길 위험)가 있고, 이는 §2 비목표에 명시된
  "SIGINT/KeyboardInterrupt graceful resume 제외"와 직접 연결된다.
- **D(`--stop-file` 같은 테스트용 옵션)**: 기각. "테스트용" 인터페이스를
  production CLI에 영구히 노출하는 것은 목적에 안 맞고, 실제 사용자
  stop UI가 아직 없는 상태에서 만들 이유가 없다(§13 비목표: "run
  directory 자동 생성"과 마찬가지로 아직 실사용처 없는 기능).
- `--stop-file`은 물론 CLI에서 `should_stop=`을 아예 넘기지 않는다
  (`None`) — Phase 4I는 core에 stop 메커니즘을 "추가"할 뿐, CLI가 그
  메커니즘을 트리거할 방법을 아직 제공하지 않는다.

### E2E에는 callback/stop을 넘기지 않는다

`run_imagefolder_training_e2e.py`의 `_run_workflow_stage()`(`:97-111`)는
**변경하지 않는다** — 사용자 권장대로 "E2E는 callback 없이 기존 동작
유지". `_run_workflow_stage()`가 이미 `run_imagefolder_training_workflow(request)`를
호출한 뒤 자기 방식대로 history를 순회해 출력하는 코드(`:106-109`)를
갖고 있으므로, `progress_callback`/`should_stop`을 넘기지 않으면
`run_training()` 내부에서 그 두 검사는 전부 스킵되고 수치 결과는
Phase 4H와 **완전히 동일**하다.

### CLI 실시간 출력 변경이 regression anchor/테스트에 미치는 영향 분석

- **regression anchor 수치(§14-4, `docs/phase4h_production_training_cli_design.md`)**:
  영향 없음 — E2E 스크립트 자체를 안 건드리므로 anchor 재현 여부와
  무관.
- **`tests/scripts/test_train_imagefolder_args.py`**: `parse_args()`만
  테스트하므로 영향 없음(CLI에 새 플래그를 추가하지 않으므로 이
  파일은 변경조차 필요 없음).
- **`tests/scripts/test_train_imagefolder_cli.py`**: 전부 exit
  code/artifact 파일 존재 여부만 확인하고 **stdout 내용을 검사하는
  테스트가 하나도 없다**(코드 확인 완료) — 출력 시점이 사후에서
  실시간으로 바뀌어도 이 테스트들은 전부 그대로 통과한다. 다만 §16에서
  "출력이 실제로 매 epoch마다 나오는지"를 확인하는 가벼운 신규 테스트를
  추가하기로 했고(brittle하지 않은 방식으로, §16 참고), §14-4대로
  실제 구현에도 추가됐다.
- **`tests/training/test_imagefolder_workflow.py`**: `progress_callback`/
  `should_stop`을 넘기지 않는 기존 9개 테스트는 그대로 유지 —
  `run_imagefolder_training_workflow()`가 이 두 인자에 기본값 `None`을
  두므로 시그니처 확장이 기존 호출을 깨지 않는다.

---

## 12. `ImageFolderWorkflowRequest`/함수 시그니처 설계

### 채택: keyword-only 함수 인자 (사용자 권장 방향과 동일)

```python
def run_imagefolder_training_workflow(
    request: ImageFolderWorkflowRequest,
    *,
    progress_callback: TrainingProgressCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
) -> ImageFolderWorkflowResult:
    ...
```

`ImageFolderWorkflowRequest`에는 필드를 추가하지 않는다. 근거:

- **serialization 가능성**: `ImageFolderWorkflowRequest`는 지금 전부
  `Path`/`TrainingConfig`(자체도 str/int/float/bool 필드뿐)/`bool`/`int`로만
  구성돼 있어, 원칙적으로 `asdict()` 후 JSON으로 직렬화 가능한
  모양이다(§13 비목표의 "run_config.json"이 나중에 이 구조를 그대로
  써먹을 수 있다는 뜻이기도 하다). callable을 필드로 넣으면 이 성질이
  깨진다 — `asdict(request)`가 callable을 만나면 즉시 사용할 수 없는
  값이 되거나 실패한다.
- **데이터 모델 오염**: `Request`는 "이 학습을 어떻게 설정할지"를
  나타내는 값 객체이고, callback/stop은 "이 실행을 프로그램적으로 어떻게
  제어/관찰할지"를 나타내는 실행 제어 요소다. 성격이 다른 두 개념을
  한 dataclass에 넣으면 향후 `Request`를 로그로 남기거나 비교하거나
  캐시 키로 쓰는 등의 용도에서 매번 "callable 필드는 어떻게 하지"라는
  질문이 반복된다.
- **호출부 가독성**: `run_imagefolder_training_workflow(request,
  progress_callback=cb, should_stop=stop)`가
  `run_imagefolder_training_workflow(ImageFolderWorkflowRequest(...,
  progress_callback=cb, should_stop=stop))`보다 "이 실행에 한정된
  선택적 동작"이라는 의도를 더 명확히 드러낸다 — `Request`를 재사용하며
  callback만 바꿔 여러 번 호출하는 경우(예: 같은 설정으로 재시도)에도
  자연스럽다.
- `run_training()` 자체도 이미 `progress_callback`/`should_stop`을
  keyword-only로 두므로(§6/§7), workflow 계층도 같은 관례를 그대로
  잇는다.

`run_imagefolder_training_workflow()` 내부 변경은 단 한 줄이다
(`imagefolder_workflow.py:188-190`의 `run_training(...)` 호출에
`progress_callback=progress_callback, should_stop=should_stop` 추가) —
이 두 인자를 `run_training()`에 그대로 전달하는 것 외에 workflow가
직접 `should_stop()`을 호출하거나 `progress_callback`을 스스로
소비하는 로직은 **전혀 두지 않는다**(§17 회귀 불변조건과 직결).

---

## 13. 파일별 변경 계획

**수정**:
- `src/image_ai_studio/training/loop.py` — `TrainingProgress`
  dataclass, `TrainingProgressCallback`/`ShouldStopCallback` 타입 별칭,
  `TrainingHistory.stopped_by_user` 필드, `run_training()` 시그니처에
  `progress_callback`/`should_stop` keyword-only 인자 추가 + epoch
  루프에 callback 호출/stop 판정 삽입 + resume 분기에 `stopped_by_user`
  리셋 한 줄, docstring 갱신.
- `src/image_ai_studio/training/imagefolder_workflow.py` —
  `run_imagefolder_training_workflow()` 시그니처에 두 keyword-only
  인자 추가, 내부 `run_training(...)` 호출에 그대로 전달(1줄).
- `scripts/train_imagefolder.py` — 사후 `zip()` 출력 루프를
  `progress_callback` 기반 실시간 출력으로 교체, 요약부에
  `stopped_by_user` 출력 추가.
- `README.md` — Phase 4I 절 신설.

**신규**: `docs/phase4i_training_progress_and_stop_design.md`(본 문서).

**변경하지 않음(분석 결과 필요성을 찾지 못함)**:
- `src/image_ai_studio/training/config.py` — callback/stop은
  `TrainingConfig`가 아니라 `run_training()`의 별도 인자이고,
  `RESUME_CONFIG_FIELDS`(optimizer/scheduler 구조 비교용)와는 무관.
- `src/image_ai_studio/training/checkpoint.py` — §9에서 근거 확인
  (제네릭 `asdict()`/`**dict` 패턴이라 신규 필드가 자동으로 흘러감,
  `_REQUIRED_HISTORY_FIELDS`에 `stopped_by_user`를 **추가하지 않는
  것**이 정답).
- `src/image_ai_studio/training/history.py` — 같은 이유로 제네릭.
- `src/image_ai_studio/training/imagefolder_resume.py` — ModelSpec/
  dataset 호환성 검증과 무관, 손댈 이유 없음.
- `scripts/run_imagefolder_training_e2e.py` — §11에서 "callback 없이
  기존 동작 유지"로 확정.
- `model_definition/*`/`export/*`/`parity/*`/C++ 코드/다른 E2E
  스크립트 — 전부 무관.

---

## 14. 테스트 계획 및 구현 결과

> **아래 §14-1~§14-4 항목은 설계 당시 세운 검증 계획이다.** 실제
> 구현에서는 서로 겹치는 항목을 자연스럽게 병합해
> `tests/training/test_loop.py`에 신규 테스트 **함수 14개**로
> 구현했다(설계 당시 계획한 16개 "항목"과는 별개 수치 — 계획 항목
> 수와 실제 테스트 함수 수를 혼동하지 말 것, §18 참고). 추가로:
> - `tests/training/test_imagefolder_workflow.py`에 4개
> - `tests/scripts/test_train_imagefolder_cli.py`에 2개
> - `tests/training/test_history.py`에 하위 호환 테스트 2개 추가 및
>   기존 키 목록 검증 테스트 1개 갱신
> - `tests/training/test_checkpoint.py`에 legacy checkpoint 하위 호환
>   테스트 1개 추가(`_REQUIRED_HISTORY_FIELDS`는 수정하지 않음)
>
> **최종 결과: `tests/training/` + `tests/scripts/` 205 passed, 전체
> `pytest` 362 passed, 0 failed.** 아래 §14-1~§14-4의 번호가 매겨진
> 목록은 설계 당시의 **계획 항목**으로서 기록 목적으로 그대로
> 유지한다 — 실제 구현된 테스트 함수명/개수는 위 요약과 §18을 기준으로
> 삼을 것.

### 14-1. `tests/training/test_loop.py` (신규 테스트, 기존 파일에 추가)

기존 테스트 명명 규칙(`test_run_training_...`)을 그대로 따른다.

1. `test_run_training_progress_callback_none_matches_no_callback_argument` —
   `progress_callback=None`을 명시적으로 넘긴 결과와 아예 안 넘긴
   결과가 (같은 seed로) 완전히 동일한지 확인. 기존 `progress_callback`
   없는 모든 테스트가 이미 "callback=None 기본값" 경로를 검증하고
   있으므로, 이 테스트는 "명시적 None"과 "생략"이 같은 코드 경로임을
   짧게 못박는 역할만 한다.
2. `test_run_training_progress_callback_called_once_per_completed_epoch` —
   3 epoch 정상 완료, 콜백 호출 리스트 길이 == 3.
3. `test_run_training_progress_callback_reports_correct_epoch_and_metrics` —
   콜백으로 수집한 각 `TrainingProgress`의 `train_loss`/`val_loss`/
   `val_accuracy`/`best_epoch`/`best_val_loss`/`epochs_without_improvement`가
   반환된 `TrainingResult.history`의 같은 인덱스 값과 정확히 일치.
4. `test_run_training_progress_callback_epoch_numbering_across_resume` —
   2 epoch 완료 후 resume으로 3 epoch 추가 실행, 수집된
   `run_epoch`이 `[1,2,3]`, `global_epoch`이 `[3,4,5]`,
   `total_run_epochs`가 매번 `3`인지 확인.
5. `test_run_training_progress_callback_learning_rate_reflects_value_used_this_epoch` —
   `lr_scheduler="plateau"`로 LR이 실제로 줄어드는 시나리오(기존
   `test_build_scheduler_reduces_lr_after_patience_bad_steps` 패턴
   재사용)에서, LR이 줄어든 **바로 다음** 콜백에서야 새 LR이 보이는지
   확인(줄어든 그 epoch의 콜백에는 아직 이전 LR이 보여야 함).
6. `test_run_training_progress_callback_exception_propagates` —
   특정 epoch에서 예외를 던지는 콜백을 넘기고 `pytest.raises`로 확인.
7. `test_run_training_progress_callback_exception_leaves_history_and_model_updated_through_that_epoch` —
   위 테스트를 확장해, 예외 발생 후에도 호출자가 들고 있는 `model`
   객체의 `state_dict()`가 그 epoch까지 학습된 값과 일치하는지 확인
   (§5의 문서화 내용을 테스트로 고정).
8. `test_run_training_should_stop_none_matches_no_argument` — 3번과 대응.
9. `test_run_training_should_stop_true_from_start_still_runs_at_least_one_epoch` —
   `config.epochs=3`(1이면 애초에 `should_stop`이 평가되지 않으므로
   반드시 1보다 커야 함, 아래 13번 참고), `should_stop=lambda: True`를
   처음부터 넘기고, fresh 학습이 정확히 1 epoch만 실행되고
   `history.stopped_by_user is True`, `best_state_dict is not None`인지
   확인(§7 시나리오 1 + §10 근거를 테스트로 고정).
10. `test_run_training_should_stop_set_inside_callback_prevents_next_epoch_immediately` —
    **콜백 내부에서** 공유 플래그를 세팅하고 `should_stop`이 그 플래그를
    읽도록 구성(예: `flag = []`, `progress_callback=lambda p: flag.append(1)`,
    `should_stop=lambda: bool(flag)`), `config.epochs=3`으로 실행해 정확히
    1 epoch만 실행되고 2번째 epoch가 시작조차 되지 않는지 확인 — 이번
    리뷰로 바로잡은 "콜백 → should_stop 평가" 순서(§4/§7) 자체를
    증명하는 핵심 테스트.
11. `test_run_training_should_stop_evaluated_after_progress_callback_returns` —
    10번과 상호보완적으로, `should_stop` mock의 호출 시점이
    `progress_callback` mock의 호출 **이후**였는지를 두 mock의 호출
    순서를 기록해 직접 확인(예: 공유 리스트에 `"callback"`/
    `"should_stop"` 문자열을 순서대로 append).
12. `test_run_training_early_stopping_takes_priority_over_user_stop` —
    `early_stopping_patience=1`이면서 `should_stop`이 처음부터
    `True`인 조합에서 `history.stopped_early is True`이고
    `history.stopped_by_user is False`이며, `should_stop` mock의 호출
    횟수가 `0`인지(아예 평가되지 않았는지)까지 확인(§8).
13. `test_run_training_should_stop_true_on_final_requested_epoch_does_not_mark_stopped_by_user`
    (**정책 변경, 리뷰 반영**) — `config.epochs=2`이고 `should_stop`이
    정확히 마지막(2번째) epoch의 콜백 이후에만 `True`가 되도록 구성,
    `len(history.train_losses)==2`(더 실행되지 않음, 자연 종료와 epoch
    수가 같음)이면서 **`history.stopped_by_user is False`**인지 확인
    (§7 시나리오 3 — 이전 설계에서는 `True`를 기대했으나 정책이
    바뀌었다).
14. `test_run_training_should_stop_never_evaluated_when_config_epochs_is_one`
    (신규, §7 "특수 사례") — `config.epochs=1`, `should_stop`이 항상
    `True`를 반환하도록 구성하고도 `should_stop` mock의 호출 횟수가
    `0`이고 `history.stopped_by_user is False`, 학습이 정상 완료되는지
    확인.
15. `test_training_resume_state_accepts_stopped_by_user_history` —
    `stopped_early=False, stopped_by_user=True`인 `TrainingHistory`로
    `TrainingResumeState`를 만들어도 예외가 나지 않는지 확인(§9 핵심
    결정의 음성 테스트 — "막지 않는다"는 계약을 직접 고정).
16. `test_run_training_resume_resets_stopped_by_user_flag` —
    `stopped_by_user=True`인 `TrainingHistory`로 만든
    `resume_state`로 `run_training()`을 호출한 뒤, 반환된
    `TrainingResult.history.stopped_by_user`가(이번 호출이
    `should_stop` 없이 끝까지 돌았다면) `False`인지 확인(§9 리셋 로직
    검증).

### 14-2. `tests/training/test_imagefolder_resume.py`(checkpoint/resume)

기존 `test_imagefolder_checkpoint_resume_matches_continuous_run_exactly`
패턴 재사용, 자동 checkpoint는 구현하지 않으므로 워크플로우 없이
`checkpoint.py`의 `save_training_checkpoint()`/`load_training_checkpoint()`를
직접 쓰는 기존 저수준 패턴을 그대로 따른다.

- `test_user_stop_checkpoint_is_resumable` — `config.epochs=4`로 호출하되
  `should_stop`을 2번째 epoch의 콜백 이후 `True`가 되도록 구성해 **아직
  epoch가 남아 있는 상태에서** 멈추게 한다(§7 정책상 마지막 epoch에서
  멈추면 `stopped_by_user`가 기록되지 않으므로, 이 테스트는 반드시
  "아직 실행 안 한 epoch가 남은 채로 멈추는" 시나리오여야 함).
  `history.stopped_by_user is True`, `len(history.train_losses)==2`
  확인 후 checkpoint 저장 → 그 checkpoint에서 resume이 예외 없이
  성공하는지(= `stopped_early` checkpoint였다면 여기서 반드시
  실패해야 했을 것과 대비).
- `test_user_stop_then_resume_matches_continuous_run_exactly` — 연속
  4 epoch 실행 vs (`config.epochs=4`로 호출하되 should_stop으로 2
  epoch만에 멈춤 → checkpoint → resume 2 epoch)이 model/optimizer/
  scheduler/history 전부 정확히 일치하는지 — 기존 exact-resume 테스트의
  "epoch 수를 config로 나눈" 버전 대신 "should_stop으로 도중에 멈춘"
  버전이라는 점만 다름.

### 14-3. `tests/training/test_imagefolder_workflow.py`

- `test_progress_callback_is_forwarded_to_run_training` —
  `run_imagefolder_training_workflow(request, progress_callback=spy)`
  호출 후 `spy`가 `request.training_config.epochs`번 호출됐는지.
- `test_should_stop_is_forwarded_to_run_training` — `request.training_config.epochs`를
  2 이상으로 두고 `should_stop`이 1 epoch 후(아직 남은 epoch가 있는
  상태에서) 멈추도록 설정하고, `result.history.stopped_by_user is
  True`이고 실제로 1 epoch만 실행됐는지 확인(§7 정책상 요청한
  epoch를 전부 채우고 멈추면 `stopped_by_user`가 기록되지 않으므로,
  이 테스트는 반드시 "epoch가 남은 채로 멈추는" 시나리오여야 함).
- `test_user_stop_with_checkpoint_out_produces_resumable_checkpoint` —
  같은 이유로 `epochs`를 2 이상으로 두고 아직 epoch가 남은 상태에서
  멈춘 `should_stop` + `checkpoint_out` 조합으로 만든 checkpoint를
  다음 `run_imagefolder_training_workflow()` 호출에서 `resume_from`으로
  써서 예외 없이 이어지는지, 누적 `history` 길이가 정확한지.
- 기존 9개 테스트는 `progress_callback`/`should_stop`을 넘기지 않는
  호출 그대로 유지 — 시그니처 확장이 하위 호환됨을 이 테스트들의
  무수정 통과 자체로 증명.

### 14-4. `tests/scripts/test_train_imagefolder_cli.py`

- `test_main_prints_progress_for_each_epoch_via_callback`(신규,
  brittle하지 않게 설계) — `capsys`로 stdout을 캡처한 뒤, **정확한
  문자열 전체를 비교하지 않고** `"epoch "` 접두사를 가진 줄의 개수가
  실제 실행된 epoch 수와 일치하는지만 센다. 그리고 학습 완료 후
  중복 출력이 없는지(같은 개수 재확인)만 본다 — 사용자 지침("출력
  문자열 전체를 지나치게 엄격하게 고정하는 brittle test는 피해주세요")을
  그대로 따른다.
- `test_main_resume_prints_only_newly_completed_global_epochs`(신규,
  §11 정책 검증, **checkpoint 생성 절차를 명시(리뷰 반영)** — production
  CLI는 checkpoint를 자동 저장하지 않고 `--checkpoint-out`을 줬을
  때만 저장하므로, 1회차 실행에 반드시 그 플래그가 있어야 2회차에서
  `--resume-from`을 쓸 수 있다):
  1. 1회차 CLI 실행: `cli.main(["--model-json", ..., "--dataset-root", ...,
     "--output-dir", ..., "--epochs", "2", "--checkpoint-out",
     str(tmp_path / "checkpoint.pt")])`.
  2. 1회차 성공(`exit_code == 0`) 확인 후, 실제 코드
     (`imagefolder_resume.py`의 `metadata_path_for_checkpoint()`)로
     확인한 명명 규칙 그대로 `tmp_path / "checkpoint.pt"`와
     `tmp_path / "checkpoint.pt.meta.json"` 두 파일이 모두 생성됐는지
     확인.
  3. 2회차 CLI 실행(다른 `--output-dir`, 같은 model-json/dataset-root):
     `cli.main(["--model-json", ..., "--dataset-root", ..., "--output-dir",
     ..., "--epochs", "2", "--resume-from", str(tmp_path / "checkpoint.pt")])` —
     이번 호출의 stdout만 `capsys`로 캡처(1회차 출력과 섞이지 않도록
     `capsys.readouterr()`를 1회차 직후 한 번 소비해 버퍼를 비워 둠).
  4. `"epoch "`로 시작하는 줄에서 정규식 등으로 epoch 번호만 뽑는다.
  5. 검증: progress epoch 번호가 정확히 `[3, 4]`(이전 `[1, 2]`가 다시
     나타나지 않음), progress 줄의 총 개수가 정확히 `2`, 학습 종료
     후(요약 출력 포함) 그 개수가 그대로 유지됨(중복/재출력 없음).
     문자열 전체 비교는 피하고 epoch 번호가 포함된 줄만 파싱한다(사용자
     지침 그대로).
- 기존 6개 테스트(exit code/artifact 존재 확인)는 무수정 — §11 분석에서
  이미 stdout을 검사하지 않음을 확인했으므로 영향 없음.

### 14-5. 전체 회귀

`tests/training/` + `tests/scripts/` 전체, 전체 `pytest`, 기존 4개
E2E 스크립트(`run_training_e2e.py`/`run_real_training_e2e.py`/
`run_resume_training_e2e.py`/`run_imagefolder_training_e2e.py`) 재실행 —
전부 `progress_callback`/`should_stop`을 넘기지 않으므로 Phase 4H와
수치까지 완전히 동일해야 한다. `run_imagefolder_training_e2e.py`는
TorchScript export + C++ CPU/CUDA parity까지 포함해 재확인.

---

## 15. 구현 순서 (작은 단계, 실제로 이 순서대로 적용됨)

> 아래 목록은 설계 당시 세운 계획이자, 실제 구현이 그대로 따른 순서의
> 기록이다 — 구현 완료 후 다시 확인한 결과 이 순서와 다르게 진행된
> 단계는 없었다.

1. `loop.py`에 `TrainingProgress`/`TrainingProgressCallback`/
   `ShouldStopCallback` 추가 (다른 코드에서 아직 쓰이지 않는 순수 추가).
2. `TrainingHistory.stopped_by_user` 필드 추가(기본값 `False`).
3. `run_training()` 시그니처에 두 keyword-only 인자 추가, 본문은 아직
   안 바꾼 상태로 "항상 `None`이면 기존과 동일"만 우선 확인(회귀
   0건이어야 함 — 이 시점까지는 실질적으로 아무 동작도 안 바뀜).
4. epoch 루프에 §7 pseudocode 그대로 삽입(learning_rate 캡처, stop 판정,
   callback 호출) + resume 분기에 `stopped_by_user` 리셋 한 줄.
5. §14-1 `test_loop.py` 신규 테스트 16개 작성/통과.
6. `imagefolder_workflow.py`에 두 keyword-only 인자 추가 + 내부
   `run_training()` 호출에 전달(1줄).
7. §14-2/§14-3 신규 테스트 작성/통과.
8. `train_imagefolder.py`를 콜백 기반 실시간 출력으로 전환,
   `stopped_by_user` 출력 추가.
9. §14-4 신규 CLI 테스트 작성/통과.
10. README 갱신(Phase 4I 절 신설).
11. 전체 회귀(§14-5) — `tests/training/` + `tests/scripts/` + 전체
    pytest + 기존 4개 E2E 스크립트 + C++ CPU/CUDA parity.

---

## 16. 위험 요소

- **callback 예외 후 상태 유실**: §5에서 문서화했지만, 실제 사용자가
  "학습을 의도적으로 멈추고 싶어서" callback에서 예외를 던지는 실수를
  할 위험이 있다(`ShouldStopCallback` 자체가 checkpoint를 저장하는
  것은 아니다 — `should_stop`을 쓰면 `run_training()`이 정상적인
  `TrainingResult`를 반환해 상위 workflow가 `checkpoint_out` 등 기존
  artifact 후처리를 계속 수행할 수 있게 될 뿐이다) — docstring/설계
  문서에 "의도적으로 멈추려면 예외 대신 `should_stop`을 쓰라"는 안내를
  명확히 남기는 것 외에 코드 차원의 강제는 하지 않는다(과설계 방지,
  실사용 사례가 아직 없음).
- **stop 지연(리뷰로 완화됐지만 완전히 없어지지는 않음)**: §4/§7에서
  콜백을 `should_stop()` 평가보다 먼저 호출하도록 바꾼 덕분에, **콜백
  안에서 동기적으로 stop 플래그를 세팅하는 UI 패턴**은 지연 없이 같은
  epoch 경계에서 바로 반영된다. 하지만 콜백과 무관하게 완전히
  비동기로(다른 스레드에서 임의 시점에) stop이 요청되는 경우에는
  여전히 epoch 경계에서만 관측되므로 최대 1 epoch만큼 기다려야 할 수
  있다 — 이는 설계상 트레이드오프이며 batch 단위 중단(비목표)을
  도입하지 않는 한 완전히 없앨 수 없다. 문서/CLI 출력에 두 경우의
  차이를 명시해 사용자 기대치를 관리한다.
- **`config.epochs == 1`일 때 `should_stop`이 절대 평가되지 않음**:
  §7 "특수 사례"에서 설계한 그대로지만, 이 API를 처음 쓰는 개발자가
  "should_stop을 줬는데 왜 안 먹히지"라고 오인할 수 있는 지점이다 —
  `run_training()`/`ShouldStopCallback` 관련 docstring에 이 조건을
  명시하고, §14-1의
  `test_run_training_should_stop_never_evaluated_when_config_epochs_is_one`
  로 동작을 고정해 둔다.
- **`stopped_by_user` 리셋 누락 위험**: §9에서 지적한 resume 시 리셋
  코드(`history.stopped_by_user = False`)를 빠뜨리면, 사용자가 이미
  한 번 멈췄다 재개한 학습이 실제로는 끝까지 다 돌았는데도
  `stopped_by_user=True`로 잘못 보고될 수 있다 — §14-1의
  `test_run_training_resume_resets_stopped_by_user_flag`가 이 회귀를
  직접 잡는다.
- **`_REQUIRED_HISTORY_FIELDS`에 실수로 `stopped_by_user`를 추가하는
  위험**: 리뷰 시 반드시 확인해야 할 지점 — 추가하면 옛 checkpoint가
  전부 깨진다. §9에서 "추가하지 않는 것이 정답"이라고 명시적으로
  못박아 둔다.
- **`threading.Event`를 실제로 넘겨 쓸 때의 스레드 안전성**:
  `threading.Event.is_set()`은 표준 라이브러리 문서상 여러 스레드에서
  동시에 호출해도 안전하다고 보장된다 — Phase 4I core API가 이를
  직접 구현하지 않으므로(§6에서 타입으로 강제하지 않기로 결정) 이
  안전성은 core의 책임이 아니라 "그런 객체를 넘긴 호출자"의 책임이며,
  표준 라이브러리가 이미 보장하는 성질이라 추가 검증이 필요 없다.
- **CLI 실시간 출력 전환 자체의 회귀**: §11에서 분석한 대로 기존
  CLI 테스트는 stdout을 검사하지 않아 이 위험은 낮지만, §14-4의 신규
  테스트로 "매 epoch마다 정확히 한 줄씩 나온다"를 최소한으로
  확인한다.

---

## 17. 회귀 불변조건 재확인 (사용자 제시 목록 그대로)

```text
- callback=None, should_stop=None이면 기존 학습 수치와 tensor 결과 동일
  -> §14-1 항목 1/8, §15 3번 단계에서 "본문 변경 전 시그니처만 추가"로
     이미 이 불변조건을 별도로 검증
- Phase 4F exact checkpoint/resume 유지
  -> checkpoint.py/config.py 무변경(§13), loop.py의 기존 resume 분기
     로직도 stopped_by_user 리셋 한 줄 외에는 무변경
- Phase 4G ImageFolder metadata 검증 유지
  -> imagefolder_resume.py 무변경(§13)
- Phase 4H production CLI/workflow/E2E 책임 분리 유지
  -> E2E는 callback/stop 미사용(§11), workflow는 그대로 전달만(§12).
     CLI는 fresh 학습의 출력 형식은 유지하되, resume에서는 누적
     history 전체를 다시 출력하던 기존 방식 대신 이번 호출에서 새로
     완료된 epoch만 global epoch 번호로 실시간 출력하도록 의도적으로
     바뀐다(§11) -- production CLI/workflow/E2E의 책임 경계 자체는
     그대로 유지됨
- current model checkpoint 저장 시점 유지
  -> imagefolder_workflow.py의 checkpoint 저장 코드(:213-226) 무변경
- CPU RNG 복원 직후 즉시 run_training() 호출 구조 유지
  -> imagefolder_workflow.py :185-190 두 줄 사이에 아무것도 추가하지
     않음(§12에서 명시 확인). should_stop 평가는 run_training() 내부,
     epoch 루프 안에서만 일어나므로 이 두 줄 사이와는 애초에 무관한
     위치다
- 기존 ImageFolder E2E 3+2 epoch anchor 수치 유지
  -> E2E가 callback/stop을 넘기지 않으므로 §14-5에서 수치 동일 확인
- TorchScript export 및 C++ CPU/CUDA parity 유지
  -> 동일한 이유로 E2E 무변경, §14-5에서 재확인
```

---

## 18. 미결정 사항 (구현 완료 후 실제 결정 기록)

구현 전에는 아래 4가지가 미결정이었다. 실제 구현 시점에 다음과 같이
확정했다:

1. **`learning_rate_next` 필드** — 추가하지 않았다(§3/§4 결정 그대로
   유지). `TrainingProgress.learning_rate`는 여전히 "이번 epoch가 실제로
   사용한 값" 하나뿐이다.
2. **CLI의 `stopped_by_user` 출력 문구** — 상단 요약에 `stopped_early`와
   나란히 한 줄만 추가하는 것으로 정했다(`  stopped_by_user={history.stopped_by_user}`,
   `scripts/train_imagefolder.py`의 `main()`). checkpoint 안내 옆에
   `stopped_early`처럼 별도 note를 붙이지는 않았다 -- `stopped_by_user`는
   resume이 그대로 가능하므로(`stopped_early`처럼 "이 checkpoint로는 더
   이상 resume할 수 없다"는 경고를 붙일 이유 자체가 없다) 대칭적인
   문구가 애초에 불필요했다.
3. **§14-1 테스트 개수** — 설계에서 제시한 16개 항목을 실제로는
   `tests/training/test_loop.py`에 신규 테스트 함수 14개로 구현했다
   (일부 항목을 자연스럽게 병합; 예를 들어 "callback이 should_stop보다
   먼저 호출된다"와 "콜백 안에서 세팅한 stop 플래그가 즉시 반영된다"는
   하나의 테스트로 동시에 검증된다). 최종적으로 `tests/training/test_loop.py`
   52 passed(기존 38 + 신규 14), `tests/training/test_imagefolder_workflow.py`
   13 passed(기존 9 + 신규 4), `tests/scripts/test_train_imagefolder_cli.py`
   8 passed(기존 6 + 신규 2) -- 모두 §17의 회귀 불변조건을 깨지 않았다.
4. **`run_epoch`/`global_epoch`/`total_run_epochs` 이름** — 그대로
   채택했다. 세 이름 모두 `TrainingProgress`의 실제 필드명이자
   구현/테스트 전체에서 일관되게 쓰인다.

---

## 19. 향후 확장과의 연결 (짧게)

- **epoch별 자동 checkpoint(리뷰 반영 — 이번 Phase의 progress callback
  만으로는 구현할 수 없음을 명시)**: `save_training_checkpoint()`가
  요구하는 `model`(현재 가중치), `training_result`(사실은
  `optimizer_state_dict`/`scheduler_state_dict`/`epochs_without_improvement`
  까지 포함하는 `TrainingResult` 전체), `loader_generator_state`,
  `cpu_rng_state`를 `TrainingProgress`는 **하나도 담고 있지 않다**(§3
  — 의도적으로 경로/지표만 담도록 설계했으며, 이번 리뷰로도 이
  원칙을 지켰다). `optimizer`/`scheduler`는 애초에 `run_training()`의
  지역 변수라 callback 바깥에서 접근할 방법이 없고, DataLoader
  generator state/CPU RNG state는 지금 `imagefolder_workflow.py`가
  **학습이 전부 끝난 뒤** 한 번만 채취한다(`:194-195`) — epoch 경계마다
  채취하도록 만들어져 있지 않다. 즉 **Phase 4I의 `progress_callback`은
  관찰/UI 갱신 전용이고, 그 안에서 완전한(정확히 재개 가능한) checkpoint를
  저장할 수 없다.** epoch별 자동 checkpoint를 실제로 구현하려면 다음
  Phase에서 별도로 설계해야 하며, 가능한 방향은 다음 둘 중 하나다:
  - `run_training()` 내부에 **epoch-end checkpoint hook**을 새로
    추가(콜백이 아니라 `run_training()`이 스스로
    `model`/`optimizer`/`scheduler`/`history`/`best_state_dict`를 알고
    있는 지점에서 저장을 수행하거나, 그 정보를 온전히 담은 별도
    콜백/훅을 새로 정의), 또는
  - `model`/`optimizer`/`scheduler`/`history`/`best_state_dict`를 포함하는
    **내부 snapshot API**를 별도로 설계하고, `loader_generator_state`/
    `cpu_rng_state`를 **같은 epoch 경계**에서 함께 채취하도록
    `imagefolder_workflow.py`(또는 그 대체)와 연동하는 방식.

  두 방향 모두 이번 Phase 범위 밖이며, `TrainingProgress`에
  model/state_dict를 추가해 억지로 해결하지 않는다(§3의 "model/state_dict
  제외" 결정을 그대로 유지).
- **SIGINT graceful stop**: `signal.signal(SIGINT, handler)`에서
  `handler`가 `stop_event.set()`을 호출하고 `should_stop=stop_event.is_set`을
  넘기면 자연스럽게 연결된다 — 이번 Phase는 이 배선 자체를 구현하지
  않지만 API가 이미 이를 지원하는 형태다.
- **GUI/background worker**: GUI 스레드가 학습 스레드에
  `threading.Event`를 공유해 `should_stop=stop_event.is_set`,
  `progress_callback=`으로 Qt 시그널을 방출하는 함수를 넘기는 방식이
  바로 이 API가 상정한 사용 시나리오다.
- **ETA/TensorBoard/W&B**: 전부 `progress_callback`이 매 epoch 받는
  `TrainingProgress`를 소비하는 외부 코드에서 구현 가능 — core는
  이들에 대해 전혀 알 필요가 없다(콜백 패턴의 핵심 이점).

이 설계는 §1의 실제 코드 분석(구현 전 기준, 함수명/처리 순서 중심)에
근거하며, 사용자가 제시한 예시 API와 방향을 대부분 그대로 채택하되,
epoch 번호 필드명(`completed_epochs` → `global_epoch`)과 학습률 필드
(단일 `learning_rate`로 확정) 두 가지에서 근거를 들어 대안을
제시했다. 최초 초안에 대한 리뷰를 반영해 (1) `progress_callback`을
`should_stop()` 평가보다 먼저 호출하도록 순서를 바꾸고 그 결과로
`TrainingProgress.stop_requested` 필드를 제거했으며(§3/§4/§7), (2)
마지막 요청 epoch에서의 stop 요청은 `stopped_by_user`로 기록하지
않도록 정책을 좁혔고(§7/§8/§14), (3) progress callback만으로는
완전한 epoch별 자동 checkpoint를 구현할 수 없음을 명시했다(§19).

**최종 설계는 §15의 순서에 따라 구현됐으며, 전체 단위 테스트와 기존
E2E, TorchScript export 및 C++ CPU/CUDA parity 검증을 모두 통과했다**
(문서 상단 "구현 결과 요약"과 §14 참고 — §15 자체는 실제로 적용된
구현 순서의 기록으로 그대로 유지한다).
