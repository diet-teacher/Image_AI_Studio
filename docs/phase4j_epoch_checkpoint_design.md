# Phase 4J: Epoch-end Automatic Checkpointing and Recovery — 설계안

**상태: 설계 확정 및 구현 완료.** `run_training()`이 완료된 epoch
경계마다 exact-resume 가능한 checkpoint를 안전하게(원자적으로) 저장할
수 있도록 Phase 4F(checkpoint/resume) + Phase 4I(progress callback/
cooperative stop)를 확장한 설계이며, 이 문서가 정의한 정책이 production
코드(`loop.py`/`checkpoint.py`/`imagefolder_resume.py`/
`imagefolder_workflow.py`/`scripts/train_imagefolder.py`/
`scripts/run_imagefolder_training_e2e.py`)와 회귀 테스트에 그대로
반영되어 있다.

**전제(Phase 4I §19에서 이미 예고됨)**: Phase 4I 설계 문서는 이 확장을
"다음 Phase에서 별도로 설계"하도록 명시적으로 미뤄뒀고, 가능한 방향을
두 가지로 좁혀뒀다 — (1) `run_training()` 내부에 epoch-end checkpoint
hook 추가, (2) model/optimizer/scheduler/history/best_state_dict를
포함하는 별도 snapshot API. 이 문서는 저장소를 직접 조사한 뒤 (1)을
채택하고 그 구체적인 형태를 정의한다.

**리뷰 1차 반영**: `EpochCheckpointSnapshot`이 암시하던 "독립 복사본"
인상을 지우고 이름/계약을 "hook 호출 중에만 유효한 뷰"로 명시, 자동
저장 기본값을 off로 통일, 저장 주기를 global epoch 기준으로 재정의,
"callback 예외가 나도 항상 저장된다"는 주장을 scheduled/non-scheduled
epoch로 정확히 좁힘, `stopped_by_user`는 mid-loop 자동 저장에서 항상
`False`임을 명시, metadata를 checkpoint보다 먼저 준비하는 순서로 변경.

**리뷰 2차 반영**: 최종(post-hoc) 저장도 metadata-first로 통일하고
mid-loop hook과 공유하는 `_ensure_checkpoint_metadata()` helper 도입,
`EpochCheckpointView`가 optimizer/scheduler/generator의 살아있는
참조만 담도록 재설계(비용은 scheduled epoch에서만 발생), metadata
sidecar를 fresh(무조건 현재 상태로 교체)와 resume(이미 검증된 sidecar
재사용)으로 분리, scheduled epoch와 최종 epoch가 겹치면 중복 저장을
허용하기로 결정, CLI/workflow 테스트 책임 분리, `checkpoint_every`
입력 검증 확정.

**리뷰 3차 반영**: `checkpoint_every` 검증을 `config.py`의 private
helper 재사용에서 `imagefolder_workflow.py` 자체의 private validator로
교체, fresh 실행의 경로 재사용 한계를 정직하게 재분석.

**리뷰 4차 반영(이 버전에서 수정, 정책 전환)**: 3차 리뷰가 남겨둔
"fresh가 metadata를 무조건 교체"하는 정책 자체에 **실제로 성립하지
않는 안전 주장**이 있었음을 발견했다 — "dataset이 다르면 이후 resume이
일반적으로 명확히 거부된다"는 설명이 틀렸다(§7-3에서 정확히 분석).
metadata 검증은 "디스크의 저장된 metadata" vs "지금 다시 계산한
metadata"를 비교하는데, fresh가 이미 저장된 metadata를 **현재 학습
기준으로 먼저 덮어써버리면** 그 비교는 사실상 "현재 값 vs 현재 값"이
되어 항상 통과한다 — 그 밑에 깔린 checkpoint.pt가 실제로 어떤
dataset에서 만들어졌는지는 전혀 검증되지 않는다. 이 문제를 근본적으로
막기 위해, 이번 개정은 **"fresh(또는 다른 경로에서의 resume)가 기존
`checkpoint_out`/sidecar를 덮어쓰는 것 자체를 금지"**하는 정책으로
전환했다 — content-fingerprint 기반의 "다르면 거부, 같으면 통과"라는
느슨한 검증 대신, "기존 파일이 있으면 무조건 거부, 오직 정확히
`resume_from == checkpoint_out`인 in-place resume만 기존 경로를 갱신할
수 있다"는 훨씬 단순하고 완전한 규칙으로 대체했다(§2/§6/§7/§8/§10/
§11/§12/§13/§14/§16/§17/§18/§19 전면 갱신). 이 정책 전환에 따라
`scripts/run_imagefolder_training_e2e.py`도 더 이상 "무수정"으로
남을 수 없다는 것을 확인해 §11-5/§13에 반영했다(재사용 중인
`checkpoint_out` 경로를 그대로 두면 두 번째 실행부터 실패한다).

**리뷰 5차 반영**: `progress_callback`/`should_stop`/`checkpoint_hook`이
학습 RNG를 소비하거나 model/optimizer/scheduler/DataLoader generator를
변형하면 exact-resume이 깨질 수 있다는 것을 확인하고, 세 API 각각에
RNG-purity/비변형 계약을 명시(§3-5) — exact-resume 보장 범위를
"built-in hook + 계약을 지키는 callback" 조합으로 정확히 좁힘(§5/§9/§17).

**리뷰 6차 반영(이 버전에서 수정)**: `_ensure_checkpoint_metadata()`가
scheduled epoch마다 반복 호출되며 매번 같은 metadata를 다시 쓰던
비효율을 바로잡았다 — metadata는 workflow 실행(한 번의
`run_imagefolder_training_workflow()` 호출)당 최대 한 번만 준비하고,
scheduled hook과 최종 저장이 `metadata_ready`라는 **하나의 공유
상태**를 통해 그 사실을 함께 추적하도록 재설계했다(§7-3/§11-3).

---

## 1. 현재 구조 분석

### 1-1. checkpoint가 저장되는 시점(현재, Phase 4F부터 변경 없음)

`run_training()`은 **모든 epoch를 마친 뒤에야** `TrainingResult`를
반환하고, 파일을 전혀 쓰지 않는다. Phase 4F 설계 문서
(`docs/phase4f_checkpoint_resume_design.md` §5)는 이걸 명시적인 정책으로
못박아뒀다:

> "지원하지 않음: 단일 run_training() 호출 내부에서 매 epoch 자동 저장,
> epoch callback, 프로그램 강제 종료 직전 자동 저장, batch 중간 저장.
> 긴 학습에서 주기적 checkpoint가 필요하면, caller가 학습을 여러
> chunk로 나눠 run_training()을 반복 호출하고 매 호출 종료 후 저장하는
> 방식을 쓴다."

Phase 4J는 정확히 이 배제 목록의 첫 항목("매 epoch 자동 저장")을
다시 여는 작업이다.

`imagefolder_workflow.py`(production 경로)의 실제 저장 흐름은 다음과
같다: `run_training(...)` 호출이 반환한 뒤 → RNG snapshot 캡처 →
(`checkpoint_out`이 주어졌으면) `save_training_checkpoint()` 1회 호출 →
이어서 `save_imagefolder_resume_metadata()`로 metadata sidecar 1회
저장(순서: checkpoint 먼저, metadata 나중). 즉 지금은 학습 전체에
걸쳐 정확히 한 번만 저장 가능하고, `run_training()`이 예외 없이
정상 반환해야만 도달하는 코드다.

`_prepare_resume()`(resume 분기)이 fresh 분기와 다르게 동작하는 것도
확인해뒀다 — `resume_from is None`(fresh)이면 metadata를 **전혀 읽지도
검증하지도 않는다**. 즉 **오늘은 fresh 학습이 `checkpoint_out` 경로를
재사용할 때, 그 경로에 남아있던 이전(무관할 수도 있는) checkpoint/
metadata를 검증 없이 그냥 학습 종료 후 덮어쓴다.** 이번 개정
(§7-3/§10)은 이 기존 동작 자체가 Phase 4J의 metadata-first 순서와
결합했을 때 안전하지 않다고 결론짓고, fresh의 경로 재사용을 금지하는
쪽으로 정책을 바꾼다 — 즉 이 부분은 **의도적인 기존 동작 변경**이다
(§12에서 하위 호환 영향을 정직하게 설명한다).

### 1-2. 현재 epoch 처리 순서 (Phase 4I 적용 후, `loop.py`의 `run_training()`)

```text
train_one_epoch()
→ evaluate()
→ history.train_losses/val_losses/val_accuracies 기록
→ best_val_loss/best_state_dict/epochs_without_improvement 갱신
→ scheduler.step(val_loss)  [scheduler 있으면]
→ early stopping 조건 확인 (만족하면 history.stopped_early = True)
→ progress_callback 호출 (있으면)
→ should_stop 평가 (early stopping 아니고, 다음 epoch가 남아 있을 때만)
→ (stopped_early 또는 stopped_by_user면) break, 아니면 다음 epoch
```

이 순서는 Phase 4I가 이미 확정한 것으로, 이번 Phase는 **이 순서
자체를 바꾸지 않는다** — 새 hook을 어디에 끼워 넣을지만 결정한다(§3).

### 1-3. checkpoint 저장에 필요한 모든 상태는 이미 `run_training()`
내부에서 접근 가능하다 (핵심 발견)

```python
>>> gen = torch.Generator().manual_seed(0)
>>> loader = DataLoader(ds, batch_size=4, shuffle=True, generator=gen)
>>> loader.generator is gen
True
```

`torch.utils.data.DataLoader.__init__`은 생성자에 넘긴 `generator`를
`self.generator`에 그대로 보관한다. 이 저장소의 모든 학습 경로는
`train_loader`를 만들 때 항상 명시적으로 `generator=`를 넘긴다 — 즉
`run_training()`이 이미 파라미터로 받는 `train_loader` 객체 자체가
그 generator의 살아있는 참조를 갖고 있다. CPU RNG는
`torch.get_rng_state()`로 어디서든 읽기 전용으로 접근 가능하다.
`model.state_dict()`/`optimizer.state_dict()`/`scheduler.state_dict()`도
전부 읽기 전용이며 RNG를 소비하지 않는다.

**결론**: `run_training()`은 오늘 이미 `model`/`optimizer`/`scheduler`/
`history`/`best_state_dict`/`epochs_without_improvement`(지역 변수)와
`train_loader.generator`를 전부 갖고 있다. 이 사실이 §5의 아키텍처
선택과 §4의 view 설계를 좌우한다.

`run_training()`이 구조적으로 알 수 없는 것은 딱 두 가지뿐이다: (1)
어디에 저장할지, (2) ImageFolder 전용 metadata. 이 둘은 앞으로도
`imagefolder_workflow.py`의 책임으로 남아야 한다.

### 1-4. checkpoint 저장이 현재 원자적이지 않다 (별도 발견, §7에서 다룸)

`checkpoint.py`의 `save_training_checkpoint()`와 `imagefolder_resume.py`의
`save_imagefolder_resume_metadata()` 둘 다 임시 파일이나
`os.replace()` 없이 대상 경로에 바로 쓴다. 저장소 전체에서
원자적 쓰기 패턴을 검색했지만 어디에도 없다. Phase 4J로 저장 빈도가
늘어나면 이 위험의 발생 빈도도 함께 늘어나므로, 원자적 쓰기는 이번
Phase에서 선택이 아니라 필수다(§7).

### 1-5. metadata 검증이 실제로 무엇을 비교하는지 (핵심 재확인, 4차 리뷰의 근거)

`require_compatible_imagefolder_resume_metadata(saved, current)`은
**디스크에 저장된 metadata**와 **지금 이 순간 ModelSpec/dataset으로부터
다시 계산한 metadata**를 비교한다. 이 함수는 `checkpoint.pt`의 실제
내용을 전혀 들여다보지 않는다 — checkpoint.pt가 정말로 `saved`
metadata가 기록됐던 그 저장 이벤트에서 나온 것인지 증명할 방법이
애초에 없다. 그래서 **"먼저 saved를 지금 값으로 덮어쓴 뒤 나중에
검증한다"는 순서 자체가 검증을 무력화한다** — saved가 이미 current와
같아진 뒤이므로 비교는 항상 통과하고, checkpoint.pt가 실제로 무엇인지는
전혀 확인되지 않는다. 이 사실이 §7-3/§8/§10/§11의 정책 전환 전체의
근거다.

### 1-6. 관련 파일 재확인 결과 요약

- `src/image_ai_studio/training/history.py`/`config.py`: Phase 4J는
  이 두 파일을 수정하지 않는다(§13). `checkpoint_every` 검증은
  `imagefolder_workflow.py` 자체의 private validator로 둔다(§11-2).
- `src/image_ai_studio/training/imagefolder_resume.py`: metadata
  내용은 학습 도중 절대 바뀌지 않는다 — 한 `run_training()` 호출
  내내 ModelSpec도 dataset도 그대로이기 때문이다.
- `tests/training/test_checkpoint.py`/`test_loop.py`/
  `test_imagefolder_resume.py`/`test_imagefolder_workflow.py`/
  `tests/scripts/test_train_imagefolder_cli.py`: 기존 monkeypatch/spy
  패턴을 그대로 재사용한다(§14).
- `scripts/run_imagefolder_training_e2e.py`: **이번 정책 전환으로 더
  이상 무수정으로 남을 수 없다** — 이 스크립트는 고정 경로
  `CHECKPOINT_PATH`에 fresh 학습(stage 1)을 매번 다시 저장한다. 새
  정책에서는 그 경로에 이전 실행의 checkpoint/metadata가 이미 있으면
  stage 1이 즉시 `ValueError`로 실패한다 — 이 스크립트를 두 번째부터
  재실행 가능하게 유지하려면 stage 1 직전에 그 경로를 정리(삭제)하는
  아주 작은 변경이 필요하다(§11-5/§13에서 상세).

---

## 2. 목표/비목표

### 목표

1. `run_training()`에 epoch 경계 checkpoint 저장 hook을 추가한다 —
   저장에 필요한 상태(model 포함)에 접근할 수 있는 별도 타입으로, 단
   실제 저장 여부를 결정하기 전까지는 추가 계산 비용을 강제하지
   않는다(§4).
2. `imagefolder_workflow.py`가 이 hook을 이용해 학습 도중 사용자가
   명시적으로 켠 주기(`checkpoint_every`)마다 `checkpoint_out`을
   exact-resume 가능한 상태로 계속 갱신한다.
3. checkpoint(및 metadata sidecar) 저장을 원자적으로 만든다 — 기존
   Phase 4F의 1회성 저장 경로도 포함해서, 저장 순서를 metadata
   먼저로 통일한다.
4. 학습이 예외로 중단되거나 프로세스가 크래시해도, 디스크에 남은
   마지막 checkpoint는 항상 "완전히 완료된 어떤 scheduled epoch"의
   정확한 상태를 나타낸다 — 손상되거나 반쯤 쓰인 파일이 아니다.
5. **`checkpoint_out`이 가리키는 경로에 이미 유효한 checkpoint/metadata가
   있으면, 오직 `resume_from == checkpoint_out`인 in-place resume만
   그 경로를 갱신할 수 있다.** fresh 실행이나 다른 경로에서의 resume이
   기존 `checkpoint_out`(또는 그 metadata sidecar)을 발견하면 학습을
   시작하지 않고 명확한 `ValueError`로 거부한다 — §1-5에서 확인했듯
   "다시 쓴 뒤 검증"으로는 checkpoint/metadata 쌍의 정합성을 보장할
   수 없으므로, "기존 파일이 있으면 애초에 건드리지 않는다"는 훨씬
   단순하고 완전한 규칙으로 이를 대체한다.
6. 기존 `--checkpoint-out`의 "그 경로 하나"라는 의미와 새 자동 저장
   기능의 기본 off는 유지하되, 목표 5의 안전성 요구가 기존 fresh
   재사용 동작보다 우선한다는 것을 명확히 한다(§12).

### 비목표 (사용자 제시 목록 그대로)

GUI 구현, background worker/thread 구현, batch 중간 checkpoint/resume,
multi-worker DataLoader exact resume, distributed training, AMP/CUDA
training 추가, cloud/object storage, checkpoint retention/rotation
개수 관리, SIGINT/KeyboardInterrupt graceful recovery, TensorBoard/W&B.
이번 개정에서 추가로 확정한 비목표: `--overwrite-checkpoint` 같은
명시적 덮어쓰기 플래그, 자동 파일 삭제, checkpoint/metadata pair ID
도입, checkpoint 포맷 버전 변경 — 전부 이번 Phase에서 다루지 않는다
(§19).

---

## 3. epoch checkpoint 경계

(1~2차 리뷰에서 확정, 이번 개정에서 변경 없음)

### 3-1. 정확한 삽입 지점

```text
train → validate → history 기록 → best/카운터 갱신 → scheduler.step()
→ early stopping 판정 → checkpoint_hook (신규) → progress_callback
→ should_stop 평가 → break 판정
```

`checkpoint_hook`은 완료된 epoch마다 항상 호출된다(있으면). 호출
자체는 저렴하고(§4), 무거운 계산은 hook이 scheduled epoch로 판단한
뒤에만 발생한다. "hook 호출 여부"와 "실제 저장 여부"는 서로 다른
질문이다.

`checkpoint_hook`을 `progress_callback`/`should_stop`보다 먼저 실행하는
이유: 둘 다 사용자 callable이라 예외를 던질 수 있는데, hook을 먼저
실행하면 scheduled epoch에서는 그 예외가 나더라도 이번 epoch까지의
상태가 이미 디스크에 저장된 뒤가 된다. `checkpoint_every=1`일 때만
이 보장이 "모든 epoch"로 강해진다.

### 3-2. `scheduler.step()`과의 관계

`checkpoint_hook`은 `scheduler.step()` 이후에 실행된다.

### 3-3. `should_stop` 평가 이전/이후

`checkpoint_hook`은 `should_stop` 평가 이전에 실행되므로, scheduled
epoch에서 저장되는 `stopped_by_user`는 항상 `False`다 — 정확한 값은
학습 종료 후 기존 최종 저장이 채운다(§9-3). `stopped_early`는 이
hook보다 먼저 결정되므로 항상 정확한 값이 저장된다.

### 3-4. "마지막 epoch"과 최종 저장의 관계 (값 기준 동등성, 조건부)

hook 실행 지점과 `run_training()` 반환 지점 사이의 **core 코드
자체**(`scheduler` 관련 마무리, `optimizer.state_dict()`/
`copy.deepcopy()` 등)는 RNG를 소비하지 않는다. 하지만 그 사이에는
`progress_callback`/`should_stop`(사용자가 넘긴 임의의 callable)도
실행된다 — 이 값 기준 동등성 주장은 **§3-5의 계약을 그 callable들이
지킨다는 전제 위에서만** 성립한다. 그 전제가 성립하면(§3-5), 마지막
epoch이 scheduled epoch였고 `should_stop`이 그 사이 stop을 결정하지
않았다면 두 저장은 값 기준으로 동일하다(파일 byte-for-byte 동일은
주장하지 않는다). 마지막 epoch이 scheduled가 아니었다면 최종 저장이
간극을 메우고, scheduled와 겹치면 §9-4의 중복 저장이 발생한다.

### 3-5. `progress_callback`/`should_stop`/`checkpoint_hook`의 RNG-purity 및
비변형 계약 (신규, exact-resume 보장의 전제 조건)

**문제**: `checkpoint_hook`은 `progress_callback`/`should_stop`보다
먼저 실행되므로(§3-1) scheduled epoch의 checkpoint는 이 두 callable이
실행되기 **전**의 model/optimizer/scheduler/history/CPU RNG/
DataLoader generator 상태를 저장한다. 하지만 이 둘은 Python
callable이므로, closure나 전역 참조를 통해 기술적으로
`torch.rand()`/`torch.randn()` 같은 RNG 소비 연산을 호출하거나,
model 파라미터를 바꾸거나, optimizer/scheduler 상태를 바꾸거나,
`loader_generator.set_state()`/`.manual_seed()`로 DataLoader
generator를 바꿀 수 있다 — core는 임의 callable의 부수효과를 막을
방법이 없다. 예를 들어 `progress_callback`이 매 epoch `torch.rand()`를
한 번 호출하면, checkpoint에 저장된 CPU RNG state는 그 callback
실행 이후 실제로 다음 epoch가 시작할 때의 RNG state와 달라지고,
그 checkpoint에서 resume한 다음 epoch의 Dropout 등 확률적 연산
결과가 continuous run과 어긋난다.

**정책**: 이 문제를 core 코드로 막는 대신, 세 API 각각에 **명시적
계약**을 두고 문서화(docstring + 이 설계 문서)한다 — Python 타입
시스템으로 완전히 강제할 수는 없으므로, `imagefolder_workflow.py`가
제공하는 built-in 구현(`_print_progress()`, `_make_checkpoint_hook()`)이
이 계약을 지키도록 구현하고 코드 리뷰로 확인하는 것으로 확정한다
(§14/§16).

```text
progress_callback:
  - 진행 상황 관찰, 출력, UI 전달 전용
  - PyTorch CPU RNG를 소비하면 안 됨(torch.rand()/torch.randn() 등 호출 금지)
  - model/optimizer/scheduler/DataLoader generator를 변경하면 안 됨
  - closure나 전역 변수로 이 객체들에 접근할 수 있더라도 변경하지 않아야 함

should_stop:
  - 외부 stop flag를 읽어 bool을 반환하는 용도
  - PyTorch RNG를 소비하면 안 됨
  - model/optimizer/scheduler/DataLoader generator를 변경하면 안 됨
  - 반환값 외의 학습 상태 side effect를 만들면 안 됨

checkpoint_hook(§4의 EpochCheckpointView 계약과 통합):
  - view의 상태를 읽고 동기적으로 저장하는 용도
  - model/optimizer/scheduler/loader_generator를 변경하면 안 됨
  - torch.get_rng_state()/generator.get_state()/.state_dict() 같은
    읽기 전용 호출만 허용(이들 자체는 RNG를 소비하지 않음, §1-3)
  - hook 내부에서 학습에 사용되는 RNG를 소비하는 다른 연산을 호출하면 안 됨
```

**이 계약을 위반한 사용자 정의 callback/hook으로 인한 비결정성은
core의 버그가 아니라 caller의 계약 위반으로 정의한다** — §5/§17의
exact-resume 보장 범위를 이 계약을 지키는 경우로 좁히는 근거다.

---

## 4. 상태 View 설계

(1~2차 리뷰에서 확정, 이번 개정에서 변경 없음)

### 4-1. 소유권/수명 계약과 non-scheduled epoch 비용

`EpochCheckpointView`는 model/history/optimizer/scheduler/
loader_generator의 **살아있는 참조**만 담는다 — `.state_dict()`/RNG
읽기는 hook이 scheduled epoch로 판단한 뒤에만 수행한다:

```python
@dataclass(frozen=True)
class EpochCheckpointView:
    """checkpoint_hook 호출 동안만 유효한 읽기 전용 뷰(synchronous
    ephemeral view) -- 독립 snapshot이 아니다.

    model/history/optimizer/scheduler/loader_generator는 전부
    run_training() 내부의 살아있는 참조다. hook은 이 view가 유효한
    동안(자기 자신의 동기 호출 범위 안)에서 필요한 조회와 직렬화를
    전부 끝내야 하고, 이 view나 그 어떤 참조도 나중에 쓰려고
    보관하면 안 된다. 비동기/백그라운드 저장에는 쓸 수 없다(§19).

    optimizer/scheduler/loader_generator는 읽기 전용으로만 접근해야
    한다 -- 이 객체들을 변형하면(특히 loader_generator) exact-resume이
    깨진다. 이 hook은 또한 학습에 사용되는 RNG를 소비해서도 안 된다
    (torch.rand()/torch.randn() 등 호출 금지) -- .state_dict()/
    .get_state()/torch.get_rng_state() 같은 읽기 전용 호출 자체는
    RNG를 소비하지 않으므로(§1-3) 허용되지만, 그 밖의 RNG 소비
    연산은 다음 epoch가 시작할 때의 실제 상태와 checkpoint에 저장된
    상태를 어긋나게 만든다(§3-5, exact-resume 계약의 일부).

    best_state_dict/epochs_without_improvement는 run_training()이
    이미 매 epoch 유지하는 값이라 담는 데 추가 비용이 없다.
    """

    model: nn.Module
    history: TrainingHistory
    best_state_dict: dict[str, torch.Tensor]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None
    epochs_without_improvement: int
    loader_generator: torch.Generator | None


CheckpointHook = Callable[[EpochCheckpointView], None]
```

non-scheduled epoch에서는 view 생성(참조 전달뿐, 비용 무시할 수준)
외에 `optimizer.state_dict()`/`scheduler.state_dict()`/
`torch.get_rng_state()`/`loader_generator.get_state()` 중 어느 것도
호출되지 않는다 — `checkpoint_every`는 디스크 I/O뿐 아니라 이
조회/RNG 비용까지 scheduled epoch에서만 발생하도록 걸러낸다.

`loader_generator`가 `None`일 수 있는 이유는 `run_training()`이
dataset-agnostic한 core API이기 때문이다 — `imagefolder_workflow.py`의
concrete hook은 이를 명확한 실패로 처리한다(§8-1).

### 4-2. `TrainingResult`를 재사용하지 않는 이유

`TrainingResult`는 학습 종료 후 caller가 독립적으로 소유하는 값이고,
`EpochCheckpointView`는 계속 mutate되는 참조다 — 계약이 다른 두
용도에 같은 타입을 쓰지 않기 위해 평평한(flat) 별도 dataclass로
정의한다.

### 4-3. checkpoint에 필요한 전체 상태 vs 기존 payload

payload 형식 자체는 전혀 바뀌지 않는다. `CHECKPOINT_FORMAT_VERSION`은
그대로 1이다. hook은 scheduled epoch로 판단한 뒤, view의 필드로부터
`TrainingResult`를 새로 조립해 `save_training_checkpoint()`에 넘긴다
(§11) — `checkpoint.py`의 외부 계약은 바뀌지 않는다.

---

## 5. API 후보 비교와 최종 선택

(1차 리뷰에서 확정, 변경 없음)

**최종 선택: 후보 A** — `run_training()`에 epoch-end hook 추가.
후보 B(core가 파일 직접 저장, 계층 위반), C(`progress_callback` 확장,
사용자 지침으로 금지), D(workflow가 epoch마다 재호출, 재구성 비용/
`should_stop` 계약 붕괴)는 기각.

**exact resume 보장의 범위(4차 리뷰에서 정정)**: 후보 A가 exact
resume을 유지한다는 것은 **`checkpoint_hook`이 `progress_callback`/
`should_stop`보다 먼저 실행되고, 그 두 callable이 §3-5의 RNG-purity/
비변형 계약을 지킨다는 전제 위에서** 성립한다 — 임의의 사용자 정의
callback이 RNG를 소비하거나 학습 상태를 변형해도 exact resume이
유지된다고 주장하지 않는다(§3-5/§9/§17에서 이 범위를 일관되게
명시한다).

---

## 6. 저장 파일/경로 정책

### 6-1. 저장 정책 후보 비교

| 후보 | 평가 |
|---|---|
| latest checkpoint를 (사용자가 켰다면) 정해진 주기마다 저장 | **채택** |
| N epoch마다 저장 | 채택 — `checkpoint_every`가 기본 메커니즘 |
| best validation epoch에서만 저장 | 기각(주 정책으로는) — crash 복구 목적에 안 맞음 |
| latest + best 각각 저장 | v1에는 포함하지 않음(§19) |
| 최종 종료 시에만 별도 final checkpoint 저장 | 기존 동작 보존, 저장 순서만 metadata-first로 변경 |

### 6-2. 기본값: 자동 저장은 off

`checkpoint_every=None`이 기본값이다. `checkpoint_out`을 주는 것만으로는
자동 저장이 켜지지 않는다 — 이는 유지된다. 다만 §6-5의 새 정책 때문에,
`checkpoint_every`를 켜지 않아도 **`checkpoint_out`을 준 fresh 실행
자체의 동작이 바뀐다**(기존 경로 재사용이 거부됨) — 이건 자동 저장
기능의 on/off와 무관한, 별도의 안전성 정책 변경이다(§12).

### 6-3. `checkpoint_out` 하나만 지정했을 때 실제 파일명 정책

바뀌지 않는다. `checkpoint_out`이 "latest" checkpoint 파일이고,
metadata sidecar는 `metadata_path_for_checkpoint(checkpoint_out)`이다.

### 6-4. 기존 최종(post-hoc) 저장 코드는 그대로 유지하되, metadata-first로 통일한다

`imagefolder_workflow.py`가 `run_training()` 반환 직후 수행하는 저장은
자동 저장 기능을 켜든 안 켜든 무조건 실행한다. 저장 순서는 §7-3의
metadata-first 절차를 따른다. `should_stop`으로 학습이 중단된 경우,
이 최종 저장이 정확한 `stopped_by_user=True`를 반영하는 유일한
저장이다(§9-3).

### 6-5. 출력 경로 재사용 정책 (신규 결정, 4차 리뷰)

**기존 파일이 존재하는 `checkpoint_out`을 갱신할 수 있는 경우는
`resume_from == checkpoint_out`인 in-place resume뿐이다.**

| 시나리오 | 정책 |
|---|---|
| `resume_from is None`(fresh) + `checkpoint_out`이 가리키는 checkpoint 또는 sidecar가 이미 존재 | **`ValueError`, 학습 시작 전에 거부. 기존 파일은 건드리지 않는다.** |
| `resume_from is None`(fresh) + `checkpoint_out` 경로가 완전히 비어 있음 | 정상 진행, metadata-first로 새로 저장(§7-3) |
| `resume_from is not None`이고 `resume_from == checkpoint_out` | in-place resume. `_prepare_resume()`이 이미 검증한 바로 그 파일 — scheduled/최종 저장이 원자적으로 갱신 가능, metadata는 재작성하지 않음 |
| `resume_from is not None`이고 `resume_from != checkpoint_out` + `checkpoint_out`이 가리키는 checkpoint 또는 sidecar가 이미 존재 | **`ValueError`, 학습 시작 전에 거부.** source(`resume_from`)는 읽기 전용으로 남고, 출력 경로도 건드리지 않는다 |
| `resume_from is not None`이고 `resume_from != checkpoint_out` + `checkpoint_out` 경로가 완전히 비어 있음 | 정상 진행 — resume은 `resume_from`에서, 저장은 새 `checkpoint_out`에, metadata-first로 새로 저장(§7-3) |

**명시적으로 채택하지 않는 것(비목표, §2)**: `--overwrite-checkpoint`
같은 플래그, 자동 삭제, checkpoint/metadata pair ID, 포맷 버전 변경 —
필요성이 확인되면 별도 Phase에서 설계한다.

이 검증은 `_validate_checkpoint_output_paths()`(§11-2/§11-3)가
`run_imagefolder_training_workflow()` 맨 앞, ModelSpec 로드보다도
먼저 수행한다(파일 경로만 보면 되므로 비용이 전혀 들지 않는다).

---

## 7. atomic save 정책

### 7-1. 왜 지금 필요한가

§1-4에서 확인했듯 오늘의 저장은 원자적이지 않다. Phase 4J는 저장
빈도를 (학습 전체 1회) → (사용자가 켰다면 주기적)으로 올리므로, 원자적
쓰기는 필수다(§8).

### 7-2. 절차와 private helper

(1~2차 리뷰에서 확정, 변경 없음)

```python
# checkpoint.py
def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # 실패하면 예외 그대로 전파, 재시도 없음
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass  # 정리 실패가 원래 저장 예외를 가리지 않는다
        raise
```

```python
# imagefolder_resume.py
def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
```

계약: 임시 파일은 목적지와 같은 디렉터리에 `tempfile.mkstemp()`로
고유하게 생성 → `flush()` + `os.fsync()` → `os.replace()`(재시도 없음)
→ 예외 시 임시 파일 삭제 시도(실패해도 원래 예외 우선) → 실패 시
목적지 파일은 전혀 건드려지지 않음.

`save_training_checkpoint()`/`save_imagefolder_resume_metadata()`는
내부적으로 이 helper를 쓰도록 바뀌지만, 외부 시그니처/반환값/파일
포맷은 바뀌지 않는다(§12).

### 7-3. metadata 처리 정책: 출력 경로 검증(§6-5)이 먼저, 그 다음 공유 helper

**이전 개정의 "fresh는 무조건 현재 상태로 (재)교체, resume은 이미
검증된 sidecar 재사용"이라는 정책을 폐기한다.** §1-5에서 확인했듯,
그 정책은 "쓴 뒤에 검증"하는 순서라 checkpoint/metadata 쌍의 정합성을
보장하지 못했다 — saved metadata를 current로 먼저 덮어쓰면 이후의
비교가 항상 통과해버려서, 그 밑의 checkpoint.pt가 실제로 무엇인지는
전혀 확인되지 않는다(구체적 반례: 이전 checkpoint가 Dataset A로
학습됐고, fresh가 sidecar를 Dataset B 기준으로 교체한 뒤 checkpoint
저장에 실패하면, 이후 Dataset B로 resume 시도 시 metadata 검증은
"Dataset B vs Dataset B"라 통과하고, 모델 구조가 같으면
`model.load_state_dict()`도 성공해 **Dataset A의 가중치가 조용히
Dataset B 학습으로 착각되어 로드된다** — dataset이 다름에도 막히지
않는다).

**새 정책: `_validate_checkpoint_output_paths()`(§6-5)가 워크플로우
시작 시 이미 "in-place resume" 또는 "완전히 비어 있는 새 경로" 둘
중 하나임을 보장한다.** 그 보장 위에서 metadata 처리는 단순해진다.

**추가 확정(이번 개정): metadata는 workflow 실행(한 번의
`run_imagefolder_training_workflow()` 호출)당 최대 한 번만 준비한다.**
§1-5의 불변성(한 `run_training()` 호출 내내 ModelSpec/dataset은 절대
바뀌지 않는다)에 따라, "완전히 새 경로"일 때 metadata를 매 scheduled
epoch마다 다시 쓰는 것은 항상 같은 내용을 반복해서 `fsync()`/
`os.replace()`하는 낭비다(예: `checkpoint_every=1`인 fresh 5 epoch
학습이면 5번의 scheduled 저장 + 1번의 최종 저장 = 6번의 checkpoint
저장 기회가 있는데, 그때마다 metadata를 다시 쓸 이유가 없다).
scheduled hook과 최종 저장이 **하나의 공유 상태**를 통해 "이번
workflow 실행에서 metadata를 이미 준비했는지"를 함께 추적한다:

```python
# imagefolder_workflow.py, run_imagefolder_training_workflow() 내부
metadata_ready = _is_in_place_resume(request)  # in-place면 이미 준비된 것으로 시작

def ensure_checkpoint_metadata() -> None:
    """checkpoint_out의 metadata sidecar가 준비된 상태가 되도록 보장한다
    -- 이 workflow 실행 동안 최대 한 번만 실제로 쓴다. 이 함수가 예외
    없이 반환한 뒤에만 checkpoint_out을 갱신해야 한다. scheduled hook과
    최종 저장 양쪽 모두 이 클로저 하나를 공유해서 호출한다(두 곳에
    독립적인 metadata_ready 상태를 따로 두지 않는다).

    이미 준비됐으면(in-place resume이거나 이번 실행에서 한 번 성공했으면)
    아무 것도 하지 않는다. 아니라면 현재 ModelSpec/dataset 기준
    metadata를 원자적으로 쓴다 -- "덮어쓰기"가 아니라 "최초 생성"이다
    (§6-5가 이미 이 경로가 비어있었음을 보장했으므로). 쓰기가
    성공해야만 metadata_ready를 True로 바꾼다 -- 실패하면(예외 전파)
    metadata_ready는 False로 남고, 그러면 caller가 이어서 checkpoint
    저장을 호출하지 않는다(§8).
    """
    nonlocal metadata_ready
    if metadata_ready:
        return
    metadata_path = metadata_path_for_checkpoint(request.checkpoint_out)
    current_metadata = build_imagefolder_resume_metadata(model_spec, splits)
    save_imagefolder_resume_metadata(current_metadata, metadata_path)  # 원자적(§7-2)
    metadata_ready = True
```

`_is_in_place_resume()`은 §11-2에서 정의하는 것과 같은 함수를
재사용한다(중복 정의하지 않음). 실제 구현은
`run_imagefolder_training_workflow()` 내부의 closure(`metadata_ready`/
`ensure_checkpoint_metadata()`)로 정했다 — **핵심은 scheduled hook과
최종 저장이 같은 `metadata_ready` 상태를 공유한다는 것**이다 — 새
클래스나 별도 상태 객체는 만들지 않았다.

- **in-place resume**: `metadata_ready`가 처음부터 `True`이므로 이번
  workflow 실행 동안 metadata를 전혀 다시 쓰지 않는다(`_prepare_resume()`이
  이미 검증했으므로).
- **완전히 새 경로**(fresh 또는 `resume_from != checkpoint_out`):
  이번 실행에서 첫 번째로 실제 checkpoint를 저장하려는 순간(첫
  scheduled epoch가 있으면 거기서, scheduled 저장이 한 번도 없었으면
  최종 저장 직전에) metadata가 정확히 한 번만 원자적으로 생성되고,
  그 뒤로는 다시 쓰지 않는다.

**"완전히 새 경로"이므로 checkpoint 저장이 실패해도 안전한 이유**:
이 경로에는 애초에 아무 파일도 없었음이 `_validate_checkpoint_output_paths()`로
이미 증명됐으므로, metadata write 성공 후 checkpoint write가 실패해도
남는 상태는 "metadata만 있고 checkpoint 없음"뿐이다 — 이는 §7-4에서
설명하듯 이미 잘 알려진, 명확하게 실패하는 경로다(`load_training_checkpoint()`가
파일 없음으로 거부). **"무관한 이전 checkpoint와 새 metadata가
섞이는" 조합 자체가 더 이상 발생할 수 없다** — §6-5가 애초에 그런
상황(기존 파일이 있는 경로 재사용)을 학습 시작 전에 차단하기
때문이다. metadata 쓰기 자체가 실패하면 `metadata_ready`는 `False`로
남고, 그 예외가 그대로 전파되어 checkpoint 저장은 호출조차 되지
않는다(§8).

### 7-4. 한계 (정직하게 명시, "개별 파일 무결성"과 "쌍의 일관성"을 구분)

이 설계는 서로 다른 두 가지 보장을 각각 다른 메커니즘으로 제공한다:

- **개별 파일의 무결성**(잘린/반쯤 쓰인 파일이 되지 않는 것)은
  §7-2의 원자적 교체(`_atomic_torch_save()`/`_atomic_write_text()`)가
  보장한다. **각 개별 atomic save에서 `os.replace()` 이전에 실패하면,
  그 저장 대상 파일의 기존 버전은 보존된다** — 이 이상의 다중 파일
  트랜잭션은 제공하지 않는다(checkpoint 저장과 metadata 저장은
  서로 다른 두 번의 원자적 교체이지, 하나의 트랜잭션이 아니다).
- **checkpoint+metadata 쌍의 일관성**(이 두 파일이 실제로 서로
  대응하는 내용인가)은 §6-5의 **출력 경로 검증**(`_validate_checkpoint_output_paths()`)이
  보장한다 — 기존 파일이 있는 경로는 애초에 건드리지 않으므로, "새
  metadata + 무관한 이전 checkpoint" 같은 잘못된 조합이 만들어질
  가능성 자체가 없다. in-place resume 경로에서는 `_prepare_resume()`의
  사전 검증이 이 역할을 대신한다.

다른 한계:

- **정전/OS 크래시까지의 완전한 내구성은 보장하지 않는다.**
  `os.fsync()`는 OS 페이지 캐시의 flush를 요청할 뿐, 디스크 자체의
  쓰기 캐시/파일시스템 저널링까지 통제하지 않는다.
- **Windows 관련**: `os.replace()`는 POSIX/Windows 양쪽에서 원자적
  이다. 백신/탐색기 미리보기 등이 파일을 일시적으로 잠그면
  `PermissionError`가 날 수 있고, 재시도하지 않고 그대로 전파한다.
- **경로 동일성 판단의 한계**: `_is_in_place_resume()`(§11-2)는
  `Path.resolve()` + `os.path.normcase()` 기반으로 "같은 파일을
  가리키는 경로"를 판단한다(§11-2에서 상세) — 심볼릭 링크가 실제로
  존재하는 경우 `.resolve()`가 그 대상까지 따라가 정확히 판단하지만,
  네트워크 드라이브 매핑처럼 OS/파일시스템 수준에서 별개로 보이는
  두 경로가 실제로는 같은 파일을 가리키는 극단적인 경우까지 완벽히
  잡아내지는 못한다 — 이런 경우는 "다른 경로"로 오인되어
  `checkpoint_out`이 비어있지 않다는 이유로 정상적으로 `ValueError`가
  나므로(안전한 방향의 실패), 데이터 손상으로 이어지지는 않는다.

---

## 8. 저장 실패 정책

### 8-1. 후보 비교

- **A. 예외를 그대로 전파하고 학습 실패 처리 — 채택.** 이 저장소
  전체가 이미 이 원칙을 따른다.
- **B. warning만 출력하고 학습 계속 — 기각.** silent failure 금지
  지침에 걸린다.
- **C. 사용자 설정으로 선택 — 기각.** hook 구현체가 자기 `try/except`로
  이미 커버 가능.

**최종 선택: A.**

### 8-2. scheduled epoch vs non-scheduled epoch에서 정확히 무엇이 실패하는가

저장 관련 예외(`ValueError`(generator None), `save_training_checkpoint()`
실패)는 오직 scheduled epoch에서만 발생할 수 있다. `_validate_checkpoint_output_paths()`가
워크플로우 시작 시 이미 통과했으므로, hook/최종 저장이 시도하는
metadata 쓰기는 §7-3에서 설명한 대로 "최초 생성"뿐이라 그 자체가
실패하는 경우(디스크 가득 참 등)만 남는다 — 예전처럼 "호환성 검증
실패"로 인한 예외는 이제 학습 시작 **전**(§6-5)에서 이미 걸러졌으므로
scheduled epoch 중에는 더 이상 발생하지 않는다.

### 8-3. 실패해도 보장되는 것

**개별 파일 무결성**: scheduled epoch에서 저장이 실패해도 직전에
성공한 checkpoint는 손상되지 않고 그대로 남는다(§7-2/§7-4).
**checkpoint+metadata 쌍의 일관성**: §6-5의 출력 경로 검증 덕분에,
in-place resume이 아닌 모든 경로는 애초에 비어있는 상태에서
시작하므로 "무관한 파일과 섞이는" 실패 모드 자체가 없다. in-place
resume 경로에서 저장이 실패해도, 남는 것은 직전에 성공한(같은 학습의)
scheduled checkpoint뿐이다.

---

## 9. early stopping/user stop/예외 시나리오별 checkpoint

(1~2차 리뷰에서 확정 — §6-5의 출력 경로 검증은 학습 시작 **전**에
끝나는 절차라 이 절이 다루는 "학습 도중" 시나리오와는 독립적이다.
이번 개정에서 추가된 전제: 아래 표는 §3-5의 RNG-purity/비변형
계약을 `progress_callback`/`should_stop`이 지킨다고 가정한다 —
계약을 어기는 callback을 쓰면 "scheduled epoch에서 이미 저장됨"이
값 기준으로 정확하다는 보장이 깨질 수 있다.)

### 9-1. scheduled epoch vs non-scheduled epoch (핵심 계약)

> `checkpoint_every=1`일 때만 모든 완료된 epoch가 `progress_callback`/
> `should_stop` 예외보다 먼저 저장된다. `checkpoint_every>1`이면
> non-scheduled epoch에서 발생한 예외는 그 epoch의 저장 기회를 얻지
> 못하고, 디스크에는 그 이전의 마지막 scheduled epoch까지만 남는다.

| 시나리오 | scheduled epoch | non-scheduled epoch |
|---|---|---|
| `progress_callback` 예외 | hook이 먼저 실행돼 이미 저장된 뒤에 예외 발생 | hook이 곧바로 반환(저장 안 함) → 예외 발생, 디스크는 직전 scheduled epoch까지만 반영 |
| `should_stop` 예외 | 위와 동일 논리 | 위와 동일 논리 |
| `checkpoint_hook` 자신의 저장 예외 | 이 epoch은 저장 안 되고 예외 전파, 직전 scheduled epoch은 보존(§8-3) | 해당 없음 |

### 9-2. 그 외 시나리오

| 시나리오 | hook 호출 여부 | scheduled epoch `stopped_early` | scheduled epoch `stopped_by_user` | 비고 |
|---|---|---|---|---|
| 정상 완료 | 매 epoch마다 | `False` | `False` | 최종 저장(§6-4)이 종료 후 최종 상태 재보장 |
| early stopping | 매 epoch마다(발동 포함) | 발동 epoch이 scheduled면 `True` | `False` | resume 거부 대상 |
| `should_stop` 사용자 중단 | 매 epoch마다 | `False`(§3-3) | 항상 `False` | §9-3 참고 |
| `train_one_epoch`/`evaluate` 예외 | 호출 안 됨 | — | — | 오늘과 동일 |
| `config.epochs == 1` | 정확히 1번 | 배수면 저장 | 위와 동일 | 특수 분기 불필요 |
| 마지막 요청 epoch에서 stop flag 설정 | 정상 호출 | scheduled 여부에 따라 | `False`(Phase 4I 규칙) | hook은 cadence만으로 판단 |

### 9-3. 자동 checkpoint(mid-loop)와 user-stop 이후 최종 checkpoint의 차이

mid-loop 저장은 `stopped_by_user=False`(아직 결정 전), 학습 종료 후
최종 저장이 정확한 `True`를 반영한다. exact-resume 상태는 같은
epoch을 가리킬 수 있지만 종료 사유 metadata는 다를 수 있다.

### 9-4. scheduled epoch와 최종 epoch가 같으면 같은 경로에 두 번 저장될 수 있다 (결정)

중복을 허용한다(생략 최적화는 `stopped_by_user` 정확성 문제와
결합도 증가를 이유로 기각). I/O 낭비일 뿐 정확성 문제는 아니다.

---

## 10. resume 동작

### 10-1. global epoch 번호 / 누적 history / cadence

`EpochCheckpointView.history`는 `run_training()`이 유지하는 같은
객체이므로 `len(view.history.train_losses)`는 global epoch과 항상
일치한다. resume 후에도 저장 주기가 절대 epoch 기준으로 정확히
이어진다 — global epoch 7에서 resume 후 `checkpoint_every=5`면 다음
scheduled epoch은 10이다(12가 아니다).

### 10-2. `stopped_by_user` 리셋

Phase 4I 로직(resume 직후 `False`로 리셋)을 그대로 유지한다.

### 10-3. 새 checkpoint의 metadata / 기존 checkpoint 덮어쓰기 여부 (정책 갱신)

§6-5의 출력 경로 검증이 학습 시작 전에 다음 둘 중 하나만 허용한다:

- **`resume_from == checkpoint_out`(in-place resume)**: `_prepare_resume()`이
  이미 검증한 바로 그 checkpoint/metadata를 그대로 갱신한다 — metadata는
  다시 쓰지 않고, checkpoint만 scheduled epoch마다 원자적으로
  덮어쓴다.
- **`resume_from != checkpoint_out`(다른 경로로 resume 결과를 저장)**:
  `checkpoint_out`(과 그 sidecar)이 이미 존재하면 `ValueError` — 반드시
  비어있는 새 출력 경로여야 한다. `resume_from`이 가리키는 원본
  checkpoint/metadata는 읽기 전용으로만 쓰이고 절대 갱신되지 않는다.
  새 출력 경로의 metadata는 `_prepare_resume()`이 이미 검증한(현재
  ModelSpec/dataset과 일치하는) metadata로 최초 생성된다(§7-3).

기존에 있던 "resume이면 metadata를 재사용, fresh면 무조건 교체"라는
이분법은 폐기됐다 — 이제 실제로 갈리는 기준은 "`resume_from ==
checkpoint_out`인가"뿐이고, fresh와 "다른 경로로의 resume"은 §6-5
관점에서 완전히 동일하게 취급된다(둘 다 "새 경로여야 하며, 있으면
거부").

### 10-4. resume source와 checkpoint output이 같은 경로인 경우 (in-place resume)

이미 오늘 명시적으로 지원/권장되는 사용법이다(Phase 4G/4H CLI 예시
`--resume-from X --checkpoint-out X`). `_prepare_resume()`은
`run_training()` 호출 **전에** 기존 파일 전체를 메모리로 완전히
읽어들이므로, hook이 나중에 같은 경로를 원자적 교체로 갱신해도 이미
메모리에 있는 값과 충돌하지 않는다. §6-5의 새 검증은 이 시나리오를
막지 않는다(정확히 `resume_from == checkpoint_out`인 경우가
in-place resume으로 인식돼 그대로 허용됨, §11-2).

---

## 11. workflow/CLI 연결

### 11-1. `ImageFolderWorkflowRequest`에 새 필드 추가

```python
@dataclass
class ImageFolderWorkflowRequest:
    ...
    checkpoint_every: int | None = None  # None(기본) = 자동 저장 비활성
```

### 11-2. 유효성 검증 (확정, 더 이상 미결정 아님)

`run_imagefolder_training_workflow()` 맨 앞, ModelSpec 로드보다도
먼저 다음을 순서대로 검증한다(둘 다 비용이 들지 않는 순수 값/경로
검사이므로):

**(a) `checkpoint_every` 검증**(§6-2):

```python
def _validate_checkpoint_every(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("checkpoint_every must be an integer or None")
    if value < 1:
        raise ValueError("checkpoint_every must be at least 1")
```

`None` 허용, `bool` 거부, non-int 거부, `1` 미만 거부, `1` 이상 int만
허용. `config.py`의 `_require_positive_int()`(밑줄로 시작하는 그
모듈 내부 helper)를 import해서 쓰지 않는다 — `imagefolder_workflow.py`
자체의 private validator로 둔다. `checkpoint_every is not None and
checkpoint_out is None`이면 별도로 `ValueError`.

**(b) 출력 경로 재사용 검증**(§6-5):

```python
def _is_in_place_resume(request: ImageFolderWorkflowRequest) -> bool:
    if request.resume_from is None or request.checkpoint_out is None:
        return False
    return _normalized_path(request.resume_from) == _normalized_path(request.checkpoint_out)


def _normalized_path(path: str | Path) -> str:
    # Path.resolve()로 상대/절대 경로 표기 차이를 없애고, os.path.normcase()로
    # Windows의 대소문자 비구분 파일시스템에서의 오탐/누락을 줄인다
    # (POSIX에서 normcase는 no-op).
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _validate_checkpoint_output_paths(request: ImageFolderWorkflowRequest) -> None:
    if request.checkpoint_out is None:
        return
    if _is_in_place_resume(request):
        return  # resume_from == checkpoint_out -- _prepare_resume()이 검증을 담당

    checkpoint_path = Path(request.checkpoint_out)
    metadata_path = metadata_path_for_checkpoint(checkpoint_path)
    if checkpoint_path.exists():
        raise ValueError(
            f"{checkpoint_path} already exists -- a fresh training run (or a resume "
            "that writes to a different path than --resume-from) must use a new, "
            "unused checkpoint_out path. To continue training this exact checkpoint, "
            "pass it as both --resume-from and --checkpoint-out."
        )
    if metadata_path.exists():
        raise ValueError(
            f"{metadata_path} already exists -- a fresh training run (or a resume "
            "that writes to a different path than --resume-from) must use a new, "
            "unused checkpoint_out path."
        )
```

이 검증은 `checkpoint_every`가 켜져 있는지와 **무관하게** 항상
수행된다 — `checkpoint_out`이 주어지면 이번 학습이 끝날 때 어차피
최종 저장이 그 경로를 건드리기 때문이다(§6-4). `_is_in_place_resume()`은
§7-3/§11-3의 `metadata_ready` 초기값을 정하는 데도 그대로 재사용한다
(같은 판단을 두 곳에서 다르게 구현하지 않는다).

### 11-3. `run_imagefolder_training_workflow()` 내부 배선

**`ensure_checkpoint_metadata`/`metadata_ready`는 이 함수 호출 하나당
정확히 한 번씩 만들어지는 closure 상태이고, scheduled hook과 최종
저장이 그 하나를 함께 공유한다**(§7-3):

```python
def run_imagefolder_training_workflow(
    request: ImageFolderWorkflowRequest,
    *,
    progress_callback: TrainingProgressCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
) -> ImageFolderWorkflowResult:
    _validate_checkpoint_every(request.checkpoint_every)               # §11-2(a)
    _validate_checkpoint_output_paths(request)                          # §11-2(b)
    ...
    metadata_ready = _is_in_place_resume(request)  # §7-3, 이 호출 전체가 공유하는 상태

    def ensure_checkpoint_metadata() -> None:
        nonlocal metadata_ready
        if metadata_ready:
            return
        metadata_path = metadata_path_for_checkpoint(request.checkpoint_out)
        current_metadata = build_imagefolder_resume_metadata(model_spec, splits)
        save_imagefolder_resume_metadata(current_metadata, metadata_path)  # 원자적(§7-2)
        metadata_ready = True

    checkpoint_hook = (
        _make_checkpoint_hook(request, ensure_checkpoint_metadata)
        if request.checkpoint_every is not None else None
    )
    training_result = run_training(
        model, train_loader, val_loader, request.training_config, device="cpu",
        resume_state=resume_state, progress_callback=progress_callback,
        should_stop=should_stop, checkpoint_hook=checkpoint_hook,
    )
    ...
    # 학습 종료 후 최종 저장 (§6-4, 무조건 실행, checkpoint_out이 주어진 경우)
    if request.checkpoint_out is not None:
        ensure_checkpoint_metadata()  # §7-3 -- 이미 준비됐으면(scheduled 저장이 있었으면) 아무 것도 안 함
        save_training_checkpoint(  # 원자적(§7-2)
            request.checkpoint_out, model=model, training_result=training_result,
            training_config=request.training_config,
            loader_generator_state=loader_generator_state_after,
            cpu_rng_state=cpu_rng_state_after,
        )
        ...
```

`_make_checkpoint_hook()`(view 설계, 공유 `ensure_checkpoint_metadata`
클로저, global epoch 기준 cadence, `loader_generator` None 정책을
전부 반영한 최종 형태 — §6-5의 upfront 검증 덕분에 이 함수 내부는
더 단순해졌다). `model_spec`/`splits`를 다시 클로저로 캡처할 필요가
없다 — 이미 `ensure_checkpoint_metadata`가 그것들을 캡처했으므로,
hook은 그 함수 하나만 받으면 된다:

```python
def _make_checkpoint_hook(
    request: ImageFolderWorkflowRequest, ensure_checkpoint_metadata: Callable[[], None],
) -> CheckpointHook:
    def hook(view: EpochCheckpointView) -> None:
        global_epoch = len(view.history.train_losses)
        if global_epoch % request.checkpoint_every != 0:
            return  # non-scheduled epoch -- state_dict()/RNG 조회를 전혀 하지 않는다

        if view.loader_generator is None:
            raise ValueError(
                "auto checkpoint requires an explicit DataLoader generator for exact "
                "resume, but loader_generator is None"
            )

        ensure_checkpoint_metadata()  # §7-3, checkpoint보다 먼저 -- 이번 실행에서 아직 준비 안 됐을 때만 실제로 씀

        training_result = TrainingResult(
            history=view.history,
            best_state_dict=view.best_state_dict,
            optimizer_state_dict=view.optimizer.state_dict(),
            scheduler_state_dict=(
                view.scheduler.state_dict() if view.scheduler is not None else None
            ),
            epochs_without_improvement=view.epochs_without_improvement,
        )
        save_training_checkpoint(  # 원자적(§7-2)
            request.checkpoint_out, model=view.model, training_result=training_result,
            training_config=request.training_config,
            loader_generator_state=view.loader_generator.get_state(),
            cpu_rng_state=torch.get_rng_state(),
        )

    return hook
```

`loop.py`는 "N epoch마다"/"몇 번째 global epoch인지"/"metadata를
언제 준비해야 하는지"/"출력 경로가 재사용 가능한지"를 전혀 모른다 —
전부 workflow 계층의 책임이다.

### 11-4. `train_imagefolder.py` CLI 변경

```text
--checkpoint-every N   학습 도중 global epoch이 N의 배수가 될 때마다
                        자동으로 checkpoint를 갱신한다(--checkpoint-out
                        필수, N은 1 이상의 정수). 생략하면 기존과
                        동일하게 학습 종료 시에만 저장된다.
```

새 CLI 인자는 이것 하나뿐이다. `_validate_checkpoint_output_paths()`가
던지는 `ValueError`는 CLI의 기존 `except (ModelValidationError,
TrainingConfigError, ValueError, OSError): return 1` 패턴이 이미
포괄하므로 CLI 쪽에 새 예외 처리 코드가 필요 없다 — 메시지도
`print(f"Error: {exc}", file=sys.stderr)`로 그대로 사용자에게 보인다.

**CLI 실시간 checkpoint 저장 알림은 이번 Phase에 포함하지 않는다**:
`_print_progress()`는 metrics만 계속 찍는다.

### 11-5. `run_imagefolder_training_e2e.py`: 더 이상 무수정으로 남을 수 없다 (정책 전환의 직접적 결과)

이전 버전들은 이 스크립트를 "새 필드 기본값이 off라 무수정으로
남는다"고 결론지었다 — **이는 §6-5의 새 정책에서는 더 이상 성립하지
않는다.** 이 스크립트는 `checkpoint_every`를 전혀 쓰지 않지만
(자동 저장과 무관), `checkpoint_out=CHECKPOINT_PATH`(고정 경로)를
매번 주는 **fresh** 학습(stage 1)을 실행한다 — `CHECKPOINT_PATH`는
`REPO_ROOT / "artifacts" / "training" / ARTIFACT_NAME / "checkpoint.pt"`로
고정돼 있고, 스크립트가 스스로 정리하지 않는다. 오늘은 fresh가
무조건 덮어쓰므로 이 스크립트를 몇 번을 다시 실행해도 항상 성공했지만,
§6-5 정책 아래서는 **두 번째 실행부터 stage 1이 곧바로
`ValueError`로 실패한다**(첫 실행이 남긴 `checkpoint.pt`/
`checkpoint.pt.meta.json`이 이미 그 경로에 있으므로).

**필요한 최소 변경**: stage 1(fresh 학습) 실행 직전에
`CHECKPOINT_PATH`와 그 metadata sidecar(`metadata_path_for_checkpoint(CHECKPOINT_PATH)`)를
`unlink(missing_ok=True)`로 정리하는 두 줄을 추가한다. 이건 "새
자동 저장 기능을 쓰기 위한 변경"이 아니라 "Phase 4J가 core
`imagefolder_workflow.py`의 fresh-save 안전성 정책을 바꿨기 때문에
불가피하게 필요한 적응"이다 — 그 외 이 스크립트의 나머지 로직
(fresh 3 epoch → resume 2 epoch, C++ parity 등)은 전혀 바뀌지 않는다.
이 변경을 §13 파일별 변경 계획에 명시적으로 포함한다(이전 버전들의
"건드리지 않는다"는 결론을 이번 개정에서 정정).

---

## 12. 하위 호환

- **`run_training()`**: 새 `checkpoint_hook: CheckpointHook | None =
  None` 키워드 전용 파라미터. 기본값 `None`이면 기존과 완전히 동일.
- **`TrainingHistory`/`TrainingResult`**: 필드 추가 없음.
  `CHECKPOINT_FORMAT_VERSION`도 그대로 1.
- **`checkpoint.py`/`imagefolder_resume.py`**: 외부 계약은 바뀌지
  않는다. 내부 구현만 원자적 쓰기로 바뀐다(§7).
- **`ImageFolderWorkflowRequest`**: 새 필드 `checkpoint_every` 추가.
- **`run_imagefolder_training_workflow()`의 함수 시그니처**: 변경
  없음.

**명시적으로 인정하는 외부 동작 변경(하위 호환을 깨는 부분,
정직하게 명시)**: **`checkpoint_out`이 가리키는 경로에 이미
checkpoint 또는 sidecar가 있는 상태에서 fresh 학습을 실행하거나,
그 경로가 `resume_from`과 다른 경로로 resume 결과를 저장하려고
하면, 이전에는 조용히 덮어썼지만 이제는 `ValueError`로 명확히
거부된다.** 이 동작은 `checkpoint_every`(새 자동 저장 기능)를 켰는지
여부와 **무관하게** 적용된다 — `checkpoint_out`만 쓰는 기존 사용자도
영향을 받는다.

- 기존 checkpoint **포맷**과 in-place resume 호환성은 그대로 유지된다
  (payload 구조/필드는 안 바뀜, §4-3).
- 기존 API 기본값(`checkpoint_every=None`, 자동 저장 off)은 그대로
  유지된다.
- **다만 fresh 학습이 기존 `checkpoint_out` 경로를 덮어쓰는 동작
  자체는 이제 명확히 거부된다.** 이 변경을 정당화하는 근거: §1-5/§7-3에서
  증명했듯, "덮어쓴 뒤 나중에 검증"하는 방식으로는 checkpoint/metadata
  쌍의 정합성을 구조적으로 보장할 수 없다 — 그 상태에서 저장이
  중간에 실패하면 무관한 이전 checkpoint가 새 metadata와 조용히
  짝지어져 이후 resume 시 잘못된 가중치가 조용히 로드될 수 있다.
  **명시적인 실패가, 조용히 잘못된 checkpoint/metadata 쌍을 만드는
  것보다 낫다** — 이 원칙이 이번 정책 전환 전체의 근거다.
- 이 변경의 실질적 영향을 받는 이 저장소 내부의 유일한 기존 코드는
  `scripts/run_imagefolder_training_e2e.py`이고, 그 스크립트도 두
  줄의 정리(cleanup) 코드로 계속 정상 동작한다(§11-5/§13).
- README에 이 정책을 명확한 안내로 포함한다(§13).

---

## 13. 파일별 변경 계획

**수정**:
- `src/image_ai_studio/training/loop.py` — `EpochCheckpointView`
  dataclass, `CheckpointHook` 타입 별칭, `run_training()` 시그니처에
  `checkpoint_hook` 키워드 전용 인자 추가 + epoch 루프에 hook 호출
  삽입(§3), docstring 갱신. **Phase 4I가 이미 정의한
  `TrainingProgressCallback`/`ShouldStopCallback` 타입 별칭의 docstring에도
  §3-5의 RNG-purity/비변형 계약(학습 RNG 소비 금지, model/optimizer/
  scheduler/DataLoader generator 변경 금지)을 추가한다** — Phase 4I
  시점에는 이 계약이 필요하지 않았지만(파일 저장이 없었으므로), Phase
  4J의 exact-resume 보장이 이 계약에 의존하게 되면서 처음으로
  요구되는 제약이다.
- `src/image_ai_studio/training/checkpoint.py` — `_atomic_torch_save()`
  private helper 추가, `save_training_checkpoint()` 내부에서 사용.
  외부 시그니처/포맷 불변.
- `src/image_ai_studio/training/imagefolder_resume.py` —
  `_atomic_write_text()` private helper 추가,
  `save_imagefolder_resume_metadata()` 내부에서 사용. 외부 시그니처/
  포맷 불변.
- `src/image_ai_studio/training/imagefolder_workflow.py` —
  `ImageFolderWorkflowRequest.checkpoint_every` 필드 추가,
  `_validate_checkpoint_every()`/`_normalized_path()`/
  `_is_in_place_resume()`/`_validate_checkpoint_output_paths()`(§11-2)
  private 함수 신설, `run_imagefolder_training_workflow()` 내부에
  scheduled hook과 최종 저장이 공유하는 `metadata_ready`/
  `ensure_checkpoint_metadata`(§7-3/§11-3)를 두고 `_make_checkpoint_hook()`(§11-3)이
  그 클로저를 받도록 신설, 기존 최종 저장 코드를
  `ensure_checkpoint_metadata()` 호출 + metadata-first 순서로 재배선,
  `run_training(...)` 호출에 `checkpoint_hook=` 전달.
- `scripts/train_imagefolder.py` — `--checkpoint-every` 플래그 추가.
- **`scripts/run_imagefolder_training_e2e.py`** — stage 1(fresh 학습)
  직전에 `CHECKPOINT_PATH`/그 metadata sidecar를 `unlink(missing_ok=True)`로
  정리하는 두 줄 추가(§11-5). **이전 버전 문서가 "건드리지 않는다"고
  했던 것을 이번 개정에서 정정한다** — Phase 4J의 core 정책 변경이
  이 스크립트의 재실행 가능성에 직접 영향을 주기 때문에 불가피하다.
  나머지 로직(fresh 3 epoch → resume 2 epoch, TorchScript export,
  C++ CPU/CUDA parity, anchor 수치)은 전혀 바뀌지 않는다.
- `README.md` — Phase 4J 절 신설. **다음 취지의 강한 안내를 실제로
  포함했다**:

  > 새 학습(fresh)은 반드시 새로운 checkpoint 출력 경로를 사용해야
  > 한다. 기존 checkpoint 경로를 계속 이어서 쓰려면 그 경로를
  > `--resume-from`이자 `--checkpoint-out`으로 함께 지정하는
  > in-place resume이어야 한다 — 그 외의 경우 기존 경로에 이미
  > checkpoint/metadata가 있으면 학습이 시작되지 않고 명확한 오류로
  > 거부된다.

**신규**: `docs/phase4j_epoch_checkpoint_design.md`(본 문서).

**변경하지 않음**: `config.py`, `history.py`,
`model_definition/*`/`export/*`/`parity/*`/C++ 코드.

---

## 14. 테스트 계획 및 구현 결과

기존 파일에 추가했다(신규 파일 없음). **stdout/문자열 전체를 고정하는
brittle test는 만들지 않는다** — 오류 메시지 검증은 핵심 substring만
확인한다.

### `tests/scripts/test_train_imagefolder_cli.py` (책임 좁힘)

CLI 테스트는 배선(parsing/forwarding)과 최종 상태만 검증한다 — 중간
저장이 실제로 몇 번, 어느 epoch에서 일어났는지는 이 파일의 책임이
아니다.

- **`--checkpoint-every N`이 `ImageFolderWorkflowRequest.checkpoint_every`로
  정확히 전달됨을 request 캡처 방식으로 직접 증명한다**:
  `run_imagefolder_training_workflow`를 monkeypatch로 가짜 구현으로
  바꿔 CLI가 실제로 넘긴 `ImageFolderWorkflowRequest` 객체를 캡처하고
  `checkpoint_every`/`checkpoint_out` 필드 값을 직접 검사한다 — final
  post-hoc checkpoint는 forwarding이 끊어져도 만들어지므로 "checkpoint
  파일이 생겼다"는 사실만으로는 forwarding을 증명하지 못하기 때문이다.
  실제 workflow를 그대로 실행해 최종 checkpoint가 만들어지는 end-to-end
  확인은 별도의 정상 실행 테스트가 담당한다.
- `--checkpoint-every`만 주고 `--checkpoint-out`을 생략하면 exit
  code 1 + stderr에 `checkpoint_every`/`checkpoint_out` 핵심 문구
  포함(전체 메시지 고정은 하지 않음).
- `--checkpoint-every 0`/`-1`이 exit code 1 + stderr에 `at least 1`
  핵심 문구 포함으로 거부됨.
- `--checkpoint-every 1.5`가 argparse 자체의 파싱 오류(`SystemExit(2)`)로
  거부됨.
- `--checkpoint-every N`을 준 정상 실행(완전히 새 출력 경로)이 exit
  code 0으로 끝나고, 기존과 동일한 전체 artifact set을 만들어냄.
- **fresh 학습이 이미 checkpoint가 있는 `--checkpoint-out` 경로를
  가리키면 exit code 1 + stderr에 `already exists` 핵심 문구 포함,
  기존 파일은 그대로 남음**(§6-5 정책의 CLI 레벨 회귀 테스트).
- 기존 6+2개 CLI 테스트가 전부 무수정으로 통과(하위 호환 재확인 완료).

### `tests/training/test_imagefolder_workflow.py` (중간 저장 검증 + 출력 경로 정책의 실제 책임)

`save_training_checkpoint`를 `monkeypatch`로 spy해 호출마다 그 호출이
쓴 `training_result.history`의 길이(= 호출 시점의 global epoch)를
기록한다. **mid-loop(hook) 저장인지 최종(post-hoc) 저장인지를 별도
플래그로 구분해서 기록하지는 않는다** — 기록된 global epoch 목록과
호출 순서만으로 이미 scheduled/최종 저장을 구분해 증명할 수 있기
때문이다(예: `[2, 4, 5]`는 scheduled 2회 + 최종 1회를, `[10, 10]`은
scheduled와 최종이 같은 global epoch에서 겹쳐 총 2회 호출됐음을 각각
증명한다). 테스트를 위해 `inspect`/call stack 검사 같은 별도 구분
로직을 추가하지 않는다.

**cadence/중복 저장 관련(1~2차 리뷰에서 확정, 유지)**:
- `checkpoint_every=None`(기본)이면 `run_training()` 반환 전
  `save_training_checkpoint`가 전혀 호출되지 않고, 반환 후 정확히
  1번만 호출됨.
- fresh 5 epoch, `checkpoint_every=2`(완전히 새 출력 경로) →
  scheduled 저장이 global epoch 2, 4에서, 최종 저장이 5에서 →
  기록된 global epoch 목록이 정확히 `[2, 4, 5]`.
- in-place resume(기존 global epoch 7, `checkpoint_every=5`로 3
  epoch 추가) → scheduled 저장과 최종 저장이 둘 다 global epoch
  10에서 발생(총 2번 호출) — cadence가 호출 횟수가 아니라 global
  epoch 기준임을 함께 고정.
- `checkpoint_every`가 주어졌는데 `checkpoint_out`이 없으면 `ValueError`.
- `checkpoint_every=0`/`-1`/`True`/`1.5`가 direct workflow 호출에서
  전부 `ValueError`(`_validate_checkpoint_every()` 계약).

**출력 경로 정책 관련(신규, 4차 리뷰 — 아래 목록으로 이전 버전의
"fresh가 무관한 stale metadata를 새 metadata로 교체" 및 "metadata
교체 후 이전 checkpoint가 남아도 dataset이 다르면 resume 거부"
테스트를 완전히 대체한다)**:

- **fresh + 기존 `checkpoint_out`이 이미 존재 → `ValueError`, 기존
  checkpoint 파일이 변경되지 않음**(내용을 사전에 기록해두고 실행
  후 바이트 단위로 동일함을 확인).
- **fresh + checkpoint는 없지만 metadata sidecar만 존재 →
  `ValueError`, sidecar가 변경되지 않음.**
- **fresh + 완전히 새 경로 → metadata-first로 정상 저장 성공**(신규
  metadata 파일이 먼저 생기고 이어서 checkpoint가 생김을 확인).
- **`resume_from != checkpoint_out` + 출력 경로에 checkpoint가 이미
  존재 → `ValueError`**, `resume_from`이 가리키는 원본 checkpoint/
  metadata는 전혀 읽히거나 바뀌지 않음.
- **`resume_from != checkpoint_out` + 출력 경로에 sidecar만 존재 →
  `ValueError`.**
- **`resume_from != checkpoint_out` + 완전히 새 출력 경로 → 정상
  저장 성공**(resume은 `resume_from`에서, 결과는 새 경로에 저장,
  새 경로의 metadata가 `resume_from`이 검증받은 metadata와 값
  기준으로 동일함을 확인).
- **`resume_from == checkpoint_out`(in-place resume) → 기존처럼
  정상 동작**(회귀 재확인, exact-resume 테스트 포함). **exact-resume
  테스트의 전제(§3-5)**: built-in `_make_checkpoint_hook()` 사용,
  `progress_callback=None` 또는 RNG/학습 상태를 변경하지 않는
  callback 사용, `should_stop=None` 또는 단순 외부 bool flag 조회만
  수행 — continuous run과 resume run의 model/optimizer/scheduler/
  history/best_state_dict/RNG 결과를 값 기준으로 비교한다. RNG를
  소비하는 악의적/오용 callback을 만들어 "exact resume이 깨진다"는
  것 자체를 단위 테스트로 고정할 필요는 없다 — 이는 지원 계약 밖의
  오용이며, §3-5의 문서화와 아래 "callback/hook purity 문서화 검증"
  항목으로 관리한다.
- **상대 경로와 절대 경로가 같은 파일을 가리킬 때도 in-place
  resume으로 정확히 인식됨**(`_normalized_path()` 계약 테스트 — 예:
  `Path("checkpoint.pt")`를 특정 cwd에서 만든 것과 그 절대 경로가
  같은 파일로 인식되는지).
- **완전히 새 출력 경로에서 metadata 저장은 성공하고 checkpoint
  저장이 실패하면, 디스크에는 metadata만 남고 checkpoint는 생기지
  않음**(`save_training_checkpoint`를 monkeypatch로 실패시켜 확인).
- **위 상태에서 그 경로로 resume을 시도하면 `FileNotFoundError`로
  명확히 실패함**(`load_training_checkpoint()`가 존재 확인 없이 곧바로
  `torch.load()`를 호출하기 때문 -- 정확한 예외 타입과 파일명을 담은
  핵심 메시지까지 검증한다. `pytest.raises(Exception)`처럼 지나치게
  넓은 예외로 검증하지 않는다).
- **"무관한 이전 checkpoint + 새 metadata"가 조합되는 상태 자체가
  생성되지 않음**을 위 테스트들의 조합으로 간접 증명(그런 조합을
  만들려는 모든 시도가 사전 검증에서 차단됨).

**metadata/atomic save(1~2차 리뷰에서 확정, 유지)**:
- **resume은 기존 metadata를 재사용하고 재검증하지 않음**: in-place
  resume에서 `save_imagefolder_resume_metadata`가 hook의 scheduled
  저장 이후에도 한 번도 호출되지 않았음(spy 호출 횟수 0)과 sidecar
  파일의 내용(bytes)이 그대로임을 확인 -- 호출 횟수 0이 이미
  "재작성되지 않음"을 직접 증명하므로, 파일시스템 timestamp 정밀도
  차이로 flaky해질 수 있는 mtime 비교에는 의존하지 않는다.
- **legacy checkpoint 하위 호환**: Phase 4J 이전에 저장된
  checkpoint(포맷 불변)를 in-place resume으로 이어서 학습해도
  문제없이 로드/재개됨.
- 기존 9+4개 테스트가 전부 무수정으로 통과 — 단, 각 테스트가 같은
  `tmp_path`/`output_dir`를 여러 번 fresh로 재사용하고 있지 않은지
  §6-5 정책 도입 시 반드시 재검토(재사용하고 있다면 하위 디렉터리를
  분리하도록 그 테스트들도 함께 손봐야 한다 — 이는 코드 변경이지
  이 설계 문서의 정책 변경 대상은 아니다).

**callback/hook purity 문서화 검증(신규, §3-5)**:
- **production CLI의 `_print_progress()`는 출력만 하며 torch RNG/
  model/optimizer/scheduler/generator를 변경하지 않는다**: 코드
  리뷰로 확인하는 항목이지만, `_print_progress()`가 `progress`
  인자의 필드를 문자열로 포맷해 `print()`하는 것 외에 아무 부수효과가
  없음을 (필요하다면) `torch.get_rng_state()`를 호출 전후로 비교하는
  가벼운 회귀 테스트로도 고정할 수 있다.
- **built-in `_make_checkpoint_hook()`이 반환하는 hook은
  `.state_dict()`/`.get_state()`/`torch.get_rng_state()`와 파일
  저장(`save_training_checkpoint()`/`ensure_checkpoint_metadata()`)만
  수행하고 학습 상태를 변경하지 않는다**: hook 호출 전후로
  `view.model.state_dict()`/`view.optimizer.state_dict()`의 값이
  달라지지 않았음을 확인하는 가벼운 회귀 테스트로 고정할 수 있다.

### `tests/training/test_loop.py`

(이 파일은 `loop.py` 레벨 계약만 다루므로 §6-5의 workflow 레벨 경로
정책과 무관하다)

- **`checkpoint_hook=None`이면 기존과 동일한 observable 결과**:
  history 전체(train/val losses, val accuracies, best_epoch,
  stopped_early, stopped_by_user), 최종 model.state_dict(),
  best_state_dict, optimizer/scheduler state_dict,
  epochs_without_improvement까지 전부 비교한다(단순히 train_losses/
  best_state_dict 일부만 비교하지 않음).
- `checkpoint_hook`이 완료된 epoch마다 정확히 한 번 호출됨.
- **non-scheduled epoch 비용 회피를 실제 spy로 증명**:
  `torch.optim.Adam.state_dict`(scheduler 계약은
  `ReduceLROnPlateau.state_dict`로 별도 짧은 테스트)를 monkeypatch로
  감싸 호출 횟수를 추적하고, 연속된 hook 호출 사이(hook이 반환한
  뒤부터 다음 epoch의 view 생성/hook 진입 전까지)에는 호출이 전혀
  없음을 누적 호출 횟수의 델타로 확인한다 -- hook 내부 카운터만으로는
  core/view 생성 쪽의 실수를 검출하지 못하므로, 실제 메서드 호출을
  spy한다.
- **`EpochCheckpointView`가 그 epoch 시점의 실제 상태와 정확히 일치**:
  `evaluate()`를 monkeypatch해 val_loss 개선/비개선 순서를 결정론적으로
  고정하고(예: `1.0, 0.8, 0.9, 1.1` → `epochs_without_improvement`
  기대값 `[0, 0, 1, 2]`, `best_epoch` 기대값 `[1, 2, 2, 2]`),
  `train_one_epoch()`도 monkeypatch해 매 epoch 파라미터를 epoch
  번호로 채워 각 hook 호출 시점의 `best_state_dict` snapshot이 실제로
  어느 epoch의 값인지 정확히 검증한다(살아있는 참조를 hook 밖에서
  비교하지 않고, hook 내부에서 tensor `.clone()`으로 snapshot을 만들어
  기록).
- ephemeral 계약: `view.model is model`, `view.history is` 그 history
  객체, hook 호출 시점의 `len(view.history.train_losses)`가 실제
  global epoch과 일치(identity 기반, brittle하지 않음).
- `checkpoint_hook`이 `progress_callback`보다 먼저 호출됨.
- scheduled/non-scheduled epoch에서 `progress_callback`/`should_stop`
  예외 시 저장 여부(§9-1).
- scheduled checkpoint의 `stopped_by_user`는 항상 `False`.
- CPU RNG/`train_loader.generator` 상태 정확한 캡처.
- **자동 checkpoint에서 resume한 결과와 continuous run exact
  equality**(전제 명시, §3-5): 이 테스트가 쓰는 `checkpoint_hook`은
  `EpochCheckpointView`를 읽기만 하고 즉시 저장하는 순수 구현이고,
  `progress_callback`은 `None`이거나 RNG/state를 건드리지 않는
  순수 callback이며, `should_stop`도 `None`이거나 단순 외부 bool
  flag 조회만 수행한다 — §3-5 계약을 어기는 콜백으로는 이 동등성이
  성립한다고 주장하지 않는다. hook 안에서 `view.scheduler`/
  `view.loader_generator`가 실제로 `None`이 아님을 `.state_dict()`/
  `.get_state()`를 부르기 전에 명시적으로 assert하고, 최종 비교에는
  `best_state_dict`/`epochs_without_improvement`도 포함한다.
- early stopping 발동 scheduled epoch에서 `stopped_early is True`.
- `checkpoint_hook` 예외 전파, `TrainingResult` 미반환.
- `config.epochs == 1`에서도 정확히 1번 호출.

### `tests/training/test_checkpoint.py`

- `save_training_checkpoint()`/`_atomic_torch_save()`가 원자적으로
  저장함(저장 도중 예외 → 기존 파일 보존, 임시 파일 미잔존).
- **`os.replace` 실패 시 기존 파일 보존 + 임시 파일도 남지 않음을 함께
  확인**(`tmp_path.iterdir()`로 디렉터리 전체가 원본 파일 하나뿐임을
  검증).
- 임시 파일 정리(`unlink`) 자체가 실패해도 사용자에게 보이는 예외는
  원래 저장 실패 예외임을 확인.
- 기존 라운드트립/에러 계약 테스트 전부 무수정으로 통과.

### `tests/training/test_imagefolder_resume.py`

- metadata sidecar도 원자적으로 저장됨(`_atomic_write_text()`), 임시
  파일 미잔존까지 확인.
- metadata 저장 실패 시(완전히 새 경로에서) checkpoint가 아직
  생성되지 않은 상태 그대로 남음, `os.replace` 실패 후에도 temp 파일이
  남지 않음을 확인.

### 회귀 전체

`tests/training/` + `tests/scripts/` + 전체 `pytest`, 그리고 4개
기존 E2E 스크립트 재실행(§11-5의 두 줄 cleanup 추가 후) —
`checkpoint_hook=None`/`checkpoint_every=None`(모든 E2E의 기본 경로)이
Phase 4I까지의 수치와 완전히 동일함을 확인한다. **특히
`run_imagefolder_training_e2e.py`는 연속으로 두 번 실행해도(로컬
재실행/CI 재시도 시나리오) 둘 다 성공하는지 반드시 확인한다**(§11-5의
cleanup이 실제로 이 시나리오를 커버하는지의 핵심 검증).

---

## 15. 구현 순서 (작은 단계)

1. `loop.py`에 `EpochCheckpointView`/`CheckpointHook` 추가(순수 추가).
2. `run_training()` 시그니처에 `checkpoint_hook` 키워드 전용 인자
   추가, 본문은 아직 호출 안 함(회귀 0건 확인).
3. epoch 루프에 §3 위치대로 hook 호출 삽입.
4. `test_loop.py`에 §14 테스트 작성/통과.
5. `checkpoint.py`/`imagefolder_resume.py`에 원자적 쓰기 helper 추가.
6. `test_checkpoint.py`/`test_imagefolder_resume.py`에 원자성 테스트
   작성/통과.
7. `imagefolder_workflow.py`에 `checkpoint_every` 필드 + §11-2의 두
   검증 함수(`_validate_checkpoint_every()`/
   `_validate_checkpoint_output_paths()`) + 공유 `metadata_ready`/
   `ensure_checkpoint_metadata`(§7-3/§11-3) + `_make_checkpoint_hook()` +
   기존 최종 저장 코드를 metadata-first로 재배선 + `run_training()`
   호출에 전달.
8. `test_imagefolder_workflow.py`에 §14 테스트(출력 경로 정책 포함)
   작성/통과.
9. `train_imagefolder.py`에 `--checkpoint-every` 플래그 추가.
10. `test_train_imagefolder_cli.py`에 §14 테스트 작성/통과.
11. **`run_imagefolder_training_e2e.py`에 stage 1 직전 cleanup 두 줄
    추가**(§11-5) — 연속 두 번 실행 테스트로 확인.
12. README 갱신(Phase 4J 절 신설, §6-5 정책 안내 포함).
13. 전체 회귀 — `tests/training/` + `tests/scripts/` + 전체 pytest +
    기존 4개 E2E(연속 두 번 실행 포함) + C++ CPU/CUDA parity.

---

## 16. 위험 요소

- **기존 사용자/스크립트가 fresh 학습에 `checkpoint_out`을 재사용하는
  습관에 의존하고 있었다면 이번 변경으로 깨짐**: §12에서 인정한
  의도적 하위 호환 변경 — 이 저장소 안의 유일한 그런 코드
  (`run_imagefolder_training_e2e.py`)는 §11-5로 이미 대응했지만,
  저장소 밖에서 이 워크플로우 함수를 직접 호출하는 외부 코드가
  있다면 영향을 받는다. README의 강한 안내(§13)로 완화한다.
- **`checkpoint_hook` 안에서 무거운 I/O가 매 scheduled epoch 학습을
  지연시킴**: `checkpoint_every`로 빈도를 낮추는 것이 유일한 완화
  수단.
- **Windows에서 파일이 일시적으로 잠겨 `os.replace()`가 실패**: §7-4 —
  직전 성공한 scheduled checkpoint는 손상되지 않는다.
- **`checkpoint_hook`을 `progress_callback`보다 먼저 두는 순서를
  실수로 반대로 구현할 위험**: §14의 호출 순서 테스트가 회귀를 잡는다.
- **cadence 계산을 hook의 호출 횟수 카운터로 잘못 구현할 위험**:
  global epoch 기준 회귀 테스트(§14)가 잡는다.
- **`_normalized_path()`의 경로 동일성 판단 한계**: §7-4에서 설명한
  심볼릭 링크/네트워크 드라이브 매핑 같은 극단적인 경우 — 안전한
  방향(과도하게 "다른 경로"로 판단해 정당한 in-place resume을
  실수로 거부)으로만 실패하므로 데이터 손상 위험은 없지만, 사용자
  경험상 불편할 수 있다.
- **`view.optimizer`/`.scheduler`/`.loader_generator`를 hook이 실수로
  변형할 위험**: 읽기 전용 계약이 Python 타입 시스템으로 강제되지
  않으므로 코드 리뷰로 확인해야 한다.
- **custom `progress_callback`/`should_stop`/`checkpoint_hook`이 §3-5의
  계약을 어길 위험**: custom `progress_callback`/`should_stop`/
  `checkpoint_hook`이 학습 RNG를 소비하거나 model/optimizer/scheduler/
  DataLoader generator를 변경하면, mid-loop checkpoint가 실제 다음
  epoch 시작 상태를 나타내지 않게 되어 exact-resume 보장이 깨질 수
  있다. core는 임의 callable의 closure/global side effect를 막을
  수 없으므로, 이 API들은 관찰/flag 조회/읽기 전용 저장 계약을
  따라야 한다(§3-5) — 이 계약은 Python 타입 시스템으로 강제되지
  않고 docstring/코드 리뷰로만 관리되므로, built-in 구현
  (`_print_progress()`/`_make_checkpoint_hook()`)이 실제로 이를
  지키는지는 §14의 문서화 검증 테스트로 고정한다.
- **`train_loader.generator`가 `None`인 caller가 미래에 생길 가능성**:
  core 레벨은 허용하지만 ImageFolder production 경로는 명시적으로
  거부한다(§8-1... 정확히는 §4-1/§11-3).
- **`os.fsync()` 비용**: scheduled epoch마다 발생 — 완화 수단은
  `checkpoint_every`를 늘리는 것.

---

## 17. 회귀 불변조건

```text
- checkpoint_hook=None, checkpoint_every=None이면 기존 학습 수치와
  tensor 결과 완전히 동일(단, checkpoint_out을 이미 존재하는 경로에
  재사용하는 fresh 실행 자체는 이제 ValueError로 거부됨 -- §12의
  의도적 동작 변경)
- checkpoint_every=1일 때만 모든 완료된 epoch가 progress_callback/
  should_stop 예외보다 먼저 저장된다(§9-1)
- 저장 cadence는 hook의 호출 횟수가 아니라 global epoch 기준이며,
  resume 후에도 절대 epoch 번호로 정확히 이어진다(§10-1)
- non-scheduled epoch에서는 optimizer/scheduler state_dict()나
  RNG 읽기가 전혀 발생하지 않는다(§4-1)
- resume_from == checkpoint_out인 in-place resume만 기존
  checkpoint_out을 갱신할 수 있다 -- 그 외(fresh, 또는 resume_from !=
  checkpoint_out)는 checkpoint_out이 완전히 비어있는 새 경로일 때만
  진행되고, 그렇지 않으면 학습 시작 전에 ValueError로 거부된다(§6-5)
- metadata가 유효하게 준비된 뒤에만 checkpoint가 갱신된다(mid-loop
  저장과 최종 저장 양쪽 모두, §7-3) -- metadata는 workflow 실행당
  최대 한 번만 실제로 쓰이고, scheduled hook과 최종 저장이 그 상태
  (`metadata_ready`)를 공유한다(§7-3/§11-3)
- mid-loop 자동 저장의 stopped_by_user는 항상 False이고, 정확한
  값은 학습 종료 후 최종 저장만 반영한다(§9-3)
- scheduled epoch와 이번 호출의 마지막 epoch가 같으면 같은 경로에
  두 번 저장될 수 있다(허용된 정책, §9-4)
- **자동 checkpoint exact-resume은 built-in hook과 RNG/state-pure
  progress_callback/should_stop 조합에서 보장된다(§3-5) — 무조건적인
  보장이 아니다.** `progress_callback`/`should_stop`/`checkpoint_hook`은
  학습에 사용되는 RNG를 소비하거나 model/optimizer/scheduler/
  DataLoader generator를 변경해서는 안 된다(§3-5); 이 계약을 어긴
  custom callback으로 인한 비결정성은 core의 버그가 아니라 caller의
  계약 위반이다
- progress_callback/should_stop의 상호 순서(Phase 4I가 확정)는 그대로 유지
- Phase 4F exact checkpoint/resume 유지(in-place resume 경로에서)
- Phase 4G ImageFolder metadata 검증 유지
- Phase 4H production CLI/workflow/E2E 책임 분리 유지
- Phase 4I progress_callback/should_stop 하위 호환 유지
- current model checkpoint 저장 시점 유지("현재" != "best")
- CPU RNG 복원 직후 즉시 run_training() 호출 구조 유지
- 기존 ImageFolder E2E 3+2 epoch anchor 수치 유지(cleanup 두 줄
  추가 후에도, 연속 두 번 실행 모두에서)
- TorchScript export 및 C++ CPU/CUDA parity 유지
```

---

## 18. 미결정 사항

**Phase 4J 구현 전에 남은 정책 미결정 사항은 없다.** 자동 저장
기본값, cadence 계산 기준, metadata 처리 순서, `checkpoint_every`
입력 검증, view 설계와 비용, **출력 경로 재사용 정책**(in-place
resume만 허용, 그 외에는 기존 파일 존재 시 거부), 그리고 이번 개정의
`progress_callback`/`should_stop`/`checkpoint_hook`의 RNG-purity/
비변형 계약과 그에 따른 exact-resume 보장 범위(§3-5/§5/§17)까지
전부 확정했다.

에러 메시지의 구체적인 문구, private 함수/변수의 정확한 이름, 테스트
함수명 같은 것들은 구현 세부사항이며 설계 미결정 사항으로 취급하지
않는다. `_normalized_path()`가 다루지 못하는 극단적인 경로 동일성
케이스(§7-4/§16)는 "결정을 못 내린 것"이 아니라 안전한 방향으로만
실패하는 것으로 이미 확인된, 문서화된 한계다.

---

## 19. 향후 확장

- **비동기/백그라운드 저장**: 명시적 비목표이며, §4-1의 ephemeral
  view 계약상 현재 설계를 그대로 백그라운드 스레드에 넘기는 것은
  안전하지 않다 — 지원하려면 독립 복사 snapshot에 해당하는 별도
  타입을 새로 설계해야 한다.
- **명시적 overwrite 지원(`--overwrite-checkpoint` 등)**: 이번
  Phase는 §6-5의 "기존 파일이 있으면 무조건 거부"를 채택했다 —
  사용자가 정말로 기존 (무관한) checkpoint를 의도적으로 덮어쓰고
  싶어하는 사용 사례가 실제로 확인되면, 별도 Phase에서 명시적
  플래그(예: `--overwrite-checkpoint`)와 함께 "기존 파일을 삭제하고
  진행"하는 옵트인 경로를 추가할 수 있다 — 이번 설계는 그 확장을
  구조적으로 막지 않는다(§6-5의 검증 함수에 새 조건 분기를 추가하는
  정도로 확장 가능). 만약 그런 덮어쓰기를 "완전히 무관한 이전
  학습"과 "자기 자신의 예전 시도"로 구분해서 다르게 다뤄야 한다면,
  그때는 checkpoint/metadata 쌍에 식별자(저장 이벤트 UUID 등)를
  함께 기록하는 메커니즘이 다시 필요해질 수 있다.
- **latest + best 이중 저장**: §6-1에서 가치는 인정했지만 이번
  Phase에서는 미룬 정책.
- **checkpoint retention/rotation**: 이번 Phase의 명시적 비목표.
- **SIGINT graceful stop과의 연결**: Phase 4I §19가 이미 언급한 배선이
  이미 가능하다 — 이번 Phase는 구현하지 않는다.
- **CLI 실시간 checkpoint 알림**: §11-4에서 이번 Phase 범위 밖으로
  확정했다.
- **cloud/object storage 저장**: hook이 동기적으로, hook 호출 범위
  안에서 완료한다는 전제 아래 core API 변경 없이 가능하다.
- **scheduled/final 중복 저장 최적화**: §9-4에서 "중복 허용"으로
  결정했지만, 나중에 I/O가 실제로 문제가 되면 재검토할 수 있다 —
  단 `stopped_by_user` 정확성 문제(§9-3)를 별도로 해결해야 한다.
