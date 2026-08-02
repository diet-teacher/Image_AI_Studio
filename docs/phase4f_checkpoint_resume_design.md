# Phase 4F: Full Checkpoint + Resume

이 문서는 설계 검토와 실제 구현/실행 결과를 함께 정리한다. Phase 4A~4E는
`save_state_dict()`/`load_state_dict()`로 **모델 가중치만** 저장/재로드할
수 있었다 -- optimizer momentum, LR scheduler 진행 상태, epoch 카운터,
`TrainingHistory`, early stopping 카운터는 `run_training()` 호출이
끝나면 전부 사라졌다. Phase 4F의 목표는 이 상태를 전부 저장해 **중단된
학습을 이어서(resume) 실행**할 수 있게 하는 것이다.

핵심 질문은 "optimizer state를 저장할 것인가?"가 아니라 **"재개한
학습이 중단 없이 연속 실행한 학습과 어디까지 같아야 하는가?"**였다.
이 문서는 그 계약을 정확히 정의하고, 실제로 실행해서 증명한다.

## 1. bare state_dict와 full checkpoint의 차이

| | `save_state_dict()`/`load_state_dict()` (Phase 4A, 변경 없음) | `save_training_checkpoint()`/`load_training_checkpoint()` (Phase 4F, 신규) |
|---|---|---|
| 내용 | `model.state_dict()`만 | model + optimizer + scheduler + history + best model + early stopping 카운터 + DataLoader generator + CPU RNG |
| 용도 | best model 배포/TorchScript export | 중단된 학습 재개 |
| 포맷 | 텐서 dict 그대로 | `"format_version"` 키를 가진 self-contained payload dict |

두 포맷은 서로 다른 파일이며 섞어 쓰면 안 된다. `load_state_dict()`에
full checkpoint 파일을 넣거나, `load_training_checkpoint()`에 bare
state_dict 파일을 넣으면 각각 어느 쪽을 쓰려던 것인지 안내하는 명확한
`ValueError`가 발생한다(9절).

## 2. current model과 best model의 차이 (혼동 시 실제 버그가 됨)

```text
model.state_dict()               = 마지막으로 완료된 epoch의 "현재" 모델 (= optimizer state와 같은 시점, resume 시작점)
training_result.best_state_dict  = 지금까지 validation loss가 가장 낮았던 epoch의 snapshot (최종 평가/export용)
```

예를 들어 현재 epoch=5, best epoch=3이면 checkpoint는
`model_state_dict`에 epoch 5, `best_state_dict`에 epoch 3의 가중치를
담아야 한다. `save_training_checkpoint()`의 `model` 인자에 실수로
`best_state_dict`를 로드한 모델을 넘기면 이 둘이 같아져 버리는데, 이는
resume 시작점이 "지금까지 가장 좋았던 지점"으로 조용히 바뀌는 버그다.

`tests/training/test_checkpoint.py::test_checkpoint_distinguishes_current_model_from_best_model`
이 이 상황을 직접 재현해 고정한다 -- best epoch(2)가 마지막 epoch(3)가
아닌 상황을 만들고, `model_state_dict`는 epoch 3, `best_state_dict`는
epoch 2의 값을 각각 유지하며 서로 다름을 확인한다.

## 3. `TrainingConfig.epochs`의 의미와 completed epoch 계산

`epochs`는 resume 여부와 무관하게 **"이번 `run_training()` 호출에서
추가로 실행할 epoch 수"**를 뜻한다 (총 목표 epoch가 아니다). 이전에
3 epoch를 완료하고 `epochs=2`로 resume하면 실행되는 epoch는 4, 5다.

completed epoch 수는 별도 필드로 저장하지 않는다 -- `len(history.
train_losses)`가 유일한 출처다:

```python
completed_epochs = len(history.train_losses)
for epoch in range(completed_epochs + 1, completed_epochs + config.epochs + 1):
    ...
```

`resume_state`가 없으면 `completed_epochs = 0`이라 `range(1,
config.epochs + 1)`이 되어 Phase 4E까지의 동작과 완전히 동일하다.

`TrainingResumeState.__post_init__`은 `history.train_losses`/
`val_losses`/`val_accuracies` 세 리스트의 길이가 같은지, 비어있지
않은지 검증한다 -- 이 셋 중 하나라도 다르면 `completed_epochs` 자체를
신뢰할 수 없기 때문이다.

## 4. config 변경 허용/금지 필드

`optimizer.load_state_dict()`/`scheduler.load_state_dict()`는 저장된
param group 값(예: learning_rate, momentum)을 그대로 복원한다 -- 즉
resume 시 다른 `learning_rate`를 주더라도, 로드 시점에 checkpoint의
값으로 덮어써져 새 값이 조용히 무시된다(실제로 `optimizer =
_build_optimizer(model, new_config); optimizer.load_state_dict(saved)`
순서로 확인). 그래서 Phase 4F는 이 필드들을 **자유 변경 대상에서
제외**했다:

```text
반드시 checkpoint와 일치해야 함:
    optimizer, learning_rate, momentum,
    lr_scheduler, lr_scheduler_factor, lr_scheduler_patience,
    batch_size

자유롭게 변경 가능:
    epochs, early_stopping_patience
```

`batch_size`도 강제 일치 대상이다 -- 바뀌면 같은 sample 순서라도 batch
그룹핑과 optimizer step 수 자체가 달라져 exact resume 계약이 깨진다.
`optimizer`/`lr_scheduler`가 다르면 애초에
`optimizer.load_state_dict()`/`scheduler.load_state_dict()`가 다른
종류의 객체에 안 맞는 state를 넣으려다 실패하거나(또는 더 나쁘게, 값이
일부만 반영되고) 조용히 잘못된 상태가 될 수 있다.

`early_stopping_patience`는 바뀌어도 되지만, `epochs_without_improvement`
카운터는 항상 checkpoint의 값을 그대로 이어받는다 -- patience=10으로
늘려서 resume해도 checkpoint 시점에 이미 4번 연속 개선이 없었다면
카운터는 4에서 계속된다(0으로 리셋되지 않는다).

비교는 `config.py`의 `require_compatible_resume_config()` 한 함수가
필드 목록(`RESUME_CONFIG_FIELDS` 튜플)을 순회하며 담당한다 -- registry나
별도 비교 클래스는 만들지 않았다. `checkpoint.py`가 이 함수를
re-export하므로 `from image_ai_studio.training.checkpoint import
require_compatible_resume_config`로도 그대로 가져다 쓸 수 있다(11절).

**이 검증은 caller가 알아서 호출해야 하는 관례가 아니라, `run_training()`
이 `resume_state`를 받을 때 항상 스스로 강제하는 core API 계약이다.**
`TrainingResumeState`에 `training_config`(checkpoint 저장 당시 config를
`asdict()`한 dict) 필드를 추가해, `run_training()`이 optimizer/scheduler
state를 로드하기 **전에** 이 값과 새 `config`를 항상 비교한다:

```python
if resume_state is not None:
    if resume_state.history.stopped_early:
        raise ValueError(...)
    require_compatible_resume_config(resume_state.training_config, config)  # 항상 실행됨
    optimizer.load_state_dict(...)
    ...
```

`scripts/run_resume_training_e2e.py`가 `load_training_checkpoint()`
직후 이 함수를 별도로 호출하는 것은 **조기 실패(fail fast)를 위한
선택 사항**일 뿐이다 -- 그 호출을 지우고 바로 `run_training(...,
resume_state=...)`를 호출해도 config가 다르면 여전히 거부된다. 이건
실제로 `tests/training/test_loop.py::test_run_training_resume_rejects_
incompatible_saved_config`가 caller helper를 전혀 거치지 않고
`run_training()`을 직접 호출해서 증명한다.

## 5. Checkpoint 저장 가능 시점 / resume 시작 시점

`run_training()`은 **모든 epoch를 마친 뒤에야** `TrainingResult`를
반환한다. 따라서 이번 Phase에서 실제로 가능한 저장 시점은 정확히
"`run_training()` 호출이 정상 종료된 뒤, caller가 저장"뿐이다:

```text
지원하지 않음:
    - 단일 run_training() 호출 내부에서 매 epoch 자동 저장
    - epoch callback
    - 프로그램 강제 종료 직전 자동 저장
    - batch 중간 저장
```

긴 학습에서 주기적 checkpoint가 필요하면, caller가 학습을 여러
chunk로 나눠 `run_training()`을 반복 호출하고 매 호출 종료 후 저장하는
방식을 쓴다 (예: 10 epoch 실행 -> checkpoint -> 10 epoch resume ->
checkpoint -> ...). `run_training()`은 여전히 파일을 쓰지 않는다
(Phase 4A부터 유지된 원칙).

## 6. loader generator와 CPU RNG가 둘 다 필요한 이유 (실측 근거)

DataLoader의 shuffle 순서와 Dropout의 mask는 **서로 다른 두 개의
독립적인 RNG 상태**에 의존한다는 것을 직접 실행해서 확인했다:

* **DataLoader shuffle generator** -- 3개 기존 E2E 스크립트 전부
  `torch.Generator().manual_seed(SEED)`를 만들어 `DataLoader(shuffle=True,
  generator=...)`에 넘긴다. 이 로컬 generator의 `.get_state()`를
  저장했다가 새 `Generator()`에 `.set_state()`로 복원하면, 이후
  epoch들의 셔플 순서가 연속 실행과 정확히 일치함을 실측 확인했다.
  이 상태는 **전역 RNG(`torch.get_rng_state()`)와 무관**하다 -- 완전히
  별개의 객체다.
* **CPU RNG (`torch.get_rng_state()`)** -- `nn.Dropout`은 이 로컬
  generator가 아니라 전역 CPU RNG를 쓴다. 실제로 `torch.manual_seed(0)`
  이후 `Dropout(p=0.5)`를 두 번 호출하면 서로 다른 mask가 나오지만,
  `torch.get_rng_state()`를 저장했다가 `torch.set_rng_state()`로
  복원한 뒤 다시 호출하면 첫 번째 호출과 bit-identical한 mask가
  나옴을 실측 확인했다. `examples/models/phase4_training_model.json`
  (Phase 4A/4B 회귀 앵커가 실제로 쓰는 모델)에 `dropout` 레이어가 있어,
  이건 가상의 우려가 아니라 지금 이미 실행되고 있는 경로다.

그래서 checkpoint는 이 둘을 **별도 필드**(`loader_generator_state`,
`cpu_rng_state`)로 각각 저장한다. `optimizer.state_dict()`/
`scheduler.state_dict()` 생성 및 `load_state_dict()` 자체는 CPU RNG를
소비하지 않음도 실측 확인했다(아래 8절 순서의 안전성 근거).

### RNG 복원 순서

```text
1. ModelSpec 로드, model 생성 (가중치는 이후 곧 덮어써지므로 이 시점의 초기화가 결과에 영향 없음)
2. model_state_dict 로드
3. DataLoader(train/val) 생성 -- train_loader는 새 torch.Generator()에
   loader_generator_state를 set_state()한 뒤 그 generator로 생성
4. torch.set_rng_state(cpu_rng_state)  -- 반드시 다른 모든 준비가 끝난 뒤, 마지막에
5. RNG를 소비하는 다른 작업 없이 바로 run_training(..., resume_state=...) 호출
```

optimizer/scheduler 생성과 `load_state_dict()`는 `run_training()`
내부(4단계 이후)에서 일어나지만, 이 두 연산이 CPU RNG를 소비하지 않음을
실측했으므로 5단계 이후에 일어나도 안전하다. 이 순서를 어기고(예: RNG
복원 뒤에 `torch.randn()`이나 새 model 생성 같은 다른 RNG 소비 작업을
끼워 넣으면) exact resume이 깨진다 -- `scripts/run_resume_training_e2e.py`
와 `tests/training/test_loop.py::test_run_training_resume_matches_
continuous_run_exactly`가 정확히 이 순서를 그대로 실행해서 증명한다.

## 7. Exact resume 보장 조건 (범위를 벗어난 주장을 하지 않음)

**아래 조건에서 검증되었다** (일반적인 모든 CPU/PyTorch 환경에 대한
보편적 bitwise determinism 주장이 아니다):

```text
- CPU training (device="cpu")
- num_workers=0
- 이 저장소에 고정된 PyTorch/torchvision 버전, 동일한 실행 환경
- 동일한 ModelSpec, 동일한 dataset/split, 동일한 batch_size
- 동일한 optimizer/scheduler 설정(4절의 "반드시 일치" 필드)
- run_training() 호출 경계(epoch 경계)에서만 checkpoint
- DataLoader shuffle generator state 복원
- CPU RNG state 복원
```

이 조건에서, 연속 실행과 "checkpoint 저장 후 resume 실행"의 다음
항목에 대해 **tensor-level exact equality**(`torch.equal`/정확한
`==`)를 목표로 했고, 실제 테스트/E2E로 고정했다:

```text
model parameters, optimizer state tensors, scheduler state,
TrainingHistory 값(train_losses/val_losses/val_accuracies),
best_state_dict, best_epoch, best_val_loss, epochs_without_improvement
```

CUDA RNG state와 batch-level(sampler iterator/worker) 상태는 저장하지
않는다 -- 3개 기존 E2E 스크립트 전부 `run_training(..., device="cpu")`로
학습이 **항상 CPU에서만** 실행됨을 grep으로 확인했고(CUDA는 이후 C++
parity 단계에서만 등장), `num_workers=0`이라 sampler/worker 상태
자체가 존재하지 않는다. `random`/`numpy`도 `training/*.py` 전체에서
전혀 쓰이지 않음을 grep으로 확인했다 -- 저장할 상태 자체가 없다.

## 8. `stopped_early=True` checkpoint -- 조회는 허용, resume만 거부

**checkpoint "조회/가중치 추출"과 "resume 실행 가능 여부"는 서로 다른
질문이다.** `history.stopped_early=True`인 checkpoint라도
`load_training_checkpoint()`는 이를 거부하지 않고 payload를 그대로
반환한다 -- 이 함수의 책임은 "이 파일이 구조적으로 유효한
checkpoint인가"까지다. 사용자가 `payload["best_state_dict"]`나
`payload["model_state_dict"]`를 공식 API로 꺼내 새 모델에
`model.load_state_dict(...)`로 로드해 쓰는 것은 정당한 용도이므로 막을
이유가 없다:

```python
payload = load_training_checkpoint(path)  # stopped_early=True여도 성공
model.load_state_dict(payload["best_state_dict"])  # 정상 동작
```

("이 로드가 정상적으로 가능해야 한다"는 계약을
`tests/training/test_checkpoint.py::test_load_training_checkpoint_
allows_stopped_early_for_weight_extraction`이 직접 실행해서 고정한다.)

**resume 실행 자체를 거부하는 것은 완전히 다른 두 계층**이다:

* `loop.py`의 `TrainingResumeState.__post_init__` -- `stopped_early=True`
  history로 `TrainingResumeState`를 만드는 시점 자체를 거부
* `loop.py`의 `run_training()` 진입 시 -- `TrainingResumeState`는
  frozen dataclass가 아니라 생성 이후 `resume_state.history.
  stopped_early = True`처럼 직접 mutate될 수 있으므로, `run_training()`
  이 다시 한번 방어한다

그 가중치에서 새로 학습을 시작하려면, checkpoint의 `best_state_dict`
또는 `model_state_dict`를 `model.load_state_dict(...)`(PyTorch
`nn.Module`의 메서드)로 새 모델에 로드하고 새 `TrainingConfig`로 새
학습을 시작하면 된다(수준 A, 기존 Phase 4A 기능으로 이미 가능) --
에러 메시지에 이 대안을 함께 안내한다. 이건 `training.checkpoint.
load_state_dict(model, path)`(파일 경로를 받는 이 프로젝트의 helper)
와는 다른 함수이므로 혼동하지 않도록 에러 메시지/문서 모두
`model.load_state_dict(...)`처럼 명시적으로 `model.`을 붙여 표기한다.

## 9. Checkpoint payload와 validation

```python
CHECKPOINT_FORMAT_VERSION = 1

payload = {
    "format_version": 1,
    "model_state_dict": ...,      # 현재(마지막 완료 epoch) 모델
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,  # None 가능
    "history": ...,                # TrainingHistory를 asdict()한 순수 dict
    "best_state_dict": ...,
    "epochs_without_improvement": ...,
    "training_config": ...,        # TrainingConfig를 asdict()한 순수 dict (정보 + 호환성 검증용)
    "loader_generator_state": ...,
    "cpu_rng_state": ...,
}
```

`torch.save(payload, path)`/`torch.load(path, weights_only=True)`로
왕복 가능함을 확인했다(nested dict/list/tensor/스칼라는
`weights_only=True`의 허용 범위 안에 있다 -- 기존 `load_state_dict()`가
쓰던 것과 동일한 안전한 로드 방식을 그대로 유지). `TrainingConfig`/
`TrainingHistory` 인스턴스를 그대로 넣지 않고 `asdict()`로 순수 dict화
해서 넣는다 -- 새로운 커스텀 클래스를 pickle에 노출시키지 않기
위함이다.

`load_training_checkpoint()`가 구조적으로 검증하는 것:

```text
- 로드된 객체가 dict인지, "format_version" 키가 있는지 (없으면 bare state_dict로 착각한 것일 수 있음을 안내)
- format_version == CHECKPOINT_FORMAT_VERSION (다르면 명확히 거부, migration 코드는 없음 -- 지금 버전이 1개뿐이므로)
- 필수 key 전부 존재
- history가 dict이고, 6개 필수 필드(train_losses/val_losses/val_accuracies/
  best_epoch/best_val_loss/stopped_early)를 전부 갖고 있는지
- history 세 리스트 길이 일치
- training_config가 dict이고, RESUME_CONFIG_FIELDS 7개를 전부 갖고 있는지
- epochs_without_improvement가 0 이상의 정수인지
- loader_generator_state/cpu_rng_state가 각각 torch.Tensor인지
- training_config.lr_scheduler와 scheduler_state_dict 존재 여부의 내적 일관성
  (scheduler 있는데 state 없음, 또는 그 반대는 파일 자체가 손상된 것으로 간주)
```

**`stopped_early=True`는 더 이상 여기서 거부하지 않는다** (8절) --
model_state_dict/best_state_dict 조회는 stopped_early 여부와 무관하게
항상 가능해야 하기 때문이다. resume 실행 자체를 막는 것은
`TrainingResumeState`/`run_training()`의 책임이다.

model_state_dict/optimizer_state_dict/best_state_dict 내부 텐서의
shape나 optimizer state의 세부 구조는 검사하지 않는다 -- 그건
`model.load_state_dict()`/`optimizer.load_state_dict()`가 이미 명확한
에러로 검증해 주므로 다시 구현할 필요가 없다(원칙: PyTorch가 이미 잘
검증하는 것을 다시 재구현하지 않는다).

학습 설정(optimizer/learning_rate 등) **호환성은 여기서 검사하지
않는다** -- 이 함수는 이번에 resume에 쓸 새 `TrainingConfig`를 모르기
때문이다. 그 검증은 `run_training()`이 `resume_state`를 받을 때 항상
스스로 강제한다(4절) -- caller가 `require_compatible_resume_config()`
를 직접 호출하는 것은 조기 검증용 선택 사항일 뿐이다.

`load_state_dict()`에도 2줄짜리 방어를 추가했다 -- 로드한 객체가
`"format_version"` 키를 가진 dict면 "이건 full checkpoint인 것 같다,
`load_training_checkpoint()`를 쓰라"고 안내하고 거부한다. 두 함수
이름이 비슷해 실제로 헷갈리기 쉬운 지점이라 예외적으로 정당화했다
(다른 곳에서는 이런 선제적 방어를 추가하지 않았다).

payload 조립 helper(`TrainingResumeState`를 dict에서 자동으로 만들어
주는 함수)는 만들지 않았다 -- 지금 이걸 쓰는 곳이
`scripts/run_resume_training_e2e.py` 하나뿐이라 아직 중복이
발생하지 않고 있고, 4~5줄짜리 조립 코드를 추상화할 만큼의 무게가
없다고 판단했다.

## 10. `run_training()` / `TrainingResumeState` / `TrainingResult` 변경

외부 시그니처는 선택 인자 하나만 추가했다:

```python
def run_training(
    model, train_loader, val_loader, config, device="cpu",
    resume_state: TrainingResumeState | None = None,   # 신규, 기본값 None
) -> TrainingResult:
```

`resume_state=None`(기본값)이면 Phase 4E까지의 동작과 100% 동일 --
기존 3개 E2E 스크립트, `tests/training/test_loop.py`의 기존 테스트가
전부 코드 수정 없이 그대로 통과했다(회귀 결과는 14절).

`resume_state`가 주어지면:

```text
1. resume_state.history.stopped_early를 다시 한번 확인 (8절)
2. require_compatible_resume_config(resume_state.training_config, config)로
   config 호환성을 강제 (4절) -- optimizer/scheduler state를 로드하기 전에 확인
3. optimizer/scheduler를 config로 새로 생성
4. optimizer/scheduler에 resume_state의 state_dict를 deepcopy 후 로드
   (deepcopy하는 이유: optimizer.load_state_dict()가 내부 텐서를
   aliasing한다는 것을 실측 확인했다 -- deepcopy 없이 로드하면 이후
   optimizer.step()이 caller가 들고 있는 resume_state/checkpoint
   payload의 텐서를 조용히 변형시킬 수 있다)
5. history = copy.deepcopy(resume_state.history) -- 원본 객체를 직접
   수정하지 않는다 (caller가 같은 resume_state를 나중에 다시 참조할
   수 있어야 하므로)
6. best_state_dict / best_val_loss / best_epoch을 resume_state에서 이어받음
7. epochs_without_improvement를 resume_state에서 이어받음 (0으로 리셋하지 않음)
8. completed_epochs = len(history.train_losses)
9. epoch 번호를 completed_epochs+1부터 시작, config.epochs만큼 추가 실행
```

`TrainingResumeState`(신규, `loop.py`)는 optimizer/scheduler state,
이전 history, early stopping 카운터, best snapshot, **checkpoint 저장
당시 config**(`training_config: dict`)를 하나로 묶는 작은 dataclass다
-- `run_training()`에 필요한 재개용 입력이 6개나 되어 개별 키워드
인자로 늘어놓으면 시그니처가 지저분해지기 때문에 추가했다(확장성을
위해서가 아니라 인자 개수 문제 해결을 위한 최소한의 묶음).
`training_config` 필드는 config 호환성 강제(위 2단계)에 필요해서
추가됐다 -- 이게 없으면 caller가 별도로 넘겨야 하는데, 그러면
caller가 그 인자를 빼먹었을 때 검증 자체가 우회될 수 있다. 생성
시점에 `history` 일관성/`stopped_early` 등을 자체 검증한다(3·8절).

`TrainingResult`(기존)에 3개 필드를 추가했다: `optimizer_state_dict`,
`scheduler_state_dict`, `epochs_without_improvement`. 전부 함수 종료
시점의 독립적인 deepcopy snapshot이다. `TrainingResult`는 코드
전체에서 `run_training()` 내부에서만 생성되므로(다른 생성 지점 없음)
필드 추가에 따른 외부 호환성 위험이 없다.

별도 `run_training_resume()` 함수는 만들지 않았다 -- 그러면 train/val/
history/best/early-stopping 루프 로직이 두 함수에 중복되어 앞으로
이 로직을 고칠 때마다 두 곳을 같이 고쳐야 하는 회귀 위험이 생긴다.
optimizer/scheduler를 caller가 직접 만들어 넘기는 방식도 채택하지
않았다 -- `TrainingConfig -> optimizer/scheduler` 생성 책임이 지금
전적으로 `_build_optimizer`/`_build_scheduler`에 있는데, 이걸 caller로
옮기면 그 로직이 중복되거나 caller가 private 함수를 다시 import해서
쓰는 어색한 결합이 생긴다.

## 11. `checkpoint.py`/`config.py` 구조와 `require_compatible_resume_config`의 위치

기존 `save_state_dict()`/`load_state_dict()`는 **한 줄도 바꾸지
않았다**(`load_state_dict()`에 추가한 2줄짜리 방어 제외). 새 함수
`save_training_checkpoint()`/`load_training_checkpoint()`를
`checkpoint.py`에 추가했다 -- 별도 `full_checkpoint.py` 파일을 만들지
않은 이유는 `checkpoint.py`가 이미 "학습된 텐서 상태를 저장하는 곳"
이라는 역할을 갖고 있어 그 역할을 살짝 넓히는 것으로 충분하기
때문이다. 기존 `save_state_dict()`/`load_state_dict()`를 이 payload
포맷으로 바꾸는 것(포맷 자체를 변경)은 처음부터 배제했다 -- 3개 E2E
스크립트와 TorchScript export 흐름이 이미 이 함수들을 bare state_dict
전제로 쓰고 있어(총 6곳 이상 호출), 포맷을 바꾸면 전부 깨진다.

`require_compatible_resume_config()`(와 `RESUME_CONFIG_FIELDS` 튜플)는
**`config.py`에 둔다.** 처음에는 `checkpoint.py`에 있었지만, "이
검증을 `run_training()`(`loop.py`)이 항상 스스로 강제해야 한다"는
요구를 반영하면서 옮겼다 -- `loop.py`가 이 함수를 호출하려면
`checkpoint.py`를 import해야 하는데, `checkpoint.py`는 이미
`loop.py`의 `TrainingResult`를 import하고 있어(`loop.py -> checkpoint.py`
방향은 없지만 `checkpoint.py -> loop.py`는 있음) `loop.py ->
checkpoint.py`를 추가하면 **순환 의존**이 생긴다. `config.py`는
`loop.py`/`checkpoint.py` 어느 쪽도 import하지 않는 가장 아래 계층이라,
여기 두면 `loop.py`(`run_training()`)와 `checkpoint.py`(파일 검증)
양쪽이 순환 없이 같은 함수를 가져다 쓸 수 있다:

```text
loop.py       -> config.py   (require_compatible_resume_config 등 사용)
checkpoint.py -> config.py   (〃)
checkpoint.py -> loop.py     (TrainingResult 타입 사용)
loop.py       -X-> checkpoint.py   (이 방향은 없음 -- 있었다면 순환 의존)
```

즉 `config.py`가 최하위 계층(설정 정의 + resume 호환성 검증)이고,
`loop.py`는 `config.py`만 import하며, `checkpoint.py`는 `config.py`와
`loop.py`를 둘 다 import한다. `loop.py`가 `checkpoint.py`를 import하는
방향은 존재하지 않으므로 순환이 생기지 않는다.

`checkpoint.py`는 이 함수를 다시 구현하지 않고 `config.py`에서 import한
뒤 그대로 re-export한다 (`from image_ai_studio.training.checkpoint
import require_compatible_resume_config`로도 여전히 쓸 수 있도록) --
로직이 두 곳에 중복되지 않는다.

`checkpoint.py`가 `TrainingConfig`/`TrainingResult`를 다루기 위해
`config.py`/`loop.py`를 import한다 -- `history.py`가 이미 `loop.py`를
import하고 있는 것과 정확히 같은 의존 방향이라 새로운 결합 패턴이
아니다. `checkpoint.py`는 여전히 `model_definition`을 import하지
않는다(단방향 경계 유지).

## 12. 신규 E2E: `scripts/run_resume_training_e2e.py`

책임은 오직 하나 -- resume exactness 증명. `examples/models/
phase4_training_model.json`(Dropout + BatchNorm 포함, `run_training_e2e.py`
가 이미 검증에 쓰는 모델)과 `training/dataset.py`의 synthetic
dataset(네트워크 불필요)을 사용한다. `SGD + momentum + ReduceLROnPlateau`
조합으로 실행해 더 많은 상태 종류를 실제로 왕복시킨다.

```text
(a) 연속 5 epoch 실행
(b) 3 epoch 실행 -> save_training_checkpoint() 실제 파일 저장
    -> 새 model/DataLoader/generator 생성(새 프로세스를 흉내)
    -> load_training_checkpoint() 실제 파일 로드
    -> require_compatible_resume_config()로 조기 설정 호환성 확인
       (선택 사항 -- 생략해도 이후 run_training()이 다시 강제한다, 4절)
    -> DataLoader generator state 복원, CPU RNG state 복원
    -> 2 epoch resume
(a)와 (b)의 model/optimizer/scheduler state, history, best_state_dict,
best_epoch/best_val_loss, epochs_without_improvement를 항목별로 비교
```

**`run_resume_training_e2e.py` 자체는 TorchScript export/C++ parity를
전혀 수행하지 않는다.** Phase 4F는 export/parity 경로(`export/*`,
`parity/*`, C++ 코드)를 전혀 수정하지 않았으므로, 그 경로의 검증은
기존 3개 E2E(`run_training_e2e.py`/`run_real_training_e2e.py`/
`run_imagefolder_training_e2e.py`)를 그대로 재실행해 **회귀가 없는지만**
확인했다(13절) -- Phase 4F의 resume 기능 자체가 C++에서 실행되거나
검증되는 것이 아니다. 기존 3개 E2E 스크립트는 이번 Phase에서 수정하지
않았다. ImageFolder E2E에 `--resume-from` 등을 추가하는 것도 이번
Phase에서는 하지 않았다 -- 먼저 core resume 계약과 synthetic exact
E2E를 완성하는 데 집중했다.

## 13. 실제 실행 검증 결과

Windows 11, PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, GTX 1080에서
전부 실제로 실행하여 확인했다 (추정치 없음):

* **`tests/training/test_config.py`**: 30 passed
* **`tests/training/test_loop.py`**: 39 passed
* **`tests/training/test_checkpoint.py`**: 29 passed
* **`tests/training/` 전체**: 140 passed
* **전체 `pytest`**: 297 passed
* **Phase 0 regression**: `tiny_cnn`/`tiny_residual_cnn` CPU/CUDA 전부 PASS
* **Phase 1~3 E2E regression**: 4개 예시 JSON 전부 PASS
* **Phase 4A/4B synthetic E2E** (`run_training_e2e.py`, 수정 없음):
  기존과 완전히 동일한 수치(train loss 1.3386 -> 0.2867, best epoch 10) 재현
* **Phase 4C CIFAR-10 E2E** (`run_real_training_e2e.py`, 수정 없음):
  기존과 완전히 동일한 수치(best epoch 4, test_accuracy=0.1953) 재현
* **Phase 4D ImageFolder E2E** (`run_imagefolder_training_e2e.py`,
  수정 없음): 기존과 완전히 동일한 수치(best epoch 5, test_accuracy=0.2600) 재현
* **신규 Phase 4F Resume E2E** (`run_resume_training_e2e.py`): 연속
  5 epoch vs 3 epoch+checkpoint+resume 2 epoch 비교 결과 -- model
  state/optimizer state/scheduler state/history(train_losses/
  val_losses/val_accuracies)/best_state_dict/best_epoch/best_val_loss/
  epochs_without_improvement **10개 항목 전부 PASS**
* **신규 scheduler patience-boundary unit test**
  (`test_run_training_resume_scheduler_lr_reduction_crosses_checkpoint_
  boundary`): `lr_scheduler_patience=2`, val_loss를 계속 1.0으로 고정한
  deterministic 시나리오 -- 연속 4 epoch에서는 4번째 epoch에서 LR이
  1.0 -> 0.5로 감소(bad epoch 3회 > patience 2, `test_build_scheduler_
  reduces_lr_after_patience_bad_steps`가 이미 실측한 호출 순번과 동일한
  근거). (3 epoch 실행 + resume 1 epoch)에서도 동일하게 4번째 epoch
  (resume 후 1번째)에서 LR이 0.5로 감소하고, `scheduler_state_dict`
  전체가 두 경로에서 정확히 일치함을 확인 -- "state_dict가 왕복된다"를
  넘어 "num_bad_epochs 카운트가 checkpoint 경계를 넘어 정확히
  이어진다"는 것 자체를 증명

```text
(a) Continuous:  epoch1~5 train_loss = 1.629621, 1.408341, 1.248359, 1.139046, 1.017424
(b) Resume:      epoch4~5 train_loss = 1.139046, 1.017424  (연속 실행의 epoch4~5와 정확히 일치)
best_epoch=5, best_val_loss=1.220610 (양쪽 동일)

PHASE 4F E2E: PASS
```

## 14. 의도적으로 구현하지 않은 것

```text
- batch-level resume (sampler iterator/worker 상태) -- num_workers=0이라 애초에 존재하지 않음
- epoch callback/자동 checkpoint 저장 -- run_training()은 여전히 파일을 쓰지 않음
- CUDA RNG state 저장 -- 학습이 CPU 전용으로 하드코딩되어 있어 검증 불가능한 죽은 코드가 됨
- multi-worker(num_workers>0) 지원
- distributed checkpoint
- config override(learning_rate 등 자유 변경) -- optimizer.load_state_dict()가
  덮어써버리므로 이번 Phase에서는 강제 일치로 처리
- ImageFolder E2E의 --resume-from CLI -- core resume 계약을 먼저 확정하는 데 집중
```

CUDA를 학습에서 실제로 지원하게 되면, `format_version`이 이미 있으므로
그때 버전을 올리고 `cuda_rng_state`(`torch.cuda.get_rng_state_all()`)
필드를 추가하면 된다 -- 지금 미리 넣어봐야 검증할 방법이 없는 필드를
payload에 얹는 것이므로 넣지 않았다.
