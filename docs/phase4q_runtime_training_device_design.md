# Phase 4Q: Runtime Training Device Exposure — 설계안

## 1. 목적

ImageFolder 학습에 CPU/CUDA/CUDA:N device를 명시적으로 선택할 수 있게
한다. generic training core(`loop.py`)는 Phase 4A~4P를 거치며 이미
`device` 파라미터를 완비해 뒀으므로(§2), 이번 Phase의 실제 작업은 그
파라미터를 ImageFolder CLI/workflow 계층에서 실제로 선택 가능하게
연결하는 배선(wiring)이다.

## 2. 조사로 확인한 기존 구조

`loop.py`의 `_build_criterion(config, device)`, `train_one_epoch(...,
device)`, `evaluate(..., device)`, `evaluate_classification_metrics(...,
device)`, `run_training(..., device)`는 전부 이미 device를 받아 배치마다
`.to(device)`를 수행하도록 구현돼 있었다(Phase 4A~4P가 매번 `device="cpu"`
로만 호출했을 뿐 시그니처 자체는 완성돼 있었음) — **`loop.py`는 이번
Phase에서 한 글자도 수정하지 않는다.**

반면 `imagefolder_workflow.py`는 `run_training(...)`/
`evaluate_classification_metrics(...)` 호출부 두 곳에 `device="cpu"`가
하드코딩돼 있었고, **`model.to(device)` 호출이 어디에도 없었다** —
지금까지는 `build_model()`이 항상 CPU 모델을 만들고 `device="cpu"`만
쓰였기 때문에 이 누락이 드러나지 않았을 뿐이다. 이 누락을 메우는 것이
이번 Phase의 핵심 production 변경이다.

## 3. PyTorch 실측 (로컬 CUDA, GPU 1개)

**cross-device state 이관**: GPU에서 1 step 학습한 `optimizer.state_dict()`
(Adam의 `exp_avg`가 `cuda:0`)를 CPU model 기반의 새 CPU optimizer에
`load_state_dict()`하면 자동으로 `cpu`로 이관되고 `.step()`이 정상
동작함을 확인했다. 반대 방향(CPU 저장 state → GPU optimizer)도 자동으로
`cuda:0`로 이관되고 정상 동작함을 확인했다. `model.load_state_dict()`도
방향과 무관하게 자기 자신의 현재 device를 유지한 채 값만 올바르게
복사됨을 확인했다(`Tensor.copy_()`의 cross-device 지원). **따라서 이
프로젝트가 별도 device migration 코드를 작성할 필요가 없다.**

**device 문자열 검증**: `torch.device("CUDA:0")`, `torch.device("cuda:00")`,
`torch.device("CPU")`가 전부 `RuntimeError`로 거부됨을 확인했다(대소문자
구분, zero-padding 거부) — `torch.device()` 자체가 이미 엄격해서 별도
canonicalization parser가 불필요하다. 다만 `torch.device()`는 `mps`/
`xpu`/`hip` 등 이 프로젝트가 검증하지 않은 backend도 그대로 통과시키므로,
프로젝트가 직접 허용 목록(`cpu`/`cuda`/`cuda:N`)을 관리한다(§5).

**out-of-range CUDA index**: `torch.device("cuda:99")` 자체는 구성
시점에 에러가 나지 않고, 실제 `.to("cuda:99")` 호출에서야 저수준
`AcceleratorError: CUDA error: invalid device ordinal`가 남을 확인했다.
이 backstop에 의존하지 않고 `torch.cuda.device_count()`로 조기 검증한다
(§5).

## 4. device ownership

`ImageFolderWorkflowRequest.device: str = "cpu"`를 추가했다.
`TrainingConfig`에는 두지 않는다 — `device`는 학습 objective를 바꾸는
hyperparameter가 아니라 "이번 실행을 어떻게 돌릴지"를 나타내는 runtime
파라미터이고, `seed`처럼 `TrainingConfig` 밖에서 관리되는 run-level
파라미터라는 점에서 같은 계층이다(`seed`는 initialization/RNG 결과
자체에 영향을 주고 `device`는 execution backend를 정하는 값이라 완전히
동일한 개념은 아니지만, 둘 다 checkpoint의 `training_config`/
`RESUME_CONFIG_FIELDS` 비교 대상이 아니라는 architecture적 결론은
같다 -- `--seed`의 기존 docstring도 "resume 시 사실상 무시됨"이라고
명시하는 실행 전용 파라미터다). `run_training()` 자신이 애초에 `device`를
`TrainingConfig`가 아니라 독립 파라미터로 받아 온 사실(Phase 4A부터)이
이 설계를 저장소 자체가 이미 증명하고 있다. `TrainingConfig`에 두면
checkpoint의 `training_config`에 저장되어 "hyperparameter인가 실행
환경인가" 혼동과 `RESUME_CONFIG_FIELDS` 포함 여부 판단 부담이 불필요하게
생긴다.

## 5. device validation

`imagefolder_workflow.py`에 `_validate_device(value: str) -> None`을
추가했다:

```python
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda|cuda:(0|[1-9][0-9]*))$")

def _validate_device(value: str) -> None:
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
            raise ValueError(f"device={value!r} is out of range -- torch.cuda.device_count()=={device_count}")
```

정규식이 `cuda:(0|[1-9][0-9]*)`로 leading zero를 이미 거부하므로
`torch.device()`를 추가 형식 검증에 쓰지 않았다(과도한 이중 검증
회피). `ValueError`를 그대로 사용했다 — `device`가 `TrainingConfig`
필드가 아니므로 `TrainingConfigError`를 쓰지 않았고,
`require_matching_num_classes()`/`_validate_checkpoint_every()`가 이미
이 모듈에서 `ValueError`를 쓰는 기존 패턴을 그대로 재사용했다(새
exception class를 만들지 않음). CUDA 미가용/index 범위 초과는 CPU로
조용히 대체하지 않고 학습 시작 전에 명확히 거부한다 — 이 프로젝트가
`learning_rate`/`label_smoothing`/`class_weights` 전부에서 지켜온 strict
validation 철학과 동일하다.

**안정화 수정**: 최초 구현은 `_DEVICE_PATTERN.match(value)`를 썼는데,
Python `re`의 `$`는 문자열 끝뿐 아니라 trailing newline 직전에도 매치될
수 있어 `match("cpu\n")`가 `True`를 반환함을 직접 실측 확인했다(strict
syntax validation 의도와 어긋남). `fullmatch()`로 교체해 문자열 전체가
패턴과 정확히 일치할 때만 통과하도록 강화했고, `"cpu\n"`/`"cuda\n"`/
`"cuda:0\n"`을 거부 케이스로 회귀 테스트에 추가했다. 정규식 anchors
(`^`/`$`)는 그대로 두고 최소 diff로 처리했다(재작성 불필요).

## 6. model/device lifecycle

```
_prepare_resume(request, model_spec, splits)   # 항상 CPU model build/load
    ↓
model = model.to(request.device)                # Phase 4Q 신규 -- optimizer 생성 전
    ↓
DataLoader 생성 (device 무관, CPU tensor)
    ↓
run_training(model, ..., device=request.device, ...)
    ↓ (run_training() 내부)
    optimizer = _build_optimizer(model, config)  # model.parameters()가 이미 target device
    criterion = _build_criterion(config, device=device)  # Phase 4P의 class_weights도 자동으로 같은 device
    ↓
best_model = build_model(model_spec)             # 항상 CPU(무수정)
best_model.load_state_dict(training_result.best_state_dict)  # cross-device 자동 처리(§3)
    ↓
evaluate_classification_metrics(best_model, test_loader, ..., device="cpu")  # request.device 전달하지 않음(무수정)
    ↓
TorchScriptExporter().export(best_model, example_input, ...)  # CPU model + CPU example_input(무수정)
```

`model.to(device)`는 `_prepare_resume()` 직후, DataLoader/optimizer
생성 전에 위치한다 — PyTorch semantics상 optimizer는
`model.parameters()`가 가리키는 실제 tensor를 참조하므로, `.to()`는
optimizer 생성보다 반드시 먼저여야 새 device의 parameter가 optimizer에
올바르게 등록된다. `run_training()`은 이미 이 순서를 스스로 지키고
있으므로(§2), workflow가 `run_training()` 호출 전에 model을 옮기기만
하면 된다(`run_training()` 자신의 기존 docstring 계약: "model은 호출
전에 이미 device로 옮겨져 있어야 함").

`_prepare_resume()`가 CPU model을 build하고 checkpoint state를 CPU로
load하는 기존 구조는 그대로 유지했다(수정 없음) — `.to(device)`는 그
함수의 반환값을 받은 뒤 `run_imagefolder_training_workflow()`에서
수행한다.

## 7. checkpoint map_location — 무수정 확인

`load_training_checkpoint()`/`load_state_dict()`는 이미
`map_location: str = "cpu"` 기본값을 가지며, 실제 호출부
(`_prepare_resume()`)도 이 기본값을 그대로 쓴다. `save_training_checkpoint()`
/`save_state_dict()`는 `.cpu()` 변환 없이 `torch.save()`만 수행하므로
GPU 학습 중 저장된 checkpoint에는 CUDA tensor가 그대로 들어갈 수 있지만,
로드 시 `map_location="cpu"` 기본값으로 항상 CPU로 복원된다. 정확히
말하면: (1) checkpoint loader가 `map_location="cpu"`를 이미 기본값으로
쓰고 있고, (2) model/optimizer state의 CPU↔CUDA 이관 자체가 정상
동작함을 §3에서 PyTorch 레벨로 실측했으며, (3) 대표 경로인 CPU→CUDA
workflow resume을 실제 smoke test로 확인했다(§13) — 이 세 가지를 종합하면
**checkpoint state의 device portability에 필요한 기본 메커니즘은 이미
갖춰져 있다.** 이 근거로 `checkpoint.py`는 이번 Phase에서 한 글자도
수정하지 않았다.

## 8. CUDA RNG state — 저장하지 않음(핵심 gate)

`torch.cuda.get_rng_state_all()`/`torch.cuda.set_rng_state_all()`는
코드베이스 어디에도 없다. `imagefolder_workflow.py`의 `_set_seed()`가
`torch.cuda.manual_seed_all(seed)`를 호출하지만, 이는 fresh 학습
시작 시 1회 초기 시드 설정일 뿐 checkpoint/resume 메커니즘과 무관하다
(이번 Phase에서도 이 함수는 무수정). checkpoint가 저장/복원하는 RNG는
`cpu_rng_state`(`torch.get_rng_state()`)와 `loader_generator_state`
(DataLoader의 CPU `torch.Generator`) 둘뿐이다.

**결론**: CUDA 학습 중 RNG를 소비하는 어떤 연산(Dropout 등)이든 그
스트림 상태를 복원할 방법이 없으므로, CUDA를 포함하는 어떤 resume
조합도 bitwise exact-resume으로 공식 주장할 수 없다.

## 9. exact-resume vs cross-device portability — 정확한 구분

* **CPU → CPU**: 기존 exact-resume 계약(tensor-level exact equality)을
  완전히 그대로 유지한다. `run_resume_training_e2e.py` 재실행으로
  무영향을 확인했다(§14).
* **CUDA가 포함되는 resume**: model/optimizer state가
  device-portable한 구조를 갖고 있음을 low-level PyTorch 실측(§3,
  GPU↔CPU 양방향 `optimizer.load_state_dict()`/`model.load_state_dict()`
  가 자동으로 device를 이관하고 `.step()`까지 정상 동작)으로 확인했고,
  Phase 4Q workflow regression에서는 **대표 경로인 CPU→CUDA resume
  하나**를 optional CUDA smoke test로 직접 고정했다(§13). CUDA→CPU
  전체 workflow resume과 CUDA→CUDA resume은 dedicated workflow
  regression test로 별도 고정하지 않았다 — §3의 실측이 양방향 모두를
  대상으로 했으므로 동작 자체는 개연성이 높지만, "모든 방향의
  workflow가 pytest로 검증됨"이라고 과장하지 않는다. 어느 방향이든
  §8의 CUDA RNG gate 때문에 **bitwise exact는 공식 주장하지 않는다.**

`portable`(=underlying model/optimizer state를 다른 device에서
load/resume할 수 있다)과 `모든 방향의 workflow resume이 각각 dedicated
pytest로 고정됨`을 같은 의미로 쓰지 않는다. 잘못된 표현("GPU에서도
exact resume을 지원한다")은 코드/문서 어디에도 쓰지 않았다. 정확한
표현("CUDA를 포함한 resume은 model/optimizer state portability
차원에서 지원하지만, CUDA RNG state를 checkpoint하지 않으므로 bitwise
exact-resume은 이번 Phase의 보장 범위가 아니다")을 README/이
문서 전체에서 일관되게 사용했다.

## 10. final test/export — CPU 경계 무수정

`TorchScriptExporter.export()`를 직접 읽은 결과, `example_input =
example_input.to("cpu")`로 입력을 **강제로 CPU에 고정**하지만 `model`
자체는 이동시키지 않는다 — 즉 "model이 이미 CPU에 있다"는 암묵적
가정을 갖고 있다(model이 CUDA에 있으면 `model(example_input)`에서
device mismatch `RuntimeError`가 남). 이 가정은 `best_model`이
`build_model(model_spec)`으로 항상 새로 만들어져 어떤 `.to()`도 거치지
않으므로(무수정) 계속 성립한다. 따라서:

* `evaluate_classification_metrics(best_model, test_loader, ...,
  device="cpu")` 호출부는 `request.device`를 전달하지 않고 **그대로
  유지**했다(명시적 주석 추가, imagefolder_workflow.py 참고).
* `TorchScriptExporter`/`export/*` production 코드는 무수정.
* C++ parity(`tools/run_and_compare.py`)는 export된 TorchScript
  아티팩트만 다루므로 Python 학습 device와 완전히 무관 — 무수정.

## 11. CLI

`--device DEVICE`(기본값 `cpu`)를 추가했다. `argparse`의 `choices`로는
`cuda:N`을 전부 열거할 수 없으므로 실제 validation은 workflow entry
(`_validate_device`)에서 수행한다 — CLI는 문자열 전달만 담당하고,
Python API(`ImageFolderWorkflowRequest(..., device=...)`)로 직접
호출하는 경로도 동일하게 검증된다(CLI에만 validation을 두지 않음).
`Device: {args.device}` stdout 한 줄을 `Model JSON`/`Dataset root`/
`Resume from`/`Checkpoint out`과 같은 위치에 추가했다 — 새로 계산된
지표가 아니라 사용자가 방금 지정한 값의 echo이므로, Phase 4O가 확립한
"상세 metric으로 stdout을 확대하지 않는다"는 원칙과 충돌하지 않는다.

## 12. Phase 4P class_weights 연동

`_build_criterion(config, device)`가 이미 device를 받으므로,
`run_imagefolder_training_workflow()`가 `device="cpu"` 대신
`request.device`를 `run_training()`에 전달하는 것만으로 class_weights
tensor가 자동으로 올바른 device에 생성된다 — `_build_criterion()`을
우회하지 않는다는 계약을 optional CUDA smoke test로 직접 확인했다(§13).

## 13. 테스트 전략

* **CPU-only**: `ImageFolderWorkflowRequest` 생성 시 `device`
  생략 → `"cpu"` 기본값 확인. `_validate_device()` 직접 단위 테스트 --
  정상(`cpu`), 잘못된 syntax(`gpu`/`CUDA`/`CPU`/`mps`/`xpu`/`hip`/
  `cuda:`/`cuda:-1`/`cuda:00`/빈 문자열/비-문자열), `torch.cuda.is_available`
  monkeypatch로 CUDA 미가용 시 `cuda`/`cuda:0` 거부, `torch.cuda.device_count`
  monkeypatch로 index 범위 초과 거부, 조건 만족 시(`is_available=True,
  device_count=2`) `cuda`/`cuda:0`/`cuda:1` 정상 통과. workflow 레벨에서
  잘못된 device가 `run_training()` 호출 전에 거부됨을 monkeypatch로
  직접 증명(class_weights 길이 mismatch 조기 검증과 동일 패턴). `device`
  가 `run_training()`의 `device` kwarg로 실제 전달되고, 그 시점의
  `model`이 이미 그 device 위에 있는지(`next(model.parameters()).device`)
  spy로 확인 — "cpu" 경로로도 wiring 전체를 실측 가능. workflow의
  최종 detailed evaluation 호출(`evaluate_classification_metrics`)이
  request.device를 그대로 쓰지 않고 명시적으로 `device="cpu"`를 쓰는지
  spy로 확인(`request.device="cpu"`인 경우만 다룸 -- CUDA training 뒤
  최종 test/export 전체 경로가 실제로 CPU에서 정상 완료되는지는 아래
  optional CUDA smoke test (3)이 담당). CLI: `--device` forwarding/
  기본값/stdout echo/invalid syntax/CUDA 미가용/index 범위 초과가 clean
  failure(exit code 1)인지 확인.
* **optional CUDA(`pytest.mark.skipif(not torch.cuda.is_available(),
  ...)`)**: 이 프로젝트에 없던 skipif 패턴을 신규 도입했다. 로컬에 실제
  CUDA(1 GPU)가 있어 전부 실행/통과를 직접 확인했다(GPU 없는 CI에서는
  자동 skip). `pytest --collect-only`로 실측한 결과 **4개**로 구성했다:
  (1) `_build_criterion(..., device="cuda")`
  의 weight tensor가 실제 CUDA에 생성되는지, (2) `run_training(...,
  device="cuda")` 1 epoch가 성공하고 model parameter device가 cuda이며
  train/val loss가 finite인지, (3) ImageFolder workflow가 `device="cuda"`
  +`class_weights`로 학습→최종 test→TorchScript export까지(CPU
  best_model 기반) 한 번에 성공하는지(generic smoke + export boundary를
  이 테스트 하나로 커버, 중복 GPU 테스트를 늘리지 않음), (4) CPU에서
  저장한 checkpoint를 CUDA에서 resume했을 때 에러 없이 완료되는지(portability
  smoke, bitwise equality assertion 없음, CUDA→CPU 대칭 테스트는
  추가하지 않음 -- 한 방향으로 충분).
* CPU→CPU exact-resume은 기존 테스트를 복제하지 않고 전체 pytest +
  `run_resume_training_e2e.py` 재실행으로만 확인했다(§14).

## 14. E2E 전략

새 mandatory GPU E2E는 만들지 않았다. 기존 5개 E2E
(`run_phase1_e2e.py`, `run_training_e2e.py`, `run_real_training_e2e.py`,
`run_resume_training_e2e.py`, `run_imagefolder_training_e2e.py`)를
`device` 관련 수정 없이(기본값 CPU) 그대로 재실행해 numerical anchor가
전부 이전과 완전히 동일함을 확인했다(전부 PASS). `run_imagefolder_training_e2e.py`
에 CLI device flag를 추가하는 등 스크립트 API를 확장하지 않았다.

## 15. 제외 범위 (재확인)

CUDA exact-resume(CUDA RNG state checkpoint, `torch.use_deterministic_algorithms()`
등 결정론적 설정), AMP/mixed precision, gradient accumulation,
multi-GPU/distributed training, `mps`/`xpu`/`hip` 등 CUDA 외 backend,
학습 device와 다른 별도 evaluation device, artifact(TrainingHistory/
checkpoint metadata/test_result.json/class_mapping.json/TorchScript
metadata/ImageFolderWorkflowResult)에 device 기록, checkpoint schema
변경, GPU 성능 튜닝(pin_memory/num_workers).
