# Phase 6B: Single-Image Inference Core + Application/Qt Worker — 구현 결과

Phase 6A(docs/phase6a_inference_architecture.md)가 설계한 것을 그대로
구현했다. GUI는 없다 -- `InferencePage`/`MainWindow` 통합은 Phase 6C의
몫이다.

## 1. implemented files

```text
src/image_ai_studio/training/device.py                (신규, 순수 이동 리팩터)
src/image_ai_studio/inference/__init__.py               (신규)
src/image_ai_studio/inference/single_image_inference.py (신규)
src/image_ai_studio/application/inference_controller.py (신규)
src/image_ai_studio/gui/qt_inference_worker.py           (신규)

src/image_ai_studio/training/imagefolder_workflow.py    (수정 -- device.py에서 import만)
src/image_ai_studio/training/torchvision_dataset.py      (수정 -- load_class_mapping() 최소 구조 검증 추가)

tests/inference/test_single_image_inference.py                (신규, 21개)
tests/application/test_inference_controller.py                (신규, 10개)
tests/gui/test_qt_inference_worker.py                          (신규, 7개)
tests/gui/test_qt_inference_worker_integration.py               (신규, 2개)
```

## 2. public API

```python
# image_ai_studio.inference.single_image_inference
@dataclass(frozen=True)
class InferenceRequest:
    model_json_path: Path
    state_dict_path: Path
    class_mapping_path: Path
    image_path: Path
    device: str
    precision: str

@dataclass(frozen=True)
class InferenceResult:
    predicted_index: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    inference_duration_seconds: float

def run_single_image_inference(request: InferenceRequest) -> InferenceResult: ...


# image_ai_studio.application.inference_controller
InferenceControllerState = Literal["idle", "running", "finished", "failed"]

class InferenceAlreadyRunningError(RuntimeError): ...

def build_inference_request(*, model_json_path, state_dict_path, class_mapping_path,
                             image_path, device="cpu", precision="fp32") -> InferenceRequest: ...

class InferenceController:
    def __init__(self, backend: InferenceBackend = run_single_image_inference) -> None: ...
    @property
    def state(self) -> InferenceControllerState: ...
    @property
    def is_running(self) -> bool: ...
    def begin_run(self) -> None: ...
    def run(self, request: InferenceRequest) -> InferenceResult: ...


# image_ai_studio.gui.qt_inference_worker (PySide6 import는 이 파일에만)
class QtInferenceWorker(QObject):
    finished = Signal(object)  # InferenceResult
    failed = Signal(str)
    def __init__(self, controller: InferenceController, request: InferenceRequest) -> None: ...
    def run(self) -> None: ...  # QThread.started에 연결
```

## 3. loading flow(canonical)

```text
ModelSpec JSON
    -> load_model_spec()          [model_definition/serialization.py]
    -> validate_model_spec()       [model_definition/validation.py]  -- final_shape 확보
    -> class_mapping.json
       -> load_class_mapping()     [training/torchvision_dataset.py] -- 구조 검증 포함(§5)
       -> require_matching_num_classes(len(classes), final_shape)     -- 새 검증 없음, 기존 public 함수
    -> build_model()               [model_definition/builder.py]
    -> load_state_dict()           [training/checkpoint.py]
    -> model.to(device)
    -> model.eval()
```

`imagefolder_workflow.py`의 `_prepare_resume()`(fresh-model 경로)와
본질적으로 동일한 순서다 -- inference 전용 model builder는 만들지
않았다(Phase 6A §3/§5 계약 그대로).

## 4. preprocessing contract

`build_transform(model_spec.input_shape)`(`training/torchvision_dataset.py`,
public, 무수정)를 그대로 import해서 쓴다:

```python
image = Image.open(request.image_path).convert("RGB")
inputs = build_transform(model_spec.input_shape)(image).unsqueeze(0).to(request.device)
```

**preprocessing parity regression test**
(`tests/inference/test_single_image_inference.py::test_preprocessing_matches_training_imagefolder_exactly`)
가 이 계약을 고정한다 -- 정사각형이 아닌 `input_shape=(3,16,12)`와
단색이 아닌 gradient 이미지를 써서, 실제 `torchvision.datasets.
ImageFolder(transform=build_transform(...))`가 만드는 tensor와
single-image inference 경로가 만드는 tensor를 `torch.equal()`(근사
아닌 완전 일치)로 비교한다 -- 둘 다 PIL `.convert("RGB")` + 동일
`build_transform()` 호출이므로 bit-identical함을 실측으로 확인했다.
두 번째 테스트(`test_run_single_image_inference_uses_build_transform_end_to_end`)
는 `run_single_image_inference()` 전체를 통해서도(model의 `forward()`
를 monkeypatch해서 실제로 들어간 입력 tensor를 가로채) 같은 결과가
나오는지 재확인한다 -- "함수 조립"과 "실제 실행 경로" 양쪽에서 parity를
고정했다.

## 5. class_mapping 최소 구조 검증

Phase 6A가 발견한 gap(`load_class_mapping()`이 구조 검증 없이 그대로
반환)을 이번에 메웠다 -- **가장 작은 수정 지점**이라고 판단해
`load_class_mapping()` 자체(`training/torchvision_dataset.py`)에
`_require_valid_class_mapping()`을 추가했다(inference 쪽에 별도
wrapper를 만들지 않음, training/inference가 정확히 같은 함수를 공유).

검증 내용: `classes` 키 존재 + list + 비어있지 않음 + 모든 원소가
`str` + **classes 이름은 unique**(Phase 6B 마무리 라운드에서 추가 --
`class_to_idx`가 없는 mapping은 순서-일치 검증이 아예 실행되지 않아
중복 이름을 걸러내지 못했던 gap을 메움. inference가 만드는
`{class_name: probability}` dict는 이름이 중복되면 앞선 확률이 조용히
덮어써져 결과 의미가 깨지므로, artifact load 경계에서 이를 막는다)
+ `class_to_idx`가 있으면 `{name: index for index, name in
enumerate(classes)}`와 정확히 일치. `save_class_mapping()`이 실제로
저장하는 정상 artifact(`ImageFolder.classes`/`class_to_idx`에서 온
값, 항상 이 조건을 만족)는 검증을 그대로 통과하므로 **training
behavior는 전혀 바뀌지 않는다** -- `tests/training`(486개, 신규
`test_load_class_mapping_rejects_duplicate_class_names_without_class_to_idx`
포함) + `tests/scripts`가 이 변경 후에도 정상 artifact 기준 전부
무수정으로 PASS함을 확인했다.

## 6. precision execution contract

```python
with torch.inference_mode():
    if precision == "fp32":
        logits = model(inputs)
    else:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits = model(inputs)
```

- `model.eval()`/`torch.inference_mode()`는 precision과 무관하게 항상
  적용(Phase 6A §6).
- `fp32`는 autocast 없음(의도적).
- `fp16`/`bf16`은 `torch.amp.autocast(device_type="cuda", dtype=...)`
  로 forward만 감싼다 -- `loop.py`의 `train_one_epoch()`이 forward+loss
  구간을 감싸는 것과 동일한 API 호출(`loop.py:279`).
- 입력 tensor/model parameter를 `.half()`/`.bfloat16()`로 직접 캐스팅
  하지 않는다 -- state_dict는 항상 fp32로 로드된다.
- `GradScaler`는 쓰지 않는다(backward가 없으므로 scaled-backward
  개념 자체가 적용되지 않음).
- CPU + fp16/bf16은 `_validate_precision_device_compatibility()`가
  `run_single_image_inference()` 진입 초반에 거부한다(model/이미지를
  준비하기 전에 fail-fast).

**duration 측정 구간**(`InferenceResult.inference_duration_seconds`):
"model/입력 준비 완료 -> forward 시작"부터 "softmax/argmax/confidence/
probabilities dict 추출 완료"까지. CUDA device면 이 구간 끝에 명시적
`torch.cuda.synchronize(device)`를 추가했다 -- `.item()` 호출들이
이미 암묵적으로 동기화를 강제하지만(그 값이 나올 때까지 블록), 의도를
코드로 명확히 하고 향후 코드가 바뀌어도 안전하도록 명시적으로
넣었다. 즉 host-side enqueue latency가 아니라 **실제 GPU 완료 시간에
최대한 가깝게** 측정한다(Phase 6A §11 권장 방향).

## 7. device/precision validation 공용화

`_DEVICE_PATTERN`/`_validate_device`/`_is_cuda_device`/
`_validate_precision_device_compatibility`/`_CUDA_ONLY_PRECISIONS`를
`imagefolder_workflow.py`에서 신규 `training/device.py`로 **순수
이동**했다(로직/docstring/예외 메시지 전부 그대로, 이름도 그대로 --
public으로 승격하지 않았다, 같은 패키지 내부이므로 그대로
import해서 쓸 수 있다는 것이 Phase 6A의 최소주의 판단이었다).
`imagefolder_workflow.py`는 이제 이 모듈에서 import만 하고(재사용
가능하도록 그대로 이름을 노출), `inference/single_image_inference.py`
도 같은 함수를 import해서 쓴다 -- 로직 복사 없음.

**기존 `tests/training/test_imagefolder_workflow.py`가 이 private
함수들을 `imagefolder_workflow` 모듈에서 직접 import하는 기존 테스트
코드는 한 글자도 바꾸지 않았다** -- `imagefolder_workflow.py`가 그
함수들을 여전히 자기 namespace에 갖고 있으므로(재-import) 기존 import
경로가 그대로 유효하다.

**training regression 결과**: `pytest -q tests/training` (device
이동 + class_mapping 검증/uniqueness 추가 이후) **486 passed**
(신규 duplicate-class-name 테스트 1개 포함) -- device validation/
precision-device compatibility/CUDA index/CPU fp16·bf16 거부 관련
기존 테스트 전부 무수정으로 PASS.

## 8. prediction contract

```python
probabilities_tensor = torch.softmax(logits, dim=1)
predicted_index = int(probabilities_tensor.argmax(dim=1).item())
confidence = float(probabilities_tensor[0, predicted_index].item())
probabilities = {classes[i]: float(probabilities_tensor[0, i].item()) for i in range(len(classes))}
```

`raw logits`은 `InferenceResult`에 담지 않는다(Phase 6A §4). 확인한
계약: `sum(probabilities.values()) ≈ 1.0`, `confidence ==
probabilities[predicted_class]`, `predicted_class == classes[predicted_index]`
-- 전부 `tests/inference/test_single_image_inference.py`가 실제
forward pass로 검증한다.

## 9. InferenceController lifecycle

`TrainingController`보다 단순한 4-state machine(`idle/running/
finished/failed`, `stopping` 없음, `threading.Event` 없음) --
`TrainingController`와 공통 base class를 만들지 않았다(Phase 6A §8
판단 그대로). `is_running`은 `state == "running"`만 확인한다.
`begin_run()`은 이미 `running`이면 `InferenceAlreadyRunningError`,
`run()`은 `state != "running"`이면 `RuntimeError`. `finished`/`failed`
에서 곧바로 새 `begin_run()`이 가능하다(별도 reset 단계 없음,
`TrainingController`와 동일한 계약).

## 10. QtInferenceWorker

- signal: `finished = Signal(object)`, `failed = Signal(str)` --
  **progress signal 없음**(단일 이미지는 진행률 개념이 없다, Phase
  6A §8).
- `run()`: `controller.begin_run()`(이미 실행 중이면 `failed` emit 후
  즉시 반환) -> `controller.run(request)`(예외는 `failed` emit,
  traceback 포함) -> 성공하면 `finished` emit. `QtTrainingWorker.run()`
  과 정확히 같은 shape에서 `progress_callback` 인자만 뺐다.
- **worker thread 확인**: `tests/gui/test_qt_inference_worker.py::
  test_worker_runs_off_the_gui_thread_and_emits_finished`가 backend가
  실행되는 thread id가 main thread id와 다름을 직접 확인.
- **receiver thread 확인**: 실제 `QObject` subclass(`_RecordingReceiver`,
  monkeypatch 아님 -- Phase 5C §17이 발견한 "class attribute 교체가
  Qt의 connect() 시점 thread-affinity 판정을 오염시킨다"는 함정을
  다시 밟지 않기 위해 진짜 subclass만 썼다)로 `finished`/`failed`
  둘 다 main thread에서 실행됨을 확인
  (`test_finished_delivered_to_real_qobject_receiver_runs_on_main_thread`,
  `test_failed_delivered_to_real_qobject_receiver_runs_on_main_thread`).
  대조군으로 plain 함수 slot은 여전히 worker thread에서 직접 실행됨도
  재확인(`test_plain_function_slot_runs_on_emitting_worker_thread_not_gui_thread`).
- **deleteLater contract**: `worker.finished/failed -> thread.quit`,
  `worker.finished/failed -> worker.deleteLater`(worker 자신의 신호에),
  `thread.finished -> thread.deleteLater`. Phase 5C stabilization이
  발견한 잘못된 패턴(`thread.finished -> worker.deleteLater`)은
  어디에도 없다 -- `test_repeated_worker_run_and_cleanup_with_full_deletelater_contract`
  가 이 전체 wiring을 2회 연속 실행해 `QThread.isRunning() is False`
  (또는 이미 삭제되어 접근 시 `RuntimeError`)까지 확인한다. 이 확인은
  "QThread 실행이 끝났다"는 것이지, worker/thread의 `deleteLater()`로
  예약된 C++ 객체 해제가 그 시점에 이미 처리됐음을 별도로 증명하지는
  않는다(`_wait_for_thread_cleanup()` 참고, 과장 표현 정리).
- **실제 backend 통합**: `test_qt_inference_worker_integration.py`가
  fake가 아닌 실제 `run_single_image_inference()`를 진짜 `QThread`로
  CPU 1회 + CUDA 1회(skipif) 실행 -- application/GUI wiring이 실제
  production inference core와 맞물려 정상 동작함을 확인.

### 10-1. native abort 조사 및 해결(stabilization)

최초 구현 직후 GUI worker 테스트를 35회 반복 실행하는 중 드물게(2회)
프로세스 레벨 `Fatal Python error: Aborted`가 관찰됐다 -- "PySide6/
pytest-qt known risk"로 넘기지 않고 원인을 끝까지 좁혔다.

**재현 좁히기**: `test_qt_inference_worker.py`/
`test_qt_inference_worker_integration.py`를 각각 단독으로 30회씩
반복하면 각각 1회씩 재현됐지만(`Windows fatal exception: access
violation`, 크래시 발생 thread에 Python frame이 전혀 없음 -- 네이티브
코드 내부에서 발생), **7개 테스트 각각을 단독으로 20~30회 반복하면
전혀 재현되지 않았다.** 즉 어떤 개별 테스트의 로직 결함이 아니라
**같은 프로세스/`QApplication` 안에서 여러 `QThread` 생애주기가
연속으로 일어날 때만** 나타나는 문제였다.

**원인 특정**: 반복 실행 중 다음 `RuntimeWarning`을 실측으로
포착했다(16회 중 1회 재현):

```text
RuntimeWarning: libpyside: Failed to disconnect
(<bound method SignalBlocker._quit_loop_by_signal ...>)
from signal "finished(PyObject)". signal.disconnect(slot)
```

`qtbot.waitSignal(worker.finished, ...)`는 그 signal에 **임시로**
`SignalBlocker`를 connect했다가 `with` 블록이 끝날 때 다시
disconnect한다. 그런데 `worker.finished`에는 canonical wiring에 따라
`worker.deleteLater()`도 연결돼 있다 -- `finished.emit()`이 일어나면
`worker.deleteLater()`(worker 자신의 thread에서 direct 처리)가 거의
즉시 실제 C++ 객체 삭제로 이어질 수 있는데, `SignalBlocker.__exit__`
(main thread에서 실행)가 그 직후 같은 signal에서 disconnect를
시도하면서 **이미 삭제됐거나 삭제 중인 객체에 접근**하게 된다. 이
경합이 대부분은 경고로만 끝나지만, 드문 타이밍에서는 access
violation으로 이어진다는 것이 이번 조사의 결론이다 -- 관찰된 크래시의
"크래시 thread에 Python frame이 없음"(네이티브 C++ 내부에서 발생)이
이 설명과 정확히 부합한다.

**원인 분류: B(test teardown/fixture timing bug).** production
`QtInferenceWorker`/`InferenceController`(canonical deleteLater
ordering 자체는 Phase 5C 계약대로 정확했다 -- `thread.finished ->
worker.deleteLater` 같은 잘못된 패턴은 어디에도 없었다)의 문제가
아니라, **테스트 코드가 `deleteLater()`로 자기 자신을 삭제하는
객체의 signal에 `qtbot.waitSignal()`을 거는 조합** 자체가 문제였다.
Category C(PySide6/pytest-qt 자체 issue, 원인 미확정)로 넘기지
않은 이유: 위 RuntimeWarning으로 정확한 경합 지점을 특정했고, 그
경합을 원천적으로 없애는 재현 가능한 수정을 적용해 재현 횟수가
0으로 떨어짐을 확인했기 때문이다(아래).

**수정**: `tests/gui/test_qt_inference_worker.py`,
`tests/gui/test_qt_inference_worker_integration.py` 두 파일에서
`qtbot.waitSignal(worker.finished/failed, ...)` 사용을 전부 제거하고,
signal에 **영구적으로**(임시 connect/disconnect 없이) 연결하는 plain
관찰자(`events.append(...)`, worker thread에서 직접 실행되지만
CPython의 `list.append()`는 GIL 덕분에 atomic이라 안전하다) +
`qtbot.waitUntil()` polling으로 대체했다 -- signal에 임시로 connect/
disconnect하는 코드 자체가 더 이상 없으므로 이 경합이 구조적으로
발생할 수 없다. 추가로 `test_qt_inference_worker.py`의 나머지
테스트(원래 `deleteLater()`를 연결하지 않고 `thread.quit(); thread.
wait()`만 하던 것들)도 전부 `_wire_full_lifecycle()`(canonical
wiring)로 통일했다 -- `worker`(`moveToThread()`로 다른 thread에
affinity가 있는 QObject)를 명시적 `deleteLater()` 없이 Python GC에만
맡기는 것도 cross-thread QObject 삭제라는 정의되지 않은 동작이라,
production 코드가 실제로 문서화한 사용 패턴과 테스트를 일치시키는
것이 맞다고 판단했다. **production 코드는 전혀 수정하지 않았다.**

**regression 결과**: 수정 후 `test_qt_inference_worker.py` 30회,
`test_qt_inference_worker_integration.py` 30회, `tests/gui` 전체
20회 반복 -- **abort 0회, "Failed to disconnect" warning 0회**(총
80회 반복, 수정 전 관찰 빈도 대비 명확한 개선).

## 11. CPU E2E / CUDA E2E / fp16 / bf16

이 개발 머신의 실측 환경(Phase 6B stabilization에서 직접 확인):

```text
torch: 2.12.1+cu126
cuda runtime: 12.6
device: NVIDIA GeForce GTX 1080
compute capability: (6, 1)              -- Pascal 세대
torch.cuda.is_bf16_supported(): True
```

**중요한 구분**: `torch.cuda.is_bf16_supported() == True`는 "현재
PyTorch API 기준으로 bf16 연산을 예외 없이 실행할 수 있다고 판정됨"을
뜻할 뿐, **native Tensor Core bf16 하드웨어 가속을 보유한다는 뜻이
아니다.** bf16 Tensor Core는 Ampere(compute capability 8.0)부터
도입됐고, 이 GPU는 Pascal(6.1)이라 애초에 Tensor Core 자체가 없다
(fp16 Tensor Core조차 Volta/Turing부터다). 즉 이 장치의 bf16 실행은
Tensor Core 가속이 아니라 다른 경로(CUDA core 연산 또는 PyTorch
내부의 다른 처리)로 이뤄지는 것으로 보이며, 이 프로젝트는 그 내부
구현을 검증하지 않는다 -- **PyTorch API가 "실행 가능"으로 판정했고,
실제로 예외 없이 유효한 확률을 반환하는 것만 실측으로 확인했다.**
"하드웨어 가속 지원"까지는 단정하지 않는다.

이 구분에 따라 **skip 없이** 전부 실행:

- CPU: `run_single_image_inference()`(core 직접) + `QtInferenceWorker`
  (실제 QThread) 양쪽에서 완주, 확률 합/confidence/artifact 없음 확인.
- CUDA fp32: core + QThread 양쪽에서 완주, GUI thread 비블로킹,
  progress-handler-equivalent인 `finished`가 main thread에서 처리됨.
- CUDA fp16/bf16: core 레벨에서 autocast 경로가 예외 없이 동작하고
  유효한 확률(합 ≈ 1)을 반환함을 확인 -- 목적은 wiring/execution
  contract 확인이지 성능 벤치마크가 아니다(Phase 6A §21). bf16 test의
  skip 조건은 `torch.cuda.is_available() and torch.cuda.
  is_bf16_supported()` 둘 다를 확인한다(`tests/inference/
  test_single_image_inference.py`의 `_bf16_supported_on_this_cuda_device()`)
  -- "CUDA만 있으면 bf16도 있다"는 하드코딩된 가정이 아니라 PyTorch
  자신의 API로 실제 판정한다. 이 장치는 그 판정 결과가 `True`라서
  skip되지 않고 실행됐다.

## 12. test coverage 요약

```text
tests/inference/test_single_image_inference.py    21개
tests/application/test_inference_controller.py    10개
tests/gui/test_qt_inference_worker.py               7개
tests/gui/test_qt_inference_worker_integration.py    2개
합계 40개 신규(전체 764 -> 804)
```

마무리 라운드(class_mapping uniqueness)에서 `tests/training/
test_imagefolder_dataset.py`에 1개
(`test_load_class_mapping_rejects_duplicate_class_names_without_class_to_idx`)
가 추가돼 전체는 **805개**가 됐다 -- 이 1개는 inference 전용이 아니라
`load_class_mapping()`(training/inference 공유 artifact 경계) 자체의
검증이라 위 표에는 포함하지 않았다.

training core에서 이미 검증된 것(model 빌드/`build_transform()` 수식/
device-precision validator 로직 자체의 정확성)은 반복하지 않았다 --
Phase 6B의 신규 테스트는 "이 값들을 올바른 순서로 조립했는가"와
"training과 inference가 절대 갈라지지 않는가"(preprocessing parity)
에 집중한다.

## 13. production training 코드 영향

`imagefolder_workflow.py`: `_DEVICE_PATTERN`/`_validate_device`/
`_is_cuda_device`/`_validate_precision_device_compatibility`/
`_CUDA_ONLY_PRECISIONS` 정의를 제거하고 `training/device.py`에서
import(순수 이동, `import re` 제거). **동작 변경 없음.**

`torchvision_dataset.py`: `load_class_mapping()`에 구조 검증(class
name uniqueness 포함) 추가. **정상 artifact에 대한 동작 변경 없음**
-- 손상된/수동 작성된 잘못된 파일만 새로 거부됨.

두 변경 모두 `tests/training`(486개) + `tests/scripts` + 전체
`pytest`(805개)가 반복 재확인 PASS했다.

## 14. Phase 6C handoff

`InferencePage`가 그대로 쓸 수 있는 public API:

```python
from image_ai_studio.application.inference_controller import (
    InferenceController, build_inference_request, InferenceAlreadyRunningError,
)
from image_ai_studio.gui.qt_inference_worker import QtInferenceWorker

request = build_inference_request(model_json_path=..., state_dict_path=...,
                                   class_mapping_path=..., image_path=...,
                                   device=..., precision=...)
controller = InferenceController()  # 기본 backend

thread = QThread()
worker = QtInferenceWorker(controller, request)
worker.moveToThread(thread)
thread.started.connect(worker.run)
worker.finished.connect(page.on_finished)   # 실제 QObject bound method에!
worker.failed.connect(page.on_failed)        # 〃
worker.finished.connect(thread.quit)
worker.failed.connect(thread.quit)
worker.finished.connect(worker.deleteLater)  # worker 자신의 신호에!
worker.failed.connect(worker.deleteLater)
thread.finished.connect(thread.deleteLater)
thread.start()
```

`controller.state`로 현재 상태 조회 가능. `is_running`이 True인 동안
Start 버튼 등을 비활성화하면 된다. 새 run은 finished/failed에서 바로
새 `QtInferenceWorker`+`QThread`를 만들어 시작하면 된다(같은
`controller` 재사용, `TrainingPage`가 이미 증명한 패턴).

미해결 blocker 없음. Phase 6A §11에서 결정한 대로, Phase 6C의
`InferencePage`는 output directory 자동 탐색(`best_model_state_dict.pt`/
`class_mapping.json`) + model JSON 별도 picker(2 picker) 조합을
구현하면 된다 -- 이 조합에 필요한 파일 이름(`best_model_state_dict.pt`,
`class_mapping.json`)은 Phase 4/5부터 고정돼 있고 이번 Phase에서
바뀌지 않았다.
