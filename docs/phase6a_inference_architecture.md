# Phase 6A: Inference Architecture — 설계안

**이 문서는 design-only다.** Phase 6A는 production/test 코드를 전혀
수정하지 않았다 -- 아래는 전부 실제 코드(Phase 4~5D)를 직접 읽고
확인한 조사 결과와, Phase 6B~6D가 그 위에서 구현할 architecture
설계다.

## 0. 원칙

**Phase 5에서 안정화된 training 코드를 건드려 inference까지 generic하게
만들지 않는다.** 기존 API(`build_model`/`load_model_spec`/
`build_transform`/`load_state_dict`/...)를 최대한 재사용하면서,
inference를 training과 나란히 놓이는 별도의 얇은 vertical slice로
추가한다. `TrainingController`/`QtTrainingWorker`는 이번 Phase에서
**구조를 복제하지 않는다** -- 검증된 *패턴*(state machine, QThread
lifecycle, signal thread-affinity, deleteLater ordering)만 재사용하고,
inference 고유의 훨씬 단순한 lifecycle(아래 §8)에 맞게 독립적으로
작게 만든다.

---

## 1. 현재 artifact 구조 조사

`run_imagefolder_training_workflow()`(`src/image_ai_studio/training/
imagefolder_workflow.py`)가 만드는 것을 실제 코드 기준으로 정리한다.

| artifact | 생성 위치 | 항상 생성? | inference에 필요? |
|---|---|---|---|
| `best_model_state_dict.pt` | `output_dir/best_model_state_dict.pt` (고정 파일명) | 항상 | **필수** |
| model definition JSON | 사용자가 학습 시 지정한 `model_json_path` -- **output_dir에 복사되지 않음** | 학습 입력(출력 아님) | **필수** |
| `class_mapping.json` | `output_dir/class_mapping.json` (고정 파일명) | 항상 | **필수** |
| `training_history.json` | `output_dir/training_history.json` | 항상 | 불필요 |
| `test_result.json` | `output_dir/test_result.json` | 항상 | 불필요 |
| checkpoint(`*.pt`) + metadata | 사용자가 지정한 `checkpoint_out`(output_dir 밖일 수 있음), optional | `checkpoint_out` 지정 시만 | **불필요**(training resume 전용, 아래 §1-B) |
| `model.ts`(TorchScript) | `output_dir/model.ts` | `export_torchscript=True`(GUI 기본값)일 때만 | 선택(canonical 아님, 아래 §1-C) |
| `model_metadata.json` | `output_dir/model_metadata.json` | TorchScript export 시만 | 선택 |

**중요한 발견**: `output_dir`에는 model definition JSON이 **저장되지
않는다.** 학습 시 사용자가 고른 `model_json_path`는 입력일 뿐, 워크플로우
어디에서도 `output_dir`로 복사/보존하지 않는다(`imagefolder_workflow.py`
전체를 확인했다 -- `model_json_path`를 쓰는 곳은 `load_model_spec()`
호출 한 곳뿐). 이는 §11(artifact bundle)의 UX 설계에 직접 영향을 준다.

### A. `best_model_state_dict.pt`만으로 모델을 복원할 수 있는가?

**아니다.** `save_state_dict()`(`training/checkpoint.py`)는
`model.state_dict()`만 저장한다 -- 아키텍처 정보가 전혀 없다. 복원하려면
최소한:

```text
1. model definition JSON  (ModelSpec -> nn.Module 구조, input_shape 포함)
2. best_model_state_dict.pt (가중치)
3. class_mapping.json      (output index -> class name, 사람이 읽는 결과에 필수)
```

세 개가 함께 있어야 "이미지 -> 사람이 이해하는 예측"까지 완성된다.
normalization contract(mean/std)는 파일로 저장되지 않고 **코드
상수**(`torchvision_dataset.NORMALIZE_MEAN`/`NORMALIZE_STD`, 둘 다
`(0.5, 0.5, 0.5)`)로 고정돼 있다 -- `build_transform()`을 그대로
재사용하면 이 상수도 자동으로 따라온다(아래 §2).

### B. checkpoint는 inference에 필요한 구조 정보까지 담고 있는가?

**아니다 -- training resume 전용이다.** `save_training_checkpoint()`의
payload(`training/checkpoint.py`)를 직접 확인했다:

```text
format_version, model_state_dict, optimizer_state_dict,
scheduler_state_dict, history, best_state_dict,
epochs_without_improvement, training_config, loader_generator_state,
cpu_rng_state, cuda_rng_state, scaler_state_dict
```

`model_state_dict`/`best_state_dict`는 들어있지만 **ModelSpec(아키텍처)도
class_mapping도 들어있지 않다** -- `training_config`는
`TrainingConfig`(optimizer/lr/precision 등 hyperparameter)일 뿐 모델
구조 정의가 아니다. checkpoint에서 추론하려 해도 결국 `best_model_
state_dict.pt`와 똑같이 별도의 ModelSpec JSON + class_mapping.json이
필요하므로, checkpoint를 inference 입력으로 쓰는 것은 `best_model_
state_dict.pt`보다 이점이 없고(오히려 optimizer/scheduler 등 불필요한
state까지 로드해야 함) **inference 경로에서 제외한다.**

### C. canonical inference path: `state_dict + ModelSpec` vs `TorchScript`

**`state_dict + ModelSpec`을 canonical inference path로 선택한다.**
TorchScript는 향후 확장(특히 C++ inference, 이 프로젝트의 장기 목표인
Python/C++ 비교)으로 남겨두고 Phase 6에서는 만들지 않는다.

이유:

1. **TorchScript export는 선택 사항이다**(`export_torchscript`
   checkbox, GUI 기본값은 체크돼 있지만 사용자가 끌 수 있다). canonical
   path가 TorchScript면 이 체크박스를 끈 모든 학습 결과에 대해
   inference가 불가능해진다 -- `best_model_state_dict.pt`는
   **항상** 생성되므로 이쪽이 훨씬 견고하다.
2. state_dict 경로는 `build_model()`/`load_model_spec()`/
   `load_state_dict()`를 그대로 재사용한다 -- "inference 전용 model
   builder를 새로 만들지 마라"는 요구사항을 code 없이 만족시킨다.
3. `TorchScriptExporter`(`export/torchscript_exporter.py`)의
   docstring 자체가 "TorchScript is deprecated upstream but kept as
   the compatibility backend for this spike"라고 명시한다 -- 이
   프로젝트에서 TorchScript의 역할은 애초에 "C++ parity 검증용
   export backend"이지 "Python 쪽 canonical inference 수단"이 아니다.
4. TorchScript metadata(`model_metadata.json`, `export/base.py`의
   `build_metadata()`)에도 class_mapping이 없다 -- TorchScript를
   선택해도 class_mapping.json은 어차피 별도로 필요하므로, 아키텍처
   정보를 얻는 이점이 GUI 입력 개수를 줄여주지 못한다.
5. state_dict 경로가 하나 더 필요로 하는 것(model definition JSON)은
   `build_transform()`이 필요로 하는 `input_shape`와도 이미 같은
   출처(ModelSpec)라 어차피 로드해야 한다 -- 별도 비용이 없다.

**성급하게 둘 다 지원하지 않는다** -- TorchScript inference(`torch.
jit.load()` + forward)는 Phase 6 범위 밖의 명시적 non-goal이다(§16).

---

## 2. preprocessing contract 조사

`training/torchvision_dataset.py`의 `build_transform(input_shape)`가
학습/검증/평가 전체에서 쓰는 **유일한** 전처리 경로다(CIFAR-10과
ImageFolder 둘 다 이 함수 하나를 공유한다). 실제 코드로 확인한 계약:

```python
def build_transform(input_shape: tuple[int, int, int]) -> transforms.Compose:
    _, height, width = input_shape
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
```

| 항목 | 값 |
|---|---|
| 이미지 라이브러리 | PIL(torchvision `ImageFolder`/`transforms`가 내부적으로 PIL 사용) |
| RGB 변환 | `ImageFolder`가 내부적으로 `Image.open(...).convert("RGB")` 수행(3채널 고정, `_require_rgb_input_shape()`가 `input_shape[0]==3`을 미리 강제) |
| Resize | `transforms.Resize((height, width))` -- `input_shape`에서 유도, augmentation 없음(train/val/test 완전히 동일) |
| ToTensor | `[0,255]` PIL -> `[0,1]` float32 tensor, `(H,W,C)` -> `(C,H,W)` |
| Normalize | `mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)` 고정 상수(dataset별 튜닝 아님) |
| channel order | RGB, `(C,H,W)` |
| dtype | `float32` |
| batch dimension | `build_transform()`은 만들지 않는다 -- `DataLoader`의 collation이 배치 축을 붙인다. 단일 이미지 inference는 `tensor.unsqueeze(0)`으로 직접 만들어야 한다(새 코드지만 1줄, PyTorch 표준 관용구) |
| device 이동 | `build_transform()`은 device를 모른다 -- 항상 CPU tensor를 반환하고, `.to(device)`는 호출자(현재는 `run_imagefolder_training_workflow()`가 DataLoader 배치 단위로) 책임 |
| precision 처리 | `build_transform()`은 항상 `float32`를 반환한다 -- fp16/bf16은 model/autocast 쪽 관심사이지 전처리 단계의 관심사가 아니다(training도 동일: DataLoader는 항상 fp32를 내보내고, autocast가 forward 내부에서만 다운캐스트) |

**결론: 이 함수는 이미 public(`_` 없음)이고, input_shape 하나만
받는 순수 함수라 inference가 그대로 import해서 쓸 수 있다. 추출/리팩터
불필요.** 동일 transform 로직을 inference에서 복사해 새로 구현하지
않는다 -- `from image_ai_studio.training.torchvision_dataset import
build_transform`을 inference 코드가 직접 import한다. training
behavior는 전혀 바뀌지 않는다(이 함수를 read-only로 재사용할 뿐 수정
없음).

단일 이미지 inference가 추가로 하는 일(새 코드, 작음):

```python
image = PIL.Image.open(image_path).convert("RGB")   # RGB 강제는 여기서 직접
tensor = build_transform(model_spec.input_shape)(image).unsqueeze(0)  # (1,C,H,W)
tensor = tensor.to(device)
```

---

## 3. model reconstruction 조사

실제 코드로 다음 경로가 이미 전부 존재/재사용 가능함을 확인했다:

```text
ModelSpec JSON
    -> load_model_spec(path)                          [model_definition/serialization.py, public]
    -> validate_model_spec(model_spec)                 [model_definition/validation.py, public -- shape 검증]
    -> build_model(model_spec) -> nn.Sequential         [model_definition/builder.py, public]
    -> load_state_dict(model, path, map_location=...)   [training/checkpoint.py, public]
```

| 함수 | 위치 | 역할 |
|---|---|---|
| `load_model_spec(path) -> ModelSpec` | `model_definition/serialization.py` | JSON -> ModelSpec |
| `validate_model_spec(model_spec) -> shape_trace` | `model_definition/validation.py` | shape 검증, 최종 output shape(`shape_trace[-1].output_shape`) 제공 -- class 개수 검증에 필요 |
| `build_model(model_spec) -> nn.Sequential` | `model_definition/builder.py` | ModelSpec -> 실제 `nn.Module` |
| `load_state_dict(model, path, map_location="cpu") -> nn.Module` | `training/checkpoint.py` | 저장된 가중치를 in-place 로드(bare state_dict 포맷 검증 포함) |
| `require_matching_num_classes(num_classes, final_shape)` | `training/torchvision_dataset.py` | class 개수 vs model 출력 shape 일치 검증(public, 이미 재사용 가능) |
| `load_class_mapping(path) -> dict` | `training/torchvision_dataset.py` | class_mapping.json 로드(public) |

**inference 전용 model builder를 새로 만들 필요가 없다** -- 위 다섯
함수를 순서대로 호출하는 것이 Phase 6B의 "model reconstruction"
전체다. `imagefolder_workflow.py`의 `_prepare_resume()`이 fresh-model
경로에서 하는 일(`build_model(model_spec)` 후 `.load_state_dict(...)`)
과 본질적으로 동일한 패턴이다 -- 다만 inference는 `.to(request.device)`
뒤에 `.eval()`을 추가로 호출한다(training은 `model.train()`을 학습
루프 안에서 별도로 호출, inference는 항상 eval 고정).

---

## 4. class mapping / prediction 의미

`class_mapping.json`(`save_class_mapping()`/`load_class_mapping()`,
`torchvision_dataset.py`) 구조를 실제 코드로 확인:

```json
{"classes": ["cat", "dog"], "class_to_idx": {"cat": 0, "dog": 1}}
```

`classes[i]`가 모델 출력 logits의 index `i`와 정확히 대응한다 --
`ImageFolder`가 만드는 `class_to_idx`를 train/val/test 세 split이
`_require_matching_classes()`로 동일함을 강제하고, `imagefolder_
workflow.py`가 이 순서 그대로(`len(splits.classes)`) 모델 출력 차원과
`require_matching_num_classes()`로 일치시키므로, `classes` 리스트
순서 = 출력 차원 순서가 항상 보장된다.

**loss/output contract**: 모델은 항상 raw logits을 출력한다(어떤
`LayerSpec`에도 Softmax 레이어가 없다 -- `model_definition/
serialization.py`의 `_LAYER_REGISTRY`를 직접 확인: conv2d/
batch_norm2d/relu/max_pool2d/adaptive_avg_pool2d/flatten/linear/
dropout/residual_block/branch/identity뿐). loss는 항상
`nn.CrossEntropyLoss`(`loop.py`)이고, 이 loss는 내부적으로 log_softmax
+ NLLLoss를 결합한다 -- 즉 **모델 출력은 unnormalized logits이고,
confidence를 얻으려면 inference 쪽에서 명시적으로 `softmax(logits)`를
적용해야 한다**(training/`evaluate()` 계열이 `outputs.argmax(dim=1)`로
정확도만 계산하고 확률을 만들지 않는 것과 대칭적 -- softmax는
inference에서 새로 필요한 유일한 수식이며, training 어디에도 이미
구현돼 있지 않다).

### `InferenceResult` 설계(구현하지 않음, 데이터 모델만)

```python
@dataclass
class InferenceResult:
    predicted_index: int
    predicted_class: str
    confidence: float                    # probabilities[predicted_class]
    probabilities: dict[str, float]       # class name -> softmax 확률, 합 ~= 1.0
    inference_duration_seconds: float     # TrainingProgress.epoch_duration_seconds와 동일한 선례
```

`probabilities`를 `list[float]`(classes 순서 aligned)가 아니라
`dict[str, float]`로 설계한다 -- GUI가 class 이름으로 바로 표시할 수
있어 `classes` 리스트와 다시 zip할 필요가 없다. raw logits/tensor는
이 dataclass에 담지 않는다(GUI가 요구하지 않음, Phase 5C의 "test_metrics
상세 테이블 불필요" 결정과 동일한 최소주의).

---

## 5. device / precision 재사용 조사

Training이 쓰는 검증 로직을 실제 코드로 확인:

| 검증 | 위치 | 공개 여부 |
|---|---|---|
| `_DEVICE_PATTERN`(`^(cpu\|cuda\|cuda:(0\|[1-9][0-9]*))$`) | `imagefolder_workflow.py` | **private**(`_` 접두) |
| `_validate_device(value)` -- 정규식 + `torch.cuda.is_available()`/`device_count()` | `imagefolder_workflow.py` | **private** |
| `_is_cuda_device(device)` | `imagefolder_workflow.py` | **private** |
| `_validate_precision_device_compatibility(precision, device)` -- fp16/bf16은 CUDA 전용 | `imagefolder_workflow.py` | **private** |
| `PRECISION_CHOICES = ("fp32","fp16","bf16")` | `training/config.py` | public(값 자체는 `TrainingConfig`가 `_require_one_of()`로 검증 -- device와 무관한 단독 검증) |

**발견**: preprocessing(`build_transform`)과 달리, device/precision
검증 로직은 **private 함수로 `imagefolder_workflow.py` 안에 갇혀
있다.** `TrainingConfig`(config.py) 자신은 precision 값 자체만 알고
device는 전혀 모른다 -- cross-field 검증(`fp16`/`bf16`은 CUDA 전용)은
의도적으로 workflow 레벨에서만 한다(defense-in-depth, `loop.py`의
`_build_precision_execution()`과는 다른 경계 보호, 코드 주석에 명시).

CPU + fp16/bf16 허용 여부(실제 코드로 확인): **거부한다.** CPU AMP는
"이번 Phase의 범위 밖"이라고 주석에 명시돼 있고, `device=="cpu"`이면
`fp16`/`bf16` 둘 다 `ValueError`. CUDA availability/index 검증은
`_validate_device()`가 `torch.cuda.is_available()`/`device_count()`로
조기 검증(늦게 `.to(device)`에서 저수준 에러가 나는 것을 막기 위해).
autocast는 **training(정확히는 `run_training()`의 forward+loss
구간)에서만** 쓰이고, `evaluate()`/`evaluate_classification_metrics()`
(항상 CPU 고정 평가)와 `TorchScriptExporter.export()`는 autocast를
전혀 쓰지 않는다 -- 이 프로젝트에서 "평가/추론 유사 경로"가 이미
autocast 없이 fp32로만 동작해 온 선례가 있다.

**Phase 6B 권장 조치(설계만, 이번에 구현하지 않음)**: 위 4개 private
함수(`_DEVICE_PATTERN`/`_validate_device`/`_is_cuda_device`/
`_validate_precision_device_compatibility`)를 `imagefolder_workflow.py`
밖의 작은 공용 모듈(예: `training/device.py`)로 **순수 이동**(이름
변경 없이 옮기고 `imagefolder_workflow.py`는 그 모듈에서 import하도록
수정)하는 것을 권장한다 -- 동작 변경이 전혀 없는 리팩터이므로 기존
training 회귀 테스트(`tests/training/test_imagefolder_workflow.py`
등)가 그대로 안전망이 된다. 이 이동 후 inference도 같은 4개 함수를
import해서 쓴다(중복 구현 없음). 이 리팩터가 부담스럽다면 대안으로
inference 코드가 이 private 함수들을 `imagefolder_workflow`에서 직접
import하는 것도 기술적으로는 가능하지만(같은 패키지 내부이므로 금지된
것은 아님), 관례상 지저분하므로 **작은 이동 리팩터 쪽을 권장**한다.
어느 쪽이든 **로직을 복사해서 새로 구현하지 않는다.**

`precision` inference 시 선택 가능 여부: state_dict는 항상 fp32로
저장된다(AMP는 순수 실행 시 dtype 힌트일 뿐 파라미터 저장 dtype을
바꾸지 않는다 -- PyTorch AMP의 표준 동작) -- 즉 학습 시 precision과
무관하게, inference에서 fp16/bf16을 CUDA에서 선택하는 것은 training과
동일한 계약(CUDA 전용)으로 독립적으로 지원 가능하다.

---

## 6. inference mode의 PyTorch semantics

프로젝트 전체에서 이미 3곳(`loop.py`의 `evaluate()`/
`evaluate_classification_metrics()`, `export/torchscript_exporter.py`)
이 정확히 같은 관용구를 쓰고 있음을 확인했다:

```python
model.eval()
with torch.inference_mode():
    outputs = model(inputs)
```

**Phase 6의 canonical 관용구도 동일하게 `model.eval()` +
`torch.inference_mode()`로 고정한다**(`torch.no_grad()`는 쓰지 않는다
-- `inference_mode()`가 `no_grad()`보다 더 강한 최적화를 제공하는
상위 호환이고, 이 프로젝트가 이미 세 곳에서 일관되게 `inference_mode()`
를 선택해 왔으므로 새로운 결정을 내릴 필요가 없다).

두 API는 **역할이 다르다**(문서화 목적, Phase 6B 코드 주석에도 반영
권장):

- `model.eval()`: `nn.Module`의 **모드**를 바꾼다 -- `Dropout`을
  비활성화하고 `BatchNorm2d`가 배치 통계 대신 저장된 running
  mean/var를 쓰게 한다. autograd와는 무관하다.
- `torch.inference_mode()`: **autograd graph 생성 자체를 막는다** --
  `.backward()`가 불가능해지는 대신 메모리/속도 이득을 얻는다.
  `model.eval()`과 무관하며, 둘 다 없으면(예: `model.eval()`만 하고
  `inference_mode()`를 빠뜨리면) 불필요한 autograd graph가 계속
  쌓인다.

둘 다 필요하고 서로 대체할 수 없다 -- inference 코드는 항상 둘 다
쓴다.

### precision execution contract(fp32/fp16/bf16)

§5가 확인한 device/precision **검증** 계약(`fp16`/`bf16`은 CUDA
전용, CPU면 거부)과는 별개로, Phase 6A 초안에는 검증을 통과한 뒤
**forward 실행 자체**를 precision별로 어떻게 수행할지가 빠져 있었다
-- 이 문서에서 그 계약을 명확히 한다. `loop.py`의
`_build_precision_execution()`/`train_one_epoch()`이 이미 쓰는
실행 방식을 그대로 재사용하고, inference 전용 precision 코드를 새로
설계하지 않는다:

```python
model.eval()
with torch.inference_mode():
    if precision == "fp32":
        outputs = model(inputs)
    else:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            outputs = model(inputs)
```

| precision | 실행 방식 |
|---|---|
| `fp32` | 일반 forward, autocast 비활성(`training`이 `autocast_dtype=None`일 때와 동일한 무분기 경로) |
| `fp16` | CUDA에서 `torch.amp.autocast(device_type="cuda", dtype=torch.float16)`로 forward를 감싼다 -- `train_one_epoch()`의 fp16 forward+loss 구간과 동일한 API 호출(`loop.py:279`) |
| `bf16` | CUDA에서 `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`로 forward를 감싼다 -- fp16과 동일한 autocast 메커니즘, dtype만 다름 |
| CPU + fp16/bf16 | §5의 기존 training 계약과 동일하게 거부(`_validate_precision_device_compatibility()`가 이미 이 조합을 `ValueError`로 막으므로, inference core까지 도달하지 않는다) |

**입력 tensor를 미리 `.half()`/`.bfloat16()`로 캐스팅하지 않는다** --
training이 `autocast_dtype`으로 forward+loss 구간만 감싸고 model
parameter/입력 tensor 자체는 그대로 fp32로 두는 것과 동일한 방식이다
(autocast가 내부적으로 필요한 연산만 선택적으로 낮은 정밀도로
실행한다). 이 대칭성 덕분에 inference도 항상 fp32 state_dict를 그대로
로드하기만 하면 되고, precision별로 다른 방식으로 가중치를 변환/저장할
필요가 없다.

**`GradScaler`는 inference에 필요 없다** -- `_build_precision_execution()`
이 반환하는 두 값(`autocast_dtype`, `scaler`) 중 `scaler`는 scaled
backward(역전파) 전용이고, inference는 backward를 전혀 수행하지
않는다(`torch.inference_mode()`가 애초에 이를 막는다). Phase 6B는
`_build_precision_execution()` 전체를 재사용하지 않고, 그 함수가
`autocast_dtype`을 계산하는 매핑 규칙(`"fp16"→torch.float16`,
`"bf16"→torch.bfloat16`, `"fp32"→None`)만 가져와 쓴다 -- 이 매핑은
한 줄짜리 규칙이라 별도 함수 추출 없이 inference core 안에서 직접
분기해도 중복 구현이라 보지 않는다(§5에서 실제로 이동을 권장한
device/precision **검증** 로직과는 성격이 다르다 -- 저건 여러 줄의
분기/예외 메시지를 가진 검증 함수라 이동 가치가 있고, 이건 3-way
literal 매핑이라 추출할 만한 로직이 없다).

세 역할은 서로 다르고 적용 범위도 다르다:

- `model.eval()` -- module 모드(Dropout/BatchNorm 등)를 바꾼다.
  **모든** inference(fp32 포함)에서 항상 필요하다.
- `torch.inference_mode()` -- autograd graph 생성을 차단한다.
  마찬가지로 **모든** inference에서 항상 필요하다.
- `autocast` -- forward 연산의 실행 dtype을 제어한다. **fp16/bf16을
  선택한 경우에만** 추가로 필요하다.
- `fp32`에서는 **의도적으로 autocast를 쓰지 않는다** -- 위 표의
  "일반 forward, autocast 비활성"이 바로 이 뜻이다.

즉 `model.eval()`/`inference_mode()`는 precision과 무관하게 항상
켜지는 고정 계약이고, autocast는 그 위에 fp16/bf16일 때만 조건부로
얹히는 별도의 축이다 -- `fp32`에서 autocast가 없는 것은 "계약이
깨진" 것이 아니라 애초에 그 계약의 일부가 아니다(예: `model.eval()`을
빠뜨리면 precision과 무관하게 BatchNorm이 배치 통계를 계속 쓰게 되어
결과가 학습 시 evaluation과 달라지지만, autocast를 fp32에서 빠뜨리는
것은 정상 동작이다).

---

## 7. single-image vs folder inference 범위 판단

**Phase 6B/6C는 single-image inference만 구현한다(Option A).** folder
batch inference는 Phase 6 안에서 만들지 않고 명시적으로 미룬다.

이유:

1. single image의 `InferenceResult` 하나만 설계/테스트하면 되므로
   표면적이 작다 -- folder batch는 "결과를 어떻게 집계/표시할
   것인가"(테이블? 요약 통계? 실패한 이미지 처리?)라는 아직 답하지
   않은 UX 질문을 새로 열어야 한다.
2. YAGNI: 이번 Phase의 최소 목표("학습된 모델을 실제 이미지에 적용해
   추론")는 single image로 이미 완전히 달성된다.
3. **구조적으로 손해가 없다** -- folder batch가 나중에 필요해지면
   `run_single_image_inference()`를 반복 호출하는 얇은 loop만
   추가하면 된다(inference core 자체를 다시 설계할 필요 없음). 지금
   미리 batch를 지원하도록 core를 일반화해도 당장 이득이 없다.
4. `multi-run`/대규모 batch job manager/비동기 큐/DB는 이미 Phase 5의
   non-goal 계승 항목이었고, 폴더 batch를 지금 넣으면 그 경계와
   맞닿는다(예: "폴더 처리 중 Stop"이 사실상 mini job manager다).

`InferenceRequest.image_path: Path`(단수)로 설계하고, 폴더 지원은
Phase 6 이후 backlog로 남긴다(§16).

---

## 8. application layer 설계

### `InferenceController`가 필요한가? → **필요하다, 단 훨씬 단순하게.**

`TrainingController`의 state machine(`idle/running/stopping/finished/
failed` + `threading.Event` cooperative stop)을 그대로 복제하지
않는다. Single-image inference는 **원자적**(한 번의 forward pass,
중간에 의미 있게 멈출 지점이 없다)이므로 `stopping` 상태와
`request_stop()`이 필요 없다.

```python
InferenceControllerState = Literal["idle", "running", "finished", "failed"]

class InferenceController:
    def __init__(self, backend: InferenceBackend = run_single_image_inference) -> None: ...
    @property
    def state(self) -> InferenceControllerState: ...
    @property
    def is_running(self) -> bool: ...   # state == "running"만
    def begin_run(self) -> None: ...    # is_running이면 InferenceAlreadyRunningError
    def run(self, request: InferenceRequest) -> InferenceResult: ...
```

`request_stop()`/`threading.Event`/`stopping` 상태 전부 없음 --
single-active-run guard(`is_running`)만 `TrainingController`와
동일한 이유(중복 실행 방지)로 유지한다.

### 별도 `QtInferenceWorker`가 필요한가? → **필요하다.**

`QtTrainingWorker`와 같은 이유(§9 CUDA 참고)로, CUDA 첫 호출(context
초기화/cuDNN 알고리즘 선택)은 수 초가 걸릴 수 있어 GUI thread에서
동기 실행하면 화면이 멈춘다. CPU 단일 이미지는 보통 매우 빠르지만,
"device에 따라 GUI가 막히거나 안 막히거나"를 조건부로 만드는 것보다
**항상 QThread를 쓰는 것이 더 단순하고 예측 가능하다**(Phase 5가
이미 증명한 패턴 하나로 일관되게 추론 가능). progress signal은
필요 없다(단일 이미지는 진행률 개념이 없음) -- `finished`/`failed`
두 signal만 있으면 된다:

```python
class QtInferenceWorker(QObject):
    finished = Signal(object)  # InferenceResult
    failed = Signal(str)       # f"{ExceptionType}: {message}\n{traceback}"

    def run(self) -> None:
        # controller.begin_run() -> controller.run(request) -> emit
        # (QtTrainingWorker.run()과 정확히 같은 shape, progress 없음)
```

**worker.deleteLater() ordering은 Phase 5C stabilization이 확정한
계약을 그대로 따른다**(`finished`/`failed` → `worker.deleteLater()`,
`thread.finished` → `thread.deleteLater()`) -- 이 부분은 새로 설계할
필요가 없다, 이미 옳은 것으로 검증된 패턴을 그대로 베낀다.

### `TrainingController`와 추상 base class가 필요한가? → **아니다.**

두 controller의 state machine 모양이 다르다(`stopping`의 유무 자체가
다르다) -- 억지로 공통 base를 뽑으면 "언젠가 쓸 수도 있는" 추상화이지
지금 당장 실제로 공유되는 로직이 거의 없다(`is_running`
property/`begin_run()` 이름 정도). 두 개의 독립적인 작은 클래스가
하나의 억지 추상화보다 읽기 쉽고 안전하다(YAGNI, "세 번째 사용처가
생기기 전엔 추상화하지 않는다"라는 이 프로젝트의 기존 원칙과 일치).

### 기존 worker/controller를 generic하게 refactor할 가치가 있는가? → **아니다.**

Phase 5의 `TrainingController`/`QtTrainingWorker`는 이번 Phase에서
**한 줄도 수정하지 않는다.** Phase 6는 그 옆에 새 파일
(`application/inference_controller.py`, `gui/qt_inference_worker.py`)
을 추가할 뿐이다 -- "Phase 6 때문에 안정화된 training architecture를
generic framework로 대규모 refactor하지 않는다"는 이 조사의 핵심
원칙을 그대로 지킨다.

---

## 9. GUI architecture 설계

### `InferencePage` 초안(Phase 6C가 구현할 화면, 지금은 스케치만)

```text
Inference

Model
  Training output directory: [...]        Browse...
  (자동 탐색: best_model_state_dict.pt, class_mapping.json)
  Model JSON:                [...]        Browse...
  (output_dir에 저장되지 않으므로 별도 선택 필요, §11)

Input
  Image: [...]                             Browse...

Runtime
  Device:    [cpu/cuda/cuda:0/...]         (Training과 동일한 combo 재사용)
  Precision: [fp32/fp16/bf16]              (Training과 동일한 combo 재사용)

[Run Inference]

Result
  Status: Idle / Running... / Completed / Failed
  Predicted class:      ...
  Confidence:           ...
  Class probabilities:  class A: 0.83, class B: 0.17, ...
  Inference time:       ...ms
  Error(있으면): 요약 + Details(Phase 5C의 Failed UX와 동일 패턴)
```

이미지 preview(선택한 이미지를 QLabel/QPixmap으로 미리보기)는 **Phase
6의 필수 기능이 아니다** -- "사용자가 올바른 이미지를 선택했는지
확인"이라는 실질적 가치는 있지만, Phase 5C가 지킨 "화려하게 만들지
않는다" 원칙과 그래프/차트 non-goal 정신에 맞춰 **backlog로 미룬다**
(작은 `QLabel` + `QPixmap.scaled()` 정도라 구현 자체는 어렵지 않지만,
"필수 기능인지"를 판단하라는 §12 지시에 따라 미필수로 분류한다).

### Training/Inference navigation: `QTabWidget`

`MainWindow`는 현재 `TrainingPage` 하나만 `setCentralWidget()`으로
담는다(`gui/main_window.py`, Phase 5C). Phase 6C는 이를 `QTabWidget`
(central widget) + 두 tab("Training", "Inference")으로 바꾼다 --
**sidebar/stacked widget/router/plugin architecture는 만들지
않는다**(Phase 5C가 의도적으로 만들지 않았던 것을 이번에도 근거 없이
들여오지 않는다, 화면이 2개뿐인 상황에서 `QTabWidget`보다 복잡한
구조를 정당화할 근거가 없다).

```python
class MainWindow(QMainWindow):
    def __init__(self, ...):
        ...
        self._tabs = QTabWidget(self)
        self._training_page = TrainingPage(self)
        self._inference_page = InferencePage(self)
        self._tabs.addTab(self._training_page, "Training")
        self._tabs.addTab(self._inference_page, "Inference")
        self.setCentralWidget(self._tabs)
```

### close-during-training vs close-during-inference lifecycle

`closeEvent()`는 두 페이지 모두의 활동 여부를 확인해야 하지만,
**training과 inference의 "종료를 기다리는 방식"은 다르다** -- §8에서
이미 확정한 대로 inference에는 `request_stop()`/cooperative
stop/`stopping` 상태가 아예 없다(원자적 단일 forward pass라 중간에
멈출 지점이 없다, §7). 따라서 training에 쓰는 "stop 요청" 문구를
inference에 그대로 적용하면 안 된다 -- **inference는 강제 중단하지
않고 자연 종료(끝날 때까지 기다림)만 한다.** `QThread.terminate()`나
inference용 새 stop API는 추가하지 않는다(Phase 6B/6C 전체에 적용되는
원칙, §8과 동일). blocking `wait()`도 쓰지 않는다(Phase 5C 원칙
그대로).

**close ownership은 `MainWindow`가 중앙에서 조정한다.** 각 page의
`close_requested`를 `MainWindow.close`에 직접 연결하면 안 된다 --
Training과 Inference가 동시에 실행 중일 때 먼저 끝난 쪽의
`close_requested`가 곧바로 `MainWindow.close()`를 호출해
`closeEvent()`가 다시 실행되고, 아직 실행 중인 다른 page 때문에 확인
다이얼로그가 또 뜨는 문제가 생긴다(둘 다 active인 상태에서
`closeEvent()`가 "활동 있음"으로 판단해 다이얼로그를 다시 띄움).
이를 막기 위해 각 page의 `close_requested`는 `MainWindow`의 coordination
handler 하나에 연결하고, 그 handler가 "정말 닫아도 되는지"(양쪽 다
비활성인지)를 매번 다시 확인한 뒤에만 실제로 `self.close()`를
호출한다.

```python
class MainWindow(QMainWindow):
    def __init__(self, ...):
        ...
        self._close_pending = False   # "사용자가 닫기를 확인했고, 진행 중인 작업이 끝나길 기다리는 중"
        self._training_page.close_requested.connect(self._try_finish_pending_close)
        self._inference_page.close_requested.connect(self._try_finish_pending_close)

    def closeEvent(self, event):
        any_active = (self._training_page.is_training_active()
                      or self._inference_page.is_inference_active())
        if not any_active:
            event.accept(); return
        if self._close_pending:
            # 이미 확인 다이얼로그를 통과해 종료를 기다리는 중이다 --
            # 다시 묻지 않는다(다이얼로그는 정확히 한 번만 표시).
            event.ignore(); return

        if not user_confirms():   # 확인 다이얼로그, 정확히 한 번만 표시
            event.ignore(); return

        self._close_pending = True
        if self._training_page.is_training_active():
            self._training_page.request_stop_and_close()   # cooperative stop 요청
        if self._inference_page.is_inference_active():
            self._inference_page.wait_for_completion_and_close()  # stop 없음, 자연 종료만 기다림
        event.ignore()  # 실제 close는 _try_finish_pending_close()가 판단해서 수행

    def _try_finish_pending_close(self) -> None:
        """두 page의 `close_requested`가 공통으로 연결하는 handler --
        먼저 끝난 page가 emit해도, 아직 다른 page가 active면 아무 일도
        하지 않는다(다이얼로그 재등장 방지의 핵심)."""
        if (self._close_pending
                and not self._training_page.is_training_active()
                and not self._inference_page.is_inference_active()):
            self._close_pending = False
            self.close()
```

**Training + Inference 둘 다 실행 중인 경우**, 어느 쪽이 먼저
끝나든 동일하게 안전하다(대칭적):

```text
확인 다이얼로그 1회만 표시 -> Yes -> MainWindow._close_pending = True
    -> training_page.request_stop_and_close()  (cooperative stop 요청)
    -> inference_page.wait_for_completion_and_close()  (자연 종료 대기)

[경우 1] Training이 먼저 끝남
    -> training_page가 close_requested emit
    -> _try_finish_pending_close(): inference_page가 아직 active
       -> 아무 일도 하지 않음(실제 close 보류)
    -> inference도 끝남 -> inference_page가 close_requested emit
    -> _try_finish_pending_close(): 이제 둘 다 inactive -> self.close()

[경우 2] Inference가 먼저 끝남 -- 순서만 반대, 결과는 동일
    -> inference_page가 close_requested emit
    -> _try_finish_pending_close(): training_page가 아직 active
       -> 아무 일도 하지 않음(실제 close 보류)
    -> training도 끝남 -> training_page가 close_requested emit
    -> _try_finish_pending_close(): 이제 둘 다 inactive -> self.close()
```

두 번째로 도착한 `close_requested`가 `_try_finish_pending_close()`를
호출했을 때 `self.close()`가 실제로 실행되면, 이 시점엔 이미
`is_training_active()`/`is_inference_active()` 둘 다 `False`이므로
`closeEvent()`가 재진입해도 즉시 `accept()`한다 -- 다이얼로그가 다시
뜨지 않는다.

**page-level `close_pending`과 MainWindow-level `_close_pending`은
서로 다른 계층의 개념이다.** `TrainingPage.request_stop_and_close()`/
`InferencePage.wait_for_completion_and_close()`는 Phase 5C의
`_close_pending`/`close_requested` 패턴을 각자 페이지 안에서 그대로
쓴다(그 page 자신이 끝나면 `close_requested`를 emit하라는 의미) --
이건 "이 page가 끝나면 스스로 알려라"라는 page-local 신호다.
`MainWindow._close_pending`은 그와 별개로 "앱 전체가 닫히기를
기다리는 중"이라는 상위 신호이고, 여러 page의 신호를 모아 언제 실제
`close()`를 부를지 판단하는 것은 전적으로 `MainWindow`의 책임이다.
`InferenceController`/`QtInferenceWorker`(그리고 기존
`TrainingController`/`QtTrainingWorker`)는 close coordination의
존재 자체를 모른다 -- 새 메서드나 signal을 추가하지 않는다. 기존
`TrainingController`/`QtTrainingWorker` architecture도 이 조정 때문에
수정하지 않는다.

---

## 10. error handling 설계 — layer별 책임

Training GUI(Phase 5C)와 동일한 원칙: **GUI는 재검증하지 않는다**,
기존 함수가 이미 명확한 예외를 던지면 그것을 그대로 GUI Failed 상태로
전달한다.

| 오류 | 검증 layer | 근거(재사용할 기존 동작) |
|---|---|---|
| artifact(state_dict/model JSON/class mapping) 파일 없음 | inference core(실제 로드 함수) | `load_model_spec`/`load_state_dict`/`load_class_mapping`이 이미 `FileNotFoundError`/`ModelValidationError` 던짐 |
| model JSON 파싱 실패 | inference core | `load_model_spec()` → `ModelValidationError` |
| class mapping 파싱/구조 문제 | inference core | `load_class_mapping()`은 현재 `json.loads`만 하고 구조 검증이 없다 -- **Phase 6B가 확인해야 할 작은 gap**(아래 참고) |
| state_dict mismatch(구조 불일치) | PyTorch runtime | `nn.Module.load_state_dict()`의 기존 `RuntimeError`(missing/unexpected keys 명시) 그대로 전파 |
| 잘못된 이미지/미지원 포맷 | PyTorch runtime(PIL) | `PIL.Image.open()`의 `UnidentifiedImageError`/`OSError` 그대로 전파. GUI는 파일 dialog에 이미지 확장자 필터를 UX sugar로 둘 수 있다(중복 검증 아님, Phase 5C의 precision/device 콤보와 동일한 성격) |
| CUDA unavailable / 잘못된 CUDA index | application/request 경계 | §5에서 이동 권장한 `_validate_device()`(CUDA `is_available()`/`device_count()` 조기 검증) 재사용 |
| precision/device incompatibility | application/request 경계 | §5의 `_validate_precision_device_compatibility()` 재사용 |
| class count mismatch(class_mapping 클래스 수 ≠ model 출력 차원) | inference core | 기존 public `require_matching_num_classes()` 재사용(새 검증 함수 불필요) |
| corrupted model file | PyTorch runtime | `torch.load()`의 기존 예외 그대로 전파 |
| GUI textual parsing(빈 문자열 → None 등) | GUI(`InferencePage`) | Phase 5C의 `_empty_to_none()`과 동일한 성격의 얇은 변환, semantic validation 아님 |

**작은 gap 하나 발견**: `load_class_mapping()`은 현재 구조 검증을
전혀 하지 않는다(`{"classes": [...], "class_to_idx": {...}}`를
가정하고 바로 반환) -- 손상된/형식이 다른 class_mapping.json이 들어오면
`KeyError`/`TypeError`가 inference core 어딘가에서 불명확하게 발생할
수 있다. Phase 6B가 이 함수에 최소한의 구조 검증을 추가할지 검토할
가치가 있다(이 파일은 `torchvision_dataset.py`에 있으므로 손대면
training 쪽에도 영향 -- 순수 방어적 검증 추가이고 정상 입력의 동작은
바뀌지 않으므로 기존 회귀 테스트로 안전하게 확인 가능하다). **이번
Phase 6A에서는 수정하지 않는다** -- Phase 6B가 실제로 필요하다고
판단하면 그때 다룬다.

---

## 11. artifact bundle 문제 검토

§1에서 확인한 대로 `output_dir`에는 `best_model_state_dict.pt`/
`class_mapping.json`은 있지만 **model JSON은 없다.**

- **Option A(디렉터리 하나만 선택 → 자동 탐색)**: `output_dir`
  선택만으로는 model JSON을 못 찾는다 -- 근본적으로 불완전하다.
- **Option B(파일 각각 선택)**: 항상 동작하지만 3개 picker(model
  JSON/state_dict/class_mapping)는 번거롭다.
- **Option C(새 manifest/bundle 포맷)**: 새 artifact format이며,
  기존 training 산출물과 무관한 새 개념을 도입한다 -- "정말 필요하지
  않다면 만들지 마라"는 지시에 정면으로 위배된다. **채택하지 않는다.**

**권장: Option A와 B의 절충** -- `output_dir`을 선택하면
`best_model_state_dict.pt`/`class_mapping.json`을 고정 파일명으로
자동 탐색하고(training 쪽이 이미 이 두 파일을 항상 이 이름으로
`output_dir`에 저장하므로 안전하게 가정 가능), **model JSON만 별도
picker로 남긴다.** 3개 picker → 2개 picker로 줄어들고, training
코드를 전혀 건드리지 않는다.

향후(Phase 6 이후) `run_imagefolder_training_workflow()`가 model
JSON을 `output_dir`에 함께 복사/저장하도록 확장하면 picker를 1개로
더 줄일 수 있지만, 이는 training-core에 작은 변경을 가하는 것이라
**Phase 6 범위에서는 하지 않는다** -- "필요하지 않다면 만들지
않는다"는 원칙과 "training 코드를 근거 없이 건드리지 않는다"는 원칙이
여기서 함께 작용한다. Phase 6D 이후 실제 UX 피드백이 쌓이면
재검토할 backlog로 남긴다.

---

## 12. 테스트 전략 설계(Phase 6B~6D가 구현, 지금은 설계만)

training core에서 이미 검증된 내용(model 빌드 자체의 정확성, transform
수식 자체의 정확성, device/precision 검증 로직 자체의 정확성 -- 이동만
하고 로직은 그대로이므로)은 inference 테스트에서 반복하지 않는다.

| 영역 | 무엇을 검증하는가 | 중복 회피 |
|---|---|---|
| `InferenceRequest` 구성/변환 | `build_inference_request()`의 타입 변환(str→Path 등), semantic validation 없음 확인 | `test_build_training_request_...` 패턴 재사용, 로직은 새로 검증 |
| model reconstruction 연결 | `load_model_spec→build_model→load_state_dict` 체인이 올바르게 이어지는지 **한 번만** | `build_model()` 자체 정확성은 `tests/model_definition/`이 이미 담당 |
| **preprocessing parity**(가장 중요한 신규 테스트) | 같은 이미지를 inference 경로로 읽었을 때, training의 `ImageFolder`가 만드는 tensor와 **동일한 tensor**가 나오는지 | 신규 -- 이게 "완전히 동일한 preprocessing contract"를 실제로 잠그는 유일한 테스트 |
| single image prediction | `run_single_image_inference()`가 알려진 결과를 반환하는지(합성 모델/이미지) | 신규, 작음 |
| class mapping 매핑 | `predicted_index → predicted_class`가 `classes` 순서와 정확히 일치 | 신규, 작음 |
| confidence/probabilities | softmax 합이 1에 근접, `confidence == probabilities[predicted_class]` | 신규, 작음 |
| CPU inference(실제 E2E) | 실제 tiny model + 실제 이미지로 완주 | `test_training_page_integration.py`와 동일한 패턴 |
| CUDA inference(skipif) | wiring만(수치 정확성 아님) -- 이미 Phase 4/5가 CUDA 수치 정확성을 졸업시킴 | `test_qt_training_worker_integration.py`의 CUDA 테스트와 동일 철학 |
| precision(fp16/bf16) validation | 이동된 `_validate_precision_device_compatibility()` 재사용 확인 | 로직 자체는 기존 테스트가 이미 커버, 이동 후 import 경로만 확인 |
| `InferenceController` lifecycle | `idle→running→finished/failed`, single-active-run guard | `test_training_controller.py`보다 훨씬 작음(`stopping` 없음) |
| `QtInferenceWorker` thread affinity | `finished`/`failed`가 GUI thread에서 실행되는지 | Phase 5C의 thread-affinity 테스트와 동일한 필요성(새 클래스이므로 독립적으로 잠가야 함) |
| `InferencePage` widget → request 매핑 | GUI 필드 → `InferenceRequest` | `test_training_page.py`의 필드 매핑 테스트와 동일 패턴 |
| GUI integration(실제 E2E) | 실제 `InferencePage`를 통한 전체 흐름 | 신규, 1개면 충분(Phase 5C 철학) |
| Training/Inference navigation | tab 전환, 두 페이지 동시 존재 확인 | `test_main_window.py` 확장 |

---

## 13. Phase 6 non-goals

```text
object detection, segmentation, video inference, camera/webcam, RTSP,
multi-model ensemble, multi-GPU inference, benchmark dashboard,
experiment DB, inference history DB, ONNX export, TensorRT, OpenVINO,
C++ inference GUI, deployment server/API, packaging/installer,
drag-and-drop, custom theme, complex image viewer/editor,
folder/batch inference(§7 -- 구조적으로 나중에 추가 가능하도록 설계는
해두되 지금 만들지 않음), TorchScript inference path(§1-C, canonical
아님), inference cooperative stop/progress(§8, 원자적 작업이라 불필요),
artifact manifest/bundle 새 포맷(§11), output_dir에 model JSON 자동
저장(§11, training-core 변경이라 미룸), 이미지 preview(§9),
TrainingController/QtTrainingWorker의 generic 추상화(§8)
```

---

## 14. Phase 6 분할 계획

### Phase 6A(완료, 이 문서) — architecture/design only

- 구현 대상: 없음(문서만)
- 수정 가능 파일: `docs/phase6a_inference_architecture.md` 신규만
- 완료 조건: 이 문서, baseline 764/764 유지, git diff는 새 문서
  파일 하나뿐
- 테스트 범위: 없음(회귀 재확인만)
- 다음 Phase handoff: 이 문서 전체(특히 §3/§5/§8/§11의 결정 사항)

### Phase 6B — inference core + application/controller/worker

- 구현 대상:
  - `src/image_ai_studio/inference/__init__.py`,
    `src/image_ai_studio/inference/single_image_inference.py`
    (`InferenceRequest`, `InferenceResult`, `run_single_image_inference()`)
  - `src/image_ai_studio/application/inference_controller.py`
    (`InferenceController`, `build_inference_request()`)
  - `src/image_ai_studio/gui/qt_inference_worker.py`
    (`QtInferenceWorker`) -- PySide6 import는 이 파일에만 허용(Phase
    5B의 `qt_training_worker.py`와 동일한 경계 원칙)
  - (권장, §5) `src/image_ai_studio/training/device.py` 신설 +
    `imagefolder_workflow.py`가 거기서 import하도록 이동 -- 순수
    리팩터, 동작 변경 없음
- 수정 가능 파일 범위: 위 신규 파일들 + (권장 리팩터 채택 시)
  `imagefolder_workflow.py`의 import 문/4개 함수 위치만. **training
  로직/behavior는 전혀 바꾸지 않는다.**
- 완료 조건: GUI 없이 inference core가 실제 tiny model+실제 이미지로
  CPU/CUDA(가능하면) 양쪽에서 동작, 전체 회귀(기존 764 + 신규) PASS,
  training 관련 기존 테스트 전부 무변경으로 PASS(리팩터 시 특히 중요)
- 테스트 범위: `tests/inference/`, `tests/application/
  test_inference_controller.py`, `tests/gui/test_qt_inference_worker.py`
  (+ integration)
- 다음 Phase handoff: Phase 5B가 5C에 넘긴 것과 동일한 형태 -- 동작하는
  `InferenceController`+`QtInferenceWorker`+`run_single_image_inference()`
  스택, GUI는 아직 없음

### Phase 6C — inference GUI + MainWindow integration

- 구현 대상: `src/image_ai_studio/gui/inference_page.py`
  (`InferencePage`)
- 수정 대상: `src/image_ai_studio/gui/main_window.py`(`QTabWidget`
  전환, `closeEvent()` 확장)
- 수정 가능 파일 범위: 위 2개 + 관련 테스트만. inference core/
  controller/worker(6B 산출물)는 **API를 바꾸지 않고 그대로 사용**
- 완료 조건: Phase 5C의 완료 체크리스트와 동일한 형태(GUI 필드
  매핑, Run 버튼 lifecycle, controls enable/disable, 실제 CPU E2E,
  CUDA wiring smoke, close-during-inference(§9의 자연 종료 대기
  계약 -- stop 요청 없음), `MainWindow`의 중앙 close coordination
  (§9 `_close_pending`/`_try_finish_pending_close()` -- 다이얼로그
  1회만 표시, 먼저 끝난 page 때문에 재등장하지 않음), tab navigation,
  전체 회귀 PASS)
- 테스트 범위: `tests/gui/test_inference_page.py`,
  `tests/gui/test_main_window.py` 확장
- 다음 Phase handoff: 완성된 Training+Inference GUI 전체

### Phase 6D — final integration validation / graduation

- 구현 대상: 없음(Phase 5D와 동일하게 verification-first)
- 수정 가능 파일: 문제가 실제로 발견될 때만, 최소로
- 완료 조건: 실제 GUI launcher smoke(Inference tab 포함), 실제
  CPU/CUDA inference E2E(GUI 경로), **Training→Inference 연속
  시나리오**(같은 세션에서 방금 학습한 모델로 바로 추론, Phase 6D
  고유의 새로운 검증 가치), close-during-inference(실제 backend,
  §9 계약대로 stop 요청 없이 자연 종료를 기다린 뒤 close),
  **training+inference 동시 실행 중 close를 양쪽 종료 순서 모두**
  (training 먼저/inference 먼저) **실제 backend로 재확인**(§9 --
  확인 다이얼로그가 정확히 1회만 뜨고, 먼저 끝난 page 때문에
  재등장하지 않는지까지 확인), 전체 회귀 반복 PASS, README/전체
  design doc 정합성,
  `docs/phase6_final_integration.md`, "PHASE 6 COMPLETE" 판정
- 테스트 범위: 기존 전체 재확인, 신규 permanent test는 실제 gap
  발견 시에만
- 다음 Phase handoff: Phase 6 종료, 후속 Phase(있다면)는 별도 계획

세분화는 이 4단계로 충분하다고 판단한다 -- 6B를 더 쪼갤 만큼 각
조각이 독립적으로 크지 않다(inference core 자체가 training core보다
훨씬 작다: cooperative stop/checkpoint/resume/epoch loop가 전부
없다).

---

## 15. 이번 Phase 6A 변경 사항

**production/test 코드 수정 없음.** 신규 파일은
`docs/phase6a_inference_architecture.md` 하나뿐이다. README는 이번
라운드에서 수정하지 않는다(설계 문서만으로는 사용자에게 보여줄 새
기능이 없다는 지시에 따름).
