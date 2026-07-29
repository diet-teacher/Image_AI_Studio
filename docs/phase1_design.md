# Phase 1 설계: Model Definition Layer

## 1. Phase 1 목표

Phase 0에서 TorchScript와 AOTInductor 두 배포 경로를 직접 검증한 결과
(`docs/phase0_results.md` 참고), Windows(1차 타겟)에서 AOTInductor는
CPU 런타임 종료 시 크래시, CUDA는 테스트 GPU의 Compute Capability
제약으로 export 자체가 불가능함이 확인되었다. 따라서 Phase 1부터는
**C++ 배포/추론 경로를 TorchScript 하나로 한정**한다. 기존
AOTInductor 코드(`export/aoti_exporter.py`, `cpp/aoti_*`)는 기록/검증
목적으로 유지하지만, Phase 1의 신규 코드는 이를 전혀 참조하지 않는다.

Phase 1의 목표는 PySide6 UI를 만들기 전에, 앞으로 Image AI Studio의
중심이 될 **Model Definition Layer**를 구현하는 것이다. 이 레이어는
다음 흐름을 코드로 성립시킨다:

```text
Model Definition
    ↓
Shape Inference / Validation
    ↓
PyTorch Model Builder
    ↓
torch.nn.Module
    ↓
TorchScript Export
    ↓
C++ Inference (Phase 0 인프라 재사용)
```

이번 범위에는 PySide6 UI, 학습 루프, IPC, Detection/Segmentation이
포함되지 않는다.

## 1-1. 완료 범위 / 미지원 범위

완료 (아래 섹션에서 각각 상세 설명):

* Sequential 기반 `ModelSpec` (`specs.py`)
* `LayerSpec` 9종: `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`,
  `AdaptiveAvgPool2d`, `Flatten`, `Linear`, `Dropout`,
  `ResidualBlock`(Phase 2에서 추가, `docs/phase2_residual_block_design.md`
  참고 -- 고정된 skip connection 하나뿐이며 일반 Branch/Merge는 아님)
* JSON serialization/deserialization (`serialization.py`)
* parameter validation (각 `LayerSpec`/`ModelSpec`의 `__post_init__`)
* shape inference / layer connection validation (`shape_inference.py`,
  `validation.py`)
* PyTorch `nn.Sequential` build (`builder.py`)
* TorchScript export (Phase 0 `TorchScriptExporter` 재사용)
* 기존 C++ TorchScript runner(`run_torchscript.exe`) 연동
  (`scripts/run_phase1_e2e.py`)
* Python/C++ parity E2E 검증 (`run_phase1_e2e.py`, Phase 0
  `parity.compare_outputs` 재사용)

현재 미지원:

* 일반 Branch / Merge (임의 분기/합류)
* DAG (`GraphSpec`/`NodeSpec`/`EdgeSpec` 등 임의의 그래프 구조)
* Detection / Segmentation
* Training (학습 루프)
* UI (PySide6)

## 2. 디렉터리 구조

```text
src/image_ai_studio/model_definition/
    __init__.py
    errors.py            ModelValidationError (공용 예외 타입)
    specs.py              LayerSpec / ModelSpec dataclass 정의 + 파라미터 검증
    shape_inference.py     layer별 output shape 계산 + 레이어 연결 검증
    validation.py           ModelSpec 전체를 검증하는 단일 진입점
    builder.py               ModelSpec -> torch.nn.Sequential
    serialization.py          ModelSpec <-> JSON

tests/model_definition/
    test_specs_validation.py
    test_shape_inference.py
    test_serialization.py
    test_builder.py
    test_torchscript_integration.py
```

모듈 간 의존 관계는 순환이 없는 단방향 DAG다:

```text
errors <- specs <- shape_inference <- validation <- builder
       <- specs <- serialization (errors, specs만 사용)
```

`specs.py`/`shape_inference.py`/`validation.py`/`serialization.py`는
**torch를 import하지 않는다**. 실제 `torch.nn.Module`을 만드는
`builder.py`만 torch에 의존한다. 따라서 모델 정의 자체(JSON 로드,
shape 계산, validation)는 PyTorch가 설치되지 않은 환경(예: 나중에
UI 프로세스가 별도 프로세스로 분리되는 경우)에서도 재사용할 수 있다.

## 3. ModelSpec 구조

```python
@dataclass
class ModelSpec:
    name: str
    input_shape: tuple[int, int, int]   # (channels, height, width), batch 제외
    layers: list[LayerSpec]
```

`input_shape`은 Phase 1에서는 이미지 classification용 3차원
`(C, H, W)`만 지원한다. `__post_init__`에서 `name`이 비어있지 않은
문자열인지, `input_shape`이 `list`/`tuple`이고 3개의 양의 정수인지
검증한다. `input_shape`/`layers`가 애초에 `list`/`tuple`이 아니면
(예: 정수, 문자열, `None`) `tuple()`/`list()` 변환을 시도하기 전에
타입을 먼저 검사해 `ModelValidationError`를 낸다 -- 그렇지 않으면
`tuple(123)` 같은 호출이 raw `TypeError`를 먼저 던져 버린다.

`layers`는 **평평한(Sequential) 리스트**만 지원한다. 임의의 DAG나
분기 구조는 Phase 1 범위 밖이다 (10번 섹션 참고). `layers`는 문자열
같은 임의의 iterable을 받아들이지 않으며 (그렇지 않으면 `"conv"`가
`['c','o','n','v']`로 조용히 쪼개질 수 있다), 각 원소가 실제
`LayerSpec` 인스턴스인지도 검사한다 -- 그래야 `["conv"]`처럼 잘못된
원소가 `shape_inference` 단계까지 흘러가지 않고 `ModelSpec` 생성
시점에 바로 걸린다. `layers`가 빈 리스트/튜플이면
`ModelValidationError`를 던진다 -- 레이어가 하나도 없는 모델은
`build_model()`/`shape_inference` 입장에서 "정의된 모델"이 아니라
빈 자리표시자에 가깝고, 이를 허용하면 `shape_trace[-1]` 같은 최종
shape 조회(예: `run_phase1_e2e.py`의 요약 출력)가 소비자 쪽마다
빈 리스트를 방어하는 코드를 반복해야 한다. `ModelSpec`이 생성되는
순간부터 항상 유효한 모델이라는 불변조건을 보장하는 쪽을 택했다.

## 4. 지원 Layer

| JSON `type` | dataclass | 비고 |
|---|---|---|
| `conv2d` | `Conv2dSpec` | `in_channels`은 이전 레이어에서 자동 추론 |
| `batch_norm2d` | `BatchNorm2dSpec` | `num_features`는 이전 레이어에서 자동 추론 |
| `relu` | `ReLUSpec` | |
| `max_pool2d` | `MaxPool2dSpec` | `stride` 생략 시 `kernel_size`와 동일 (torch.nn.MaxPool2d와 동일한 기본값). `padding`은 `kernel_size // 2` 이하만 허용 (아래 참고) |
| `adaptive_avg_pool2d` | `AdaptiveAvgPool2dSpec` | 정사각형 `output_size x output_size`만 지원 |
| `flatten` | `FlattenSpec` | |
| `linear` | `LinearSpec` | `in_features`는 이전 레이어에서 자동 추론 |
| `dropout` | `DropoutSpec` | |
| `residual_block` | `ResidualBlockSpec` | Phase 2에서 추가. `in_channels`는 이전 레이어에서 자동 추론. 기존 `ResidualBlock`을 사용하는 고정 composite layer이며 일반 DAG는 미지원 |

각 dataclass는 자기 파라미터만 검증한다 (예: `Conv2dSpec.__post_init__`이
`kernel_size > 0`을 확인). 이 검증은 JSON에서 역직렬화될 때도 그대로
적용된다 -- `serialization.py`가 `LayerCls(**kwargs)`로 생성자를
직접 호출하기 때문이다.

## 5. Shape Inference

`shape_inference.infer_model_shapes(model_spec)`이 핵심 함수다.
`ModelSpec.input_shape`부터 시작해 각 레이어를 순서대로 지나며
`LayerShapeInfo` 리스트를 만든다:

```python
@dataclass
class LayerShapeInfo:
    index: int
    layer: LayerSpec
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    inferred: dict[str, int]   # 예: {"in_features": 401408}, {"num_features": 32}
```

`inferred`에는 이전 레이어의 shape에서만 결정 가능한 파라미터
(`Conv2d.in_channels`, `BatchNorm2d.num_features`, `Linear.in_features`)가
담긴다. `builder.py`는 이 값을 그대로 사용해 `torch.nn.Module`을
생성하므로, shape 계산 로직이 두 곳에 중복되지 않는다.

`format_shape_trace(trace)`는 향후 UI가 다음과 같은 형태로 표시할 수
있도록 사람이 읽기 쉬운 문자열을 만들어준다:

```text
Conv2d
[3, 224, 224] -> [32, 224, 224]
```

### 레이어 연결 오류도 여기서 잡힌다

`Conv2d`/`MaxPool2d`/`AdaptiveAvgPool2d`/`BatchNorm2d`는 3차원
`(C, H, W)` 입력을, `Linear`는 1차원 `(features,)` 입력을 요구한다.
예를 들어 `Flatten` 다음에 `Conv2d`가 오면:

```text
ModelValidationError: Layer 1 (Conv2d): expected a 3D input shape
(channels, height, width) but got shape (192,) (1D). Check the layer
connected before this one.
```

처럼 어떤 레이어(인덱스 + 이름)가 문제인지 명시하는 메시지를 낸다.
Conv2d/MaxPool2d 출력 크기가 0 이하로 줄어드는 경우도 같은 방식으로
잡는다.

## 6. Validation

파라미터 검증(예: `kernel_size <= 0`)과 shape 연결 검증(예: Conv2d에
1D 입력)은 서로 다른 레이어에서 이뤄진다:

* **파라미터 검증**: `specs.py`의 각 dataclass `__post_init__`.
  shape 정보 없이 그 자체로 판단 가능한 값만 검사한다
  (`kernel_size`, `stride`, `out_channels`, `out_features`,
  `padding`, `p` 등). `bool` 필드(`ReLUSpec.inplace`,
  `LinearSpec.bias`)는 `True`/`False`만 허용하고, 문자열/정수
  등 truthy/falsy 값을 암묵적으로 변환하지 않는다 (JSON을 사람이
  직접 편집할 수 있기 때문). `MaxPool2dSpec.padding`은
  `kernel_size // 2` 이하인지도 여기서 확인한다 -- `torch.nn.MaxPool2d`는
  이 제약을 생성 시점이 아니라 `forward()` 호출 시점에서만 검사하므로,
  그대로 두면 `build_model()`을 통과한 뒤에야(심지어 TorchScript
  export/추론 시점에) 실패할 수 있다.
* **shape 연결 검증**: `shape_inference.py`. 이전 레이어의 출력
  shape이 있어야만 판단 가능하다.

`validation.validate_model_spec(model_spec)`은 이 둘을 조합한 단일
진입점이다: `ModelSpec`/`LayerSpec` 생성 시점에 이미 파라미터 검증이
끝나 있으므로, 이 함수는 `shape_inference.infer_model_shapes`를 호출해
shape 체인을 검증하고 트레이스를 반환한다. `builder.build_model`도
내부적으로 동일한 함수를 호출하므로, "빌드하면 항상 검증도 된다."

모든 검증 실패는 `image_ai_studio.model_definition.errors.ModelValidationError`
(= `ValueError`의 서브클래스) 하나로 통일되어 있고, 메시지는 향후
UI에 그대로 노출해도 이해할 수 있도록 작성했다.

## 7. PyTorch Model Builder

```python
model = build_model(model_spec)   # -> nn.Sequential
```

`build_model`은 먼저 `validation.validate_model_spec`을 호출해 shape
트레이스를 얻고, 각 `LayerShapeInfo`를 대응하는 `nn.Module` 생성
함수에 넘긴다.
예를 들어 `Linear`는 다음처럼 만들어진다:

```python
def _build_linear(layer: LinearSpec, inferred: dict[str, int]) -> nn.Module:
    return nn.Linear(in_features=inferred["in_features"], out_features=layer.out_features, bias=layer.bias)
```

즉 JSON/`LinearSpec`에는 `out_features`만 있으면 되고, `in_features`는
사용자가 계산할 필요가 없다. shape 검증에 실패하면 `build_model`은
예외를 던지고 **부분적으로 생성된 모델을 반환하지 않는다**.

## 8. JSON 직렬화

```python
save_model_spec(model_spec, "example_model.json")
model_spec = load_model_spec("example_model.json")
```

레이어 JSON은 판별자(discriminator) 필드 `"type"` + 그 레이어
dataclass의 필드 그대로다:

```json
{
  "name": "example_model",
  "input_shape": [3, 224, 224],
  "layers": [
    {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "batch_norm2d"},
    {"type": "relu"},
    {"type": "flatten"},
    {"type": "linear", "out_features": 10}
  ]
}
```

`in_channels`/`num_features`/`in_features`처럼 shape에서 유도되는
값은 JSON에 절대 포함되지 않는다 -- 로드/빌드할 때마다
`shape_inference`가 다시 계산한다. `model_spec_from_dict`은 알 수
없는 `type`, 누락된 `type` 필드, 문자열이 아닌 `type`(예:
`{"type": ["conv2d"]}`), 누락된 필수 파라미터를 모두
`ModelValidationError`로 변환해 사용자에게 이해 가능한 메시지를
전달한다. `type`이 문자열인지는 registry에서 `dict.get(type_name)`을
호출하기 전에 검사한다 -- 그렇지 않으면 리스트처럼 hashable하지 않은
값이 raw `TypeError`(unhashable type)를 먼저 던진다.

Round-trip(`ModelSpec -> JSON -> ModelSpec`)은 dataclass의 기본
`__eq__`로 의미적 동일성이 보장된다 (테스트로 검증됨).

## 9. TorchScript Export 흐름

Phase 0에서 만든 `image_ai_studio.export.torchscript_exporter.TorchScriptExporter`를
그대로 재사용한다 (중복 구현 없음):

```text
ModelSpec
    -> build_model()           (model_definition/builder.py)
    -> nn.Module.eval()
    -> TorchScriptExporter().export(...)   (Phase 0 코드, 변경 없음)
    -> model.pt + metadata.json
    -> torch.jit.load(model.pt)
    -> Phase 0 parity.compare_outputs로 원본 모델과 재로드된 모델의 출력 비교
```

`tests/model_definition/test_torchscript_integration.py`가 이 전체
흐름이 실제로 동작하는지 검증한다. `TorchScriptExporter.export()`는
`state_dict_path`가 실제로 존재할 것을 요구하지 않으므로 (파일이
없으면 `state_dict_sha256`을 `None`으로 기록), Phase 1 모델처럼
디스크에 저장된 `state_dict`가 없는 경우에도 그대로 사용할 수 있다.

AOTInductor 연동은 Phase 1에서 하지 않는다.

## 9-1. C++ TorchScript Runner까지의 E2E 검증

위 흐름은 `torch.jit.load()`로 Python 프로세스 안에서만 재로드/비교한다.
`scripts/run_phase1_e2e.py`는 여기서 한 단계 더 나아가, `ModelSpec`으로
정의한 모델이 **실제 Phase 0 C++ TorchScript 러너**(`run_torchscript.exe`)
까지 도달하는지 검증한다:

```text
Model JSON (examples/models/phase1_e2e_model.json)
    -> load_model_spec()                (serialization.py, 변경 없음)
    -> validate_model_spec()            (validation.py, 변경 없음)
    -> build_model()                    (builder.py, 변경 없음)
    -> TorchScriptExporter().export()   (Phase 0 코드, 변경 없음)
    -> model.pt
    -> run_torchscript.exe              (Phase 0 C++ 러너, 변경 없음)
    -> Phase 0 parity.compare_outputs로 Python 참조 출력과 비교
```

새로 만든 것은 없다:

* **새 C++ runner를 만들지 않았다.** `run_torchscript.exe`는 `--model`을
  CLI 인자로 받는 범용 러너이므로, TinyCNN/TinyResidualCNN에 쓰던 바로
  그 실행 파일이 이 모델도 그대로 실행한다.
* **새 TorchScript exporter를 만들지 않았다.** Phase 0의
  `TorchScriptExporter`를 그대로 호출한다.
* **새 tensor I/O/parity 코드를 만들지 않았다.** `parity.tensor_io.save_tensor`,
  `parity.compare_outputs.compare_outputs`, 그리고 러너 실행/비교/결과
  로깅을 담당하는 `tools.run_and_compare.run_case()`를 그대로 재사용한다.
  `run_case()`에는 입력 텐서 경로를 지정할 수 있는 `input_bin`/`input_meta`
  파라미터만 추가했다 (미지정 시 기존 Phase 0 동작과 100% 동일) --
  Phase 1 E2E 모델의 입력 shape `(3, 16, 16)`이 Phase 0 공유 입력의
  `(3, 224, 224)`와 다르기 때문에 필요한 최소한의 변경이다.

**JSON이 각 예시 모델의 단일 소스(single source of truth)다.** 처음에는
`examples/models/phase1_e2e_model.json`과 짝을 이루는 `_reference_model_spec()`
Python 함수를 두고 둘이 같은지(`==`) 매번 검사했으나, 이 방식은 예시
모델이 하나 늘어날 때마다 JSON과 Python 코드를 둘 다 손으로 동기화해야
해서 확장성이 없었다 (실제로 두 번째 예시
`examples/models/phase1_e2e_alt_model.json`을 추가하면서 이 패턴이
깨졌다). JSON ↔ `ModelSpec` 동등성 자체는 어느 한 예시 모델에 묶이지
않는 일반적인 성질이라 `tests/model_definition/test_serialization.py`의
round-trip 테스트가 이미 검증하므로, `run_phase1_e2e.py`는 `--model-json`로
받은 JSON을 그대로 신뢰하고 `load_model_spec()` + `validate_model_spec()`만
수행한다.

실행:

```bash
python scripts/run_phase1_e2e.py
python scripts/run_phase1_e2e.py --model-json examples/models/phase1_e2e_alt_model.json
```

`--model-json`은 `examples/models/phase1_e2e_model.json`을 기본값으로
하므로, 인자 없이 실행하면 기존과 동일하게 동작한다. 다른 JSON을
지정하면 같은 검증 흐름(JSON -> ModelSpec -> validate -> build_model ->
TorchScript export -> `run_torchscript.exe` -> parity)을 그대로 재사용해
임의의 `ModelSpec` 모델을 검증할 수 있다 -- 아티팩트(`model.pt`, 입력
텐서, 참조 출력)는 `ModelSpec.name` 기준으로 경로가 정해진다. **이름이
다른 모델끼리는 아티팩트가 분리**되지만, 서로 다른 JSON이라도 `name`
필드가 같으면 같은 경로를 가리키게 되어 아티팩트를 덮어쓸 수 있다 --
현재는 이를 막는 별도의 이름 유일성 검사가 없으므로, 예시 모델을
추가할 때는 `name`이 겹치지 않도록 주의해야 한다.

`run_torchscript.exe`가 아직 빌드되지 않았다면 `scripts/build_torchscript.py`를
그대로 호출해 자동으로 빌드한다 (새 빌드 로직 없음). CUDA가 없는
머신에서는 CUDA 케이스가 `SKIPPED`로 보고되며 CPU로 조용히
폴백하지 않는다 (Phase 0과 동일한 정책).

이 스크립트는 `pytest`가 아니라 별도 스크립트다 -- C++ 러너 빌드를
요구하지 않는 `pytest`의 전체 unit test와, 빌드된 C++ 러너가 있어야
의미 있는 E2E 검증을 분리하기 위해서다 (정확한 테스트 개수는 새
테스트가 추가될 때마다 바뀌므로 여기서는 명시하지 않는다).

## 10. 현재 제한 사항

* `ModelSpec.layers`는 평평한 리스트만 지원한다. `ResidualBlockSpec`
  (고정된 skip connection 하나, Phase 2에서 추가)처럼 "레이어 하나로
  보이는 고정된 composite"는 표현할 수 있지만, 사용자가 임의로 분기/합류
  구조를 정의하는 일반 DAG는 여전히 지원하지 않는다.
* `input_shape`은 3차원 `(C, H, W)` 이미지 입력만 지원한다.
* `AdaptiveAvgPool2d`는 정사각형 `(output_size, output_size)`만
  지원한다 (직사각형 `(h, w)` 출력 미지원).
* `Conv2d`/`MaxPool2d`는 정사각형 kernel/stride/padding(정수 하나)만
  지원한다 (`(kh, kw)` 튜플 미지원).
* `Conv2d`/`MaxPool2d`의 `dilation`은 지원하지 않는다 (항상 1로
  가정하고 shape을 계산한다).
* CUDA 경로에 대한 Phase 1 전용 테스트는 없다. Phase 0의 CUDA 검증
  코드는 그대로 유지되며, Phase 1의 모든 unit test는 CPU에서만
  동작하도록 작성했다.
* AOTInductor 경로는 Phase 1 신규 코드에서 의도적으로 제외했다
  (1번 섹션 참고). 필요해지면 별도로 재평가한다.

## 11. ResidualBlock(구현됨) / 향후 DAG 확장 방향

`LayerSpec`은 필드가 없는 마커 베이스 클래스이고, `specs.py` /
`shape_inference.py` / `builder.py` 모두 "레이어 타입 -> 처리 함수"
딕셔너리 조회(`_SHAPE_HANDLERS`, `_BUILDERS`, `_LAYER_REGISTRY`)로
동작한다. 이 설계 덕분에 Phase 2에서 `ResidualBlockSpec`을 추가할 때
기존 레이어 처리 코드는 전혀 바꾸지 않고 세 딕셔너리에 항목만
추가하면 됐다. 상세 설계와 실제 구현/검증 결과(shape inference 공식,
builder 재사용 방식, 테스트 결과, C++ E2E parity)는
`docs/phase2_residual_block_design.md`에 정리했다.

`ModelSpec.layers`가 완전한 DAG(임의의 분기/합류)를 지원해야 하는
시점이 오면, `ModelSpec`은 그대로 두고 `layers: list[LayerSpec]`
옆에 `graph: GraphSpec | None` 같은 대안 표현을 추가하는 방향을
권장한다 -- 기존 Sequential 기반 모델과 테스트를 깨지 않기 위해서다.
고정된 composite layer(`ResidualBlockSpec`)만으로 충분한 현재로서는
이런 일반 DAG 계층을 미리 만들지 않는다. 기존 `TinyResidualCNN`은
Phase 0 테스트용 모델로 그대로 유지된다.
