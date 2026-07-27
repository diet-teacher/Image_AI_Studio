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
시점에 바로 걸린다.

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

## 10. 현재 제한 사항

* `ModelSpec.layers`는 평평한 리스트만 지원한다. 분기/합류가 있는
  DAG(예: ResidualBlock)는 아직 표현할 수 없다.
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

## 11. 향후 ResidualBlock / DAG 확장 방향

`LayerSpec`은 필드가 없는 마커 베이스 클래스이고, `specs.py` /
`shape_inference.py` / `builder.py` 모두 "레이어 타입 -> 처리 함수"
딕셔너리 조회(`_SHAPE_HANDLERS`, `_BUILDERS`, `_LAYER_REGISTRY`)로
동작한다. 따라서 향후 다음과 같은 composite spec을 추가할 때, 기존
레이어 처리 코드를 변경할 필요가 없다:

```python
@dataclass
class ResidualBlockSpec(LayerSpec):
    out_channels: int
    stride: int = 1
    # 내부적으로 Conv-BN-ReLU-Conv-BN + shortcut을 표현
```

필요한 작업은 세 딕셔너리에 `ResidualBlockSpec`에 대한 항목을
추가하는 것뿐이다:

1. `shape_inference._SHAPE_HANDLERS[ResidualBlockSpec]` -- 내부
   sub-layer들의 shape을 순서대로 계산해 최종 output shape과
   (필요하다면) 내부 in_channels를 반환.
2. `builder._BUILDERS[ResidualBlockSpec]` -- `models/residual_block.py`의
   `ResidualBlock`처럼 내부 sub-module들을 조립한 `nn.Module`(또는
   전용 `nn.Module` 서브클래스)을 반환.
3. `serialization._LAYER_REGISTRY["residual_block"]` -- JSON
   `"type"` 매핑 추가.

`ModelSpec.layers`가 완전한 DAG(임의의 분기/합류)를 지원해야 하는
시점이 오면, `ModelSpec`은 그대로 두고 `layers: list[LayerSpec]`
옆에 `graph: GraphSpec | None` 같은 대안 표현을 추가하는 방향을
권장한다 -- 기존 Sequential 기반 모델과 테스트를 깨지 않기 위해서다.
이번 Phase 1 구현에서는 `ResidualBlockSpec` 자체를 만들지 않았고,
기존 `TinyResidualCNN`은 Phase 0 테스트용 모델로 그대로 유지된다.
