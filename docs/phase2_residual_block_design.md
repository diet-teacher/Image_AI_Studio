# Phase 2 설계 검토: ResidualBlockSpec (composite layer)

이 문서는 설계 검토 결과만 정리한다. **이번 작업에서 아래 내용은
구현하지 않았다** -- Phase 1 코드는 변경 없음.

## 1. 범위

Phase 2에서 바로 일반 DAG(`GraphSpec`/`NodeSpec`/`EdgeSpec`)로
확장하지 않고, 현재 Sequential 구조(`ModelSpec.layers: list[LayerSpec]`)를
그대로 유지하면서 `ResidualBlockSpec` 하나를 "레이어 하나처럼 보이는
composite"로 추가하는 방향을 검토한다.

```text
ModelSpec.layers = [
    Conv2dSpec(...),
    ResidualBlockSpec(out_channels=32, stride=2),   # 내부에 skip connection 포함
    FlattenSpec(),
    LinearSpec(...),
]
```

`ModelSpec`/`shape_inference.infer_model_shapes`/`builder.build_model`
입장에서 `ResidualBlockSpec`은 다른 `LayerSpec`과 동일하게 "하나의
`input_shape` -> 하나의 `output_shape`"만 갖는 블랙박스로 취급된다.
내부에 skip connection(분기+합류)이 있다는 사실은 `ResidualBlockSpec`
자신의 shape 핸들러/builder 함수 안에서만 다뤄지고, `ModelSpec` 수준의
리스트 구조나 다른 레이어 처리 코드는 전혀 몰라도 된다.

## 2. 현재 구조가 이미 지원하는 확장 지점

`specs.py`/`shape_inference.py`/`builder.py`/`serialization.py` 네
모듈 모두 "레이어 타입(class) -> 처리 함수" 딕셔너리 조회로 동작한다:

| 모듈 | 레지스트리 | 키 |
|---|---|---|
| `shape_inference.py` | `_SHAPE_HANDLERS` | `type(layer)` (클래스) |
| `builder.py` | `_BUILDERS` | `type(layer)` (클래스) |
| `serialization.py` | `_LAYER_REGISTRY` / `_TYPE_NAMES` | JSON `"type"` 문자열 <-> 클래스 |

`validate_model_spec`/`infer_model_shapes`는 레이어 타입을 하나도
모른 채 이 딕셔너리들만 조회하므로, **새 딕셔너리 항목 3개를 추가하는
것만으로 새 레이어 타입이 기존 파이프라인(검증 -> 빌드 -> export ->
E2E) 전체를 그대로 통과**한다. 이 설계가 Phase 1에서 이미 검증된
상태이므로, `ResidualBlockSpec` 추가는 "새 기능"이 아니라 "기존
레지스트리에 항목 추가"에 가깝다.

## 3. 데이터 구조: `ResidualBlockSpec`

`src/image_ai_studio/models/residual_block.py`의 기존 `ResidualBlock`
(`Conv-BN-ReLU-Conv-BN + shortcut -> Add -> ReLU`, Phase 0에서
`TinyResidualCNN`을 통해 TorchScript CPU/CUDA 양쪽에서 이미 bit-identical
parity로 검증됨)을 그대로 감싼다. 내부 kernel_size(3), padding(1),
bias(False) 등은 `ResidualBlock`과 동일하게 고정하고 확장하지 않는다
(필요해지면 별도로 재검토).

```python
# specs.py (추가안, 미구현)
@dataclass
class ResidualBlockSpec(LayerSpec):
    """Conv-BN-ReLU-Conv-BN + shortcut -> Add -> ReLU.
    in_channels는 이전 레이어 출력에서 자동 계산.
    내부 kernel_size/padding은 ResidualBlock과 동일하게 고정(3x3, padding=1)."""

    out_channels: int
    stride: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("out_channels", self.out_channels)
        _require_positive_int("stride", self.stride)
```

`Conv2dSpec`과 동일한 검증 헬퍼(`_require_positive_int`)를 그대로
재사용하므로 `specs.py`에 새 헬퍼 함수는 필요 없다.

## 4. Shape Inference

`ResidualBlock`의 두 conv(`conv1`: stride=`stride`, `conv2`: stride=1)와
shortcut(`stride`, kernel=1)은 모두 동일한 출력 크기 공식
`floor((h - 1) / stride) + 1`로 귀결된다 (아래 "검증" 참고). 따라서
기존 `_spatial_output_size` 헬퍼를 `Conv2dSpec`과 동일하게 한 번만
호출하면 된다 -- 새 산술 로직이 필요 없다.

```python
# shape_inference.py (추가안, 미구현)
def _residual_block_shape(
    layer: ResidualBlockSpec, input_shape: Shape, index: int
) -> tuple[Shape, dict[str, int]]:
    _require_rank(layer, input_shape, index, rank=3, expected_desc="(channels, height, width)")
    in_channels, h_in, w_in = input_shape
    h_out = _spatial_output_size(
        h_in, kernel_size=3, stride=layer.stride, padding=1,
        index=index, layer_name="ResidualBlock", dim_name="height",
    )
    w_out = _spatial_output_size(
        w_in, kernel_size=3, stride=layer.stride, padding=1,
        index=index, layer_name="ResidualBlock", dim_name="width",
    )
    return (layer.out_channels, h_out, w_out), {"in_channels": in_channels}


_SHAPE_HANDLERS[ResidualBlockSpec] = _residual_block_shape
```

`_require_rank`를 재사용하므로 `Flatten` 다음에 `ResidualBlockSpec`이
오는 등 잘못된 연결도 `Conv2d`와 동일한 형식의 메시지로 자동으로
잡힌다.

**검증(공식이 실제 `ResidualBlock`과 일치하는지):**

* `conv1` (main path): `kernel=3, padding=1, stride=s` ->
  `floor((h + 2 - 3) / s) + 1 = floor((h - 1) / s) + 1`
* `shortcut` (projection, `in_channels != out_channels or stride != 1`일 때):
  `kernel=1, padding=0, stride=s` -> `floor((h - 1) / s) + 1`
* `conv2` (main path, stride 항상 1): `kernel=3, padding=1, stride=1` ->
  `floor((h_out + 2 - 3) / 1) + 1 = h_out` (그대로 유지, 축소 없음)

세 경로가 동일한 최종 크기로 수렴하므로 `Add(main, shortcut)`이
항상 shape이 맞는다 -- 이는 새로 만드는 성질이 아니라 기존
`ResidualBlock` 구현이 이미 보장하고 있는 불변조건을 shape_inference
쪽에서 한 번 더 명시적으로 계산해주는 것뿐이다.

## 5. Builder

새 `nn.Module`을 만들지 않고 기존 `ResidualBlock`을 그대로 인스턴스화한다.

```python
# builder.py (추가안, 미구현)
from image_ai_studio.models.residual_block import ResidualBlock

def _build_residual_block(layer: ResidualBlockSpec, inferred: dict[str, int]) -> nn.Module:
    return ResidualBlock(
        in_channels=inferred["in_channels"],
        out_channels=layer.out_channels,
        stride=layer.stride,
    )


_BUILDERS[ResidualBlockSpec] = _build_residual_block
```

`ResidualBlock.forward()`는 입력값에 따라 분기하는 control flow가
없다 (shortcut을 `nn.Identity()`로 할지 `nn.Sequential(Conv,BN)`으로
할지는 `__init__` 시점에 고정됨). 따라서 `torch.jit.trace` 관점에서
`nn.Sequential` 안에 다른 레이어들과 함께 들어가도 특별 취급이
필요 없다 -- `TinyResidualCNN`이 이미 이 사실을 Phase 0에서 CPU/CUDA
양쪽 TorchScript export로 검증했다.

## 6. Serialization

`_LAYER_REGISTRY`/`_TYPE_NAMES`는 클래스와 JSON 필드를 범용적으로
다루므로 (`asdict(layer)`), 레지스트리에 한 줄만 추가하면 끝난다 --
`_layer_to_dict`/`_layer_from_dict`/`model_spec_to_dict`/`model_spec_from_dict`
어느 것도 수정할 필요가 없다.

```python
# serialization.py (추가안, 미구현)
_LAYER_REGISTRY: dict[str, type[LayerSpec]] = {
    ...,
    "residual_block": ResidualBlockSpec,
}
```

JSON 예시:

```json
{"type": "residual_block", "out_channels": 32, "stride": 2}
```

## 7. Validation

`validation.validate_model_spec`은 이미 레이어 타입에 무관하게
`infer_model_shapes`를 호출하는 구조라 **수정이 전혀 필요 없다**.
`ResidualBlockSpec.__post_init__`(파라미터 검증)과
`shape_inference._SHAPE_HANDLERS[ResidualBlockSpec]`(shape 연결
검증)만 있으면 기존과 동일한 수준의 검증이 자동으로 적용된다.

## 8. 테스트 계획

기존 테스트 파일 구조를 그대로 따라 각 파일에 케이스를 추가한다
(새 테스트 파일 불필요):

* `test_specs_validation.py`
  * `out_channels <= 0` / `stride <= 0` 거부
* `test_shape_inference.py`
  * `stride=1`: 공간 크기 유지 (`(16,32,32) -> ResidualBlockSpec(out_channels=32) -> (32,32,32)`)
  * `stride=2`: 공간 크기 절반 (`-> (32,16,16)`)
  * `Flatten` 다음에 `ResidualBlockSpec` -> `ModelValidationError` (3D 입력 요구)
  * `inferred == {"in_channels": ...}` 확인
* `test_serialization.py`
  * `ResidualBlockSpec` 포함 모델의 JSON round-trip
  * `{"type": "residual_block", ...}` 로드
* `test_builder.py`
  * 생성된 모듈이 `ResidualBlock` 인스턴스인지, `in_channels`/`out_channels`/`stride`가
    올바르게 전달됐는지
  * dummy tensor forward 후 출력 shape이 `shape_inference` 예측과 일치하는지
  * `in_channels != out_channels` (projection shortcut 경로)와
    `in_channels == out_channels, stride=1` (`nn.Identity()` shortcut 경로)
    둘 다 forward 확인
* `test_torchscript_integration.py`
  * `ResidualBlockSpec`을 포함한 모델을 `build_model` -> `torch.jit.trace` ->
    reload -> parity까지 확인 (기존 테스트와 동일한 패턴, 모델 스펙만 교체)

## 9. 예상 변경 파일 (구현 시)

| 파일 | 변경 내용 |
|---|---|
| `src/image_ai_studio/model_definition/specs.py` | `ResidualBlockSpec` dataclass 추가 |
| `src/image_ai_studio/model_definition/shape_inference.py` | `_residual_block_shape` 추가, `_SHAPE_HANDLERS`에 등록 |
| `src/image_ai_studio/model_definition/builder.py` | `_build_residual_block` 추가 (`models.residual_block.ResidualBlock` import), `_BUILDERS`에 등록 |
| `src/image_ai_studio/model_definition/serialization.py` | `_LAYER_REGISTRY`에 `"residual_block"` 등록 |
| `tests/model_definition/test_specs_validation.py` | 파라미터 검증 테스트 추가 |
| `tests/model_definition/test_shape_inference.py` | shape/연결 검증 테스트 추가 |
| `tests/model_definition/test_serialization.py` | round-trip 테스트 추가 |
| `tests/model_definition/test_builder.py` | 빌드/forward 테스트 추가 |
| `tests/model_definition/test_torchscript_integration.py` | ResidualBlock 포함 모델 export 테스트 추가 |
| `docs/phase1_design.md` | 4번(지원 Layer) 표에 `residual_block` 추가, 11번 섹션을 "구현됨"으로 갱신 |

**변경 불필요**: `errors.py`, `validation.py`,
`export/torchscript_exporter.py`, `models/residual_block.py`(그대로
재사용), C++ 코드 전부, `run_phase1_e2e.py`(JSON 기반이라 레이어
타입에 무관).

## 10. 이번 Phase 2 검토에서 의도적으로 제외한 것

* **`GraphSpec`/`NodeSpec`/`EdgeSpec`** 등 일반 DAG 표현. 아직 필요한
  모델이 ResidualBlock 하나뿐이므로, 이를 위해 그래프 추상화 계층을
  미리 만들지 않는다. `ModelSpec.layers: list[LayerSpec]`는 그대로
  유지.
* **`ResidualBlockSpec` 내부의 임의 구조화** (예: sub-layer 리스트를
  사용자가 직접 정의). `ResidualBlock`과 동일하게 `out_channels`/`stride`
  두 파라미터로 고정. 커스터마이징이 필요해지는 시점에 재검토.
* **`kernel_size`/`dilation` 등 추가 옵션**. `Conv2d`/`MaxPool2d`도
  아직 정사각형 정수 하나만 지원하는 것과 동일한 이유로 보류.
* **중첩 composite** (ResidualBlock 안에 또 다른 composite). 현재
  설계는 `LayerSpec` 하나가 정확히 하나의 `nn.Module`에 대응하는
  1단계 구조만 가정한다.
