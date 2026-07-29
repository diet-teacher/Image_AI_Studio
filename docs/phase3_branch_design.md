# Phase 3: BranchSpec (제한된 분기/합류)

이 문서는 설계 검토와 실제 구현 결과를 함께 정리한다. Phase 2의
`ResidualBlockSpec`(내부 구조가 고정된 composite)을 일반화해서, 사용자가
branch 내부 구성을 직접 정의할 수 있는 `BranchSpec`을 추가했다. 일반
DAG(`GraphSpec`/`NodeSpec`/`EdgeSpec`)는 이번 Phase 범위가 아니다.

## 1. 범위

`ModelSpec.layers: list[LayerSpec]` 구조는 그대로 유지한다. `BranchSpec`은
"입력 하나 -> N개 병렬 branch -> merge로 하나로 합류"라는 **제한된** 패턴만
표현한다 (즉시 합류하지 않는 skip, 임의 거리 skip, 다중 레벨 분기는 지원
안 함):

```text
ModelSpec.layers = [
    Conv2dSpec(...),
    ReLUSpec(),
    BranchSpec(
        branches=[
            [Conv2dSpec(...), BatchNorm2dSpec()],
            [IdentitySpec()],
        ],
        merge="add",
    ),
    ReLUSpec(),
]
```

`ResidualBlockSpec`은 사실 `BranchSpec`의 "내부가 고정된 특수 케이스"로 볼 수
있다: `BranchSpec(branches=[[Conv,BN,ReLU,Conv,BN],[Identity 또는 Conv,BN]],
merge="add")`가 정확히 `ResidualBlock`의 구조다. `BranchSpec`은 이걸
일반화해서 branch 개수/내용/merge 종류를 사용자가 고르게 한 것이다.

## 2. `BranchSpec` / `IdentitySpec` 데이터 구조

```python
@dataclass
class BranchSpec(LayerSpec):
    branches: list[list[LayerSpec]]
    merge: str = "add"   # "add" | "concat"


@dataclass
class IdentitySpec(LayerSpec):
    pass   # 입력을 그대로 통과시키는 passthrough
```

검증 규칙 (`specs.py`):

* `merge`는 `"add"`/`"concat"`만 허용.
* `branches`는 최소 2개.
* 각 branch는 비어있지 않은 `list[LayerSpec]`이어야 함 -- **빈 리스트(`[]`)를
  Identity로 암묵 해석하지 않는다.** skip path가 필요하면 반드시
  `IdentitySpec()`을 명시해야 한다 (아래 3번 참고).
* branch 내부에 `BranchSpec`이 다시 나오면(중첩) 거부한다 (Phase 3 범위 밖).

## 3. 빈 branch 대신 `IdentitySpec()`을 쓰기로 한 이유

구현 전 검토에서 두 방식을 비교했다:

* `branches=[[Conv2dSpec(...)], []]` -- 빈 리스트를 Identity로 암묵 해석
* `branches=[[Conv2dSpec(...)], [IdentitySpec()]]` -- 명시적 Identity 레이어

**`IdentitySpec()`을 선택했다.** 이유:

1. **JSON 가독성** -- `{"type": "identity"}`는 그 자체로 "의도적으로 아무것도
   안 함"이 드러나지만, `[]`는 그 의미를 미리 알고 있어야만 해석된다.
2. **검증 정책의 일관성** -- `ModelSpec.layers`가 항상 "비어있지 않은
   list[LayerSpec]"을 요구하는 것과 동일한 규칙을 branch에도 그대로 적용할
   수 있다 (특수 케이스 없음). 이 프로젝트는 지금까지 일관되게 암묵적
   변환보다 명시적 검증을 택해왔다 (bool 필드 strict 검증, MaxPool2d padding
   명시적 제약 등).
3. **구현 비용이 거의 0** -- `shape_inference.py`에 이미 있던 `_identity_shape`
   (ReLU/Dropout이 이미 재사용 중)를 그대로 재사용하면 되고, builder에
   `nn.Identity()` 한 줄, registry에 `"identity"` 한 줄이면 끝이다.

`[]`와 `IdentitySpec()`을 **둘 다** 허용하는 절충안은 채택하지 않았다 --
"같은 걸 표현하는 두 가지 방법"이 생기면 저장 시 어느 쪽으로 직렬화할지
애매해지기 때문이다.

## 4. Concat은 channel(axis 0/dim=1) 방향만 지원, `concat_dim` 필드 없음

`concat_dim`을 파라미터로 노출하지 않고 채널 방향으로 고정했다. 이유:

* 표준 CNN에서 branch 병합은 사실상 전부 channel 방향이다 (Inception,
  DenseNet 등). H/W 방향 concat은 이번에 확인된 요구가 없다.
* `concat_dim`을 없애면 shape inference/builder/JSON이 전부 단순해진다 --
  검증은 "H,W는 완전히 같아야 하고 C만 합산"으로 고정되고, builder는
  `torch.cat(outputs, dim=1)`로 하드코딩되어 "설계 시점 (C,H,W) 공간과
  런타임 batch-포함 텐서 사이의 오프셋(`concat_dim + 1`)" 같은 실수하기 쉬운
  지점이 아예 사라진다.
* 나중에 필요해지면 `concat_dim: int = 1`(런타임 기준, 채널이 기본값)을
  **추가**하는 것만으로 기존 JSON(필드 없음)은 자동으로 지금과 동일하게
  해석되어 breaking change 없이 확장할 수 있다.

## 5. Shape Inference

`_branch_shape`는 각 branch에 대해 기존 `infer_layer_shape`를 그대로 재귀
호출한다 (새 shape 엔진 없음):

```python
def _branch_shape(layer, input_shape, index):
    branch_output_shapes = []
    for branch in layer.branches:
        shape = input_shape
        for sub_layer in branch:
            shape, _ = infer_layer_shape(sub_layer, shape, index)
        branch_output_shapes.append(shape)

    if layer.merge == "add":
        # 모든 branch output shape이 완전히 같아야 함
        ...
    else:  # concat
        # 3D(C,H,W)만 허용, H/W는 같아야 하고 C만 합산
        ...
```

`BranchSpec` 자체는 3D로 제한하지 않는다 (각 branch의 sub-layer가 자기
rank 요구사항을 스스로 강제하므로 -- 예를 들어 `Flatten` 뒤에서 1D
branch+add도 형식상 허용된다). `Flatten` 다음에 `BranchSpec`이 3D를
요구하는 레이어(Conv2d 등)를 잘못 연결하면 기존과 동일한 형식의 에러
메시지로 잡힌다.

## 6. Builder

`_build_branch`는 branch마다 기존 `infer_layer_shape` + 개별 레이어
builder(`_build_plain_layer`)를 재사용해 `nn.Sequential`을 구성한 뒤
`BranchBlock`으로 묶는다:

```python
class BranchBlock(nn.Module):
    def __init__(self, branches: list[nn.Module], merge: str):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.merge = merge

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        if self.merge == "add":
            ...
        return torch.cat(outputs, dim=1)  # merge == "concat"
```

`BranchSpec`은 (다른 레이어와 달리) 자기 `input_shape` 문맥이 더 필요해서,
`_build_layer`에서 `isinstance(info.layer, BranchSpec)`으로 한 번 분기해
`info.input_shape`를 넘겨준다 -- 나머지 레이어 타입의 순수 딕셔너리 조회
방식(`_BUILDERS[type(layer)]`)은 그대로다. 이 한 곳을 제외하면 Phase 2와
동일하게 "딕셔너리에 항목만 추가" 패턴을 유지했다.

`nn.ModuleList` 크기는 생성 시점에 고정되고 `forward`는 데이터값에 의존하는
분기가 없어(`self.merge`는 인스턴스 생성 시 고정된 문자열), `ResidualBlock`과
동일하게 `torch.jit.trace`와 호환된다.

## 7. Serialization

`branches`가 `LayerSpec` 중첩 리스트라서, 일반 `asdict()`만으로는 각 서브
레이어의 `"type"` 판별자를 잃어버린다 (필드 값만 재귀 변환하지 클래스 정보는
모름). 이 부분만 `_layer_to_dict`/`_layer_from_dict`에 전용 코드를 추가했다
(Phase 2의 "레지스트리 한 줄"보다 손이 가는 유일한 지점):

```json
{
  "type": "branch",
  "merge": "add",
  "branches": [
    [{"type": "conv2d", "out_channels": 8, "kernel_size": 3, "padding": 1}, {"type": "batch_norm2d"}],
    [{"type": "identity"}]
  ]
}
```

## 8. Validation

`validation.py`는 수정하지 않았다 -- `validate_model_spec`은 레이어 타입을
몰라도 되는 구조라서 그대로다. `BranchSpec.__post_init__`(파라미터 검증)과
`shape_inference._SHAPE_HANDLERS[BranchSpec]`(shape 연결 검증)만으로 기존과
동일한 수준의 검증이 자동 적용된다.

## 9. 구현 순서 (스파이크 우선)

구현 전 검토에서 합의한 순서대로 진행했다:

1. `BranchSpec`(merge="add"만) 최소 버전을 실제 대상 파일에 바로 구현
2. **기존 `TorchScriptExporter`/`run_phase1_e2e.py`/C++ `run_torchscript.exe`를
   전혀 수정하지 않고** 최소 Add 모델로 CPU/CUDA parity 실측 -- 성공
   (10번 섹션)
3. `IdentitySpec` 추가
4. channel-only `Concat` 추가
5. validation 재확인, 5개 테스트 파일에 케이스 추가, 예시 JSON, 문서

이 순서 덕분에 "BranchBlock이 trace되고 기존 파이프라인을 그대로 통과하는가"
라는 유일한 미검증 가정을 가장 먼저, 가장 싼 시점에 확인했다.

## 10. 구현 결과 확인

실제 실행으로 확인한 결과 (Windows 11, PyTorch 2.12.1+cu126, GTX 1080):

* **최소 Add 스파이크**: `examples/models/phase3_branch_model.json`(당시
  Add만)으로 `python scripts/run_phase1_e2e.py --model-json ...`을 실행,
  **기존 스크립트/exporter/C++ 러너를 한 글자도 안 고치고** CPU/CUDA 모두
  `PASS`, `max_abs_error=0.0`(bit-identical) 확인.
* **Concat 추가 후**: shape inference 예측과 실제 `BranchBlock`
  forward(`torch.cat(dim=1)`) 결과가 홀수 spatial size(7x7, stride=2)에서도
  일치함을 교차 검증 (`test_branch_concat_matches_actual_branch_block_on_odd_spatial_size`).
* **TorchScript 통합 테스트**: `Conv -> ReLU -> Branch(Add: Conv+BN/Identity)
  -> ReLU -> Branch(Concat: Conv/MaxPool) -> Flatten -> Linear` 모델을
  `build_model -> torch.jit.trace -> 저장/재로드 -> parity`까지 실행, PASS.
* **최종 Phase 3 E2E**(Add+Identity+Concat을 모두 포함한
  `examples/models/phase3_branch_model.json`): CPU/CUDA 둘 다 `PASS`,
  `max_abs_error=0.0`.
* **pytest**: 전체 unit test PASS, FAIL 0, SKIPPED 0 (정확한 개수는
  테스트가 추가될 때마다 바뀌므로 여기서는 명시하지 않는다).
* **Phase 0/1/2 regression**: `tiny_cnn`/`tiny_residual_cnn` CPU/CUDA,
  기존 Phase 1/2 E2E 예시 모델 전부 재실행하여 PASS 확인 (영향 없음).

## 11. 실제 변경 파일

| 파일 | 변경 내용 |
|---|---|
| `src/image_ai_studio/model_definition/specs.py` | `BranchSpec`, `IdentitySpec` 추가 |
| `src/image_ai_studio/model_definition/shape_inference.py` | `_branch_shape` 추가/등록, `IdentitySpec`은 기존 `_identity_shape` 재사용 |
| `src/image_ai_studio/model_definition/builder.py` | `_build_branch`/`_build_identity` 추가/등록, `_build_layer`에 `BranchSpec` 전용 분기 1곳 (input_shape 전달 목적), `_build_plain_layer` 헬퍼로 기존 로직 재사용 |
| `src/image_ai_studio/model_definition/serialization.py` | `branches` 중첩 필드 전용 (역)직렬화, `"branch"`/`"identity"` registry 등록 |
| `src/image_ai_studio/models/branch_block.py` | `BranchBlock` (신규, `ResidualBlock`과 나란히) |
| `tests/model_definition/test_*.py` (5개) | Phase 2와 동일 패턴으로 케이스 추가 |
| `examples/models/phase3_branch_model.json` | Add+Identity+Concat을 모두 포함한 예시 모델 (신규) |
| `docs/phase1_design.md` | 지원 Layer 표에 `branch`/`identity` 반영 (최소) |

**변경 없음**: `errors.py`, `validation.py`, `export/torchscript_exporter.py`,
`models/residual_block.py`, C++ 코드 전부, `run_phase1_e2e.py`,
`run_and_compare.py`.

## 12. 이번 Phase 3에서 의도적으로 구현하지 않은 것

* `GraphSpec`/`NodeSpec`/`EdgeSpec`, 일반 DAG, 임의 Branch/Merge
* 즉시 합류하지 않는 skip (U-Net/DenseNet류 long skip connection)
* 중첩 `BranchSpec` (branch 안에 또 다른 branch)
* multi-input / multi-output 모델 (외부 입력/출력은 계속 1개)
* `concat_dim` 노출, `Mul` 등 Add/Concat 외 merge 연산
* dynamic control flow, PyTorch FX
* training / UI / detection / segmentation

이 목록은 필요성이 구체적으로 확인되기 전까지 보류하며, `ModelSpec` 구조
변경 없이 `BranchSpec`을 그래프의 한 노드 종류로 재사용하거나 독립적으로
대체하는 방향으로 나중에 확장할 수 있다.
