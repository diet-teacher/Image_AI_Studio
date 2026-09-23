# Phase 14 Model Designer

## 목적과 사용자 흐름

Phase 14는 기존 `model_definition` 계층을 대체하지 않고 그 canonical
`ModelSpec` JSON을 GUI에서 편집하는 `Model Designer` 탭을 제공합니다.
기본 흐름은 다음과 같습니다.

1. `Model Designer` 탭에서 모델 이름과 `(C, H, W)` 입력 shape를 정합니다.
2. 순서형 레이어 목록에서 레이어를 추가·삭제·이동하고 타입별 파라미터를
   편집합니다.
3. `Validate`로 canonical validation과 레이어별 shape trace를 확인합니다.
4. `Save...`로 JSON을 저장하거나 `Save for Training...`으로 저장과 Training
   탭 handoff를 한 번에 수행합니다.
5. Training 탭에서 dataset, output, 학습 옵션을 설정하고 기존 학습 흐름을
   실행합니다.

`Save for Training...`은 새로운 request나 controller 경로가 아닙니다.
성공한 저장 파일의 정규화된 절대경로를 기존 TrainingPage의 `Model JSON`
입력에 채우며, 학습 시작 시 기존 `_build_request()`와
`build_training_request()`가 그 값을 소비합니다. 사용자는 handoff 이후에도
Browse나 직접 입력으로 다른 외부 JSON을 선택할 수 있고, 그 manual override가
다음 요청의 권위 있는 값입니다.

## Canonical schema와 지원 편집 범위

편집기의 단일 진실 공급원은 기존 `ModelSpec`, `LayerSpec`, serialization,
shape inference, validation, builder API입니다. GUI 전용 schema나 별도의
ViewModel 규칙은 만들지 않습니다. 현재 폼은 다음 등록 레이어를 지원합니다.

- `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`
- `AdaptiveAvgPool2d`, `Flatten`, `Linear`, `Dropout`
- `ResidualBlock`, `Identity`
- `Branch`의 `add` 또는 channel `concat`

레이어는 표시된 순서대로 실행됩니다. 추가·삭제·위/아래 이동으로 순서를
바꾸며, 선택한 레이어의 타입별 파라미터 폼이 갱신됩니다. `Branch`는 여러
개의 구조화된 branch와 각 branch 안의 순서형 레이어를 편집하지만, nested
`Branch`는 제공하지 않습니다. 일반 arbitrary DAG, graph canvas, 자유로운
long skip 연결도 지원 범위가 아닙니다.

## Validation과 shape feedback

`Validate`와 저장 동작은 편집기 값을 `model_spec_from_dict()`로 구성한 뒤
기존 `validate_model_spec()`을 호출합니다. 따라서 필드 범위, 레이어 연결,
shape 규칙은 GUI가 복제하지 않고 canonical 계층이 결정합니다. 성공하면 각
레이어의 입력·출력 shape와 inferred parameter를 기반으로 한 trace가 표시되고,
실패하면 첫 오류를 간결하게 표시한 채 사용자가 값을 수정할 수 있습니다.

입력 shape와 레이어 integer 필드는 Python arbitrary-precision 정수를 보존하는
텍스트 기반 편집기를 사용합니다. `QSpinBox`의 32-bit 한계로 값을 자르지
않으며, 예를 들어 `2147483648`도 canonical 규칙이 허용하는 위치에서는 저장과
재로드까지 같은 정수로 유지됩니다. 기존 최소값과 semantic validation은
그대로 적용됩니다.

Float 필드는 finite 값에 대해 사용자가 입력한 tiny/scientific notation을
읽고 canonical float로 전달합니다. 표시 재구성에는 round-trip 가능한 표현을
사용하므로 작은 `eps`나 scientific notation이 조용히 0 또는 다른 값으로
바뀌지 않습니다. 빈 문자열, malformed 숫자, `nan`, `+inf`, `-inf` 같은
non-finite 입력은 이전의 유효값으로 몰래 대체되지 않습니다. 잘못된 원문이
화면에 남아 build/validate/save를 차단하고, 사용자가 유효한 값으로 고치면
같은 편집 세션에서 다시 검증할 수 있습니다.

## Transactional load와 rollback

Load는 다음 단계를 모두 통과하기 전까지 live widget을 commit하지 않습니다.

1. JSON deserialize와 canonical `ModelSpec` 구성
2. semantic 및 shape validation
3. 모든 값이 현재 폼에서 손실 없이 표현 가능한지 staging
4. widget population

어느 단계에서든 실패하면 이름, input shape, 레이어 데이터와 순서뿐 아니라
선택된 top-level 레이어 및 지원되는 nested `BranchEditor` 선택까지 이전
상태로 rollback합니다. 목록 widget의 signal-block 상태도 호출 전 값으로
복원됩니다. 선택 상태는 editor UI 상태이며 canonical JSON에는 직렬화되지
않습니다. 예상하지 못한 rollback 자체의 실패를 성공으로 숨기지는 않습니다.

## Atomic save와 Training handoff

Save는 먼저 canonical model 구성과 validation을 끝낸 후 기존 atomic
`save_model_spec()`을 사용합니다. 목적지와 같은 디렉터리의 임시 파일을
완전히 기록하고 원자적으로 교체하므로 validation, serialization 또는 replace
실패가 기존 목적지의 부분 파일을 정상 결과처럼 남기지 않습니다.

`Save for Training...`은 이 저장이 성공한 뒤에만
`model_saved_for_training`을 방출합니다. 신호에는 방금 저장한 파일의
`Path.resolve()` 기반 절대경로가 담깁니다. MainWindow는 단일 연결로
TrainingPage의 기존 경로 입력을 갱신하고 Training 탭으로 이동합니다.
대화상자 취소, validation 실패, serialization/save 실패에는 신호가 없고,
기존 Training 경로와 현재 탭도 바뀌지 않습니다. 반복 사용은 성공 한 번당
한 번의 handoff만 만들며 탭 재진입이 연결을 추가하지 않습니다.

## CPU graduation evidence

`tests/gui/test_model_designer_integration.py`는 `qtbot`으로 실제 MainWindow,
ModelDesigner, TrainingPage를 연결합니다. 테스트가 GUI control로 작은
Conv2d → ReLU → Flatten → Linear 모델을 편집하고 다음 경계를 관통합니다.

```text
GUI fields and ordered layer form
  -> canonical validation and shape trace
  -> atomic model JSON save
  -> normalized TrainingPage handoff
  -> TrainingPage._build_request()
  -> build_training_request()
  -> existing TrainingController / QtTrainingWorker
  -> local CPU ImageFolder workflow
  -> canonical portable artifacts
```

테스트는 `tmp_path` 아래에 두 클래스와 train/val/test split을 가진 작은 PIL
이미지 fixture를 직접 만들고 1 epoch만 실행합니다. 저장된 정의를 기존
`load_model_spec()`과 `build_model()`로 읽어 CPU forward shape도 확인하며,
학습 완료와 QThread cleanup, `model_definition.json`, best state dict,
history, class mapping, test result를 프로그램 방식으로 검사합니다. 잘못된
입력의 no-handoff와 수정 후 성공, manual override, 반복 handoff도 함께
검증합니다. 네트워크, 외부 다운로드, CUDA, screenshot, 임의 sleep, busy
loop 또는 `QThread.terminate()`를 사용하지 않습니다.

## 수동 확인

자동 테스트 외에 다음 흐름을 수동으로 확인할 수 있습니다.

1. GUI를 열어 세 번째 `Model Designer` 탭이 한 번만 보이는지 확인합니다.
2. starter model을 Validate하고 shape trace가 레이어 순서대로 표시되는지
   확인합니다.
3. 잘못된 integer/float를 입력해 저장이 차단되고 원문을 고칠 수 있는지
   확인합니다.
4. `Save for Training...`을 취소했을 때 탭과 기존 Training 경로가 유지되는지
   확인합니다.
5. 성공한 저장 뒤 Training 탭으로 이동하고 절대 JSON 경로가 채워지는지
   확인합니다.
6. Browse나 직접 입력으로 다른 canonical JSON을 선택한 뒤 요청에 그 경로가
   사용되는지 확인합니다.

## 한계와 명시적 비범위

- 현재 UI는 ordered/form-based editor이며 graph canvas, drag-and-drop topology,
  arbitrary DAG, nested branch 또는 model execution preview가 아닙니다.
- 편집기 validation은 tensor shape와 canonical 파라미터 계약을 검사하지만
  정확도, 수렴 또는 특정 데이터셋에 대한 모델 품질을 보장하지 않습니다.
- CPU graduation은 작은 로컬 fixture와 1 epoch의 연결 검증입니다. 실제
  장시간 training, 대규모 dataset 성능, CUDA correctness/performance를
  대표하지 않습니다.
- Hyperparameter search, cancellation 기능 추가, 새로운 training/inference
  core API, dependency 변경은 Phase 14의 범위가 아닙니다.
- C++ GUI, packaging, installer, deployment migration과 Phase 15 기능도
  포함하지 않습니다.
- 외부에서 작성한 canonical Model JSON은 계속 Browse/직접 입력으로 사용할 수
  있지만, 현재 폼이 지원하지 않는 미래 schema 확장은 staging 단계에서 load가
  거부될 수 있습니다.
