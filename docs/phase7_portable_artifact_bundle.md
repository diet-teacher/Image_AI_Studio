# Phase 7 Portable Training-Output Bundle

Phase 6은 `InferencePage`가 학습 output directory에서 `best_model_state_dict.pt`
/`class_mapping.json` 두 고정 파일명을 자동으로 유도하되, 그 학습에 실제로
쓰인 model definition JSON은 `output_dir`에 남지 않아 사용자가 별도로
다시 지정해야 한다는 비대칭을 `docs/phase6_final_integration.md` §4/§9에
명시적인 non-goal로 남겼다("output_dir에 model JSON을 자동 저장하도록
training core를 확장하지 않음"). Phase 7은 정확히 이 비대칭을 메운다 --
학습에 실제로 쓰인, 검증까지 끝난 `ModelSpec`을 `output_dir` 안에 세
번째 고정 파일명으로 함께 저장해서, **output directory 하나만으로**
(원본 `--model-json` 파일의 위치/파일명/포맷팅과 무관하게) 추론까지 이어질
수 있게 한다.

이 문서 자신은 검증을 수행하지 않는다 -- Phase 5D/6D와 동일한 원칙으로,
아래 §5의 자동화 테스트와 Phase 게이트(고정 checkpoint test 목록 + 전체
project harness)가 실제 검증 주체다. §7의 "PHASE 7 COMPLETE" 판정은 이
문서를 작성했다는 사실이 아니라, 커밋된 상태가 그 게이트를 통과한다는
조건에 대한 서술이다.

이 산출물 세 개를 묶는 **새 manifest 포맷이나 archive 포맷은 도입하지
않는다.** 여전히 각자 독립적인 파일 세 개(`.json`, `.json`, `.pt`)가 같은
디렉터리에 나란히 있을 뿐이며, 그것들을 하나로 감싸는 새 컨테이너/색인
파일은 없다.

## 1. 세 아티팩트의 소유권과 canonical 경로

`src/image_ai_studio/training/imagefolder_workflow.py`의
`run_imagefolder_training_workflow()`가 `request.output_dir` 아래에 항상
쓰는 고정 파일명 세 개:

```text
output_dir/
├─ model_definition.json      학습에 실제로 쓰인, load_model_spec() +
│                              validate_model_spec()까지 통과한 ModelSpec
│                              (신규, Phase 7)
├─ best_model_state_dict.pt   best epoch의 model state_dict (Phase 4A/4B부터)
└─ class_mapping.json         class 이름/인덱스 매핑 (Phase 4D부터)
```

세 파일 모두 이 워크플로우 함수 하나가 소유한다 -- 다른 코드 경로가 같은
이름으로 별도로 쓰지 않는다. `model_definition.json`은 사용자가 지정한
원본 `--model-json`/`model_json_path` 파일의 사본이 아니다 --
`load_model_spec()`으로 읽고 `validate_model_spec()`으로 검증까지 마친
바로 그 `ModelSpec` 객체를 `save_model_spec()`으로 **다시 직렬화**한 것이다
(`imagefolder_workflow.py`의 "Phase 7 checkpoint 1" 주석 참고). 그래서:

* 원본 파일이 압축 JSON(들여쓰기 없음)이든 다른 파일명이든, `output_dir`의
  결과물은 항상 `save_model_spec()`의 표준 포맷(들여쓰기 2칸)이다.
* 원본 파일이 나중에 이동/삭제돼도 `output_dir`의 사본은 영향받지 않는다.
* 비교 계약은 바이트 단위가 아니라 `load_model_spec()`으로 다시 읽은
  `ModelSpec` 객체 동등성이다(`tests/training/test_imagefolder_workflow.py`의
  `test_model_definition_json_round_trips_regardless_of_source_formatting_and_filename`).

## 2. 저장 방식과 overwrite 정책

세 파일 모두 **plain overwrite**다 -- `Path.write_text()`(JSON 두 개) 또는
`torch.save()`(state_dict)를 직접 호출할 뿐, Phase 4J가 `checkpoint.py`/
`imagefolder_resume.py`에 도입한 임시 파일 + `os.replace()` 원자적 저장
패턴을 쓰지 않는다. 즉:

* 같은 `output_dir`을 재사용하는 두 번째 학습 실행은 기존 파일을 확인 없이
  덮어쓴다(이 프로젝트의 다른 산출물 저장 함수들과 동일한 기존 정책,
  Phase 4H가 이미 문서화한 "고정 파일명 덮어쓰기" 정책을 `model_definition.json`
  에도 그대로 적용한 것뿐이다). 두 번째 실행에서 쓰인 spec이 다르면
  `model_definition.json`은 그 새 spec으로 결정적으로 교체되고, 같은
  `output_dir`의 다른 고정 산출물(`best_model_state_dict.pt`/
  `class_mapping.json`)은 그대로 남는다
  (`test_model_definition_json_replaced_by_later_run_with_different_spec`).
* 저장 도중 프로세스가 중단되면(디스크 가득 참, 강제 종료 등) 반쯤 쓰인
  파일이 남을 수 있다 -- checkpoint/metadata 저장과 달리 이 세 파일에는
  원자적 저장의 보호가 없다(아래 §6 residual risk 참고).

## 3. 실패 시 동작

`model_definition.json` 저장(`save_model_spec()`)이 실패하면(예: `OSError`)
그 예외는 감싸이거나 무시되지 않고 그대로 호출자에게 전파된다 -- 다른
산출물 저장(checkpoint, history, class mapping 등)과 동일한 fail-fast 계약
(`test_model_definition_json_write_failure_propagates_and_is_not_swallowed`).
이 저장은 `training_history.json`/`class_mapping.json`/checkpoint보다
**먼저** 실행되므로, 저장이 실패하면 그 이후의 산출물 저장 단계는 아예
실행되지 않는다.

읽기 실패(추론 시 `model_definition.json`이 없거나 손상된 경우)는 이
워크플로우가 아니라 추론 경로(§4)의 책임이다.

## 4. 새 output 워크플로우와 legacy output에 대한 명시적 Model JSON override

`InferencePage._build_request()`(`src/image_ai_studio/gui/inference_page.py`)
는 Model JSON 입력란을 **선택 필드**로 바꿨다:

```text
Model JSON 입력란이 비어 있으면:
    -> training_output_dir/model_definition.json 을 그대로 derive한다
       (best_model_state_dict.pt/class_mapping.json과 동일하게, 단순
       경로 결합 -- 파일 존재 여부를 미리 확인하지 않는다)
Model JSON 입력란에 값이 있으면:
    -> 그 값이 항상 우선한다(auto-discovery 경로를 거치지 않음)
```

이 override는 **Phase 7 이전에 생성된 legacy output directory**(즉
`model_definition.json`이 없는 output_dir)를 계속 쓸 수 있게 하기 위해
존재한다 -- 사용자가 그 학습에 실제로 쓰인 model definition JSON을 직접
가리키면, `output_dir`에 canonical 파일이 없어도 추론이 정상 동작한다
(`tests/gui/test_training_inference_integration.py`의
`test_phase7_cp3_explicit_model_json_still_works_for_legacy_output_dir`).

Model JSON을 비워 두고 canonical 파일이 실제로 없으면(legacy output_dir을
override 없이 쓴 경우 등), `InferencePage`는 이를 미리 검증하지 않는다 --
Phase 6A/6C가 확립한 "GUI는 재검증하지 않는다" 원칙 그대로, 존재하지 않는
경로로 `InferenceRequest`가 조립되고 그 실패는 기존 worker/controller
`failed` 경로(상태 표시 + controls 복원 + thread/worker cleanup)를 통해서만
드러난다.

**canonical 새 output 워크플로우**(Phase 7 이후 생성된 output_dir):

```text
1. scripts/train_imagefolder.py로 학습 -- output_dir에 model_definition.json
   /best_model_state_dict.pt/class_mapping.json 세 파일이 모두 생성됨
2. InferencePage에서 Training Output Dir만 선택
3. Model JSON은 비워 둔다 -- output_dir/model_definition.json이 자동으로 쓰임
4. Input Image 선택, Device/Precision 선택, Run Inference
```

이 워크플로우는 **새로운 manifest나 archive 포맷을 도입하지 않는다** -- 세
파일은 여전히 같은 디렉터리에 나란히 존재하는 독립 파일일 뿐이며, 이
문서가 "bundle"이라고 부르는 것은 그 세 파일의 관례적 묶음을 가리키는
서술일 뿐 새 파일 포맷의 이름이 아니다.

## 5. 검증 범위 -- 자동화 CPU 커버리지 / CUDA 조건부 구분

이 checkpoint가 참조하는, 저장소에 실제로 존재하는 테스트만 나열한다 --
특정 실행의 pass 개수/시간/실측치를 이 문서 자체가 새로 주장하지 않는다.

### 5-1. 영구 자동화 테스트(CPU에서 항상 실행, CUDA 가용성과 무관)

```text
tests/training/test_imagefolder_workflow.py
    (Phase 7 checkpoint 1 절)
    - model_definition.json이 canonical 경로에 저장됨
    - 원본 파일의 포맷팅/파일명과 무관하게 ModelSpec 객체 동등성으로 round-trip
    - 같은 output_dir 재사용 시 최신 실행의 spec으로 결정적으로 교체
    - 저장 실패(OSError) 시 예외가 감싸이지 않고 그대로 전파

tests/gui/test_inference_page.py
    (Phase 7 CP2 절, pytest.mark.phase7_cp2_inference_bundle_discovery)
    - Model JSON 필드가 비어 있으면 training_output_dir/model_definition.json으로 derive
    - 고정 파일명/부모 디렉터리 조합이 정확함
    - 명시적 Model JSON 값이 auto-discovery보다 항상 우선함
    - placeholder 텍스트가 auto-discovery를 안내함
    - 초기 상태에서 필드 자체는 여전히 빈 문자열(placeholder는 값이 아님)
    - canonical 파일이 없을 때 기존 failed 경로로만 실패가 드러남
    - auto-discovery 모드에서도 CP2/CP3/CP4의 기존 lifecycle 계약
      (Running/Finished/Failed, controls 복원, stale 결과 초기화, rerun)이
      그대로 유지됨

tests/gui/test_training_inference_integration.py
    (Phase 7 CP3 절)
    - 실제 CPU backend로: 학습이 만든 output_dir + 입력 이미지만으로
      (Model JSON 완전히 비움) 추론이 끝까지 성공
    - 실제 CPU backend로: model_definition.json을 수동으로 제거해
      legacy output_dir을 재현한 뒤, 명시적 Model JSON override로
      추론이 여전히 성공
```

이 목록은 CPU만으로 완결된다 -- 세 테스트 모듈 모두 fake 또는 실제 CPU
backend를 사용하며, CUDA 설치 여부와 무관하게 항상 실행/검증된다.

### 5-2. CUDA 조건부(conditional) 커버리지

Phase 7이 이 checkpoint에서 새로 추가한 CUDA 전용 테스트는 없다 --
`model_definition.json` 저장/로드 자체는 device와 무관한 순수 파일 I/O이고,
추론 경로의 CUDA fp16/bf16 forward 검증은 Phase 6B/6D가 이미 확립한
기존 조건부 테스트(`docs/phase6_final_integration.md` §8-2)가 계속
담당한다 -- 이 문서는 그 커버리지를 재확인하거나 새로 생성하지 않는다.
Phase 7의 auto-discovery 경로가 실제 CUDA 환경에서 end-to-end로
실행됐다는 것을 이 문서는 주장하지 않는다.

### 5-3. 수동 실행(자동화 테스트 아님)

```bash
python scripts/run_gui.py
```

이 launcher로 Training tab에서 학습 후 Inference tab에서 Model JSON을
비운 채 실행해 보는 경로는 자동화된 pytest 스위트에 포함되지 않는다 --
§5-1의 `test_training_inference_integration.py`가 이 시나리오의 실제
회귀 안전망이다.

## 6. Residual risks / non-goals

```text
legacy output(model_definition.json이 없는 output_dir): 자동으로
    migrate/backfill하지 않는다 -- 사용자가 명시적 Model JSON override로
    직접 지정해야 하며(§4), 이 프로젝트는 이런 legacy output_dir을 감지해
    경고하거나 canonical 파일을 사후 생성하는 기능을 제공하지 않는다.

integrity/signature manifest 없음: 세 파일이 서로 짝이 맞는지(같은 학습
    실행에서 나왔는지) 검증하는 체크섬/서명/manifest가 없다 -- output_dir
    안의 파일들을 사용자가 수동으로 섞어 놓아도(예: 다른 실행의
    class_mapping.json을 덮어쓰기) 이를 감지하는 메커니즘은 없다.

packaging/installer 없음: 이 bundle을 단일 배포 아카이브(zip 등)로 묶어
    다른 머신에 설치하는 기능은 없다 -- 여전히 디렉터리를 통째로 복사하는
    것이 유일한 이동 방법이다.

batch inference 없음: InferencePage/InferenceController는 여전히 단일
    이미지만 지원한다(Phase 6A §7의 기존 범위 판단, Phase 7은 이를
    바꾸지 않았다).

이미지 preview 없음: 선택한 이미지나 결과를 시각적으로 렌더링하는 기능은
    여전히 없다(Phase 6A §9의 기존 backlog, Phase 7은 이를 바꾸지 않았다).

inference cancellation 없음: 단일 이미지 forward pass는 여전히 원자적
    작업으로 취급되며 취소 지점이 없다(Phase 6A §7/§8의 기존 설계 판단,
    Phase 7은 이를 바꾸지 않았다).

원자적 저장 없음: §2에서 서술한 대로 세 파일 모두 plain overwrite다 --
    checkpoint/metadata가 갖는 임시 파일 + os.replace() 보호가 없다.
```

Phase 4~6부터 유효한 다른 non-goal(그래프/차트, 실험 이력 DB, multi-run,
custom theme 등)도 Phase 7에서 계속 유효하다.

## 7. Phase 7 graduation criteria

```text
[ ] 이 checkpoint가 참조하는 고정 checkpoint test 목록(§5-1)이 실제
    커밋된 상태에서 PASS
[ ] 전체 프로젝트 harness(pytest 전체 스위트 + 필요한 회귀 스크립트)가
    실제 커밋된 상태에서 PASS
```

**PHASE 7 COMPLETE**는 위 두 조건(고정 checkpoint test 목록과 전체 project
harness가 커밋된 상태에서 실제로 PASS하는 것)이 모두 충족될 때에만
성립하는 조건부 졸업 판정이다. 이 문서를 작성하는 행위 자체는 그 실행을
수행하지 않으며, 이 문서는 "검증이 이 문서 작성으로 완료되었다"고
주장하지 않는다 -- 실제 판정은 그 테스트/harness 실행 결과에 달려 있다.
