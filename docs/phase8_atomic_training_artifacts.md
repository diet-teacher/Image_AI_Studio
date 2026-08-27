# Phase 8 Atomic Training Artifacts

Phase 7(`docs/phase7_portable_artifact_bundle.md` §2/§6)은 학습 output
directory에 나란히 놓이는 세 개의 휴대 가능한 산출물

```text
output_dir/
├─ model_definition.json      학습에 실제로 쓰인, 검증 완료된 ModelSpec
├─ class_mapping.json         class 이름/인덱스 매핑
└─ best_model_state_dict.pt   best epoch의 model state_dict
```

을 **plain overwrite**(`Path.write_text()` / `torch.save()` 직접 호출)로
저장한다고 명시했고, "원자적 저장 없음 -- checkpoint/metadata가 갖는
임시 파일 + `os.replace()` 보호가 없다"를 residual risk로 남겼다.

Phase 8은 그 gap을 좁힌다:

* **checkpoint 1** -- `src/image_ai_studio/training/artifact_io.py`에 내부
  원자적 I/O primitive 두 개(`atomic_write_text` / `atomic_torch_save`)를
  도입했다(다른 코드는 건드리지 않음).
* **checkpoint 2** -- `save_model_spec()`(`model_definition/serialization.py`)
  / `save_class_mapping()`(`training/torchvision_dataset.py`) /
  `save_state_dict()`(`training/checkpoint.py`)의 **게시 단계만** 그
  primitive를 거치도록 바꿨다. 직렬화 표현(결정론적 `json.dumps(...,
  indent=2)` UTF-8 텍스트 / `torch.save()` payload), 고정 파일명, 덮어쓰기
  정책, `load_*` 호환성은 전부 그대로다.
* **checkpoint 3**(이 문서) -- 커밋된 real CPU ImageFolder 워크플로우가
  실제로 위 세 산출물을 각자 canonical 경로에 **개별적으로 원자적으로
  게시**한다는 것을 focused 통합/fault-injection 테스트로 확인하고,
  그 보장의 정확한 경계 -- 그리고 그것이 **세 파일에 걸친 트랜잭션이
  아니라는 점** -- 을 문서화한다.

이 문서 자신은 검증을 수행하지 않는다 -- Phase 5D/6D/7과 동일한 원칙으로,
아래 §6의 자동화 테스트와 Phase 게이트(§8)가 실제 검증 주체다. "PHASE 8
COMPLETE" 판정은 이 문서를 작성했다는 사실이 아니라, 커밋된 구현이 그
게이트를 통과한다는 조건에 대한 서술이다.

이 checkpoint는 **새 packaging/archive/manifest/signature/migration 포맷을
도입하지 않으며**, 취소/batch 추론/이미지 preview/CUDA 실행/복구 보장도
추가하지 않는다(§7).

## 1. 소유권과 canonical 경로 (Phase 7에서 변경 없음)

`src/image_ai_studio/training/imagefolder_workflow.py`의
`run_imagefolder_training_workflow()`가 `request.output_dir` 아래에 고정
파일명으로 쓰는 산출물과, 정상 실행에서의 저장 순서:

```text
1. model_definition.json      save_model_spec()      -- 원자적 (CP2)
2. training_history.json       save_training_history() -- plain write_text (변경 없음)
3. class_mapping.json          save_class_mapping()   -- 원자적 (CP2)
4. (checkpoint.pt / .meta.json) save_training_checkpoint() -- 원자적 (Phase 4J, 별개 경로)
5. best_model_state_dict.pt    save_state_dict()      -- 원자적 (CP2)
6. test_result.json            Path.write_text()      -- plain write_text (변경 없음)
7. (model.ts / model_metadata.json) TorchScriptExporter -- 이 checkpoint 범위 밖
```

세 산출물(`model_definition.json` / `class_mapping.json` /
`best_model_state_dict.pt`) 모두 이 워크플로우 함수 하나가 소유한다 --
다른 코드 경로가 같은 이름으로 별도로 쓰지 않는다. `model_definition.json`
은 원본 `--model-json` 파일의 바이트 사본이 아니라, `load_model_spec()`
+ `validate_model_spec()`까지 통과한 바로 그 `ModelSpec`을
`save_model_spec()`으로 다시 직렬화한 것이다(Phase 7 §1과 동일).

`training_history.json` / `test_result.json` / `checkpoint.pt`는 이 세
산출물에 포함되지 않는다. `training_history.json` / `test_result.json`은
CP2에서 건드리지 않았으므로 여전히 plain `write_text`다.
`checkpoint.pt` / `.meta.json`은 Phase 4J부터 자체 원자적 저장
(`checkpoint.py`의 `_atomic_torch_save` / `imagefolder_resume.py`의
`_atomic_write_text`)을 이미 갖고 있고, `artifact_io.py`는 그 검증된
패턴을 한 곳에 모은 것이다 -- 두 구현은 동일한 "같은 디렉터리 임시 파일
-> flush/fsync -> `os.replace()`" 계약을 따른다.

## 2. per-file 원자적 게시 방식

`artifact_io._publish_atomically()`(두 primitive 공통 구현)의 계약:

```text
- 목적지 디렉터리(Path(path).parent) 안에 tempfile.mkstemp()로 유일한
  이름의 임시 파일(.{name}.XXXX.tmp)을 만든다 -- 다른 디렉터리로
  폴백하지 않는다.
- 임시 파일에 직렬화한 뒤 flush() + os.fsync()로 완성된 바이트의 디스크
  반영을 요청하고, os.replace()로 목적지에 게시한다
  (POSIX/Windows 양쪽에서 원자적 교체).
- 게시에 성공하면 목적지는 표준 Path.read_text(encoding=...) /
  torch.load() 경로로 다시 읽을 수 있고, helper 소유의 임시 파일은
  남지 않는다.
```

한 번 호출 = 정확히 한 파일의 원자적 게시. 이 primitive는 미리 존재하던
임의의 파일을 지우거나 재사용하지 않으며, 여러 산출물에 걸친 다중 파일
트랜잭션(all-or-nothing)을 제공하지 않는다.

## 3. 덮어쓰기 정책 / 임시 파일

* **덮어쓰기 정책은 Phase 7에서 바뀌지 않았다.** 같은 `output_dir`을
  재사용하는 다음 학습 실행은 세 산출물을 확인 없이 최신 유효 버전으로
  교체한다(다른 저장 함수와 동일한 기존 정책). 게시가 원자적이므로,
  덮어쓰기 순간에 목적지 경로에는 "옛 완전한 파일" 또는 "새 완전한
  파일"만 존재하고 반쯤 쓰인 중간 상태는 노출되지 않는다.
* 반복 성공 실행은 매번 그 실행의 최신 유효 산출물을 게시하며, 성공한
  실행이 끝난 뒤 `output_dir`에는 helper 소유의 임시 파일
  (`.model_definition.json.*.tmp` 등)이 남지 않는다 -- 게시 성공 시
  임시 파일은 `os.replace()`로 목적지가 되어 사라지고, 게시 실패 시에도
  best-effort로 정리된다(§4).

## 4. 실패 시 동작과 경계

### 4-1. per-file 보존

세 산출물 중 하나의 게시(`atomic_write_text` / `atomic_torch_save`)가
게시 이전 어느 단계에서든 실패하면(직렬화 / `flush` / `os.fsync` /
`os.replace`):

* 그 **특정** 목적지 파일에 이미 존재하던 바이트는 **바이트 단위로 그대로
  보존된다** -- 임시 파일에만 쓰다 실패했으므로 목적지는 건드려지지
  않는다.
* helper가 만든 임시 파일만 best-effort로 지운다. 그 정리(unlink)마저
  실패해도 사용자에게 보이는 예외는 **원래 실패 예외**다(정리 실패가
  원래 예외를 가리지 않는다).
* 원래 예외(`OSError` / `RuntimeError` / `UnicodeEncodeError` 등)는
  재시도나 폴백 없이 그대로 호출자에게 전파되고, 워크플로우는 실패로
  끝난다.

### 4-2. 다중 파일 트랜잭션이 아니다

이 보장은 **각 파일 단위**다. 세 산출물을 하나로 묶는 트랜잭션은 없다:

* §1의 저장 순서에서 **실패 지점보다 앞서 게시된 형제 산출물은 이미 새
  바이트로 게시된 채 그대로 남는다** -- 롤백되지 않는다. 예를 들어
  `best_model_state_dict.pt` 게시가 실패하면, 그 실행에서 이미 게시된
  `model_definition.json` / `class_mapping.json`은 **새 실행의 내용**을
  담은 채 남고(옛 실행 버전으로 되돌아가지 않음), `best_model_state_dict.pt`
  자신만 이전 실행의 바이트를 유지한다.
* 실패 지점보다 **뒤의** 저장 단계(§1의 이후 항목)는 fail-fast로 아예
  실행되지 않는다.
* 따라서 게시 실패 후의 `output_dir`은 "서로 다른 실행에서 온, 각자
  개별적으로는 온전한 파일"이 섞인 상태일 수 있다. 이 조합을 감지하는
  manifest/checksum/서명은 없다(§7).

## 5. 추론 auto-discovery / legacy Model JSON override (변경 없음)

Phase 7 CP2가 만든 `InferencePage._build_request()`의 동작 --

```text
Model JSON 입력란이 비어 있으면:
    -> training_output_dir/model_definition.json 을 그대로 derive
Model JSON 입력란에 값이 있으면:
    -> 그 값이 항상 우선 (auto-discovery 경로를 거치지 않음)
```

-- 은 이 checkpoint에서 **전혀 바뀌지 않았다**. `model_definition.json`
게시가 plain write에서 원자적 write로 바뀐 것은 추론이 그 파일을 읽는
방식과 무관하다(원자적 게시는 오히려 "반쯤 쓰인 `model_definition.json`을
추론이 읽을" 창을 없앤다). Phase 7 이전에 생성된 legacy output directory
(즉 `model_definition.json`이 없는 `output_dir`)에 대한 명시적 Model JSON
override도 그대로 유효하다.

이 두 경로는 고정 통합 allowlist로 커버된다:

```text
tests/gui/test_training_inference_integration.py
    - test_phase7_cp3_output_dir_alone_drives_inference_without_model_json
        실제 CPU backend: 학습이 만든 output_dir + 입력 이미지만으로
        (Model JSON 완전히 비움) 추론이 끝까지 성공 -- auto-discovery
    - test_phase7_cp3_explicit_model_json_still_works_for_legacy_output_dir
        실제 CPU backend: model_definition.json을 수동 제거해 legacy
        output_dir을 재현한 뒤, 명시적 Model JSON override로 추론 성공
    - (Phase 8 CP3) test_phase8_cp3_atomic_publication_leaves_clean_output_dir
        실제 CPU backend: 학습 직후 output_dir에 helper 임시 파일이
        없고 세 canonical 산출물이 로드 가능하며, 그 산출물만으로
        auto-discovery 추론이 성공
```

## 6. 검증 범위 -- 자동화 CPU 커버리지 / CUDA 조건부 구분

이 checkpoint가 참조하는, 저장소에 실제로 존재하는 테스트만 나열한다 --
특정 실행의 pass 개수/시간/실측치/비용을 이 문서 자체가 새로 주장하지
않는다.

### 6-1. 영구 자동화 테스트 (CPU에서 항상 실행, CUDA 가용성과 무관)

```text
tests/training/test_artifact_io.py
    (Phase 8 CP1, pytest.mark.phase8_cp1_atomic_artifact_io_primitives)
    - primitive 자체의 성공 round-trip / 기존 목적지 보존
      (os.replace 실패 / serializer 실패) / 임시 파일 정리 /
      정리 실패가 원래 예외를 가리지 않음 / 목적지 디렉터리에만 임시
      파일 생성 / shell·network-free

tests/training/test_imagefolder_workflow.py
    (Phase 8 CP3, pytest.mark.phase8_cp3_atomic_bundle_workflow_graduation)
    - 성공 실행이 model_definition.json / class_mapping.json /
      best_model_state_dict.pt 세 개를 canonical 경로에 게시하고, 셋 다
      artifact_io._publish_atomically()(= CP2 writer)를 거치며, load_*
      경로로 다시 읽힌다
    - 같은 output_dir 반복 성공 실행이 최신 유효 산출물을 게시하고
      helper 소유 임시 파일(.tmp)을 남기지 않는다
    - model_definition.json 게시(os.replace)가 실패하면 기존 파일이
      바이트 단위로 보존되고 원래 예외가 전파되며 임시 파일이 남지 않는다
    - best_model_state_dict.pt 게시(torch.save 직렬화)가 실패하면 그
      파일만 이전 바이트를 유지하고, 그 실행에서 먼저 게시된
      model_definition.json은 새 내용을 담은 채 남는다(트랜잭션 롤백을
      주장하지 않는다)

tests/gui/test_training_inference_integration.py
    (§5의 고정 allowlist -- 실제 CPU backend)
```

이 목록은 CPU만으로 완결된다 -- 모든 모듈이 fake 또는 실제 CPU backend를
사용하며, CUDA 설치 여부와 무관하게 항상 실행/검증된다. Phase 8이 이
checkpoint에서 새로 추가한 CUDA 전용 테스트는 없다.

### 6-2. CUDA 조건부 (conditional) 범위

`model_definition.json` / `class_mapping.json` / `best_model_state_dict.pt`
게시는 device와 무관한 순수 파일 I/O이며, `save_state_dict()`는
`best_model`(항상 CPU에서 `build_model()`로 새로 만들어진 인스턴스, Phase
4Q)을 저장한다. 이 checkpoint의 원자적 게시 계약은 CUDA 실행을 요구하거나
가정하지 않으며, 이 문서는 Phase 8 원자적 게시가 실제 CUDA 환경에서
end-to-end로 실행됐다고 주장하지 않는다. CUDA training/resume/추론 forward
자체의 조건부 커버리지는 Phase 4R~4V / 6이 이미 확립한 기존
`@pytest.mark.skipif(not torch.cuda.is_available())` 테스트가 계속
담당하며, 이 문서는 그것을 재확인하거나 새로 생성하지 않는다.

### 6-3. 수동 실행 (자동화 테스트 아님)

```bash
python scripts/train_imagefolder.py --model-json ... --dataset-root ... --output-dir ...
python scripts/run_gui.py
```

production CLI로 실제 학습을 돌려 `output_dir`을 눈으로 확인하는 경로는
자동화 스위트에 포함되지 않는다 -- §6-1의 테스트가 회귀 안전망이다.

## 7. Residual risks / non-goals

```text
다중 파일 트랜잭션 없음: §4-2. 세 산출물은 각자 원자적으로 게시될 뿐,
    "셋 다 성공 아니면 셋 다 옛 버전"을 보장하는 트랜잭션은 없다.
    한 게시가 실패하면 먼저 게시된 형제는 새 버전으로, 실패한 파일은
    옛 버전으로 남는 혼합 상태가 가능하다.

integrity/signature manifest 없음(Phase 7 §6에서 계속): 세 파일이 같은
    학습 실행에서 나왔는지 검증하는 체크섬/서명/manifest가 없다 --
    §4-2의 혼합 상태나, 사용자가 수동으로 다른 실행의 파일을 섞어 놓은
    경우를 감지하는 메커니즘은 없다.

training_history.json / test_result.json은 원자적이 아니다: CP2 범위
    밖이라 여전히 plain write_text다 -- 저장 도중 프로세스가 중단되면
    반쯤 쓰인 파일이 남을 수 있다.

fsync 내구성은 best-effort: flush()/os.fsync()는 완성된 바이트의 디스크
    반영을 "요청"할 뿐, 특정 OS/파일시스템/하드웨어에서의 crash-durability
    를 이 문서가 보장하지는 않는다. os.replace()의 원자성(교체가 일어나면
    완전한 파일)은 POSIX/Windows에서 성립한다.

packaging/archive/installer 없음, batch 추론 없음, 이미지 preview 없음,
    추론 cancellation 없음, legacy output 자동 migrate/backfill 없음,
    학습 취소/복구(crash recovery) 보장 없음: 전부 Phase 6/7의 기존
    non-goal 그대로이며 Phase 8 CP3에서 바뀌지 않았다.

공개 API / goal / manifest / config / launcher / packaging / 이후 Phase
    기능은 이 문서-and-통합 checkpoint에서 변경되지 않는다.
```

## 8. Phase 8 graduation criteria

```text
[ ] 이 checkpoint가 참조하는 고정 checkpoint test 목록(§6-1)이 실제
    커밋된 구현에서 PASS
[ ] 전체 프로젝트 harness(pytest 전체 스위트 + 필요한 회귀 스크립트)를
    커밋된 구현에서 정확히 한 번 최종 실행해 PASS
```

**PHASE 8 COMPLETE**는 위 두 조건이 모두 충족될 때에만 성립하는 조건부
졸업 판정이다. 이 문서를 작성하는 행위 자체는 그 실행을 수행하지 않으며,
이 문서는 "검증이 이 문서 작성으로 완료되었다"고 주장하지 않는다 --
실제 판정은 그 테스트/harness 실행 결과에 달려 있다.
