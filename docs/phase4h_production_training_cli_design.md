# Phase 4H: Production ImageFolder Training CLI Separation — 설계안

**상태: 구현 및 검증 완료.** 실제 사용자용 ImageFolder 학습 CLI와 회귀
검증용 E2E 스크립트의 책임을 분리하는 상세 설계와, 그 설계를 그대로
구현한 뒤 검증한 결과를 담는다.

**개정 (리뷰 반영, 구현 완료 후)**: (1) `run_imagefolder_training_e2e.py`의
"best model save/reload" 재검증 블록을 완전히 제거함 — 같은
`best_model_state_dict_path` 파일을 두 model에 각각 로드해 출력을
비교하는 방식은 항상 같은 결과만 내는(같은 파일을 같은 방법으로 두 번
읽을 뿐인) 무의미한 검증이었고, "저장 전 원본 model vs 재로드 model"
비교와 동일하지 않았다. `Result`에 살아있는 model을 추가하지 않는
설계(§6)를 유지한 채, 이 검증 자체를 삭제하는 쪽으로 정리했다 —
state_dict 저장/재로드의 수치적 정확성은 이미
`tests/training/test_imagefolder_workflow.py`(best model 평가,
TorchScript export 비교)와 `tests/training/test_checkpoint.py`가 단위
테스트로 충분히 커버한다(§10/§12/§14-4 갱신). (2) E2E regression anchor
수치는 자동 gate가 아니라 "출력 후 수동 재현 확인" 대상임을 명시함 —
환경/PyTorch 버전 차이에 민감한 소수점 값을 엄격한 자동 비교 대상으로
삼지 않는다(§14-4). (3) 삭제된 Phase 4G CLI 테스트 9개 중 `OSError`
전용 케이스(metadata는 있는데 checkpoint 파일만 없는 경우)가 새 CLI
테스트 이전 과정에서 누락되어 있었던 것을 발견해
`test_train_imagefolder_cli.py`에 추가함(§12/§14-2).

최종 검증 결과 (모두 이 저장소에서 직접 실행 확인):

* `tests/training/` + `tests/scripts/`: **182 passed**
* 전체 `pytest`: **339 passed**
* `run_imagefolder_training_e2e.py` 재구성 후 실제 CIFAR-10 ImageFolder
  fixture로 regression anchor 수치(§14-4)를 정확히 재현: PASS
* TorchScript export, C++ CPU/CUDA parity: PASS

---

## 1. 현재 구조 분석

### 1-1. `scripts/run_imagefolder_training_e2e.py`(566줄, Phase 4D/4E/4G에
걸쳐 확장)의 책임 분해

실제 코드(`main()`, `:201-561`)를 순서대로 추적한 결과, 이 스크립트는 현재
**11단계**를 한 함수 안에서 전부 수행한다:

| 순서 | 코드 위치 | 책임 | 분류(§1-2) |
|---|---|---|---|
| 1 | `:211-227` | `TrainingConfig` 생성 (`epochs`만 CLI, `batch_size`/`learning_rate`는 하드코딩 상수) | 학습 본질 |
| 2 | `:244-253` | `load_model_spec()` + `validate_model_spec()` | 학습 본질 |
| 3 | `:265-282` | resume checkpoint/metadata 로드 (Phase 4G) | 학습 본질 |
| 4 | `:284-302` | `make_imagefolder_datasets()` + class 수 검증 | 학습 본질 |
| 5 | `:308-317` | resume metadata 호환성 검증 (Phase 4G) | 학습 본질 |
| 6 | `:319-351` | model build/resume, DataLoader 구성(train/val/**test**) | 학습 본질(test_loader만 결과적으로 §7 평가에 필요, 학습 자체엔 불필요) |
| 7 | `:359-378` | `TrainingResumeState` 조립 + RNG 복원 (Phase 4G) | 학습 본질 |
| 8 | `:380-403` | `run_training()` 호출 + **`loss_improved` 게이트** (`:396-403`) | 학습 실행(본질) + **게이트(E2E 전용)** |
| 9 | `:408-438` | history 저장, checkpoint 저장(Phase 4G) | 학습 본질 |
| 10 | `:440-474` | best model 저장, **class mapping 저장+재로드 검증**(`:450-461`), test 평가 | 학습 본질 + **재로드 검증 게이트(E2E 전용)** |
| 11 | `:476-561` | **best model save/reload allclose 검증**(`:476-491`), TorchScript export, C++ CPU/CUDA parity, PASS/FAIL 판정 | **E2E/parity 전용** |

### 1-2. 사용자가 요청한 분류 (추정 없이 코드 근거로)

- **실제 학습에 본질적으로 필요한 로직**: ModelSpec 로드/검증, dataset
  로드/검증(`make_imagefolder_datasets`, `require_matching_num_classes`),
  `TrainingConfig` 구성, model build, DataLoader 구성, resume
  준비(checkpoint/metadata 로드+검증+RNG 복원), `run_training()` 호출,
  full checkpoint 저장, best model state_dict 저장, training history 저장,
  class mapping 저장, test 평가(`evaluate()`).
- **E2E 검증에만 필요한 로직**: `:396-403`의 `loss_improved` 게이트(`return
  1` 트리거), `:454-461`의 class mapping 재로드 후 `==` 비교 게이트,
  `:487-491`의 best model save/reload `torch.allclose()` 게이트, 마지막
  `overall_ok`/`PHASE 4D E2E`(현재 `IMAGEFOLDER TRAINING E2E`) PASS/FAIL
  판정 전체.
- **C++ runner/parity 전용 로직**: `:493-527`(reference tensor 저장,
  `save_tensor`), `:529-557`(`find_runner_binary`/`run_case`, CPU/CUDA
  루프, `subprocess.run`으로 `build_torchscript.py` 자동 빌드).
- **artifact 경로를 하드코딩하는 부분**: `:141-145`의 `ARTIFACTS_COMMON`/
  `ARTIFACTS_TRAINING`/`ARTIFACTS_TORCHSCRIPT`/`ARTIFACTS_REFERENCE`/
  `BUILD_DIR` — 전부 `REPO_ROOT` 기준 고정 경로이고 사용자가 CLI로 바꿀
  방법이 없다. 파일명도 `artifact_name = f"{model_spec.name}{ARTIFACT_SUFFIX}"`
  (`:255`, `:139`)로 고정 규칙을 따른다.
- **CIFAR-10 fixture 기본값에 의존하는 부분**: `MODEL_JSON`(`:132`,
  `examples/models/phase4c_cifar10_model.json`), `DEFAULT_DATASET_ROOT`
  (`:133`, `artifacts/datasets/cifar10_imagefolder`) — 둘 다
  `scripts/prepare_cifar10_imagefolder_fixture.py`가 만든 산출물을
  가리키는 값이며, 실제 사용자 dataset과는 무관한 회귀 검증용 기본값이다.
- **테스트 성공/실패 판정을 위한 gate**: 위 "E2E 검증에만 필요한 로직" 전체
  + `overall_ok = parity_ok`(`:559`) + 각 단계의 `return 1` 조기 종료.
- **CLI argument parsing**: `parse_args()`(`:154-198`) — `--model-json`,
  `--dataset-root`, `--optimizer`/`--momentum`/`--lr-scheduler`/
  `--lr-scheduler-factor`/`--lr-scheduler-patience`/
  `--early-stopping-patience`(Phase 4E), `--epochs`/`--resume-from`/
  `--checkpoint-out`(Phase 4G). **`--batch-size`/`--learning-rate`는
  여전히 노출되지 않는다**(`:215-216`에 `DEFAULT_BATCH_SIZE`/
  `DEFAULT_LEARNING_RATE` 하드코딩) — Phase 4G 설계 문서(§15-1)가 "이번
  Phase에서는 범위 밖"이라고 명시적으로 미룬 항목이 그대로 남아 있다.
- **사용자에게 노출하면 안 되는 내부 테스트 옵션**: 현재는 이런 옵션이
  CLI에 없다(즉 노출된 "내부 전용" 플래그는 없음) — 문제는 반대 방향이다:
  **C++ parity/E2E 전용 실행 자체가 옵트아웃 불가능하게 항상 실행된다.**
  사용자가 실제 학습만 하고 싶어도 `run_torchscript` 빌드가 없으면
  자동으로 `build_torchscript.py`를 서브프로세스로 실행하고(`:531-538`),
  `find_runner_binary`/`run_case`가 항상 호출된다(`:529-557`) — 이건
  "내부 전용 옵션이 노출됨"이 아니라 "내부 전용 동작이 끌 수 없게
  강제됨"이라는, 사실상 더 나쁜 형태의 같은 문제다.
- **공통 함수로 추출 가능한 로직**: `run_training_e2e.py`/
  `run_real_training_e2e.py`/`run_imagefolder_training_e2e.py` 세 스크립트를
  나란히 읽은 결과(§1-3), **best model 저장 → save/reload allclose 검증 →
  TorchScript export → CPU/CUDA reference 저장 → C++ runner 자동 빌드/실행 →
  parity 판정**(대략 90~100줄)이 세 스크립트에 **거의 글자 그대로**
  중복되어 있다. 이번 Phase는 ImageFolder 경로 분리만 요청받았으므로 세
  스크립트를 전부 통합하는 것은 범위 밖이지만, 이 사실 자체는 §4(아키텍처
  후보)의 근거로 쓴다.

### 1-3. 다른 파일에서 확인한 사실

- `src/image_ai_studio/training/imagefolder_resume.py`(Phase 4G): 이미
  `ImageFolderResumeMetadata`, `build_imagefolder_resume_metadata()`,
  `save_/load_imagefolder_resume_metadata()`,
  `require_compatible_imagefolder_resume_metadata()`,
  `metadata_path_for_checkpoint()`를 제공한다 — **워크플로우가 필요한
  metadata 관련 기능은 전부 이미 존재**하고, 신규로 만들 필요가 없다.
- `src/image_ai_studio/training/checkpoint.py`: `save_training_checkpoint()`/
  `load_training_checkpoint()`가 dataset-agnostic하게 이미 존재. 워크플로우
  모듈은 이걸 그대로 호출만 하면 된다.
- `src/image_ai_studio/training/config.py`: `TrainingConfig`가 이미
  optimizer/scheduler/early-stopping 전부 검증한다.
  `require_compatible_resume_config()`도 이미 존재.
- `src/image_ai_studio/model_definition/errors.py:5` — **`class
  ModelValidationError(ValueError)`**로 확인됨(재추정 아님, 직접 읽음).
  `training/config.py`의 `TrainingConfigError`도 이미 `ValueError`
  서브클래스(Phase 4F/4G 설계 문서에서 이미 확인됨). 즉 **이 프로젝트의
  모든 커스텀 검증 예외는 결국 `ValueError`다** — `OSError`(파일 I/O)만
  별도 계열이다. 이 사실이 §11(error handling) 설계의 핵심 근거다.
- `src/image_ai_studio/model_definition/specs.py:222-260` —
  `ModelSpec.__post_init__`은 `name`을 "비어있지 않은 문자열"로만
  검증한다. **파일시스템에 안전한 문자만 허용한다는 제약이 전혀 없다.**
  현재 E2E는 `artifact_name = f"{model_spec.name}{ARTIFACT_SUFFIX}"`를
  디렉터리명(`ARTIFACTS_TORCHSCRIPT / artifact_name / "model.pt"`,
  `:498`)에 직접 쓰므로, 사용자가 `name`에 `/`나 `..` 같은 문자를 넣으면
  경로가 깨질 수 있는 **실재하는 위험**이다(§9에서 이걸 구조적으로
  없애는 방법을 설계한다).
- `src/image_ai_studio/tools/run_and_compare.py` — `find_runner_binary()`
  (`:35-51`)는 `build_dir/cpp/<subdir>/run_torchscript(.exe)`를 찾고,
  `run_case()`(`:54-135`)는 `subprocess.run()`으로 네이티브 바이너리를
  호출하고 `artifacts/reference/*`의 저장된 참조 텐서와 비교한다. **순수
  Python 프로세스만으로는 절대 실행할 수 없는 외부 의존성**(빌드된 C++
  바이너리)이다 — production CLI 기본 동작에 넣으면 "빌드 안 한 사용자는
  학습도 못 한다"는 나쁜 사용성이 생긴다.
- `src/image_ai_studio/export/torchscript_exporter.py` — `TorchScriptExporter.export()`
  (`:23-31`)는 순수 `torch.jit.trace` + 검증이다. **외부 프로세스/빌드
  의존성이 없다** — CUDA 없이도, C++ 없이도 항상 동작 가능.
- `tests/training/test_train_export_parity.py` — 이미 "학습 → state_dict
  저장/재로드 → TorchScript export → **Python 쪽 `compare_outputs()`**"까지만
  검증하는 pytest가 존재한다(C++ 서브프로세스 없음). 즉 **"TorchScript
  export가 수치적으로 맞는가"를 C++ 바이너리 없이 pytest로 빠르게
  검증하는 선례가 이미 있다** — §14 테스트 설계에서 이 패턴을 그대로
  재사용한다.
- `tests/scripts/` 디렉터리에는 현재 Phase 4G에서 추가한
  `test_run_imagefolder_training_e2e_args.py`(5개)와
  `test_run_imagefolder_training_e2e_resume_cli.py`(4개), 총 **9개
  테스트**가 있다. `run_training_e2e.py`/`run_real_training_e2e.py`는
  대응하는 CLI 테스트 파일이 **없다** — 이 두 스크립트는 `--model-json`
  (그리고 `run_real_training_e2e.py`는 `--data-root`/`--*-limit`)만
  노출하는 좁은 CLI라서 별도 parser 테스트가 필요할 만큼 복잡해진 적이
  없었다. **`run_imagefolder_training_e2e.py`만 Phase 4E/4G를 거치며
  production CLI 수준으로 복잡해졌고, 그래서 처음으로 전용 CLI 테스트가
  필요해졌다** — 이 자체가 "이 스크립트가 원래 역할을 넘어섰다"는
  사용자의 문제의식을 코드로 뒷받침한다.

---

## 2. 문제점

E2E와 production 책임이 섞인 구체적 지점을 위 분석에 근거해 정리한다.

1. **`--batch-size`/`--learning-rate`가 여전히 하드코딩** — 실제 사용자
   dataset은 크기/GPU 메모리가 CIFAR-10 fixture(200장)와 전혀 다른데도
   `batch_size=8`, `learning_rate=1e-3`을 바꿀 방법이 없다. Phase 4G가
   "resume 연결"에만 집중하려고 의도적으로 미룬 항목인데, 지금 목표가
   "실제 프로덕션 사용"이라면 더 이상 미룰 수 없다.
2. **C++ parity가 옵트아웃 불가능** — `run_torchscript` 빌드가 없으면
   `subprocess.run(["...", "build_torchscript.py"])`가 자동 실행된다.
   빌드 도구 체인(MSVC 등)이 없는 환경에서 실제 학습만 하려는 사용자도
   이 경로를 강제로 거친다.
3. **`loss_improved`/class mapping 재로드/best model save-reload
   allclose — 세 가지 게이트가 실패하면 checkpoint/history/best model이
   이미 디스크에 저장된 뒤에도 `return 1`로 끝난다.** 이건 회귀
   테스트로서는 올바른 동작(파이프라인이 깨졌다는 신호)이지만, 실제
   사용자 입장에서는 "내 dataset이 조금 어려워서 1 epoch 만에 loss가
   내려가지 않았다"는 정상적인 상황도 "E2E: FAIL"이라는 무서운 메시지와
   0이 아닌 exit code로 끝나 버린다 — 학습 자체는 완전히 유효했는데도.
4. **고정된 artifact 경로/이름** — `ARTIFACTS_TRAINING`,
   `ARTIFACTS_TORCHSCRIPT` 등은 저장소 루트 기준 고정 경로다. 사용자가 두
   개의 다른 dataset을 같은 `model_spec.name`으로 학습하면 결과물이
   서로 덮어써진다(`ARTIFACT_SUFFIX`가 "같은 phase4c_cifar10_model.json을
   재사용하는 두 E2E"만 구분하도록 설계된 것이지, "여러 사용자 실행"을
   구분하도록 설계된 게 아니다).
5. **`model_spec.name`이 경로에 직접 쓰이는데 문자 제약이 없다** — §1-3에서
   확인. 지금은 예시 모델들의 이름이 전부 안전한 snake_case라 드러나지
   않았을 뿐, 실제 사용자가 임의의 이름을 쓰는 production CLI에서는
   잠재 버그다.
6. **CLI 테스트가 이 스크립트에만 새로 필요해짐** — §1-3에서 확인한 대로,
   이 스크립트만 Phase 4E/4G를 거치며 "일반 학습 CLI"로 성장했다는 것
   자체가 책임 분리가 필요하다는 가장 직접적인 증거다.
7. **(추가 발견) `class mapping 재로드 검증`과 `best model save/reload
   allclose 검증`은 이미 단위 테스트로 커버된 불변조건을 매 실행마다
   다시 확인하는 것과 같다** — `tests/training/test_imagefolder_dataset.py::
   test_save_and_load_class_mapping_round_trip`과
   `tests/training/test_checkpoint.py`(state_dict round-trip)가 이미 이
   계약을 pytest로 고정해 두었다. 프로덕션 실행마다 같은 것을 다시
   검증하는 건 안전성을 더하지 않고 forward pass 1회 + JSON round-trip을
   추가 비용으로 쓸 뿐이다 — 이건 사용자가 준 목록에 명시되진 않았지만
   "테스트 성공/실패 판정을 위한 gate"로 분류해 production에서 제거할
   것을 권장한다(§17에 재확인 항목으로 남김).

---

## 3. Phase 4H 목표/범위

### 구현할 것

- `scripts/train_imagefolder.py` — 실제 사용자용 production CLI(신규).
- `src/image_ai_studio/training/imagefolder_workflow.py` — CLI와 E2E가
  공유하는 orchestration 모듈(신규).
- `scripts/run_imagefolder_training_e2e.py`의 재구성 — production
  workflow를 고정 설정으로 호출하고, 회귀 gate + parity만 이 스크립트에
  남긴다.
- `--batch-size`/`--learning-rate`를 production CLI에 신규 노출(§5).
- `--output-dir` 기반 artifact 경로 정책(§9).
- TorchScript export를 기본 포함하되 옵트아웃 가능하게(§10).
- C++ parity를 production CLI에서 완전히 제거(§10).

### 제외할 것 (§12 사용자 목록 그대로 확인)

자동/주기적 checkpoint, latest/best checkpoint rotation, SIGINT 자동 저장,
checkpoint N개 보관, CUDA 학습, AMP, multi-worker resume, distributed
training, GUI, random augmentation, 다른 dataset 타입 CLI, hyperparameter
search, experiment tracking 도구 연동, YAML config. 이 항목들은 모두 Phase
4H의 "책임 분리"라는 목표와 독립적인 신규 기능이며, production CLI 분리가
끝난 뒤 별도 Phase에서 다룰 문제다.

---

## 4. 아키텍처 후보 비교

### 후보 A: 새 CLI가 기존 E2E 함수를 재사용

```text
train_imagefolder.py -> run_imagefolder_training_e2e.py 내부 함수 import
```

**기각.** 이유:

- `run_imagefolder_training_e2e.py`는 스크립트지 라이브러리가 아니다 —
  최상단에서 `sys.path.insert(0, str(REPO_ROOT / "src"))`(`:90-91`)를
  모듈 import 시점에 실행하는 부작용이 있고, `main()`이 dataset 검증부터
  C++ parity까지 전부 한 함수에 있어 "학습만" 재사용할 방법이 없다(함수를
  쪼개지 않는 한). 실제로 이 프로젝트의 기존 관례도 `scripts/`를
  라이브러리로 import하는 건 **테스트 목적으로만**(Phase 4G의
  `tests/scripts/test_run_imagefolder_training_e2e_*.py`가 `sys.path`에
  `scripts/`를 추가해 모듈로 import) 존재하지, 다른 production 코드가
  이를 의존하는 선례는 없다.
- 근본적으로 **의존 방향이 거꾸로**다: E2E/회귀 검증 코드는 production
  코드의 정확성을 확인하는 "테스트"이어야 하는데, production CLI가 E2E
  스크립트에 의존하면 "테스트가 제품 코드에 의존"하는 게 아니라 "제품이
  테스트에 의존"하게 된다. E2E 스크립트가 나중에 회귀 검증 목적으로
  자유롭게 바뀌면(예: 새로운 gate 추가) production CLI의 동작이 의도치
  않게 함께 바뀔 위험이 생긴다.

### 후보 B: 공통 orchestration 모듈 추출

```text
src/image_ai_studio/training/imagefolder_workflow.py
    scripts/train_imagefolder.py -> imagefolder_workflow.py
    scripts/run_imagefolder_training_e2e.py -> imagefolder_workflow.py
```

**채택.** 근거:

- §1-2에서 확인했듯, "학습 본질" 로직(모델/데이터셋 검증, 학습 실행,
  checkpoint/history/best model/class mapping 저장, test 평가)은 이미
  지금도 CLI와 E2E 양쪽에 **동시에** 필요하다(사용처가 이미 2곳
  존재 — 가정이 아니라 현재 요청 자체가 "CLI도 필요, E2E도 유지"다).
  Phase 4F 설계 문서가 정립한 원칙("사용처가 하나뿐이면 추상화 정당화
  안 됨")이 여기서는 반대로 적용된다 — 사용처가 이미 둘이다.
- Phase 4F/4G가 여러 리뷰 라운드를 거치며 증명했듯, resume의 RNG
  복원 순서는 **미묘하고 손으로 두 곳에 나눠 구현하면 어긋나기 쉽다**
  (Phase 4G 리뷰에서 실제로 순서 버그가 한 번 지적되어 수정됨,
  `docs/phase4g_imagefolder_resume_design.md` 개정 1 참고). 이 로직을
  하나의 함수/모듈에 두면 향후 수정(예: 버그 수정, 새 검증 추가)이
  한 곳에서만 일어나고 두 호출자 모두 자동으로 적용받는다.
- 후보 A의 의존 역전 문제가 없다 — CLI/E2E 둘 다 "아래" 계층인
  `src/image_ai_studio/training/`을 향해서만 의존한다. 이는 Phase 4F가
  이미 확립한 계층 규칙(`config.py` ← `loop.py` ← `checkpoint.py`)과
  같은 방향이다.

### 후보 C: production CLI와 E2E를 거의 독립 유지

**기각.** 이유(중복 비용 분석):

- 지금 이미 존재하는 로직만으로도 fresh-training 경로(모델 build →
  dataset 검증 → DataLoader 구성 → `run_training()` → checkpoint/history/
  best model/class mapping 저장)가 약 100줄, resume 경로(metadata 로드 →
  검증 → model/generator 복원 → `TrainingResumeState` 조립 → RNG 복원)가
  약 60줄이다. 이 160줄을 두 파일에 손으로 복제하면, RNG 순서처럼 눈에
  잘 안 띄는 버그가 한쪽에만 있거나 나중에 한쪽만 수정될 위험이 상시
  존재한다.
- 향후(다음 Phase) 자동 checkpoint 같은 기능을 추가할 때, 독립
  유지라면 두 파일을 매번 동시에 고쳐야 한다 — 이건 사용자가 이번
  Phase에서 명시적으로 피하고 싶어하는 "자동 checkpoint를 다음 Phase에서
  안전하게 얹을 수 있는 기반"이라는 목표와 정면으로 충돌한다.

**최종 선택: 후보 B.**

---

## 5. Production CLI 설계

### 파일 이름: `scripts/train_imagefolder.py`

기존 명명 규칙 확인 결과(`scripts/export_models.py` — `동사_명사.py`,
`_e2e`/`test` 접미사 없음, production-style 유틸리티 스크립트) 사용자가
제안한 `train_imagefolder.py`가 이 규칙과 정확히 일치한다. 더 나은 후보를
찾지 못해 그대로 채택한다.

### CLI 옵션 분류

| 옵션 | 분류 | 근거 |
|---|---|---|
| `--model-json` | **이번 Phase 필수 노출** | 기존과 동일 |
| `--dataset-root` | **이번 Phase 필수 노출** | 기존과 동일, 단 CIFAR-10 fixture 기본값은 제거(§5 하단) |
| `--epochs` | **이번 Phase 필수 노출** | Phase 4G에서 이미 노출됨 |
| `--batch-size` | **이번 Phase 필수 노출(신규)** | §2-1에서 지적한 문제 해결. 기본값 8(기존 `DEFAULT_BATCH_SIZE`) 유지 |
| `--learning-rate` | **이번 Phase 필수 노출(신규)** | 위와 동일 이유. 기본값 1e-3(기존 `DEFAULT_LEARNING_RATE`) 유지 |
| `--optimizer`/`--momentum`/`--lr-scheduler`/`--lr-scheduler-factor`/`--lr-scheduler-patience`/`--early-stopping-patience` | **이번 Phase 필수 노출** | Phase 4E에서 이미 노출됨, 그대로 이전 |
| `--resume-from`/`--checkpoint-out` | **이번 Phase 필수 노출** | Phase 4G에서 이미 노출됨, 그대로 이전 |
| `--output-dir` | **이번 Phase 필수 노출(신규)** | §9 참고, artifact 경로 정책의 핵심 |
| `--seed` | **이번 Phase 필수 노출(신규)** | 기본값은 기존 `SEED=20260730` 상수. 아래 "seed와 resume" 참고 — resume 계약에 영향 없음을 확인했으므로 노출해도 안전 |
| `--export-torchscript`/`--no-export-torchscript` | **이번 Phase 필수 노출(신규)** | `argparse.BooleanOptionalAction`(Python 3.10+, `pyproject.toml`의 `requires-python = ">=3.10"`과 호환 확인됨) 사용. 기본값 `True`(export 기본 포함, §10) |
| `--device` | **production CLI에는 부적절** | Phase 4F/4G가 CUDA RNG resume을 명시적으로 지원하지 않음(`docs/phase4f_checkpoint_resume_design.md`) — `--device cuda`를 열면 exact-resume 계약이 조용히 깨진다. 하드코딩 `"cpu"` 유지 |
| `--run-parity`/`--runner-path` | **production CLI에는 부적절** | C++ parity는 "알려진 참조값과 비교"가 목적인 배포 파이프라인 검증 도구이지 실제 dataset 학습 결과를 검증하는 도구가 아니다(비교할 참조값 자체가 없음). E2E 전용으로 완전히 제거(§10) |
| `--artifact-name` | **향후 Phase로 미룸** | `--output-dir` 자체가 이름 충돌을 구조적으로 없애므로(§9) 이번 Phase에서는 불필요. 여러 실행을 같은 상위 디렉터리에 정리하고 싶다는 요구가 생기면 그때 추가 |

**`batch_size`/`learning_rate`와 resume 호환성**: 두 필드 모두 이미
`RESUME_CONFIG_FIELDS`(`config.py:24-32`)에 포함되어 있어 `--resume-from`
사용 시 `require_compatible_resume_config()`가 checkpoint와 다르면
명확한 `ValueError`로 거부한다(기존 Phase 4F 계약 그대로, 새로 만들
필요 없음) — CLI로 노출한다고 새로운 검증 로직이 필요하지 않다.

**`--seed`와 resume 계약**: fresh 학습에서는 `--seed`가 `torch.manual_seed()`
(model 초기화용)와 DataLoader `loader_generator.manual_seed()` 둘 다에
쓰인다. **resume에서는 `--seed`가 사실상 아무 영향이 없다** — model은
`set_seed()` 직후 `build_model()`로 초기화되지만 그 즉시
`model.load_state_dict(payload["model_state_dict"])`로 덮어써지고,
DataLoader generator는 `--seed`가 아니라 `payload["loader_generator_state"]`로
복원되며, 전역 CPU RNG도 최종적으로 `payload["cpu_rng_state"]`로 덮어써진다
(Phase 4G 설계 §3-2). 따라서 resume 시 `--seed`에 어떤 값을 줘도 결과가
같다 — 이 사실을 `--help` 문서와 README에 명시해 사용자 혼란을 막는다.

**CIFAR-10 fixture 기본값 제거**: `--model-json`/`--dataset-root`는
production CLI에서 **기본값을 두지 않고 필수 인자로 만든다**(`required=True`).
현재 E2E의 기본값(`MODEL_JSON`, `DEFAULT_DATASET_ROOT`)은 회귀 검증용
CIFAR-10 fixture를 가리키는 값이라 production CLI에 그대로 두면 "인자를
깜빡한 사용자가 자기도 모르게 CIFAR-10 fixture를 학습"하는 혼란스러운
기본 동작이 된다.

### 실행 예시

```bash
# 새로 학습
python scripts/train_imagefolder.py \
    --model-json my_model.json --dataset-root path/to/dataset \
    --epochs 20 --batch-size 32 --learning-rate 5e-4 \
    --output-dir artifacts/my_run --checkpoint-out artifacts/my_run/checkpoint.pt

# 이어서 학습
python scripts/train_imagefolder.py \
    --model-json my_model.json --dataset-root path/to/dataset \
    --epochs 10 --batch-size 32 --learning-rate 5e-4 \
    --output-dir artifacts/my_run \
    --resume-from artifacts/my_run/checkpoint.pt --checkpoint-out artifacts/my_run/checkpoint.pt

# TorchScript export 생략 (가중치만 필요한 경우)
python scripts/train_imagefolder.py --model-json my_model.json --dataset-root ... \
    --output-dir artifacts/my_run --no-export-torchscript
```

---

## 6. Workflow 모듈 설계

**모듈 위치: `src/image_ai_studio/training/imagefolder_workflow.py`(신규).**
`imagefolder_resume.py`와 같은 계층에 둔다 — `checkpoint.py`/`loop.py`/
`config.py`/`imagefolder_resume.py`/`torchvision_dataset.py`를 모두
import하지만 그 반대(이 모듈을 저 아래 계층이 import)는 없다. 순환 의존
위험 없음.

### dataclass 설계 — 사용자가 제안한 3개(Request/Artifacts/Outcome) 대신 **2개**로 단순화

과설계 방지 원칙(§14 사용자 지침) 적용: "artifact 경로"와 "학습 결과
지표"를 별도 dataclass로 나눌 만큼 사용처(CLI/E2E)의 요구가 다르지
않다 — 두 호출자 모두 "저장된 파일 경로 + 학습이 어떻게 됐는지"를
**함께** 필요로 한다(E2E는 경로로 TorchScript reload/parity를 하고,
동시에 `history`로 loss-decreased gate도 판단해야 함). 하나의 결과
dataclass로 합쳐도 필드 수가 과도해지지 않는다.

```python
@dataclass
class ImageFolderWorkflowRequest:
    model_json_path: Path
    dataset_root: Path
    training_config: TrainingConfig      # 이미 검증된 TrainingConfig를 그대로 받는다
                                          # (호출자가 optimizer/scheduler/epochs를
                                          # 어디서 어떻게 조립하든 상관없이 재사용 --
                                          # CLI는 argparse에서, E2E는 고정 상수에서)
    output_dir: Path
    resume_from: Path | None = None
    checkpoint_out: Path | None = None
    export_torchscript: bool = True
    seed: int = 20260730                 # 기존 SEED 상수와 동일한 기본값


@dataclass
class ImageFolderWorkflowResult:
    history: TrainingHistory
    test_loss: float
    test_accuracy: float
    best_model_state_dict_path: Path
    training_history_path: Path
    class_mapping_path: Path
    test_result_path: Path                      # (리뷰 반영) test_loss/test_accuracy를
                                                  # 담은 JSON -- §9/§7 참고, 기존 E2E와
                                                  # 동일한 계약을 Result에도 명시
    checkpoint_path: Path | None
    checkpoint_metadata_path: Path | None
    torchscript_model_path: Path | None
    torchscript_metadata_path: Path | None


def run_imagefolder_training_workflow(request: ImageFolderWorkflowRequest) -> ImageFolderWorkflowResult:
    ...
```

`ImageFolderWorkflowResult`는 **경로와 지표만** 담고, 살아있는 `nn.Module`/
텐서 객체는 담지 않는다 — 호출자가 필요하면 저장된 파일에서 다시
`load_state_dict()`/`load_model_spec()`으로 읽으면 된다(둘 다 순수하고
저렴한 호출). 이렇게 하면 워크플로우 함수의 "출력 계약"이 파일 시스템
경계와 정확히 일치해, CLI(다음 프로세스 실행에서 resume) / E2E(다음
단계에서 parity 검증) 양쪽에게 자연스럽다.

### 워크플로우가 print()를 직접 하지 않는다

**핵심 설계 결정**: `run_imagefolder_training_workflow()`는 진행 상황을
직접 `print()`하지 않는다. 예외를 던지고 `Result`를 반환할 뿐이다.
근거:

- `run_training()`(`loop.py`)은 이미 epoch별 콜백이 없다(Phase 4F/4G가
  콜백을 명시적으로 범위 밖으로 뒀음, `docs/phase4g_imagefolder_resume_design.md`
  §12). 즉 **지금도** "epoch 1: train_loss=..." 같은 줄은 실시간 로그가
  아니라, `run_training()`이 끝난 뒤 반환된 `history.train_losses` 등을
  루프 돌며 사후에 출력하는 것이다(`:390-394`). 워크플로우가 `history`를
  `Result`에 그대로 담아 반환하면, CLI/E2E 각자가 원하는 형식으로 **똑같이
  사후 출력**할 수 있다 — 콜백이나 로거 주입 같은 새 추상화가 전혀
  필요 없다.
- CLI와 E2E는 원하는 출력 스타일이 다르다(CLI는 사용자 친화적 요약, E2E는
  "PASS"/"FAIL" 단계별 진단). 출력 형식 결정을 워크플로우 밖으로 빼면 두
  스타일이 서로 간섭하지 않는다.

### resume 준비를 하나의 사설 함수로 격리

**(리뷰 반영, 아래 순서로 수정됨)** 사용자가 §10에서 요청한 대로, resume
준비 순서(Phase 4G §3-2에서 확립한 9단계 — metadata 검증 → model/generator
준비 → DataLoader 생성 → ResumeState/config 검증 → CPU RNG 복원)를
**`_prepare_resume()`이라는 워크플로우 모듈 내부의 단일 비공개 함수**에
모아 두되, **CPU RNG 복원 자체는 이 함수 밖(호출자)에서 한다.**

`DataLoader` 생성은 `_prepare_resume()`이 반환된 **뒤**
`run_imagefolder_training_workflow()`에서 일어난다(model/generator 준비와
DataLoader 생성이 서로 다른 함수에 걸쳐 있음). 만약 `_prepare_resume()`
안에서 `torch.set_rng_state()`까지 호출하고 반환하면, 반환 이후 실행되는
호출자의 DataLoader 생성 코드가 "RNG 복원"과 "`run_training()` 호출"
사이에 끼어들게 된다 — DataLoader 생성 자체는 RNG를 소비하지 않아
수치 결과에는 영향이 없지만(실제로 이전 초안은 이 순서였는데도
Phase 4G 실측 검증에서는 문제가 없었다), **"RNG 복원과
`run_training()` 사이에 다른 코드를 절대 두지 않는다"는 불변조건을
코드 구조로 강제**하기 위해 복원 자체를 호출자로 옮긴다. 즉
`_prepare_resume()`은 저장된 CPU RNG state를 **반환만** 하고, 실제
`torch.set_rng_state()` 호출은 DataLoader 생성이 전부 끝난 다음, 호출자
쪽에서 `run_training()` 바로 앞줄에 둔다.

```python
def _prepare_resume(
    request: ImageFolderWorkflowRequest,
    model_spec: ModelSpec,
    splits: ImageFolderSplits,
) -> tuple[
    nn.Module,
    torch.Generator,
    TrainingResumeState | None,
    torch.Tensor | None,
]:
    """request.resume_from이 None이면 (신규 model, 신규 generator, None, None)을
    반환한다. 있으면 Phase 4G §3-2의 순서(metadata 로드/검증 -> model
    build+load -> generator 복원 -> ResumeState 조립 -> config 검증)를
    전부 수행한 뒤 (model, restored_generator, resume_state,
    payload["cpu_rng_state"])를 반환한다.

    **이 함수는 전역 CPU RNG를 절대 건드리지 않는다** -- 네 번째 반환값
    (cpu_rng_state)은 호출자가 DataLoader 생성을 전부 마친 뒤, run_training()
    호출 바로 직전에 torch.set_rng_state()로 직접 적용해야 한다. 이 함수
    안에서 미리 복원해버리면, 함수가 반환된 뒤 호출자가 하는 DataLoader
    생성 코드가 복원 시점과 run_training() 사이에 끼어들게 되어 "복원은
    항상 마지막, 그 뒤 바로 run_training()"이라는 불변조건이 함수 경계
    때문에 깨진다."""
```

`run_imagefolder_training_workflow()` 쪽 호출 지점(§7/§8에서 공유하는
코드 블록):

```python
model, loader_generator, resume_state, cpu_rng_state = _prepare_resume(request, model_spec, splits)

train_loader = DataLoader(splits.train, generator=loader_generator, ...)
val_loader = DataLoader(splits.val, ...)
test_loader = DataLoader(splits.test, ...)

if cpu_rng_state is not None:
    torch.set_rng_state(cpu_rng_state)

training_result = run_training(
    model, train_loader, val_loader, request.training_config,
    device="cpu", resume_state=resume_state,
)
```

`torch.set_rng_state(cpu_rng_state)`와 `run_training(...)` 사이에는 출력,
텐서 생성, model 생성, 추가 validation, DataLoader 생성 등 **어떤 코드도
두지 않는다** — 이 두 줄이 항상 바로 붙어 있도록 유지하는 것이
불변조건을 지키는 유일한 방법이다.

---

## 7. Fresh training 흐름

`request.resume_from is None`일 때 `run_imagefolder_training_workflow()`의
순서(전부 기존 E2E 로직 재배치, 신규 로직 없음). §6에서 정리한 대로
fresh 경로도 `_prepare_resume()`을 거친다(`resume_from=None`이면 신규
model/generator/`resume_state=None`/`cpu_rng_state=None`을 반환) — fresh와
resume이 동일한 함수 경계, 동일한 "DataLoader 생성 후 RNG 처리" 구조를
공유한다:

1. `load_model_spec()` + `validate_model_spec()` — 실패 시 `ModelValidationError`.
2. `make_imagefolder_datasets()` + `require_matching_num_classes()` — 실패 시 `ValueError`.
3. `model, loader_generator, resume_state, cpu_rng_state = _prepare_resume(request, model_spec, splits)`
   — fresh 경로에서는 `set_seed(request.seed)` → `build_model(model_spec)` →
   `loader_generator = torch.Generator().manual_seed(request.seed)`,
   `resume_state`/`cpu_rng_state` 둘 다 `None`.
4. train/val/test `DataLoader` 구성(`loader_generator` 사용).
5. `cpu_rng_state is not None`이면 `torch.set_rng_state(cpu_rng_state)`
   (fresh 경로에서는 이 블록이 아무 일도 하지 않음) → **다른 RNG 소비
   작업 없이 즉시**
   `run_training(model, train_loader, val_loader, request.training_config, device="cpu", resume_state=resume_state)`.
6. `cpu_rng_state_after = torch.get_rng_state().clone()`,
   `loader_generator_state_after = loader_generator.get_state().clone()` (checkpoint 저장용,
   `request.checkpoint_out`이 주어졌을 때만 실제로 사용).
7. `request.output_dir`에 `training_history.json`/`class_mapping.json` 저장.
8. `request.checkpoint_out`이 주어졌으면 `save_training_checkpoint()` +
   `save_imagefolder_resume_metadata()` — **`best_model` 생성 전에** 수행
   (Phase 4G §10과 동일한 버그 회피).
9. `best_model = build_model(model_spec); best_model.load_state_dict(training_result.best_state_dict)`
   → `save_state_dict()`로 `best_model_state_dict.pt` 저장.
10. `evaluate(best_model, test_loader, device="cpu")` → test_loss/test_accuracy →
    `output_dir/test_result.json`에 `{"test_loss": ..., "test_accuracy": ...}`
    저장(**리뷰 반영, §9 계약 통일** — 기존 E2E와 동일한 내용/형식이지만
    저장할 전용 helper 함수가 프로젝트에 따로 없으므로(`run_imagefolder_training_e2e.py`/
    `run_real_training_e2e.py` 둘 다 `json.dumps(...)` + `Path.write_text(...)`를
    그 자리에서 직접 쓰는 게 기존 관례임, §1-3 확인) 워크플로우도 같은
    표준 `json`/`pathlib` 조합을 그대로 재사용한다 — 이 저장을 감싸는
    새 함수를 만들지 않는다).
11. **TorchScript export 또는 기존 산출물 정리(리뷰 반영, §9/§10 신규 정책)**:
    - `request.export_torchscript`가 참이면: `set_seed(request.seed)` →
      `example_input = torch.randn(1, *model_spec.input_shape)` →
      `TorchScriptExporter().export(...)` → `output_dir/model.ts` +
      `output_dir/model_metadata.json`. 이 `example_input`은 디스크에
      저장하지 않는다(§9) — 오직 trace용으로만 메모리에서 쓰고 버린다.
    - 거짓이면: `(output_dir / "model.ts").unlink(missing_ok=True)`,
      `(output_dir / "model_metadata.json").unlink(missing_ok=True)`로
      **이전 실행이 같은 `output_dir`에 남긴 TorchScript 산출물을
      제거**한다(§9 "stale artifact" 정책). 삭제 대상은 워크플로우가
      고정 이름으로 관리하는 이 두 파일뿐이고, 파일이 없어도 오류 없이
      통과(`missing_ok=True`)하며, 삭제 자체가 실패하면(권한 등) `OSError`를
      그대로 호출자에게 전파한다(감싸지 않음, §11과 동일 원칙).
12. `ImageFolderWorkflowResult` 조립(export를 안 했거나 정리했다면
    `torchscript_model_path`/`torchscript_metadata_path`는 `None`) 후 반환.

기존 E2E에 있던 다음 세 가지는 **워크플로우에서 제거**(§2-7의 이유,
production 기본 경로에서 빠짐):
- class mapping 저장 후 재로드해서 `==` 비교하는 자체 검증.
- best model save 후 재로드해서 `torch.allclose()` 비교하는 자체 검증.
- `loss_improved` 게이트(loss가 실제로 줄었는지 판정).

이 세 가지는 §12에서 설명하듯 E2E 스크립트가 `Result`를 받은 뒤 **자기
책임으로** 다시 확인한다(워크플로우가 하는 게 아니라 E2E가 호출 후 직접).

---

## 8. Resume 흐름

`request.resume_from is not None`일 때 `_prepare_resume()`(§6) 내부 순서
— Phase 4G 설계 §3-2를 그대로 계승하되, **CPU RNG 복원 단계는 이 함수
안에 두지 않는다**(리뷰 반영, §6 참고):

1. `saved_metadata = load_imagefolder_resume_metadata(metadata_path_for_checkpoint(request.resume_from))`
   + `payload = load_training_checkpoint(request.resume_from)` — `except (ValueError, OSError)`로
   호출자에게 그대로 전파(워크플로우가 감싸지 않음, §11).
2. `splits`(호출자가 이미 만들어 전달, §7의 2번과 공유)로
   `current_metadata = build_imagefolder_resume_metadata(model_spec, splits)`.
3. `require_compatible_imagefolder_resume_metadata(saved_metadata, current_metadata)` —
   **model/DataLoader를 만들기 전에** 끝낸다.
4. `set_seed(request.seed)` → `model = build_model(model_spec)` →
   `model.load_state_dict(payload["model_state_dict"])`(`best_state_dict` 아님).
5. `restored_generator = torch.Generator(); restored_generator.set_state(payload["loader_generator_state"])`.
6. `TrainingResumeState` 조립(`history=TrainingHistory(**payload["history"])` 등) +
   `require_compatible_resume_config(resume_state.training_config, request.training_config)` —
   **이 시점까지 전역 CPU RNG 미변경**.
7. `(model, restored_generator, resume_state, payload["cpu_rng_state"])` 반환.
   **`torch.set_rng_state()`는 여기서 호출하지 않는다** — 그 값을 그대로
   호출자에게 넘길 뿐이다.

`_prepare_resume()`이 반환된 뒤 `run_imagefolder_training_workflow()`는
§6에서 보인 공유 코드 블록을 그대로 실행한다: 이 `model`/`restored_generator`로
train/val/test `DataLoader`를 **전부 구성**한 다음, `cpu_rng_state`(4번째
반환값)가 `None`이 아니면 그제서야 `torch.set_rng_state(cpu_rng_state)`를
호출하고, **다른 RNG 소비 작업 없이 즉시**
`run_training(..., resume_state=resume_state)`를 호출한다. 최종 순서는:

```text
checkpoint/metadata 검증 (1~3)
-> model/generator/ResumeState 준비 (4~6)
-> DataLoader 생성 (호출자, §6 공유 블록)
-> config 검증 완료 (6, _prepare_resume 안에서 이미 끝남)
-> CPU RNG 복원 (호출자, DataLoader 생성 다음)
-> 다른 RNG 소비 작업 없이 즉시 run_training()
```

이후(checkpoint 저장, best model/history/class mapping 저장, test 평가,
TorchScript export/정리)는 §7의 6~12단계와 완전히 동일한 코드 경로를
탄다 — fresh/resume 분기는 `_prepare_resume()`이 반환하는 값(신규 vs
복원된 model/generator/resume_state/cpu_rng_state)으로만 결정되고, 그
뒤로는 단일 코드 경로다.

---

## 9. Artifact 정책

### 저장되는 artifact 전수 조사 (현재 E2E 기준)

| Artifact | 현재 위치(하드코딩) | production CLI에서 |
|---|---|---|
| best model state_dict | `ARTIFACTS_TRAINING/{artifact_name}_state_dict.pt` | `output_dir/best_model_state_dict.pt` |
| training history JSON | `ARTIFACTS_TRAINING/{artifact_name}_history.json` | `output_dir/training_history.json` |
| class mapping JSON | `ARTIFACTS_TRAINING/{artifact_name}_classes.json` | `output_dir/class_mapping.json` |
| test 평가 결과 JSON | `ARTIFACTS_TRAINING/{artifact_name}_test_result.json` | `output_dir/test_result.json` |
| TorchScript model | `ARTIFACTS_TORCHSCRIPT/{artifact_name}/model.pt` | `output_dir/model.ts`(선택, off 시 이전 실행 산출물 제거 — 아래 "stale artifact 정책" 참고) |
| TorchScript metadata | `ARTIFACTS_TORCHSCRIPT/{artifact_name}/metadata.json` | `output_dir/model_metadata.json`(선택, 위와 동일) |
| example input/output tensor(`.bin`/`.json`) | `ARTIFACTS_COMMON`/`ARTIFACTS_REFERENCE` | **저장하지 않음** — C++ parity 전용 산출물이라 production에서 필요 없음(아래 설명) |
| full checkpoint | (Phase 4G, `--checkpoint-out`) | `request.checkpoint_out`(사용자가 명시적으로 지정, §9 "output-dir과의 관계") |
| ImageFolder resume metadata | `<checkpoint>.meta.json`(자동 유도) | 동일 규칙 유지(`metadata_path_for_checkpoint()`) |

**example input/output tensor를 저장하지 않는 이유**: 이 파일들은 오직
C++ runner가 Python 쪽 참조값과 비교하기 위한 입력을 지정된 형식으로
전달하는 용도다(`save_tensor()`/`load_tensor()`, `parity/tensor_io.py`).
production CLI는 C++ parity를 아예 실행하지 않으므로(§10) 이 파일을 만들
이유가 없다 — `TorchScriptExporter.export()`가 요구하는 `example_input`은
메모리에서 만들어 trace에만 쓰고 버린다.

### 설계 질문별 정책

- **기존 파일이 있을 때 덮어쓸지**: **덮어쓴다(확인 없음).** 기존
  프로젝트의 모든 저장 함수(`save_state_dict`/`save_training_history`/
  `save_class_mapping`/`TorchScriptExporter.export`/`save_training_checkpoint`)가
  이미 예외 없이 덮어쓰기 방식이다 — 새 확인 절차를 넣으면 이 프로젝트의
  기존 관례와 어긋나고, 비대화형 스크립트 철학과도 맞지 않는다. 대신
  CLI가 각 저장 직후 어떤 파일이 쓰였는지 명확히 출력한다(기존 패턴
  유지).
- **resume 시 같은 output-dir 사용 가능 여부**: **가능하다.** `output_dir`
  아래 파일들은 "최신 상태"를 나타내는 것이지 버전 스냅샷이 아니다 — 매
  실행마다 덮어써진다. 버전 관리가 필요하면 사용자가 직접 다른
  `--output-dir`를 지정해야 한다(자동 타임스탬프 디렉터리 같은 기능은
  과설계이므로 만들지 않음).
- **checkpoint-out과 output-dir의 관계**: **독립적으로 유지한다(자동
  유도하지 않음).** `--checkpoint-out`을 생략하면 checkpoint를 저장하지
  않는다 — Phase 4G의 4-조합 정책(§8, `docs/phase4g_imagefolder_resume_design.md`)을
  그대로 계승. `--output-dir`가 주어졌다고 checkpoint가 자동으로
  생기는 "숨은 동작"을 만들지 않는다 — 가장 단순하고 예측 가능한 정책을
  우선한다(사용자 지침 §6 "과설계를 피하고 가장 단순한 정책을 추천"에
  따름). 실사용 예시(§5)에서는 `--checkpoint-out <output-dir>/checkpoint.pt`를
  관례로 문서화한다.
- **사용자 지정 경로와 자동 경로의 우선순위**: `--output-dir`는 이번
  Phase에서 **필수 인자**로 만든다(자동/기본 경로 없음) — "artifact
  경로를 하드코딩하는 부분을 없앤다"는 목표와 직접 연결된다.
- **artifact_name 충돌**: `output_dir` 기반 고정 파일명(`best_model_state_dict.pt`
  등)으로 전환하면서 **`model_spec.name` 기반 파일명 조합 자체가
  사라진다** — 즉 "서로 다른 실행의 산출물이 이름이 같아서 덮어써지는"
  문제는 "사용자가 서로 다른 `--output-dir`를 쓰면 된다"는 훨씬 단순한
  규칙으로 대체된다.
- **model name의 경로 부적절 문자 위험**: 위와 같은 이유로 **구조적으로
  사라진다** — `model_spec.name`은 이제 어디에서도 경로 구성에 쓰이지
  않는다(TorchScript export의 `model_name=` 인자는 `metadata.json` 안에
  문자열로 기록될 뿐 경로에 쓰이지 않음, `export/torchscript_exporter.py`
  확인됨). 별도 sanitize 로직을 추가할 필요가 없다 — `--output-dir`
  전환의 부수 효과로 문제 자체가 없어진다.
- **Windows 경로 지원**: 전부 `pathlib.Path` 기반(`argparse`의
  `type=Path`)이라 기존 스크립트들과 동일하게 Windows 경로를 그대로
  지원한다 — 새로 처리할 것 없음.
- **stale artifact 정책(리뷰 반영, 신규)**: 같은 `output_dir`를 재사용하며
  `--export-torchscript`였다가 `--no-export-torchscript`로 바꿔 다시
  실행하면, 이전 실행이 남긴 `model.ts`/`model_metadata.json`이 그대로
  남아 있으면 사용자가 이번 실행 결과로 착각할 위험이 있다. 이를 막기
  위해 `export_torchscript=False`일 때 워크플로우가 **직접**
  `output_dir/model.ts`와 `output_dir/model_metadata.json`을
  `Path.unlink(missing_ok=True)`로 제거한다(§7 11번, §10). 규칙:
  - 삭제 대상은 워크플로우가 고정 이름으로 관리하는 이 두 파일뿐이다
    — `output_dir` 안의 다른(사용자가 직접 넣은) 파일은 절대 건드리지
    않는다.
  - 파일이 애초에 없어도(첫 실행부터 export를 껐거나 이미 지워진 경우)
    `missing_ok=True`라 오류 없이 통과한다.
  - 삭제 자체가 실패하면(권한 문제 등) 그 `OSError`를 감싸지 않고
    호출자에게 그대로 전파한다(§11의 "예외를 삼키지 않는다" 원칙과
    동일).
  - 이 경우 `ImageFolderWorkflowResult.torchscript_model_path`/
    `torchscript_metadata_path`는 항상 `None`이다(파일이 실제로 없는
    상태와 Result가 항상 일치).

---

## 10. Export와 parity 책임 분리

| 단계 | production CLI(워크플로우) | E2E |
|---|---|---|
| test 평가 | **기본 실행** (§7 10번) | 워크플로우 재사용, 결과로 gate 판단 |
| best model 저장 | **기본 실행** | 워크플로우 재사용 |
| best model save/reload allclose 검증 | **제거**(§2-7, 이미 단위 테스트로 커버됨) | **개정: E2E에도 재도입하지 않고 완전히 삭제**(§12) — `Result`가 살아있는 model을 반환하지 않으므로 "원본 vs 재로드" 비교가 애초에 불가능하고, 같은 파일을 두 번 읽어 비교하는 대안은 항상 같은 값만 나오는 무의미한 검증이라 회귀 가치가 없음. state_dict 저장/재로드 정확성은 `test_imagefolder_workflow.py`/`test_checkpoint.py`가 이미 단위 테스트로 커버 |
| class mapping 재로드 검증 | **제거** | **E2E가 직접 재도입** |
| TorchScript export | **기본 포함, `--no-export-torchscript`로 옵트아웃(끄면 같은 `output_dir`의 이전 산출물도 정리, §9 stale artifact 정책)** | 워크플로우 호출 시 `export_torchscript=True`로 고정 |
| example_input/output tensor 저장 | **하지 않음**(§9) | **E2E가 직접** 별도로 만들어 저장(parity에 필요하므로) |
| C++ CPU/CUDA parity | **완전히 제외**(옵트인 플래그도 없음) | **E2E 전용, 항상 강제 실행** |

production CLI가 C++ runner 빌드 위치나 CUDA 가용성에 의존하면 안 된다는
사용자 우려가 정확히 맞다 — `find_runner_binary`/`run_case`/
`subprocess.run(["...", "build_torchscript.py"])`(§1-3)를 워크플로우와
CLI 양쪽 모두에서 완전히 제거하는 것으로 해결한다. **워크플로우 모듈은
`image_ai_studio.tools.run_and_compare`를 아예 import하지 않는다** —
import 자체를 없애 실수로도 CLI 경로에서 C++ 의존성이 섞여 들어올 수
없게 한다.

**E2E가 production workflow 결과물을 받아 parity만 추가 검증하는 구조가
가능한가**: 가능하고, 그렇게 설계한다. E2E는:

1. `run_imagefolder_training_workflow(request)`를 호출해 `Result`를
   받는다(`export_torchscript=True`로 요청, `output_dir`은 E2E 전용 고정
   경로).
2. `Result.torchscript_model_path`를 그대로 C++ runner에 넘긴다 — TorchScript
   재수출이 필요 없다.
3. `Result.best_model_state_dict_path`를 다시 로드해(`load_state_dict()`) 자기
   책임으로 example_input을 만들고 Python 참조 출력을 계산 → `save_tensor()`로
   `ARTIFACTS_REFERENCE`에 저장 → `find_runner_binary`/`run_case`로 C++
   CPU/CUDA 실행.

---

## 11. Error handling

### 현재 흩어진 예외 조사 결과

§1-3에서 확인한 대로 **`ModelValidationError`/`TrainingConfigError` 둘
다 이미 `ValueError`의 서브클래스**다. 코드베이스 전체에서 이 계층
바깥의 예외는 `checkpoint.py`/`imagefolder_resume.py`가 명시적으로 내는
`ValueError`(이 역시 `ValueError` 자체)와, `torch.load()`가 파일이 없을 때
내는 `OSError`(`FileNotFoundError`) 뿐이다. TorchScript export
실패(`TorchScriptExporter.export()`)는 예외를 던지지 않고 `metadata.json`에
`status: "FAIL"` + `error_log`를 기록하는 **결과 기반** 실패 신호를
쓴다(`export/torchscript_exporter.py`, 이미 확인된 기존 동작 — 변경 없음).
C++ runner 오류(`run_case()`)도 마찬가지로 예외가 아니라 `status` 필드로
표현된다.

### 권장 구조

```text
imagefolder_workflow.py (workflow):
    예외를 삼키거나 재포장하지 않는다. ModelValidationError/
    TrainingConfigError/ValueError/OSError를 그대로 전파한다.
    TorchScript export 실패는 여기서 명시적으로 체크해서
    ValueError로 승격한다("TorchScript export failed: {error_log}") --
    결과 기반 실패 신호를 계속 호출자에게 떠넘기지 않고, 워크플로우의
    출력 계약을 "성공하면 Result, 실패하면 예외" 하나로 통일하기 위함.

scripts/train_imagefolder.py (main()):
    전체 워크플로우 호출을 단일 try/except로 감싼다:

        try:
            training_config = TrainingConfig(...)   # TrainingConfigError 여기서 발생 가능
            result = run_imagefolder_training_workflow(request)
        except (ModelValidationError, TrainingConfigError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    기존 E2E처럼 단계별로 나눠 잡지 않는다 -- "어느 내부 단계가
    실패했는지"는 회귀 진단 목적일 때만 의미 있고, 실제 사용자에게는
    "왜 실패했는지 메시지"만 있으면 충분하다. 예외 메시지 자체가 이미
    구체적이므로(Phase 4F/4G가 전부 "무엇이 왜 다른지" 담은 메시지를
    설계했음) 손실이 없다.

scripts/run_imagefolder_training_e2e.py (E2E main()):
    워크플로우 예외를 여전히 단계별로 잡아서 "FAIL: {exc}" +
    "IMAGEFOLDER TRAINING E2E: FAIL"을 출력한다(기존 패턴 유지) --
    회귀 진단에서는 "어느 단계에서 깨졌는지"가 실제로 유용한 정보이기
    때문에, 이 계층에서는 세분화된 처리를 유지하는 것이 맞다.
```

`except (ModelValidationError, TrainingConfigError, ValueError, OSError)`가
사실상 `except (ValueError, OSError)`와 동일한 예외 집합을 잡지만(둘 다
`ValueError` 서브클래스이므로), **명시적으로 구체 타입을 나열**한다 —
"이 함수가 어떤 예외 계열을 의도적으로 처리하는지" 코드만 보고 알 수
있게 하기 위함이며, 나중에 어느 하나가 `ValueError`를 상속하지 않는
독립 예외로 리팩터링되어도(가능성은 낮지만) 조용히 처리 범위에서
빠지는 대신 즉시 드러난다.

---

## 12. 기존 E2E 재구성

### E2E CLI 표면을 대폭 축소한다 (예상 밖 파급 효과 — 명시적으로 알림)

사용자가 제공한 "수정 후보" 목록에는 `scripts/run_imagefolder_training_e2e.py`와
`README.md`만 있었지만, 실제로 설계를 끝까지 따라가 보면 **이 두 파일
외에 `tests/scripts/test_run_imagefolder_training_e2e_args.py`와
`test_run_imagefolder_training_e2e_resume_cli.py`(Phase 4G에서 신설, 총 9개
테스트)도 재구성 대상**이 된다. 이유:

- production 목표(실제 사용자 config 자유도)와 회귀 검증 목표(고정된
  값으로 반복 가능한 결과)는 근본적으로 반대 방향이다. `run_training_e2e.py`/
  `run_real_training_e2e.py`(§1-3에서 확인)는 `--model-json`(+ `run_real_training_e2e.py`의
  `--data-root`/`--*-limit`)만 노출하고 optimizer/scheduler/epochs 등은
  **전부 모듈 상단 상수로 고정**한다(`TRAINING_CONFIG = TrainingConfig(epochs=10, ...)`,
  `run_training_e2e.py:61`). Phase 4E/4G가 `run_imagefolder_training_e2e.py`에
  붙인 `--optimizer`/`--epochs`/`--resume-from`/`--checkpoint-out` 등은
  전부 "production 사용성"을 위해 추가된 것이었지, "회귀 검증"에는
  필요했던 적이 없다(회귀는 항상 같은 고정값으로 돌려야 의미가 있다).
- 따라서 이 옵션들을 `train_imagefolder.py`로 이전하면, E2E의
  `parse_args()`는 형제 스크립트들과 같은 좁은 표면(`--model-json`/
  `--dataset-root`)으로 되돌아가는 것이 **자연스러운 결과**이지 임의
  선택이 아니다.
- 이 축소로 인해 Phase 4G가 만든 CLI 테스트 9개는 더 이상 대상이 되는
  플래그가 존재하지 않게 된다. **이 테스트들의 "의도"(resume 배선이
  맞는가, 잘못된 checkpoint를 명확히 거부하는가, epoch 누적이
  맞는가)는 사라지지 않는다 — `tests/scripts/test_train_imagefolder_cli.py`
  (신규)로 그대로 옮겨간다.** 이건 삭제가 아니라 "테스트가 검증하는
  대상 파일이 바뀌었으니 위치도 함께 옮긴다"에 가깝다. §13에서 파일
  단위로 정리한다.

### 재구성된 E2E의 흐름

```text
1. ModelSpec 로드 (--model-json, 기본값 유지 가능 -- 이제 이 스크립트
   본연의 회귀 검증 목적에 CIFAR-10 fixture 기본값이 다시 적절해짐)
2. CIFAR-10 ImageFolder fixture 준비 확인 (기존과 동일, 없으면 안내만
   -- 자동 준비는 하지 않음, 기존 동작 유지)
3. 고정 설정으로 워크플로우를 두 번 호출:
   (a) fresh: epochs=3, checkpoint_out=E2E 전용 고정 경로,
       export_torchscript=True
   (b) resume: epochs=2, resume_from=(a)의 checkpoint_out,
       checkpoint_out=동일 경로(Phase 4G의 "동일 경로 지원" 계약을
       회귀 검증에 그대로 활용)
4. (b)의 Result.history로 회귀 gate 확인:
   - loss_improved (train_losses[-1] < train_losses[0])
   - best_epoch/best_val_loss/test_accuracy를 §14-4의 anchor 수치와 함께
     **출력**(자동 gate 아님 -- §14-4 참고)
5. class mapping 재로드 검증 (E2E 자체 책임으로 재도입)
6. (b)의 Result.torchscript_model_path로 C++ CPU/CUDA parity 실행
7. PASS/FAIL 출력

**(리뷰 반영) best model save/reload 재검증은 여기 없다.** 이전 버전은
이걸 "E2E 자체 책임으로 재도입"한다고 적었지만, `ImageFolderWorkflowResult`
가 살아있는 model을 반환하지 않는 설계(§6)를 유지하는 한 "저장 전 원본
model vs 재로드 model" 비교 자체가 불가능하다 -- 대안으로 같은
`best_model_state_dict_path` 파일을 두 model에 각각 로드해서 서로
비교하는 방법은 항상 같은 값만 나오는(같은 파일을 같은 방법으로 두 번
읽을 뿐인) 검증이라 회귀 가치가 없다고 판단해 완전히 제거했다. state_dict
저장/재로드 자체의 수치적 정확성(원본 vs 재로드)은 이미
`tests/training/test_imagefolder_workflow.py`(best model 평가,
TorchScript export 출력 비교)와 `tests/training/test_checkpoint.py`가
단위 테스트로 충분히 커버한다.
```

**부수 효과(가치 있는 발견)**: 이 재구성은 현재 pytest/E2E 어디에도 없는
검증 하나를 자동으로 얻는다 — **실제 CIFAR-10 ImageFolder fixture로
"3 epoch 학습 + checkpoint 저장 -> resume 2 epoch"가 실제로 동작하는지가
매번 재현 가능한 스크립트로 고정된다.** 지금까지 이건 이 대화
세션에서 사람이 직접 커맨드 두 번을 수동으로 실행해서만 확인했던
것이다(Phase 4G 구현 완료 보고에 기록됨) — 재구성된 E2E가 이걸
자동화된 반복 가능한 검증으로 승격시킨다.

### 두 방식 비교: `main(argv)` 직접 호출 vs workflow 함수 직접 호출

**workflow 함수를 직접 호출하는 쪽을 채택한다.** 근거:

- E2E는 고정된 설정(CIFAR-10 fixture 경로, `epochs=3`/`2`, 고정 seed)만
  쓴다 — `ImageFolderWorkflowRequest`를 코드로 직접 조립하는 쪽이
  타입 안전하고, 문자열 CLI 인자 리스트를 만들었다가 다시 argparse로
  파싱하는 불필요한 왕복이 없다.
- `main(argv)`를 호출하면 반환값이 exit code(정수)뿐이라 E2E가 필요한
  `Result`(경로/지표)에 접근할 수 없다 — parity 실행에 필요한
  `torchscript_model_path` 등을 받으려면 어차피 워크플로우 함수를
  직접 불러야 한다.
- CLI 인자 파싱 자체의 정확성(`--epochs` 값이 올바르게 전달되는가 등)은
  `tests/scripts/test_train_imagefolder_args.py`/`test_train_imagefolder_cli.py`
  (§13, §14)가 이미 별도로 검증한다 — E2E가 이를 다시 검증할 필요가
  없다(사용자 질문 "E2E가 CLI argument wiring까지 검증해야 하는지"에
  대한 답: **아니오, CLI 테스트에 위임한다**).

---

## 13. 파일 변경 계획

**수정**:
- `scripts/run_imagefolder_training_e2e.py` — §12대로 재구성(워크플로우
  호출 + parity/gate만 남김, CLI 표면 축소).
- `tests/scripts/test_run_imagefolder_training_e2e_args.py`,
  `tests/scripts/test_run_imagefolder_training_e2e_resume_cli.py` — 대상
  플래그가 사라지므로 **삭제하고, 그 검증 의도를 `test_train_imagefolder_args.py`/
  `test_train_imagefolder_cli.py`로 이전**(§12). E2E 자체의 좁아진 CLI
  표면(`--model-json`만 남음)은 형제 스크립트들처럼 별도 테스트 파일 없이
  두는 것을 기본으로 하되, 재구성된 fresh+resume 두 번 호출 흐름 자체의
  스모크 테스트가 필요하면 `tests/scripts/test_run_imagefolder_training_e2e.py`
  (신규, 소규모)를 하나만 추가하는 것을 검토(§17 미결정).
- `README.md` — `train_imagefolder.py` production CLI 절 신설, 기존
  Phase 4G ImageFolder E2E 절을 재구성된 내용으로 갱신.

**신규**:
- `scripts/train_imagefolder.py`
- `src/image_ai_studio/training/imagefolder_workflow.py`
- `tests/training/test_imagefolder_workflow.py`
- `tests/scripts/test_train_imagefolder_args.py`
- `tests/scripts/test_train_imagefolder_cli.py`
- `docs/phase4h_production_training_cli_design.md`(본 문서)

**변경하지 않음(분석 결과 필요성을 찾지 못함)**:
- `config.py`/`loop.py`/`checkpoint.py`/`history.py`/`imagefolder_resume.py`/
  `torchvision_dataset.py` — 워크플로우 모듈이 이들의 기존 공개 API만
  호출하면 충분함을 §6~§8에서 확인. core training loop(`run_training()`)도
  변경 없음(사용자 지침 §14 "core training loop는 변경하지 않음"과 일치).
- `model_definition/*`/`export/*`/`parity/*`/C++ 코드 — production CLI가
  이들을 그대로(export는 그대로, parity는 아예 안 씀) 재사용.
- `scripts/run_training_e2e.py`/`run_real_training_e2e.py`/
  `run_resume_training_e2e.py` — ImageFolder 경로와 무관, §1-2에서 확인한
  중복은 이번 Phase 범위 밖(§14 "이번 Phase에서 제외할 것"에 명시된
  범위를 지킴).

---

## 14. 테스트 계획

### 14-1. `tests/training/test_imagefolder_workflow.py`(신규)

`tmp_path` + PIL fixture(기존 `test_imagefolder_resume.py` 패턴 재사용):

- fresh training 성공: `Result`의 모든 경로 파일이 실제로 생성됨,
  `history`/`test_loss`/`test_accuracy`가 채워짐.
- resume 성공: 2 epoch fresh + 2 epoch resume한 `Result.history`가
  연속 4 epoch 실행과 정확히 일치(기존 `test_imagefolder_checkpoint_resume_matches_continuous_run_exactly`와
  동일한 exact-equality 계약을 **워크플로우 함수를 통해서** 재확인 —
  Phase 4G의 함수 수준 테스트는 `checkpoint.py`/`loop.py`를 직접 호출했지만,
  이번엔 `run_imagefolder_training_workflow()`가 그 배선을 올바르게
  하는지까지 확인).
- `output_dir` 정책: 지정한 디렉터리 아래 고정 파일명으로만 생성됨,
  존재하는 파일은 덮어씀.
- **checkpoint가 best model이 아니라 current(마지막 epoch) model을
  저장하는지 확인(리뷰 반영, 재설계)**: `ImageFolderWorkflowResult`에는
  살아있는 model/state_dict가 없으므로(§6 설계를 그대로 유지 — 이 테스트
  때문에 Result API를 무겁게 만들지 않는다), **별도의 기준(reference)
  실행**을 만들어 비교한다:
  1. 같은 `model_spec`/`dataset`/`seed`/`TrainingConfig`로 `_prepare_resume()`을
     거치지 않는 **순수 `run_training()` 직접 호출**(워크플로우를 통하지
     않고 `loop.py`를 그대로 사용, `test_imagefolder_resume.py`의
     exact-resume 테스트와 동일한 저수준 패턴)로 연속 실행을 만들고,
     그 `model`(호출 후에도 마지막 epoch 가중치 그대로인 객체)의
     `state_dict()`를 기준값으로 확보한다.
  2. `run_imagefolder_training_workflow(request)`를 같은 설정으로 호출해
     `Result.checkpoint_path`를 얻는다.
  3. `payload = load_training_checkpoint(result.checkpoint_path)`로
     checkpoint를 로드한다.
  4. `payload["model_state_dict"]`의 각 텐서가 1번 기준 model의
     `state_dict()`와 `torch.equal()` 기준으로 정확히 일치하는지 확인한다.
  5. validation 결과를 결정론적으로 구성(`monkeypatch`로 `evaluate()`가
     초반 epoch에서만 낮은 loss를 반환하게 하는 식, `test_loop.py`의
     기존 monkeypatch 패턴 재사용)해 **best epoch가 마지막 epoch와
     다른 케이스**를 하나 만들고, 그 경우
     `payload["model_state_dict"] != payload["best_state_dict"]`(적어도
     한 텐서는 다름)임을 추가로 확인한다 — checkpoint가 정말 "최고 성능
     epoch"가 아니라 "마지막 epoch"를 저장한다는 사실을 이 케이스에서
     확실하게 구분해 고정한다.
- best model 평가: `test_loss`/`test_accuracy`가 `evaluate(best_model, ...)`와
  일치.
- **`test_result.json`(리뷰 반영, §9 계약 통일)**: `Result.test_result_path`가
  가리키는 파일이 실제로 생성됨 + 그 JSON 내용(`test_loss`/`test_accuracy`)이
  `Result.test_loss`/`Result.test_accuracy`와 정확히 일치.
- resume history 누적: `--epochs`가 추가 epoch 수로 정확히 반영됨(누적
  길이 검증).
- metadata mismatch 거부: 다른 dataset/ModelSpec으로 resume 시도 시
  `ValueError`.
- `export_torchscript=False`일 때 `torchscript_model_path`가 `None`이고
  실제 파일도 생성되지 않음(빈 `output_dir`에서 처음부터 끄는 경우).
- **stale artifact 정리(리뷰 반영, 신규)**: 같은 `output_dir`를 재사용하는
  시나리오로 설계 — (1) `export_torchscript=True`로 1회 실행 →
  `model.ts`/`model_metadata.json` 생성 확인, (2) **같은 `output_dir`**에
  `export_torchscript=False`로 다시 실행, (3) 1번에서 생긴 두 파일이
  실제로 삭제됐는지 확인(`Path.exists()`가 `False`), (4) 두 번째 실행의
  `Result.torchscript_model_path`/`torchscript_metadata_path`가 `None`인지
  확인. 추가로 `output_dir`에 워크플로우가 만들지 않은 임의 파일(예:
  `output_dir/user_notes.txt`)을 미리 넣어 두고, `export_torchscript=False`
  실행 후에도 그 파일이 그대로 남아 있는지 확인해 "고정 이름 두 파일만
  지운다"는 범위를 검증한다.
- TorchScript export가 성공적으로 되면 순수 Python
  `compare_outputs()`(`tests/training/test_train_export_parity.py`의
  기존 패턴 재사용)로 원본 모델 출력과 일치함을 확인 — **C++ 바이너리
  없이** export 정확성을 검증하는 이미 확립된 패턴.

### 14-2. `tests/scripts/test_train_imagefolder_args.py`(신규)

Phase 4G의 `test_run_imagefolder_training_e2e_args.py`와 같은 패턴
(scripts/를 sys.path에 추가해 `parse_args()` 직접 호출):

- **필수 인자 3개(리뷰 반영, 명시적으로 전부 확인)**: `--model-json`/
  `--dataset-root`/`--output-dir`는 전부 `required=True`다(§5). 셋 중
  하나씩만 빠뜨린 조합을 parametrize해서, `parse_args()` 호출 시
  argparse가 항상 `SystemExit`(비영 exit code)로 끝나는지 확인:

  ```python
  BASE_ARGS = ["--model-json", "m.json", "--dataset-root", "d", "--output-dir", "o"]

  @pytest.mark.parametrize("missing_flag", ["--model-json", "--dataset-root", "--output-dir"])
  def test_parse_args_requires_model_json_dataset_root_output_dir(missing_flag: str) -> None:
      args = [a for a in _drop_flag_and_value(BASE_ARGS, missing_flag)]
      with pytest.raises(SystemExit):
          e2e_script... # 실제로는 train_imagefolder 모듈의 parse_args()
      # SystemExit.code가 0이 아님을 함께 확인 (pytest.raises로 잡은 뒤 exc_info.value.code != 0)
  ```

  (실제 구현 시 헬퍼 이름/세부 구조는 자유, 핵심은 "3개 필수 인자 각각
  하나씩 빠진 경우"를 빠짐없이 parametrize로 커버하는 것.)
- 기본값(`--epochs`/`--batch-size`/`--learning-rate`/`--seed`/
  `--export-torchscript`) — 필수 3개 인자를 채운 최소 커맨드라인에서 확인.
- `--resume-from`/`--checkpoint-out` 파싱(Phase 4G 테스트에서 그대로
  이전).
- `--no-export-torchscript`로 `export_torchscript=False`가 되는지.

### 14-3. `tests/scripts/test_train_imagefolder_cli.py`(신규)

Phase 4G의 `test_run_imagefolder_training_e2e_resume_cli.py`와 같은 패턴
(`main(argv)` 실제 호출, `tmp_path` fixture, C++ 의존성 없음 — 애초에
production CLI는 C++를 호출하지 않으므로 monkeypatch도 불필요):

- fresh 학습 성공 → `output_dir`에 기대한 파일들이 전부 생성됨.
- resume 성공(checkpoint→resume) → 누적 epoch 수 확인(Phase 4G 패턴 이전).
- `--resume-from` 없는 경로 → traceback 없이 exit code 1(Phase 4G에서
  발견한 `OSError` 케이스 포함, 그대로 이전).
- metadata sidecar 없이 resume → exit code 1(Phase 4G 패턴 이전).
- `--no-export-torchscript` → `output_dir`에 `model.ts`가 생성되지 않음.

### 14-4. E2E regression (`scripts/run_imagefolder_training_e2e.py`)

**현재 저장소에서 실제로 재현된 regression anchor 수치(2026-08-02 확인,
`docs/phase4g_imagefolder_resume_design.md` 개정 2 보고 내용과 동일)**:

```text
5 epoch(3+2 split)  train_loss: 2.3903 -> 2.2862 -> 2.2040 -> 2.1784 -> 2.1509
                    val_loss:   2.2666 -> 2.2082 -> 2.1678 -> 2.1553 -> 2.1269
                    val_acc:    0.1000 -> 0.1600 -> 0.1800 -> 0.2200 -> 0.3000
best_epoch=5, best_val_loss=2.1269
test_loss=2.1859, test_accuracy=0.2600
TorchScript export: PASS
C++ CPU/CUDA parity: PASS
```

**정책(리뷰 반영, 옵션 B 채택)**: 이 수치를 스크립트 내부의 자동 실패
조건(hard assertion)으로 넣지 않는다. E2E는 지금처럼 이 값들을 그대로
**출력**만 하고, 자동 PASS/FAIL 판정은 `loss_improved`(방향성만 확인),
class mapping 일치, TorchScript export 성공, C++ parity처럼 환경에
안정적인 조건에만 건다. 정확한 소수점 값 일치는 **사람이 이 문서의
anchor 수치와 실제 출력을 수동으로 대조**해서 확인한다 — PyTorch
버전/CPU 아키텍처가 바뀌면 부동소수점 마지막 자리가 흔들릴 수 있는데,
그런 환경 차이를 스크립트 자동 실패로 처리하면 실제 회귀와 환경
잡음을 구분할 수 없어진다(옵션 A: 허용 오차를 둔 자동 gate는 오차
폭을 얼마로 잡을지가 또 다른 임의 결정이 되므로 채택하지 않았다).
워크플로우로의 이관이 순수 리팩터링이므로 수치 자체는 바뀌면 안 되고,
재구성 직후 실제로 재실행해 이 문서의 값과 정확히 일치함을 사람이
직접 확인했다(위 "최종 검증 결과" 참고) — 이후 이 스크립트를 다시
손댈 때도 같은 방식(재실행 + 수동 대조)으로 회귀를 확인할 것을
권장한다.

### 14-5. Phase 4G 회귀

fresh 3 epoch + checkpoint, resume 2 epoch, 동일 경로 resume-from/
checkpoint-out, metadata mismatch, missing checkpoint/metadata clean
failure — 전부 §14-1/§14-3에서 워크플로우/CLI 테스트로 그대로 재현됨을
확인.

### 14-6. 전체 회귀

`tests/training/`, `tests/scripts/`, 전체 `pytest`, 기존
`run_training_e2e.py`(Phase 4A/4B)/`run_real_training_e2e.py`(Phase 4C)/
`run_resume_training_e2e.py`(Phase 4F) — 전부 변경하지 않았으므로(§13)
기존과 동일한 결과가 나와야 함, 회귀 없음을 재실행으로 확인.

---

## 15. 구현 순서 (작은 단계)

1. `imagefolder_workflow.py` 구현 — `ImageFolderWorkflowRequest`/`Result`
   dataclass + `_prepare_resume()` + `run_imagefolder_training_workflow()`.
   순수 라이브러리 코드, CLI/E2E 미배선.
2. §14-1 워크플로우 단위/통합 테스트 작성/통과(fresh, resume-exact,
   output-dir, metadata mismatch 등).
3. `scripts/train_imagefolder.py` 구현(`parse_args()` + `main()`, 워크플로우
   호출 + §11 에러 처리).
4. §14-2/§14-3 CLI 테스트 작성/통과.
5. `scripts/run_imagefolder_training_e2e.py` 재구성(§12) — CLI 표면 축소,
   워크플로우 2회 호출(fresh+resume), 제거했던 게이트 중 loss/class
   mapping만 재도입(best model save/reload 재검증은 재도입하지 않고
   완전히 제거, §10/§12), parity 로직 이전.
6. 기존 `tests/scripts/test_run_imagefolder_training_e2e_*.py` 삭제(내용은
   3~4단계에서 이미 이전 완료).
7. §14-4 E2E regression 앵커 수치 재현 확인(수동 실행 + 필요 시 소규모
   자동 테스트).
8. README 갱신(production CLI 절 신설, E2E 절 갱신).
9. 전체 회귀(§14-6) — `tests/training/` + `tests/scripts/` + 전체 pytest +
   4개 기존 E2E 스크립트 + C++ CPU/CUDA parity.

---

## 16. 위험 요소

- **중복**: 워크플로우 추출로 대부분 없앴지만, `run_training_e2e.py`/
  `run_real_training_e2e.py`와의 TorchScript export/parity 블록 중복
  (§1-2에서 확인)은 이번 Phase에서 **의도적으로 남긴다** — 범위 밖.
  다음 Phase에서 이 세 스크립트의 공통 export/parity 로직까지 추출할지
  검토할 여지를 §17에 남긴다.
- **RNG 순서**: `_prepare_resume()`이 CPU RNG state를 직접 복원하지 않고
  호출자에게 반환값으로 넘기는 구조(§6, 리뷰로 수정됨)라, "DataLoader
  생성 → RNG 복원 → 즉시 `run_training()`"이라는 순서를 지키는 책임이
  `_prepare_resume()`(model/generator 준비)과
  `run_imagefolder_training_workflow()`(DataLoader 생성 + RNG 복원 +
  `run_training()` 호출) **두 곳에 걸쳐** 있다는 점은 여전히 위험
  요소다 — `run_imagefolder_training_workflow()`를 나중에 수정할 때
  `torch.set_rng_state(cpu_rng_state)`와 `run_training()` 사이에 실수로
  다른 코드를 끼워 넣을 수 있다. §14-1의 exact-equality 테스트가 이
  위험에 대한 안전망이다(순서가 깨지면 반드시 테스트가 실패함) — 이
  구조를 선택한 것 자체가 "함수 하나에 완전히 격리"보다 방어력이
  약간 낮아진 트레이드오프이므로, 구현 시 이 두 줄을 나란히 유지하는
  것을 코드 리뷰에서 특별히 확인할 것을 권장한다.
- **artifact 덮어쓰기**: 확인 없이 덮어쓰는 정책(§9)은 사용자가
  `--output-dir`를 재사용 실수로 잘못 지정하면 이전 결과를 조용히 잃을
  수 있다 — 완화책으로 CLI가 저장 직후 각 파일 경로를 명확히 출력하는
  것 외에 추가 안전장치는 넣지 않기로 결정했다(§9 근거 참고). 사용자가
  이 트레이드오프에 동의하지 않으면 `--output-dir`가 이미 존재하고
  비어있지 않을 때 경고만 출력하는 경량 방어를 추가하는 선택지가 있다
  (§17).
- **E2E 재구성 자체의 회귀 위험**: E2E를 "워크플로우를 두 번 호출"하는
  구조로 다시 쓰는 것 자체가 회귀 검증 스크립트의 신뢰도에 영향을 줄 수
  있는 변경이다 — §14-4에서 기존 수치와 **정확히** 일치하는지 확인하는
  것이 이 위험에 대한 직접적인 완화책이다.
- **테스트 이전 누락**: §13에서 삭제 대상으로 표시한 9개 테스트의 검증
  의도가 새 테스트 파일로 빠짐없이 옮겨졌는지 수동 대조가 필요하다 —
  구현 단계(§15의 4번)에서 항목별로 체크리스트를 만들어 확인할 것을
  권장.
- **stale artifact 삭제(리뷰 반영, 신규 위험)**: `export_torchscript=False`일
  때 워크플로우가 `output_dir`의 파일을 능동적으로 `unlink()`하는 것은
  이 워크플로우가 갖는 유일한 "삭제" 동작이다(다른 모든 저장은
  덮어쓰기일 뿐 삭제가 없음, §9). 삭제 대상을 워크플로우가 만든
  고정 이름 두 파일로 엄격히 제한하고 그 범위를 §14-1 테스트로
  고정했지만, 향후 파일명 상수가 바뀌거나 새 artifact가 추가될 때 이
  삭제 로직이 함께 갱신되지 않으면 "새 artifact는 안 지워지고 옛
  이름만 지워지는" 식의 불일치가 생길 수 있다 — 파일명을 하드코딩
  두 곳(저장 시점/삭제 시점)에 반복하지 않고 모듈 상수 하나로 관리할
  것을 구현 시 권장.

---

## 17. 미결정 사항

1. **재구성된 E2E 자체의 소규모 스모크 테스트를 pytest로도 추가할지** —
   현재 `run_training_e2e.py`/`run_real_training_e2e.py`는 별도
   `tests/scripts/test_run_*.py`가 없다(선례 없음). 하지만
   재구성된 `run_imagefolder_training_e2e.py`는 "fresh+resume 두 번
   호출"이라는 새로운 내부 구조를 갖게 되므로, 이 배선 자체가 깨지지
   않는지 확인하는 아주 작은 pytest(예: tiny synthetic ImageFolder로
   `main()`을 직접 호출, C++ 부분은 스킵되도록)를 추가할 가치가 있다 —
   추천: **추가한다**(구현 부담이 작고, §16 "E2E 재구성 자체의 회귀
   위험"을 pytest 수준에서도 빠르게 잡을 수 있음). 최종 결정은 구현
   단계에서 실제 소요를 보고 확정.
2. **`--output-dir`가 이미 존재하고 비어있지 않을 때 경고를 출력할지** —
   추천: **이번 Phase에서는 넣지 않는다**. 기존 프로젝트의 모든 저장
   경로가 동일하게 "조용히 덮어쓰기" 정책이라 일관성을 우선한다. 사용자
   피드백이 실제로 발생하면 그때 가벼운 경고(에러 아님)를 추가.
3. **`run_training_e2e.py`/`run_real_training_e2e.py`도 향후 같은 방식으로
   분리할지(공용 export/parity 헬퍼 추출)** — 이번 Phase 범위 밖으로
   명시적으로 확인됨(§13). §1-2/§16에서 발견한 중복은 다음 Phase의
   후보로만 기록해 둔다.
4. **`--seed`의 기본값을 CLI 도움말에서 얼마나 자세히 설명할지**(특히
   "resume 시 사실상 무시됨") — 추천: `--help` 문자열에 한 줄 요약 +
   README에 §5의 전체 설명을 링크하는 정도로 충분, 과도한 CLI 도움말
   장문화는 지양.
5. **class mapping/best-model-reload 자체 검증을 production에서 제거하는
   결정(§2-7)** — **해결됨(리뷰로 확정)**. class mapping 재검증은 E2E에
   재도입하는 것으로 확정(§12). best model save/reload 재검증은 E2E에도
   재도입하지 않고 완전히 제거하는 것으로 확정했다 — `Result`가 살아있는
   model을 반환하지 않는 한(설계 유지) "원본 vs 재로드" 비교가 불가능해,
   대안(같은 파일을 두 번 읽어 비교)은 회귀 가치가 없었기 때문이다(§10/
   §12).

이 설계는 §1의 실제 코드 분석(정확한 줄 번호 포함)에 근거하며, 사용자가
제시한 파일 후보를 대부분 그대로 채택하되, `tests/scripts/test_run_imagefolder_training_e2e_*.py`
두 파일의 삭제/이전이라는 추가 파급 효과를 §12/§13에서 명시적으로
드러냈다. 구현은 §15 순서대로 완료됐고, 문서 상단의 최종 검증 결과와
개정 내역이 그 결과를 반영한다.
