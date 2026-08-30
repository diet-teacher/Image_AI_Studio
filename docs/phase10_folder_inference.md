# Phase 10 Folder Inference

Phase 6~7은 Inference tab에서 **단일 이미지 한 장**만 추론할 수 있었다
(`docs/phase6_final_integration.md`, `docs/phase7_portable_artifact_bundle.md`).
Phase 10은 그 위에 **폴더 하나를 통째로** 결정론적으로 추론하는 흐름을
추가한다 -- 기존 단일 이미지 inference core와 Phase 7 portable artifact
계약을 **재사용**할 뿐, 새 inference 경로/새 artifact 포맷/취소/병렬/
preview/migration은 도입하지 않는다.

Phase 10 checkpoint 구성:

* **CP1** -- `src/image_ai_studio/inference/folder_inference.py`: GUI/Qt를
  전혀 모르는 폴더 단위 contract(discovery + 조립 + per-image 오류 격리 +
  집계).
* **CP2** -- `src/image_ai_studio/application/folder_inference_controller.py`
  (framework-agnostic lifecycle) + `src/image_ai_studio/gui/qt_folder_inference_worker.py`
  (PySide6 `QThread` worker).
* **CP3** -- `src/image_ai_studio/gui/inference_page.py`: Single Image /
  Folder 모드 토글과 비동기 폴더 실행/결과 표시.
* **CP4**(이 문서) -- 커밋된 CP1~CP3 구현이 실제 `MainWindow` +
  `InferencePage`의 비동기 폴더 경로에서 canonical portable bundle 하나를
  **실제 CPU backend로** 끝까지 소비한다는 것을 focused 통합 테스트로
  확인하고, 정확한 동작/호환성/한계/졸업 조건을 문서화한다.

이 문서 자신은 검증을 수행하지 않는다 -- Phase 5D/6D/7/8과 동일한
원칙으로, §7의 자동화 테스트와 Phase 게이트(§9)가 실제 검증 주체다.
"PHASE 10 COMPLETE" 판정은 이 문서를 작성했다는 사실이 아니라, 커밋된
구현이 그 게이트를 통과한다는 조건에 대한 서술이다. 이 문서는 특정
실행의 pass 개수/시간/실측치/비용을 새로 주장하지 않는다.

## 1. 지원 이미지 확장자

`folder_inference.SUPPORTED_IMAGE_EXTENSIONS`(소문자, 점 포함)의 값은
정확히 다음 8개다:

```text
.bmp  .gif  .jpeg  .jpg  .png  .tif  .tiff  .webp
```

* 확장자 비교는 **대소문자를 구분하지 않는다** -- `photo.JPG`,
  `scan.PNG`, `frame.WebP`도 동일하게 포함된다(`Path.suffix.lower()`
  기준).
* 이 목록에 없는 확장자(`.txt`, `.csv`, `.pdf`, `.zip` 등)와 **확장자가
  전혀 없는 파일**은 discovery에서 제외된다.
* 이 목록은 파일 이름만 본다. 실제 파일 내용이 유효한 이미지인지는
  discovery 단계에서 검사하지 않는다 -- 확장자만 지원 목록에 있으면
  후보로 들어가고, 열 수 없는 파일이면 §3의 per-image 오류로 격리된다.

## 2. discovery와 정렬

`folder_inference.discover_supported_images(folder_path)`:

```text
- folder_path는 반드시 "존재하는 디렉터리"여야 한다.
    존재하지 않는 경로 / 디렉터리가 아닌 경로 -> FolderInferenceError
    (backend 호출 전에, 같은 입력에 대해 같은 메시지로).
- folder_path "바로 아래"의 항목만 본다. 하위 폴더로 재귀하지 않는다.
- 디렉터리는 제외한다(확장자를 가진 디렉터리 이름 "looks_like.png"도
  파일이 아니므로 제외).
- 지원 확장자(§1)를 가진 "파일"만 골라 파일 이름(Path.name) 기준
  오름차순으로 정렬해 돌려준다.
- 반환 순서는 파일시스템 열거 순서(Path.iterdir())와 무관하다 --
  같은 폴더에 대해 항상 같은 순서.
- 지원 이미지가 하나도 없으면 빈 리스트를 돌려준다(그 자체는 오류가
  아니다 -- fatal 판정은 run_folder_inference가 한다).
```

`folder_inference.run_folder_inference(request, backend=run_single_image_inference)`:

* discovery 결과가 **빈 리스트면** backend를 한 번도 부르지 않고
  `FolderInferenceError`(`no supported images in folder: ...`)를 던진다.
* 그 외에는 discovery 순서 그대로 이미지를 **한 장씩 순차적으로**
  처리하고, 결과 항목(`FolderInferenceResult.items`)의 순서는 discovery
  순서와 동일하다.
* `discover_supported_images()`를 두 번 불러도, `run_folder_inference()`를
  두 번 불러도 동일한 폴더에 대해 순서와 집계가 같다(반복 가능).

## 3. per-image 오류 격리

한 이미지에서 backend가 예외를 던지면:

* 그 이미지는 `result=None`, `error=<bounded 문자열>`인 실패
  `ImageOutcome`으로 기록된다.
* **오류 메시지는 상한이 있다**: `"{ExceptionType}: {message}"` 형태로
  만든 뒤 500자를 넘으면 잘라내고 `"..."`를 붙인다 -- backend가 아무리
  긴 예외 문자열을 던져도 aggregate 결과가 무한정 커지지 않는다.
* **이후 이미지 처리는 계속된다** -- 실패는 그 항목 하나에만 격리되고
  batch를 중단시키지 않는다.

`FolderInferenceResult`의 `total` / `succeeded` / `failed`는 `items`에서
파생되는 관측값이며 따로 저장되지 않는다(`total == succeeded + failed`가
항상 성립). per-image 실패가 섞여 있어도 이것은 **정상적으로 완료된
배치**다 -- `run_folder_inference()`는 예외를 던지지 않고 aggregate를
반환한다. controller/worker/page는 이를 `finished`로 다룬다. `failed`
경로(controller `failed` 상태, worker `failed` signal, page
`Failed: ...` status)는 **폴더 연산 자체가 치명적 예외를 던졌을 때만**
쓰인다(존재하지 않는 폴더, 지원 이미지 0장 등 `FolderInferenceError`).

## 4. 기존 single-image backend 재사용

`run_folder_inference()`는 discovery한 각 이미지 경로를 **공유
artifact/device/precision 값**과 합쳐 기존
`inference.single_image_inference.InferenceRequest`를 만들고(새 필드/새
포맷 없음), 기본값이 `run_single_image_inference`인 `backend`에 그대로
넘긴다.

```text
FolderInferenceRequest
    model_json_path / state_dict_path / class_mapping_path  (폴더 공유)
    folder_path                                             (discovery 대상)
    device / precision                                      (폴더 공유)

  -- 각 이미지마다 -->

InferenceRequest(
    model_json_path, state_dict_path, class_mapping_path,   (동일 값 재사용)
    image_path=<discovery가 찾은 경로>,
    device, precision,                                      (동일 값 재사용)
)
```

* 성공한 이미지의 `ImageOutcome.result`는 backend가 돌려준
  `InferenceResult` **객체 그대로**다 -- 폴더 계층은 예측 클래스/
  confidence/probabilities/duration을 다시 계산하지 않는다.
* `InferenceRequest` / `InferenceResult` / `run_single_image_inference`의
  시그니처와 동작은 Phase 6B 그대로이며 이 Phase에서 바뀌지 않는다
  (`tests/inference/test_folder_inference.py`가 두 dataclass의 필드
  집합이 불변임을 고정한다).
* 이미지마다 backend가 model/artifact를 새로 로드하는 비용은 그대로
  남는다 -- §8 참고.

## 5. UI 상태 전이 / lifecycle

`InferencePage`(Phase 10 CP3)는 `Mode:` 콤보로 **Single Image**(기본) /
**Folder**를 명시적으로 고른다. 폴더 실행은 `Folder`가 선택된 상태에서
Run Inference를 눌렀을 때만 시작된다.

```text
[Idle]
  Mode = Folder, Training Output Dir / (선택) Model JSON / Input Folder /
  Device / Precision 입력 후 Run Inference
    -> _build_folder_request()가 보이는 입력을 CP1 FolderInferenceRequest로
       스냅샷 (조립 실패 시 controller를 건드리지 않고 "Failed: ..." 표시)
    -> 별도 QThread + QtFolderInferenceWorker 쌍으로 비동기 실행
    -> 모든 입력 컨트롤 비활성화, status "Running", 두 결과 영역 clear
[Running]
  worker.finished(FolderInferenceResult)  (per-image 실패가 섞여도 여기로)
    -> 발견 순서대로 이미지당 한 행 채우기, 요약 라벨 갱신,
       단일 이미지 결과 영역은 clear 유지, status "Finished"
  worker.failed(str)                       (fatal 폴더 오류만)
    -> 폴더 결과 영역 clear, status "Failed: <첫 줄>"
  두 경우 모두: thread.quit -> worker.deleteLater(자기 신호에) ->
    thread.finished -> thread.deleteLater + _on_folder_thread_finished()
[Cleanup 완료]
  _folder_thread / _folder_worker = None, 모든 입력 컨트롤 재활성화
  close_pending이면 여기서 close_requested emit
```

* **결과 표시**: `Folder Inference Results` 그룹에 요약 라벨(`Total: N
  Succeeded: N  Failed: N`)과 테이블(`Image` / `Status` / `Predicted
  Class` / `Confidence` / `Error`)이 있다. 행 순서는 aggregate `items`
  순서(= discovery 순서) 그대로다. 성공 행은 `Status=Success`,
  실제 예측 클래스와 confidence, `Error=--`. 실패 행은 `Status=Failure`,
  `Predicted Class`/`Confidence=--`, `Error`에 bounded 오류의 **첫 줄**.
* **rerun**: 매 실행은 `setRowCount(0)` 후 처음부터 다시 채우므로,
  이어지는 실행이 이전 실행의 행을 남기거나 중복하지 않는다. 이전
  성공/실패 표시도 새 실행 시작 시 clear된다.
* **overlap 방지**: 단일 이미지/폴더 실행은 독립된 QThread/worker
  쌍이지만 하나의 status 라벨, 하나의 control-enable 헬퍼, 하나의
  `_close_pending` 플래그를 공유한다 -- 둘 중 하나가 active인 동안
  Run을 다시 눌러도 두 번째 실행은 시작되지 않는다.
* **canonical QThread 계약**: `worker.deleteLater()`는 반드시 worker
  자신의 `finished`/`failed`에 연결하고, `thread.deleteLater()`만
  `thread.finished`에 연결한다(Phase 5C stabilization,
  `docs/phase5c_training_gui_design.md` §18). signal은 반드시 실제
  `QObject` bound method에 connect한다 -- plain 함수/lambda에 connect하면
  emit이 일어난 worker thread에서 직접 실행된다(Phase 5B 실측).
* **창 닫기 조율**: `is_inference_active()`는 단일 이미지 `_thread`나
  폴더 `_folder_thread` 중 하나라도 cleanup이 끝나지 않았으면 True다.
  `request_close()`는 추론을 **취소하지 않는다** -- 자연 종료 후 해당
  thread cleanup 핸들러에서 `close_requested`를 한 번 emit한다
  (Phase 6C CP4 계약, 폴더 경로에서도 동일).
* **단일 이미지 모드는 변하지 않는다** -- 같은 request 조립, 같은
  `InferenceController`/`QtInferenceWorker`, 같은 결과 포맷, 같은
  overlap 방지/rerun/cleanup/close-defer 동작.

## 6. 공개 API / portable artifact 호환성

* **single-image public API 불변**: `InferenceRequest` /
  `InferenceResult` / `run_single_image_inference`
  (`inference/single_image_inference.py`)의 시그니처·필드·동작은 Phase
  6B 그대로다. 폴더 계층은 그 위에 얹히는 조립 layer일 뿐이다.
* **Phase 7 canonical 경로 재사용**: 폴더 모드의 아티팩트 경로 유도는
  단일 이미지 모드(`_build_request()`)와 동일하다 -- Training Output
  Dir 아래 고정 파일명 `best_model_state_dict.pt` / `class_mapping.json`,
  그리고 Model JSON 입력란이 비어 있으면
  `<Training Output Dir>/model_definition.json`을 자동으로 유도한다.
  차이는 단일 `image_path` 대신 `folder_path`뿐이다.
* **legacy explicit Model JSON override 유지**: Model JSON 입력란에 값을
  넣으면 그 값이 항상 우선한다 -- Phase 7 이전에 생성된(즉
  `model_definition.json`이 없는) output directory도 폴더 모드에서
  원본 Model JSON을 명시적으로 골라 그대로 쓸 수 있다.
* **새 포맷 없음**: manifest/archive/packaging/signature/migration
  포맷을 도입하지 않는다. Phase 7/8의 세 휴대 산출물을 소비만 한다.
* **pytest strict-marker 정책 무영향**: 이 Phase의 테스트 모듈은 새
  pytest 마커를 등록하거나 사용하지 않는다(Phase 9 marker registry
  hygiene와 충돌 없음).

## 7. 검증 범위 -- CPU 자동화 / CUDA 조건부

이 checkpoint가 참조하는, 저장소에 실제로 존재하는 테스트만 나열한다.
특정 실행의 pass 개수/시간/실측치를 이 문서가 새로 주장하지 않는다.

### 7-1. 영구 자동화 테스트 (CPU에서 항상 실행, CUDA 가용성과 무관)

```text
tests/inference/test_folder_inference.py                     (CP1, fake backend + tmp 이미지)
    discovery 순서/확장자 대소문자/하위 폴더 제외/빈 폴더,
    이미지 -> InferenceRequest 조립(artifact 공유), all-success/
    mixed/all-failure 집계, per-image 오류 bound(<=500자), 빈/누락/
    비지원-only 폴더가 backend 호출 전에 FolderInferenceError,
    반복 실행의 동일 순서/집계, dataclass 불변성

tests/application/test_folder_inference_controller.py        (CP2, fake backend)
    idle/running/finished/failed 전이, 단일 active run 강제,
    aggregate(혼합 포함) -> finished, fatal 예외 -> failed + 재던짐

tests/gui/test_qt_folder_inference_worker.py                 (CP2, fake backend + 실제 QThread)
    off-GUI-thread 실행, 정확히 한 번의 finished/failed,
    혼합 aggregate -> finished / fatal -> failed, 실제 QObject
    receiver에서 GUI thread 전달, canonical deleteLater 반복 cleanup

tests/gui/test_inference_page.py                             (CP3, fake controller 주입)
    Single/Folder 모드 토글과 위젯 가시성, 폴더 request 스냅샷,
    per-image 행/집계 표시, 혼합 완료 -> Finished, fatal -> Failed,
    rerun에서 중복 행 없음, overlap 방지, 컨트롤 복원, close-defer

tests/gui/test_folder_inference_integration.py               (CP4, 실제 CPU run_single_image_inference)
    실제 MainWindow + InferencePage 폴더 경로로 canonical bundle
    (Model JSON 비움 = auto-discovery) 하나를 소비: 지원 확장자
    이미지 3장(유효 2 + 지원 확장자를 가진 깨진 파일 1),
    발견 순서(이름 오름차순) 그대로 이미지당 한 행, 중간의 깨진
    이미지만 격리 실패하고 그 뒤 유효 이미지도 완료,
    Total/Succeeded/Failed = 3/2/1, 성공 행은 실제 예측/confidence,
    실패 행은 bounded 오류 첫 줄, 혼합 완료 뒤 Run/입력 복원 +
    worker/thread cleanup, 매 폴더 실행이 정확히 finished 1회 /
    failed 0회만 emit(실행마다 plain 관찰자로 확인),
    같은 창에서 이어지는 두 번째 폴더 실행이 중복 행/중복 신호/
    이전 오류 없이 성공(행 수 3 -> 2).
    같은 모듈이 (a) legacy output_dir(model_definition.json 제거)에
    대한 explicit Model JSON override 폴더 실행과 (b) 같은 bundle의
    단일 이미지 모드 auto-discovery 추론을 함께 회귀로 고정한다.
```

전부 CPU만으로 완결된다 -- fake backend 또는 실제 CPU
`run_single_image_inference`만 쓰고, CUDA 설치 여부, 외부 서비스,
네트워크, packaging 도구, 스크린샷 비교와 무관하게 실행되며 pytest
임시 디렉터리 밖의 저장소 아티팩트를 만들거나 바꾸지 않는다.

### 7-2. CUDA 조건부 범위

폴더 계층은 device 문자열을 각 `InferenceRequest`에 그대로 전달할 뿐
CUDA 실행을 요구하거나 가정하지 않는다. CUDA inference forward 자체의
조건부 커버리지는 Phase 6이 이미 확립한
`@pytest.mark.skipif(not torch.cuda.is_available())` 단일 이미지 테스트가
계속 담당한다 -- 이 Phase는 CUDA 전용 폴더 테스트를 새로 추가하지
않으며, 이 문서는 폴더 추론이 실제 CUDA 환경에서 end-to-end로
실행됐다고 주장하지 않는다.

### 7-3. 수동 실행 (자동화 테스트 아님)

```bash
python scripts/run_gui.py   # Inference tab -> Mode: Folder
```

GUI에서 실제 폴더를 골라 눈으로 확인하는 경로는 자동화 스위트에
포함되지 않는다 -- §7-1의 테스트가 회귀 안전망이다.

## 8. Residual risks / non-goals

```text
성능(모델 재로드): 이미지마다 backend가 model/state_dict/class mapping을
    새로 로드하고 model을 다시 build/eval한다 -- 폴더가 크면 이 반복
    로드 비용이 지배적이다. 한 번 로드한 model을 배치 내에서 재사용하는
    최적화는 이 Phase의 범위 밖이다.

순차 실행뿐: 이미지는 정확히 한 장씩, 발견 순서대로 처리된다. 병렬/
    멀티프로세스/배치 텐서 추론은 없다.

취소 없음: 폴더 실행에는 취소/중단 API가 전혀 없다(cooperative stop도,
    QThread.terminate()도 쓰지 않는다). 시작된 배치는 항상 끝까지
    진행되고, 창을 닫아도 자연 종료를 기다린다(§5).

진행률/preview 없음: batch 진행률 표시, 남은 개수, 처리 중 미리보기,
    이미지 썸네일은 없다. 결과는 배치가 끝난 뒤 한꺼번에 표시된다.

flat discovery뿐: folder_path 바로 아래만 본다. 하위 폴더 재귀,
    glob 패턴, 포함/제외 필터는 없다. 항목 판정은 Path.is_file()
    기준이라 심볼릭 링크는 따라간다 -- 지원 확장자를 가진 "파일을
    가리키는" 링크는 일반 파일과 똑같이 후보에 포함되고, "디렉터리를
    가리키는" 링크는 파일이 아니므로 제외되며 그 안으로 재귀하지도
    않는다.

내용 검증 없음: discovery는 확장자만 본다. 확장자만 지원 목록에 있고
    실제로는 열 수 없는 파일은 per-image 오류로 격리될 뿐, discovery
    단계에서 걸러지지 않는다.

결과 export 없음: 배치 결과를 CSV/JSON 등으로 저장하는 기능은 없다 --
    화면 테이블 표시가 전부다. 다중 폴더/작업 큐 관리도 없다.

packaging/installer 없음, TorchScript 현대화 없음, CUDA 검증 없음:
    Phase 6~8의 기존 non-goal 그대로이며 Phase 10에서 바뀌지 않는다.

공개 API / artifact 포맷 / 의존성 / launcher / goal / manifest / config /
    production source / 이 focused 통합 모듈 밖의 테스트 assertion /
    이후 Phase 기능은 이 문서-and-통합 checkpoint에서 변경되지 않는다.
```

## 9. Phase 10 graduation criteria

```text
[ ] 네 개의 고정 required-test allowlist
    (phase10_cp1_folder_inference_contract /
     phase10_cp2_folder_inference_controller_worker /
     phase10_cp3_folder_inference_page /
     phase10_cp4_folder_inference_cpu_graduation)가 커밋된 구현에서 PASS
[ ] 네 checkpoint verifier가 모두 PASS 판정(각 checkpoint의 allowed
    files 밖 변경 없음, verifier mutation 없음)
[ ] 고정된 전체 프로젝트 harness를 커밋된 구현에서 정확히 한 번 최종
    실행해 PASS(protected-file / HEAD / staged-index / 범위 밖 worktree /
    harness mutation 없음)
```

**PHASE 10 COMPLETE**는 위 세 조건이 모두 충족될 때에만 성립하는 조건부
졸업 판정이다. 이 문서를 작성하는 행위 자체는 그 실행을 수행하지 않으며,
이 문서는 "검증이 이 문서 작성으로 완료되었다"고 주장하지 않는다 --
실제 판정은 그 테스트/verifier/harness 실행 결과에 달려 있다. Phase는 이
manifest에서 멈춘다: 이후 Phase를 실행하지 않고 add/commit/push/
pull-request도 수행하지 않는다.
