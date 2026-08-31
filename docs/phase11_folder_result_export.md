# Phase 11 Folder Result Export

Phase 10은 Inference tab의 Folder 모드에서 폴더 하나를 통째로 추론해
per-image 결과를 화면 테이블로 보여주지만, 그 결과를 파일로 저장하는
기능은 명시적으로 non-goal이었다(`docs/phase10_folder_inference.md` §8).
Phase 11은 그 위에 **화면에 보이는 바로 그 혼합 결과를 결정론적 JSON
또는 고정 스키마 CSV로 저장**하는 흐름을 추가한다 -- 기존 폴더 추론
경로와 Phase 7 portable artifact 계약을 **재사용/소비만** 하며, 추론을
다시 돌리거나 예측값을 재계산하지 않고, import/round-trip, 진행률, 취소,
preview, drag-and-drop, 병렬 실행, packaging/installer, TorchScript
현대화, CUDA 검증은 도입하지 않는다.

Phase 11 checkpoint 구성:

* **CP1** -- `src/image_ai_studio/inference/folder_result_export.py`:
  GUI/Qt를 전혀 모르는 framework-independent CSV/JSON export 계약
  (버전 붙은 JSON 스키마, 고정 CSV 헤더, 순서/escaping/숫자 직렬화/
  원자적 게시/실패 보존). `tests/inference/test_folder_result_export.py`.
* **CP2** -- `src/image_ai_studio/gui/inference_page.py`: `Folder
  Inference Results` 영역에 CSV / JSON export 액션 두 개. `_on_folder_finished`
  가 받은 정확한 `FolderInferenceResult` 객체를 CP1 경계로 넘긴다
  (테이블 텍스트 아님). `tests/gui/test_inference_page.py`의 Phase 11
  CP2 절.
* **CP3**(이 문서) -- 커밋된 CP1~CP2 구현이 실제 `MainWindow` +
  `InferencePage`의 비동기 폴더 경로에서 canonical portable bundle
  하나를 **실제 CPU backend + 실제 CP1 exporter**로 끝까지 소비하고,
  화면에 표시된 것과 파일로 나간 것이 일치함을 focused 통합 테스트로
  확인하며, 정확한 스키마/동작/호환성/한계/졸업 조건을 문서화한다.
  `tests/gui/test_folder_result_export_integration.py`.

이 문서 자신은 검증을 수행하지 않는다 -- Phase 8/10과 동일한 원칙으로,
§9의 자동화 테스트와 Phase 게이트(§10)가 실제 검증 주체다. 이 문서는
특정 실행의 pass 개수/시간/실측치/비용을 새로 주장하지 않는다.

## 1. JSON 스키마 (versioned)

`folder_result_export.folder_result_to_json_text(result)`는 다음 스키마의
UTF-8 텍스트를 돌려준다. 최상위 키 순서는 **선언 순서 그대로**이며
알파벳 재정렬을 하지 않는다(`sort_keys=False`). `indent=2`, 마지막에
개행 한 줄.

```json
{
  "format_version": 1,
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "items": [
    {
      "image_path": "<supplied path string>",
      "status": "success" | "failed",
      "predicted_class": "<str> | null",
      "confidence": "<float> | null",
      "probabilities": { "<class name>": "<float>", "...": "..." } | null,
      "inference_duration_seconds": "<float> | null",
      "error": "<str> | null"
    }
  ]
}
```

* `format_version` = `EXPORT_FORMAT_VERSION` = **1**. 필드 구성이나 의미가
  바뀌면(추가/삭제/재정의) 반드시 올린다. 값 1은 이 절이 문서화한
  스키마 그대로다.
* `total` / `succeeded` / `failed`는 `FolderInferenceResult`에서 파생되는
  관측값이며 항상 `total == succeeded + failed`, `total == len(items)`.
* `items`의 각 원소는 **정확히 위 7개 키를, 그 순서로** 가진다.
* `image_path`는 backend에 넘어간 경로 문자열 그대로다 -- display용
  파일 이름으로 줄이지 않는다(화면 테이블은 폴더 기준 상대 경로/파일
  이름만 보여주지만 export는 전체 경로를 보존한다).
* **성공 항목**: `predicted_class`/`confidence`/`probabilities`/
  `inference_duration_seconds`가 backend가 돌려준 `InferenceResult`
  값 그대로 채워지고 `error`는 `null`. 값을 재계산하지 않는다.
* **실패 항목**: 그 넷이 전부 `null`이고 `error`만 채워진다 -- 예측/
  confidence/probability/duration을 지어내지 않는다.
* `probabilities`는 중첩 object이며 key는 **class 이름 오름차순**으로
  정렬된다(dict 삽입 순서와 무관, 결정론적).

## 2. CSV 스키마 (고정 헤더)

`folder_result_export.folder_result_to_csv_text(result)`는 헤더 한 줄 +
항목당 정확히 한 데이터 행을 RFC4180 스타일 quoting으로 직렬화한다
(`csv.writer`, `lineterminator="\n"`). 헤더는 `CSV_COLUMNS` 상수와
동일한 **정확한 순서**(재정렬 금지):

```text
image_path,status,predicted_class,confidence,probabilities,inference_duration_seconds,error
```

* 데이터 행 수 == `len(result.items)`. 각 행은 헤더와 같은 길이 7.
* `status`는 `success` / `failed`(JSON과 동일한 소문자 값).
* **성공 행**: `predicted_class`는 문자열 그대로, `confidence`/
  `inference_duration_seconds`는 §3의 숫자 직렬화, `probabilities`는
  class 이름 오름차순으로 정렬한 **JSON 텍스트 문자열**(CSV 셀 하나에
  dict를 담을 수 없으므로), `error`는 빈 문자열.
* **실패 행**: `predicted_class`/`confidence`/`probabilities`/
  `inference_duration_seconds`가 전부 **빈 문자열**, `error`에 bounded
  오류 전문.
* 콤마/따옴표/개행/유니코드를 포함한 경로·오류·클래스 이름은
  `csv` 모듈이 자동으로 quote/escape하며 파싱 시 원래 값으로 복원된다.

## 3. 결정론적 순서와 형식

* **항목 순서**: JSON `items`와 CSV 데이터 행은 `FolderInferenceResult.items`
  의 순서를 **그대로** 보존한다 -- 그 순서는 Phase 10의 discovery 순서
  (폴더 바로 아래 지원 확장자 파일을 `Path.name` 오름차순 정렬)와
  동일하다. 화면 테이블 행 순서와도 1:1로 대응한다.
* **probabilities 순서**: JSON object의 key와 CSV 셀 안 JSON 텍스트의
  key 모두 class 이름 오름차순. 원본 `InferenceResult.probabilities`
  dict는 수정되지 않는다(정렬은 새 dict를 만들어 수행).
* **숫자 직렬화(로케일 독립)**: `confidence`와 `inference_duration_seconds`
  는 JSON에서 `float`를 그대로 담아 `json.dumps`가 처리하고(파이썬 `json`
  은 로케일을 참조하지 않고 `float.__repr__`와 같은 round-trip 알고리즘을
  쓴다), CSV에서는 같은 알고리즘인 `repr(value)`로 문자열화한다. 두
  표현의 숫자 문자열은 서로 일치하며 현재 로케일(콤마 소수점 구분자
  등)의 영향을 받지 않는다. 반올림/자리수 고정은 하지 않는다 -- 원래
  `float` 값을 왕복 가능하게 그대로 담는다.
* **반복 가능**: 같은 `FolderInferenceResult`에 대해 `folder_result_to_*_text`
  를 여러 번 불러도 바이트 단위로 동일한 텍스트가 나온다.
* JSON 텍스트는 항상 개행 한 줄로 끝난다. CSV는 마지막 행 뒤 개행 하나.

## 4. UTF-8 동작

* JSON은 `ensure_ascii=False`로 직렬화된다 -- 유니코드 경로/클래스 이름/
  오류 문자열이 `\uXXXX` escape 없이 그대로 담긴다. 파일은 UTF-8로
  기록되고 UTF-8로 다시 읽으면 동일한 문자열이 복원된다.
* CSV도 UTF-8 텍스트로 기록되며 유니코드 문자를 그대로 보존한다.
* BOM은 붙이지 않는다.

## 5. partial 실패 표현

* per-image 실패가 섞여 있어도 `FolderInferenceResult`는 **정상적으로
  완료된 배치**다(Phase 10 §3). export는 그 aggregate를 있는 그대로
  직렬화한다 -- 실패 항목은 §1/§2대로 예측 필드가 null/빈 문자열,
  `error`에 bounded 문자열이 담긴 채 **두 export 모두에 그대로 남는다**.
* per-image `error` 문자열의 상한은 Phase 10의 폴더 추론 계약이 정한다
  (`"{ExceptionType}: {message}"` 형태로 만든 뒤 500자 초과 시 잘라내고
  `"..."` 부착). export 계층은 그 문자열을 더 자르거나 늘리지 않고
  그대로 옮긴다. 화면 테이블은 그 오류의 **첫 줄만** 보여주지만
  CSV/JSON은 개행 포함 전문을 보존한다.
* 혼합 배치와 전부 실패한 배치 모두 `total`/`succeeded`/`failed` 수와
  행/항목이 정확히 유지된다. 성공 항목은 실패 항목의 존재와 무관하게
  값이 보존된다(깨진 이미지 뒤에 온 유효 이미지도 성공으로 남는다).
* 표시되거나 내보낸 결과가 조용히 누락되거나 날조되지 않는다 --
  항목 수 = 발견 이미지 수 = 테이블 행 수 = CSV 데이터 행 수 = JSON
  `items` 길이.

## 6. 목적지의 원자적 동작

파일 게시는 기존 `image_ai_studio.training.artifact_io.atomic_write_text`
primitive를 그대로 재사용한다:

* 목적지와 **같은 디렉터리**에 임시 파일을 만들어 전체 텍스트를 쓴 뒤
  `os.replace()`로 교체한다. 성공하면 목적지에는 이전의 완전한 파일이나
  새 완전한 파일만 존재하고, 반쯤 쓰인 파일이나 helper의 임시 파일은
  남지 않는다.
* 직렬화나 교체가 실패하면 **기존 목적지 파일은 바이트 단위로
  보존**되고, 임시 파일만 정리된 뒤 원래 예외가 그대로 전파된다 --
  재시도나 폴백 경로가 없다.
* `format`이 `SUPPORTED_EXPORT_FORMATS`(`"csv"`, `"json"`)에 없거나
  `path`가 이미 디렉터리로 존재하면(파일이 아닌 목적지), 파일시스템을
  건드리기 전에 `FolderResultExportError`를 던진다.
* `write_folder_result_export`는 어떤 경로에서도 입력
  `FolderInferenceResult`를 수정하지 않는다.

## 7. GUI enablement과 오류 동작

`InferencePage`의 `Folder Inference Results` 그룹에 **Export CSV** /
**Export JSON** 버튼 두 개가 요약 라벨·결과 테이블 아래에 있다.

* **enablement**: 두 액션은 (완료된 폴더 aggregate가 retained 되어 있고)
  **그리고** (단일 이미지/폴더 실행이 active가 아닐 때)에만 활성화된다.
  따라서 초기 상태, 새 실행이 시작될 때(비동기 작업 전에 retained
  aggregate를 버림), Running 중, fatal 폴더 실패 뒤, stale 폴더 결과에서
  다른 모드로 전환한 뒤에는 전부 비활성화된다. partial 실패가 섞인
  완료 배치는 여전히 완전히 export 가능하다.
* **직렬화 대상**: `_on_folder_finished`에 전달된 **바로 그
  `FolderInferenceResult` 객체**를 retain 했다가 그대로 CP1
  `write_folder_result_export` 경계로 넘긴다 -- 테이블 셀 텍스트를 다시
  파싱하지 않는다.
* **save 다이얼로그**: 각 액션은 `QFileDialog.getSaveFileName`을 열고
  결정론적 제안 파일명(`folder_inference_results.csv` /
  `folder_inference_results.json`)과 해당 확장자 필터를 제시한다.
  사용자가 취소하면(빈 경로) **no-op**이다 -- exporter를 부르지 않고
  상태 라벨도 바꾸지 않는다.
* **정확히 한 번 호출**: 다이얼로그가 경로를 돌려주면 retained
  aggregate와 그 경로를 CP1 exporter에 **정확히 한 번** 넘긴다.
* **성공 피드백**: 상태 라벨이 `Exported CSV: <파일명>` /
  `Exported JSON: <파일명>`으로 바뀐다. 표시된 결과 행은 건드리지
  않는다.
* **쓰기 오류**: `FolderResultExportError` 또는 `OSError`는 **GUI
  thread에서** 잡아 `Export failed: <첫 줄>` 형태의 bounded 메시지
  (200자 상한)로 상태 라벨에 보여준다 -- traceback을 노출하지 않고,
  크래시하지 않으며, 추론을 시작하지 않는다. 현재 결과는 그대로
  retained 되어 재시도할 수 있다.
* **lifecycle 무변경**: export는 QThread/worker/signal ownership을 전혀
  바꾸지 않는다. Phase 10의 비동기 폴더 lifecycle, overlap 방지,
  thread cleanup, `MainWindow`의 취소 없는 창 닫기 조율은 그대로다.
  자동 export는 없다 -- 항상 사용자가 버튼을 눌러야 한다.

## 8. 호환성 보장

* **single-image / folder 추론 public API 불변**: `InferenceRequest` /
  `InferenceResult` / `run_single_image_inference` /
  `FolderInferenceRequest` / `ImageOutcome` / `FolderInferenceResult` /
  `run_folder_inference`의 시그니처·필드·동작은 Phase 6B/10 그대로다.
  export 계층은 그 위에 얹히는 **순수 읽기 변환**이다.
* **controller / worker 계약 불변**: `FolderInferenceController` /
  `QtFolderInferenceWorker` / `InferenceController` / `QtInferenceWorker`
  의 상태 전이와 signal은 변경되지 않는다.
* **Phase 7 canonical 경로 재사용**: export는 아티팩트 경로 유도에
  관여하지 않는다. 폴더 모드는 여전히 Training Output Dir 아래 고정
  파일명 `best_model_state_dict.pt` / `class_mapping.json`을 쓰고, Model
  JSON 입력란이 비어 있으면 `<Training Output Dir>/model_definition.json`
  을 자동으로 유도한다.
* **legacy explicit Model JSON override 유지**: Model JSON 입력란에 값을
  넣으면 그 값이 항상 우선한다 -- `model_definition.json`이 없는(Phase 7
  이전) output directory도 폴더 모드로 추론하고 그 결과를 export할 수
  있다.
* **새 포맷/의존성 없음**: manifest/archive/packaging/signature/
  migration 포맷을 도입하지 않는다. CP1 모듈은 표준 라이브러리(`csv`,
  `io`, `json`)와 기존 저장소 primitive(`atomic_write_text`)만 쓴다 --
  새 서드파티 의존성이 없다.
* **launcher / pyproject / config 무변경**: `scripts/run_gui.py`,
  `pyproject.toml`(의존성·strict-marker registry 포함), 로컬 config는
  이 Phase에서 바뀌지 않는다.
* **strict-marker 정책 무영향**: Phase 11 테스트 모듈들은 새 pytest
  마커를 등록하거나 사용하지 않는다(Phase 9 marker registry hygiene와
  충돌 없음).

## 9. 검증 범위 -- CPU 자동화 / CUDA 조건부

이 checkpoint가 참조하는, 저장소에 실제로 존재하는 테스트만 나열한다.
특정 실행의 pass 개수/시간/실측치를 이 문서가 새로 주장하지 않는다.

### 9-1. 영구 자동화 테스트 (CPU에서 항상 실행, CUDA 가용성과 무관)

```text
tests/inference/test_folder_result_export.py                  (CP1, 구성된 FolderInferenceResult + tmp 경로)
    versioned JSON 최상위 스키마와 항목 필드 순서/성공·실패 값,
    고정 CSV 헤더와 행 값, image_path가 display name이 아닌 전체 경로,
    항목 순서 보존, class-name 정렬 probabilities, 로케일 독립 숫자
    직렬화, UTF-8/ensure_ascii=False/trailing newline, CSV quoting
    (콤마·따옴표·개행·유니코드), 혼합·전부 실패 aggregate의 정확한
    수/행, 반복 호출의 동일 텍스트, 원자적 write 성공(임시 파일
    잔존 없음)과 pre-replace 실패 시 기존 목적지 보존, 직렬화 실패
    시 목적지 미생성, 지원하지 않는 format / 디렉터리 목적지 거부,
    입력 result 불변, 모듈 소스에 Qt 미참조

tests/gui/test_inference_page.py (Phase 11 CP2 절)              (CP2, fake controller 주입 + patched 다이얼로그/exporter 경계)
    폴더 결과 영역의 CSV/JSON 액션 존재와 초기 비활성, 완료·부분
    성공 배치에서 delivered 인스턴스 retain + 두 액션 활성,
    각 액션이 CP1 exporter를 정확히 한 번 (result identity + 경로 +
    format)으로 호출, 실제 CP1 writer로 tmp 파일 기록, 다이얼로그
    취소 no-op, GUI thread의 bounded write 오류 + 재시도 가능,
    fatal 실패 / 새 실행 / 모드 전환 / rerun에서 stale export
    source 정리와 액션 비활성/복원, 최신 aggregate만 export

tests/gui/test_folder_result_export_integration.py             (CP3, 실제 CPU run_single_image_inference + 실제 CP1 exporter)
    실제 MainWindow + InferencePage 폴더 경로로 canonical bundle
    (Model JSON 비움 = auto-discovery) 하나를 소비: 지원 확장자
    이미지 3장(유효 2 + 지원 확장자를 가진 깨진 파일 1), 발견
    순서(이름 오름차순) 그대로 이미지당 한 행, 중간의 깨진
    이미지만 격리 실패하고 그 뒤 유효 이미지도 성공,
    Total/Succeeded/Failed = 3/2/1. 표시된 혼합 배치를 실제 Export
    CSV / Export JSON 버튼(다이얼로그만 patch)과 실제 CP1 exporter로
    저장하고, 파일로 나간 CSV/JSON이 delivered FolderInferenceResult
    를 verbatim 직렬화한 것이며 화면 테이블 각 행과 경로/상태/예측/
    confidence/probability/duration/error 의미가 동일하고 JSON 집계가
    정확히 3/2/1임을 확인. partial 실패가 두 export 모두에 남고,
    각 액션이 exporter를 retained aggregate로 정확히 한 번만 호출.
    같은 창의 두 번째 성공 폴더 실행이 완료 전에 stale export
    데이터를 비우고(액션 비활성 + source None), 새 실행만 중복 행/
    stale 오류/중복 exporter 호출 없이 내보내며 첫 실행의 export
    파일은 그대로 유지됨을 확인. 같은 모듈이 (a) legacy output_dir
    (model_definition.json 제거)에 대한 explicit Model JSON override
    폴더 실행 + export와 (b) 같은 bundle의 단일 이미지 모드
    auto-discovery 추론(폴더 결과/​export 상태 무영향)을 함께 회귀로
    고정한다.
```

전부 CPU만으로 완결된다 -- fake backend 또는 실제 CPU
`run_single_image_inference` + 실제 CP1 exporter만 쓰고, CUDA 설치 여부,
외부 모델 다운로드, 네트워크, 새 의존성, packaging 도구, 스크린샷
비교와 무관하게 실행되며 pytest 임시 디렉터리 밖의 저장소 아티팩트를
만들거나 바꾸지 않는다.

### 9-2. CUDA 조건부 범위

export 계층은 device 문자열을 해석하지 않는다 -- `InferenceResult` 값이
어떤 device에서 계산됐든 텍스트로 직렬화만 한다. CUDA inference forward
자체의 조건부 커버리지는 Phase 6이 확립한
`@pytest.mark.skipif(not torch.cuda.is_available())` 단일 이미지 테스트가
계속 담당한다 -- 이 Phase는 CUDA 전용 export 테스트를 추가하지 않으며,
이 문서는 폴더 결과 export가 실제 CUDA 환경에서 실행됐다고 주장하지
않는다.

### 9-3. 수동 실행 (자동화 테스트 아님)

```bash
python scripts/run_gui.py   # Inference tab -> Mode: Folder -> Run -> Export CSV / Export JSON
```

GUI에서 실제 폴더를 골라 추론하고 저장 다이얼로그로 파일을 눈으로
확인하는 경로는 자동화 스위트에 포함되지 않는다 -- §9-1의 테스트가
회귀 안전망이다.

## 10. Residual risks / non-goals

```text
import / round-trip 없음: CSV/JSON은 사람이 읽고 다른 도구에 넘기는
    출력 전용이다. 이 파일들을 다시 읽어 FolderInferenceResult로
    복원하는 loader는 없다.

단일 목적지 파일: 각 액션은 사용자가 고른 파일 하나에 쓴다. 다중
    폴더/작업 큐, 자동 파일명 순번, 디렉터리 일괄 export는 없다.

자동 export 없음: 배치가 끝나도 저장은 자동으로 일어나지 않는다 --
    항상 사용자가 Export CSV / Export JSON을 눌러야 한다.

진행률 / 취소 / preview / drag-and-drop 없음: export는 동기적이고
    작은 텍스트 쓰기 한 번이라 진행률 표시나 취소가 없다. Phase 10의
    폴더 추론 자체에도 취소/진행률/썸네일이 없다는 점은 그대로다.

packaging / installer 없음, TorchScript 현대화 없음, CUDA 검증 없음:
    Phase 6~10의 기존 non-goal 그대로이며 Phase 11에서 바뀌지 않는다.

세 휴대 산출물은 트랜잭션이 아님: export는 목적지 파일 하나 단위로만
    원자적이다. 여러 파일을 하나의 원자적 연산으로 묶지 않는다
    (Phase 8과 동일한 파일 단위 계약).

성능(모델 재로드): 이미지마다 backend가 model/artifact를 새로 로드하는
    Phase 10의 비용은 그대로다 -- export 계층은 이미 계산된 결과만
    직렬화하므로 이 비용에 관여하지 않는다.

공개 API / artifact 포맷 / 의존성 / launcher / goal / manifest / config /
    production source / 이 세 allowed 파일 밖의 테스트 assertion /
    이후 Phase 기능은 이 문서-and-통합 checkpoint에서 변경되지 않는다.
```

## 11. Phase 11 graduation criteria

```text
[ ] 세 개의 고정 required-test allowlist
    (phase11_cp1_folder_result_export_contract /
     phase11_cp2_folder_result_export_gui /
     phase11_cp3_folder_result_export_cpu_graduation)가 커밋된 구현에서 PASS
[ ] 세 checkpoint verifier가 모두 PASS 판정(각 checkpoint의 allowed
    files 밖 변경 없음, verifier mutation 없음)
[ ] 고정된 전체 프로젝트 harness를 커밋된 구현에서 정확히 한 번 최종
    실행해 PASS(protected-file / HEAD / staged-index / 범위 밖 worktree /
    harness mutation 없음)
```

**PHASE 11 COMPLETE**는 위 세 조건이 모두 충족될 때에만 성립하는 조건부
졸업 판정이다. 이 문서를 작성하는 행위 자체는 그 실행을 수행하지 않으며,
이 문서는 "검증이 이 문서 작성으로 완료되었다"고 주장하지 않는다 --
실제 판정은 그 테스트/verifier/harness 실행 결과에 달려 있다. Phase는 이
manifest에서 멈춘다: 이후 Phase를 실행하지 않고 add/commit/push/
pull-request도 수행하지 않는다.
