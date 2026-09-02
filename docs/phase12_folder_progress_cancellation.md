# Phase 12: 폴더 추론 진행률 및 협조적 취소

Phase 12는 기존 순차 폴더 추론에 관찰 가능한 진행률과 이미지 경계 기반의
협조적 취소를 추가합니다. Phase 10의 폴더 탐색·순차 실행 계약과 Phase
11의 CSV/JSON 결과 형식은 그대로 유지합니다.

## 사용자 흐름

1. Inference 탭에서 `Folder` 모드를 선택하고 portable artifact 출력
   디렉터리와 이미지 폴더를 지정합니다.
2. `Run Inference`를 누르면 발견된 전체 이미지 수와 완료 수가 진행률에
   표시되고, 각 이미지 결과가 발견 순서대로 누적됩니다.
3. 실행 중 `Cancel`을 누르면 상태가 `Cancelling...`으로 바뀝니다. 현재
   실행 중인 한 장의 forward는 끝까지 완료되고, 다음 이미지 시작 전에
   취소가 관찰됩니다.
4. 취소된 실행에서도 이미 처리된 성공 및 격리된 실패 행은 유지됩니다.
   `Export CSV` 또는 `Export JSON`으로 이 partial 결과를 저장할 수 있습니다.
5. 다시 실행하면 이전 진행률, 취소 플래그, 결과 행, 오류 및 export 원본이
   먼저 초기화됩니다.

## 진행률 계약

폴더 탐색이 성공하면 첫 이미지 실행 전에 `completed=0`인 초기 진행
스냅샷이 한 번 전달됩니다. 이후 이미지 하나의 성공 또는 격리된 실패가
결과에 추가될 때마다 스냅샷이 한 번 전달됩니다.

- `total`: 실행 시작 시 발견된 지원 이미지의 전체 수
- `completed`: 실제 처리가 끝난 이미지 수
- `succeeded` / `failed`: 완료된 항목 중 성공 및 격리된 실패 수
- 항상 `succeeded + failed == completed`이고 `0 <= completed <= total`입니다.
- `completed`는 실행 중 0부터 단조 증가하며 이미지 하나당 1씩 증가합니다.

Worker thread가 progress signal을 내보내고 `InferencePage`의 QObject bound
handler가 GUI thread에서 progress bar와 상태 문구를 갱신합니다. Widget은
worker thread에서 직접 변경하지 않습니다.

## 취소 경계와 terminal 순서

취소 요청은 thread-safe하고 여러 번 요청해도 같은 효과만 내는
idempotent 플래그입니다. 폴더 backend는 첫 이미지 시작 전과 각 이미지가
끝난 뒤 다음 이미지로 넘어가기 전에 이 플래그를 확인합니다.

이미 시작된 단일 이미지 forward는 강제로 중단하지 않습니다. 따라서
사용자가 Cancel을 누른 시점에 처리 중이던 이미지는 정상 성공하거나 해당
이미지만 실패한 결과로 기록된 뒤 취소가 적용됩니다. 취소가 관찰된 뒤에는
새 이미지 backend를 시작하지 않습니다.

취소 terminal은 정상 완료나 fatal failure와 구분됩니다. Worker는 실행당
`finished`, `cancelled`, `failed` 중 정확히 하나를 내보내며, 취소에는
완료된 `FolderInferenceResult`와 발견된 전체 수가 함께 전달됩니다. 그 뒤
thread quit, worker/thread `deleteLater`, page reference 정리가 기존 Qt
lifecycle 순서로 수행됩니다. `QThread.terminate()`, blocking `wait()`, busy
loop 또는 고정 sleep에 의존하지 않습니다.

## 처리 수와 partial 결과

취소 시 세 종류의 수를 구분해야 합니다.

- **processed**: 취소 전에 forward가 끝나 결과 항목이 만들어진 수
- **discovered**: 시작 시 폴더에서 발견된 지원 이미지의 전체 수
- **unprocessed**: `discovered - processed`; 취소가 관찰된 뒤 시작하지 않은 수

Partial `FolderInferenceResult.total`, `succeeded`, `failed`와 `items`는
processed 항목만 설명합니다. `discovered_total`과 unprocessed 수, 그리고
취소라는 terminal 의미는 `FolderInferenceCancelled` 경계와 UI 상태에
존재하며 `FolderInferenceResult` 필드를 확장하지 않습니다. 따라서 미처리
이미지를 성공이나 실패로 표시하지 않습니다.

## CSV/JSON 내보내기 호환성

취소된 partial 결과는 정상 완료 결과와 동일한 Phase 11 exporter로
내보냅니다. 화면 테이블을 다시 파싱하거나 예측값을 재계산하지 않고,
retained `FolderInferenceResult`를 그대로 직렬화합니다.

- CSV 헤더와 JSON `format_version: 1` 스키마는 변경되지 않습니다.
- 행과 JSON item에는 처리된 결과만 포함됩니다.
- `total`, `succeeded`, `failed`는 processed 기준입니다.
- 취소 여부, `discovered_total`, unprocessed 수는 format version 1 파일에
  포함되지 않습니다.

이는 기존 Phase 11 소비자와의 호환성을 지키기 위한 의도적인 경계입니다.
취소 파일만 보고 원래 발견 수나 미처리 수를 복원할 수는 없습니다.

## 재실행과 창 닫기

새 폴더 실행을 시작할 때 controller의 취소 상태와 page의 진행률, terminal
상태, 이전 결과·오류·export 원본을 초기화합니다. 따라서 취소 후 같은
창에서 다시 실행해도 stale cancel 요청이 새 실행에 전달되지 않고 진행률은
0부터 시작합니다. 완료 또는 취소 후 worker/thread 참조가 정리되어야 다음
실행이 가능합니다.

활성 폴더 추론 중 MainWindow 닫기를 확인하면 창은 즉시 파괴되지 않습니다.
`InferencePage.request_close()`가 협조적 취소를 요청하고 현재 forward와
terminal cleanup이 끝날 때까지 close를 보류합니다. 정리가 끝난 뒤 기존
close coordination이 닫기를 정확히 한 번 다시 시도합니다. 중복 close
요청은 중복 취소나 signal 연결을 만들지 않습니다. Idle 상태와 기존 단일
이미지 추론의 close 동작은 유지됩니다.

## 호환성

- `progress_callback`과 `should_cancel`은 선택적 keyword hook입니다. 이를
  사용하지 않는 기존 폴더 호출과 기존 one-argument injected backend는
  종전처럼 동작합니다.
- Phase 6B 단일 이미지 controller/worker 공개 API와 single-image UI 흐름은
  변경하지 않습니다.
- `FolderInferenceResult`, portable artifact, Phase 11 export schema 및
  launcher 계약은 변경하지 않습니다.
- Per-image 오류 격리와 파일 이름 기준 결정적 순차 처리를 유지합니다.

## CPU 통합 검증 범위

통합 검증은 테스트 임시 디렉터리에 canonical portable bundle과 작은 로컬
이미지를 만들고 실제 `MainWindow`, `InferencePage`, QThread worker,
`run_folder_inference`, `run_single_image_inference` CPU 경로를 연결합니다.
동기화는 worker 측 `threading.Event`와 GUI 측 상태 기반 `qtbot.waitUntil`을
사용합니다. 취소 전 실제 완료 이미지, 취소 시점의 현재 forward 완료,
이후 이미지 미시작, 단일 terminal, partial CSV/JSON, cleanup, 성공 rerun과
close-triggered cancellation을 프로그램 상태와 파일 내용으로 확인하도록
구성됩니다.

## 한계와 명시적 비범위

- 취소는 이미지 경계에서만 적용되며 현재 forward를 mid-forward abort하지
  않습니다. 이미지 한 장이 오래 걸리면 취소 완료도 그만큼 지연될 수 있습니다.
- 순차 처리만 지원하며 병렬 inference나 처리 우선순위는 제공하지 않습니다.
- CUDA 동작, 네트워크 다운로드, 외부 모델 서비스는 이 CPU 통합 범위에서
  검증하지 않습니다.
- 이미지 preview, 결과 이미지 렌더링, drag-and-drop, 재귀 폴더 탐색,
  packaging/installer 및 새로운 cancellation/stop 공개 API는 포함하지 않습니다.
- Screenshot 비교나 시간 기반 고정 sleep을 correctness 근거로 사용하지 않습니다.
- Export format version 1에는 취소 및 미처리 메타데이터가 없으므로 이를
  파일에 포함하려면 별도의 향후 스키마 설계가 필요합니다.
