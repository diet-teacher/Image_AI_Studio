# Phase 13: 단일 이미지 입력 미리보기

Phase 13은 Inference 탭의 `Single Image` 모드에 선택한 **입력 이미지**를
보여 주는 읽기 전용 미리보기를 추가합니다. 추론 파이프라인, 아티팩트
형식, 폴더 모드 동작은 바꾸지 않습니다.

## 입력 미리보기 vs 추론 결과

이 미리보기는 **사용자가 고른 원본 입력 이미지**를 그대로 보여 줄 뿐입니다.

- 예측 클래스, confidence, 확률 분포, 결과 이미지, saliency/overlay와는
  아무 관련이 없습니다. 그런 값은 여전히 `Inference Result` 영역에만
  나타납니다.
- 미리보기 갱신은 추론을 시작하지 않고, 모델·state_dict·class mapping을
  읽지 않으며, thread·worker·CUDA·네트워크를 건드리지 않습니다.
- `Folder` 모드에는 미리보기가 없습니다. 폴더 결과 표와 혼동되지 않도록
  모드를 바꾸면 숨기고 동시에 비웁니다.

## 사용자 흐름

1. Inference 탭에서 `Single Image` 모드를 선택합니다. `Input Image Preview`
   영역은 중립 placeholder(`No image selected`) 상태로 시작합니다.
2. `Browse...`로 이미지를 고르거나, `Input Image` 칸에 경로를 입력한 뒤
   Enter를 누르거나 포커스를 옮깁니다. 입력 중 매 글자마다가 아니라
   **확정 시점(`editingFinished`)에만** 미리보기가 갱신됩니다.
3. 유효한 로컬 이미지면 종횡비를 유지한 축소 미리보기가 표시됩니다.
4. 경로가 비면 placeholder로 돌아가고, 읽을 수 없는 경로면 간결한
   `Preview unavailable` 상태가 됩니다. 어느 경우에도 `Input Image` 칸의
   텍스트나 요청 매핑은 바뀌지 않습니다.
5. `Folder` 모드로 전환하면 미리보기가 사라지고 비워집니다. `Single Image`
   로 돌아오면 현재 `Input Image` 경로가 다시 반영됩니다.

## 디코딩 동작

- 디코딩은 Qt의 `QImageReader`로 로컬 파일에서 **동기적으로** 수행됩니다.
  `size()`로 원본 크기 metadata를 확인하고, orientation과 사용자 지정 preview
  상한을 함께 고려한 source-coordinate 크기를 `setScaledSize()`로 `read()` 전에
  요청합니다. 별도 thread/worker는 없습니다.
- `setAutoTransform(True)`이므로 EXIF orientation을 기록하는 형식
  (JPEG/TIFF 등)에서는 방향이 반영됩니다.
- 크기 metadata가 없거나 0 이하이면 원본 전체 디코딩으로 fallback하지 않고
  `Preview unavailable`로 안전하게 종료합니다.
- 원본 파일은 읽기 전용으로만 열리며, 바이트 내용과 수정 시각이 바뀌지
  않습니다.

## 스케일링 동작

- 표시 픽스맵은 문서화된 상한
  (`PREVIEW_MAX_SIZE`, 기본 320x320 device-independent 픽셀) 안으로만
  들어갑니다.
- **축소만** 합니다. 어느 한 축이라도 상한을 넘으면 종횡비를 유지한 채
  상한 안으로 줄이고, 이미 상한보다 작은 이미지는 원본 크기로 표시합니다
  (업스케일 없음).
- 90도/270도 및 mirror/flip+90도 orientation은 표시 좌표에서 축을 바꿔
  상한에 맞춘 뒤 source 좌표로 되돌려 decode 크기를 요청합니다. 따라서
  정사각형이 아닌 사용자 지정 상한에서도 회전 후 결과가 두 축 안에 듭니다.
- 종횡비는 항상 보존되므로 상한에 닿는 축은 최대 하나입니다. 따라서 큰
  이미지가 주변 레이아웃을 원본 크기로 늘리지 못합니다.

## 오류 동작

다음 입력은 전부 예외 없이 `Preview unavailable` 상태로 처리됩니다.

- 존재하지 않는 경로
- 파일이 아닌 경로(디렉터리 등)
- 지원하지 않는 형식
- 헤더만 그럴듯하고 본문이 깨진 파일

`load_image()`는 이런 경우 `False`를 반환하고 예외를 Qt 이벤트 루프로
전파하지 않습니다. 잘못된 경로 뒤에 유효한 경로를 주면 stale 픽스맵과
stale 오류 플래그가 모두 버려지고 정상 상태로 회복됩니다. `clear()`는
placeholder로 되돌리며 마찬가지로 이전 상태를 완전히 대체합니다.

## 모드 동작

- `Single Image` 모드에서만 미리보기 그룹이 보입니다.
- `Folder` 모드로 전환하면 그룹을 숨기고 `clear()`로 비웁니다(숨기기만
  하지 않습니다).
- `Single Image` 모드로 돌아오면 현재 `Input Image` 경로를 결정론적으로
  다시 반영합니다. 경로가 비어 있으면 placeholder를 보여 줍니다.
- 모드 전환은 폴더 export 원본 초기화, 진행률 초기화 등 기존 Phase 10~12
  동작을 그대로 유지합니다.

## 컴포넌트 상태 전이

`ImagePreview`는 항상 다음 세 상태 중 정확히 하나입니다.

- **placeholder**: 초기 상태와 `clear()` 이후. 픽스맵 없음, 중립 문구.
- **image**: 유효한 디코드. 축소된 픽스맵을 하나의 `QLabel`이 소유,
  상태 문구는 빈 문자열.
- **unavailable**: 위 "오류 동작"의 입력. 픽스맵 없음, 간결한 문구.

모든 전이는 결정론적으로 재현 가능합니다(`load_image()` 성공/실패,
`clear()`).

## 접근성과 자원 상한

- 미리보기 영역은 제목이 있는 `QGroupBox`(`Input Image Preview`)로
  라벨링됩니다. 상태는 색이 아니라 텍스트 문구(`No image selected` /
  `Preview unavailable`)로 전달되고, 라벨은 가운데 정렬 + 자동 줄바꿈
  입니다.
- 미리보기는 읽기 전용입니다. 입력 위젯이 아니며 값을 편집하거나
  submit하지 않습니다.
- 표시 픽스맵은 항상 상한 크기 이하이고, 그 픽스맵을 소유하는 것은 위젯의
  `QLabel` 하나뿐입니다. Qt format handler가 native scaled decoding을
  지원하면 원본 전체 allocation을 피하거나 줄일 수 있습니다. 다만 일부
  handler는 내부 fallback scaling을 사용할 수 있어 모든 형식·plugin의
  절대 peak-memory 상한을 보장하지 않습니다.
- 모듈 수준 픽스맵/이미지 캐시가 없고, 이전 픽스맵은 다음 로드나
  `clear()` 시 교체됩니다. 위젯은 background worker/thread를 만들지
  않습니다.
- 로컬 decode는 여전히 GUI thread에서 동기적으로 실행되므로 느린 storage나
  decoder에서는 짧은 UI 지연이 생길 수 있습니다.

## 호환성

- `_build_request()`의 아티팩트 경로 유도(state_dict / class mapping 고정
  파일명, `model_definition.json` 자동 탐색 + 명시적 Model JSON override)와
  검증 경로는 변경되지 않습니다. 미리보기는 `Input Image` 칸 텍스트를 다시
  쓰지 않습니다.
- Phase 6 단일 이미지 async QThread lifecycle(Running/Finished/Failed,
  controls disable/enable, overlap 방지, worker/thread cleanup, rerun)은
  변경되지 않습니다.
- Phase 10~12 폴더 탐색·순차 실행, 진행률 스냅샷, 이미지 경계 협조적
  취소, Phase 11 CSV/JSON `format_version` 1 export 스키마는 변경되지
  않습니다.
- `MainWindow` close coordination(`is_inference_active()`,
  `request_close()`, `close_requested`)과 idle/단일/폴더 close 동작은
  변경되지 않습니다.
- 공개 추론 API(`InferenceRequest`/`InferenceResult`,
  `run_single_image_inference`, `run_folder_inference`,
  `FolderInferenceResult`)와 portable artifact 형식은 소비만 하며 바꾸지
  않습니다.
- `image_preview` 모듈은 inference/training/application 패키지, `torch`,
  `numpy`, 네트워크 모듈을 import하지 않습니다.

## 자동 검증 근거

- **CP1 -- 컴포넌트 단독** (`tests/gui/test_image_preview.py`):
  placeholder/image/unavailable 상태, `size()`/orientation 확인과
  `setScaledSize()`가 `read()`보다 앞서는 순서, landscape/portrait 및
  회전된 non-square 상한의 source-coordinate decode 요청, unknown-size
  fail-closed, 축소 전용 종횡비 스케일링, 커스텀 상한, EXIF orientation
  (Pillow가 있을 때만, `importorskip`), clear/reload 전이, 원본 파일 비수정,
  inference/무거운 의존성 미import, 모듈 수준 픽스맵 캐시 부재,
  background thread 부재.
- **CP2 -- 페이지 배선** (`tests/gui/test_inference_page.py`의 `Phase 13
  CP2` 절): 라벨된 미리보기 하나가 단일 이미지 입력 영역에 위치, Browse
  액션과 committed manual path가 모두 갱신, 빈 경로 -> placeholder,
  읽을 수 없는/깨진 경로 -> unavailable(예외 없음), 요청 매핑/검증 불변,
  `Folder` 전환 시 hide+clear, 복귀 시 재반영, 미리보기 작업이 inference
  thread/worker/scan/watcher를 만들지 않음.
- **CP3 -- 실제 MainWindow 졸업**
  (`tests/gui/test_inference_preview_integration.py`): 실제 `MainWindow` +
  `InferencePage` 객체 그래프에서 미리보기 인스턴스가 정확히 하나이고
  라벨링됨, Browse와 committed manual path 갱신 + 종횡비 상한 준수,
  없는 경로 -> 깨진 경로 -> 유효 경로 회복, Single Image -> Folder ->
  Single Image 왕복 후 stale 미리보기 없음, **가짜** 비동기 단일 이미지
  inference lifecycle과 공존하면서도 위젯이 중복 생성되지 않음.
  주입된 `InferenceController` 백엔드는 실제 forward 없이 canned
  `InferenceResult`만 반환하고(필요 시 `threading.Event` gate), 미리보기
  디코드만 실제 Qt 로컬 디코드입니다. 관찰은 위젯 상태 플래그,
  `pixmap()` 크기, `findChildren` 같은 상태 기반과 `threading.Event` /
  `qtbot.waitUntil`뿐이며 CUDA, 네트워크, 외부 다운로드, screenshot 비교,
  고정 sleep, file watcher, 새 의존성을 쓰지 않습니다.

자동 검증은 특정 예측값이나 렌더링된 픽셀을 재계산/비교하지 않습니다.
이 문서는 통과한 테스트 수를 명시하지 않습니다.

## 수동 확인 항목

다음은 자동 검증 대상이 아니며 필요할 때 수동으로 확인합니다.

- 실제 데스크톱에서의 시각적 렌더링 품질과 HiDPI 디스플레이 스케일링.
- 실제 카메라 사진의 다양한 형식에서 EXIF orientation 반영 결과.
- 매우 큰 원본 이미지를 반복 로드할 때의 실제 메모리 사용 추이.
- format handler별 native scaled decode 여부와 OS 수준 peak memory.
- 실제 파일 선택 다이얼로그와 포커스 이동을 통한 상호작용 감각.

## 한계

- 미리보기는 로컬 파일 시스템 경로만 지원합니다. URL, 원격 스토리지,
  아카이브 내부 항목은 대상이 아닙니다.
- 파일이 바뀌어도 자동으로 다시 읽지 않습니다. 갱신은 Browse 또는
  committed manual edit 같은 명시적 사용자 동작에서만 일어납니다
  (file watcher/polling 없음).
- Qt 빌드가 디코드할 수 없는 형식은 `Preview unavailable`로만 표시되며,
  별도 디코더를 번들하지 않습니다.
- 미리보기 크기는 고정 상한이며 사용자 조절 UI(줌/팬)는 없습니다.

## 명시적 비범위

이번 Phase는 다음을 추가하지 않습니다.

- drag-and-drop 입력
- 폴더 썸네일 그리드
- 이미지 편집(크롭/회전/필터 등)
- 추론 결과 이미지 렌더링 또는 overlay
- EXIF 메타데이터 편집
- 재귀 폴더 탐색
- packaging / installer
- TorchScript 마이그레이션
- CUDA inference
- 네트워크 기능
