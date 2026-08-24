# Phase 6 Final Integration Validation & Graduation

Phase 6A~6D 전체를 아우르는 최종 정리 문서다. Phase 6A는 architecture
설계(`docs/phase6a_inference_architecture.md`), Phase 6B는 inference
core + application controller + Qt worker 구현
(`docs/phase6b_inference_core_design.md`), Phase 6C는 `InferencePage` +
`MainWindow` 두-tab 통합(Training/Inference)을 추가했다. Phase 6D는
Phase 5D와 동일한 원칙으로 **verification-first**다 -- 새 기능을
추가하지 않고, Phase 6A~6C가 만든 것이 실제 사용자 관점에서 "방금 학습한
모델로 바로 추론"까지 하나의 세션 안에서 정상 동작하는지, 그리고
training/inference가 동시에 활성인 상태에서 창을 닫는 경로가 안전한지를
검증한다.

이 문서 자신은 검증을 수행하지 않는다 -- 아래 §8의 자동화 테스트와
Phase 게이트(고정 integration test 목록 + 전체 project harness)가 실제
검증 주체다. §10의 "PHASE 6 COMPLETE" 판정은 이 문서를 작성했다는 사실이
아니라, 커밋된 상태가 그 게이트를 통과한다는 조건에 대한 서술이다.

## 1. Architecture

```text
InferencePage / MainWindow            (GUI, PySide6, Phase 6C)
        |  _build_request() -> build_inference_request()
        v
InferenceController                   (application, framework-agnostic, Phase 6B)
        |  QtInferenceWorker.run() 안에서 begin_run()+run() 호출
        v
QtInferenceWorker (QObject) / QThread  (GUI-framework worker boundary, Phase 6B)
        |  블로킹 호출
        v
run_single_image_inference()           (inference core public entrypoint, Phase 6B)
        |
        v
기존 training core의 재사용 가능한 public API
    load_model_spec / validate_model_spec / build_model / load_state_dict
    build_transform / load_class_mapping / require_matching_num_classes
    training/device.py: _validate_device / _validate_precision_device_compatibility
```

의존 방향은 Phase 5의 training 스택과 동일하게 위에서 아래로만 흐른다.
`inference` 패키지는 GUI/Qt를 전혀 모르고, `InferenceController`는
PySide6를 전혀 import하지 않는다(PySide6와
`image_ai_studio.application`을 함께 import하는 유일한 지점은
`src/image_ai_studio/gui/qt_inference_worker.py`, Phase 5B의
`qt_training_worker.py`와 동일한 경계 원칙).

**inference 전용 model builder/transform은 새로 만들지 않았다** --
`run_single_image_inference()`은 training core가 이미 갖고 있는 public
함수 5개를 정해진 순서로 호출할 뿐이다(위 도식의 마지막 줄, Phase 6A §3).
`TrainingController`/`QtTrainingWorker`(Phase 5)는 Phase 6에서 한 줄도
수정하지 않았다 -- `InferenceController`/`QtInferenceWorker`는 그 옆에
새로 추가된 독립적인 클래스이며, `stopping` 상태나
`threading.Event` 기반 cooperative stop이 없는 더 단순한 4-state machine
(`idle/running/finished/failed`)이다(Phase 6A §8 -- single-image
inference는 원자적 forward pass 한 번이라 중간에 멈출 지점이 없다는
판단에 근거).

## 2. Phase별 역할

| Phase | 추가한 것 | 건드리지 않은 것 |
|---|---|---|
| 6A | architecture 조사/설계(canonical inference path, preprocessing/device/precision 재사용 조사, artifact bundle 절충안, close coordination 설계 등), **코드 없음** | 전체 |
| 6B | `inference/single_image_inference.py`, `application/inference_controller.py`, `gui/qt_inference_worker.py`, `training/device.py`(순수 이동 리팩터), `load_class_mapping()`에 최소 구조 검증 추가 | GUI 화면, training core의 동작(리팩터/검증 추가는 정상 artifact 기준 무변경) |
| 6C | `gui/inference_page.py`(`InferencePage`), `gui/main_window.py`를 `QTabWidget` 2-tab(Training/Inference) + 중앙집중형 close coordination으로 확장 | inference core/controller/worker의 public API(6B 산출물을 그대로 사용) |
| 6D | **없음**(verification-only, Phase 5D와 동일 원칙) -- Training→Inference 연속 시나리오와 동시 활성 close 시나리오를 실제(fake 아닌) backend로 검증하는 integration test 2개 모듈 추가 | inference core, `InferenceController`, `QtInferenceWorker`, `InferencePage`/`MainWindow`의 기능 계약 |

## 3. Runtime flow (single-image inference)

```text
GUI input(위젯 snapshot)
    -> InferencePage._build_request() -> build_inference_request()
       (training output dir + 고정 파일명 2개를 조합, 아래 §4)
    -> InferenceRequest 생성(semantic validation은 device/precision
       검증 로직과 inference core 자신의 artifact 로드 단계가 수행)
    -> 실패 시: controller.begin_run()을 호출하지 않고 GUI에만
       "Failed: ..." 표시(controller.state는 idle로 유지)
    -> 성공 시: QThread + QtInferenceWorker 생성, moveToThread, signal 연결
    -> thread.start()
       -> (worker thread) QtInferenceWorker.run()
          -> controller.begin_run() (InferenceAlreadyRunningError면 failed emit)
          -> controller.run(request)
             -> run_single_image_inference(request) 블로킹 호출
                (model 재구성 -> preprocessing -> forward -> softmax/argmax)
          -> 성공: state="finished" 전이 후 finished.emit(result)
          -> 예외: state="failed" 전이 후 failed.emit(message+traceback)
    -> (GUI thread, queued) InferencePage._on_finished/_on_failed가
       위젯을 갱신(predicted class/confidence/probabilities/duration,
       또는 실패 메시지 첫 줄)
    -> cleanup: worker.finished/failed -> thread.quit(), ->
                worker.deleteLater(); thread.finished ->
                thread.deleteLater() + InferencePage._on_thread_finished()
                (`_thread`/`_worker`를 None으로 되돌리고 controls 재활성)
```

Training과의 핵심 차이: **progress signal이 없다**(단일 이미지는 진행률
개념이 없음, `finished`/`failed` 두 signal만 존재)와 **stop 요청이
없다**(원자적 작업이라 취소 지점이 없음, 아래 §7).

## 4. Training-to-Inference artifact flow

Phase 6A §1/§11이 확인한 사실을 그대로 따른다: 학습 산출물
(`run_imagefolder_training_workflow()`가 `output_dir`에 쓰는 것)에는
`best_model_state_dict.pt`(고정 파일명, 항상 생성)와
`class_mapping.json`(고정 파일명, 항상 생성)은 있지만, 학습에 쓰인
model definition JSON은 `output_dir`에 복사/보존되지 않는다(사용자가
학습 시 지정한 입력 파일일 뿐).

`InferencePage`는 이 사실을 그대로 반영한 **2-picker** 구성이다
(`src/image_ai_studio/gui/inference_page.py`):

```text
Training Output Dir: [디렉터리 선택]
    -> best_model_state_dict.pt / class_mapping.json을 고정 파일명으로
       자동 유도(InferencePage._build_request()가 output_path / 파일명
       으로 경로를 조립 -- 별도 탐색 로직 없음, 단순 경로 결합)
Model JSON: [파일 선택]
    -> 학습에 쓴 model definition JSON을 사용자가 직접 지정
       (output_dir에 없으므로 별도 선택 필수)
Input Image: [파일 선택]
    -> 추론할 단일 이미지
```

3개 artifact(state_dict/model JSON/class_mapping)를 각각 개별
picker로 요구하는 대신, training 산출물 구조를 활용해 2개 picker로
줄인 절충안이다(Phase 6A §11이 검토한 Option A/B/C 중 A+B 절충 --
새 manifest/bundle 포맷은 도입하지 않았고, training 코드도 이
목적으로 수정하지 않았다). `state_dict`/`class_mapping` 파일이 없거나
손상된 경우의 오류는 GUI가 재검증하지 않고 `run_single_image_inference()`
내부의 기존 로드 함수(`load_state_dict`/`load_class_mapping` 등)가
던지는 예외를 그대로 `failed` 상태로 전달한다(Phase 5C의 "GUI는
재검증하지 않는다" 원칙을 그대로 계승, Phase 6A §10).

## 5. Canonical user workflow

이 checkpoint가 문서화하는, 실제 사용자가 Training tab에서 Inference
tab까지 이어서 쓰는 표준 흐름이다(`docs/phase6a_inference_architecture.md`
§9의 화면 스케치와 §11의 artifact 절충안, Phase 6C 구현, Phase 6D의
`tests/gui/test_training_inference_integration.py`가 이 흐름 전체를
실제 backend로 검증한다):

```text
1. Training tab에서 새로 학습하거나, 이미 완료된 학습의 output
   directory를 그대로 재사용한다
   -- 새로 학습: Model JSON/Dataset root/Output directory 지정 후 Start,
      완료되면 output_dir에 best_model_state_dict.pt/class_mapping.json
      생성(§4)
   -- 기존 재사용: 이전에 완료된 학습이 남긴 output_dir을 그대로 사용
      (같은 세션일 필요 없음 -- 디스크에 파일만 있으면 됨)
2. Inference tab으로 전환한다(같은 MainWindow, 같은 세션 -- 새 창을
   띄우지 않는다)
3. Training Output Dir에 위 output_dir을 선택한다(best_model_state_dict.pt
   /class_mapping.json을 고정 파일명으로 자동 유도, §4)
4. Model JSON에 그 학습에 실제로 사용된 model definition JSON을
   선택한다(output_dir에 없으므로 별도 지정 필수)
5. Input Image에 추론할 단일 이미지를 선택한다
6. Device(cpu/cuda/cuda:N)와 Precision(fp32/fp16/bf16)을 선택한다
   (fp16/bf16은 CUDA 전용 -- CPU와 조합하면 명확한 오류로 거부, §4)
7. Run Inference를 누른다 -- GUI thread를 막지 않고 QThread에서 실행되며
   (§3), 진행 중에는 입력 필드/버튼이 비활성화된다
8. 완료되면 결과 영역에서 결정론적으로 계산된 필드를 확인한다:
   Predicted Class, Confidence(%), Class Probabilities(class별 %,
   class 이름 기준 정렬), Duration(ms) -- 재계산 없이 InferenceResult의
   기존 필드를 그대로 표시한다(값 자체는 model/입력에 따라 달라지므로
   이 문서가 특정 수치를 예단하지 않는다)
9. 작업이 진행 중인 상태(학습만, 추론만, 또는 둘 다)에서 창을 닫으려
   하면 확인 다이얼로그가 뜨고, 동의하면 각 활성 작업이 안전하게
   끝날 때까지 창이 실제로는 닫히지 않고 기다린다(training은
   cooperative stop, inference는 취소 없이 자연 종료 대기, §7) --
   진행 중이던 작업의 결과/부분 산출물을 잃지 않고 창을 닫을 수 있다
```

## 6. QThread ownership / `deleteLater` 계약 (`QtInferenceWorker`)

```python
self._thread = QThread(self)                       # parent=InferencePage
self._worker = QtInferenceWorker(self._controller, request)
self._worker.moveToThread(self._thread)
self._thread.started.connect(self._worker.run)
self._worker.finished.connect(self._on_finished)    # QObject bound method
self._worker.failed.connect(self._on_failed)         # QObject bound method
self._worker.finished.connect(self._thread.quit)
self._worker.failed.connect(self._thread.quit)
self._worker.finished.connect(self._worker.deleteLater)   # worker 자신의 신호에!
self._worker.failed.connect(self._worker.deleteLater)     # 〃
self._thread.finished.connect(self._thread.deleteLater)
self._thread.finished.connect(self._on_thread_finished)
```

Phase 5C stabilization이 확정한 계약(`worker.deleteLater()`는 반드시
worker 자신의 `finished`/`failed`에 연결, `thread.deleteLater()`만
`thread.finished`에 연결)을 그대로 재사용한다 -- 새로 설계하지 않고
검증된 패턴을 그대로 베꼈다(Phase 6A §9의 명시적 결정). Phase 6B
stabilization 라운드에서 이 wiring 자체는 처음부터 올바르게
구현되었음을 확인했고(`docs/phase6b_inference_core_design.md` §10-1),
드물게 재현된 native abort의 원인은 production 코드가 아니라 테스트
코드(`qtbot.waitSignal()`을 `deleteLater()`가 연결된 signal에 임시로
붙였다 뗐다 하는 경합)였다 -- 해당 절 참고.

`InferencePage`는 `_thread`/`_worker`를 `_on_thread_finished()`에서
`None`으로 되돌린다(`TrainingPage`는 그렇게 하지 않는다 -- 이 차이는
Phase 6D의 두 integration test 모듈이 `_thread_cleaned_up()` helper로
양쪽을 모두 처리하도록 명시적으로 흡수한다, 아래 §8).

## 7. MainWindow close coordination

`MainWindow`(`src/image_ai_studio/gui/main_window.py`)는 Phase 5C의
단일-page close 처리를 Training+Inference 두 page를 조정하는 중앙집중형
로직으로 확장했다(Phase 6A §9가 설계한 그대로):

```text
closeEvent()
    -> 이미 _close_pending이면: event.ignore()만 하고 아무 것도
       다시 하지 않는다(다이얼로그/시그널 연결 중복 방지)
    -> training_active = training_page.is_training_active()
       inference_active = inference_page.is_inference_active()
    -> 둘 다 False면: event.accept() (즉시 종료)
    -> 하나라도 True면: 확인 다이얼로그(Yes/No) -- 정확히 1회만 표시
       No  -> event.ignore(), 아무 상태도 바뀌지 않음
       Yes -> _close_pending = True
              training_active면: close_requested를
                  _on_training_close_ready에 connect +
                  training_page.request_stop_and_close()
                  (cooperative stop 요청 -- Phase 5의 기존 계약)
              inference_active면: close_requested를
                  _on_inference_close_ready에 connect +
                  inference_page.request_close()
                  (stop 요청 없음 -- 자연 종료를 기다릴 뿐)
              event.ignore() (즉시 닫지 않음)
    -> 활성이었던 각 page가 스스로 끝나면 close_requested를 emit
       -> 대응하는 _on_*_close_ready()가 disconnect 후
          _maybe_finish_pending_close() 호출
       -> 활성이었던 page 전부가 done으로 표시된 시점에만
          self.close()를 다시 호출 (이 시점엔 둘 다 idle이므로
          closeEvent가 즉시 accept)
```

**training과 inference의 "종료를 기다리는 방식"은 다르다**(Phase 6A
§9의 명시적 설계): training은 cooperative stop(`request_stop_and_close()`
-> 다음 epoch 경계에서 안전하게 중단)을 요청하지만, inference는
**취소하지 않고 원자적 forward pass가 끝날 때까지 그대로 기다린다**
(`request_close()`는 진행 중인 추론을 중단시키지 않는다 -- Phase 6의
non-goal, 아래 §8). 어느 page가 먼저 끝나든 대칭적으로 안전하다 --
먼저 끝난 page의 `close_requested`가 도착해도 다른 page가 아직
active면 `_maybe_finish_pending_close()`가 조용히 반환하고 실제
`close()`는 나중에 끝난 page의 신호가 두 번째로 도착했을 때 정확히
한 번 호출된다. 이 대칭성과 "다이얼로그가 정확히 1회만 뜬다"는 계약은
Phase 6C의 `tests/gui/test_main_window.py`(fake backend)와 Phase 6D의
`tests/gui/test_phase6d_close_integration.py`(실제 backend, 아래 §8)
양쪽에서 확인한다.

`QThread.terminate()`나 GUI thread를 막는 `thread.wait()`는 Phase 6
전체에서 쓰지 않는다(Phase 5 원칙 그대로 유지).

## 8. 검증 범위 -- 자동화 테스트 / CUDA 조건부 / 수동 실행 구분

Phase 6D는 새 permanent test를 실제 검증 gap이 발견된 경우에만
추가한다는 Phase 5D 원칙을 그대로 따랐다. 아래는 **이 checkpoint
시점에 저장소에 실제로 존재하는 파일**을 기준으로 한 구분이며, 특정
실행에서 나온 pass 개수/시간/GPU 실측치를 이 문서 자체가 새로
주장하지 않는다(그런 수치가 필요하면 Phase 6B 구현 문서
`docs/phase6b_inference_core_design.md`처럼 그 Phase가 실제로 실행한
시점의 기록을 인용해야 하며, 이 문서는 그 수치를 재생산하거나
갱신하지 않는다).

### 8-1. 영구 자동화 테스트(CPU에서 항상 실행, CUDA 가용성과 무관)

```text
tests/inference/test_single_image_inference.py           inference core (request/result 조립, model reconstruction 연결, preprocessing parity, prediction/softmax 계약)
tests/application/test_inference_controller.py            InferenceController lifecycle(idle/running/finished/failed, single-active-run guard)
tests/gui/test_qt_inference_worker.py                     QtInferenceWorker thread affinity, deleteLater 계약(fake backend)
tests/gui/test_inference_page.py                           InferencePage 위젯 -> InferenceRequest 매핑, Run lifecycle, 결과 표시(fake backend)
tests/gui/test_main_window.py                              tab navigation, 단일/복수 page close coordination(fake backend)
tests/gui/test_training_inference_integration.py           Phase 6D CP1 -- 같은 MainWindow 세션 안에서 실제 학습 -> 실제 추론 연속 시나리오(실제 CPU backend, fake 없음)
tests/gui/test_phase6d_close_integration.py                Phase 6D CP2 -- 실제 CPU backend로 "추론 중 close"/"학습+추론 동시 활성 중 close"(양쪽 종료 순서 모두) 검증
```

이 목록은 CPU만으로 완결되며 device 선택 UI 자체(`cpu` 옵션)는 CUDA
설치 여부와 무관하게 항상 실행/검증된다.

### 8-2. CUDA 조건부(conditional) 커버리지

```text
tests/inference/test_single_image_inference.py             CUDA fp32/fp16/bf16 forward 경로 -- torch.cuda.is_available() 없으면 skip, bf16은 추가로 torch.cuda.is_bf16_supported()도 확인(둘 다 True일 때만 실행)
tests/gui/test_qt_inference_worker_integration.py            실제 run_single_image_inference()를 실제 QThread로 CPU 1회 + CUDA 1회(skipif) 실행
```

이 테스트들은 CUDA가 없는 환경에서는 **skip되며 실패로 취급되지
않는다.** 이 checkpoint는 어떤 특정 머신에서 이 조건부 테스트가
실행됐는지, 어떤 GPU/실측값이 나왔는지를 새로 주장하지 않는다 --
그런 실측 기록은 그 검증을 실제로 수행한 Phase(예: Phase 6B의 실측은
`docs/phase6b_inference_core_design.md` §11에 이미 기록되어 있다)의
문서를 참고해야 하며, 이 문서는 그 기록을 대신 재확인하거나 새로
생성하지 않는다.

### 8-3. 수동 실행(launcher, 자동화 테스트 아님)

```bash
python scripts/run_gui.py
```

`MainWindow`를 실제로 띄워 Training/Inference 두 tab을 눈으로 확인하는
경로는 **자동화된 pytest 스위트에 포함되지 않는다.** 이 명령을 실행하면
사용자가 직접 Training tab에서 학습을 진행하거나 Inference tab에서
기존 학습 산출물로 추론을 실행해 볼 수 있지만, 이 문서는 특정 시점에
이 명령이 실행되었다거나 그 결과 화면이 어땠는지를 실행 기록으로
주장하지 않는다 -- 이는 §8-1의 자동화된 integration test들(특히
`test_training_inference_integration.py`가 실제 `MainWindow`/
`TrainingPage`/`InferencePage`/실제 backend 조합을 프로그램적으로
구동하는 것)이 대신 담당하는 회귀 안전망이다.

## 9. Phase 6 non-goals / residual risks

```text
inference cancellation(stop 버튼 없음 -- 원자적 단일 forward pass라
    중간에 멈출 지점이 없다는 Phase 6A §7/§8의 명시적 설계 판단, 취소가
    필요해지면 별도 Phase에서 재검토)
batch/folder inference(단일 이미지만 지원, Phase 6A §7의 명시적 범위
    판단 -- InferenceRequest.image_path는 단수(Path)이고 다중 파일
    반복 처리/결과 집계 UX는 설계되지 않음)
이미지 preview / result image rendering(선택한 이미지나 결과를
    QLabel/QPixmap 등으로 시각적으로 보여주는 기능 없음, Phase 6A §9가
    backlog로 명시적으로 미룸)
CUDA 실행은 조건부(conditional)다 -- CUDA가 없는 환경에서는 §8-2의
    테스트가 skip되며, 그 환경에서 CUDA 경로 자체가 검증되었다는
    뜻이 아니다. `device`/`precision` 콤보박스에 cuda 옵션이 뜨는지
    여부도 그 머신의 torch.cuda.is_available() 값에 따라 달라진다
artifact manifest/bundle 새 포맷 없음(state_dict/model JSON/
    class_mapping을 하나로 묶는 새 파일 포맷 없음, §4의 2-picker 절충
    유지, Phase 6A §11)
output_dir에 model JSON을 자동 저장하도록 training core를 확장하지
    않음(training-core 변경이라 Phase 6 범위 밖으로 미룸, §4)
TorchScript inference path 없음(canonical inference는 state_dict +
    ModelSpec 조합뿐, TorchScript export는 여전히 C++ parity 검증
    용도로만 존재, Phase 6A §1-C)
Windows에서 자식/하위 프로세스(subprocess)에 대한 시그널 전달/종료
    제어 관련 알려진 제약(Phase 4K가 문서화한, Windows에서 자식
    프로세스에 Ctrl+C를 전달하려면 CREATE_NEW_PROCESS_GROUP +
    GenerateConsoleCtrlEvent가 필요해 POSIX보다 까다롭다는 제약)은
    Phase 6 GUI 자체에는 **적용되지 않는다** -- Inference/Training GUI
    lifecycle은 OS subprocess가 아니라 in-process QThread만 쓰므로
    이 제약의 대상이 아니다. 이 항목은 프로젝트에 존재하는 별개의
    Windows 관련 제약(CLI 스크립트의 subprocess 기반 시그널 전달)을
    Phase 6 문맥에서 명시적으로 배제하기 위해 기록한다.
```

Phase 5부터 유효한 non-goal(그래프/차트, 실험 이력 DB, multi-run,
packaging/installer, custom theme 등)도 Phase 6에서 계속 유효하다.

## 10. Phase 6 graduation criteria

```text
[ ] 이 checkpoint가 참조하는 고정 integration test 목록(§8-1)이 실제
    커밋된 상태에서 PASS
[ ] 전체 프로젝트 harness(pytest 전체 스위트 + 필요한 회귀 스크립트)가
    실제 커밋된 상태에서 PASS
[x] Training-to-Inference 연속 시나리오가 실제 backend로 검증하는
    자동화 테스트로 저장소에 존재함(§8-1)
[x] close coordination(추론 중/학습+추론 동시 활성, 양쪽 종료 순서)이
    실제 backend로 검증하는 자동화 테스트로 저장소에 존재함(§8-1)
[x] 문서 정합성(README/phase6a/phase6b/phase6_final_integration) 완료
[x] non-goal/residual risk 명시(§9)
```

**PHASE 6 COMPLETE**는 위 두 조건(고정 integration test 목록과 전체
project harness가 커밋된 상태에서 실제로 PASS하는 것)이 충족될 때에만
성립하는 졸업 판정이다. 이 문서를 작성하는 행위 자체는 그 실행을
수행하지 않으며, 이 문서는 "검증이 이 문서 작성으로 완료되었다"고
주장하지 않는다 -- 실제 판정은 그 테스트/harness 실행 결과에 달려
있다.
