# Phase 5C: Training GUI — 설계안

## 1. 범위

Phase 5B가 만든 `TrainingController`/`QtTrainingWorker` 위에 실제
사용자가 조작하는 화면(`TrainingPage`/`MainWindow`)을 올린다. 사용자가
할 수 있는 것: Model JSON/Dataset root/Output directory 선택, Basic/
Advanced TrainingConfig 옵션 설정, device/precision 선택, resume
checkpoint 선택(optional), 학습 시작/observe/cooperative stop,
completed/early_stopped/user_stopped 결과 확인, 실패 메시지 확인,
artifact 경로 확인, 완료 후 새 학습 시작.

**training core, `TrainingController`/`QtTrainingWorker` 아키텍처,
Phase 4 public API는 전혀 수정하지 않았다** -- UI가 기존 API에
맞췄다(그 반대가 아님).

## 2. `MainWindow`/`TrainingPage` 구조

```text
src/image_ai_studio/gui/main_window.py    MainWindow(QMainWindow)
src/image_ai_studio/gui/training_page.py  TrainingPage(QWidget)
scripts/run_gui.py                        launcher(QApplication + MainWindow + exec())
```

`widgets/`/`views/`/`presenters/`/`models/` 같은 하위 계층을 만들지
않았다 -- application scope가 Training 화면 하나뿐이므로 `MainWindow`는
`TrainingPage` 하나만 `setCentralWidget()`으로 담는 얇은 껍데기이고,
tab/sidebar/router는 없다.

## 3. widget → request 필드 매핑

`TrainingPage._build_request()`가 현재 widget 값의 snapshot에서
Phase 5B의 `build_training_request(**kwargs)`를 그대로 호출한다 --
`TrainingConfig`/`ImageFolderWorkflowRequest`를 직접 생성하지 않는다.
`model_json_path`/`dataset_root`/`output_dir`는 `QLineEdit.text().
strip()`(경로 문자열, `build_training_request()`가 `Path`로 변환),
나머지는 각 widget의 `.value()`/`.currentText()`/`.isChecked()`를
그대로 kwarg로 전달한다.

## 4. Basic / Advanced 설정 UI

**Basic** (`QGroupBox("Training Settings")`, `QFormLayout`): Epochs
(`QSpinBox`, 기본 5), Batch size(기본 8), Learning rate(`QDoubleSpinBox`,
`decimals=8`, 기본 1e-3), Optimizer(`OPTIMIZER_CHOICES` 콤보, 기본
`adam`), Device(`_detect_device_choices()` 콤보), Precision
(`PRECISION_CHOICES` 콤보, 기본 `fp32`).

**Advanced** (`QGroupBox("Advanced Settings")`): momentum(기본 0.9),
weight_decay(기본 0.0), gradient_clip_norm([Enable]+spin, 기본
비활성), label_smoothing(기본 0.0), class_weights(`QLineEdit`,
comma-separated), lr_scheduler(`None`/`plateau` 콤보) + factor/
patience(scheduler가 `None`이면 비활성), early_stopping_patience
([Enable]+spin), checkpoint_every([Enable]+spin), pin_memory/
non_blocking(`QCheckBox`, 기본 미체크), export_torchscript
(`QCheckBox`, **기본 체크** -- CLI 기본값과 일치), seed(기본
`imagefolder_workflow.SEED`), resume_from(선택 파일 + Browse/Clear),
checkpoint_out(선택 저장 경로 + Browse).

optional 숫자 필드는 전부 `[Enable QCheckBox] + [SpinBox]` 패턴이다
(`_build_optional_double_row`/`_build_optional_int_row`) -- 체크
해제 상태는 `build_training_request()`에 `None`으로 전달되지, `0` 같은
sentinel 값으로 몰래 바뀌지 않는다. `momentum`은 optimizer가 `sgd`가
아니어도 항상 노출한다(SGD로 나중에 바꿀 수 있어야 하므로) --
`TrainingConfig`도 momentum을 optimizer와 무관하게 항상 검증한다.

## 5. request 조립과 실패 처리

`TrainingPage._build_request()`는 semantic validation을 전혀
하지 않는다 -- `build_training_request()`/`TrainingConfig.__post_init__`
이 던지는 예외를 그대로 전파한다. `_on_start_clicked()`가 이를
`try/except`로 감싸서, 실패 시 **`controller.begin_run()`을 호출하기
전에** GUI에 `Failed` 상태만 표시한다(`_show_failure()`) --
worker/thread는 만들어지지 않고, `controller.state`는 `idle`로
그대로 남는다. 이것은 의도된 비대칭이다: GUI-visible "Failed"와
`controller.state`가 항상 1:1로 대응하지는 않는다(request 조립
실패는 controller가 아예 관여하기 전의 GUI-layer 실패이므로).

## 6. `QThread` lifecycle

Phase 5B의 패턴을 그대로 재사용한다:

```python
self._thread = QThread(self)
self._worker = QtTrainingWorker(self._controller, request)
self._worker.moveToThread(self._thread)
self._thread.started.connect(self._worker.run)
self._worker.progress.connect(self._on_progress)   # QObject bound method
self._worker.finished.connect(self._on_finished)   # QObject bound method
self._worker.failed.connect(self._on_failed)        # QObject bound method
self._worker.finished.connect(self._thread.quit)
self._worker.failed.connect(self._thread.quit)
self._worker.finished.connect(self._worker.deleteLater)
self._worker.failed.connect(self._worker.deleteLater)
self._thread.finished.connect(self._thread.deleteLater)
```

**§9(Phase 5B design doc)의 signal thread-affinity 경고를 그대로
지킨다** -- `progress`/`finished`/`failed`를 plain 함수/lambda가
아니라 `TrainingPage`(QObject) bound method에만 connect한다.
`Start`를 다시 누르면 `self._thread`/`self._worker`를 새로 만들어
덮어쓴다(같은 `TrainingController` 인스턴스는 재사용) -- 여러 번
연속 실행을 지원한다. `thread.wait()`를 GUI thread 핸들러 안에서
호출하지 않는다(이벤트 루프를 막지 않기 위해) -- cleanup은 signal
connection(`finished`/`failed` → `thread.quit()` → `deleteLater()`)
에 맡긴다. `QThread.terminate()`는 어디에도 쓰지 않는다.

## 7. progress UI

`_on_progress(progress: TrainingProgress)`가 매 epoch마다 갱신하는
값: progress bar(**`run_epoch`/`total_run_epochs`, `global_epoch`은
절대 분모/분자로 쓰지 않는다** -- resume 이후 잘못된 비율이 나오는
버그를 Phase 4V/5B에서 이미 고쳤고 여기서 다시 지킨다), Global epoch,
Train loss, Val loss, Val accuracy, Learning rate, Best epoch, Best
val loss, Epoch duration. 첫 progress 콜백 전에는 `_reset_progress_
display()`가 모든 라벨을 `"--"`/progress bar `0/1`로 두어 가짜 값을
보여주지 않는다.

## 8. Stop UX

`Stop` 버튼은 Running일 때만 활성화된다. 클릭 시
`controller.request_stop()`을 호출하고 즉시 상태 텍스트를
`"Stopping after current epoch..."`로 바꾼 뒤 자기 자신을 비활성화한다
(중복 클릭 방지) -- **"즉시 취소"가 아니라 epoch 경계 cooperative
stop**임을 문구로 분명히 한다. 실제 `Finished`로 바뀌는 시점은
`worker.finished`/`worker.failed`가 도착한 뒤(`_finish_common()`)이다.

## 9. 결과(Finished) UX

`_on_finished(result)`는 `result.stop_reason`을
`_STOP_REASON_TEXT`(`completed`/`early_stopped`/`user_stopped` →
사람이 읽는 문구)로 매핑해 상태 라벨에 표시한다 -- **`history.
stopped_early`/`stopped_by_user`를 다시 읽어 재계산하지 않는다**
(single source of truth, Phase 4V/5B 계약 그대로). Test loss/Test
accuracy와 artifact 6종(best model/training history/class mapping/
test result/checkpoint/TorchScript model) 경로를 표시하고, `None`인
artifact는 `"Not generated"`로 보여준다.

## 10. 실패(Failed) UX

`_on_failed(message)`는 Phase 5B의 `failed` signal이 보내는 한 개의
`str`(`"{ExceptionType}: {message}\n{traceback}"`)을 받아 첫 줄만
짧은 요약으로, 전체를 별도 읽기전용 `QPlainTextEdit`(Details)에
보여준다 -- worker signal API는 변경하지 않았다. 로깅 프레임워크는
추가하지 않았다(요구 범위 밖).

## 11. close 처리

```text
MainWindow.closeEvent():
    학습 중 아님 → event.accept()
    학습 중 → QMessageBox.question("Stop it and exit?")
        No  → event.ignore()
        Yes → training_page.request_stop_and_close(); event.ignore()

TrainingPage.request_stop_and_close():
    학습 중 아님 → close_requested.emit() 즉시
    학습 중 → self._close_pending = True; self._on_stop_clicked()

TrainingPage._finish_common()  (worker.finished/failed 이후 항상 호출됨):
    _close_pending이 True면 클리어하고 close_requested.emit()
    → MainWindow.close()가 다시 호출됨 → 이번엔 학습 중이 아니므로 accept()
```

`QThread.terminate()`나 `closeEvent` 안에서의 blocking `thread.wait()`
는 쓰지 않는다 -- close는 항상 학습이 실제로(비동기로) 끝난 뒤에만
일어난다. shutdown-manager 같은 별도 추상화 없이 `_close_pending`
boolean 하나로 충분했다.

## 12. tests

```text
tests/gui/test_training_page.py               19개, fake backend, qtbot
tests/gui/test_main_window.py                   3개, smoke + close-during-training
tests/gui/test_training_page_integration.py     1개, 실제 workflow(CPU 1 epoch)
```

`test_training_page.py`는 초기 상태, Basic/Advanced 필드 → request
매핑, CUDA device 선택 → request 매핑(fake backend, §"CUDA GUI
wiring"), optional 필드 비활성 → `None` 매핑, request 조립 실패 시
`controller.state == "idle"` 유지, Start → controls 비활성/Stop
활성, progress bar가 `run_epoch`/`total_run_epochs`를 쓰고
`global_epoch`을 쓰지 않음(resume 유사 케이스로 직접 확인), **`_on_
progress`가 실제로 main/GUI thread에서 실행됨을 실제 subclass override
로 고정(§9의 thread-affinity 계약 검증)**, Stop → "Stopping..." →
cooperative stop 실제 전달, `stop_reason`별 완료 문구, 실패 시
요약/상세/controls 복원, 반복 실행, **8회 연속 run/cleanup을 몰아
실행해 QThread/worker deleteLater ordering을 stress하는 회귀
테스트**(§18)를 각각 덮는다.
`test_main_window.py`는 `MainWindow` 생성/`TrainingPage` 존재/show-close,
그리고 학습 중 close가 확인 다이얼로그(monkeypatch로 Yes) →
cooperative stop → 학습 종료 후에만 실제로 닫히는 흐름을 검증한다.
`test_training_page_integration.py`는 실제
`run_imagefolder_training_workflow()`를 CPU 1 epoch로 GUI를 통해
끝까지 돌려 Completed 상태/test metric/artifact 경로/worker-thread
정리를 확인한다(Phase 4 학습 correctness나 Phase 5B의 QThread+CUDA
wiring을 다시 검증하지 않는다 -- 그건 각각 `tests/training/`과
Phase 5B 테스트가 이미 담당).

## 13. Phase 5B 계약 재사용 확인

새로 만든 코드는 없다 -- `TrainingController`/`QtTrainingWorker`/
`build_training_request()`를 import해서 그대로 쓴다.
`src/image_ai_studio/application/training_controller.py`,
`src/image_ai_studio/gui/qt_training_worker.py`는 이번 Phase에서
diff가 없다(`git status --short`로 확인). signal thread-affinity
경고(Phase 5B design doc §9)를 실제로 지켜서 `TrainingPage`의 bound
method에만 connect했다.

## 14. training core 무수정 확인

**무수정.** `src/image_ai_studio/training/*.py`는 이번 Phase에서
전혀 건드리지 않았다 -- `git status --short`로 확인, full pytest로
회귀 없음 재확인(764 passed, 기존 741 + 신규 23).

## 15. Phase 5D handoff

Phase 5D가 해야 할 일은 최종 통합 검증/문서 정합성/졸업 처리뿐이어야
한다. 이번 Phase에서 미룬 것 없이 최소 요구사항을 전부 구현했으므로,
Phase 5D에 남는 것은 다음과 같은 순수 검증/정리 작업이다: 전체
pytest 최종 재확인, README/design doc 정합성 점검, Phase 5A~5C를
아우르는 최종 그래프(Phase 5 전체 요약), 필요하다면 이번 Phase에서
발견된 미미한 flakiness(§16 참고)에 대한 후속 조사.

## 16. non-goals(Phase 5C)

```text
그래프/차트, 실험 이력 DB, 자동 timestamp run 디렉터리,
artifact "탐색기에서 열기" 버튼, multi-run, inference GUI, packaging,
custom stylesheet/theme/dark mode/애니메이션/커스텀 위젯 라이브러리,
test_metrics 상세 테이블, 새 dependency, training core 기능 추가
```

이 항목들은 Phase 5C의 완료 기준이 아니며, backlog로 유지한다.

## 17. 테스트 작성 중 발견: `monkeypatch.setattr(Class, ...)`는
thread-affinity 테스트를 오염시킬 수 있다

`_on_progress`가 main thread에서 도는지 확인하는 테스트를 처음에는
`monkeypatch.setattr(TrainingPage, "_on_progress", spy_func)`로
작성했는데, 이 경우 spy가 **worker thread에서** 호출되는 것으로
관찰됐다(실제 production 동작과 다름). 진짜 subclass(`class
_ThreadRecordingTrainingPage(TrainingPage): def _on_progress(self,
progress): ...`)로 바꿔 다시 확인하니 스크래치 스크립트(모듈
import부터 실제 QApplication까지 pytest 없이 직접 실행)와 동일하게
main thread에서 정상적으로 실행됨을 확인했다. 즉 **버그는
production 코드가 아니라 처음 작성한 테스트 방법 자체에 있었다** --
사후에 class attribute를 갈아끼우는 방식이 Qt의 connect() 시점 슬롯
해석에 어떤 식으로든 영향을 준 것으로 보인다(정확한 내부 메커니즘은
추적하지 않았다). 최종 테스트는 진짜 subclass override 방식으로
고정했고, production 코드(`training_page.py`)는 이 조사로 인해
전혀 바뀌지 않았다.

## 18. PySide6 + pytest 전체 스위트의 드문 native crash — observed issue / root cause / fix / regression coverage

**Observed issue.** 전체 `pytest -q`를 여러 차례 연속 실행하는 중
드물게(대략 1% 내외) 테스트 로직 실패가 아니라 프로세스 레벨
`Fatal Python error: Aborted`로 중간에 죽는 현상이 관찰됐다. 파일
단위로 좁혀본 결과 `tests/gui/test_qt_training_worker.py`(15회),
`tests/gui/test_qt_training_worker_integration.py`(15회),
`tests/gui/test_training_page.py`(15회),
`tests/gui/test_training_page_integration.py`(15회)는 각각 0회
abort였고, `tests/gui/test_main_window.py`만 단독 실행(15회) 중
1회 abort가 재현되어 원인이 `TrainingPage`가 매 run마다 만드는
`QThread`/`QtTrainingWorker` lifecycle 쪽임을 좁혔다(이 파일이
close-during-training 시나리오에서 실제 QThread를 생성·종료시키는
유일한 gui 테스트 파일이었다).

**Root cause.** `TrainingPage._on_start_clicked()`가 원래

```python
self._worker.finished.connect(self._thread.quit)
self._worker.failed.connect(self._thread.quit)
self._thread.finished.connect(self._worker.deleteLater)   # 문제
self._thread.finished.connect(self._thread.deleteLater)
```

로 연결하고 있었다. `worker`는 `moveToThread(thread)`로 worker
thread에 속해 있으므로 `worker.deleteLater()`가 실제로 안전하게
처리되려면 **worker thread의 event loop가 그 deferred-delete
event를 처리할 시점에 아직 돌고 있어야** 한다. 그런데 `thread.
finished`는 `quit()` 처리로 그 event loop가 이미 멈춘 **뒤에만**
발생한다 -- 즉 `thread.finished.connect(worker.deleteLater)`는
이미 멈춘 thread의 큐에 삭제 이벤트를 posting하는 것과 같아서
처리 여부가 보장되지 않는다. 이는 Qt가 공식적으로 권장하는
canonical `moveToThread` 패턴(`connect(worker, finished, worker,
deleteLater)` -- worker 자신의 finished/failed에 직접 connect)과
다르며, `thread.deleteLater()`(자기 자신의 thread인 main thread의
event loop가 여전히 살아있으므로 안전)와 거의 동시에 같은 신호에서
발화되는 두 `deleteLater()` 사이의 드문 timing window에서 native
crash가 나는 것과 부합한다.

**Fix.** `src/image_ai_studio/gui/training_page.py`
`_on_start_clicked()`에서 `worker.deleteLater()`를 `thread.
finished`가 아니라 **worker 자신의** `finished`/`failed`에
connect하도록 변경했다(Qt canonical 패턴과 일치):

```python
self._worker.finished.connect(self._thread.quit)
self._worker.failed.connect(self._thread.quit)
self._worker.finished.connect(self._worker.deleteLater)
self._worker.failed.connect(self._worker.deleteLater)
self._thread.finished.connect(self._thread.deleteLater)
```

`worker.finished`/`worker.failed`는 worker 자신의 thread에서
emit되고 receiver(`worker` 자신)도 같은 thread이므로 direct
connection으로 그 자리에서 안전하게 처리된다 -- worker thread의
event loop가 아직 멈추지 않은 시점이다. `thread.deleteLater()`는
그대로 `thread.finished`에 남겨뒀다(`thread` 객체 자신은 main
thread 소속이라 안전).

이 외에 `_on_finished`(GUI 갱신, Start 버튼 재활성화)가 `thread.
quit()`보다 먼저 큐에서 처리되어(같은 `worker.finished`에 연결
순서상 먼저 connect됨) "Finished 표시 → 사용자가 바로 Start
클릭 → 이전 QThread가 아직 quit 처리 전"인 순간이 존재할 수
있음도 확인했다(§9에서 요청된 조사). 다만 `TrainingController.
run()`이 `state="finished"`로 바꾸는 시점은 `finished.emit()`보다
먼저이고(같은 worker thread 호출 스택 안), `begin_run()`은 매번
새 `threading.Event`를 만들며, 이전 run의 worker thread는 이미
자기 `run()`을 반환한 뒤라 controller state 변경의 영향을 받는
코드가 더 이상 실행 중이지 않다 -- 즉 이 ordering은 실제 데이터
손상이나 crash로 이어지는 경로를 찾지 못했다(`QThread::quit()`는
어느 thread에서 호출해도 안전하다는 것이 Qt 문서에 명시돼 있다).
근거 없는 추가 수정은 하지 않았다.

**Regression coverage.**
`tests/gui/test_training_page.py::test_repeated_run_thread_lifecycle_stress`
(신규)가 fake backend로 8회 연속 run/cleanup을 짧은 시간 안에 몰아
실행하고, 마지막에는 최종 QThread가 실제로 멈추거나 `deleteLater()`
로 해제됐는지까지 확인한다(Start 버튼 재활성화 시점과 실제 thread
teardown 완료 시점은 서로 다른 queued event이므로 별도로 기다림) --
이 deleteLater ordering을 반복적으로 노출시킨다.
검증: 수정 전에는 `tests/gui/test_main_window.py` 단독 15회 실행 중
1회 abort가 재현됐고, 수정 후에는 같은 파일 30회, `tests/gui` 전체
20회+20회(총 40회), 신규 stress 테스트 단독 25회, 전체 `pytest -q`
8회(초기 6회 + 문서 정합성 재확인 라운드 2회) -- 도합 100회 이상의
반복 실행에서 abort가 **0회**였다.

