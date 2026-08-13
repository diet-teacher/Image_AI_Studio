# Phase 5B: Application + Qt Worker Integration — 설계안

## 1. Phase 5A 결정 요약

Phase 5A(architecture investigation)에서 확정한 것:

```text
GUI framework: PySide6(프로젝트 초기부터 문서상 결정됨, 새 framework 비교 없음)
architecture: Design B(얇은 application/service layer)
thread 전략: QThread(GUI framework native worker)
CUDA 정책: 학습 전체(model 생성 포함)를 worker thread 안에서 시작~완결
progress 전달: Qt signal/slot
stop 전달: threading.Event(cooperative, epoch 경계)
exception 전달: try/except + signal, training core는 무수정
GUI state model: Idle/Running/Stopping/Finished/Failed + result.stop_reason
single active run만 지원
training core public API 변경 불필요
```

Phase 5B는 이 결정들을 실제 코드로 만드는 단계다 -- **실제 Training
Page/MainWindow는 Phase 5C**.

## 2. application/gui dependency boundary

```text
src/image_ai_studio/application/training_controller.py
    PySide6 import 금지 -- framework-agnostic

src/image_ai_studio/gui/qt_training_worker.py
    PySide6 import 허용 -- 이 repository에서 PySide6와
    image_ai_studio.application을 함께 import하는 유일한 위치
```

`import image_ai_studio.gui.qt_training_worker`만으로는 `QApplication`
생성, thread 시작, CUDA 초기화 등 어떤 side effect도 없음을
`tests/application`/`tests/gui`의 import 자체(모듈 최상위에 실행 코드
없음)로 보장한다(§29 확인 완료).

## 3. `TrainingController` responsibility

`src/image_ai_studio/application/training_controller.py`:

- `build_training_request(**kwargs) -> ImageFolderWorkflowRequest`:
  UI 입력값(문자열 경로 포함)을 `TrainingConfig`+
  `ImageFolderWorkflowRequest`로 조립하는 얇은 경계. **semantic
  validation을 전혀 하지 않는다** -- `TrainingConfig.__post_init__`이
  던지는 예외를 그대로 전파한다(§7).
- `TrainingController`: `backend`(기본값
  `run_imagefolder_training_workflow`, 테스트에서 주입 가능)를 감싸
  `state`(`idle`/`running`/`stopping`/`finished`/`failed`)와
  cooperative stop을 위한 `threading.Event`를 관리한다.

**QThread 생성이나 Signal emit은 이 클래스의 책임이 아니다** -- 순수
Python 객체이며, `begin_run()`/`run()`을 어느 thread에서 호출할지는
전적으로 caller(Qt worker)가 정한다.

## 4. Qt worker responsibility

`src/image_ai_studio/gui/qt_training_worker.py`의 `QtTrainingWorker
(QObject)`:

```text
progress = Signal(object)  # TrainingProgress
finished = Signal(object)  # ImageFolderWorkflowResult
failed = Signal(str)       # "{ExceptionType}: {message}\n{traceback}"
```

`run()`(→ `QThread.started`에 연결)이 하는 일 전부:

```text
controller.begin_run()   # TrainingAlreadyRunningError면 failed emit 후 즉시 반환
controller.run(request, progress_callback=self.progress.emit)
    → 성공: finished emit
    → 예외: failed emit(예외 메시지 + traceback.format_exc())
```

이 worker 인스턴스는 살아있는 model/optimizer/tensor를 전혀 소유하지
않는다 -- `run_imagefolder_training_workflow()`가 모든 학습 state를
자기 내부에서만 관리하고 밖으로 반환하지 않으므로, GUI/main thread가
CUDA tensor/model을 만들어 이 worker로 넘기는 일은 구조적으로
발생하지 않는다(Phase 5A CUDA+thread 결론 그대로).

## 5. backend dependency injection

```python
TrainingBackend = Callable[..., ImageFolderWorkflowResult]
```

기본값은 실제 `run_imagefolder_training_workflow`. 테스트는
`TrainingController(backend=fake_backend)`로 실제 학습 없이 상태
전이/인자 전달만 검증한다(§9).

## 6. state model

```text
idle → begin_run() → running
running → request_stop() → stopping
running/stopping → run() 반환(성공) → finished
running/stopping → run() 반환(예외) → failed
idle/finished/failed → begin_run() → running (별도 reset 단계 없음)
```

`finished` 상태 안에서 `result.stop_reason`
(`"completed"`/`"early_stopped"`/`"user_stopped"`)으로 세분한다 --
이를 application state로 중복 인코딩하지 않는다(Phase 5A 조사에서
결정한 GUI state model 방향).

## 7. single-active-run contract

`TrainingController.is_running`(`state in ("running", "stopping")`)이
유일한 판단 기준이다. `begin_run()`이 이미 실행 중일 때 호출되면
`TrainingAlreadyRunningError`를 던지고 **상태를 바꾸지 않는다**.
`QtTrainingWorker.run()`은 이 예외를 잡아 `failed` signal로 전달한다
(테스트: `test_worker_rejects_second_run_while_first_is_active`).

## 8. cooperative stop

```python
self._stop_event = threading.Event()   # begin_run()에서 생성
...
run_imagefolder_training_workflow(request, ..., should_stop=self._stop_event.is_set)
```

`request_stop()`은 GUI thread에서 언제든 호출 가능하고, `running`이
아니면 조용히 no-op이다(중복 클릭/이미 끝난 뒤 요청을 에러로 취급하지
않음). **Phase 4의 기존 epoch 경계 cooperative stop 그대로다** --
`request_stop()` 호출 즉시 학습이 멈추지 않고, `state`가
`stopping`으로 바뀔 뿐 실제 종료는 backend가 반환할 때 일어난다.
`QThread.terminate()`/강제 kill은 어디에도 쓰지 않는다.

## 9. progress signal flow — **중요한 empirical 경고**

`worker.progress.emit(progress)`는 worker thread에서 호출된다.
Qt의 signal-slot이 이를 GUI thread로 "자동으로" 안전하게 넘겨줄 것
같지만, **실제로는 receiver가 무엇이냐에 따라 다르다**:

- receiver가 실제 `QObject` 인스턴스의 메서드(예: `QMainWindow`의
  슬롯)면 `Qt.AutoConnection`이 emitter/receiver의 thread affinity를
  비교해 자동으로 `QueuedConnection`으로 처리한다 -- GUI thread에서
  안전하게 실행됨.
- **receiver가 평범한 Python 함수/lambda/bound method(QObject가
  아닌 객체, 예: `list.append`)면 thread affinity 자체가 없어서
  `AutoConnection`이 direct connection으로 처리되고, emit이 일어난
  worker thread에서 그 자리에서 바로(동기) 실행된다.**

이 사실을 scratch(`phase5b_signal_thread_affinity_check.py`)와 실제
pytest 테스트(`test_plain_function_slot_runs_on_emitting_worker_thread_not_gui_thread`)
양쪽으로 직접 확인했다(가정이 아님). **Phase 5C는 반드시** `worker.
progress`/`finished`/`failed`를 실제 `QObject`(예:
`QMainWindow`/`QWidget`) 인스턴스 메서드에 connect하거나
`type=Qt.ConnectionType.QueuedConnection`을 명시해야 한다 -- 그러지
않고 평범한 함수 안에서 QWidget을 직접 수정하면 Qt의 "위젯은 GUI
thread에서만" 규칙을 위반해 미정의 동작/크래시 위험이 생긴다.

## 10. result flow

`controller.run()`이 정상 반환한 `ImageFolderWorkflowResult`를
그대로 `finished.emit(result)`한다. **worker/controller가
`history.stopped_early`/`stopped_by_user`를 다시 읽어 stop_reason을
재계산하지 않는다** -- `result.stop_reason`이 유일한 authoritative
source다(Phase 4V 계약 그대로 유지).

## 11. failure flow

```python
except Exception as exc:
    self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
```

training core에는 어떤 예외 처리도 추가하지 않았다 -- 이 try/except는
application boundary(`QtTrainingWorker.run()`)에만 있다. `"failed"`는
`TrainingStopReason`에 포함되지 않는다(Phase 4 계약 무변경) -- GUI
state의 `Failed`는 순수 application-layer 개념이다.

## 12. worker/QThread lifecycle

```text
QThread() 생성
QtTrainingWorker(controller, request) 생성 + moveToThread(thread)
thread.started.connect(worker.run)
thread.start()
  ...
finished 또는 failed 수신
thread.quit(); thread.wait()
```

`tests/gui/test_qt_training_worker.py::test_repeated_worker_run_and_cleanup`
가 같은 controller로 worker/thread를 두 번 연속 생성·정리해도
문제없음을 확인했다(Phase 5C의 "Start를 여러 번" 기본 시나리오).

## 13. CUDA + QThread empirical 결과

**PASS** -- scratch(`phase5b_qthread_cuda_smoke.py`)와 정식 pytest
(`test_qt_worker_runs_real_cuda_workflow_off_the_gui_thread`,
`@pytest.mark.skipif(not torch.cuda.is_available())`) 양쪽에서
실제 로컬 GPU로 확인:

```text
worker thread에서 model 생성 → .to("cuda") → forward/backward
→ progress callback → 정상 완료
크래시/hang 없음
thread 정상 종료(QThread.isRunning() == False)
```

CUDA training correctness 자체는 Phase 4가 이미 졸업했으므로 FP32
1개만 확인했다(FP16/BF16까지 반복할 필요가 없다는 것이 Phase 5A
조사의 결론이었다).

## 14. validation responsibility(무변경)

application layer(`build_training_request()`)는 semantic validation을
하지 않는다 -- `TrainingConfig`/`ImageFolderWorkflowRequest`/workflow
자체의 기존 검증이 그대로 authoritative하다. 이 경계는 테스트
(`test_build_training_request_does_not_swallow_config_validation_errors`)
로 고정했다.

## 15. tests

```text
tests/application/test_training_controller.py   12개, PySide6 미사용
tests/gui/test_qt_training_worker.py              6개, pytest-qt(qtbot), fake backend
tests/gui/test_qt_training_worker_integration.py  2개, 실제 workflow(CPU 1개 + CUDA 1개, skipif)
```

CUDA correctness 반복 검증은 하지 않는다(Phase 4가 이미 담당) --
Phase 5B의 CUDA 테스트는 오직 QThread integration 경계만 본다.

## 16. training core 무수정 여부

**무수정.** `src/image_ai_studio/training/*.py`, `scripts/*.py` 전부
diff 없음(`git status --short`로 확인). Phase 4 회귀는 이 사실
자체로 이미 보장되며, full pytest(739 passed, 기존 719 + 신규 20)로
재확인했다.

## 17. Phase 5C handoff contract

Phase 5C가 알아야 할 전부:

```python
from image_ai_studio.application.training_controller import (
    TrainingController, build_training_request, TrainingAlreadyRunningError,
)
from image_ai_studio.gui.qt_training_worker import QtTrainingWorker

request = build_training_request(model_json_path=..., dataset_root=..., output_dir=...,
                                  epochs=..., batch_size=..., learning_rate=..., ...)
controller = TrainingController()  # 기본 backend

thread = QThread()
worker = QtTrainingWorker(controller, request)
worker.moveToThread(thread)
thread.started.connect(worker.run)
worker.progress.connect(main_window.on_progress)   # QObject 메서드에! (§9)
worker.finished.connect(main_window.on_finished)    # result.stop_reason로 결과 분기
worker.failed.connect(main_window.on_failed)        # 메시지 표시, GUI state = Failed
thread.start()

# Stop 버튼:
controller.request_stop()

# 종료:
thread.quit(); thread.wait()
```

`controller.state`로 현재 상태 조회 가능. 새 run은 `finished`/
`failed`에서 바로 `QtTrainingWorker`+`QThread`를 다시 만들어 시작하면
된다(같은 `controller` 재사용).

## 18. non-goals(Phase 5B)

```text
실제 Training Page, MainWindow 디자인, model file picker widget,
dataset picker widget, TrainingConfig widgets, graph, artifact browser,
close confirmation dialog UI, multi-run, experiment history, packaging,
inference GUI, batch progress, ETA, GPU monitor, TensorBoard/W&B,
training core 기능 추가
```

이 항목들은 전부 Phase 5C 이후로 미룬다.
