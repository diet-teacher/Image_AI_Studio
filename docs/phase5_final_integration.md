# Phase 5 Final Integration Validation & Graduation

Phase 5A~5D 전체를 아우르는 최종 정리 문서다. Phase 5D는 새 기능을
추가하지 않고, Phase 5A~5C에서 만든 것이 실제 사용자 관점에서 하나의
완성된 학습 애플리케이션으로 정상 동작하는지 최종 검증했다.

## 1. Architecture

```text
TrainingPage / MainWindow            (GUI, PySide6)
        |  build_training_request()
        v
TrainingController                   (application, framework-agnostic)
        |  QtTrainingWorker.run() 안에서 begin_run()+run() 호출
        v
QtTrainingWorker (QObject) / QThread  (GUI-framework worker boundary)
        |  블로킹 호출
        v
run_imagefolder_training_workflow()   (training core public entrypoint)
        |
        v
training core (model/optimizer/loop/checkpoint/metrics/...)
```

의존 방향은 위에서 아래로만 흐른다 -- training core는 GUI/application을
전혀 모르고, `TrainingController`는 PySide6를 전혀 import하지 않는다
(PySide6와 `image_ai_studio.application`을 함께 import하는 유일한
지점은 `src/image_ai_studio/gui/qt_training_worker.py`).

## 2. Phase별 역할

| Phase | 추가한 것 | 건드리지 않은 것 |
|---|---|---|
| 5A | architecture 조사(PySide6/Design B/QThread/cooperative stop 등 결정), **코드 없음** | 전체 |
| 5B | `TrainingController`, `build_training_request()`, `QtTrainingWorker` | training core, 실제 GUI 화면 |
| 5C | `TrainingPage`, `MainWindow`, `scripts/run_gui.py` | training core, `TrainingController`/`QtTrainingWorker` architecture |
| 5D | **없음**(verification-only) -- Phase 5C stabilization 라운드에서 `training_page.py`의 signal 연결 순서(§6/§7 참고)만 수정 | training core, `TrainingController`, `QtTrainingWorker`, GUI 기능 계약 |

## 3. Runtime flow

```text
GUI input(위젯 snapshot)
    -> TrainingPage._build_request() -> build_training_request()
    -> ImageFolderWorkflowRequest 생성(semantic validation은
       TrainingConfig/ImageFolderWorkflowRequest 자체가 수행)
    -> 실패 시: controller.begin_run()을 호출하지 않고 GUI에만 Failed 표시
       (controller.state는 idle로 유지)
    -> 성공 시: QThread + QtTrainingWorker 생성, moveToThread, signal 연결
    -> thread.start()
       -> (worker thread) QtTrainingWorker.run()
          -> controller.begin_run()  (TrainingAlreadyRunningError면 failed emit)
          -> controller.run(request, progress_callback=self.progress.emit)
             -> backend(run_imagefolder_training_workflow) 블로킹 호출
             -> 매 epoch마다 progress_callback 호출(→ progress.emit, worker thread)
          -> 성공: state="finished" 전이 후 finished.emit(result)
          -> 예외: state="failed" 전이 후 failed.emit(message+traceback)
    -> (GUI thread, queued) TrainingPage._on_progress/_on_finished/_on_failed
       가 위젯을 갱신
    -> cleanup: worker.finished/failed -> thread.quit(), -> worker.deleteLater()
                thread.finished -> thread.deleteLater()
```

## 4. Stop/close lifecycle

```text
Stop 클릭
    -> controller.request_stop()  (running이 아니면 조용히 no-op)
    -> 상태 "Stopping after current epoch..." 즉시 표시, Stop 버튼 비활성
    -> (worker thread) 다음 epoch 경계에서 should_stop()==True 감지 ->
       result.stop_reason == "user_stopped"로 정상 반환
    -> finished.emit(result) -> GUI가 "Training stopped by user" 표시,
       모든 configuration control + Browse/Clear 버튼 재활성

학습 중 MainWindow.close()
    -> QMessageBox.question("Stop it and exit?")
       No  -> event.ignore(), 창 유지, 학습 계속
       Yes -> training_page.request_stop_and_close()
              (내부적으로 controller.request_stop() + _close_pending=True)
              event.ignore()  (즉시 닫지 않음)
    -> 학습이 finished/failed로 끝나면 _finish_common()이
       close_requested.emit() -> MainWindow.close()가 다시 호출됨
       -> 이번엔 is_training_active()==False이므로 closeEvent가 accept
```

`QThread.terminate()`, GUI thread를 막는 `thread.wait()`는 어디에도
쓰지 않는다 -- 전부 실제 코드 재확인 완료(§9 참고).

## 5. QThread ownership / `deleteLater` 계약

```python
self._thread = QThread(self)                       # parent=TrainingPage
self._worker = QtTrainingWorker(self._controller, request)
self._worker.moveToThread(self._thread)
self._thread.started.connect(self._worker.run)
self._worker.progress.connect(self._on_progress)    # QObject bound method
self._worker.finished.connect(self._on_finished)    # QObject bound method
self._worker.failed.connect(self._on_failed)         # QObject bound method
self._worker.finished.connect(self._thread.quit)
self._worker.failed.connect(self._thread.quit)
self._worker.finished.connect(self._worker.deleteLater)   # worker 자신의 신호에!
self._worker.failed.connect(self._worker.deleteLater)     # 〃
self._thread.finished.connect(self._thread.deleteLater)
```

`worker.deleteLater()`는 반드시 **worker 자신의** `finished`/`failed`에
연결해야 한다(Qt canonical `moveToThread` 패턴) -- `worker`는 worker
thread 소속이므로 그 thread의 event loop가 아직 살아있는 동안(=worker
자신이 방금 emit한 신호를 같은 thread에서 direct connection으로 처리하는
시점) 처리돼야 안전하다. `thread.finished`(= `quit()` 처리로 그 event
loop가 이미 멈춘 뒤에만 발생)에 연결하면 이미 멈춘 thread에 삭제
이벤트를 posting하는 것과 같아 처리 여부가 보장되지 않는다 -- 이것이
Phase 5C stabilization에서 드물게 재현된 native abort(`Fatal Python
error: Aborted`)의 실측 원인이었다(docs/phase5c_training_gui_design.md
§18). `thread.deleteLater()`는 그대로 `thread.finished`에 남아있다
(`thread` 객체 자신은 항상 main thread 소속이라 안전).

**Phase 5D에서 이 계약이 여전히 유지되는지 실제 코드로 재확인했다**
(`src/image_ai_studio/gui/training_page.py` `_on_start_clicked()`,
2026-08-14 기준) -- 회귀 없음.

## 6. GUI contract 최종 확인 (Phase 5D)

실제 코드 기준으로 재확인한 것:

* GUI는 training core 객체를 직접 조립하지 않는다 -- `TrainingPage.
  _build_request()`는 `build_training_request()`만 호출한다.
* GUI thread에서 학습을 직접 수행하지 않는다 -- 전부 `QtTrainingWorker.
  run()`(worker thread)에서 일어난다.
* `progress`/`finished`/`failed`는 `TrainingPage`의 실제 QObject bound
  method에 connect돼 있고, main Qt thread에서 실행됨을 실제 CUDA 학습
  경로까지 포함해 재확인했다(§7).
* Start마다 새 `QThread`/`QtTrainingWorker`를 만들고, 같은
  `TrainingController` 인스턴스는 재사용한다.
* `QThread.terminate()`, GUI thread를 막는 `thread.wait()` 사용 없음.
* `TrainingController`: run lifecycle state(`idle`/`running`/
  `stopping`/`finished`/`failed`) 관리, `is_running`(`running`/
  `stopping`)으로 single-active-run 판단, `begin_run()`마다 새
  `threading.Event` 생성, `finished`/`failed` state 전이는 signal
  emit 이전에(같은 호출 스택 안에서) 완료된다.
* `QtTrainingWorker`: backend 호출은 worker thread에서 실행, UI에
  직접 접근하지 않는다(signal만 emit).

## 7. Phase 5D 실측 검증 결과

모두 실제 코드/실제 실행 기준(fake backend가 아닌 real backend를
우선 사용, fake는 Phase 5C의 기존 자동화 테스트가 이미 충분히
커버하는 경우에만 재사용):

* **GUI launcher smoke**: `python scripts/run_gui.py`를 백그라운드로
  6초 이상 구동, stdout/stderr에 traceback/Qt warning 없음. 별도로
  실제 `MainWindow` + 실제 `QTest.mouseClick(page._start_button, ...)`
  로 Start를 눌러 Model/Dataset/Output 지정 -> Basic/Advanced 설정 ->
  Running 중 모든 configuration control + Browse/Clear 버튼 6개
  비활성 -> 완료 -> 전부 재활성까지 end-to-end 확인.
* **CPU E2E**: 실제 tiny ImageFolder(8x8x3, cat/dog) + 실제
  `ModelSpec`으로 실제 GUI Start 경로를 통해 1 epoch 완주. Completed
  상태, test loss/accuracy 표시, `best_model_state_dict.pt`/
  `training_history.json`/`class_mapping.json`/`test_result.json`
  생성, `export_torchscript=False`일 때 TorchScript artifact가 "Not
  generated"로 정확히 표시됨을 확인.
* **resume GUI path**: 1 epoch 학습 + `checkpoint_every=1`로
  checkpoint 생성 -> **새 `TrainingPage`** 인스턴스에서
  `resume_from` 필드로 지정 -> request에 정확히 전달됨을
  `page._build_request().resume_from == checkpoint_path`로 직접
  확인 -> 추가 1 epoch 실행 -> `global_epoch==2`(누적), `run_epoch==1`/
  `total_run_epochs==1`(호출-local), progress bar는 여전히
  `run_epoch`/`total_run_epochs`(1/1)만 사용(4/2 같은 잘못된 값 아님),
  resumed run의 artifact도 정상 생성됨을 확인.
* **CUDA GUI smoke**: 이 개발 머신은 `torch.cuda.is_available() ==
  True`(1 device) -- **skip하지 않고 실제로 수행**. device combo에
  `cuda`/`cuda:0` 노출, `device="cuda"` 선택이 실제 request에
  전달됨, 실제 QThread에서 실제 CUDA 학습이 GUI thread를 막지 않고
  완료됨, progress handler가 main thread에서 실행됨(subclass override로
  thread id 직접 기록해 검증), 완료 후 QThread가 실제로 정리됨을
  확인.
* **cooperative stop**: **fake가 아닌 실제 CPU backend**로 Start ->
  Stop -> "Stopping after current epoch..." -> 실제 학습 중단 ->
  "Training stopped by user" -> 모든 control 재활성을 확인. (단,
  §8의 narrow race를 이 과정에서 발견 -- 아래 참고.)
* **close-during-training**: **fake가 아닌 실제 CPU backend**로
  학습이 실제로 진행 중(`controller.state in ("running","stopping")`,
  최소 1개의 실제 progress event 관측)인 상태에서
  `MainWindow.close()`를 호출 -> 확인 다이얼로그(Yes로 확인) -> 창이
  즉시 닫히지 않고 "Stopping after current epoch..." 표시 -> 실제
  학습이 멈춘 뒤에만 창이 실제로 닫히고 상태가 "Training stopped by
  user"임을 확인.
* **repeated-run / QThread cleanup stabilization**:
  `tests/gui/test_training_page.py::test_repeated_run_thread_lifecycle_stress`
  (fake backend, 8회 연속 run/cleanup + 최종 QThread cleanup 확인)와
  `tests/gui/test_main_window.py`(실제 QThread를 만드는 close-during-
  training 테스트 포함)를 묶어 반복 실행 -- native abort 재발 없음.

## 8. Known limitation: Start/Stop 사이의 narrow race (수정하지 않음)

Phase 5D 검증 중 실제 CPU backend로 cooperative stop을 테스트하다가
발견한 것: `TrainingPage._on_start_clicked()`가 `self._thread.start()`
를 호출한 직후, worker thread가 아직 `TrainingController.begin_run()`
을 호출하기 전(`_stop_event`가 아직 `None`인 매우 짧은 창, 실측
1ms 미만)에 `TrainingController.request_stop()`이 호출되면
`stop_event is None`이므로 조용히 no-op되고, 이후 실제로 시작되는
run은 이 stop 요청을 전혀 인지하지 못한 채 끝까지 실행된다.

실측(같은 프로세스, `_on_start_clicked()`와 `_on_stop_clicked()`를
동일 호출 스택에서 바로 연달아 호출): `_on_start_clicked()` 반환
시점에 `controller.state == "idle"`, `_stop_event is None` -- 이
상태에서 `request_stop()`은 그대로 no-op. 이후 worker thread가
`begin_run()`을 호출해 새 `_stop_event`(unset)를 만들고, 이미 지나간
`request_stop()` 호출은 이 새 이벤트에 아무 영향도 주지 못한다.

**재현 가능성**: 이 race window는 사람이 마우스로 Start를 클릭한 뒤
다시 마우스를 움직여 Stop을 클릭하는 데 걸리는 현실적인 시간(통상
100ms 이상)보다 압도적으로 짧다 -- 현재 일반적인 GUI 마우스 조작으로
재현될 가능성은 매우 낮다고 판단한다. 위 실측도 두 handler 메서드를
프로그램적으로 동일 호출 스택에서 즉시 연달아 호출해 인위적으로
만든 것이다.

**수정하지 않은 이유**: 올바른 수정은 `QtTrainingWorker.run()`이
내부적으로 호출하던 `controller.begin_run()`을 GUI thread
(`TrainingPage._on_start_clicked()`, `thread.start()` 이전)로
옮기는 것인데, 이는 `QtTrainingWorker`의 기존 공개 계약(현재
`run()` 하나가 begin_run부터 결과 emit까지 전부 책임짐)을 바꾸는
architecture 변경이며, `tests/gui/test_qt_training_worker.py`/
`test_qt_training_worker_integration.py`(Phase 5B) 등 다른 코드도
함께 손봐야 한다. Phase 5D는 "실제 실패 경로가 확인된 경우에만
최소 수정" 원칙이었고, 현재 일반적인 GUI 사용 경로에서는 도달
가능성이 매우 낮은 known limitation이라고 판단해 **문서화만 하고
production 코드는 수정하지 않기로 결정했다**(사용자 확인 완료).
향후 이 race가 실제로 문제가 되는 사례가 보고되면 위 방향으로
재검토할 것을 backlog로 남긴다.

## 9. Final test coverage

`pytest --collect-only -q` 기준 **764개**, 영역별:

| 영역 | 파일 | 개수 |
|---|---|---|
| model definition | test_specs_validation/test_shape_inference/test_serialization/test_builder/test_phase1_e2e_script/test_torchscript_integration | 168 |
| training core | test_loop/test_metrics/test_history/test_dataset/test_model_definition_integration | 176 |
| ImageFolder workflow | test_imagefolder_workflow/test_imagefolder_dataset/test_torchvision_dataset | 127 |
| checkpoint/resume | test_checkpoint/test_imagefolder_resume | 76 |
| config/CLI(precision·device 포함) | test_config/test_train_imagefolder_cli/test_train_imagefolder_args | 170 |
| application controller | test_training_controller | 14 |
| Qt worker | test_qt_training_worker/test_qt_training_worker_integration | 8 |
| TrainingPage(+ GUI integration) | test_training_page/test_training_page_integration | 20 |
| MainWindow | test_main_window | 3 |
| C++ parity / TorchScript | test_train_export_parity(+ test_torchscript_integration은 위 model definition에 포함) | 2 |

precision/device는 별도 파일이 아니라 `test_loop.py`/
`test_imagefolder_workflow.py`(CPU/CUDA FP32/FP16/BF16,
`@pytest.mark.skipif(not torch.cuda.is_available())`)와
`test_config.py`/CLI 테스트들에 걸쳐 있다.

Phase 5D는 이 기존 coverage로 계약이 충분히 검증된다고 판단해 새
permanent test를 추가하지 않았다(§7의 실측 검증은 전부 ad-hoc
scratch script로 수행 -- scratchpad에만 존재, repo에 커밋되지 않음).

## 10. Known limitations / non-goals

```text
그래프/차트, 실험 이력 DB, 자동 timestamp run 디렉터리,
artifact "탐색기에서 열기" 버튼, multi-run, inference GUI, packaging,
custom stylesheet/theme/dark mode/애니메이션/커스텀 위젯 라이브러리,
test_metrics 상세 테이블, 새 dependency, training core 기능 추가,
Start/Stop 사이 narrow race(§8, 문서화만 완료, 일반적인 GUI 사용
경로에서 도달 가능성이 매우 낮다고 판단)
```

## 11. Phase 5 graduation criteria

```text
[x] 전체 pytest 764/764 PASS(여러 차례 반복, native abort 0회)
[x] 실제 GUI launcher(scripts/run_gui.py) 정상 구동
[x] CPU E2E 정상(실제 GUI 경로, 실제 backend)
[x] stop/close/repeated-run lifecycle 정상(실제 backend 포함)
[x] resume GUI wiring 정상(실제 backend, 새 TrainingPage)
[x] CUDA 가능 환경에서 CUDA smoke 정상(skip 아님, 실제 GPU로 수행)
[x] 문서 정합성(README/phase5b/phase5c/phase5_final_integration) 완료
[x] native abort 재발 없음(반복 검증)
[x] 미해결 blocker 없음(§8의 race는 일반적인 GUI 사용 경로에서
    도달 가능성이 매우 낮은 known limitation으로 문서화, blocker 아님)
```

**PHASE 5 COMPLETE.**
