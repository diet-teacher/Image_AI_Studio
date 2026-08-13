"""`QtTrainingWorker`/`QThread` wiring 테스트(Phase 5B). `pytest-qt`의
`qtbot` fixture가 `QApplication`을 관리해 준다. 매 테스트마다 실제 학습을
돌리지 않는다 -- fake backend를 주입해 signal 전달/thread 분리/cleanup만
검증한다(실제 CPU 학습 1개는 `test_qt_training_worker_integration.py`가
따로 담당한다)."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread

from image_ai_studio.application.training_controller import TrainingController
from image_ai_studio.gui.qt_training_worker import QtTrainingWorker
from image_ai_studio.training.config import TrainingConfig
from image_ai_studio.training.imagefolder_workflow import (
    ImageFolderWorkflowRequest,
    ImageFolderWorkflowResult,
)
from image_ai_studio.training.loop import TrainingHistory, TrainingProgress

MAIN_THREAD_ID = threading.get_ident()


def _dummy_request() -> ImageFolderWorkflowRequest:
    return ImageFolderWorkflowRequest(
        model_json_path=Path("model.json"),
        dataset_root=Path("dataset"),
        training_config=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-2),
        output_dir=Path("out"),
    )


def _fake_result(stop_reason: str = "completed") -> ImageFolderWorkflowResult:
    history = TrainingHistory(
        train_losses=[0.5], val_losses=[0.5], val_accuracies=[0.5],
        best_epoch=1, best_val_loss=0.5,
    )
    return ImageFolderWorkflowResult(
        history=history,
        test_loss=0.5,
        test_accuracy=0.5,
        best_model_state_dict_path=Path("best.pt"),
        training_history_path=Path("history.json"),
        class_mapping_path=Path("class_mapping.json"),
        test_result_path=Path("test_result.json"),
        checkpoint_path=None,
        checkpoint_metadata_path=None,
        torchscript_model_path=None,
        torchscript_metadata_path=None,
        stop_reason=stop_reason,
    )


def _make_progress(global_epoch: int) -> TrainingProgress:
    return TrainingProgress(
        run_epoch=global_epoch, total_run_epochs=1, global_epoch=global_epoch,
        train_loss=0.5, val_loss=0.5, val_accuracy=0.5, learning_rate=1e-2,
        best_epoch=1, best_val_loss=0.5, epochs_without_improvement=0,
        stopped_early=False, epoch_duration_seconds=0.01,
    )


def test_worker_runs_off_the_gui_thread_and_emits_finished(qtbot) -> None:
    thread_ids: dict = {}

    def fake_backend(request, *, progress_callback=None, should_stop=None):
        thread_ids["worker"] = threading.get_ident()
        progress_callback(_make_progress(1))
        return _fake_result()

    controller = TrainingController(backend=fake_backend)
    worker = QtTrainingWorker(controller, _dummy_request())
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    progresses: list = []
    worker.progress.connect(progresses.append)

    with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
        thread.start()

    result = blocker.args[0]
    assert result.stop_reason == "completed"
    assert len(progresses) == 1
    assert progresses[0].global_epoch == 1
    assert thread_ids["worker"] != MAIN_THREAD_ID  # 실제로 다른 python thread에서 실행됨
    assert controller.state == "finished"

    thread.quit()
    thread.wait(5000)
    assert thread.isRunning() is False


def test_worker_emits_failed_on_backend_exception(qtbot) -> None:
    def failing_backend(request, *, progress_callback=None, should_stop=None):
        raise ValueError("scratch injected failure")

    controller = TrainingController(backend=failing_backend)
    worker = QtTrainingWorker(controller, _dummy_request())
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    with qtbot.waitSignal(worker.failed, timeout=5000) as blocker:
        thread.start()

    message = blocker.args[0]
    assert "ValueError" in message
    assert "scratch injected failure" in message
    assert "Traceback" in message  # traceback.format_exc() 포함 확인
    assert controller.state == "failed"

    thread.quit()
    thread.wait(5000)


def test_worker_rejects_second_run_while_first_is_active(qtbot) -> None:
    """single active run 계약: 같은 controller로 이미 begin_run()된
    상태에서 두 번째 worker.run()을 부르면 즉시 failed를 emit해야 한다."""
    controller = TrainingController(backend=lambda *a, **k: _fake_result())
    controller.begin_run()  # 이미 실행 중인 것처럼 미리 상태를 만들어 둔다

    worker = QtTrainingWorker(controller, _dummy_request())
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    with qtbot.waitSignal(worker.failed, timeout=5000) as blocker:
        thread.start()

    assert "TrainingAlreadyRunningError" in blocker.args[0]

    thread.quit()
    thread.wait(5000)


def test_worker_stop_request_reaches_should_stop(qtbot) -> None:
    """GUI thread가 실제로 학습이 진행 중인 동안 Stop을 누르는 상황을
    두 개의 `threading.Event`로 정확히 재현한다: fake backend는 시작을
    알리고 대기하다가, GUI thread가 `request_stop()`을 호출한 뒤에야
    `should_stop()`을 평가하고 반환한다."""
    backend_started = threading.Event()
    let_backend_finish = threading.Event()
    observed = {}

    def fake_backend(request, *, progress_callback=None, should_stop=None):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5), "GUI thread never released the backend"
        observed["should_stop_value"] = should_stop()
        return _fake_result(stop_reason="user_stopped")

    controller = TrainingController(backend=fake_backend)
    worker = QtTrainingWorker(controller, _dummy_request())
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
        thread.start()
        assert backend_started.wait(timeout=5), "worker thread never reached the backend call"
        controller.request_stop()
        assert controller.state == "stopping"
        let_backend_finish.set()

    assert observed["should_stop_value"] is True
    assert blocker.args[0].stop_reason == "user_stopped"

    thread.quit()
    thread.wait(5000)


def test_repeated_worker_run_and_cleanup(qtbot) -> None:
    """Phase 5C가 Start를 여러 번(순차적으로) 쓰는 기본 시나리오 --
    한 worker/thread가 끝난 뒤 같은 controller로 새 worker/thread를
    만들어도 문제없어야 한다."""
    controller = TrainingController(backend=lambda *a, **k: _fake_result())

    for _ in range(2):
        worker = QtTrainingWorker(controller, _dummy_request())
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        with qtbot.waitSignal(worker.finished, timeout=5000):
            thread.start()

        thread.quit()
        thread.wait(5000)
        assert thread.isRunning() is False
        assert controller.state == "finished"


def test_plain_function_slot_runs_on_emitting_worker_thread_not_gui_thread(qtbot) -> None:
    """**Phase 5C에게 반드시 필요한 경고**(empirical 확인, 가정 아님):
    `Signal`을 QObject가 아닌 평범한 Python 함수/bound method(예: 이
    테스트의 리스트 `.append`처럼)에 connect하면, 기본 `AutoConnection`
    은 그 슬롯을 **emit한 thread(=worker thread)에서 직접(동기) 실행**
    한다 -- GUI thread로 자동 queue되지 않는다(receiver가 QObject가
    아니라 thread affinity 자체가 없기 때문). 즉 Phase 5C가 `worker.
    progress.connect(some_plain_function)`처럼 연결하면서 그 함수 안에서
    QWidget을 직접 건드리면 Qt의 "위젯은 GUI thread에서만" 규칙을
    위반하게 된다.

    안전하게 쓰려면 Phase 5C는 반드시:
    - 실제 QObject(예: QWidget/QMainWindow) 메서드에 connect하거나
      (Qt가 receiver의 thread affinity를 인식해 자동으로 queue한다),
    - 또는 `connect(..., type=Qt.ConnectionType.QueuedConnection)`을
      명시해야 한다.

    이 테스트는 그 위험한 기본 동작 자체를 직접 고정해, Phase 5C
    설계/구현자가 이 사실을 모르고 넘어가지 않게 한다."""
    execution_thread_ids: list[int] = []

    def plain_function_slot(progress: object) -> None:
        # QObject가 아닌 평범한 함수 -- Qt 입장에서 thread affinity가 없다.
        execution_thread_ids.append(threading.get_ident())

    def fake_backend_with_one_progress_event(request, *, progress_callback=None, should_stop=None):
        progress_callback(_make_progress(1))
        return _fake_result()

    controller = TrainingController(backend=fake_backend_with_one_progress_event)
    worker = QtTrainingWorker(controller, _dummy_request())
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.progress.connect(plain_function_slot)  # 기본 AutoConnection

    with qtbot.waitSignal(worker.finished, timeout=5000):
        thread.start()

    assert len(execution_thread_ids) == 1
    # 핵심 assertion: GUI(main) thread가 아니라 emit이 일어난 worker
    # thread에서 직접 실행됐다.
    assert execution_thread_ids[0] != MAIN_THREAD_ID

    thread.quit()
    thread.wait(5000)
