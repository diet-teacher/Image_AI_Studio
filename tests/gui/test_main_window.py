"""`MainWindow` smoke + close-during-training 테스트(Phase 5C)."""
from __future__ import annotations

import threading

from PySide6.QtWidgets import QMessageBox

from image_ai_studio.application.training_controller import TrainingController
from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.gui.training_page import TrainingPage
from image_ai_studio.training.imagefolder_workflow import ImageFolderWorkflowResult
from image_ai_studio.training.loop import TrainingHistory


def _fake_result() -> ImageFolderWorkflowResult:
    history = TrainingHistory(
        train_losses=[0.5], val_losses=[0.4], val_accuracies=[0.9],
        best_epoch=1, best_val_loss=0.4,
    )
    return ImageFolderWorkflowResult(
        history=history, test_loss=0.3, test_accuracy=0.95,
        best_model_state_dict_path="best.pt", training_history_path="history.json",
        class_mapping_path="class_mapping.json", test_result_path="test_result.json",
        checkpoint_path=None, checkpoint_metadata_path=None,
        torchscript_model_path=None, torchscript_metadata_path=None,
        stop_reason="user_stopped",
    )


def test_main_window_smoke(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    assert isinstance(window.centralWidget(), TrainingPage)

    window.show()
    assert window.isVisible() is True

    window.close()
    assert window.isVisible() is False


def test_close_while_idle_closes_immediately(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    window.close()

    assert window.isVisible() is False


def test_close_while_training_prompts_and_waits_for_stop(tmp_path, monkeypatch, qtbot) -> None:
    """closeEvent가 확인 다이얼로그(Yes로 monkeypatch)를 거쳐
    cooperative stop을 요청하고, 실제 close는 학습이 끝난 뒤에만
    일어나야 한다 -- 강제 종료/blocking wait 없음."""
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request, *, progress_callback=None, should_stop=None):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        assert should_stop() is True  # close 확인 중 request_stop()이 걸렸어야 함
        return _fake_result()

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    page: TrainingPage = window.centralWidget()
    page._controller = TrainingController(backend=blocking_backend)

    (tmp_path / "model.json").write_text("{}")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "out").mkdir()
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._dataset_root_edit.setText(str(tmp_path / "dataset"))
    page._output_dir_edit.setText(str(tmp_path / "out"))

    page._on_start_clicked()
    assert backend_started.wait(timeout=5)

    window.close()  # 학습 중 -- 다이얼로그(Yes로 monkeypatch됨) -> stop 요청, 아직 안 닫힘
    assert window.isVisible() is True
    assert page._status_label.text() == "Stopping after current epoch..."

    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)
