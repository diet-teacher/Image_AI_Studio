"""`MainWindow` smoke + close-coordination 테스트(Phase 5C, CP4 확장).

CP4는 `TrainingPage` + `InferencePage`를 담은 `QTabWidget`과, 학습/추론
둘 다를 아우르는 중앙집중형 non-blocking close coordination을 검증한다."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTabWidget

from image_ai_studio.application.folder_inference_controller import FolderInferenceController
from image_ai_studio.application.inference_controller import InferenceController
from image_ai_studio.application.training_controller import TrainingController
from image_ai_studio.gui.inference_page import InferencePage
from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.gui.model_designer import ModelDesigner
from image_ai_studio.gui.training_page import TrainingPage
from image_ai_studio.model_definition.serialization import load_model_spec
from image_ai_studio.inference.folder_inference import (
    FolderInferenceCancelled,
    FolderInferenceError,
    FolderInferenceProgress,
    FolderInferenceResult,
    ImageOutcome,
)
from image_ai_studio.inference.single_image_inference import InferenceResult
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


def _fake_inference_result() -> InferenceResult:
    return InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1},
        inference_duration_seconds=0.01,
    )


def _fill_minimum_valid_inference_fields(page: InferencePage, tmp_path) -> None:
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(tmp_path / "image.png"))


def _fill_minimum_valid_training_fields(page: TrainingPage, tmp_path) -> None:
    (tmp_path / "model.json").write_text("{}")
    (tmp_path / "dataset").mkdir(exist_ok=True)
    (tmp_path / "out").mkdir(exist_ok=True)
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._dataset_root_edit.setText(str(tmp_path / "dataset"))
    page._output_dir_edit.setText(str(tmp_path / "out"))


# -- tab integration -----------------------------------------------------------


def test_main_window_smoke(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    window.show()
    assert window.isVisible() is True

    window.close()
    assert window.isVisible() is False


def test_central_widget_is_tab_widget(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.centralWidget(), QTabWidget)


def test_tab_widget_has_exactly_three_tabs(qtbot) -> None:
    """CP2 adds one Model Designer tab; the existing Training/Inference tabs
    are preserved, so the tab count goes from two to exactly three."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._tabs.count() == 3


def test_training_page_added_exactly_once(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    pages = [window._tabs.widget(i) for i in range(window._tabs.count())]
    assert pages.count(window._training_page) == 1


def test_inference_page_added_exactly_once(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    pages = [window._tabs.widget(i) for i in range(window._tabs.count())]
    assert pages.count(window._inference_page) == 1


def test_model_designer_added_exactly_once(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    pages = [window._tabs.widget(i) for i in range(window._tabs.count())]
    assert pages.count(window._model_designer) == 1
    assert isinstance(window._model_designer, ModelDesigner)


def test_stable_references_to_all_pages_in_order(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window._training_page, TrainingPage)
    assert isinstance(window._inference_page, InferencePage)
    assert isinstance(window._model_designer, ModelDesigner)
    # Training/Inference identity + order preserved; designer appended last.
    assert window._tabs.widget(0) is window._training_page
    assert window._tabs.widget(1) is window._inference_page
    assert window._tabs.widget(2) is window._model_designer


def test_designer_tab_reselection_preserves_page_identity(qtbot) -> None:
    """Navigating to the designer tab and back leaves every tab object and
    the tab ordering untouched (re-entry preserves page state)."""
    window = MainWindow()
    qtbot.addWidget(window)
    designer = window._model_designer
    pages = [window._tabs.widget(i) for i in range(window._tabs.count())]
    assert pages.count(designer) == 1

    window._tabs.setCurrentWidget(designer)
    window._tabs.setCurrentWidget(window._training_page)
    window._tabs.setCurrentWidget(designer)

    assert window._tabs.widget(0) is window._training_page
    assert window._tabs.widget(1) is window._inference_page
    assert window._tabs.widget(2) is designer


# -- Model Designer -> Training handoff (CP2) ---------------------------------


def _stub_save_dialog(monkeypatch, path) -> None:
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )


def test_designer_save_for_training_sets_training_path_and_navigates_once(tmp_path, monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window._tabs.setCurrentWidget(window._model_designer)
    target = tmp_path / "designed_model.json"
    _stub_save_dialog(monkeypatch, target)

    window._model_designer._on_use_for_training_clicked()

    expected = str(Path(target).resolve())
    assert target.exists()
    # Ordinary canonical model-definition file: externally loadable.
    load_model_spec(target)
    # The designer output lands in the *existing* Model JSON input widget --
    # the same field _build_request() reads on line
    # `model_json_path=self._model_json_edit.text().strip()`.
    assert window._training_page._model_json_edit.text() == expected
    assert window._tabs.currentWidget() is window._training_page
    assert window._training_page.is_training_active() is False


def test_designer_navigation_happens_exactly_once_per_action(tmp_path, monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window._tabs.setCurrentWidget(window._model_designer)
    training_index = window._tabs.indexOf(window._training_page)
    switches: list[int] = []
    window._tabs.currentChanged.connect(switches.append)
    _stub_save_dialog(monkeypatch, tmp_path / "m.json")

    window._model_designer._on_use_for_training_clicked()

    # Exactly one navigation, and it lands on Training.
    assert switches == [training_index]
    assert window._tabs.currentWidget() is window._training_page


def test_designer_dialog_cancel_does_not_change_training_path_or_navigate(monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window._training_page._model_json_edit.setText("/existing/authored.json")
    window._tabs.setCurrentWidget(window._model_designer)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
    )

    window._model_designer._on_use_for_training_clicked()

    assert window._training_page._model_json_edit.text() == "/existing/authored.json"
    assert window._tabs.currentWidget() is window._model_designer
    assert window._training_page.is_training_active() is False


def test_designer_failed_save_does_not_change_training_path_or_navigate(tmp_path, monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window._training_page._model_json_edit.setText("/existing/authored.json")
    window._tabs.setCurrentWidget(window._model_designer)
    window._model_designer._name_edit.setText("")  # invalid: ModelSpec.name must be non-empty
    target = tmp_path / "must_not_exist.json"
    _stub_save_dialog(monkeypatch, target)

    window._model_designer._on_use_for_training_clicked()

    assert not target.exists()
    assert window._model_designer._status_label.text().startswith("Save failed:")
    assert window._training_page._model_json_edit.text() == "/existing/authored.json"
    assert window._tabs.currentWidget() is window._model_designer


def test_designer_handoff_then_manual_override_is_authoritative(tmp_path, monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    _stub_save_dialog(monkeypatch, tmp_path / "designed.json")

    window._model_designer._on_use_for_training_clicked()
    window._training_page._model_json_edit.setText("/manually/chosen.json")

    # Manual override after the handoff wins -- the designer does not lock or
    # re-assert the field.
    assert window._training_page._model_json_edit.text() == "/manually/chosen.json"


def test_designer_repeated_use_updates_path_each_time(tmp_path, monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    first = tmp_path / "first.json"
    _stub_save_dialog(monkeypatch, first)
    window._model_designer._on_use_for_training_clicked()
    assert window._training_page._model_json_edit.text() == str(Path(first).resolve())

    window._tabs.setCurrentWidget(window._model_designer)
    second = tmp_path / "second.json"
    _stub_save_dialog(monkeypatch, second)
    window._model_designer._on_use_for_training_clicked()
    assert window._training_page._model_json_edit.text() == str(Path(second).resolve())
    assert window._tabs.currentWidget() is window._training_page


def test_designer_signal_has_one_coordinator_call_per_emission(monkeypatch, qtbot) -> None:
    """The single construction-time connection must not accumulate across
    re-entry or repeated handoffs."""
    window = MainWindow()
    qtbot.addWidget(window)
    calls: list[str] = []
    original_setter = window._training_page.set_model_json_path

    def record_path(path: str) -> None:
        calls.append(path)
        original_setter(path)

    monkeypatch.setattr(window._training_page, "set_model_json_path", record_path)
    training_index = window._tabs.indexOf(window._training_page)
    navigation: list[int] = []
    window._tabs.currentChanged.connect(navigation.append)

    window._tabs.setCurrentWidget(window._model_designer)
    navigation.clear()
    window._model_designer.model_saved_for_training.emit("C:/models/first.json")
    assert calls == ["C:/models/first.json"]
    assert navigation == [training_index]

    window._tabs.setCurrentWidget(window._model_designer)
    navigation.clear()
    window._model_designer.model_saved_for_training.emit("C:/models/second.json")
    assert calls == ["C:/models/first.json", "C:/models/second.json"]
    assert navigation == [training_index]


def test_designer_training_reentry_preserves_both_page_states(tmp_path, monkeypatch, qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    designer = window._model_designer
    designer._name_edit.setText("reentry_model")
    target = tmp_path / "reentry.json"
    _stub_save_dialog(monkeypatch, target)

    designer._on_use_for_training_clicked()
    expected_path = str(Path(target).resolve())
    assert window._tabs.currentWidget() is window._training_page

    window._tabs.setCurrentWidget(designer)

    assert designer._name_edit.text() == "reentry_model"
    assert window._training_page._model_json_edit.text() == expected_path
    assert window._tabs.count() == 3


def test_designer_tab_does_not_join_close_coordination(tmp_path, monkeypatch, qtbot) -> None:
    """The designer adds no async work: an idle close is still immediate and
    the centralized close-pending state is untouched by the designer tab."""
    window = MainWindow()
    qtbot.addWidget(window)
    _stub_save_dialog(monkeypatch, tmp_path / "m.json")
    window._model_designer._on_use_for_training_clicked()
    window.show()

    window.close()

    assert window.isVisible() is False
    assert window._close_pending is False
    assert window._training_close_done is True
    assert window._inference_close_done is True


# -- idle close ------------------------------------------------------------------


def test_close_while_idle_closes_immediately(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    window.close()

    assert window.isVisible() is False


# -- close while training active --------------------------------------------------


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
    training_page = window._training_page
    training_page._controller = TrainingController(backend=blocking_backend)
    _fill_minimum_valid_training_fields(training_page, tmp_path)

    training_page._on_start_clicked()
    assert backend_started.wait(timeout=5)

    window.close()  # 학습 중 -- 다이얼로그(Yes로 monkeypatch됨) -> stop 요청, 아직 안 닫힘
    assert window.isVisible() is True
    assert training_page._status_label.text() == "Stopping after current epoch..."

    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)


def test_close_while_training_declined_keeps_window_open(tmp_path, monkeypatch, qtbot) -> None:
    let_backend_finish = threading.Event()
    backend_started = threading.Event()

    def blocking_backend(request, *, progress_callback=None, should_stop=None):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_result()

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    training_page = window._training_page
    training_page._controller = TrainingController(backend=blocking_backend)
    _fill_minimum_valid_training_fields(training_page, tmp_path)

    training_page._on_start_clicked()
    assert backend_started.wait(timeout=5)

    window.close()
    assert window.isVisible() is True

    let_backend_finish.set()
    qtbot.waitUntil(lambda: training_page.is_training_active() is False, timeout=5000)


# -- close while inference active --------------------------------------------------


def test_close_while_inference_active_defers_until_natural_completion(tmp_path, monkeypatch, qtbot) -> None:
    """Inference에는 취소 API가 없다 -- close 확인 후에도 backend는
    끝까지 자연스럽게 실행되고, 실제 close는 그 뒤 cleanup이 끝난
    다음에만 일어나야 한다."""
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    inference_page = window._inference_page
    inference_page._controller = InferenceController(backend=blocking_backend)
    _fill_minimum_valid_inference_fields(inference_page, tmp_path)

    inference_page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    window.close()
    assert window.isVisible() is True
    assert inference_page.is_inference_active() is True  # not cancelled

    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)
    assert inference_page._thread is None
    assert inference_page._worker is None


def test_close_while_inference_failing_still_completes(tmp_path, monkeypatch, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def failing_blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        raise ValueError("boom")

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    inference_page = window._inference_page
    inference_page._controller = InferenceController(backend=failing_blocking_backend)
    _fill_minimum_valid_inference_fields(inference_page, tmp_path)

    inference_page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    window.close()
    assert window.isVisible() is True

    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)


# -- close while folder inference active -----------------------------------------


def _folder_aggregate() -> FolderInferenceResult:
    return FolderInferenceResult(
        items=(
            ImageOutcome(
                image_path=Path("a.png"),
                result=_fake_inference_result(),
                error=None,
            ),
        )
    )


def _fill_minimum_valid_folder_fields(page: InferencePage, tmp_path) -> None:
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._folder_path_edit.setText(str(tmp_path / "images"))
    page._mode_combo.setCurrentText("Folder")


def test_close_while_folder_inference_active_defers_until_natural_completion(
    tmp_path, monkeypatch, qtbot
) -> None:
    """폴더 추론도 취소 API가 없다 -- close 확인 후에도 folder worker는
    끝까지 실행되고, 실제 close는 그 뒤 cleanup이 끝난 다음에만 일어난다.
    MainWindow는 여전히 InferencePage를 정확히 하나만 갖고, 기존 비동기
    close 조정 경로(request_close -> close_requested)를 그대로 쓴다."""
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _folder_aggregate()

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    pages = [window._tabs.widget(i) for i in range(window._tabs.count())]
    assert pages.count(window._inference_page) == 1

    inference_page = window._inference_page
    inference_page._folder_controller = FolderInferenceController(backend=blocking_backend)
    _fill_minimum_valid_folder_fields(inference_page, tmp_path)

    inference_page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    window.close()
    assert window.isVisible() is True
    assert inference_page.is_inference_active() is True  # not cancelled

    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)
    assert inference_page._folder_thread is None
    assert inference_page._folder_worker is None
    assert inference_page._folder_results_table.rowCount() == 1


def test_close_while_folder_inference_failing_still_completes(tmp_path, monkeypatch, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def failing_blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        raise FolderInferenceError("no supported images in folder")

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    inference_page = window._inference_page
    inference_page._folder_controller = FolderInferenceController(backend=failing_blocking_backend)
    _fill_minimum_valid_folder_fields(inference_page, tmp_path)

    inference_page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    window.close()
    assert window.isVisible() is True

    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)
    assert window._close_pending is False
    assert inference_page._folder_thread is None
    assert inference_page._folder_worker is None


def test_close_while_folder_inference_active_requests_cooperative_cancel(
    tmp_path, monkeypatch, qtbot
) -> None:
    """Phase 12 CP3: an accepted close during an active folder run asks the
    page for a *cooperative* cancel (CP1 `should_cancel` at the next image
    boundary) and then defers the real close through the existing async
    `request_close -> close_requested` coordination -- no forced stop, no
    blocking wait. Single-image close behavior is untouched."""
    paused, resume = threading.Event(), threading.Event()

    def gated_cooperative_backend(request, *, progress_callback=None, should_cancel=None):
        names = ("a.png", "b.png", "c.png")
        outcomes: list = []
        if progress_callback is not None:
            progress_callback(FolderInferenceProgress(total=len(names), completed=0, succeeded=0, failed=0))
        for index, name in enumerate(names):
            if index == 1:
                paused.set()
                assert resume.wait(timeout=5)
            if should_cancel is not None and should_cancel():
                raise FolderInferenceCancelled(FolderInferenceResult(items=tuple(outcomes)), len(names))
            outcomes.append(
                ImageOutcome(image_path=Path(name), result=_fake_inference_result(), error=None)
            )
            if progress_callback is not None:
                progress_callback(
                    FolderInferenceProgress(
                        total=len(names), completed=len(outcomes), succeeded=len(outcomes), failed=0
                    )
                )
        return FolderInferenceResult(items=tuple(outcomes))

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    pages = [window._tabs.widget(i) for i in range(window._tabs.count())]
    assert pages.count(window._inference_page) == 1

    inference_page = window._inference_page
    inference_page._folder_controller = FolderInferenceController(backend=gated_cooperative_backend)
    _fill_minimum_valid_folder_fields(inference_page, tmp_path)

    inference_page._on_run_clicked()
    qtbot.waitUntil(paused.is_set, timeout=5000)

    window.close()
    assert window.isVisible() is True
    assert inference_page._folder_cancelling is True  # cooperative cancel requested
    assert inference_page.is_inference_active() is True  # deferred, not forced

    resume.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)
    assert window._close_pending is False
    assert inference_page._folder_thread is None
    assert inference_page._folder_worker is None
    # cooperative cancel took effect at the image boundary -- partial batch,
    # still exportable, and never shown as a fatal failure.
    assert inference_page._status_label.text().startswith("Cancelled: processed 1 of 3")
    assert inference_page._folder_export_source is not None
    assert inference_page._folder_results_table.rowCount() == 1


# -- duplicate close requests while pending ----------------------------------------


def test_repeated_close_requests_do_not_duplicate_dialog_or_retry(tmp_path, monkeypatch, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()
    question_calls = {"n": 0}

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    def fake_question(*a, **k):
        question_calls["n"] += 1
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    inference_page = window._inference_page
    inference_page._controller = InferenceController(backend=blocking_backend)
    _fill_minimum_valid_inference_fields(inference_page, tmp_path)

    inference_page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    window.close()
    window.close()  # while pending -- must not show another dialog or reconnect signals
    window.close()
    assert question_calls["n"] == 1
    assert window.isVisible() is True

    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)
    assert window._close_pending is False


# -- completion ordering -------------------------------------------------------------


@pytest.mark.parametrize("finish_training_first", [True, False])
def test_close_retries_exactly_once_regardless_of_completion_order(
    tmp_path, monkeypatch, qtbot, finish_training_first
) -> None:
    training_started = threading.Event()
    let_training_finish = threading.Event()
    inference_started = threading.Event()
    let_inference_finish = threading.Event()

    def blocking_training_backend(request, *, progress_callback=None, should_stop=None):
        training_started.set()
        assert let_training_finish.wait(timeout=5)
        return _fake_result()

    def blocking_inference_backend(request):
        inference_started.set()
        assert let_inference_finish.wait(timeout=5)
        return _fake_inference_result()

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    training_page = window._training_page
    training_page._controller = TrainingController(backend=blocking_training_backend)
    _fill_minimum_valid_training_fields(training_page, tmp_path)

    inference_page = window._inference_page
    inference_page._controller = InferenceController(backend=blocking_inference_backend)
    _fill_minimum_valid_inference_fields(inference_page, tmp_path)

    training_page._on_start_clicked()
    inference_page._on_run_clicked()
    assert training_started.wait(timeout=5)
    assert inference_started.wait(timeout=5)

    window.close()
    assert window.isVisible() is True

    if finish_training_first:
        let_training_finish.set()
        qtbot.waitUntil(lambda: training_page.is_training_active() is False, timeout=5000)
        assert window.isVisible() is True  # inference still active -- window must stay open
        let_inference_finish.set()
    else:
        let_inference_finish.set()
        qtbot.waitUntil(lambda: inference_page.is_inference_active() is False, timeout=5000)
        assert window.isVisible() is True  # training still active -- window must stay open
        let_training_finish.set()

    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)
    assert window._close_pending is False


# -- reference consistency after cleanup --------------------------------------------


def test_references_consistent_after_deferred_close(tmp_path, monkeypatch, qtbot) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _fake_inference_result()

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    inference_page = window._inference_page
    inference_page._controller = InferenceController(backend=blocking_backend)
    _fill_minimum_valid_inference_fields(inference_page, tmp_path)

    inference_page._on_run_clicked()
    assert backend_started.wait(timeout=5)

    window.close()
    let_backend_finish.set()
    qtbot.waitUntil(lambda: window.isVisible() is False, timeout=5000)

    assert window._training_page is not None
    assert window._inference_page is not None
    assert window._tabs.widget(0) is window._training_page
    assert window._tabs.widget(1) is window._inference_page
    assert inference_page._thread is None
    assert inference_page._worker is None
