"""Phase 13 CP3: single-image input preview -- real MainWindow graduation.

required-tests id: phase13_cp3_image_preview_cpu_graduation

Phase 13 CP1 (``tests/gui/test_image_preview.py``) pins the reusable
``ImagePreview`` component in isolation, and Phase 13 CP2 (the ``Phase 13
CP2`` section of ``tests/gui/test_inference_page.py``) pins how
``InferencePage`` wires that component to the Single Image input against a
directly constructed page. This module's only job is to prove those pieces
compose in the **real ``MainWindow`` + ``InferencePage``** object graph:
the preview is created exactly once, follows the existing Browse action and
a deterministically *committed* manual path, recovers from an unreadable
path, survives a Single Image -> Folder -> Single Image round trip without
leaving a stale picture, and coexists with a running (but **fake**)
asynchronous single-image inference lifecycle without a duplicate widget.

No real model forward happens here: the injected ``InferenceController``
backend is a fake that returns a canned ``InferenceResult`` (optionally
after a ``threading.Event`` gate), so the async QThread lifecycle is
exercised without CUDA, a state_dict, or a portable bundle. The preview
itself still performs its genuine bounded local image decode via Qt on
deterministic ``tmp_path`` fixtures. Observation is state-based only
(widget flags, ``pixmap()`` dimensions, ``findChildren``) plus
``threading.Event`` / ``qtbot.waitUntil`` -- no screenshot comparison, no
``time.sleep`` as correctness, no filesystem watcher, no new dependency.

The Phase 6 through Phase 12 request / thread / cancellation / export /
close contracts and the public inference APIs are only consumed here, never
changed, and no pytest marker is introduced.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtGui import QImage, qRgb
from PySide6.QtWidgets import QFileDialog

import image_ai_studio.gui.image_preview as image_preview_module
import image_ai_studio.gui.main_window as main_window_module
from image_ai_studio.application.folder_inference_controller import FolderInferenceController
from image_ai_studio.application.inference_controller import InferenceController
from image_ai_studio.gui.image_preview import ImagePreview
from image_ai_studio.gui.inference_page import InferencePage
from image_ai_studio.gui.main_window import MainWindow
from image_ai_studio.inference.single_image_inference import InferenceResult

_SINGLE = "Single Image"
_FOLDER = "Folder"
_PREVIEW_BOUND = image_preview_module.PREVIEW_MAX_SIZE


# -- deterministic local image fixtures (Qt's own writer, no PIL) ------------


def _write_png(path: Path, width: int = 48, height: int = 32) -> Path:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(qRgb(70, 130, 180))
    assert image.save(str(path), "PNG"), f"failed to write test PNG at {path}"
    return path


def _write_jpeg(path: Path, width: int = 120, height: int = 90) -> Path:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(qRgb(30, 90, 150))
    assert image.save(str(path), "JPG"), f"failed to write test JPEG at {path}"
    return path


def _write_corrupt(path: Path) -> Path:
    # Supported extension, not a decodable image -- ImagePreview must land in
    # its own "unavailable" state without an exception reaching Qt.
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not a real png body" * 8)
    return path


# -- fake single-image inference backend (no real forward) ------------------


def _canned_result() -> InferenceResult:
    return InferenceResult(
        predicted_index=0,
        predicted_class="cat",
        confidence=0.9,
        probabilities={"cat": 0.9, "dog": 0.1},
        inference_duration_seconds=0.01,
    )


def _folder_backend_must_not_run(request):  # pragma: no cover - guard only
    raise AssertionError("folder backend must not run in Phase 13 CP3 preview tests")


def _build_window(qtbot, monkeypatch, *, single_backend):
    """Build the real ``MainWindow`` but have it construct an
    ``InferencePage`` with fake controllers injected -- no production file is
    modified, and no real model forward can occur (same test-only technique
    as ``tests/gui/test_folder_cancellation_integration.py``)."""

    class _InjectedInferencePage(InferencePage):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("controller", InferenceController(backend=single_backend))
            kwargs.setdefault(
                "folder_controller",
                FolderInferenceController(backend=_folder_backend_must_not_run),
            )
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(main_window_module, "InferencePage", _InjectedInferencePage)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._inference_page
    window._tabs.setCurrentWidget(page)
    return window, page


def _previews(page) -> list[ImagePreview]:
    return page.findChildren(ImagePreview)


def _patch_open_dialog(monkeypatch, returned_path) -> None:
    def _fake(parent, caption="", directory="", filter="", *args, **kwargs):
        return (str(returned_path), filter)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(_fake))


# -- exactly one labelled preview, wired to the Single Image input ----------


def test_real_main_window_has_exactly_one_labelled_preview_in_single_mode(qtbot, monkeypatch) -> None:
    _window, page = _build_window(qtbot, monkeypatch, single_backend=lambda r: _canned_result())

    previews = _previews(page)
    assert len(previews) == 1
    assert previews[0] is page._image_preview
    assert "Preview" in page._image_preview_group.title()
    assert page._image_preview_group.isHidden() is False
    assert page._image_preview.has_image() is False
    assert page._image_preview.is_unavailable() is False
    assert page._image_preview.status_text() == image_preview_module.PLACEHOLDER_TEXT
    # the preview widget owns no background thread of its own
    assert not any(isinstance(v, QThread) for v in vars(page._image_preview).values())


# -- Browse + committed manual path both refresh it, bounded aspect ratio ---


def test_browse_and_committed_manual_path_drive_preview_within_aspect_bounds(
    tmp_path, qtbot, monkeypatch
) -> None:
    _window, page = _build_window(qtbot, monkeypatch, single_backend=lambda r: _canned_result())

    wide = _write_png(tmp_path / "wide.png", 800, 400)
    _patch_open_dialog(monkeypatch, wide)
    page._browse_image_button.click()

    assert page._image_path_edit.text() == str(wide)
    assert page._image_preview.has_image() is True
    assert page._image_preview.is_unavailable() is False
    pm = page._image_preview.pixmap()
    assert pm is not None
    # scaled down only, aspect ratio preserved, never exceeds the documented bound
    assert pm.width() <= _PREVIEW_BOUND.width()
    assert pm.height() <= _PREVIEW_BOUND.height()
    assert pm.width() == _PREVIEW_BOUND.width() or pm.height() == _PREVIEW_BOUND.height()
    assert pm.width() / pm.height() == 800 / 400

    # a committed manual edit (Enter / focus-out) refreshes it too; a small
    # source is shown at native size, never upscaled.
    small = _write_jpeg(tmp_path / "small.jpg", 120, 90)
    page._image_path_edit.setText(str(small))
    page._image_path_edit.editingFinished.emit()
    pm2 = page._image_preview.pixmap()
    assert (pm2.width(), pm2.height()) == (120, 90)
    assert page._image_path_edit.text() == str(small)  # field text untouched
    assert len(_previews(page)) == 1


# -- unreadable -> corrupt -> valid recovery, request path untouched -------


def test_unreadable_then_valid_path_recovers_without_touching_request(
    tmp_path, qtbot, monkeypatch
) -> None:
    _window, page = _build_window(qtbot, monkeypatch, single_backend=lambda r: _canned_result())
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))

    missing = tmp_path / "gone.png"
    page._image_path_edit.setText(str(missing))
    page._image_path_edit.editingFinished.emit()
    assert page._image_preview.is_unavailable() is True
    assert page._image_preview.has_image() is False
    assert page._image_preview.status_text() == image_preview_module.UNAVAILABLE_TEXT

    corrupt = _write_corrupt(tmp_path / "broken.png")
    page._image_path_edit.setText(str(corrupt))
    page._image_path_edit.editingFinished.emit()  # must not raise into Qt
    assert page._image_preview.is_unavailable() is True
    assert page._image_preview.has_image() is False

    good = _write_png(tmp_path / "good.png", 50, 40)
    page._image_path_edit.setText(str(good))
    page._image_path_edit.editingFinished.emit()
    assert page._image_preview.has_image() is True
    assert page._image_preview.is_unavailable() is False
    assert page._image_preview.status_text() == ""

    # request mapping / validation path is untouched by any of the above
    request = page._build_request()
    assert request.image_path == good
    assert request.state_dict_path == tmp_path / "best_model_state_dict.pt"
    assert request.class_mapping_path == tmp_path / "class_mapping.json"
    assert len(_previews(page)) == 1


# -- Single -> Folder -> Single round trip leaves no stale preview ---------


def test_single_folder_single_round_trip_leaves_no_stale_preview(
    tmp_path, qtbot, monkeypatch
) -> None:
    _window, page = _build_window(qtbot, monkeypatch, single_backend=lambda r: _canned_result())

    first = _write_png(tmp_path / "first.png", 40, 30)
    page._image_path_edit.setText(str(first))
    page._image_path_edit.editingFinished.emit()
    assert page._image_preview.has_image() is True

    page._mode_combo.setCurrentText(_FOLDER)
    assert page._image_preview_group.isHidden() is True
    assert page._image_preview.has_image() is False  # cleared, not merely hidden
    assert page._image_preview.is_unavailable() is False
    assert page._image_preview.status_text() == image_preview_module.PLACEHOLDER_TEXT

    # path changes while folder mode is active (no commit signal fires here)
    second = _write_png(tmp_path / "second.png", 60, 20)
    page._image_path_edit.setText(str(second))

    page._mode_combo.setCurrentText(_SINGLE)
    assert page._image_preview_group.isHidden() is False
    assert page._image_preview.has_image() is True  # deterministically re-mirrored
    pm = page._image_preview.pixmap()
    assert (pm.width(), pm.height()) == (60, 20)  # the current path, not the stale one

    # empty path on return -> neutral placeholder, still exactly one widget
    page._image_path_edit.setText("")
    page._mode_combo.setCurrentText(_FOLDER)
    page._mode_combo.setCurrentText(_SINGLE)
    assert page._image_preview.has_image() is False
    assert page._image_preview.is_unavailable() is False
    assert page._image_preview.status_text() == image_preview_module.PLACEHOLDER_TEXT
    assert len(_previews(page)) == 1


# -- coexistence with a fake asynchronous single-image inference lifecycle --


def test_preview_coexists_with_fake_async_inference_lifecycle(tmp_path, qtbot, monkeypatch) -> None:
    backend_started = threading.Event()
    let_backend_finish = threading.Event()

    def blocking_backend(request):
        backend_started.set()
        assert let_backend_finish.wait(timeout=5)
        return _canned_result()

    _window, page = _build_window(qtbot, monkeypatch, single_backend=blocking_backend)

    img = _write_png(tmp_path / "input.png", 200, 150)
    page._training_output_dir_edit.setText(str(tmp_path))
    page._model_json_edit.setText(str(tmp_path / "model.json"))
    page._image_path_edit.setText(str(img))
    page._image_path_edit.editingFinished.emit()
    assert page._image_preview.has_image() is True

    page._run_button.click()
    assert backend_started.wait(timeout=5)

    # mid-run: preview still shows the committed image, exactly one widget,
    # and the preview started no second inference path.
    assert page._status_label.text() == "Running"
    assert page._image_preview.has_image() is True
    assert page._image_preview.is_unavailable() is False
    assert len(_previews(page)) == 1

    let_backend_finish.set()
    qtbot.waitUntil(lambda: page._run_button.isEnabled(), timeout=5000)

    assert page._status_label.text() == "Finished"
    assert page._thread is None and page._worker is None
    assert page._image_preview.has_image() is True  # the run never disturbed the preview
    assert len(_previews(page)) == 1

    # the preview still refreshes normally after a completed run
    other = _write_png(tmp_path / "after.png", 24, 24)
    page._image_path_edit.setText(str(other))
    page._image_path_edit.editingFinished.emit()
    pm = page._image_preview.pixmap()
    assert (pm.width(), pm.height()) == (24, 24)
    assert len(_previews(page)) == 1


# -- preview activity alone starts no inference and spawns no extra widget --


def test_preview_activity_starts_no_inference_and_no_extra_widget(
    tmp_path, qtbot, monkeypatch
) -> None:
    def single_backend_must_not_run(request):  # pragma: no cover - guard only
        raise AssertionError("single-image backend must not run for preview-only activity")

    _window, page = _build_window(qtbot, monkeypatch, single_backend=single_backend_must_not_run)

    img = _write_png(tmp_path / "pick.png", 300, 300)
    _patch_open_dialog(monkeypatch, img)
    page._browse_image_button.click()
    page._image_path_edit.editingFinished.emit()
    page._mode_combo.setCurrentText(_FOLDER)
    page._mode_combo.setCurrentText(_SINGLE)
    page._image_path_edit.setText(str(tmp_path / "missing.png"))
    page._image_path_edit.editingFinished.emit()

    assert page._thread is None and page._worker is None
    assert page._folder_thread is None and page._folder_worker is None
    assert page._controller.state == "idle"
    assert page._folder_controller.state == "idle"
    assert len(_previews(page)) == 1
    assert not any(isinstance(v, QThread) for v in vars(page._image_preview).values())
