"""Phase 13 CP1: ImagePreview widget tests.

required-tests id: phase13_cp1_image_preview_component

Covers the reusable local-image preview component in isolation:

* neutral placeholder for the initial and post-``clear`` states,
* a concise "unavailable" state for missing / non-file / unsupported /
  corrupt inputs, produced without an exception escaping into Qt,
* orientation-aware Qt decoding (EXIF orientation honoured where the format
  and Pillow test helper support it),
* aspect-ratio-preserving display bounded by a documented preview size
  (large sources scaled down, small sources left at native size),
* deterministic clear / reload behaviour that drops stale pixmap and stale
  error state.

All fixtures are local temporary images written with Qt's own ``QImage`` (or
Pillow, only for the EXIF case, guarded by ``importorskip``). Assertions are
programmatic on widget state and pixmap dimensions -- no screenshots, no
sleeps, no timing races. No pytest marker is introduced.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage, QImageIOHandler, QPixmap, qRgb

import image_ai_studio.gui.image_preview as image_preview
from image_ai_studio.gui.image_preview import ImagePreview


def _make_preview(qtbot, **kwargs) -> ImagePreview:
    preview = ImagePreview(**kwargs)
    qtbot.addWidget(preview)
    return preview


def _write_image(path: Path, width: int, height: int, fmt: str = "PNG") -> Path:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(qRgb(90, 140, 210))
    assert image.save(str(path), fmt), f"failed to write test {fmt} at {path}"
    return path


# -- initial / cleared placeholder state -------------------------------------


def test_initial_state_is_neutral_placeholder(qtbot) -> None:
    preview = _make_preview(qtbot)
    assert preview.status_text() == image_preview.PLACEHOLDER_TEXT
    assert preview.has_image() is False
    assert preview.is_unavailable() is False
    assert preview.pixmap() is None


def test_clear_on_fresh_widget_is_idempotent(qtbot) -> None:
    preview = _make_preview(qtbot)
    preview.clear()
    assert preview.status_text() == image_preview.PLACEHOLDER_TEXT
    assert preview.has_image() is False
    assert preview.is_unavailable() is False
    assert preview.pixmap() is None


# -- documented, bounded preview size --------------------------------------


def test_preview_size_is_a_bounded_documented_constant(qtbot) -> None:
    preview = _make_preview(qtbot)
    size = preview.preview_size()
    assert isinstance(size, QSize)
    assert size == image_preview.PREVIEW_MAX_SIZE
    assert (size.width(), size.height()) == (320, 320)


def test_preview_size_accessor_returns_a_copy(qtbot) -> None:
    preview = _make_preview(qtbot)
    size = preview.preview_size()
    size.setWidth(9999)
    assert preview.preview_size().width() == 320


# -- valid decode + bounded aspect-ratio-preserving display ----------------


def test_loads_valid_png_and_shows_pixmap(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "ok.png", 200, 150)
    preview = _make_preview(qtbot)

    assert preview.load_image(path) is True
    assert preview.has_image() is True
    assert preview.is_unavailable() is False
    assert preview.status_text() == ""
    assert preview.pixmap() is not None


def test_loads_jpeg_through_qt_image_reader(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "photo.jpg", 120, 90, fmt="JPG")
    preview = _make_preview(qtbot)

    assert preview.load_image(path) is True
    assert preview.has_image() is True
    assert preview.pixmap() is not None


def test_accepts_both_str_and_path_inputs(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "p.png", 40, 40)

    from_str = _make_preview(qtbot)
    assert from_str.load_image(str(path)) is True
    assert from_str.has_image() is True

    from_path = _make_preview(qtbot)
    assert from_path.load_image(path) is True
    assert from_path.has_image() is True


def test_large_image_scaled_down_preserving_aspect_ratio(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "wide.png", 800, 400)
    preview = _make_preview(qtbot)

    assert preview.load_image(path) is True
    pm = preview.pixmap()
    assert pm is not None
    assert (pm.width(), pm.height()) == (320, 160)
    assert pm.width() <= image_preview.PREVIEW_MAX_SIZE.width()
    assert pm.height() <= image_preview.PREVIEW_MAX_SIZE.height()
    assert pm.width() / pm.height() == 800 / 400


def test_tall_image_bounded_by_height(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "tall.png", 200, 500)
    preview = _make_preview(qtbot)

    assert preview.load_image(path) is True
    pm = preview.pixmap()
    assert (pm.width(), pm.height()) == (128, 320)


def test_very_large_image_never_exceeds_preview_bounds(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "huge.png", 2000, 1500)
    preview = _make_preview(qtbot)

    assert preview.load_image(path) is True
    pm = preview.pixmap()
    assert pm.width() <= 320 and pm.height() <= 320
    # aspect ratio preserved and one axis is pinned to the bound
    assert pm.width() == 320 or pm.height() == 320
    assert pm.width() / pm.height() == pytest.approx(2000 / 1500)


def test_small_image_is_not_upscaled(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "tiny.png", 12, 8)
    preview = _make_preview(qtbot)

    assert preview.load_image(path) is True
    pm = preview.pixmap()
    assert (pm.width(), pm.height()) == (12, 8)


def test_custom_preview_size_bounds_the_pixmap(tmp_path, qtbot) -> None:
    preview = _make_preview(qtbot, preview_size=QSize(64, 64))
    path = _write_image(tmp_path / "big.png", 400, 200)

    assert preview.load_image(path) is True
    pm = preview.pixmap()
    assert (pm.width(), pm.height()) == (64, 32)
    assert pm.width() <= 64 and pm.height() <= 64


def test_bounded_decode_request_precedes_read_and_handles_orientation(
    tmp_path, qtbot, monkeypatch
) -> None:
    """Decode bounds are independently specified and applied before read()."""
    source = tmp_path / "reader-double.img"
    source.write_bytes(b"reader double only")
    transformation = QImageIOHandler.Transformation
    scenarios = [
        # natural, displayed bound, orientation, expected source decode
        (QSize(8000, 4000), QSize(320, 320), transformation.TransformationNone, QSize(320, 160)),
        (QSize(4000, 8000), QSize(320, 320), transformation.TransformationNone, QSize(160, 320)),
        (QSize(400, 800), QSize(300, 150), transformation.TransformationRotate90, QSize(150, 300)),
        (QSize(400, 800), QSize(300, 150), transformation.TransformationRotate270, QSize(150, 300)),
        (QSize(400, 800), QSize(300, 150), transformation.TransformationMirrorAndRotate90, QSize(150, 300)),
        (QSize(400, 800), QSize(300, 150), transformation.TransformationFlipAndRotate90, QSize(150, 300)),
        (QSize(12, 8), QSize(320, 320), transformation.TransformationNone, None),
        (QSize(), QSize(320, 320), transformation.TransformationNone, "unavailable"),
    ]

    for natural, bound, orientation, expected in scenarios:
        calls: list[object] = []

        class RecordingReader:
            def __init__(self, path: str) -> None:
                assert path == str(source)
                self.scaled_size: QSize | None = None

            def setAutoTransform(self, enabled: bool) -> None:
                calls.append(("setAutoTransform", enabled))

            def size(self) -> QSize:
                calls.append("size")
                return QSize(natural)

            def transformation(self):
                calls.append("transformation")
                return orientation

            def setScaledSize(self, size: QSize) -> None:
                self.scaled_size = QSize(size)
                calls.append(("setScaledSize", QSize(size)))

            def read(self) -> QImage:
                calls.append(("read", QSize(self.scaled_size) if self.scaled_size else None))
                source_size = self.scaled_size or natural
                swaps = bool(
                    int(orientation.value)
                    & int(transformation.TransformationRotate90.value)
                )
                result_size = (
                    QSize(source_size.height(), source_size.width())
                    if swaps
                    else QSize(source_size)
                )
                return QImage(result_size, QImage.Format.Format_RGB32)

        monkeypatch.setattr(image_preview, "QImageReader", RecordingReader)
        preview = _make_preview(qtbot, preview_size=bound)
        loaded = preview.load_image(source)

        assert calls[0:2] == [("setAutoTransform", True), "size"]
        if expected == "unavailable":
            assert loaded is False
            assert preview.is_unavailable() is True
            assert "transformation" not in calls
            assert not any(isinstance(call, tuple) and call[0] == "read" for call in calls)
            continue

        assert loaded is True
        assert calls[2] == "transformation"
        read_calls = [call for call in calls if isinstance(call, tuple) and call[0] == "read"]
        assert len(read_calls) == 1
        if expected is None:
            assert not any(
                isinstance(call, tuple) and call[0] == "setScaledSize"
                for call in calls
            )
            assert read_calls[0][1] is None
        else:
            assert calls[3] == ("setScaledSize", expected)
            assert read_calls[0] == ("read", expected)
        pixmap = preview.pixmap()
        assert pixmap is not None
        assert pixmap.width() <= bound.width()
        assert pixmap.height() <= bound.height()


# -- unavailable state for bad inputs, no exception into Qt ----------------


def test_missing_path_produces_unavailable_state(tmp_path, qtbot) -> None:
    preview = _make_preview(qtbot)

    assert preview.load_image(tmp_path / "does_not_exist.png") is False
    assert preview.is_unavailable() is True
    assert preview.has_image() is False
    assert preview.status_text() == image_preview.UNAVAILABLE_TEXT
    assert preview.pixmap() is None


def test_directory_path_produces_unavailable_state(tmp_path, qtbot) -> None:
    a_dir = tmp_path / "a_dir"
    a_dir.mkdir()
    preview = _make_preview(qtbot)

    assert preview.load_image(a_dir) is False
    assert preview.is_unavailable() is True
    assert preview.has_image() is False
    assert preview.pixmap() is None


def test_unsupported_file_produces_unavailable_state(tmp_path, qtbot) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("just some text, not an image", encoding="utf-8")
    preview = _make_preview(qtbot)

    assert preview.load_image(notes) is False
    assert preview.is_unavailable() is True
    assert preview.has_image() is False
    assert preview.status_text() == image_preview.UNAVAILABLE_TEXT
    assert preview.pixmap() is None


def test_corrupt_image_produces_unavailable_state_without_raising(tmp_path, qtbot) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not a real png body" * 8)
    preview = _make_preview(qtbot)

    # Must not raise -- the failure surfaces only as the unavailable state.
    assert preview.load_image(broken) is False
    assert preview.is_unavailable() is True
    assert preview.has_image() is False
    assert preview.pixmap() is None


# -- deterministic clear / reload transitions -----------------------------


def test_valid_image_after_invalid_clears_stale_error_state(tmp_path, qtbot) -> None:
    preview = _make_preview(qtbot)
    assert preview.load_image(tmp_path / "missing.png") is False
    assert preview.is_unavailable() is True

    good = _write_image(tmp_path / "good.png", 50, 50)
    assert preview.load_image(good) is True
    assert preview.has_image() is True
    assert preview.is_unavailable() is False
    assert preview.status_text() == ""
    assert preview.pixmap() is not None


def test_clear_after_valid_image_removes_pixmap_and_restores_placeholder(tmp_path, qtbot) -> None:
    preview = _make_preview(qtbot)
    assert preview.load_image(_write_image(tmp_path / "g.png", 60, 60)) is True
    assert preview.pixmap() is not None

    preview.clear()
    assert preview.has_image() is False
    assert preview.is_unavailable() is False
    assert preview.status_text() == image_preview.PLACEHOLDER_TEXT
    assert preview.pixmap() is None


def test_invalid_image_after_valid_replaces_pixmap_with_unavailable(tmp_path, qtbot) -> None:
    preview = _make_preview(qtbot)
    assert preview.load_image(_write_image(tmp_path / "g.png", 60, 60)) is True

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"definitely not an image")
    assert preview.load_image(bad) is False
    assert preview.is_unavailable() is True
    assert preview.has_image() is False
    assert preview.status_text() == image_preview.UNAVAILABLE_TEXT
    assert preview.pixmap() is None


def test_reloading_replaces_the_previous_pixmap(tmp_path, qtbot) -> None:
    preview = _make_preview(qtbot)
    assert preview.load_image(_write_image(tmp_path / "a.png", 800, 400)) is True
    first = preview.pixmap()
    assert (first.width(), first.height()) == (320, 160)

    assert preview.load_image(_write_image(tmp_path / "b.png", 300, 300)) is True
    second = preview.pixmap()
    assert (second.width(), second.height()) == (300, 300)
    assert preview.has_image() is True
    assert preview.is_unavailable() is False


# -- non-mutation of the source file ------------------------------------


def test_load_does_not_mutate_the_source_file(tmp_path, qtbot) -> None:
    path = _write_image(tmp_path / "src.png", 64, 48)
    before_bytes = path.read_bytes()
    before_mtime_ns = path.stat().st_mtime_ns

    preview = _make_preview(qtbot)
    assert preview.load_image(path) is True

    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime_ns


# -- orientation-aware decoding ---------------------------------------


def test_exif_orientation_is_applied_when_present(tmp_path, qtbot) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    source = pil_image.new("RGB", (48, 24), (30, 90, 150))  # landscape on disk
    exif = source.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW -> displayed as 24x48 portrait
    path = tmp_path / "rotated.jpg"
    source.save(path, format="JPEG", exif=exif)

    preview = _make_preview(qtbot)
    assert preview.load_image(path) is True
    pm = preview.pixmap()
    assert pm is not None
    # autoTransform honoured the tag: portrait, not the raw landscape.
    assert pm.height() > pm.width()
    assert (pm.width(), pm.height()) == (24, 48)


# -- isolation guarantees ------------------------------------------


def test_component_module_avoids_inference_and_heavy_dependencies() -> None:
    src = Path(image_preview.__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in src.splitlines()
        if line.startswith(("import ", "from "))
    ]
    blob = "\n".join(import_lines)
    for forbidden in (
        "torch",
        "cuda",
        "numpy",
        "urllib",
        "requests",
        "httpx",
        "socket",
        "http.client",
        "model_definition",
        "image_ai_studio.inference",
        "image_ai_studio.training",
        "image_ai_studio.application",
    ):
        assert forbidden not in blob, f"unexpected import in image_preview: {forbidden!r}"


def test_component_module_has_no_global_decoded_image_cache() -> None:
    for name, value in vars(image_preview).items():
        assert not isinstance(value, QPixmap), f"module-level QPixmap: {name}"
        assert not isinstance(value, QImage), f"module-level QImage: {name}"


def test_widget_creates_no_background_worker_or_thread(qtbot) -> None:
    from PySide6.QtCore import QThread

    preview = _make_preview(qtbot)
    for value in vars(preview).values():
        assert not isinstance(value, QThread)
