"""Phase 13 CP1: reusable local-image preview widget.

required-tests id: phase13_cp1_image_preview_component

``ImagePreview`` is a small, self-contained ``QWidget`` that shows a single
*local* image file for visual confirmation only. It is deliberately unaware of
inference: it never imports the training/inference/application stack, never
loads model artifacts, never touches the network or CUDA, and never mutates
the source file. Decoding goes through Qt's own ``QImageReader`` with
``setAutoTransform(True)`` so EXIF orientation is honoured on the formats that
carry it. The decoded pixmap is owned solely by this widget's ``QLabel`` --
nothing is stored in a module-level cache.

The widget is always in exactly one of three states, and every transition is
reachable deterministically:

* **placeholder** -- the initial state and the state after :meth:`clear`; a
  short neutral message and no pixmap.
* **image** -- a valid decode; the pixmap is scaled *down only*, preserving
  aspect ratio, so it can never exceed :data:`PREVIEW_MAX_SIZE`. A large
  source image therefore cannot stretch the surrounding layout to its native
  dimensions; an already-small image is shown at native size.
* **unavailable** -- a missing path, a non-file, an unsupported format, or a
  corrupt file; a short neutral message, no pixmap, and no exception is
  allowed to escape into the Qt event loop.

Loading a valid image after an unavailable one -- or calling :meth:`clear`
after a valid image -- fully replaces the previous state: the stale pixmap and
the stale error flag are both dropped.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImageIOHandler, QImageReader, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

PREVIEW_MAX_SIZE = QSize(320, 320)
"""Documented upper bound (device-independent pixels) for the displayed
pixmap. Aspect ratio is always preserved, so at most one axis reaches this
bound; a source image already smaller than this is shown at native size."""

PLACEHOLDER_TEXT = "No image selected"
"""Neutral text for the initial and post-:meth:`ImagePreview.clear` states."""

UNAVAILABLE_TEXT = "Preview unavailable"
"""Concise text for a missing / non-file / unsupported / corrupt input."""


def _transformation_swaps_axes(
    transformation: QImageIOHandler.Transformation,
) -> bool:
    """Return whether Qt's orientation transform exchanges width and height.

    ``Transformation`` is a flag enum: every 90/270-degree variant, including
    the mirror/flip combinations, contains ``TransformationRotate90``.
    """
    rotate_90 = QImageIOHandler.Transformation.TransformationRotate90
    return bool(int(transformation.value) & int(rotate_90.value))


def _bounded_source_size(
    natural_size: QSize,
    preview_size: QSize,
    transformation: QImageIOHandler.Transformation,
) -> QSize | None:
    """Calculate a down-only decode request in source coordinates.

    The preview bound is expressed after auto transformation, whereas
    ``QImageReader.setScaledSize`` expects the source coordinate system.
    """
    if (
        not natural_size.isValid()
        or natural_size.isEmpty()
        or not preview_size.isValid()
        or preview_size.isEmpty()
    ):
        return None
    swaps_axes = _transformation_swaps_axes(transformation)
    displayed = (
        QSize(natural_size.height(), natural_size.width())
        if swaps_axes
        else QSize(natural_size)
    )
    if (
        displayed.width() <= preview_size.width()
        and displayed.height() <= preview_size.height()
    ):
        return QSize(natural_size)
    bounded_display = displayed.scaled(
        preview_size,
        Qt.AspectRatioMode.KeepAspectRatio,
    )
    return (
        QSize(bounded_display.height(), bounded_display.width())
        if swaps_axes
        else bounded_display
    )


class ImagePreview(QWidget):
    """Bounded, aspect-ratio-preserving preview of one local image file."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        preview_size: QSize | None = None,
    ) -> None:
        super().__init__(parent)
        self._preview_size = (
            QSize(preview_size) if preview_size is not None else QSize(PREVIEW_MAX_SIZE)
        )
        self._has_image = False
        self._unavailable = False

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._apply_placeholder()

    # -- public API ---------------------------------------------------------

    def preview_size(self) -> QSize:
        """A *copy* of the bounding preview size (mutating it is harmless)."""
        return QSize(self._preview_size)

    def has_image(self) -> bool:
        return self._has_image

    def is_unavailable(self) -> bool:
        return self._unavailable

    def status_text(self) -> str:
        """The message-label text: a placeholder / unavailable string, or
        ``""`` while a decoded pixmap is shown."""
        return self._label.text()

    def pixmap(self) -> QPixmap | None:
        """The currently displayed pixmap, or ``None`` in the placeholder /
        unavailable states."""
        current = self._label.pixmap()
        if current is None or current.isNull():
            return None
        return current

    def clear(self) -> None:
        """Return to the neutral placeholder state, dropping any pixmap or
        error state."""
        self._apply_placeholder()

    def load_image(self, path: "str | os.PathLike[str] | None") -> bool:
        """Decode and display ``path``. Returns ``True`` on success; on any
        failure the widget enters the unavailable state and returns ``False``
        without raising."""
        pixmap = self._decode(path)
        if pixmap is None:
            self._apply_unavailable()
            return False
        if (
            pixmap.width() > self._preview_size.width()
            or pixmap.height() > self._preview_size.height()
        ):
            pixmap = pixmap.scaled(
                self._preview_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._label.clear()
        self._label.setPixmap(pixmap)
        self._has_image = True
        self._unavailable = False
        return True

    # -- internal state helpers ------------------------------------------

    def _apply_placeholder(self) -> None:
        self._label.clear()
        self._label.setText(PLACEHOLDER_TEXT)
        self._has_image = False
        self._unavailable = False

    def _apply_unavailable(self) -> None:
        self._label.clear()
        self._label.setText(UNAVAILABLE_TEXT)
        self._has_image = False
        self._unavailable = True

    def _decode(self, path: "str | os.PathLike[str] | None") -> QPixmap | None:
        """Best-effort local decode via Qt's own image APIs. Returns ``None``
        for every unusable input instead of propagating an exception."""
        try:
            raw = os.fspath(path)  # type: ignore[arg-type]
        except TypeError:
            return None
        if isinstance(raw, bytes):
            raw = os.fsdecode(raw)
        try:
            if not os.path.isfile(raw):
                return None
        except (OSError, ValueError):
            return None
        reader = QImageReader(raw)
        # Honour EXIF orientation on formats that record it (JPEG, TIFF, ...).
        reader.setAutoTransform(True)
        natural_size = reader.size()
        if not natural_size.isValid() or natural_size.isEmpty():
            # Never fall back to an unbounded full decode when metadata cannot
            # provide a safe source size.
            return None
        transformation = reader.transformation()
        decode_size = _bounded_source_size(
            natural_size,
            self._preview_size,
            transformation,
        )
        if decode_size is None:
            return None
        if decode_size != natural_size:
            # QImageReader expects source-coordinate dimensions here. This is
            # deliberately set before read(), allowing capable format handlers
            # to decode near the requested size instead of allocating native
            # dimensions first.
            reader.setScaledSize(decode_size)
        image = reader.read()
        if image.isNull():
            return None
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return None
        return pixmap
