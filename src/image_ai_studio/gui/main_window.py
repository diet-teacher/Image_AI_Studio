"""Phase 5C: top-level application window. Phase 6C CP4는 여기에 완성된
`InferencePage`를 두 번째 tab으로 추가하고, training/inference 둘 다에
대응하는 중앙집중형, non-blocking close coordination을 더한다 --
`QThread.terminate()`나 GUI thread를 얼리는 blocking wait는 여전히
쓰지 않는다."""
from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget, QWidget

from image_ai_studio.gui.inference_page import InferencePage
from image_ai_studio.gui.training_page import TrainingPage


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image AI Studio")

        self._training_page = TrainingPage(self)
        self._inference_page = InferencePage(self)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._training_page, "Training")
        self._tabs.addTab(self._inference_page, "Inference")
        self.setCentralWidget(self._tabs)

        # -- centralized pending-close coordination (CP4) -----------------------
        self._close_pending = False
        self._training_close_done = True
        self._inference_close_done = True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 -- Qt override naming
        """Training과 inference 둘 다 idle이면 바로 닫는다. 하나라도
        active면 확인 다이얼로그를 띄우고, 사용자가 동의하면 training에는
        기존 cooperative stop을 요청하고 inference는 취소 없이 자연스럽게
        끝나기를 기다린다 -- 실제 close는 active였던 모든 page가 cleanup을
        끝내고 `close_requested`를 emit한 뒤에만 정확히 한 번 재시도된다.
        Close가 이미 pending 상태면(사용자가 다시 close를 요청하는 경우)
        다이얼로그/signal 연결/cleanup 요청을 중복해서 만들지 않는다."""
        if self._close_pending:
            event.ignore()
            return

        training_active = self._training_page.is_training_active()
        inference_active = self._inference_page.is_inference_active()
        if not training_active and not inference_active:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Work in progress",
            "Training or inference is still running. Stop and exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        self._close_pending = True
        self._training_close_done = not training_active
        self._inference_close_done = not inference_active

        if training_active:
            self._training_page.close_requested.connect(self._on_training_close_ready)
            self._training_page.request_stop_and_close()
        if inference_active:
            self._inference_page.close_requested.connect(self._on_inference_close_ready)
            self._inference_page.request_close()

        event.ignore()  # 실제 close는 active했던 page들의 close_requested가 모두 도착한 뒤에만 일어난다

    # -- per-page close-ready handlers -------------------------------------------

    def _on_training_close_ready(self) -> None:
        self._training_page.close_requested.disconnect(self._on_training_close_ready)
        self._training_close_done = True
        self._maybe_finish_pending_close()

    def _on_inference_close_ready(self) -> None:
        self._inference_page.close_requested.disconnect(self._on_inference_close_ready)
        self._inference_close_done = True
        self._maybe_finish_pending_close()

    def _maybe_finish_pending_close(self) -> None:
        if not self._close_pending:
            return
        if not (self._training_close_done and self._inference_close_done):
            return
        self._close_pending = False
        self.close()  # 이 시점엔 둘 다 idle이므로 closeEvent가 즉시 accept한다
