"""Phase 5C: top-level application window. 초기 application scope는
training GUI 하나뿐이므로 tab/navigation/router 없이 `TrainingPage`
하나만 담는다 -- 향후 다른 기능이 추가될 수 있다는 이유로 지금
sidebar/plugin architecture를 만들지 않는다."""
from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QWidget

from image_ai_studio.gui.training_page import TrainingPage


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image AI Studio")

        self._training_page = TrainingPage(self)
        self._training_page.close_requested.connect(self.close)
        self.setCentralWidget(self._training_page)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 -- Qt override naming
        """학습 중이 아니면 바로 닫는다. 학습 중이면 확인 다이얼로그를
        띄우고, 사용자가 동의하면 cooperative stop을 요청한 뒤 실제
        close는 `TrainingPage.close_requested`가 emit될 때까지
        미룬다(`QThread.terminate()`나 강제 종료 없음, GUI thread를
        blocking wait로 얼리지도 않음)."""
        if not self._training_page.is_training_active():
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Training in progress",
            "Training is still running. Stop it and exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        self._training_page.request_stop_and_close()
        event.ignore()  # 실제 close는 training이 안전하게 끝난 뒤 close_requested가 처리한다
