#!/usr/bin/env python
"""Phase 5C: Image AI Studio Training GUI launcher.

얇은 launcher다 -- business logic은 전혀 없다. 실제 GUI 구성은
`image_ai_studio.gui.main_window.MainWindow`/`training_page.TrainingPage`
가 담당하고, 학습 실행은 Phase 5B의 `TrainingController`/
`QtTrainingWorker`를 그대로 쓴다.

사용법::

    python scripts/run_gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from image_ai_studio.gui.main_window import MainWindow  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
