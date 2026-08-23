"""Fixed, reviewable validation profiles.

Commands are intentionally data, not shell strings. The runner prepends the
active Python executable and always starts them with ``shell=False``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    python_args: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    steps: tuple[Step, ...]


_SYNTAX = Step(
    "python-syntax",
    ("-m", "compileall", "-q", "src", "scripts", "tools"),
    180,
)

PROFILES: dict[str, Profile] = {
    "syntax": Profile(
        "syntax",
        "Compile every project Python module without running tests.",
        (_SYNTAX,),
    ),
    "orchestrator": Profile(
        "orchestrator",
        "Validate the Claude-Codex loop and this project harness.",
        (
            _SYNTAX,
            Step(
                "tool-tests",
                ("-m", "pytest", "tests/tools", "-q", "-p", "no:cacheprovider"),
                300,
            ),
        ),
    ),
    "phase6c": Profile(
        "phase6c",
        "Run the focused GUI, controller, and Qt-worker regression gate.",
        (
            _SYNTAX,
            Step(
                "phase6c-tests",
                (
                    "-m", "pytest",
                    "tests/gui/test_inference_page.py",
                    "tests/gui/test_main_window.py",
                    "tests/gui/test_training_page.py",
                    "tests/application/test_inference_controller.py",
                    "tests/gui/test_qt_inference_worker.py",
                    "-q", "-p", "no:cacheprovider",
                ),
                600,
            ),
        ),
    ),
    "full": Profile(
        "full",
        "Compile the project and run the complete pytest suite once.",
        (
            _SYNTAX,
            Step(
                "full-tests",
                ("-m", "pytest", "-q", "-p", "no:cacheprovider"),
                1800,
            ),
        ),
    ),
}
