from __future__ import annotations

import subprocess
from unittest.mock import patch

from image_ai_studio.tools.inspect_environment import _run


class _FakeCompletedProcess:
    def __init__(self, stdout: bytes | None, stderr: bytes | None) -> None:
        self.stdout = stdout
        self.stderr = stderr


def test_windows_cp949_stdout_is_decoded_with_replace() -> None:
    # "환경" (CP949-encoded), not valid UTF-8 -- would raise/mangle under
    # PYTHONUTF8=1 forcing a UTF-8 decode of Windows console output.
    cp949_bytes = "환경".encode("cp949")
    fake = _FakeCompletedProcess(stdout=cp949_bytes, stderr=b"")

    with patch("image_ai_studio.tools.inspect_environment.sys.platform", "win32"), \
        patch("image_ai_studio.tools.inspect_environment.subprocess.run", return_value=fake) as run_mock:
        result = _run(["some-command"])

    assert result == cp949_bytes.decode("mbcs", errors="replace")
    run_mock.assert_called_once()


def test_stdout_falls_back_to_stderr_when_stdout_empty() -> None:
    fake = _FakeCompletedProcess(stdout=b"", stderr=b"stderr message")

    with patch("image_ai_studio.tools.inspect_environment.platform.system", return_value="Linux"), \
        patch("image_ai_studio.tools.inspect_environment.subprocess.run", return_value=fake):
        result = _run(["some-command"])

    assert result == "stderr message"


def test_both_streams_empty_returns_none() -> None:
    fake = _FakeCompletedProcess(stdout=b"", stderr=b"")

    with patch("image_ai_studio.tools.inspect_environment.subprocess.run", return_value=fake):
        result = _run(["some-command"])

    assert result is None


def test_both_streams_none_returns_none() -> None:
    fake = _FakeCompletedProcess(stdout=None, stderr=None)

    with patch("image_ai_studio.tools.inspect_environment.subprocess.run", return_value=fake):
        result = _run(["some-command"])

    assert result is None


def test_timeout_returns_none() -> None:
    with patch(
        "image_ai_studio.tools.inspect_environment.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="some-command", timeout=15),
    ):
        result = _run(["some-command"])

    assert result is None


def test_missing_executable_returns_none() -> None:
    with patch(
        "image_ai_studio.tools.inspect_environment.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        result = _run(["some-command"])

    assert result is None


def test_permission_error_returns_none() -> None:
    with patch(
        "image_ai_studio.tools.inspect_environment.subprocess.run",
        side_effect=PermissionError,
    ):
        result = _run(["some-command"])

    assert result is None


def test_run_uses_no_shell_and_captures_bytes() -> None:
    fake = _FakeCompletedProcess(stdout=b"ok", stderr=b"")

    with patch("image_ai_studio.tools.inspect_environment.subprocess.run", return_value=fake) as run_mock:
        _run(["cmake", "--version"])

    run_mock.assert_called_once_with(
        ["cmake", "--version"], capture_output=True, timeout=15, shell=False
    )
