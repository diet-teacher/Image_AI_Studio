"""Execution and durable reporting for the project quality harness."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .profiles import Profile

TAIL_LIMIT = 4_000


@dataclass(frozen=True)
class CommandResult:
    status: str
    exit_code: int | None
    duration_seconds: float
    stdout: str
    stderr: str
    error: str | None = None

    def summary(self) -> dict:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 3),
            "stdout_tail": self.stdout[-TAIL_LIMIT:],
            "stderr_tail": self.stderr[-TAIL_LIMIT:],
            "error": self.error,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_command(
    argv: Sequence[str], cwd: Path, env: dict[str, str], timeout: int
) -> CommandResult:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", shell=False,
        )
    except OSError as exc:
        return CommandResult(
            "START_FAILED", None, time.monotonic() - started, "", "",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate(process)
        stdout, stderr = process.communicate()
        partial_stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        partial_stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(
            "TIMEOUT", process.returncode, time.monotonic() - started,
            partial_stdout + stdout, partial_stderr + stderr,
            f"timeout after {timeout} seconds",
        )
    except KeyboardInterrupt:
        _terminate(process)
        raise

    status = "PASS" if process.returncode == 0 else "FAIL"
    return CommandResult(status, process.returncode, time.monotonic() - started, stdout, stderr)


def _git(root: Path, *args: str) -> CommandResult:
    return run_command(("git", *args), root, os.environ.copy(), 30)


def doctor(root: Path) -> dict:
    version_ok = sys.version_info >= (3, 10)
    git_check = _git(root, "rev-parse", "--show-toplevel")
    status = _git(root, "status", "--short") if git_check.status == "PASS" else None
    imports: dict[str, bool] = {}
    for module in ("pytest", "torch", "PySide6"):
        try:
            __import__(module)
            imports[module] = True
        except (ImportError, OSError):
            imports[module] = False
    healthy = version_ok and git_check.status == "PASS" and all(imports.values())
    return {
        "healthy": healthy,
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "executable": sys.executable,
            "supported": version_ok,
        },
        "repository": str(root),
        "git_repository": git_check.status == "PASS",
        "worktree_dirty": bool(status and status.stdout.strip()),
        "git_diagnostics": git_check.summary(),
        "imports": imports,
    }


def _step_environment(step_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    temp_dir = step_dir / "temp"
    pycache_dir = step_dir / "pycache"
    temp_dir.mkdir(parents=True, exist_ok=False)
    pycache_dir.mkdir(parents=True, exist_ok=False)
    env.update(
        {
            "TEMP": str(temp_dir), "TMP": str(temp_dir), "TMPDIR": str(temp_dir),
            "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "PYTHONPYCACHEPREFIX": str(pycache_dir),
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    return env


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute_profile(
    root: Path,
    profile: Profile,
    *,
    require_clean: bool = False,
    command_runner: Callable[[Sequence[str], Path, dict[str, str], int], CommandResult] = run_command,
    max_elapsed_seconds: float | None = None,
) -> tuple[int, dict]:
    health = doctor(root)
    if not health["healthy"]:
        return 2, {"state": "BLOCKED", "reason": "DOCTOR_FAILED", "doctor": health}
    if require_clean and health["worktree_dirty"]:
        return 2, {"state": "BLOCKED", "reason": "DIRTY_WORKTREE", "doctor": health}

    run_id = uuid.uuid4().hex
    run_dir = root / ".harness" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    git_head = _git(root, "rev-parse", "HEAD")
    report = {
        "run_id": run_id, "profile": profile.name, "started_at": utc_now(),
        "repository": str(root), "python": sys.executable,
        "base_commit": git_head.stdout.strip() if git_head.status == "PASS" else None,
        "initial_worktree_dirty": health["worktree_dirty"],
        "state": "RUNNING", "steps": [],
    }
    _write_json(run_dir / "manifest.json", report)

    exit_code = 0
    profile_started = time.monotonic()
    try:
        for index, step in enumerate(profile.steps, start=1):
            step_dir = run_dir / f"{index:03d}-{step.name}"
            step_dir.mkdir(parents=True, exist_ok=False)
            timeout = step.timeout_seconds
            try:
                env = _step_environment(step_dir)
            except OSError as exc:
                result = CommandResult(
                    "START_FAILED", None, 0.0, "", "",
                    f"{type(exc).__name__}: {exc}",
                )
            else:
                argv = (sys.executable, *step.python_args)
                if max_elapsed_seconds is not None:
                    remaining = max_elapsed_seconds - (time.monotonic() - profile_started)
                    if remaining <= 0:
                        result = CommandResult("TIMEOUT", None, 0.0, "", "", "profile elapsed-time limit reached")
                    else:
                        timeout = min(timeout, max(0.001, remaining))
                        result = command_runner(argv, root, env, timeout)
                else:
                    result = command_runner(argv, root, env, timeout)
                (step_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
                (step_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

            step_record = {
                "name": step.name,
                "argv": [sys.executable, *step.python_args],
                "timeout_seconds": timeout,
                "artifact_dir": str(step_dir),
                **result.summary(),
            }
            report["steps"].append(step_record)
            _write_json(step_dir / "result.json", step_record)
            if result.status != "PASS":
                exit_code = 1 if result.status == "FAIL" else 2
                break
    except KeyboardInterrupt:
        report["state"] = "BLOCKED"
        report["reason"] = "INTERRUPTED"
        exit_code = 130
    else:
        report["state"] = "PASSED" if exit_code == 0 else "FAILED" if exit_code == 1 else "BLOCKED"

    report["finished_at"] = utc_now()
    final_status = _git(root, "status", "--short")
    report["final_worktree_dirty"] = bool(final_status.stdout.strip()) if final_status.status == "PASS" else None
    _write_json(run_dir / "summary.json", report)
    return exit_code, report


def dry_run(root: Path, profile: Profile, require_clean: bool = False) -> dict:
    return {
        "state": "DRY_RUN", "repository": str(root), "profile": profile.name,
        "require_clean": require_clean, "python": sys.executable,
        "steps": [
            {
                "name": step.name,
                "argv": [sys.executable, *step.python_args],
                "timeout_seconds": step.timeout_seconds,
            }
            for step in profile.steps
        ],
        "processes_invoked": False,
    }
