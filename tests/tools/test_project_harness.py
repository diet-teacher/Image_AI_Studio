from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.project_harness import PROFILES
from tools.project_harness.runner import CommandResult, dry_run, execute_profile


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n", encoding="utf-8")
    (root / "tests").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, shell=False)
    return root


def test_profiles_use_python_argument_arrays() -> None:
    assert {"syntax", "orchestrator", "phase6c", "full"} == set(PROFILES)
    for profile in PROFILES.values():
        for step in profile.steps:
            assert step.python_args
            assert not isinstance(step.python_args, str)
            assert all(isinstance(value, str) and value for value in step.python_args)


def test_dry_run_invokes_no_process_and_exposes_exact_argv(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    report = dry_run(root, PROFILES["syntax"])
    assert report["state"] == "DRY_RUN"
    assert report["processes_invoked"] is False
    assert report["steps"][0]["argv"][0] == sys.executable
    assert report["steps"][0]["argv"][1:3] == ["-m", "compileall"]


def test_execute_isolates_environment_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr("tools.project_harness.runner.doctor", lambda value: {"healthy": True, "worktree_dirty": False})
    monkeypatch.setattr(
        "tools.project_harness.runner._git",
        lambda *args: CommandResult("PASS", 0, 0.0, "abc123\n", ""),
    )
    calls = []

    def fake_runner(argv, cwd, env, timeout):
        calls.append((list(argv), cwd, dict(env), timeout))
        return CommandResult("PASS", 0, 0.01, "ok", "")

    code, report = execute_profile(root, PROFILES["orchestrator"], command_runner=fake_runner)
    assert code == 0
    assert report["state"] == "PASSED"
    assert len(calls) == 2
    assert all(call[0][0] == sys.executable for call in calls)
    assert all(call[1] == root for call in calls)
    assert all(call[2]["TEMP"] == call[2]["TMP"] == call[2]["TMPDIR"] for call in calls)
    assert all(call[2]["QT_QPA_PLATFORM"] == "offscreen" for call in calls)
    assert calls[0][2]["TEMP"] != calls[1][2]["TEMP"]
    summary = root / ".harness" / "runs" / report["run_id"] / "summary.json"
    assert json.loads(summary.read_text(encoding="utf-8"))["state"] == "PASSED"


def test_failure_stops_later_steps(tmp_path: Path, monkeypatch) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr("tools.project_harness.runner.doctor", lambda value: {"healthy": True, "worktree_dirty": False})
    monkeypatch.setattr(
        "tools.project_harness.runner._git",
        lambda *args: CommandResult("PASS", 0, 0.0, "abc123\n", ""),
    )
    calls = 0

    def failing_runner(argv, cwd, env, timeout):
        nonlocal calls
        calls += 1
        return CommandResult("FAIL", 1, 0.01, "", "failed")

    code, report = execute_profile(root, PROFILES["full"], command_runner=failing_runner)
    assert code == 1
    assert report["state"] == "FAILED"
    assert calls == 1
    assert len(report["steps"]) == 1


def test_require_clean_blocks_before_profile_processes(tmp_path: Path, monkeypatch) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr("tools.project_harness.runner.doctor", lambda value: {"healthy": True, "worktree_dirty": True})

    def forbidden(*args, **kwargs):
        raise AssertionError("profile process must not run")

    code, report = execute_profile(root, PROFILES["syntax"], require_clean=True, command_runner=forbidden)
    assert code == 2
    assert report["state"] == "BLOCKED"
    assert report["reason"] == "DIRTY_WORKTREE"


def test_profile_elapsed_limit_is_applied_to_step_timeout(tmp_path: Path, monkeypatch) -> None:
    root = _repository(tmp_path)
    monkeypatch.setattr("tools.project_harness.runner.doctor", lambda value: {"healthy": True, "worktree_dirty": False})
    monkeypatch.setattr(
        "tools.project_harness.runner._git",
        lambda *args: CommandResult("PASS", 0, 0.0, "abc123\n", ""),
    )
    observed = []
    def fake_runner(argv, cwd, env, timeout):
        observed.append(timeout)
        return CommandResult("PASS", 0, 0.0, "", "")
    code, report = execute_profile(root, PROFILES["syntax"], command_runner=fake_runner,
                                   max_elapsed_seconds=2.0)
    assert code == 0 and report["state"] == "PASSED"
    assert len(observed) == 1 and 0 < observed[0] <= 2.0
