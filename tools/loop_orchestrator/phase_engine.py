from __future__ import annotations

import base64
import difflib
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tools.project_harness.profiles import PROFILES
from tools.project_harness.runner import execute_profile

from .goal import GoalError, path_allowed
from .models import VerifierResult
from .process import (API_CONNECTION_ERROR, MODEL_BUDGET_EXHAUSTED, MODEL_MAX_TURNS,
                      PROCESS_START_FAILED, TIMEOUT, ProcessFailure)
from .prompts import MAKER, VERIFIER
from .repository import base_commit, file_snapshot, git, worktree_snapshot

DIFF_LIMIT = 40_000
CONTENT_LIMIT = 40_000
PROTECTED_LOCAL_FILES = (
    ".loop/config.json", ".loop/budget.json",
    ".claude/settings.local.json", ".vscode/settings.json",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_digest(snapshot: dict) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _entry(path: Path) -> dict:
    if path.is_symlink():
        target = os.readlink(path)
        return {"kind": "symlink", "sha256": hashlib.sha256(target.encode("utf-8", "replace")).hexdigest(),
                "target": target}
    if not path.exists():
        return {"kind": "missing", "sha256": None}
    if not path.is_file():
        return {"kind": "non_file", "sha256": None}
    data = path.read_bytes()
    result = {"kind": "file", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    if len(data) <= CONTENT_LIMIT:
        result["content_b64"] = base64.b64encode(data).decode("ascii")
    return result


def repository_snapshot(root: Path) -> dict[str, dict]:
    tracked = git(root, "ls-files", "--cached", "--deleted").splitlines()
    untracked = git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    return {relative: _entry(root / relative) for relative in sorted(set(tracked + untracked))}


def protected_snapshot(root: Path) -> dict[str, dict]:
    return {relative: _entry(root / relative) for relative in PROTECTED_LOCAL_FILES}


def git_guard(root: Path) -> dict:
    return {
        "head": base_commit(root),
        "staged": git(root, "diff", "--cached", "--name-only", "--").splitlines(),
        "repository": repository_snapshot(root),
        "protected": protected_snapshot(root),
    }


def _decode(entry: dict | None) -> bytes | None:
    encoded = (entry or {}).get("content_b64")
    return base64.b64decode(encoded) if isinstance(encoded, str) else None


def snapshot_delta(before: dict[str, dict], after: dict[str, dict]) -> tuple[list[str], str, dict]:
    paths = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    parts, diagnostics = [], {}
    for path in paths:
        old_entry = before.get(path, {"kind": "missing", "sha256": None})
        new_entry = after.get(path, {"kind": "missing", "sha256": None})
        item = {"before_kind": old_entry.get("kind"), "after_kind": new_entry.get("kind"),
                "before_sha256": old_entry.get("sha256"), "after_sha256": new_entry.get("sha256")}
        old, new = _decode(old_entry), _decode(new_entry)
        if old is not None or new is not None:
            try:
                old_text, new_text = (old or b"").decode("utf-8"), (new or b"").decode("utf-8")
            except UnicodeDecodeError:
                item["diff_kind"] = "binary"
                parts.append(f"Binary file changed: {path} ({item['before_sha256']} -> {item['after_sha256']})\n")
            else:
                item["diff_kind"] = "text"
                parts.extend(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                                  fromfile=f"a/{path}", tofile=f"b/{path}"))
        else:
            item["diff_kind"] = "hash_only"
            parts.append(f"Hash-only file change: {path} ({item['before_kind']}:{item['before_sha256']} -> "
                         f"{item['after_kind']}:{item['after_sha256']})\n")
        diagnostics[path] = item
    diff = "".join(parts)
    if len(diff) > DIFF_LIMIT:
        diff = "[diff truncated to final characters]\n" + diff[-DIFF_LIMIT:]
    return paths, diff, diagnostics


def guard_diagnostics(before: dict, after: dict, base: str, allowed: list[str], root: Path,
                      *, require_unchanged: bool = False) -> dict:
    paths, diff, files = snapshot_delta(before["repository"], after["repository"])
    violations = []
    if require_unchanged:
        violations.extend(paths)
    else:
        for path in paths:
            try:
                if not path_allowed(path, allowed, root):
                    violations.append(path)
            except GoalError:
                violations.append(path)
    protected = sorted(path for path in PROTECTED_LOCAL_FILES
                       if before["protected"].get(path) != after["protected"].get(path))
    return {"changed_files": paths, "diff": diff, "file_diagnostics": files,
            "allowed_files_violations": sorted(set(violations)), "protected_file_violations": protected,
            "head_before": before["head"], "head_after": after["head"],
            "head_violation": before["head"] != base or after["head"] != base,
            "staged_before": before["staged"], "staged_after": after["staged"],
            "staged_violation": bool(before["staged"] or after["staged"])}


def has_guard_violation(details: dict) -> bool:
    return bool(details["allowed_files_violations"] or details["protected_file_violations"]
                or details["head_violation"] or details["staged_violation"])


class PhaseEngine:
    def __init__(self, root, config, maker, codex, budget, *, clock=time.monotonic,
                 harness_runner=execute_profile, preflight=None):
        self.root, self.config, self.maker, self.codex, self.budget = root, config, maker, codex, budget
        self.runtime = root / ".loop"
        self.clock, self.harness_runner, self.preflight = clock, harness_runner, preflight
        self._test_engine = None

    def _persist(self, state: dict, role: str, result=None) -> None:
        state["updated_at"] = _now()
        record = {"timestamp": state["updated_at"], "role": role, "exit_status": state["state"],
                  "base_commit": state["base_commit"], "state": state}
        if result is not None:
            record["result"] = result
        self.runtime.mkdir(exist_ok=True)
        (self.runtime / "state.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        run_dir = self.runtime / "runs" / state["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        sequence = len(list(run_dir.glob("*.json"))) + 1
        (run_dir / f"{sequence:03d}-{role}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _stop(self, state: dict, final: str, reason: str, details=None) -> dict:
        state["state"], state["stop_reason"], state["recent_result"] = final, reason, details or {}
        started = state.pop("started_monotonic", None)
        if started is not None:
            state["elapsed_seconds"] = self.clock() - started
        self._persist(state, "orchestrator", details)
        handoff = {"timestamp": _now(), "phase_id": state.get("phase_id"), "run_id": state["run_id"],
                   "base_commit": state["base_commit"], "state": final, "reason": reason,
                   "resume_checkpoint_index": state.get("current_checkpoint_index"), "details": details or {}}
        (self.runtime / "handoff.json").write_text(json.dumps(handoff, indent=2), encoding="utf-8")
        return state

    def _budget_valid(self, role: str) -> tuple[bool, str]:
        decision = self.budget.check(role, starting_checkpoint=True)
        if not decision.allowed:
            return False, decision.code
        return True, "OK"

    def _before_call(self, state: dict, manifest: dict, role: str) -> tuple[bool, str]:
        state["elapsed_seconds"] = self.clock() - state["started_monotonic"]
        if state["elapsed_seconds"] >= manifest["max_elapsed_seconds"]:
            return False, "MAX_ELAPSED_SECONDS"
        if state["model_calls"] >= manifest["max_model_calls"]:
            return False, "MAX_MODEL_CALLS"
        ok, reason = self._budget_valid(role)
        if not ok:
            return False, reason
        if role == "claude":
            per_call = float(self.config.get("claude_max_budget_usd", 0))
            if state["claude_cost_usd"] + per_call > manifest["max_claude_cost_usd"]:
                return False, "MAX_CLAUDE_COST_USD"
        return True, "OK"

    def _initial_state(self, manifest: dict) -> dict:
        return {"record_type": "phase", "phase_id": manifest["phase_id"],
                "manifest_digest": manifest["manifest_digest"], "run_id": uuid.uuid4().hex,
                "base_commit": base_commit(self.root), "state": "RUNNING",
                "current_checkpoint_index": 0, "current_checkpoint_id": manifest["checkpoints"][0]["checkpoint_id"],
                "completed_checkpoints": [], "checkpoints": [], "active_checkpoint": None,
                "model_calls": 0, "maker_calls": 0, "verifier_calls": 0, "planner_calls": 0,
                "claude_cost_usd": 0.0, "elapsed_seconds": 0.0, "final_harness": None,
                "stop_reason": None, "started_at": _now(), "started_monotonic": self.clock()}

    def run(self, manifest: dict, *, execute: bool = False, resume: bool = False) -> dict:
        if not execute:
            return {"record_type": "phase", "phase_id": manifest["phase_id"], "state": "DRY_RUN",
                    "manifest_digest": manifest["manifest_digest"],
                    "checkpoint_ids": [item["checkpoint_id"] for item in manifest["checkpoints"]],
                    "final_harness_profile": manifest["final_harness_profile"], "processes_invoked": False}
        if resume:
            return self._resume(manifest)
        state = self._initial_state(manifest)
        if worktree_snapshot(self.root).strip():
            return self._stop(state, "BLOCKED", "DIRTY_WORKTREE")
        phase_guard = git_guard(self.root)
        state["phase_baseline"] = phase_guard["repository"]
        state["phase_baseline_digest"] = _snapshot_digest(phase_guard["repository"])
        state["phase_baseline_protected"] = phase_guard["protected"]
        state["last_observed_snapshot"] = phase_guard["repository"]
        state["last_observed_protected"] = phase_guard["protected"]
        for role in ("claude", "codex"):
            ok, reason = self._budget_valid(role)
            if not ok:
                return self._stop(state, "BLOCKED", f"BUDGET_PRECHECK: {role}: {reason}")
        if self.preflight is not None:
            failure = self.preflight()
            if failure:
                return self._stop(state, "BLOCKED", "EXECUTABLE_PREFLIGHT_FAILED", failure)
        self._persist(state, "phase-start")
        return self._run_checkpoints(state, manifest, 0)

    def _new_checkpoint(self, state: dict, entry: dict) -> dict:
        guard = git_guard(self.root)
        if guard["head"] != state["base_commit"] or guard["staged"]:
            return {}
        record = {"checkpoint_id": entry["checkpoint_id"], "goal": entry["goal"],
                  "stage": "MAKER_PENDING", "baseline": guard["repository"],
                  "baseline_digest": _snapshot_digest(guard["repository"]),
                  "baseline_protected": guard["protected"], "baseline_head": guard["head"],
                  "baseline_staged": guard["staged"], "last_observed_snapshot": guard["repository"],
                  "last_observed_protected": guard["protected"], "attempts": [], "next_attempt": 0,
                  "session_id": None, "rework": "", "continuation_count": 0, "api_retry_count": 0,
                  "pending_call": None}
        state["active_checkpoint"] = record
        self._persist(state, "checkpoint-baseline", record)
        return record

    def _recovery_limits(self) -> tuple[int, int]:
        def _bounded(value: object, default: int = 1) -> int:
            return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default
        return (_bounded(self.config.get("max_maker_continuations", 1)),
                _bounded(self.config.get("max_api_connection_retries", 1)))

    @staticmethod
    def _full_maker_prompt(entry: dict, goal: dict, record: dict) -> str:
        return MAKER.format(goal=json.dumps(goal), checkpoint=entry["checkpoint_id"], rework=record.get("rework", ""))

    @staticmethod
    def _continuation_prompt(entry: dict, goal: dict) -> str:
        return ("CONTINUE_CHECKPOINT: resume this exact session and finish only "
               f"checkpoint {entry['checkpoint_id']} without restarting or expanding "
               f"scope.\nGOAL:\n{json.dumps(goal)}")

    @staticmethod
    def _pending_call_record(record: dict, attempt: int, call_index: int, recovery_kind, session_id) -> dict:
        return {"attempt": attempt, "call_index": call_index, "recovery_kind": recovery_kind,
                "session_id": session_id, "continuation_count": record["continuation_count"],
                "api_retry_count": record["api_retry_count"]}

    def _pending_call_intent(self, record: dict, entry: dict, goal: dict, attempt: int):
        """Reconstruct the exact next maker call from durable state.

        Returns (session_id, prompt, recovery_kind, call_index, error). ``error`` is
        non-None when a persisted ``pending_call`` exists but is incomplete or
        ambiguous, in which case resume must fail closed rather than guess.
        """
        pending = record.get("pending_call")
        if pending is None:
            attempts = record.get("attempts") or []
            last = attempts[-1] if attempts else None
            if isinstance(last, dict) and last.get("attempt") == attempt and "maker_error" in last:
                return None, None, None, None, "RESUME_RECOVERY_STATE_INCOMPLETE"
            return record.get("session_id"), self._full_maker_prompt(entry, goal, record), None, 0, None
        if not isinstance(pending, dict) or pending.get("attempt") != attempt:
            return None, None, None, None, "RESUME_RECOVERY_STATE_INCOMPLETE"
        if (pending.get("continuation_count") != record.get("continuation_count")
                or pending.get("api_retry_count") != record.get("api_retry_count")):
            return None, None, None, None, "RESUME_RECOVERY_STATE_INCOMPLETE"
        call_index = pending.get("call_index")
        if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 0:
            return None, None, None, None, "RESUME_RECOVERY_STATE_INCOMPLETE"
        recovery_kind, session_id = pending.get("recovery_kind"), pending.get("session_id")
        if recovery_kind is None:
            prompt = self._full_maker_prompt(entry, goal, record)
        elif recovery_kind == "continuation":
            if not isinstance(session_id, str) or not session_id:
                return None, None, None, None, "RESUME_RECOVERY_STATE_INCOMPLETE"
            prompt = self._continuation_prompt(entry, goal)
        elif recovery_kind == "api_retry":
            if session_id is not None:
                return None, None, None, None, "RESUME_RECOVERY_STATE_INCOMPLETE"
            prompt = self._full_maker_prompt(entry, goal, record)
        else:
            return None, None, None, None, "RESUME_RECOVERY_STATE_INCOMPLETE"
        return session_id, prompt, recovery_kind, call_index, None

    def _run_checkpoints(self, state: dict, manifest: dict, start_index: int) -> dict:
        max_continuations, max_api_retries = self._recovery_limits()
        for index in range(start_index, len(manifest["checkpoints"])):
            entry, goal = manifest["checkpoints"][index], manifest["checkpoints"][index]["goal_value"]
            state["current_checkpoint_index"], state["current_checkpoint_id"] = index, entry["checkpoint_id"]
            record = state.get("active_checkpoint") or self._new_checkpoint(state, entry)
            if not record:
                return self._stop(state, "BLOCKED", "CHECKPOINT_BASELINE_GIT_INTEGRITY_FAILURE")
            record.setdefault("continuation_count", 0)
            record.setdefault("api_retry_count", 0)
            record.setdefault("pending_call", None)
            max_reworks = min(manifest["max_rework_rounds"],
                              int(self.config.get("max_rework_rounds", manifest["max_rework_rounds"])))
            for attempt in range(int(record.get("next_attempt", 0)), max_reworks + 1):
                record.update({"stage": "MAKER_RUNNING", "next_attempt": attempt})
                session_id, prompt, recovery_kind, call_index, intent_error = self._pending_call_intent(
                    record, entry, goal, attempt)
                if intent_error:
                    return self._stop(state, "BLOCKED", intent_error, record)
                maker_record = None
                while True:
                    allowed, reason = self._before_call(state, manifest, "claude")
                    if not allowed:
                        return self._stop(state, "BLOCKED", reason, record)
                    record["pending_call"] = self._pending_call_record(record, attempt, call_index,
                                                                       recovery_kind, session_id)
                    state["model_calls"] += 1; state["maker_calls"] += 1
                    self._persist(state, "maker-before", record)
                    before = git_guard(self.root)
                    invocation = None; maker_error = None; interrupted = False
                    try:
                        invocation = self.maker.run(prompt, session_id)
                    except KeyboardInterrupt:
                        interrupted = True
                    except Exception as exc:
                        maker_error = exc
                    after = git_guard(self.root)
                    maker_guard = guard_diagnostics(before, after, state["base_commit"], goal["allowed_files"], self.root)
                    baseline_files, baseline_diff, baseline_diagnostics = snapshot_delta(record["baseline"], after["repository"])
                    record["pending_call"] = None
                    maker_record = {"attempt": attempt, "call_index": call_index, "recovery_kind": recovery_kind,
                                    "guard": maker_guard, "checkpoint_changed_files": baseline_files,
                                    "checkpoint_diff": baseline_diff, "checkpoint_file_diagnostics": baseline_diagnostics}
                    record["attempts"].append(maker_record)
                    record["last_observed_snapshot"], record["last_observed_protected"] = after["repository"], after["protected"]
                    record["stage"] = "MAKER_FINISHED"; self._persist(state, "maker-after", maker_record)

                    cost_invalid, telemetry_session = False, None
                    if invocation is not None:
                        cost = invocation.total_cost_usd
                        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
                            cost_invalid = True
                        else:
                            state["claude_cost_usd"] += float(cost)
                        telemetry_session = invocation.session_id
                        maker_record.update({"session_id": invocation.session_id, "total_cost_usd": cost,
                                             "num_turns": invocation.num_turns, "telemetry": invocation.telemetry,
                                             "maker_status": invocation.result.status})
                    elif isinstance(maker_error, ProcessFailure):
                        meta = maker_error.metadata
                        maker_record["maker_error"] = maker_error.diagnostics()
                        meta_cost = meta.get("total_cost_usd")
                        if isinstance(meta_cost, (int, float)) and not isinstance(meta_cost, bool) and meta_cost >= 0:
                            state["claude_cost_usd"] += float(meta_cost)
                        if isinstance(meta.get("session_id"), str) and meta["session_id"]:
                            telemetry_session = meta["session_id"]
                        maker_record.update({"session_id": telemetry_session, "total_cost_usd": meta.get("total_cost_usd"),
                                             "num_turns": meta.get("num_turns"), "telemetry": {
                                                 key: meta.get(key) for key in (
                                                     "requested_model", "model_usage", "canonical_models",
                                                     "primary_canonical_model", "terminal_reason", "subtype")
                                             }})
                    if telemetry_session:
                        record["session_id"] = telemetry_session
                    maker_record["continuation_count"] = record["continuation_count"]
                    maker_record["api_retry_count"] = record["api_retry_count"]
                    self._persist(state, "maker-result", maker_record)

                    if has_guard_violation(maker_guard):
                        return self._stop(state, "BLOCKED", "MAKER_SAFETY_VIOLATION", maker_record)
                    if invocation is not None and cost_invalid:
                        return self._stop(state, "BLOCKED", "CLAUDE_COST_UNKNOWN", maker_record)
                    if state["claude_cost_usd"] > manifest["max_claude_cost_usd"]:
                        return self._stop(state, "BLOCKED", "MAX_CLAUDE_COST_USD", maker_record)
                    if invocation is not None:
                        if invocation.result.status.upper() == "BLOCKED":
                            return self._stop(state, "BLOCKED", "MAKER_BLOCKED", maker_record)
                        break
                    if interrupted:
                        return self._stop(state, "BLOCKED", "INTERRUPTED", maker_record)

                    kind = maker_error.kind if isinstance(maker_error, ProcessFailure) else None
                    if kind in (MODEL_BUDGET_EXHAUSTED, MODEL_MAX_TURNS):
                        session = maker_error.metadata.get("session_id")
                        next_allowed, _ = self._before_call(state, manifest, "claude")
                        if (isinstance(session, str) and session
                                and record["continuation_count"] < max_continuations and next_allowed):
                            record["continuation_count"] += 1
                            session_id, call_index, recovery_kind = session, call_index + 1, "continuation"
                            prompt = self._continuation_prompt(entry, goal)
                            record["pending_call"] = self._pending_call_record(record, attempt, call_index,
                                                                               recovery_kind, session_id)
                            self._persist(state, "recovery-transition", record)
                            continue
                        return self._stop(state, "BLOCKED", kind, maker_record)
                    if kind == API_CONNECTION_ERROR:
                        next_allowed, _ = self._before_call(state, manifest, "claude")
                        if (not maker_guard["changed_files"]
                                and record["api_retry_count"] < max_api_retries and next_allowed):
                            record["api_retry_count"] += 1
                            session_id, call_index, recovery_kind = None, call_index + 1, "api_retry"
                            record["pending_call"] = self._pending_call_record(record, attempt, call_index,
                                                                               recovery_kind, session_id)
                            self._persist(state, "recovery-transition", record)
                            continue
                        return self._stop(state, "BLOCKED", kind, maker_record)
                    if isinstance(maker_error, ProcessFailure):
                        final = "BLOCKED" if maker_error.kind in {PROCESS_START_FAILED, TIMEOUT} else "FAILED"
                        return self._stop(state, final, f"MAKER_ERROR: {maker_error.kind}", maker_record)
                    maker_record["maker_error"] = {"type": type(maker_error).__name__, "error": str(maker_error)[-4000:]}
                    return self._stop(state, "FAILED", "MAKER_ERROR", maker_record)

                stopped = self._run_tests_and_verifier(state, manifest, record, maker_record, goal, attempt, max_reworks)
                if stopped is not None:
                    if stopped.get("state") in {"BLOCKED", "FAILED", "READY_TO_COMMIT"}:
                        return stopped
                    if stopped.get("checkpoint_passed"):
                        break
            else:
                return self._stop(state, "FAILED", "MAX_REWORK_ROUNDS", record)
        return self._final_harness(state, manifest)

    def _run_tests_and_verifier(self, state, manifest, record, maker_record, goal, attempt, max_reworks):
        before_tests = git_guard(self.root)
        record["stage"] = "TESTS_RUNNING"; self._persist(state, "tests-before", record)
        try:
            tests, missing = self._tests(goal["required_tests"], state["run_id"])
        except KeyboardInterrupt:
            return self._stop(state, "BLOCKED", "INTERRUPTED", record)
        except Exception as exc:
            return self._stop(state, "BLOCKED", "REQUIRED_TEST_INFRASTRUCTURE_FAILURE",
                              {"type": type(exc).__name__, "error": str(exc)[-4000:]})
        after_tests = git_guard(self.root)
        test_guard = guard_diagnostics(before_tests, after_tests, state["base_commit"], [], self.root,
                                       require_unchanged=True)
        maker_record.update({"tests": tests, "test_guard": test_guard})
        record["last_observed_snapshot"], record["last_observed_protected"] = after_tests["repository"], after_tests["protected"]
        record["stage"] = "TESTS_FINISHED"; self._persist(state, "tests", maker_record)
        if has_guard_violation(test_guard):
            return self._stop(state, "BLOCKED", "REQUIRED_TEST_SAFETY_VIOLATION", maker_record)
        if missing:
            return self._stop(state, "BLOCKED", "REQUIRED_TEST_NOT_ALLOWLISTED", {"tests": missing})
        if any(item["status"] in {"START_FAILED", "TIMEOUT"} for item in tests):
            return self._stop(state, "BLOCKED", "REQUIRED_TEST_INFRASTRUCTURE_FAILURE", maker_record)
        allowed, reason = self._before_call(state, manifest, "codex")
        if not allowed:
            return self._stop(state, "BLOCKED", reason, record)
        verifier_before = git_guard(self.root)
        baseline_files, baseline_diff, baseline_diagnostics = snapshot_delta(record["baseline"], verifier_before["repository"])
        phase_files, phase_diff, phase_diagnostics = snapshot_delta(state["phase_baseline"], verifier_before["repository"])
        record["stage"] = "VERIFIER_RUNNING"; self._persist(state, "verifier-before", record)
        state["model_calls"] += 1; state["verifier_calls"] += 1
        verifier = None; verifier_error = None; verifier_interrupted = False
        try:
            verifier = self.codex.verify(VERIFIER.format(
                goal=json.dumps(goal), base=state["base_commit"], checkpoint_files=baseline_files,
                checkpoint_diff=baseline_diff, phase_files=phase_files, phase_diff=phase_diff, tests=tests))
        except KeyboardInterrupt:
            verifier_interrupted = True
        except Exception as exc:
            verifier_error = exc
        verifier_after = git_guard(self.root)
        verifier_guard = guard_diagnostics(verifier_before, verifier_after, state["base_commit"], [], self.root,
                                           require_unchanged=True)
        maker_record.update({"checkpoint_changed_files": baseline_files, "checkpoint_diff": baseline_diff,
                             "checkpoint_file_diagnostics": baseline_diagnostics,
                             "phase_changed_files": phase_files, "phase_diff": phase_diff,
                             "phase_file_diagnostics": phase_diagnostics,
                             "verifier_guard": verifier_guard})
        record["last_observed_snapshot"], record["last_observed_protected"] = verifier_after["repository"], verifier_after["protected"]
        record["stage"] = "VERIFIER_FINISHED"; self._persist(state, "verifier-after", maker_record)
        if has_guard_violation(verifier_guard):
            return self._stop(state, "BLOCKED", "VERIFIER_SAFETY_VIOLATION", maker_record)
        if verifier_interrupted:
            return self._stop(state, "BLOCKED", "INTERRUPTED", maker_record)
        if verifier_error is not None:
            if isinstance(verifier_error, ProcessFailure):
                maker_record["verifier_error"] = verifier_error.diagnostics()
                final = "BLOCKED" if verifier_error.kind in {PROCESS_START_FAILED, TIMEOUT} else "FAILED"
                return self._stop(state, final, f"VERIFIER_ERROR: {verifier_error.kind}", maker_record)
            maker_record["verifier_error"] = {"type": type(verifier_error).__name__,
                                               "error": str(verifier_error)[-4000:]}
            return self._stop(state, "FAILED", "VERIFIER_ERROR", maker_record)
        failed_tests = [item for item in tests if item["status"] != "PASS"]
        if failed_tests and verifier.verdict == "PASS":
            verifier = VerifierResult("FAIL", ["Required tests failed"], [x["name"] for x in failed_tests],
                                      [], None, [], "Fix required tests")
        maker_record["verifier"] = asdict(verifier); self._persist(state, "verifier-result", maker_record)
        if verifier.verdict == "PASS":
            record.update({"stage": "COMPLETE", "final_files": baseline_files,
                           "final_hashes": file_snapshot(self.root), "next_attempt": attempt})
            state["checkpoints"].append(record); state["completed_checkpoints"].append(record["checkpoint_id"])
            state["active_checkpoint"] = None
            state["last_observed_snapshot"] = verifier_after["repository"]
            state["last_observed_protected"] = verifier_after["protected"]
            self._persist(state, "checkpoint-complete", record)
            return {"checkpoint_passed": True}
        if verifier.verdict == "BLOCKED":
            return self._stop(state, "BLOCKED", "VERIFIER_BLOCKED", record)
        if attempt >= max_reworks:
            return self._stop(state, "FAILED", "MAX_REWORK_ROUNDS", record)
        record["rework"] = "REWORK FINDINGS:\n" + json.dumps(asdict(verifier))
        record["next_attempt"] = attempt + 1; record["stage"] = "REWORK_PENDING"
        self._persist(state, "rework", record)
        return None

    def _final_harness(self, state: dict, manifest: dict) -> dict:
        if state["completed_checkpoints"] != [item["checkpoint_id"] for item in manifest["checkpoints"]]:
            return self._stop(state, "BLOCKED", "PHASE_INCOMPLETE")
        if "phase_baseline" not in state:
            return self._stop(state, "BLOCKED", "PHASE_BASELINE_MISSING")
        current = git_guard(self.root)
        phase_files, phase_diff, phase_diagnostics = snapshot_delta(state["phase_baseline"], current["repository"])
        violations = []
        for path_name in phase_files:
            try:
                if not path_allowed(path_name, manifest["allowed_files"], self.root):
                    violations.append(path_name)
            except GoalError:
                violations.append(path_name)
        if violations or current["head"] != state["base_commit"] or current["staged"]:
            return self._stop(state, "BLOCKED", "FINAL_PHASE_SCOPE_VIOLATION", {
                "phase_changed_files": phase_files, "phase_diff": phase_diff,
                "phase_file_diagnostics": phase_diagnostics,
                "allowed_files_violations": sorted(set(violations)),
                "head": current["head"], "staged": current["staged"],
            })
        remaining = manifest["max_elapsed_seconds"] - (self.clock() - state["started_monotonic"])
        if remaining <= 0:
            return self._stop(state, "BLOCKED", "MAX_ELAPSED_SECONDS")
        before = git_guard(self.root)
        try:
            code, report = self.harness_runner(self.root, PROFILES[manifest["final_harness_profile"]],
                                               max_elapsed_seconds=remaining)
        except KeyboardInterrupt:
            after = git_guard(self.root)
            details = {"error": {"type": "KeyboardInterrupt"}, "guard": guard_diagnostics(
                before, after, state["base_commit"], [], self.root, require_unchanged=True)}
            return self._stop(state, "BLOCKED", "FINAL_HARNESS_INTERRUPTED", details)
        except Exception as exc:
            after = git_guard(self.root)
            error = exc.diagnostics() if isinstance(exc, ProcessFailure) else {
                "type": type(exc).__name__, "error": str(exc)[-4000:]}
            details = {"error": error, "guard": guard_diagnostics(
                before, after, state["base_commit"], [], self.root, require_unchanged=True)}
            return self._stop(state, "BLOCKED", "FINAL_HARNESS_START_FAILED", details)
        after = git_guard(self.root)
        harness_guard = guard_diagnostics(before, after, state["base_commit"], [], self.root,
                                          require_unchanged=True)
        state["final_harness"] = {"exit_code": code, **report, "guard": harness_guard}
        self._persist(state, "project-harness", state["final_harness"])
        if has_guard_violation(harness_guard):
            return self._stop(state, "BLOCKED", "FINAL_HARNESS_MODIFIED_WORKTREE", state["final_harness"])
        if code == 0 and report.get("state") == "PASSED":
            return self._stop(state, "READY_TO_COMMIT", "PHASE_COMPLETE")
        if code == 1 or report.get("state") == "FAILED":
            return self._stop(state, "FAILED", "FINAL_HARNESS_FAILED", state["final_harness"])
        return self._stop(state, "BLOCKED", "FINAL_HARNESS_INFRASTRUCTURE_FAILURE", state["final_harness"])

    def _resume_invalid(self, manifest: dict, reason: str, details=None) -> dict:
        return self._stop(self._initial_state(manifest), "BLOCKED", reason, details)

    def _resume(self, manifest: dict) -> dict:
        path = self.runtime / "state.json"
        if not path.is_file():
            return self._resume_invalid(manifest, "RESUME_STATE_MISSING")
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))["state"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            return self._resume_invalid(manifest, "RESUME_STATE_INVALID", {"error": str(exc)[-4000:]})
        required = {"record_type", "phase_id", "manifest_digest", "run_id", "base_commit",
                    "completed_checkpoints", "checkpoints", "current_checkpoint_index"}
        if not isinstance(saved, dict) or saved.get("record_type") != "phase" or not required.issubset(saved):
            return self._resume_invalid(manifest, "RESUME_STATE_NOT_PHASE")
        phase_fields = {"phase_baseline", "phase_baseline_digest", "phase_baseline_protected", "last_observed_snapshot",
                        "last_observed_protected"}
        if not phase_fields.issubset(saved):
            return self._stop(saved, "BLOCKED", "RESUME_INCOMPLETE_CHECKPOINT_STATE")
        if saved["phase_baseline_digest"] != _snapshot_digest(saved["phase_baseline"]):
            return self._stop(saved, "BLOCKED", "RESUME_INCOMPLETE_CHECKPOINT_STATE")
        saved["started_monotonic"] = self.clock() - float(saved.get("elapsed_seconds", 0))
        if saved.get("phase_id") != manifest["phase_id"] or saved.get("manifest_digest") != manifest["manifest_digest"]:
            return self._stop(saved, "BLOCKED", "RESUME_MANIFEST_MISMATCH")
        if saved.get("base_commit") != base_commit(self.root):
            return self._stop(saved, "BLOCKED", "RESUME_BASE_COMMIT_MISMATCH")
        active = saved.get("active_checkpoint"); completed = len(saved.get("completed_checkpoints", []))
        if completed < len(manifest["checkpoints"]):
            baseline_fields = {"checkpoint_id", "goal", "baseline", "baseline_digest", "baseline_protected", "stage",
                               "last_observed_snapshot", "last_observed_protected", "next_attempt",
                               "session_id", "rework", "attempts"}
            expected = manifest["checkpoints"][completed]
            if not isinstance(active, dict) or not baseline_fields.issubset(active):
                return self._stop(saved, "BLOCKED", "RESUME_INCOMPLETE_CHECKPOINT_STATE")
            if active["baseline_digest"] != _snapshot_digest(active["baseline"]):
                return self._stop(saved, "BLOCKED", "RESUME_INCOMPLETE_CHECKPOINT_STATE")
            if active["checkpoint_id"] != expected["checkpoint_id"] or active["goal"] != expected["goal"]:
                return self._stop(saved, "BLOCKED", "RESUME_INCOMPLETE_CHECKPOINT_STATE")
            current = git_guard(self.root)
            if current["head"] != saved["base_commit"] or current["staged"]:
                return self._stop(saved, "BLOCKED", "RESUME_GIT_INTEGRITY_FAILURE")
            if current["repository"] != active["last_observed_snapshot"] or current["protected"] != active["last_observed_protected"]:
                return self._stop(saved, "BLOCKED", "RESUME_EXTERNAL_CHANGE_DETECTED")
            baseline_files, _, _ = snapshot_delta(active["baseline"], current["repository"])
            violations = []
            for path_name in baseline_files:
                try:
                    if not path_allowed(path_name, expected["goal_value"]["allowed_files"], self.root):
                        violations.append(path_name)
                except GoalError:
                    violations.append(path_name)
            if violations:
                return self._stop(saved, "BLOCKED", "RESUME_ALLOWED_FILES_VIOLATION", {"files": violations})
        elif active is not None:
            return self._stop(saved, "BLOCKED", "RESUME_INCOMPLETE_CHECKPOINT_STATE")
        else:
            current_phase = git_guard(self.root)
            if (current_phase["repository"] != saved["last_observed_snapshot"]
                    or current_phase["protected"] != saved["last_observed_protected"]):
                return self._stop(saved, "BLOCKED", "RESUME_EXTERNAL_CHANGE_DETECTED")
        for role in ("claude", "codex"):
            ok, reason = self._budget_valid(role)
            if not ok:
                return self._stop(saved, "BLOCKED", f"RESUME_BUDGET: {role}: {reason}")
        if self.preflight is not None:
            failure = self.preflight()
            if failure:
                return self._stop(saved, "BLOCKED", "RESUME_EXECUTABLE_PREFLIGHT_FAILED", failure)
        if completed >= len(manifest["checkpoints"]):
            return self._final_harness(saved, manifest)
        return self._run_checkpoints(saved, manifest, completed)

    def _tests(self, required, run_id):
        from .engine import LoopEngine
        if self._test_engine is None:
            self._test_engine = LoopEngine(self.root, self.config, self.maker, self.codex, self.budget)
            attempts = self.runtime / "test-temp" / run_id
            self._test_engine._test_attempt = len(list(attempts.iterdir())) if attempts.is_dir() else 0
        return self._test_engine._run_required_tests(required, run_id)
