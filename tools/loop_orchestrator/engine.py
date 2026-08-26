from __future__ import annotations

import hashlib, json, os, re, subprocess, uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .adapters import codex_exec_prefix
from .budget import BudgetManager
from .goal import GoalError, path_allowed, validate_executable_goal, validate_goal
from .models import State, RunState, VerifierResult
from .process import PROCESS_START_FAILED, TIMEOUT, ProcessFailure, probe_process
from .prompts import MAKER, PLANNER, VERIFIER
from .repository import base_commit, changed_files, file_snapshot, git, worktree_snapshot


class LoopEngine:
    def __init__(self, root: Path, config: dict, maker, codex, budget: BudgetManager, preflight_probe=probe_process):
        self.root, self.config, self.maker, self.codex, self.budget = root, config, maker, codex, budget
        self.runtime = root / ".loop"
        self.preflight_probe = preflight_probe
        self._test_attempt = 0

    def _save(self, state, role="orchestrator", result=None):
        self.runtime.mkdir(exist_ok=True)
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "role": role, "exit_status": state.state.value,
                  "base_commit": state.base_commit, "state": state.to_dict()}
        if result is not None: record["result"] = result
        (self.runtime / "state.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        run_dir = self.runtime / "runs" / state.run_id; run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"{len(state.transitions):03d}-{role}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _budget(self, state, role, starting=False):
        state.move(State.BUDGET_CHECK); decision = self.budget.check(role, starting_checkpoint=starting)
        if decision.allowed: return True
        state.move(State.BUDGET_STOP, f"{role}: {decision.code}")
        self._save(state, "budget", {"role": role, "decision": decision.code, "usage_percent": decision.usage_percent})
        self._write_handoff(state, state.stop_reason, {"role": role, "budget": decision.code})
        return False

    def _write_handoff(self, state, reason, details=None):
        handoff = {"timestamp": datetime.now(timezone.utc).isoformat(), "run_id": state.run_id,
                   "checkpoint_id": state.checkpoint_id, "base_commit": state.base_commit,
                   "state": state.state.value, "reason": reason, "details": details or {}}
        (self.runtime / "handoff.json").write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    def _block_process_infrastructure(self, state, role, exc):
        details = {f"{role}_error": exc.diagnostics()}
        state.recent_result = details
        state.move(State.BLOCKED, f"{role.upper()}_INFRASTRUCTURE_FAILURE: {exc.kind}")
        self._save(state, role, details)
        self._write_handoff(state, state.stop_reason, details)
        return state

    def _preflight_executables(self, state):
        timeout = min(int(self.config.get("process_timeout_seconds", 1800)), 10)
        probes = [
            ("claude", [str(self.config.get("claude_executable", "claude")), "--version"]),
            ("codex", [str(self.config.get("codex_executable", "codex")), "--version"]),
            ("codex_exec", [*codex_exec_prefix(str(self.config.get("codex_executable", "codex")), self.root), "--help"]),
        ]
        for integration, argv in probes:
            result = self.preflight_probe(argv, self.root, timeout)
            if not result["ok"]:
                failure = {"integration": integration, "argv": list(argv),
                           "kind": result["kind"], "return_code": result["return_code"],
                           "stdout_tail": result["stdout_tail"], "stderr_tail": result["stderr_tail"]}
                state.recent_result = {"preflight_failure": failure}
                state.move(State.BLOCKED, f"PRECHECK_EXECUTABLE_FAILURE: {integration}: {result['kind']}")
                self._save(state, "preflight", state.recent_result)
                self._write_handoff(state, state.stop_reason, state.recent_result)
                return False
        return True

    def run(self, goal_value: dict, max_checkpoints=1, execute=False):
        try: goal = validate_goal(goal_value)
        except GoalError as exc:
            state = RunState(uuid.uuid4().hex, "invalid", base_commit=base_commit(self.root)); state.move(State.BLOCKED, f"INVALID_GOAL: {exc}"); self._save(state); return state
        state = RunState(uuid.uuid4().hex, goal["checkpoint_id"], base_commit=base_commit(self.root))
        if not execute:
            for item in (State.BUDGET_CHECK, State.MAKER_RUNNING, State.VERIFYING, State.PLANNING): state.move(item)
            state.move(State.ACCEPTED, "DRY_RUN: no processes invoked")
            state.recent_result = {"goal": goal, "maker_prompt": MAKER.format(goal=json.dumps(goal), checkpoint=goal["checkpoint_id"], rework=""),
                                   "required_tests": goal["required_tests"], "planner_condition": "PASS only"}
            self._save(state, result=state.recent_result); return state
        try: validate_executable_goal(goal)
        except GoalError as exc:
            state.move(State.BLOCKED, f"NON_EXECUTABLE_GOAL: {exc}"); self._save(state); return state
        if worktree_snapshot(self.root).strip():
            state.move(State.BLOCKED, "DIRTY_WORKTREE: pre-existing changes cannot be attributed safely"); self._save(state); return state
        if not self._preflight_executables(state): return state

        for checkpoint_number in range(1, max_checkpoints + 1):
            state.iteration = checkpoint_number; state.checkpoint_id = goal["checkpoint_id"]
            failures, session_id, rework_text = {}, None, ""
            max_reworks = int(self.config.get("max_rework_rounds", 2))
            for rework_round in range(max_reworks + 1):
                if not self._budget(state, "claude", starting=rework_round == 0): return state
                before_maker = worktree_snapshot(self.root); before_files = file_snapshot(self.root)
                state.move(State.MAKER_RUNNING); self._save(state, "maker")
                try: invocation = self.maker.run(MAKER.format(goal=json.dumps(goal), checkpoint=goal["checkpoint_id"], rework=rework_text), session_id)
                except KeyboardInterrupt:
                    state.move(State.BLOCKED, "INTERRUPTED"); self._save(state, "maker"); return state
                except ProcessFailure as exc:
                    if exc.kind in {PROCESS_START_FAILED, TIMEOUT}:
                        return self._block_process_infrastructure(state, "maker", exc)
                    details = exc.diagnostics()
                    state.recent_result = {"maker_error": details}
                    state.move(State.FAILED, f"MAKER_ERROR: {exc}")
                    self._save(state, "maker", state.recent_result)
                    self._write_handoff(state, state.stop_reason, state.recent_result)
                    return state
                except Exception as exc:
                    state.move(State.FAILED, f"MAKER_ERROR: {exc}"); self._save(state, "maker"); return state
                maker, session_id = invocation.result, invocation.session_id
                maker_record = asdict(maker) | {"session_id": session_id, "total_cost_usd": invocation.total_cost_usd,
                                                  "num_turns": invocation.num_turns, "telemetry": invocation.telemetry,
                                                  "worktree_changed": worktree_snapshot(self.root) != before_maker}
                self._save(state, "maker", maker_record)
                if maker.status.upper() == "BLOCKED": state.move(State.BLOCKED, maker.summary); self._save(state); return state
                after_files = file_snapshot(self.root)
                maker_changed = sorted(path for path in set(before_files) | set(after_files) if before_files.get(path) != after_files.get(path))
                violations = []
                for path in maker_changed:
                    try:
                        if not path_allowed(path, goal["allowed_files"], self.root): violations.append(path)
                    except GoalError: violations.append(path)
                if violations:
                    state.move(State.BLOCKED, "ALLOWED_FILES_VIOLATION: " + ", ".join(violations)); self._save(state, "maker", maker_record | {"observed_changed_files": maker_changed}); return state
                tests, missing = self._run_required_tests(goal["required_tests"], state.run_id)
                state.recent_result = {"required_tests": tests}
                self._save(state, "tests", state.recent_result)
                if missing:
                    state.move(State.BLOCKED, "REQUIRED_TEST_NOT_ALLOWLISTED: " + ", ".join(missing)); self._save(state, "tests", {"results": tests}); return state
                infrastructure_failures = [item for item in tests if item["status"] in {"TIMEOUT", "START_FAILED"}]
                if infrastructure_failures:
                    state.recent_result = {"required_tests": tests}
                    state.move(State.BLOCKED, "REQUIRED_TEST_INFRASTRUCTURE_FAILURE")
                    self._save(state, "tests", state.recent_result)
                    self._write_handoff(state, state.stop_reason, state.recent_result)
                    return state
                before_verifier = worktree_snapshot(self.root)
                if not self._budget(state, "codex"): return state
                state.move(State.VERIFYING); self._save(state, "verifier")
                diff = git(self.root, "diff", "--binary", "--no-ext-diff")
                try: verifier = self.codex.verify(VERIFIER.format(
                    goal=json.dumps(goal), base=state.base_commit,
                    checkpoint_files=changed_files(self.root), checkpoint_diff=diff,
                    phase_files=changed_files(self.root), phase_diff=diff, tests=tests))
                except KeyboardInterrupt:
                    state.move(State.BLOCKED, "INTERRUPTED"); self._save(state, "verifier"); return state
                except ProcessFailure as exc:
                    if exc.kind in {PROCESS_START_FAILED, TIMEOUT}:
                        return self._block_process_infrastructure(state, "verifier", exc)
                    details = {"verifier_error": exc.diagnostics()}; state.recent_result = details
                    state.move(State.FAILED, f"VERIFIER_ERROR: {exc}"); self._save(state, "verifier", details); return state
                except Exception as exc:
                    state.move(State.FAILED, f"VERIFIER_ERROR: {exc}"); self._save(state, "verifier"); return state
                if worktree_snapshot(self.root) != before_verifier:
                    verifier = VerifierResult("FAIL", ["Verifier modified the worktree"], ["read_only_integrity"], [], None, [], "Investigate verifier mutation")
                    state.recent_result = asdict(verifier); state.move(State.BLOCKED, "VERIFIER_MODIFIED_WORKTREE")
                    self._save(state, "verifier", state.recent_result); return state
                failed_tests = [item for item in tests if item["status"] != "PASS"]
                if failed_tests and verifier.verdict == "PASS":
                    verifier = VerifierResult("FAIL", ["Required tests failed"], [item["name"] for item in failed_tests],
                                              [item["name"] for item in tests], verifier.visual_verification,
                                              verifier.residual_risks, "Fix required tests and rerun verification")
                state.recent_result = asdict(verifier) | {"required_tests": tests}; self._save(state, "verifier", state.recent_result)
                if verifier.verdict == "BLOCKED": state.move(State.BLOCKED, verifier.recommended_action); self._save(state); return state
                if verifier.verdict == "PASS": break
                if failed_tests: self._write_handoff(state, "REQUIRED_TEST_FAILED", {"required_tests": tests})
                signature = hashlib.sha256(json.dumps([verifier.findings, verifier.failed_checks], sort_keys=True).encode()).hexdigest()
                failures[signature] = failures.get(signature, 0) + 1; state.rework_rounds += 1
                if failed_tests and rework_round >= max_reworks:
                    state.move(State.FAILED, "MAX_REWORK_ROUNDS"); self._save(state)
                    self._write_handoff(state, state.stop_reason, {"required_tests": tests, "verifier": asdict(verifier)}); return state
                if failures[signature] >= int(self.config.get("same_failure_limit", 3)):
                    state.move(State.BLOCKED, "SAME_FAILURE_LIMIT"); self._save(state); return state
                if rework_round >= max_reworks:
                    state.move(State.FAILED, "MAX_REWORK_ROUNDS"); self._save(state)
                    self._write_handoff(state, state.stop_reason, {"required_tests": tests, "verifier": asdict(verifier)}); return state
                state.move(State.REWORK); rework_text = "REWORK FINDINGS:\n" + json.dumps(state.recent_result)
            if not self._budget(state, "codex"): return state
            state.move(State.PLANNING); self._save(state, "planner")
            try: plan = self.codex.plan(PLANNER.format(goal=json.dumps(goal), base=state.base_commit, verification=json.dumps(asdict(verifier))))
            except KeyboardInterrupt:
                state.move(State.BLOCKED, "INTERRUPTED"); self._save(state, "planner"); return state
            except ProcessFailure as exc:
                if exc.kind in {PROCESS_START_FAILED, TIMEOUT}:
                    return self._block_process_infrastructure(state, "planner", exc)
                details = {"planner_error": exc.diagnostics()}; state.recent_result = details
                state.move(State.FAILED, f"PLANNER_ERROR: {exc}"); self._save(state, "planner", details); return state
            except Exception as exc:
                state.move(State.FAILED, f"PLANNER_ERROR: {exc}"); self._save(state, "planner"); return state
            try:
                next_goal = validate_goal({"checkpoint_id": plan.checkpoint_id, "objective": plan.objective, "acceptance_criteria": plan.acceptance_criteria,
                                           "allowed_files": plan.allowed_files, "required_tests": plan.required_tests,
                                           "claude_prompt": plan.claude_prompt})
                validate_executable_goal(next_goal)
            except GoalError as exc:
                state.move(State.BLOCKED, f"INVALID_PLANNER_GOAL: {exc}"); self._save(state, "planner", asdict(plan)); return state
            next_path = self.runtime / "next-goal.json"; next_path.write_text(json.dumps(next_goal, indent=2), encoding="utf-8")
            state.recent_result = asdict(plan) | {"next_goal_path": str(next_path)}
            if checkpoint_number == max_checkpoints:
                state.move(State.ACCEPTED); self._save(state, "planner", state.recent_result); return state
            goal = next_goal
        state.move(State.ACCEPTED); self._save(state); return state

    def _run_required_tests(self, required, run_id):
        allowed = {item["name"]: item["argv"] for item in self.config.get("allowed_tests", [])}
        missing = [name for name in required if name not in allowed]
        if missing: return [], missing
        results = []
        for name in required:
            self._test_attempt += 1
            safe_name = (re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "test")[:48]
            temp_path = self.runtime / "test-temp" / run_id / f"{self._test_attempt:03d}-{safe_name}"
            argv = list(allowed[name])
            try:
                temp_path.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                results.append({"name": name, "status": "START_FAILED", "exit_code": None, "error": str(exc)[:2000],
                                "stdout_tail": "", "stderr_tail": str(exc)[-2000:],
                                "argv": argv, "temp_path": str(temp_path)})
                continue
            env = os.environ.copy()
            env.update({"TEMP": str(temp_path), "TMP": str(temp_path), "TMPDIR": str(temp_path),
                        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            try:
                done = subprocess.run(argv, cwd=self.root, text=True, encoding="utf-8", errors="replace",
                                      capture_output=True, shell=False, env=env,
                                      timeout=int(self.config.get("test_timeout_seconds", 300)), check=False)
                results.append({"name": name, "status": "PASS" if done.returncode == 0 else "FAIL", "exit_code": done.returncode,
                                "stdout_tail": done.stdout[-2000:], "stderr_tail": done.stderr[-2000:],
                                "argv": argv, "temp_path": str(temp_path)})
            except subprocess.TimeoutExpired as exc:
                def tail(value):
                    if isinstance(value, bytes): value = value.decode("utf-8", errors="replace")
                    return (value or "")[-2000:]
                results.append({"name": name, "status": "TIMEOUT", "exit_code": None,
                                "stdout_tail": tail(exc.stdout), "stderr_tail": tail(exc.stderr),
                                "argv": argv, "temp_path": str(temp_path)})
            except OSError as exc:
                results.append({"name": name, "status": "START_FAILED", "exit_code": None, "error": str(exc),
                                "stdout_tail": "", "stderr_tail": str(exc)[-2000:],
                                "argv": argv, "temp_path": str(temp_path)})
        return results, []
