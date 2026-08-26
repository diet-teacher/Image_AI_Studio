from __future__ import annotations

import io, json, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools.loop_orchestrator import adapters, cli, process
from tools.loop_orchestrator.adapters import ClaudeCLIAdapter, CodexCLIAdapter, codex_exec_prefix
from tools.loop_orchestrator.budget import BudgetManager
from tools.loop_orchestrator.engine import LoopEngine
from tools.loop_orchestrator.goal import GoalError
from tools.loop_orchestrator.phase import PhaseManifestError, validate_phase_manifest
from tools.loop_orchestrator.phase_engine import PhaseEngine
from tools.loop_orchestrator.models import ClaudeInvocation, MakerResult, PlannerResult, State, VerifierResult
from tools.loop_orchestrator.process import (
    API_CONNECTION_ERROR, JSON_PARSE_FAILED, MODEL_BUDGET_EXHAUSTED, MODEL_MAX_TURNS,
    NONZERO_EXIT, OUTPUT_TAIL_LIMIT, PROCESS_START_FAILED, PROCESS_TIMEOUT, TIMEOUT,
    ProcessFailure, ProcessResult, probe_process, run_json_process,
)
from tools.loop_orchestrator.repository import git as repository_git
from tools.project_harness.profiles import PROFILES


def command(argv, cwd): return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, shell=False, check=True)


class FakeMaker:
    def __init__(self, root, outside=False): self.root, self.outside, self.calls, self.sessions = root, outside, 0, []
    def run(self, prompt, session_id=None):
        self.calls += 1; self.sessions.append(session_id)
        name = "outside.txt" if self.outside else "change.txt"
        (self.root / name).write_text(f"change {self.calls}\n", encoding="utf-8")
        maker = MakerResult("DONE", command(["git", "rev-parse", "HEAD"], self.root).stdout.strip(), [name], ["untrusted-claim"], [], "done")
        return ClaudeInvocation(maker, f"real-session-{self.calls}", 0.01, 1)


class ChainedMaker(FakeMaker):
    def __init__(self, root): super().__init__(root); self.prompts = []
    def run(self, prompt, session_id=None):
        self.calls += 1; self.sessions.append(session_id); self.prompts.append(prompt)
        name = "change.txt" if self.calls == 1 else "next.txt"
        (self.root/name).write_text(f"checkpoint {self.calls}\n", encoding="utf-8")
        maker = MakerResult("DONE", command(["git", "rev-parse", "HEAD"], self.root).stdout.strip(), [name], [], [], "done")
        return ClaudeInvocation(maker, f"chain-session-{self.calls}", 0.01, 1)


class FakeCodex:
    def __init__(self, verdicts, root=None, mutate=False): self.verdicts, self.plans, self.verify_calls, self.root, self.mutate = list(verdicts), 0, 0, root, mutate
    def verify(self, prompt):
        self.verify_calls += 1
        if self.mutate: (self.root/"verifier-write.txt").write_text("forbidden", encoding="utf-8")
        verdict = self.verdicts.pop(0)
        return VerifierResult(verdict, [] if verdict == "PASS" else ["fix"], [] if verdict == "PASS" else ["check"], [], None, [], "rework")
    def plan(self, prompt):
        self.plans += 1
        return PlannerResult("next", "next objective", ["works"], ["next.txt"], ["required"], "implement next", "small")


class FailingCodex(FakeCodex):
    def __init__(self, *, verify_failure=None, plan_failure=None):
        super().__init__(["PASS"]); self.verify_failure, self.plan_failure = verify_failure, plan_failure
    def verify(self, prompt):
        self.verify_calls += 1
        if self.verify_failure: raise self.verify_failure
        verdict = self.verdicts.pop(0)
        return VerifierResult(verdict, [], [], [], None, [], "rework")
    def plan(self, prompt):
        if self.plan_failure: raise self.plan_failure
        return super().plan(prompt)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        command(["git", "init", "-q"], self.root); command(["git", "config", "user.email", "test@example.invalid"], self.root); command(["git", "config", "user.name", "Test"], self.root)
        (self.root / ".gitignore").write_text(".loop/\n", encoding="utf-8"); (self.root / "base.txt").write_text("base\n", encoding="utf-8")
        command(["git", "add", ".gitignore", "base.txt"], self.root); command(["git", "commit", "-qm", "base"], self.root); (self.root / ".loop").mkdir()
        self.config = {"max_rework_rounds": 2, "same_failure_limit": 3, "test_timeout_seconds": 5,
                       "allowed_tests": [{"name": "required", "argv": [sys.executable, "-c", "from pathlib import Path; Path('.loop/required-ran').write_text('yes')"]}]}
        self.goal = {"checkpoint_id":"cp1", "objective":"change", "acceptance_criteria":["works"], "allowed_files":["change.txt"], "required_tests":["required"]}

    def tearDown(self): self.temp.cleanup()
    def budget(self, claude=10, codex=10):
        now = datetime.now(timezone.utc).isoformat()
        (self.root/".loop"/"budget.json").write_text(json.dumps({"claude":{"period_usage_percent":claude,"updated_at":now,"source":"manual"},"codex":{"period_usage_percent":codex,"updated_at":now,"source":"manual"}}), encoding="utf-8")
        return BudgetManager(self.root/".loop"/"budget.json")
    @staticmethod
    def successful_probe(argv, cwd, timeout):
        return {"ok": True, "kind": None, "return_code": 0, "stdout_tail": "ok", "stderr_tail": ""}
    def engine(self, maker, codex, probe=None):
        return LoopEngine(self.root, self.config, maker, codex, self.budget(), probe or self.successful_probe)

    def test_default_fail_rework_pass_resumes_real_session(self):
        maker, codex = FakeMaker(self.root), FakeCodex(["FAIL", "PASS"])
        state = self.engine(maker, codex).run(self.goal, max_checkpoints=1, execute=True)
        self.assertEqual(State.ACCEPTED, state.state); self.assertEqual([None, "real-session-1"], maker.sessions)
        self.assertTrue((self.root/".loop"/"required-ran").is_file())
        self.assertEqual("next", json.loads((self.root/".loop"/"next-goal.json").read_text())["checkpoint_id"])

    def test_allowed_files_violation_blocks(self):
        state = self.engine(FakeMaker(self.root, outside=True), FakeCodex([])).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertIn("ALLOWED_FILES_VIOLATION", state.stop_reason)

    def test_required_test_not_allowlisted_blocks_without_verifier(self):
        goal = dict(self.goal); goal["required_tests"] = ["missing"]
        state = self.engine(FakeMaker(self.root), FakeCodex([])).run(goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertIn("REQUIRED_TEST_NOT_ALLOWLISTED", state.stop_reason)

    def test_required_test_failure_overrides_verifier_pass_and_exhausts_rework(self):
        self.config["allowed_tests"][0]["argv"] = [sys.executable, "-c", "import sys; print('forced failure'); print('tail', file=sys.stderr); sys.exit(1)"]
        state = self.engine(FakeMaker(self.root), FakeCodex(["PASS", "PASS", "PASS"])).run(self.goal, execute=True)
        self.assertEqual(State.FAILED, state.state); self.assertEqual("MAX_REWORK_ROUNDS", state.stop_reason)
        handoff = json.loads((self.root/".loop"/"handoff.json").read_text(encoding="utf-8"))
        result = handoff["details"]["required_tests"][0]
        self.assertEqual("FAIL", result["status"]); self.assertIn("forced failure", result["stdout_tail"]); self.assertIn("tail", result["stderr_tail"])

    def test_required_test_timeout_is_blocked(self):
        self.config["test_timeout_seconds"] = 0.01
        self.config["allowed_tests"][0]["argv"] = [sys.executable, "-c", "import time; print('before timeout', flush=True); time.sleep(1)"]
        state = self.engine(FakeMaker(self.root), FakeCodex([])).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertEqual("REQUIRED_TEST_INFRASTRUCTURE_FAILURE", state.stop_reason)
        self.assertEqual("TIMEOUT", state.recent_result["required_tests"][0]["status"])

    def test_required_test_start_failure_is_blocked(self):
        self.config["allowed_tests"][0]["argv"] = [str(self.root/"missing-executable.exe")]
        state = self.engine(FakeMaker(self.root), FakeCodex([])).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertEqual("START_FAILED", state.recent_result["required_tests"][0]["status"])

    def test_failed_required_test_can_pass_after_rework(self):
        script = "from pathlib import Path; import sys; sys.exit(0 if 'change 2' in Path('change.txt').read_text() else 1)"
        self.config["allowed_tests"][0]["argv"] = [sys.executable, "-c", script]
        state = self.engine(FakeMaker(self.root), FakeCodex(["PASS", "PASS"])).run(self.goal, execute=True)
        self.assertEqual(State.ACCEPTED, state.state); self.assertIn("REWORK", state.transitions)

    def test_second_checkpoint_receives_planner_claude_prompt(self):
        maker, codex = ChainedMaker(self.root), FakeCodex(["PASS", "PASS"])
        state = self.engine(maker, codex).run(self.goal, max_checkpoints=2, execute=True)
        self.assertEqual(State.ACCEPTED, state.state); self.assertIn("implement next", maker.prompts[1])
        self.assertEqual("implement next", json.loads((self.root/".loop"/"next-goal.json").read_text())["claude_prompt"])

    def test_verifier_worktree_mutation_never_reaches_planner(self):
        codex = FakeCodex(["PASS"], self.root, mutate=True)
        state = self.engine(FakeMaker(self.root), codex).run(self.goal, execute=True)
        self.assertNotEqual(State.ACCEPTED, state.state); self.assertEqual(0, codex.plans)

    def test_invalid_absolute_and_traversal_allowed_paths(self):
        for bad in ("C:\\escape.txt", "../escape.txt"):
            goal = dict(self.goal); goal["allowed_files"] = [bad]
            self.assertEqual(State.BLOCKED, self.engine(FakeMaker(self.root), FakeCodex([])).run(goal, execute=False).state)

    def test_template_and_incomplete_code_goals_cannot_execute(self):
        template = dict(self.goal); template["template"] = True
        self.assertIn("NON_EXECUTABLE_GOAL", self.engine(FakeMaker(self.root), FakeCodex([])).run(template, execute=True).stop_reason)
        for field in ("acceptance_criteria", "required_tests"):
            incomplete = dict(self.goal); incomplete[field] = []
            self.assertIn("NON_EXECUTABLE_GOAL", self.engine(FakeMaker(self.root), FakeCodex([])).run(incomplete, execute=True).stop_reason)

    def test_cli_rejects_max_checkpoints_below_one(self):
        with self.assertRaises(SystemExit) as raised:
            cli.main(["run", "--goal", str(Path(__file__).parents[2]/"tools"/"loop_orchestrator"/"example.goal.json"), "--max-checkpoints", "0", "--dry-run"])
        self.assertEqual(2, raised.exception.code)

    def test_maker_process_failure_is_saved_and_handed_off(self):
        class FailingMaker:
            def run(self, prompt, session_id=None):
                raise ProcessFailure(NONZERO_EXIT, "NONZERO_EXIT 1", return_code=1,
                                     stdout_tail='{\"error\":\"bad\"}', stderr_tail="diagnostic")
        state = self.engine(FailingMaker(), FakeCodex([])).run(self.goal, execute=True)
        self.assertEqual(State.FAILED, state.state)
        details = json.loads((self.root/".loop"/"handoff.json").read_text(encoding="utf-8"))["details"]
        self.assertEqual(1, details["maker_error"]["return_code"])
        self.assertIn("error", details["maker_error"]["stdout_tail"])
        self.assertEqual("diagnostic", details["maker_error"]["stderr_tail"])

    def test_preflight_missing_claude_blocks_without_maker(self):
        maker = FakeMaker(self.root)
        def probe(argv, cwd, timeout):
            if argv[0] == "claude": return {"ok":False,"kind":PROCESS_START_FAILED,"return_code":None,"stdout_tail":"","stderr_tail":"missing"}
            return self.successful_probe(argv, cwd, timeout)
        state = self.engine(maker, FakeCodex([]), probe).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertEqual(0, maker.calls)
        self.assertIn("PRECHECK_EXECUTABLE_FAILURE", state.stop_reason)

    def test_preflight_missing_codex_blocks_without_maker(self):
        maker = FakeMaker(self.root)
        def probe(argv, cwd, timeout):
            if argv[0] == "codex": return {"ok":False,"kind":PROCESS_START_FAILED,"return_code":None,"stdout_tail":"","stderr_tail":"missing"}
            return self.successful_probe(argv, cwd, timeout)
        state = self.engine(maker, FakeCodex([]), probe).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertEqual(0, maker.calls)

    def test_preflight_codex_exec_help_nonzero_blocks_without_maker(self):
        maker = FakeMaker(self.root)
        def probe(argv, cwd, timeout):
            if argv[-2:] == ["exec", "--help"]: return {"ok":False,"kind":NONZERO_EXIT,"return_code":2,"stdout_tail":"","stderr_tail":"bad help"}
            return self.successful_probe(argv, cwd, timeout)
        state = self.engine(maker, FakeCodex([]), probe).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertEqual(0, maker.calls)

    def test_preflight_uses_production_codex_prefix_and_help_only(self):
        calls = []
        def probe(argv, cwd, timeout):
            calls.append(list(argv)); return self.successful_probe(argv, cwd, timeout)
        class StopMaker:
            def run(inner_self, prompt, session_id=None): raise self.process_failure(NONZERO_EXIT)
        self.engine(StopMaker(), FakeCodex([]), probe).run(self.goal, execute=True)
        expected = [*codex_exec_prefix("codex", self.root), "--help"]
        self.assertEqual(expected, calls[2])
        self.assertEqual("never", calls[2][calls[2].index("--ask-for-approval") + 1])
        self.assertEqual("read-only", calls[2][calls[2].index("--sandbox") + 1])
        self.assertEqual(["exec", "--help"], calls[2][-2:])
        self.assertNotIn("--json", calls[2]); self.assertNotIn("prompt", calls[2])

    def test_preflight_timeout_blocks_and_preserves_handoff(self):
        maker = FakeMaker(self.root)
        def probe(argv, cwd, timeout):
            return {"ok":False,"kind":TIMEOUT,"return_code":None,"stdout_tail":"partial","stderr_tail":"timed out"}
        state = self.engine(maker, FakeCodex([]), probe).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state); self.assertEqual(0, maker.calls)
        details = json.loads((self.root/".loop"/"handoff.json").read_text())["details"]["preflight_failure"]
        self.assertEqual(TIMEOUT, details["kind"]); self.assertEqual("partial", details["stdout_tail"])
        self.assertEqual(["claude", "--version"], details["argv"])

    @staticmethod
    def process_failure(kind):
        return ProcessFailure(kind, kind, return_code=1 if kind == NONZERO_EXIT else None,
                              stdout_tail="out", stderr_tail="err")

    def test_maker_infrastructure_failures_block_but_nonzero_stays_failed(self):
        class FailingMaker:
            def __init__(self, failure): self.failure = failure
            def run(self, prompt, session_id=None): raise self.failure
        for kind in (PROCESS_START_FAILED, TIMEOUT):
            state = self.engine(FailingMaker(self.process_failure(kind)), FakeCodex([])).run(self.goal, execute=True)
            self.assertEqual(State.BLOCKED, state.state); self.assertTrue((self.root/".loop"/"handoff.json").is_file())
        state = self.engine(FailingMaker(self.process_failure(NONZERO_EXIT)), FakeCodex([])).run(self.goal, execute=True)
        self.assertEqual(State.FAILED, state.state)

    def test_maker_json_parse_failure_stays_failed(self):
        class FailingMaker:
            def run(inner_self, prompt, session_id=None): raise self.process_failure(JSON_PARSE_FAILED)
        state = self.engine(FailingMaker(), FakeCodex([])).run(self.goal, execute=True)
        self.assertEqual(State.FAILED, state.state)

    def test_verifier_failure_role_mapping_and_tests_record_survives(self):
        for kind, expected in ((PROCESS_START_FAILED, State.BLOCKED), (TIMEOUT, State.BLOCKED),
                               (NONZERO_EXIT, State.FAILED), (JSON_PARSE_FAILED, State.FAILED)):
            with self.subTest(kind=kind):
                codex = FailingCodex(verify_failure=self.process_failure(kind))
                state = self.engine(FakeMaker(self.root), codex).run(self.goal, execute=True)
                self.assertEqual(expected, state.state, state.stop_reason)
                run_dir = self.root/".loop"/"runs"/state.run_id
                test_records = list(run_dir.glob("*-tests.json"))
                self.assertEqual(1, len(test_records), state.stop_reason)
                result = json.loads(test_records[0].read_text())["result"]["required_tests"][0]
                self.assertEqual("PASS", result["status"])
                if expected == State.BLOCKED: self.assertTrue((self.root/".loop"/"handoff.json").is_file())
                (self.root/"change.txt").unlink()

    def test_planner_start_failure_blocks_with_handoff(self):
        codex = FailingCodex(plan_failure=self.process_failure(PROCESS_START_FAILED))
        state = self.engine(FakeMaker(self.root), codex).run(self.goal, execute=True)
        self.assertEqual(State.BLOCKED, state.state)
        handoff = json.loads((self.root/".loop"/"handoff.json").read_text())
        self.assertEqual(PROCESS_START_FAILED, handoff["details"]["planner_error"]["kind"])

    def test_planner_timeout_blocks_but_nonzero_and_json_parse_fail(self):
        for kind, expected in ((TIMEOUT, State.BLOCKED), (NONZERO_EXIT, State.FAILED),
                               (JSON_PARSE_FAILED, State.FAILED)):
            with self.subTest(kind=kind):
                codex = FailingCodex(plan_failure=self.process_failure(kind))
                state = self.engine(FakeMaker(self.root), codex).run(self.goal, execute=True)
                self.assertEqual(expected, state.state, state.stop_reason)
                if expected == State.BLOCKED:
                    handoff = json.loads((self.root/".loop"/"handoff.json").read_text())
                    self.assertEqual(TIMEOUT, handoff["details"]["planner_error"]["kind"])
                (self.root/"change.txt").unlink()

    def test_temp_mkdir_failures_are_recorded_and_block_before_verifier(self):
        original_mkdir = Path.mkdir
        required_temp_root = (self.root/".loop"/"test-temp").resolve()
        for error in (PermissionError("denied"), FileExistsError("collision")):
            with self.subTest(error=type(error).__name__):
                codex = FakeCodex(["PASS"])
                def mkdir(path, *args, **kwargs):
                    resolved = path.resolve()
                    if resolved == required_temp_root or required_temp_root in resolved.parents:
                        raise error
                    return original_mkdir(path, *args, **kwargs)
                original_argv = list(self.config["allowed_tests"][0]["argv"])
                with patch.object(Path, "mkdir", new=mkdir):
                    state = self.engine(FakeMaker(self.root), codex).run(self.goal, execute=True)
                self.assertEqual(State.BLOCKED, state.state); self.assertEqual(0, codex.verify_calls)
                self.assertEqual("REQUIRED_TEST_INFRASTRUCTURE_FAILURE", state.stop_reason)
                record = next((self.root/".loop"/"runs"/state.run_id).glob("*-tests.json"))
                result = json.loads(record.read_text())["result"]["required_tests"][0]
                self.assertEqual("START_FAILED", result["status"]); self.assertIsNone(result["exit_code"])
                self.assertIn(str(error), result["stderr_tail"]); self.assertEqual(original_argv, result["argv"])
                self.assertIn("test-temp", result["temp_path"])
                handoff = json.loads((self.root/".loop"/"handoff.json").read_text())
                self.assertIn(str(error), handoff["details"]["required_tests"][0]["stderr_tail"])
                self.assertEqual(original_argv, self.config["allowed_tests"][0]["argv"])
                (self.root/"change.txt").unlink()

    def test_required_tests_get_isolated_temp_without_changing_argv(self):
        original = list(self.config["allowed_tests"][0]["argv"])
        captures = []
        real_run = subprocess.run
        def capture(argv, **kwargs):
            if argv == original:
                captures.append((list(argv), kwargs))
                return subprocess.CompletedProcess(argv, 0, "pass", "")
            return real_run(argv, **kwargs)
        with patch("tools.loop_orchestrator.engine.subprocess.run", side_effect=capture):
            state = self.engine(FakeMaker(self.root), FakeCodex(["FAIL", "PASS"])).run(self.goal, execute=True)
        self.assertEqual(State.ACCEPTED, state.state); self.assertEqual(2, len(captures))
        paths = []
        for argv, kwargs in captures:
            self.assertEqual(original, argv); self.assertFalse(kwargs["shell"])
            env = kwargs["env"]
            self.assertEqual(env["TEMP"], env["TMP"]); self.assertEqual(env["TEMP"], env["TMPDIR"])
            self.assertEqual("1", env["PYTHONUTF8"]); self.assertEqual("utf-8", env["PYTHONIOENCODING"])
            path = Path(env["TEMP"]); paths.append(path)
            self.assertTrue(path.is_relative_to(self.root/".loop")); self.assertTrue(path.is_dir())
        self.assertNotEqual(paths[0], paths[1]); self.assertEqual(original, self.config["allowed_tests"][0]["argv"])


class AdapterTests(unittest.TestCase):
    def fixture(self):
        envelope = json.loads((Path(__file__).parent/"fixtures"/"claude_result.json").read_text(encoding="utf-8"))
        return ProcessResult(json.loads(envelope["result"]), "", "", envelope)

    def test_claude_top_level_envelope_and_no_unrestricted_bash(self):
        captured = {}
        def fake(argv, root, timeout, stdin_text=None):
            captured["argv"] = argv; captured["stdin_text"] = stdin_text; return self.fixture()
        with patch.object(adapters, "run_json_process", side_effect=fake):
            result = ClaudeCLIAdapter(Path.cwd(), 10, 1).run("prompt")
        self.assertEqual("11111111-2222-4333-8444-555555555555", result.session_id)
        self.assertEqual(0.042, result.total_cost_usd); self.assertEqual(3, result.num_turns)
        argv = captured["argv"]; self.assertIn("--disallowedTools", argv); self.assertEqual("Bash", argv[argv.index("--disallowedTools")+1])
        self.assertNotIn("Bash", argv[argv.index("--allowedTools")+1].split(","))
        self.assertNotIn("session_id", adapters.MAKER_SCHEMA["properties"])
        self.assertNotIn("prompt", argv)
        self.assertEqual("prompt", captured["stdin_text"])
        self.assertEqual("-p", argv[1])

    def test_claude_model_usage_preserves_primary_and_helpers(self):
        envelope = {
            "session_id": "session-models", "total_cost_usd": 0.3, "num_turns": 2,
            "terminal_reason": "completed", "subtype": "success",
            "result": json.dumps({"status": "DONE", "base_commit": "base", "changed_files": [],
                                  "tests_run": [], "known_risks": [], "summary": "done"}),
            "modelUsage": {
                "sonnet-key": {"canonicalModel": "claude-sonnet-5", "inputTokens": 10,
                               "outputTokens": 20, "cacheReadInputTokens": 30,
                               "cacheCreationInputTokens": 40, "costUSD": 0.29},
                "haiku-key": {"canonicalModel": "claude-haiku-4-5", "inputTokens": 1,
                              "outputTokens": 2, "cacheReadInputTokens": 3,
                              "cacheCreationInputTokens": 4, "costUSD": 0.01},
            },
        }
        payload = json.loads(envelope["result"])
        captured = {}
        def fake(argv, root, timeout, **kwargs):
            captured.update(kwargs)
            metadata = {**envelope, **process._extract_model_usage(envelope, kwargs.get("requested_model"))}
            return ProcessResult(payload, "", "", metadata)
        with patch.object(adapters, "run_json_process", side_effect=fake):
            result = ClaudeCLIAdapter(Path.cwd(), 10, 1, "claude-sonnet-5").run("prompt")
        self.assertEqual("claude-sonnet-5", result.telemetry["primary_canonical_model"])
        self.assertEqual(["claude-haiku-4-5", "claude-sonnet-5"], result.telemetry["canonical_models"])
        self.assertEqual(30, result.telemetry["model_usage"]["sonnet-key"]["cacheReadInputTokens"])
        self.assertEqual("claude-sonnet-5", captured["requested_model"])

    def test_claude_model_omitted_does_not_guess_primary(self):
        envelope = {"modelUsage": {"only": {"canonicalModel": "claude-sonnet-5", "costUSD": 0.1}}}
        telemetry = process._extract_model_usage(envelope)
        self.assertIsNone(telemetry["primary_canonical_model"])
        self.assertEqual(["claude-sonnet-5"], telemetry["canonical_models"])

    def test_claude_model_config_argv_once_and_prompt_stdin(self):
        captured = {}
        def fake(argv, root, timeout, **kwargs):
            captured["argv"], captured["kwargs"] = argv, kwargs
            return self.fixture()
        with patch.object(adapters, "run_json_process", side_effect=fake):
            ClaudeCLIAdapter(Path.cwd(), 10, 1, "claude-sonnet-5").run("secret prompt")
        self.assertEqual(1, captured["argv"].count("--model"))
        self.assertEqual("claude-sonnet-5", captured["argv"][captured["argv"].index("--model") + 1])
        self.assertNotIn("secret prompt", captured["argv"])
        self.assertEqual("secret prompt", captured["kwargs"]["stdin_text"])

    def test_malformed_model_usage_has_explicit_diagnostics(self):
        telemetry = process._extract_model_usage({"modelUsage": {
            "bad": {"canonicalModel": "claude-sonnet-5", "inputTokens": True, "costUSD": "x"}}})
        self.assertEqual(["bad"], telemetry["model_usage_invalid"])
        self.assertEqual({}, {k: v for k, v in telemetry["model_usage"]["bad"].items()
                              if k != "canonicalModel"})

    def test_nonzero_model_usage_telemetry_is_preserved(self):
        envelope = {"subtype": "error_max_budget", "session_id": "s", "total_cost_usd": 0.2,
                    "modelUsage": {"m": {"canonicalModel": "claude-sonnet-5", "inputTokens": 3,
                                             "outputTokens": 4, "costUSD": 0.2}}}
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", self._nonzero_envelope_script(envelope)],
                             Path.cwd(), 10, requested_model="claude-sonnet-5")
        metadata = raised.exception.metadata
        self.assertEqual("claude-sonnet-5", metadata["primary_canonical_model"])
        self.assertEqual(4, metadata["model_usage"]["m"]["outputTokens"])

    def test_real_cli_envelope_fixture_parser_preserves_metadata(self):
        fixture = Path(__file__).parent/"fixtures"/"claude_result.json"
        script = "from pathlib import Path; print(Path(r'%s').read_text(encoding='utf-8'))" % fixture
        parsed = run_json_process([sys.executable, "-c", script], Path.cwd(), 10)
        self.assertEqual("11111111-2222-4333-8444-555555555555", parsed.metadata["session_id"])
        self.assertEqual("DONE", parsed.payload["status"])

    def test_resume_uses_envelope_session_only(self):
        captured = {}
        def fake(argv, root, timeout, stdin_text=None): captured["argv"] = argv; return self.fixture()
        with patch.object(adapters, "run_json_process", side_effect=fake): ClaudeCLIAdapter(Path.cwd(), 10, 1).run("fix", "actual-session")
        self.assertEqual("actual-session", captured["argv"][captured["argv"].index("--resume")+1])
        self.assertNotIn("fix", captured["argv"])

    def test_codex_adapter_uses_shared_production_prefix(self):
        captured = {}
        payload = {"verdict":"PASS","findings":[],"failed_checks":[],"tests_observed":[],
                   "visual_verification":None,"residual_risks":[],"recommended_action":"none"}
        def fake(argv, root, timeout, json_lines=False, stdin_text=None):
            captured["argv"] = argv; captured["stdin_text"] = stdin_text; return ProcessResult(payload, "", "", {})
        adapter = CodexCLIAdapter(Path("repo"), 10, Path("schemas"), "codex-custom")
        with patch.object(adapters, "run_json_process", side_effect=fake): adapter.verify("actual prompt")
        prefix = codex_exec_prefix("codex-custom", Path("repo"))
        self.assertEqual(prefix, captured["argv"][:len(prefix)])
        self.assertNotIn("actual prompt", captured["argv"])
        self.assertEqual("actual prompt", captured["stdin_text"])
        self.assertIn("--json", captured["argv"]); self.assertIn("--output-schema", captured["argv"])

    def test_stdin_transport_preserves_long_korean_quoted_multiline_prompt(self):
        script = "import sys, json; print(json.dumps({'echo': sys.stdin.read()}))"
        long_prompt = ("A" * 60000) + "\n안녕하세요 \"quoted\" text\nline2\nline3\n" + ("B" * 60000)
        result = run_json_process([sys.executable, "-c", script], Path.cwd(), 10, stdin_text=long_prompt)
        self.assertEqual(long_prompt, result.payload["echo"])

    def test_dry_run_style_call_with_no_stdin_text_sends_nothing(self):
        script = "import sys; data = sys.stdin.read(); print('{\"received\": %d}' % len(data))"
        result = run_json_process([sys.executable, "-c", script], Path.cwd(), 10)
        self.assertEqual(0, result.payload["received"])

    def test_process_timeout_is_stable_alias_of_timeout(self):
        self.assertEqual(TIMEOUT, PROCESS_TIMEOUT)

    def _nonzero_envelope_script(self, envelope: dict) -> str:
        return "import sys; print(%s); sys.exit(1)" % json.dumps(json.dumps(envelope))

    def test_budget_exhausted_envelope_classifies_and_preserves_telemetry(self):
        envelope = {"type": "result", "subtype": "error_max_budget", "is_error": True,
                    "session_id": "sess-budget", "total_cost_usd": 4.9, "num_turns": 12, "result": "ran out"}
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", self._nonzero_envelope_script(envelope)], Path.cwd(), 10)
        failure = raised.exception
        self.assertEqual(MODEL_BUDGET_EXHAUSTED, failure.kind)
        self.assertEqual("sess-budget", failure.metadata["session_id"])
        self.assertEqual(4.9, failure.metadata["total_cost_usd"])
        self.assertEqual(12, failure.metadata["num_turns"])
        self.assertEqual("error_max_budget", failure.metadata["subtype"])

    def test_max_turns_envelope_classifies_and_preserves_telemetry(self):
        envelope = {"subtype": "error_max_turns", "is_error": True, "session_id": "sess-turns",
                    "total_cost_usd": 1.2, "num_turns": 50}
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", self._nonzero_envelope_script(envelope)], Path.cwd(), 10)
        failure = raised.exception
        self.assertEqual(MODEL_MAX_TURNS, failure.kind)
        self.assertEqual("sess-turns", failure.metadata["session_id"])
        self.assertEqual(50, failure.metadata["num_turns"])

    def test_api_connection_error_envelope_classifies_and_preserves_telemetry(self):
        envelope = {"subtype": "error_api_connection", "is_error": True, "session_id": "sess-conn",
                    "total_cost_usd": 0.0, "errors": ["api_connection_error"]}
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", self._nonzero_envelope_script(envelope)], Path.cwd(), 10)
        failure = raised.exception
        self.assertEqual(API_CONNECTION_ERROR, failure.kind)
        self.assertEqual("sess-conn", failure.metadata["session_id"])
        self.assertEqual(0.0, failure.metadata["total_cost_usd"])

    def test_general_nonzero_envelope_without_markers_stays_nonzero_exit(self):
        envelope = {"subtype": "success", "is_error": False, "session_id": "sess-ok", "total_cost_usd": 0.5}
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", self._nonzero_envelope_script(envelope)], Path.cwd(), 10)
        self.assertEqual(NONZERO_EXIT, raised.exception.kind)
        self.assertEqual("sess-ok", raised.exception.metadata["session_id"])

    def test_malformed_nonzero_stdout_stays_nonzero_exit_with_empty_metadata(self):
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", "print('not json'); import sys; sys.exit(1)"], Path.cwd(), 10)
        self.assertEqual(NONZERO_EXIT, raised.exception.kind)
        self.assertEqual({}, raised.exception.metadata)

    def test_codex_json_lines_nonzero_ignores_envelope_parsing(self):
        envelope = {"subtype": "error_max_budget", "session_id": "ignored"}
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", self._nonzero_envelope_script(envelope)],
                             Path.cwd(), 10, json_lines=True)
        self.assertEqual(NONZERO_EXIT, raised.exception.kind)
        self.assertEqual({}, raised.exception.metadata)

    def run_failure(self, script):
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", script], Path.cwd(), 10)
        return raised.exception

    def test_nonzero_exit_preserves_stdout_when_stderr_empty(self):
        failure = self.run_failure("import sys; print('{\"error\": \"failed\"}'); sys.exit(1)")
        self.assertEqual(NONZERO_EXIT, failure.kind)
        self.assertEqual(1, failure.return_code)
        self.assertIn('"error": "failed"', failure.stdout_tail)
        self.assertEqual("", failure.stderr_tail)

    def test_nonzero_exit_preserves_stderr(self):
        failure = self.run_failure("import sys; print('bad stderr', file=sys.stderr); sys.exit(1)")
        self.assertEqual("", failure.stdout_tail)
        self.assertIn("bad stderr", failure.stderr_tail)

    def test_nonzero_exit_preserves_both_bounded_tails(self):
        script = "import sys; print('x' * 5000); print('y' * 5000, file=sys.stderr); sys.exit(1)"
        failure = self.run_failure(script)
        self.assertEqual(OUTPUT_TAIL_LIMIT, len(failure.stdout_tail))
        self.assertEqual(OUTPUT_TAIL_LIMIT, len(failure.stderr_tail))
        self.assertTrue(failure.stdout_tail.endswith("x" * (OUTPUT_TAIL_LIMIT - 1) + "\n"))
        self.assertTrue(failure.stderr_tail.endswith("y" * (OUTPUT_TAIL_LIMIT - 1) + "\n"))

    def test_invalid_utf8_is_replaced_without_decode_failure(self):
        failure = self.run_failure("import os; os.write(1, b'bad\\xffbytes'); raise SystemExit(1)")
        self.assertIn("bad\ufffdbytes", failure.stdout_tail)

    def test_popen_keeps_shell_false_and_utf8_and_interrupt_cleanup(self):
        process = unittest.mock.MagicMock()
        process.communicate.side_effect = [KeyboardInterrupt(), ("", "")]
        with patch("tools.loop_orchestrator.process.subprocess.Popen", return_value=process) as popen:
            with self.assertRaises(KeyboardInterrupt):
                run_json_process(["fixed", "argv"], Path.cwd(), 10)
        kwargs = popen.call_args.kwargs
        self.assertFalse(kwargs["shell"])
        self.assertEqual("utf-8", kwargs["encoding"])
        self.assertEqual("replace", kwargs["errors"])
        self.assertEqual(subprocess.PIPE, kwargs["stdin"])
        process.kill.assert_called_once_with()

    def test_stdin_text_is_forwarded_to_communicate(self):
        process = unittest.mock.MagicMock()
        process.returncode = 0
        process.communicate.return_value = ('{"status":"DONE"}', "")
        with patch("tools.loop_orchestrator.process.subprocess.Popen", return_value=process):
            run_json_process(["fixed", "argv"], Path.cwd(), 10, stdin_text="the prompt text")
        self.assertEqual("the prompt text", process.communicate.call_args.kwargs["input"])

    def test_json_parse_failure_has_structured_kind_and_tails(self):
        with self.assertRaises(ProcessFailure) as raised:
            run_json_process([sys.executable, "-c", "print('not json')"], Path.cwd(), 10)
        self.assertEqual(JSON_PARSE_FAILED, raised.exception.kind)
        self.assertIn("not json", raised.exception.stdout_tail)

    def test_process_start_failure_and_timeout_have_structured_kinds(self):
        with patch("tools.loop_orchestrator.process.subprocess.Popen", side_effect=FileNotFoundError("missing")):
            with self.assertRaises(ProcessFailure) as started:
                run_json_process(["missing"], Path.cwd(), 10)
        self.assertEqual(PROCESS_START_FAILED, started.exception.kind)
        process = unittest.mock.MagicMock()
        process.communicate.side_effect = [subprocess.TimeoutExpired(["slow"], 1, output="partial", stderr="late"), ("", "")]
        with patch("tools.loop_orchestrator.process.subprocess.Popen", return_value=process):
            with self.assertRaises(ProcessFailure) as timed_out:
                run_json_process(["slow"], Path.cwd(), 1)
        self.assertEqual(TIMEOUT, timed_out.exception.kind)
        self.assertEqual("partial", timed_out.exception.stdout_tail)
        process.kill.assert_called_once_with()

    def test_probe_is_bounded_utf8_shell_false_and_timeout_aware(self):
        done = subprocess.CompletedProcess(["tool"], 3, "x" * 5000, "bad")
        with patch("tools.loop_orchestrator.process.subprocess.run", return_value=done) as run:
            result = probe_process(["tool", "--version"], Path.cwd(), 7)
        self.assertEqual(NONZERO_EXIT, result["kind"]); self.assertEqual(OUTPUT_TAIL_LIMIT, len(result["stdout_tail"]))
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs["shell"]); self.assertEqual(7, kwargs["timeout"])
        self.assertEqual("utf-8", kwargs["encoding"]); self.assertEqual("replace", kwargs["errors"])


class RepositoryTests(unittest.TestCase):
    def test_git_uses_platform_independent_utf8_decoding(self):
        done = subprocess.CompletedProcess(["git"], 0, "diff – text", "")
        with patch("tools.loop_orchestrator.repository.subprocess.run", return_value=done) as run:
            self.assertEqual("diff – text", repository_git(Path.cwd(), "diff"))
        kwargs = run.call_args.kwargs
        self.assertEqual("utf-8", kwargs["encoding"])
        self.assertEqual("replace", kwargs["errors"])
        self.assertFalse(kwargs["shell"])


class BudgetAndDoctorTests(unittest.TestCase):
    def test_soft_and_hard_stops_with_fresh_budget(self):
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"budget.json"
            for usage, expected in ((75, "SOFT_STOP"), (80, "HARD_STOP")):
                path.write_text(json.dumps({"claude":{"period_usage_percent":usage,"updated_at":now.isoformat(),"source":"manual"}}), encoding="utf-8")
                self.assertEqual(expected, BudgetManager(path, now=lambda:now).check("claude", starting_checkpoint=True).code)
    def test_stale_future_range_and_missing_budget_are_unknown(self):
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        cases = [
            {"period_usage_percent": 10, "updated_at": (now-timedelta(hours=25)).isoformat(), "source":"manual"},
            {"period_usage_percent": 10, "updated_at": (now+timedelta(seconds=1)).isoformat(), "source":"manual"},
            {"period_usage_percent": 101, "updated_at": now.isoformat(), "source":"manual"},
            {"period_usage_percent": 10, "updated_at": now.isoformat()},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"budget.json"
            for entry in cases:
                path.write_text(json.dumps({"claude":entry}), encoding="utf-8")
                self.assertEqual("BUDGET_UNKNOWN", BudgetManager(path, now=lambda:now).check("claude").code)

    def test_doctor_fails_when_codex_subprocess_cannot_start(self):
        def fake(argv, cwd, timeout):
            if "codex" in str(argv[0]):
                return {"ok":False,"kind":PROCESS_START_FAILED,"return_code":None,"stdout_tail":"","stderr_tail":"PermissionError: access denied"}
            return {"ok":True,"kind":None,"return_code":0,"stdout_tail":"ok","stderr_tail":""}
        output = io.StringIO()
        with patch.object(cli, "probe_process", side_effect=fake), redirect_stdout(output): code = cli.doctor()
        self.assertEqual(1, code); self.assertIn("PermissionError", output.getvalue())

    def doctor_with_git_output(self, stdout, stderr):
        def fake(argv, cwd, timeout):
            if argv[:2] == ["git", "status"]:
                return {"ok":True,"kind":None,"return_code":0,"stdout_tail":stdout,"stderr_tail":stderr}
            return {"ok":True,"kind":None,"return_code":0,"stdout_tail":"ok","stderr_tail":""}
        output = io.StringIO()
        with patch.object(cli, "probe_process", side_effect=fake), redirect_stdout(output):
            code = cli.doctor()
        return code, json.loads(output.getvalue())

    def test_doctor_stderr_warning_does_not_mark_clean_worktree_dirty(self):
        code, report = self.doctor_with_git_output("", "warning: inaccessible ignore file")
        self.assertEqual(0, code)
        self.assertFalse(report["worktree_dirty"])
        self.assertIn("inaccessible", report["worktree_status"]["stderr_tail"])

    def test_doctor_git_stdout_marks_worktree_dirty(self):
        code, report = self.doctor_with_git_output(" M tracked.py\n", "warning")
        self.assertEqual(0, code)
        self.assertTrue(report["worktree_dirty"])


class PhaseEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        command(["git", "init", "-q"], self.root)
        command(["git", "config", "user.email", "test@example.invalid"], self.root)
        command(["git", "config", "user.name", "Test"], self.root)
        (self.root / ".gitignore").write_text(".loop/\n.harness/\n", encoding="utf-8")
        (self.root / "cp1.txt").write_text("one-before\n", encoding="utf-8")
        (self.root / "cp2.txt").write_text("two-before\n", encoding="utf-8")
        (self.root / "goals").mkdir()
        self.goals = [
            {"checkpoint_id": "cp1", "objective": "first change", "acceptance_criteria": ["first works"],
             "allowed_files": ["cp1.txt"], "required_tests": ["required"]},
            {"checkpoint_id": "cp2", "objective": "second change", "acceptance_criteria": ["second works"],
             "allowed_files": ["cp2.txt"], "required_tests": ["required"]},
        ]
        for goal in self.goals:
            (self.root / "goals" / f"{goal['checkpoint_id']}.json").write_text(json.dumps(goal), encoding="utf-8")
        command(["git", "add", "."], self.root); command(["git", "commit", "-qm", "base"], self.root)
        (self.root / ".loop").mkdir()
        self.config = {
            "allowed_tests": [{"name": "required", "argv": [sys.executable, "-c", "print('pass')"]}],
            "test_timeout_seconds": 5, "claude_max_budget_usd": 0.5,
        }
        self.raw_manifest = {
            "phase_id": "phase", "objective": "complete two approved changes",
            "checkpoints": [{"checkpoint_id": "cp1", "goal": "goals/cp1.json"},
                            {"checkpoint_id": "cp2", "goal": "goals/cp2.json"}],
            "allowed_files": ["cp1.txt", "cp2.txt"], "allowed_tests": ["required"],
            "final_harness_profile": "orchestrator", "max_checkpoints": 2,
            "max_rework_rounds": 1, "max_model_calls": 8,
            "max_claude_cost_usd": 2.0, "max_elapsed_seconds": 60,
            "completion_conditions": ["all verifier checks pass", "final harness passes"],
        }

    def tearDown(self): self.temp.cleanup()

    def manifest(self, value=None):
        return validate_phase_manifest(value or self.raw_manifest, self.root, self.config, set(PROFILES))

    def budget(self):
        now = datetime.now(timezone.utc).isoformat()
        path = self.root / ".loop" / "budget.json"
        if not path.exists():
            path.write_text(json.dumps({
                "claude": {"period_usage_percent": 10, "updated_at": now, "source": "manual"},
                "codex": {"period_usage_percent": 10, "updated_at": now, "source": "manual"},
            }), encoding="utf-8")
        return BudgetManager(path)

    @staticmethod
    def harness(state="PASSED", code=0):
        def run(root, profile, **kwargs):
            return code, {"state": state, "profile": profile.name, "steps": [], "run_id": "harness",
                          "max_elapsed_seconds": kwargs.get("max_elapsed_seconds")}
        return run

    class Maker:
        def __init__(self, root, outside=False):
            self.root, self.calls, self.sessions, self.prompts, self.outside = root, 0, [], [], outside
        def run(self, prompt, session_id=None):
            self.calls += 1; self.sessions.append(session_id); self.prompts.append(prompt)
            name = "outside.txt" if self.outside else ("cp1.txt" if '"checkpoint_id": "cp1"' in prompt else "cp2.txt")
            (self.root / name).write_text(f"changed-{self.calls}\n", encoding="utf-8")
            result = MakerResult("DONE", "base", [name], [], [], "done")
            return ClaudeInvocation(result, f"session-{self.calls}", 0.1, 1)

    class RecoveringMaker:
        def __init__(self, root, failures):
            self.root, self.failures, self.calls, self.sessions = root, list(failures), 0, []
        def run(self, prompt, session_id=None):
            self.calls += 1; self.sessions.append(session_id)
            if self.failures:
                raise self.failures.pop(0)
            name = "cp1.txt" if '"checkpoint_id": "cp1"' in prompt else "cp2.txt"
            (self.root / name).write_text(f"changed-{self.calls}\n", encoding="utf-8")
            return ClaudeInvocation(MakerResult("DONE", "base", [name], [], [], "done"), f"session-{self.calls}", 0.1, 1)

    class Codex:
        def __init__(self, verdicts=None, mutate_root=None, failure_at=None):
            self.verdicts = list(verdicts or ["PASS", "PASS"]); self.prompts = []
            self.plans = 0; self.mutate_root = mutate_root; self.failure_at = failure_at
        def verify(self, prompt):
            self.prompts.append(prompt)
            if self.failure_at == len(self.prompts): raise ProcessFailure(TIMEOUT, "timeout")
            if self.mutate_root: (self.mutate_root / "verifier.txt").write_text("bad", encoding="utf-8")
            verdict = self.verdicts.pop(0)
            return VerifierResult(verdict, [] if verdict == "PASS" else ["fix"], [], [], None, [], "rework")
        def plan(self, prompt): self.plans += 1; raise AssertionError("phase mode never calls planner")

    def engine(self, maker=None, codex=None, **kwargs):
        return PhaseEngine(self.root, self.config, maker or self.Maker(self.root), codex or self.Codex(),
                           self.budget(), harness_runner=kwargs.pop("harness_runner", self.harness()), **kwargs)

    def test_valid_manifest_and_dry_run_invoke_no_process(self):
        manifest = self.manifest(); maker, codex = self.Maker(self.root), self.Codex()
        state = self.engine(maker, codex).run(manifest)
        self.assertEqual("DRY_RUN", state["state"]); self.assertFalse(state["processes_invoked"])
        self.assertEqual(0, maker.calls); self.assertEqual([], codex.prompts)

    def test_cli_run_phase_defaults_to_dry_run_without_adapters_or_probe(self):
        config = dict(self.config)
        config.update({"soft_stop_percent": 75, "hard_stop_percent": 80, "budget_validity_hours": 24,
                       "process_timeout_seconds": 10})
        (self.root / ".loop" / "config.json").write_text(json.dumps(config), encoding="utf-8")
        manifest_path = self.root / "phase.json"; manifest_path.write_text(json.dumps(self.raw_manifest), encoding="utf-8")
        output = io.StringIO()
        with patch.object(cli, "ROOT", self.root), patch.object(cli, "RUNTIME", self.root / ".loop"), \
             patch.object(cli, "ClaudeCLIAdapter", side_effect=AssertionError("no maker")), \
             patch.object(cli, "CodexCLIAdapter", side_effect=AssertionError("no codex")), \
             patch.object(cli, "probe_process", side_effect=AssertionError("no probe")), redirect_stdout(output):
            code = cli.main(["run-phase", "--manifest", manifest_path.name])
        self.assertEqual(0, code); self.assertFalse(json.loads(output.getvalue())["processes_invoked"])

    def test_invalid_manifest_json_and_schema(self):
        path = self.root / "broken.json"; path.write_text("{", encoding="utf-8")
        from tools.loop_orchestrator.phase import load_phase_manifest
        with self.assertRaises(PhaseManifestError): load_phase_manifest(path, self.root, self.config, set(PROFILES))
        invalid = dict(self.raw_manifest); invalid.pop("phase_id")
        with self.assertRaises(PhaseManifestError): self.manifest(invalid)

    def test_manifest_rejects_absolute_traversal_and_symlink_escape(self):
        for bad in ("C:/outside.json", "../outside.json"):
            value = dict(self.raw_manifest); value["checkpoints"] = [{"checkpoint_id": "cp1", "goal": bad}]
            with self.assertRaises((PhaseManifestError, GoalError)): self.manifest(value)
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda path: path.name == "cp1.json" or original(path)):
            value = dict(self.raw_manifest); value["checkpoints"] = [self.raw_manifest["checkpoints"][0]]
            with self.assertRaises(PhaseManifestError): self.manifest(value)

    def test_manifest_rejects_goal_id_template_placeholder_and_empty_contract(self):
        mutations = []
        mismatch = json.loads(json.dumps(self.raw_manifest)); mismatch["checkpoints"][0]["checkpoint_id"] = "wrong"; mutations.append(mismatch)
        for index, update in enumerate(({"template": True}, {"objective": "Replace with objective"},
                                        {"acceptance_criteria": []}, {"required_tests": []})):
            goal = dict(self.goals[0]); goal.update(update)
            path = self.root / "goals" / f"invalid-{index}.json"; path.write_text(json.dumps(goal), encoding="utf-8")
            value = json.loads(json.dumps(self.raw_manifest)); value["checkpoints"][0]["goal"] = f"goals/invalid-{index}.json"
            mutations.append(value)
        for value in mutations:
            with self.assertRaises(PhaseManifestError): self.manifest(value)

    def test_manifest_rejects_duplicate_id_scope_test_and_limits(self):
        cases = []
        duplicate = dict(self.raw_manifest); duplicate["checkpoints"] = [self.raw_manifest["checkpoints"][0]] * 2; cases.append(duplicate)
        scope = json.loads(json.dumps(self.raw_manifest)); scope["allowed_files"] = ["cp2.txt"]; cases.append(scope)
        tests = dict(self.raw_manifest); tests["allowed_tests"] = ["missing"]; cases.append(tests)
        for field in ("max_checkpoints", "max_rework_rounds", "max_model_calls", "max_claude_cost_usd", "max_elapsed_seconds"):
            value = dict(self.raw_manifest); value[field] = 0; cases.append(value)
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(PhaseManifestError): self.manifest(value)

    def test_manifest_rejects_malformed_shell_and_wildcard_allowlists(self):
        original = self.config["allowed_tests"]
        for entry in ({"name": "required", "argv": "python -m pytest"},
                      {"name": "required", "argv": ["cmd.exe", "/c", "pytest"]},
                      {"name": "required", "argv": ["python", "-m", "pytest", "tests/*"]}):
            self.config["allowed_tests"] = [entry]
            with self.assertRaises(PhaseManifestError): self.manifest()
        self.config["allowed_tests"] = original

    def test_two_checkpoints_pass_then_final_harness_ready(self):
        maker, codex = self.Maker(self.root), self.Codex()
        state = self.engine(maker, codex).run(self.manifest(), execute=True)
        self.assertEqual("READY_TO_COMMIT", state["state"], state)
        self.assertEqual(["cp1", "cp2"], state["completed_checkpoints"])
        self.assertEqual(2, maker.calls); self.assertEqual(2, len(codex.prompts)); self.assertEqual(0, codex.plans)
        self.assertEqual("changed-1\n", (self.root / "cp1.txt").read_text())
        self.assertEqual("changed-2\n", (self.root / "cp2.txt").read_text())

    def test_fail_rework_pass_then_next_checkpoint(self):
        maker, codex = self.Maker(self.root), self.Codex(["FAIL", "PASS", "PASS"])
        state = self.engine(maker, codex).run(self.manifest(), execute=True)
        self.assertEqual("READY_TO_COMMIT", state["state"], state); self.assertEqual(3, maker.calls)
        self.assertEqual([None, "session-1", None], maker.sessions)

    def test_checkpoint_diff_isolated_and_cumulative_changes_remain(self):
        codex = self.Codex(); state = self.engine(codex=codex).run(self.manifest(), execute=True)
        self.assertEqual("READY_TO_COMMIT", state["state"], state)
        self.assertIn("cp1.txt", codex.prompts[0]); self.assertNotIn("cp2.txt", codex.prompts[0])
        second = codex.prompts[1]
        checkpoint_section = second.split("PHASE-WIDE CHANGED FILES:", 1)[0]
        phase_section = second.split("PHASE-WIDE CHANGED FILES:", 1)[1]
        self.assertIn("cp2.txt", checkpoint_section); self.assertNotIn("cp1.txt", checkpoint_section)
        self.assertIn("cp1.txt", phase_section); self.assertIn("cp2.txt", phase_section)
        self.assertTrue((self.root / "cp1.txt").read_text().startswith("changed"))
        self.assertTrue((self.root / "cp2.txt").read_text().startswith("changed"))

    def test_phase_baseline_missing_on_resume_fails_closed(self):
        stopped = self.engine(codex=self.Codex(failure_at=2)).run(self.manifest(), execute=True)
        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        wrapper["state"].pop("phase_baseline")
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")
        resumed = self.engine().run(self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_INCOMPLETE_CHECKPOINT_STATE", resumed["stop_reason"])

    def test_checkpoint_baseline_digest_mismatch_on_resume_fails_closed(self):
        self.engine(codex=self.Codex(failure_at=2)).run(self.manifest(), execute=True)
        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        wrapper["state"]["active_checkpoint"]["baseline"]["cp2.txt"]["sha256"] = "tampered"
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")
        maker = self.Maker(self.root)
        resumed = self.engine(maker=maker).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_INCOMPLETE_CHECKPOINT_STATE", resumed["stop_reason"])
        self.assertEqual(0, maker.calls)

    def test_final_phase_guard_uses_manifest_union(self):
        class LateMutationHarness:
            def __call__(inner, root, profile, **kwargs):
                raise AssertionError("harness must not run")
        manifest = self.manifest()
        state = self.engine()._initial_state(manifest)
        state.update({"phase_baseline": {}, "completed_checkpoints": ["cp1", "cp2"],
                      "started_monotonic": 0})
        (self.root / "outside.txt").write_text("bad", encoding="utf-8")
        result = self.engine(harness_runner=LateMutationHarness())._final_harness(state, manifest)
        self.assertEqual("FINAL_PHASE_SCOPE_VIOLATION", result["stop_reason"])

    def test_prior_checkpoint_file_cannot_be_modified_without_current_permission(self):
        class BadMaker(self.Maker):
            def run(inner, prompt, session_id=None):
                result = super(BadMaker, inner).run(prompt, session_id)
                if inner.calls == 2: (inner.root / "cp1.txt").write_text("tampered\n", encoding="utf-8")
                return result
        state = self.engine(maker=BadMaker(self.root)).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual("MAKER_SAFETY_VIOLATION", state["stop_reason"])

    def test_allowed_file_violation_and_verifier_mutation_block(self):
        state = self.engine(maker=self.Maker(self.root, outside=True)).run(self.manifest(), execute=True)
        self.assertEqual("MAKER_SAFETY_VIOLATION", state["stop_reason"])
        (self.root / "outside.txt").unlink()
        state = self.engine(codex=self.Codex(mutate_root=self.root)).run(self.manifest(), execute=True)
        self.assertEqual("VERIFIER_SAFETY_VIOLATION", state["stop_reason"])

    def test_model_cost_and_elapsed_limits_block_before_call(self):
        value = dict(self.raw_manifest); value["max_model_calls"] = 1
        maker, codex = self.Maker(self.root), self.Codex()
        state = self.engine(maker, codex).run(self.manifest(value), execute=True)
        self.assertEqual("MAX_MODEL_CALLS", state["stop_reason"]); self.assertEqual([], codex.prompts)
        (self.root / "cp1.txt").write_text("one-before\n", encoding="utf-8")
        value = dict(self.raw_manifest); value["max_claude_cost_usd"] = 0.1
        maker = self.Maker(self.root); state = self.engine(maker=maker).run(self.manifest(value), execute=True)
        self.assertEqual("MAX_CLAUDE_COST_USD", state["stop_reason"]); self.assertEqual(0, maker.calls)
        ticks = iter([0, 100, 100, 100])
        maker = self.Maker(self.root); value = dict(self.raw_manifest); value["max_elapsed_seconds"] = 10
        state = self.engine(maker=maker, clock=lambda: next(ticks)).run(self.manifest(value), execute=True)
        self.assertEqual("MAX_ELAPSED_SECONDS", state["stop_reason"]); self.assertEqual(0, maker.calls)

    def test_harness_failure_and_infrastructure_mapping(self):
        state = self.engine(harness_runner=self.harness("FAILED", 1)).run(self.manifest(), execute=True)
        self.assertEqual("FAILED", state["state"], state); self.assertEqual("FINAL_HARNESS_FAILED", state["stop_reason"])
        (self.root / "cp1.txt").write_text("one-before\n", encoding="utf-8")
        (self.root / "cp2.txt").write_text("two-before\n", encoding="utf-8")
        state = self.engine(harness_runner=self.harness("BLOCKED", 2)).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual("FINAL_HARNESS_INFRASTRUCTURE_FAILURE", state["stop_reason"])

    def test_timeout_creates_handoff_and_resume_continues_without_restarting_cp1(self):
        maker, codex = self.Maker(self.root), self.Codex(failure_at=2)
        state = self.engine(maker, codex).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual(["cp1"], state["completed_checkpoints"])
        self.assertTrue((self.root / ".loop" / "handoff.json").is_file())
        self.assertTrue(any((self.root / ".loop" / "runs" / state["run_id"]).glob("*.json")))
        resumed_maker, resumed_codex = self.Maker(self.root), self.Codex(["PASS"])
        resumed = self.engine(resumed_maker, resumed_codex).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("READY_TO_COMMIT", resumed["state"], resumed); self.assertEqual(1, resumed_maker.calls)
        self.assertEqual(["cp1", "cp2"], resumed["completed_checkpoints"])

    def test_resume_noop_maker_still_verifies_pre_timeout_delta(self):
        stopped = self.engine(codex=self.Codex(failure_at=2)).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", stopped["state"])
        class NoopMaker:
            calls = 0
            def run(inner, prompt, session_id=None):
                inner.calls += 1
                return ClaudeInvocation(MakerResult("DONE", "base", [], [], [], "no change"),
                                        session_id or "resume-session", 0.1, 1)
        maker, codex = NoopMaker(), self.Codex(["PASS"])
        resumed = self.engine(maker, codex).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("READY_TO_COMMIT", resumed["state"], resumed)
        self.assertEqual(1, maker.calls)
        self.assertIn("--- a/cp2.txt", codex.prompts[0])
        self.assertIn("+changed-2", codex.prompts[0])

    def test_resume_missing_active_baseline_fails_closed(self):
        stopped = self.engine(codex=self.Codex(failure_at=2)).run(self.manifest(), execute=True)
        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        wrapper["state"]["active_checkpoint"].pop("baseline")
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")
        maker = self.Maker(self.root)
        resumed = self.engine(maker=maker).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_INCOMPLETE_CHECKPOINT_STATE", resumed["stop_reason"])
        self.assertEqual(0, maker.calls); self.assertEqual(["cp1"], stopped["completed_checkpoints"])

    def test_maker_blocked_after_edit_records_delta_and_cost(self):
        class BlockedMaker(self.Maker):
            def run(inner, prompt, session_id=None):
                invocation = super().run(prompt, session_id)
                return ClaudeInvocation(MakerResult("BLOCKED", "base", ["cp1.txt"], [], [], "blocked"),
                                        invocation.session_id, 0.25, 2,
                                        {"requested_model": "claude-sonnet-5",
                                         "canonical_models": ["claude-haiku-4-5", "claude-sonnet-5"],
                                         "primary_canonical_model": "claude-sonnet-5"})
        state = self.engine(maker=BlockedMaker(self.root)).run(self.manifest(), execute=True)
        self.assertEqual("MAKER_BLOCKED", state["stop_reason"])
        self.assertEqual(0.25, state["claude_cost_usd"])
        self.assertEqual(["cp1.txt"], state["recent_result"]["checkpoint_changed_files"])
        self.assertEqual("claude-sonnet-5", state["recent_result"]["telemetry"]["primary_canonical_model"])
        handoff = json.loads((self.root / ".loop" / "handoff.json").read_text(encoding="utf-8"))
        self.assertEqual(["claude-haiku-4-5", "claude-sonnet-5"],
                         handoff["details"]["telemetry"]["canonical_models"])

    def test_maker_process_failure_after_edit_records_delta(self):
        class FailureMaker:
            def __init__(inner, root): inner.root = root
            def run(inner, prompt, session_id=None):
                (inner.root / "cp1.txt").write_text("partial failure\n", encoding="utf-8")
                raise ProcessFailure(NONZERO_EXIT, "failed", return_code=1, stdout_tail="out", stderr_tail="err")
        state = self.engine(maker=FailureMaker(self.root)).run(self.manifest(), execute=True)
        self.assertEqual("FAILED", state["state"]); self.assertEqual("MAKER_ERROR: NONZERO_EXIT", state["stop_reason"])
        self.assertEqual(["cp1.txt"], state["recent_result"]["checkpoint_changed_files"])
        self.assertIn("before_sha256", state["recent_result"]["checkpoint_file_diagnostics"]["cp1.txt"])

    def test_maker_general_exception_and_interrupt_after_edit_record_delta(self):
        for exception in (RuntimeError("crash"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                class AbnormalMaker:
                    def __init__(inner, root): inner.root = root
                    def run(inner, prompt, session_id=None):
                        (inner.root / "cp1.txt").write_text("partial abnormal exit\n", encoding="utf-8")
                        raise exception
                state = self.engine(maker=AbnormalMaker(self.root)).run(self.manifest(), execute=True)
                expected = "INTERRUPTED" if isinstance(exception, KeyboardInterrupt) else "MAKER_ERROR"
                self.assertEqual(expected, state["stop_reason"])
                self.assertEqual(["cp1.txt"], state["recent_result"]["checkpoint_changed_files"])
                (self.root / "cp1.txt").write_text("one-before\n", encoding="utf-8")

    def test_maker_exception_with_outside_edit_prioritizes_safety_block(self):
        class UnsafeMaker:
            def __init__(inner, root): inner.root = root
            def run(inner, prompt, session_id=None):
                (inner.root / "outside.txt").write_text("unsafe\n", encoding="utf-8")
                raise RuntimeError("maker crashed")
        state = self.engine(maker=UnsafeMaker(self.root)).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual("MAKER_SAFETY_VIOLATION", state["stop_reason"])
        self.assertEqual(["outside.txt"], state["recent_result"]["guard"]["allowed_files_violations"])

    def test_maker_staged_or_head_change_is_blocked(self):
        class StagingMaker(self.Maker):
            def run(inner, prompt, session_id=None):
                result = super().run(prompt, session_id)
                command(["git", "add", "cp1.txt"], inner.root)
                return result
        state = self.engine(maker=StagingMaker(self.root)).run(self.manifest(), execute=True)
        self.assertEqual("MAKER_SAFETY_VIOLATION", state["stop_reason"])
        self.assertTrue(state["recent_result"]["guard"]["staged_violation"])

    def test_maker_head_change_is_blocked(self):
        class CommittingMaker(self.Maker):
            def run(inner, prompt, session_id=None):
                result = super().run(prompt, session_id)
                command(["git", "add", "cp1.txt"], inner.root)
                command(["git", "commit", "-qm", "unsafe fixture commit"], inner.root)
                return result
        state = self.engine(maker=CommittingMaker(self.root)).run(self.manifest(), execute=True)
        self.assertEqual("MAKER_SAFETY_VIOLATION", state["stop_reason"])
        self.assertTrue(state["recent_result"]["guard"]["head_violation"])

    def test_maker_protected_config_and_budget_changes_are_blocked(self):
        class ProtectedMaker(self.Maker):
            def run(inner, prompt, session_id=None):
                result = super().run(prompt, session_id)
                (inner.root / ".loop" / "config.json").write_text("{}", encoding="utf-8")
                (inner.root / ".loop" / "budget.json").write_text("{}", encoding="utf-8")
                (inner.root / ".claude").mkdir(exist_ok=True)
                (inner.root / ".vscode").mkdir(exist_ok=True)
                (inner.root / ".claude" / "settings.local.json").write_text("{}", encoding="utf-8")
                (inner.root / ".vscode" / "settings.json").write_text("{}", encoding="utf-8")
                return result
        state = self.engine(maker=ProtectedMaker(self.root)).run(self.manifest(), execute=True)
        self.assertEqual("MAKER_SAFETY_VIOLATION", state["stop_reason"])
        self.assertEqual([".claude/settings.local.json", ".loop/budget.json", ".loop/config.json",
                          ".vscode/settings.json"],
                         state["recent_result"]["guard"]["protected_file_violations"])

    def test_required_test_protected_file_change_is_blocked(self):
        self.config["allowed_tests"][0]["argv"] = [sys.executable, "-c",
            "from pathlib import Path; Path('.loop/config.json').write_text('{}')"]
        state = self.engine().run(self.manifest(), execute=True)
        self.assertEqual("REQUIRED_TEST_SAFETY_VIOLATION", state["stop_reason"])
        self.assertEqual([".loop/config.json"], state["recent_result"]["test_guard"]["protected_file_violations"])

    def test_verifier_protected_budget_change_is_blocked(self):
        class ProtectedCodex(self.Codex):
            def verify(inner, prompt):
                (inner.root / ".loop" / "budget.json").write_text("{}", encoding="utf-8")
                return super(ProtectedCodex, inner).verify(prompt)
        codex = ProtectedCodex(["PASS"]); codex.root = self.root
        state = self.engine(codex=codex).run(self.manifest(), execute=True)
        self.assertEqual("VERIFIER_SAFETY_VIOLATION", state["stop_reason"])
        self.assertEqual([".loop/budget.json"], state["recent_result"]["verifier_guard"]["protected_file_violations"])

    def test_harness_pass_with_product_mutation_is_blocked(self):
        def mutating(root, profile, **kwargs):
            (root / "cp1.txt").write_text("harness mutation\n", encoding="utf-8")
            return 0, {"state": "PASSED", "steps": []}
        state = self.engine(harness_runner=mutating).run(self.manifest(), execute=True)
        self.assertEqual("FINAL_HARNESS_MODIFIED_WORKTREE", state["stop_reason"])
        self.assertIn("cp1.txt", state["recent_result"]["guard"]["changed_files"])

    def test_harness_staged_change_is_blocked(self):
        def staging(root, profile, **kwargs):
            command(["git", "add", "cp1.txt", "cp2.txt"], root)
            return 0, {"state": "PASSED", "steps": []}
        state = self.engine(harness_runner=staging).run(self.manifest(), execute=True)
        self.assertEqual("FINAL_HARNESS_MODIFIED_WORKTREE", state["stop_reason"])
        self.assertTrue(state["recent_result"]["guard"]["staged_violation"])

    def test_harness_head_change_is_blocked(self):
        def committing(root, profile, **kwargs):
            command(["git", "add", "cp1.txt", "cp2.txt"], root)
            command(["git", "commit", "-qm", "unsafe harness fixture"], root)
            return 0, {"state": "PASSED", "steps": []}
        state = self.engine(harness_runner=committing).run(self.manifest(), execute=True)
        self.assertEqual("FINAL_HARNESS_MODIFIED_WORKTREE", state["stop_reason"])
        self.assertTrue(state["recent_result"]["guard"]["head_violation"])

    def test_harness_exceptions_are_structured_blocks(self):
        factories = [lambda: OSError("mkdir failed"), lambda: ProcessFailure(TIMEOUT, "timeout")]
        for factory in factories:
            with self.subTest(factory=factory):
                def failing(root, profile, **kwargs): raise factory()
                state = self.engine(harness_runner=failing).run(self.manifest(), execute=True)
                self.assertEqual("FINAL_HARNESS_START_FAILED", state["stop_reason"])
                self.assertIn("error", state["recent_result"])
                (self.root / "cp1.txt").write_text("one-before\n", encoding="utf-8")
                (self.root / "cp2.txt").write_text("two-before\n", encoding="utf-8")

    def test_harness_keyboard_interrupt_is_structured_block(self):
        def interrupted(root, profile, **kwargs): raise KeyboardInterrupt
        state = self.engine(harness_runner=interrupted).run(self.manifest(), execute=True)
        self.assertEqual("FINAL_HARNESS_INTERRUPTED", state["stop_reason"])
        self.assertEqual("KeyboardInterrupt", state["recent_result"]["error"]["type"])

    def test_post_call_cost_overrun_blocks_before_verifier(self):
        self.config["claude_max_budget_usd"] = 0.1
        value = dict(self.raw_manifest); value["max_claude_cost_usd"] = 0.15
        class ExpensiveMaker(self.Maker):
            def run(inner, prompt, session_id=None):
                result = super().run(prompt, session_id)
                return ClaudeInvocation(result.result, result.session_id, 0.2, result.num_turns)
        codex = self.Codex()
        state = self.engine(maker=ExpensiveMaker(self.root), codex=codex).run(self.manifest(value), execute=True)
        self.assertEqual("MAX_CLAUDE_COST_USD", state["stop_reason"])
        self.assertEqual(0.2, state["claude_cost_usd"]); self.assertEqual([], codex.prompts)

    def test_resume_manifest_base_and_hash_mismatch_fail_closed(self):
        state = self.engine(codex=self.Codex(failure_at=2)).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"])
        changed = dict(self.raw_manifest); changed["objective"] = "different approved objective"
        resumed = self.engine().run(self.manifest(changed), execute=True, resume=True)
        self.assertEqual("RESUME_MANIFEST_MISMATCH", resumed["stop_reason"])
        # Restore the original stopped state and exercise base-commit mismatch.
        original_base = state["base_commit"]
        state["base_commit"] = "different-base"
        state["started_monotonic"] = 0
        self.engine()._persist(state, "restore-test-state")
        resumed = self.engine().run(self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_BASE_COMMIT_MISMATCH", resumed["stop_reason"])
        # Restore again, then corrupt a completed checkpoint file.
        state["base_commit"] = original_base
        state["started_monotonic"] = 0
        self.engine()._persist(state, "restore-test-state")
        (self.root / "cp1.txt").write_text("external mutation\n", encoding="utf-8")
        resumed = self.engine().run(self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_EXTERNAL_CHANGE_DETECTED", resumed["stop_reason"])

    def test_phase_usage_at_seventy_blocks_without_model_call(self):
        maker = self.Maker(self.root); budget = self.budget()
        data = budget.read(); data["codex"]["period_usage_percent"] = 70
        budget.path.write_text(json.dumps(data), encoding="utf-8")
        state = PhaseEngine(self.root, self.config, maker, self.Codex(), budget,
                            harness_runner=self.harness()).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual(0, maker.calls)

    def test_execute_and_resume_preflight_failure_blocks_without_model_call(self):
        maker = self.Maker(self.root)
        failure = lambda: {"integration": "codex", "kind": PROCESS_START_FAILED}
        state = self.engine(maker=maker, preflight=failure).run(self.manifest(), execute=True)
        self.assertEqual("EXECUTABLE_PREFLIGHT_FAILED", state["stop_reason"]); self.assertEqual(0, maker.calls)

        stopped = self.engine(codex=self.Codex(failure_at=2)).run(self.manifest(), execute=True)
        resumed_maker = self.Maker(self.root)
        resumed = self.engine(maker=resumed_maker, preflight=failure).run(
            self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_EXECUTABLE_PREFLIGHT_FAILED", resumed["stop_reason"])
        self.assertEqual(0, resumed_maker.calls); self.assertEqual(["cp1"], stopped["completed_checkpoints"])

    def test_budget_exhausted_continues_same_session_once_then_succeeds(self):
        failure = ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget exhausted", return_code=1,
                                 metadata={"session_id": "resume-session", "total_cost_usd": 0.3,
                                           "num_turns": 10, "subtype": "error_max_budget"})
        maker, codex = self.RecoveringMaker(self.root, [failure]), self.Codex()
        state = self.engine(maker=maker, codex=codex).run(self.manifest(), execute=True)
        self.assertEqual("READY_TO_COMMIT", state["state"], state)
        self.assertEqual([None, "resume-session", None], maker.sessions)
        self.assertAlmostEqual(0.5, state["claude_cost_usd"], places=6)
        self.assertEqual(3, maker.calls); self.assertEqual(3, state["maker_calls"])
        self.assertEqual(5, state["model_calls"])

    def test_max_turns_without_session_blocks_without_continuation(self):
        failure = ProcessFailure(MODEL_MAX_TURNS, "max turns", return_code=1,
                                 metadata={"total_cost_usd": 0.2, "num_turns": 40, "subtype": "error_max_turns"})
        maker = self.RecoveringMaker(self.root, [failure])
        state = self.engine(maker=maker).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual(MODEL_MAX_TURNS, state["stop_reason"])
        self.assertEqual(1, maker.calls); self.assertEqual(0.2, state["claude_cost_usd"])

    def test_api_connection_error_retries_once_in_new_session_then_succeeds(self):
        failure = ProcessFailure(API_CONNECTION_ERROR, "conn", return_code=1,
                                 metadata={"session_id": "dropped", "total_cost_usd": 0.0,
                                           "subtype": "error_api_connection"})
        maker, codex = self.RecoveringMaker(self.root, [failure]), self.Codex()
        state = self.engine(maker=maker, codex=codex).run(self.manifest(), execute=True)
        self.assertEqual("READY_TO_COMMIT", state["state"], state)
        self.assertEqual([None, None, None], maker.sessions)

    def test_api_connection_error_repeated_blocks_after_one_retry(self):
        failures = [ProcessFailure(API_CONNECTION_ERROR, "conn", return_code=1,
                                   metadata={"subtype": "error_api_connection"}) for _ in range(2)]
        maker = self.RecoveringMaker(self.root, failures)
        state = self.engine(maker=maker).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual(API_CONNECTION_ERROR, state["stop_reason"])
        self.assertEqual(2, maker.calls)

    def test_api_connection_error_with_file_change_blocks_without_retry(self):
        class DirtyFailureMaker:
            def __init__(inner, root): inner.root, inner.calls = root, 0
            def run(inner, prompt, session_id=None):
                inner.calls += 1
                (inner.root / "cp1.txt").write_text("partial\n", encoding="utf-8")
                raise ProcessFailure(API_CONNECTION_ERROR, "conn", return_code=1,
                                     metadata={"subtype": "error_api_connection"})
        maker = DirtyFailureMaker(self.root)
        state = self.engine(maker=maker).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual(API_CONNECTION_ERROR, state["stop_reason"])
        self.assertEqual(1, maker.calls)

    def test_safety_violation_prevents_recovery_even_with_valid_session(self):
        class UnsafeRecoverableMaker:
            def __init__(inner, root): inner.root, inner.calls = root, 0
            def run(inner, prompt, session_id=None):
                inner.calls += 1
                (inner.root / "outside.txt").write_text("unsafe\n", encoding="utf-8")
                raise ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget", return_code=1,
                                     metadata={"session_id": "sess", "subtype": "error_max_budget"})
        maker = UnsafeRecoverableMaker(self.root)
        state = self.engine(maker=maker).run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual("MAKER_SAFETY_VIOLATION", state["stop_reason"])
        self.assertEqual(1, maker.calls)

    def test_cost_overflow_blocks_before_continuation_or_verifier(self):
        config = dict(self.config, claude_max_budget_usd=0.1)
        value = dict(self.raw_manifest); value["max_claude_cost_usd"] = 0.25
        failure = ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget", return_code=1,
                                 metadata={"session_id": "sess", "total_cost_usd": 0.3, "subtype": "error_max_budget"})
        maker, codex = self.RecoveringMaker(self.root, [failure]), self.Codex()
        engine = PhaseEngine(self.root, config, maker, codex, self.budget(), harness_runner=self.harness())
        state = engine.run(self.manifest(value), execute=True)
        self.assertEqual("MAX_CLAUDE_COST_USD", state["stop_reason"]); self.assertEqual(1, maker.calls)
        self.assertEqual([], codex.prompts)

    def test_maker_before_state_persists_incremented_calls_before_run(self):
        root, seen = self.root, []
        class InspectingMaker(self.Maker):
            def run(inner, prompt, session_id=None):
                wrapper = json.loads((root / ".loop" / "state.json").read_text(encoding="utf-8"))
                saved = wrapper["state"]
                seen.append((saved["model_calls"], saved["maker_calls"], saved["active_checkpoint"]["pending_call"]))
                return super().run(prompt, session_id)
        maker = InspectingMaker(self.root)
        state = self.engine(maker=maker, codex=self.Codex()).run(self.manifest(), execute=True)
        self.assertEqual("READY_TO_COMMIT", state["state"], state)
        self.assertEqual(2, len(seen))
        self.assertEqual((1, 1), seen[0][:2]); self.assertEqual((3, 2), seen[1][:2])
        self.assertIsNotNone(seen[0][2]); self.assertIsNotNone(seen[1][2])
        self.assertEqual(0, seen[0][2]["call_index"]); self.assertIsNone(seen[0][2]["recovery_kind"])
        self.assertEqual(0, seen[1][2]["call_index"]); self.assertIsNone(seen[1][2]["recovery_kind"])

    def test_continuation_interrupted_before_retry_resumes_same_session_with_finish_prompt(self):
        failure = ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget exhausted", return_code=1,
                                 metadata={"session_id": "crash-session", "total_cost_usd": 0.1,
                                           "num_turns": 5, "subtype": "error_max_budget"})
        setup_config = dict(self.config); setup_config["max_maker_continuations"] = 0
        setup_maker = self.RecoveringMaker(self.root, [failure])
        setup_engine = PhaseEngine(self.root, setup_config, setup_maker, self.Codex(), self.budget(),
                                   harness_runner=self.harness())
        state = setup_engine.run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual(1, setup_maker.calls)
        self.assertEqual(0, state["active_checkpoint"]["continuation_count"])
        self.assertIsNone(state["active_checkpoint"]["pending_call"])

        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        record = wrapper["state"]["active_checkpoint"]
        record["continuation_count"] = 1
        record["pending_call"] = {"attempt": 0, "call_index": 1, "recovery_kind": "continuation",
                                  "session_id": "crash-session", "continuation_count": 1, "api_retry_count": 0}
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")

        class ResumeMaker:
            def __init__(inner, root): inner.root, inner.calls, inner.sessions, inner.prompts = root, 0, [], []
            def run(inner, prompt, session_id=None):
                inner.calls += 1; inner.sessions.append(session_id); inner.prompts.append(prompt)
                name = "cp1.txt" if inner.calls == 1 else "cp2.txt"
                (inner.root / name).write_text(f"resumed-{inner.calls}\n", encoding="utf-8")
                return ClaudeInvocation(MakerResult("DONE", "base", [name], [], [], "done"),
                                        f"post-crash-session-{inner.calls}", 0.05, 1)
        resumed_maker = ResumeMaker(self.root)
        resumed_engine = PhaseEngine(self.root, dict(self.config, max_maker_continuations=1), resumed_maker,
                                     self.Codex(["PASS", "PASS"]), self.budget(), harness_runner=self.harness())
        resumed = resumed_engine.run(self.manifest(), execute=True, resume=True)
        self.assertEqual("READY_TO_COMMIT", resumed["state"], resumed)
        self.assertEqual(2, resumed_maker.calls)
        self.assertEqual("crash-session", resumed_maker.sessions[0])
        self.assertIn("CONTINUE_CHECKPOINT", resumed_maker.prompts[0])
        self.assertIsNone(resumed_maker.sessions[1])

    def test_continuation_resume_cannot_repeat_allowance_after_crash(self):
        failure = ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget exhausted", return_code=1,
                                 metadata={"session_id": "crash-session", "subtype": "error_max_budget"})
        setup_config = dict(self.config); setup_config["max_maker_continuations"] = 0
        setup_maker = self.RecoveringMaker(self.root, [failure])
        setup_engine = PhaseEngine(self.root, setup_config, setup_maker, self.Codex(), self.budget(),
                                   harness_runner=self.harness())
        setup_engine.run(self.manifest(), execute=True)

        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        record = wrapper["state"]["active_checkpoint"]
        record["continuation_count"] = 1
        record["pending_call"] = {"attempt": 0, "call_index": 1, "recovery_kind": "continuation",
                                  "session_id": "crash-session", "continuation_count": 1, "api_retry_count": 0}
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")

        class FailingAgainMaker:
            def __init__(inner, root): inner.root, inner.calls, inner.sessions = root, 0, []
            def run(inner, prompt, session_id=None):
                inner.calls += 1; inner.sessions.append(session_id)
                raise ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget again", return_code=1,
                                     metadata={"session_id": "crash-session", "subtype": "error_max_budget"})
        resumed_maker = FailingAgainMaker(self.root)
        resumed_engine = PhaseEngine(self.root, dict(self.config, max_maker_continuations=1), resumed_maker,
                                     self.Codex(), self.budget(), harness_runner=self.harness())
        resumed = resumed_engine.run(self.manifest(), execute=True, resume=True)
        self.assertEqual("BLOCKED", resumed["state"]); self.assertEqual(MODEL_BUDGET_EXHAUSTED, resumed["stop_reason"])
        self.assertEqual(1, resumed_maker.calls)

    def test_api_retry_interrupted_before_call_resumes_fresh_session_with_full_prompt(self):
        failure = ProcessFailure(API_CONNECTION_ERROR, "conn", return_code=1,
                                 metadata={"session_id": "dropped-session", "total_cost_usd": 0.0,
                                           "subtype": "error_api_connection"})
        setup_config = dict(self.config); setup_config["max_api_connection_retries"] = 0
        setup_maker = self.RecoveringMaker(self.root, [failure])
        setup_engine = PhaseEngine(self.root, setup_config, setup_maker, self.Codex(), self.budget(),
                                   harness_runner=self.harness())
        state = setup_engine.run(self.manifest(), execute=True)
        self.assertEqual("BLOCKED", state["state"]); self.assertEqual(1, setup_maker.calls)
        self.assertEqual(0, state["active_checkpoint"]["api_retry_count"])

        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        record = wrapper["state"]["active_checkpoint"]
        record["api_retry_count"] = 1
        record["pending_call"] = {"attempt": 0, "call_index": 1, "recovery_kind": "api_retry",
                                  "session_id": None, "continuation_count": 0, "api_retry_count": 1}
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")

        class ResumeMaker:
            def __init__(inner, root): inner.root, inner.calls, inner.sessions, inner.prompts = root, 0, [], []
            def run(inner, prompt, session_id=None):
                inner.calls += 1; inner.sessions.append(session_id); inner.prompts.append(prompt)
                name = "cp1.txt" if inner.calls == 1 else "cp2.txt"
                (inner.root / name).write_text(f"resumed-{inner.calls}\n", encoding="utf-8")
                return ClaudeInvocation(MakerResult("DONE", "base", [name], [], [], "done"),
                                        f"fresh-session-{inner.calls}", 0.05, 1)
        resumed_maker = ResumeMaker(self.root)
        resumed_engine = PhaseEngine(self.root, dict(self.config, max_api_connection_retries=1), resumed_maker,
                                     self.Codex(["PASS", "PASS"]), self.budget(), harness_runner=self.harness())
        resumed = resumed_engine.run(self.manifest(), execute=True, resume=True)
        self.assertEqual("READY_TO_COMMIT", resumed["state"], resumed)
        self.assertIsNone(resumed_maker.sessions[0])
        self.assertNotIn("CONTINUE_CHECKPOINT", resumed_maker.prompts[0])

    def test_api_retry_resume_cannot_repeat_allowance_after_crash(self):
        failure = ProcessFailure(API_CONNECTION_ERROR, "conn", return_code=1,
                                 metadata={"subtype": "error_api_connection"})
        setup_config = dict(self.config); setup_config["max_api_connection_retries"] = 0
        setup_maker = self.RecoveringMaker(self.root, [failure])
        setup_engine = PhaseEngine(self.root, setup_config, setup_maker, self.Codex(), self.budget(),
                                   harness_runner=self.harness())
        setup_engine.run(self.manifest(), execute=True)

        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        record = wrapper["state"]["active_checkpoint"]
        record["api_retry_count"] = 1
        record["pending_call"] = {"attempt": 0, "call_index": 1, "recovery_kind": "api_retry",
                                  "session_id": None, "continuation_count": 0, "api_retry_count": 1}
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")

        class FailingAgainMaker:
            def __init__(inner, root): inner.root, inner.calls, inner.sessions = root, 0, []
            def run(inner, prompt, session_id=None):
                inner.calls += 1; inner.sessions.append(session_id)
                raise ProcessFailure(API_CONNECTION_ERROR, "conn again", return_code=1,
                                     metadata={"subtype": "error_api_connection"})
        resumed_maker = FailingAgainMaker(self.root)
        resumed_engine = PhaseEngine(self.root, dict(self.config, max_api_connection_retries=1), resumed_maker,
                                     self.Codex(), self.budget(), harness_runner=self.harness())
        resumed = resumed_engine.run(self.manifest(), execute=True, resume=True)
        self.assertEqual("BLOCKED", resumed["state"]); self.assertEqual(API_CONNECTION_ERROR, resumed["stop_reason"])
        self.assertEqual(1, resumed_maker.calls)

    def test_resume_ambiguous_recovery_state_fails_closed(self):
        failure = ProcessFailure(API_CONNECTION_ERROR, "conn", return_code=1,
                                 metadata={"subtype": "error_api_connection"})
        setup_config = dict(self.config); setup_config["max_api_connection_retries"] = 0
        cases = [
            {"attempt": 0, "call_index": 1, "recovery_kind": "continuation", "session_id": None},
            {"attempt": 0, "call_index": 1, "recovery_kind": "api_retry", "session_id": "should-be-null"},
            {"attempt": 1, "call_index": 1, "recovery_kind": "continuation", "session_id": "sess"},
            {"attempt": 0, "call_index": 1, "recovery_kind": "unknown", "session_id": None},
            {"attempt": 0, "call_index": "not-an-int", "recovery_kind": None, "session_id": None},
        ]
        for pending in cases:
            with self.subTest(pending=pending):
                setup_maker = self.RecoveringMaker(self.root, [failure])
                setup_engine = PhaseEngine(self.root, setup_config, setup_maker, self.Codex(), self.budget(),
                                           harness_runner=self.harness())
                state = setup_engine.run(self.manifest(), execute=True)
                self.assertEqual("BLOCKED", state["state"])
                state_path = self.root / ".loop" / "state.json"
                wrapper = json.loads(state_path.read_text(encoding="utf-8"))
                wrapper["state"]["active_checkpoint"]["pending_call"] = pending
                state_path.write_text(json.dumps(wrapper), encoding="utf-8")
                resumed_maker = self.Maker(self.root)
                resumed = self.engine(maker=resumed_maker).run(self.manifest(), execute=True, resume=True)
                self.assertEqual("RESUME_RECOVERY_STATE_INCOMPLETE", resumed["stop_reason"])
                self.assertEqual(0, resumed_maker.calls)

    def test_repeated_recoverable_failures_block_after_exactly_one_continuation(self):
        for kind, subtype in ((MODEL_BUDGET_EXHAUSTED, "error_max_budget"), (MODEL_MAX_TURNS, "error_max_turns")):
            with self.subTest(kind=kind):
                first = ProcessFailure(kind, "first", return_code=1,
                                       metadata={"session_id": "sess-1", "subtype": subtype})
                second = ProcessFailure(kind, "second", return_code=1,
                                        metadata={"session_id": "sess-1", "subtype": subtype})
                maker = self.RecoveringMaker(self.root, [first, second])
                state = self.engine(maker=maker).run(self.manifest(), execute=True)
                self.assertEqual("BLOCKED", state["state"]); self.assertEqual(kind, state["stop_reason"])
                self.assertEqual(2, maker.calls)
                self.assertEqual(1, state["active_checkpoint"]["continuation_count"])
                self.assertEqual(2, state["maker_calls"]); self.assertEqual(2, state["model_calls"])
                (self.root / "cp1.txt").write_text("one-before\n", encoding="utf-8")

    def _crash_on_nth_claude_before_call(self, n):
        original = PhaseEngine._before_call
        calls = []
        def patched(inner, state, manifest, role):
            result = original(inner, state, manifest, role)
            if role == "claude":
                calls.append(1)
                if len(calls) == n:
                    raise SystemExit("simulated crash before next model call")
            return result
        return patched

    def test_continuation_recovery_transition_persists_before_next_call_and_survives_crash(self):
        failure = ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget exhausted", return_code=1,
                                 metadata={"session_id": "crash-session", "total_cost_usd": 0.1,
                                           "num_turns": 5, "subtype": "error_max_budget"})
        maker = self.RecoveringMaker(self.root, [failure])
        engine = PhaseEngine(self.root, dict(self.config, max_maker_continuations=1), maker,
                             self.Codex(), self.budget(), harness_runner=self.harness())
        with patch.object(PhaseEngine, "_before_call", self._crash_on_nth_claude_before_call(3)):
            with self.assertRaises(SystemExit):
                engine.run(self.manifest(), execute=True)
        self.assertEqual(1, maker.calls)

        wrapper = json.loads((self.root / ".loop" / "state.json").read_text(encoding="utf-8"))
        crashed = wrapper["state"]
        record = crashed["active_checkpoint"]
        self.assertEqual(1, record["continuation_count"])
        self.assertEqual({"attempt": 0, "call_index": 1, "recovery_kind": "continuation",
                          "session_id": "crash-session", "continuation_count": 1, "api_retry_count": 0},
                         record["pending_call"])
        self.assertAlmostEqual(0.1, crashed["claude_cost_usd"], places=6)
        self.assertEqual(1, crashed["model_calls"]); self.assertEqual(1, crashed["maker_calls"])

        resumed_maker = self.Maker(self.root)
        resumed = PhaseEngine(self.root, dict(self.config, max_maker_continuations=1), resumed_maker,
                             self.Codex(["PASS", "PASS"]), self.budget(), harness_runner=self.harness()
                             ).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("READY_TO_COMMIT", resumed["state"], resumed)
        self.assertEqual(2, resumed_maker.calls)
        self.assertEqual("crash-session", resumed_maker.sessions[0])
        self.assertIn("CONTINUE_CHECKPOINT", resumed_maker.prompts[0])
        self.assertIsNone(resumed_maker.sessions[1])
        self.assertEqual(2, len(resumed["checkpoints"][0]["attempts"]))
        self.assertAlmostEqual(0.3, resumed["claude_cost_usd"], places=6)
        self.assertEqual(5, resumed["model_calls"]); self.assertEqual(3, resumed["maker_calls"])

    def test_api_retry_recovery_transition_persists_before_next_call_and_survives_crash(self):
        failure = ProcessFailure(API_CONNECTION_ERROR, "conn", return_code=1,
                                 metadata={"session_id": "dropped-session", "total_cost_usd": 0.0,
                                           "subtype": "error_api_connection"})
        maker = self.RecoveringMaker(self.root, [failure])
        engine = PhaseEngine(self.root, dict(self.config, max_api_connection_retries=1), maker,
                             self.Codex(), self.budget(), harness_runner=self.harness())
        with patch.object(PhaseEngine, "_before_call", self._crash_on_nth_claude_before_call(3)):
            with self.assertRaises(SystemExit):
                engine.run(self.manifest(), execute=True)
        self.assertEqual(1, maker.calls)

        wrapper = json.loads((self.root / ".loop" / "state.json").read_text(encoding="utf-8"))
        crashed = wrapper["state"]
        record = crashed["active_checkpoint"]
        self.assertEqual(1, record["api_retry_count"])
        self.assertEqual({"attempt": 0, "call_index": 1, "recovery_kind": "api_retry",
                          "session_id": None, "continuation_count": 0, "api_retry_count": 1},
                         record["pending_call"])

        resumed_maker = self.Maker(self.root)
        resumed = PhaseEngine(self.root, dict(self.config, max_api_connection_retries=1), resumed_maker,
                             self.Codex(["PASS", "PASS"]), self.budget(), harness_runner=self.harness()
                             ).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("READY_TO_COMMIT", resumed["state"], resumed)
        self.assertIsNone(resumed_maker.sessions[0])
        self.assertNotIn("CONTINUE_CHECKPOINT", resumed_maker.prompts[0])
        self.assertEqual(2, len(resumed["checkpoints"][0]["attempts"]))
        self.assertAlmostEqual(0.2, resumed["claude_cost_usd"], places=6)
        self.assertEqual(5, resumed["model_calls"]); self.assertEqual(3, resumed["maker_calls"])

    def test_resume_missing_pending_call_after_recorded_recovery_decision_fails_closed(self):
        failure = ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget exhausted", return_code=1,
                                 metadata={"session_id": "crash-session", "subtype": "error_max_budget"})
        setup_config = dict(self.config); setup_config["max_maker_continuations"] = 0
        setup_maker = self.RecoveringMaker(self.root, [failure])
        setup_engine = PhaseEngine(self.root, setup_config, setup_maker, self.Codex(), self.budget(),
                                   harness_runner=self.harness())
        setup_engine.run(self.manifest(), execute=True)

        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        record = wrapper["state"]["active_checkpoint"]
        record["continuation_count"] = 1
        record["pending_call"] = None
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")

        resumed_maker = self.Maker(self.root)
        resumed = self.engine(maker=resumed_maker).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_RECOVERY_STATE_INCOMPLETE", resumed["stop_reason"])
        self.assertEqual(0, resumed_maker.calls)

    def test_resume_inconsistent_recovery_counters_fail_closed(self):
        failure = ProcessFailure(MODEL_BUDGET_EXHAUSTED, "budget exhausted", return_code=1,
                                 metadata={"session_id": "crash-session", "subtype": "error_max_budget"})
        setup_config = dict(self.config); setup_config["max_maker_continuations"] = 0
        setup_maker = self.RecoveringMaker(self.root, [failure])
        setup_engine = PhaseEngine(self.root, setup_config, setup_maker, self.Codex(), self.budget(),
                                   harness_runner=self.harness())
        setup_engine.run(self.manifest(), execute=True)

        state_path = self.root / ".loop" / "state.json"
        wrapper = json.loads(state_path.read_text(encoding="utf-8"))
        record = wrapper["state"]["active_checkpoint"]
        record["continuation_count"] = 1
        record["pending_call"] = {"attempt": 0, "call_index": 1, "recovery_kind": "continuation",
                                  "session_id": "crash-session", "continuation_count": 5, "api_retry_count": 0}
        state_path.write_text(json.dumps(wrapper), encoding="utf-8")

        resumed_maker = self.Maker(self.root)
        resumed = self.engine(maker=resumed_maker).run(self.manifest(), execute=True, resume=True)
        self.assertEqual("RESUME_RECOVERY_STATE_INCOMPLETE", resumed["stop_reason"])
        self.assertEqual(0, resumed_maker.calls)


if __name__ == "__main__": unittest.main()
