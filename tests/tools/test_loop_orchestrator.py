from __future__ import annotations

import io, json, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools.loop_orchestrator import adapters, cli
from tools.loop_orchestrator.adapters import ClaudeCLIAdapter, CodexCLIAdapter, codex_exec_prefix
from tools.loop_orchestrator.budget import BudgetManager
from tools.loop_orchestrator.engine import LoopEngine
from tools.loop_orchestrator.models import ClaudeInvocation, MakerResult, PlannerResult, State, VerifierResult
from tools.loop_orchestrator.process import (
    JSON_PARSE_FAILED, NONZERO_EXIT, OUTPUT_TAIL_LIMIT, PROCESS_START_FAILED, TIMEOUT,
    ProcessFailure, ProcessResult, probe_process, run_json_process,
)
from tools.loop_orchestrator.repository import git as repository_git


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
        for error in (PermissionError("denied"), FileExistsError("collision")):
            with self.subTest(error=type(error).__name__):
                codex = FakeCodex(["PASS"])
                def mkdir(path, *args, **kwargs):
                    if "test-temp" in path.parts: raise error
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
        def fake(argv, root, timeout): captured["argv"] = argv; return self.fixture()
        with patch.object(adapters, "run_json_process", side_effect=fake):
            result = ClaudeCLIAdapter(Path.cwd(), 10, 1).run("prompt")
        self.assertEqual("11111111-2222-4333-8444-555555555555", result.session_id)
        self.assertEqual(0.042, result.total_cost_usd); self.assertEqual(3, result.num_turns)
        argv = captured["argv"]; self.assertIn("--disallowedTools", argv); self.assertEqual("Bash", argv[argv.index("--disallowedTools")+1])
        self.assertNotIn("Bash", argv[argv.index("--allowedTools")+1].split(","))
        self.assertNotIn("session_id", adapters.MAKER_SCHEMA["properties"])

    def test_real_cli_envelope_fixture_parser_preserves_metadata(self):
        fixture = Path(__file__).parent/"fixtures"/"claude_result.json"
        script = "from pathlib import Path; print(Path(r'%s').read_text(encoding='utf-8'))" % fixture
        parsed = run_json_process([sys.executable, "-c", script], Path.cwd(), 10)
        self.assertEqual("11111111-2222-4333-8444-555555555555", parsed.metadata["session_id"])
        self.assertEqual("DONE", parsed.payload["status"])

    def test_resume_uses_envelope_session_only(self):
        captured = {}
        def fake(argv, root, timeout): captured["argv"] = argv; return self.fixture()
        with patch.object(adapters, "run_json_process", side_effect=fake): ClaudeCLIAdapter(Path.cwd(), 10, 1).run("fix", "actual-session")
        self.assertEqual("actual-session", captured["argv"][captured["argv"].index("--resume")+1])

    def test_codex_adapter_uses_shared_production_prefix(self):
        captured = {}
        payload = {"verdict":"PASS","findings":[],"failed_checks":[],"tests_observed":[],
                   "visual_verification":None,"residual_risks":[],"recommended_action":"none"}
        def fake(argv, root, timeout, json_lines=False):
            captured["argv"] = argv; return ProcessResult(payload, "", "", {})
        adapter = CodexCLIAdapter(Path("repo"), 10, Path("schemas"), "codex-custom")
        with patch.object(adapters, "run_json_process", side_effect=fake): adapter.verify("actual prompt")
        prefix = codex_exec_prefix("codex-custom", Path("repo"))
        self.assertEqual(prefix, captured["argv"][:len(prefix)])
        self.assertEqual("actual prompt", captured["argv"][len(prefix)])

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
        process.kill.assert_called_once_with()

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
if __name__ == "__main__": unittest.main()
