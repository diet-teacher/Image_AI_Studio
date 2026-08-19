from __future__ import annotations

import io, json, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools.loop_orchestrator import adapters, cli
from tools.loop_orchestrator.adapters import ClaudeCLIAdapter
from tools.loop_orchestrator.budget import BudgetManager
from tools.loop_orchestrator.engine import LoopEngine
from tools.loop_orchestrator.models import ClaudeInvocation, MakerResult, PlannerResult, State, VerifierResult
from tools.loop_orchestrator.process import ProcessResult
from tools.loop_orchestrator.process import run_json_process


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
    def __init__(self, verdicts, root=None, mutate=False): self.verdicts, self.plans, self.root, self.mutate = list(verdicts), 0, root, mutate
    def verify(self, prompt):
        if self.mutate: (self.root/"verifier-write.txt").write_text("forbidden", encoding="utf-8")
        verdict = self.verdicts.pop(0)
        return VerifierResult(verdict, [] if verdict == "PASS" else ["fix"], [] if verdict == "PASS" else ["check"], [], None, [], "rework")
    def plan(self, prompt):
        self.plans += 1
        return PlannerResult("next", "next objective", ["works"], ["next.txt"], ["required"], "implement next", "small")


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
    def engine(self, maker, codex): return LoopEngine(self.root, self.config, maker, codex, self.budget())

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
        def fake(argv, **kwargs):
            if "codex" in str(argv[0]): raise PermissionError("access denied")
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        output = io.StringIO()
        with patch.object(cli.subprocess, "run", side_effect=fake), redirect_stdout(output): code = cli.doctor()
        self.assertEqual(1, code); self.assertIn("PermissionError", output.getvalue())


if __name__ == "__main__": unittest.main()
