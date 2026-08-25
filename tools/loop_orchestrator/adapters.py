from __future__ import annotations

import json
from pathlib import Path

from .models import ClaudeInvocation, MakerResult, PlannerResult, VerifierResult
from .process import run_json_process


MAKER_SCHEMA = {"type":"object","additionalProperties":False,"required":["status","base_commit","changed_files","tests_run","known_risks","summary"],"properties":{"status":{"type":"string"},"base_commit":{"type":"string"},"changed_files":{"type":"array","items":{"type":"string"}},"tests_run":{"type":"array","items":{"type":"string"}},"known_risks":{"type":"array","items":{"type":"string"}},"summary":{"type":"string"}}}
VERIFIER_SCHEMA = {"type":"object","required":["verdict","findings","failed_checks","tests_observed","visual_verification","residual_risks","recommended_action"],"properties":{k:{} for k in ["verdict","findings","failed_checks","tests_observed","visual_verification","residual_risks","recommended_action"]}}
PLANNER_SCHEMA = {"type":"object","required":["checkpoint_id","objective","acceptance_criteria","allowed_files","required_tests","claude_prompt","estimated_scope"],"properties":{k:{} for k in ["checkpoint_id","objective","acceptance_criteria","allowed_files","required_tests","claude_prompt","estimated_scope"]}}


def codex_exec_prefix(executable: str, root: Path) -> list[str]:
    """Production Codex global options shared by execution and free preflight help."""
    return [executable, "--ask-for-approval", "never", "--sandbox", "read-only", "--cd", str(root), "exec"]


class ClaudeCLIAdapter:
    def __init__(self, root: Path, timeout: int, max_budget_usd: float):
        self.root, self.timeout, self.max_budget_usd = root, timeout, max_budget_usd
        self.executable = "claude"

    def run(self, prompt: str, session_id: str | None = None) -> ClaudeInvocation:
        argv = [self.executable, "-p", "--output-format", "json", "--json-schema", json.dumps(MAKER_SCHEMA),
                "--max-budget-usd", str(self.max_budget_usd), "--permission-mode", "acceptEdits",
                "--allowedTools", "Read,Edit,Write,Glob,Grep", "--disallowedTools", "Bash"]
        if session_id:
            argv += ["--resume", session_id]
        process = run_json_process(argv, self.root, self.timeout, stdin_text=prompt)
        actual_session = process.metadata.get("session_id")
        if not isinstance(actual_session, str) or not actual_session:
            raise ValueError("Claude JSON envelope has no valid top-level session_id")
        return ClaudeInvocation(MakerResult(**process.payload), actual_session,
                                process.metadata.get("total_cost_usd"), process.metadata.get("num_turns"))


class CodexCLIAdapter:
    def __init__(self, root: Path, timeout: int, schema_dir: Path, executable: str = "codex"):
        self.root, self.timeout, self.schema_dir, self.executable = root, timeout, schema_dir, executable

    def _run(self, prompt: str, schema_name: str) -> dict:
        argv = [*codex_exec_prefix(self.executable, self.root), "--json", "--ephemeral",
                "--output-schema", str(self.schema_dir / schema_name)]
        return run_json_process(argv, self.root, self.timeout, json_lines=True, stdin_text=prompt).payload

    def verify(self, prompt: str) -> VerifierResult:
        return VerifierResult(**self._run(prompt, "verifier.schema.json"))

    def plan(self, prompt: str) -> PlannerResult:
        return PlannerResult(**self._run(prompt, "planner.schema.json"))
