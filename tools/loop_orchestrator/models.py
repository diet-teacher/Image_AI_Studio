from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class State(str, Enum):
    IDLE = "IDLE"
    BUDGET_CHECK = "BUDGET_CHECK"
    MAKER_RUNNING = "MAKER_RUNNING"
    VERIFYING = "VERIFYING"
    REWORK = "REWORK"
    PLANNING = "PLANNING"
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    BUDGET_STOP = "BUDGET_STOP"
    FAILED = "FAILED"


@dataclass
class MakerResult:
    status: str
    base_commit: str
    changed_files: list[str]
    tests_run: list[str]
    known_risks: list[str]
    summary: str


@dataclass
class ClaudeInvocation:
    result: MakerResult
    session_id: str
    total_cost_usd: float | None = None
    num_turns: int | None = None


@dataclass
class VerifierResult:
    verdict: str
    findings: list[str]
    failed_checks: list[str]
    tests_observed: list[str]
    visual_verification: str | None
    residual_risks: list[str]
    recommended_action: str


@dataclass
class PlannerResult:
    checkpoint_id: str
    objective: str
    acceptance_criteria: list[str]
    allowed_files: list[str]
    required_tests: list[str]
    claude_prompt: str
    estimated_scope: str


@dataclass
class RunState:
    run_id: str
    checkpoint_id: str
    state: State = State.IDLE
    base_commit: str = ""
    iteration: int = 0
    rework_rounds: int = 0
    stop_reason: str | None = None
    recent_result: dict[str, Any] | None = None
    transitions: list[str] = field(default_factory=lambda: [State.IDLE.value])

    def move(self, state: State, reason: str | None = None) -> None:
        self.state = state
        self.transitions.append(state.value)
        if reason:
            self.stop_reason = reason

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result
