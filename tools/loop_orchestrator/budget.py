from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    code: str
    usage_percent: float | None


class BudgetManager:
    def __init__(self, path: Path, soft_stop: float = 75.0, hard_stop: float = 80.0,
                 validity_hours: float = 24.0, now=None):
        self.path = path
        self.soft_stop = soft_stop
        self.hard_stop = hard_stop
        self.validity_hours = validity_hours
        self.now = now or (lambda: datetime.now(timezone.utc))

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def check(self, role: str, *, starting_checkpoint: bool = False) -> BudgetDecision:
        entry = self.read().get(role, {})
        value, updated_at, source = entry.get("period_usage_percent"), entry.get("updated_at"), entry.get("source")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
            return BudgetDecision(False, "BUDGET_UNKNOWN", None)
        if not isinstance(source, str) or not source.strip() or not isinstance(updated_at, str):
            return BudgetDecision(False, "BUDGET_UNKNOWN", None)
        try:
            stamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("timezone required")
            stamp = stamp.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return BudgetDecision(False, "BUDGET_UNKNOWN", None)
        now = self.now()
        if stamp > now or now - stamp > timedelta(hours=self.validity_hours):
            return BudgetDecision(False, "BUDGET_UNKNOWN", None)
        usage = float(value)
        if usage >= self.hard_stop:
            return BudgetDecision(False, "HARD_STOP", usage)
        if usage >= self.soft_stop and starting_checkpoint:
            return BudgetDecision(False, "SOFT_STOP", usage)
        return BudgetDecision(True, "OK", usage)
