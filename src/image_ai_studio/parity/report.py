"""Append-only JSON log of test-matrix results + a markdown renderer.

Every row must use one of the statuses defined in STATUSES. SKIPPED /
UNSUPPORTED / BLOCKED / INVALID_BUILD_CONFIGURATION are never treated as
PASS by the renderer.
"""
from __future__ import annotations

import json
from pathlib import Path

STATUSES = {
    "PASS": "Executed and met the tolerance / success criteria.",
    "FAIL": "Executed but did not meet the success criteria (e.g. parity out of tolerance, crash).",
    "SKIPPED": "Not executed because a required resource is absent on this machine (e.g. no CUDA GPU).",
    "UNSUPPORTED": "The backend/API does not support this configuration on the current official distribution.",
    "BLOCKED": "Could not be executed due to a missing prerequisite that was not independently retried.",
    "INVALID_BUILD_CONFIGURATION": "Debug/Release or platform mismatch; not a backend failure.",
}


def append_result(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if log_path.exists():
        records = json.loads(log_path.read_text())
    records.append(record)
    log_path.write_text(json.dumps(records, indent=2))


def render_markdown_table(log_path: Path) -> str:
    if not log_path.exists():
        return "_No results recorded yet._"
    records = json.loads(log_path.read_text())
    if not records:
        return "_No results recorded yet._"

    columns = list(records[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in records:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)
