from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .goal import GoalError, normalize_relative, validate_executable_goal, validate_goal


PHASE_REQUIRED = {
    "phase_id", "objective", "checkpoints", "allowed_files", "allowed_tests",
    "final_harness_profile", "max_checkpoints", "max_rework_rounds",
    "max_model_calls", "max_claude_cost_usd", "max_elapsed_seconds",
    "completion_conditions",
}


class PhaseManifestError(ValueError):
    pass


def _positive(value: Any, name: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)) or value <= 0:
        kind = "integer" if integer else "number"
        raise PhaseManifestError(f"{name} must be a positive {kind}")
    return int(value) if integer else float(value)


def _inside_repository(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    normalized = normalize_relative(relative)
    path = root / normalized
    try:
        resolved = path.resolve(strict=must_exist)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise PhaseManifestError(f"path escapes repository: {relative}") from exc
    return path


def _validate_scope_path(root: Path, relative: str) -> None:
    normalized = normalize_relative(relative, allow_pattern=True)
    fixed_parts = []
    for part in Path(normalized).parts:
        if any(char in part for char in "*?[]"):
            break
        fixed_parts.append(part)
    candidate = root.joinpath(*fixed_parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise PhaseManifestError(f"scope path escapes repository: {relative}") from exc


def manifest_digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_phase_manifest(path: Path, root: Path, config: dict, harness_profiles: set[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseManifestError(f"invalid phase manifest JSON: {exc}") from exc
    return validate_phase_manifest(value, root, config, harness_profiles)


def resolve_phase_manifest_path(value: Path, root: Path) -> Path:
    path = _inside_repository(root, str(value), must_exist=True)
    if path.is_symlink() or not path.is_file():
        raise PhaseManifestError("phase manifest must be a regular non-symlink file")
    return path


def validate_phase_manifest(value: Any, root: Path, config: dict, harness_profiles: set[str]) -> dict:
    if not isinstance(value, dict):
        raise PhaseManifestError("phase manifest must be a JSON object")
    missing = sorted(PHASE_REQUIRED - value.keys())
    if missing:
        raise PhaseManifestError("missing phase fields: " + ", ".join(missing))
    for name in ("phase_id", "objective", "final_harness_profile"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise PhaseManifestError(f"{name} must be a non-empty string")
    if value["objective"].strip().lower().startswith("replace with"):
        raise PhaseManifestError("placeholder phase objective is forbidden")
    if value["final_harness_profile"] not in harness_profiles:
        raise PhaseManifestError("unknown final_harness_profile")
    for name in ("allowed_files", "allowed_tests", "completion_conditions", "checkpoints"):
        if not isinstance(value[name], list) or not value[name]:
            raise PhaseManifestError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value["allowed_tests"] + value["completion_conditions"]):
        raise PhaseManifestError("allowed_tests and completion_conditions require non-empty strings")

    allowed_files = [normalize_relative(item, allow_pattern=True) for item in value["allowed_files"]]
    for relative in allowed_files:
        _validate_scope_path(root, relative)
    allowed_tests = list(value["allowed_tests"])
    configured_tests = set()
    for item in config.get("allowed_tests", []):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise PhaseManifestError("configured test allowlist entry is invalid")
        argv = item.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
            raise PhaseManifestError(f"configured test argv is invalid: {item['name']}")
        if item["name"] in configured_tests:
            raise PhaseManifestError(f"duplicate configured test: {item['name']}")
        executable = os.path.basename(argv[0]).lower()
        if executable in {"bash", "bash.exe", "sh", "sh.exe", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
            raise PhaseManifestError(f"shell executable is forbidden in test allowlist: {item['name']}")
        if any(any(char in arg for char in "*?") for arg in argv):
            raise PhaseManifestError(f"wildcard argv is forbidden in test allowlist: {item['name']}")
        configured_tests.add(item["name"])
    unknown_tests = sorted(set(allowed_tests) - configured_tests)
    if unknown_tests:
        raise PhaseManifestError("tests are not configured in fixed argv allowlist: " + ", ".join(unknown_tests))

    limits = {
        "max_checkpoints": _positive(value["max_checkpoints"], "max_checkpoints", integer=True),
        "max_rework_rounds": _positive(value["max_rework_rounds"], "max_rework_rounds", integer=True),
        "max_model_calls": _positive(value["max_model_calls"], "max_model_calls", integer=True),
        "max_claude_cost_usd": _positive(value["max_claude_cost_usd"], "max_claude_cost_usd"),
        "max_elapsed_seconds": _positive(value["max_elapsed_seconds"], "max_elapsed_seconds"),
    }
    if len(value["checkpoints"]) > limits["max_checkpoints"]:
        raise PhaseManifestError("checkpoint count exceeds max_checkpoints")

    checkpoints, ids = [], set()
    for index, item in enumerate(value["checkpoints"]):
        if not isinstance(item, dict) or set(item) != {"checkpoint_id", "goal"}:
            raise PhaseManifestError(f"checkpoint {index} must contain only checkpoint_id and goal")
        checkpoint_id = item["checkpoint_id"]
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip() or checkpoint_id in ids:
            raise PhaseManifestError(f"invalid or duplicate checkpoint_id: {checkpoint_id!r}")
        ids.add(checkpoint_id)
        goal_relative = normalize_relative(item["goal"])
        goal_path = _inside_repository(root, goal_relative, must_exist=True)
        if goal_path.is_symlink():
            raise PhaseManifestError(f"symlink goal is forbidden: {goal_relative}")
        try:
            goal = validate_goal(json.loads(goal_path.read_text(encoding="utf-8")))
            validate_executable_goal(goal)
        except (OSError, json.JSONDecodeError, GoalError) as exc:
            raise PhaseManifestError(f"invalid executable goal {goal_relative}: {exc}") from exc
        if goal["checkpoint_id"] != checkpoint_id:
            raise PhaseManifestError(f"checkpoint ID does not match goal: {checkpoint_id}")
        if not set(goal["allowed_files"]).issubset(set(allowed_files)):
            raise PhaseManifestError(f"goal allowed_files exceed phase scope: {checkpoint_id}")
        if not set(goal["required_tests"]).issubset(set(allowed_tests)):
            raise PhaseManifestError(f"goal required_tests exceed phase scope: {checkpoint_id}")
        if not set(goal["required_tests"]).issubset(configured_tests):
            raise PhaseManifestError(f"goal required test is not allowlisted: {checkpoint_id}")
        for relative in goal["allowed_files"]:
            _validate_scope_path(root, relative)
        checkpoints.append({"checkpoint_id": checkpoint_id, "goal": goal_relative, "goal_value": goal})

    result = dict(value)
    result.update(limits)
    result["allowed_files"] = allowed_files
    result["checkpoints"] = checkpoints
    result["manifest_digest"] = manifest_digest(value)
    return result
