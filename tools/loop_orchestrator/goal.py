from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED = {"checkpoint_id", "objective", "acceptance_criteria", "allowed_files", "required_tests"}


class GoalError(ValueError):
    pass


def normalize_relative(value: str, *, allow_pattern: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoalError("path must be a non-empty string")
    raw = value.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise GoalError(f"absolute path is forbidden: {value}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise GoalError(f"path traversal or ambiguous path is forbidden: {value}")
    if not allow_pattern and any(char in raw for char in "*?[]"):
        raise GoalError(f"wildcards are forbidden in changed path: {value}")
    return path.as_posix()


def validate_goal(value: Any) -> dict:
    if not isinstance(value, dict):
        raise GoalError("goal must be a JSON object")
    missing = sorted(REQUIRED - value.keys())
    if missing:
        raise GoalError(f"missing goal fields: {', '.join(missing)}")
    if not isinstance(value["checkpoint_id"], str) or not value["checkpoint_id"].strip():
        raise GoalError("checkpoint_id must be a non-empty string")
    if not isinstance(value["objective"], str) or not value["objective"].strip():
        raise GoalError("objective must be a non-empty string")
    for field in ("acceptance_criteria", "allowed_files", "required_tests"):
        if not isinstance(value[field], list) or not all(isinstance(item, str) and item.strip() for item in value[field]):
            raise GoalError(f"{field} must be a list of non-empty strings")
    if not value["allowed_files"]:
        raise GoalError("allowed_files must not be empty")
    result = dict(value)
    result["allowed_files"] = [normalize_relative(item, allow_pattern=True) for item in value["allowed_files"]]
    if "claude_prompt" in result and (not isinstance(result["claude_prompt"], str) or not result["claude_prompt"].strip()):
        raise GoalError("claude_prompt must be a non-empty string when present")
    return result


def validate_executable_goal(value: dict) -> None:
    if value.get("template") is True:
        raise GoalError("template goals cannot be executed")
    if value["objective"].strip().lower().startswith("replace with"):
        raise GoalError("placeholder objective cannot be executed")
    if not value["acceptance_criteria"]:
        raise GoalError("acceptance_criteria must not be empty for execution")
    if not value["required_tests"]:
        raise GoalError("required_tests must not be empty for execution")


def path_allowed(changed: str, allowed: list[str], root: Path | None = None) -> bool:
    normalized = normalize_relative(changed)
    if root is not None:
        resolved_root = root.resolve()
        try:
            (resolved_root / Path(normalized)).resolve(strict=False).relative_to(resolved_root)
        except ValueError as exc:
            raise GoalError(f"resolved path escapes repository: {changed}") from exc
    return any(PurePosixPath(normalized).match(pattern) for pattern in allowed)
