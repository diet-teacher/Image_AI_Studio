from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


OUTPUT_TAIL_LIMIT = 4000
ENVELOPE_TEXT_LIMIT = 2000

TIMEOUT = "TIMEOUT"
PROCESS_TIMEOUT = TIMEOUT  # stable alias; same value, kept for forward-facing naming
PROCESS_START_FAILED = "PROCESS_START_FAILED"
NONZERO_EXIT = "NONZERO_EXIT"
JSON_PARSE_FAILED = "JSON_PARSE_FAILED"
MODEL_BUDGET_EXHAUSTED = "MODEL_BUDGET_EXHAUSTED"
MODEL_MAX_TURNS = "MODEL_MAX_TURNS"
API_CONNECTION_ERROR = "API_CONNECTION_ERROR"
MAKER_SAFETY_VIOLATION = "MAKER_SAFETY_VIOLATION"

_ENVELOPE_SCALAR_KEYS = ("session_id", "total_cost_usd", "num_turns", "terminal_reason",
                         "subtype", "stop_reason", "is_error")
_BUDGET_MARKERS = {"error_max_budget", "budget_exhausted", "error_budget_exceeded", "max_budget_usd_exceeded"}
_MAX_TURNS_MARKERS = {"error_max_turns", "max_turns_exceeded", "max_turns"}
_CONNECTION_MARKERS = {"error_api_connection", "api_connection_error", "error_network",
                       "connection_error", "api_error"}


class ProcessFailure(RuntimeError):
    def __init__(self, kind: str, message: str, *, return_code: int | None = None,
                 stdout_tail: str = "", stderr_tail: str = "", metadata: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.return_code = return_code
        self.stdout_tail = stdout_tail[-OUTPUT_TAIL_LIMIT:]
        self.stderr_tail = stderr_tail[-OUTPUT_TAIL_LIMIT:]
        self.metadata = dict(metadata) if metadata else {}

    def diagnostics(self) -> dict:
        return {"kind": self.kind, "return_code": self.return_code, "stdout_tail": self.stdout_tail,
                "stderr_tail": self.stderr_tail, "metadata": self.metadata}


def _tail(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "")[-OUTPUT_TAIL_LIMIT:]


def _bounded_text(value) -> str:
    return str(value)[:ENVELOPE_TEXT_LIMIT]


def _extract_envelope_metadata(envelope: dict) -> dict:
    """Bounded, secret-free, prompt-free telemetry preserved from a Claude JSON envelope."""
    metadata = {key: envelope[key] for key in _ENVELOPE_SCALAR_KEYS if key in envelope}
    if "errors" in envelope:
        errors = envelope["errors"]
        if isinstance(errors, list):
            metadata["errors"] = [_bounded_text(item) for item in errors[:20]]
        else:
            metadata["errors"] = _bounded_text(errors)
    if isinstance(envelope.get("result"), str):
        metadata["result"] = _bounded_text(envelope["result"])
    return metadata


def _marker_value(item) -> str:
    if isinstance(item, dict):
        item = item.get("code") or item.get("type") or item.get("reason") or ""
    return str(item).strip().lower()


def _envelope_markers(envelope: dict) -> list[str]:
    values = [envelope.get("terminal_reason"), envelope.get("subtype"), envelope.get("stop_reason")]
    errors = envelope.get("errors")
    if isinstance(errors, list):
        values.extend(errors)
    elif errors:
        values.append(errors)
    return [_marker_value(value) for value in values if value]


def _classify_nonzero_envelope(envelope: dict) -> str:
    """Conservative, explicit-field classification; ambiguous or unrecognized envelopes stay NONZERO_EXIT."""
    for marker in _envelope_markers(envelope):
        if marker in _BUDGET_MARKERS:
            return MODEL_BUDGET_EXHAUSTED
        if marker in _MAX_TURNS_MARKERS:
            return MODEL_MAX_TURNS
        if marker in _CONNECTION_MARKERS:
            return API_CONNECTION_ERROR
    return NONZERO_EXIT


def probe_process(argv: Sequence[str], cwd: Path, timeout: int) -> dict:
    """Probe a fixed, non-model argv and return bounded structured diagnostics."""
    try:
        done = subprocess.run(list(argv), cwd=cwd, text=True, encoding="utf-8", errors="replace",
                              capture_output=True, shell=False, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "kind": TIMEOUT, "return_code": None,
                "stdout_tail": _tail(exc.stdout), "stderr_tail": _tail(exc.stderr)}
    except OSError as exc:
        return {"ok": False, "kind": PROCESS_START_FAILED, "return_code": None,
                "stdout_tail": "", "stderr_tail": _tail(str(exc))}
    kind = None if done.returncode == 0 else NONZERO_EXIT
    return {"ok": done.returncode == 0, "kind": kind, "return_code": done.returncode,
            "stdout_tail": _tail(done.stdout), "stderr_tail": _tail(done.stderr)}


@dataclass
class ProcessResult:
    payload: dict
    stdout: str
    stderr: str
    metadata: dict


def run_json_process(argv: Sequence[str], cwd: Path, timeout: int, *, json_lines: bool = False,
                     stdin_text: str | None = None) -> ProcessResult:
    """Run a fixed adapter-built argv; never interprets model-produced commands.

    The prompt (if any) is supplied only via stdin_text over stdin, never argv, so it never
    appears in the child's command line or in any ProcessFailure diagnostics derived from argv.
    """
    process = None
    try:
        process = subprocess.Popen(list(argv), cwd=cwd, text=True, encoding="utf-8", errors="replace",
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=False)
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            process.kill(); process.communicate()
        raise ProcessFailure(TIMEOUT, f"TIMEOUT after {timeout}s",
                             stdout_tail=_tail(exc.stdout), stderr_tail=_tail(exc.stderr)) from exc
    except KeyboardInterrupt:
        if process is not None:
            process.kill(); process.communicate()
        raise
    except OSError as exc:
        raise ProcessFailure(PROCESS_START_FAILED, f"PROCESS_START_FAILED: {exc}",
                             stderr_tail=str(exc)) from exc
    if process.returncode != 0:
        stdout_tail = stdout[-OUTPUT_TAIL_LIMIT:]
        stderr_tail = stderr[-OUTPUT_TAIL_LIMIT:]
        kind, metadata = NONZERO_EXIT, {}
        if not json_lines:
            try:
                envelope = json.loads(stdout)
            except json.JSONDecodeError:
                envelope = None
            if isinstance(envelope, dict):
                metadata = _extract_envelope_metadata(envelope)
                kind = _classify_nonzero_envelope(envelope)
        raise ProcessFailure(kind,
            f"{kind} {process.returncode}: stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}",
            return_code=process.returncode, stdout_tail=stdout_tail, stderr_tail=stderr_tail, metadata=metadata)
    try:
        if json_lines:
            events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            payload = next((e.get("result") for e in reversed(events) if isinstance(e.get("result"), dict)), None)
            if payload is None:
                texts = []
                for event in events:
                    item = event.get("item", {})
                    if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                        texts.append(item.get("text"))
                    elif event.get("type") == "message":
                        texts.append(event.get("text") or event.get("content"))
                payload = texts[-1] if texts else None
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise ValueError("no structured final result event")
        else:
            envelope = json.loads(stdout)
            payload = envelope.get("structured_output", envelope.get("result", envelope))
            if isinstance(payload, str):
                payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("result is not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProcessFailure(JSON_PARSE_FAILED, f"JSON_PARSE_FAILED: {exc}",
                             stdout_tail=stdout, stderr_tail=stderr) from exc
    metadata = {} if json_lines else envelope
    return ProcessResult(payload, stdout, stderr, metadata)
