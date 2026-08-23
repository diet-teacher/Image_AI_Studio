from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


OUTPUT_TAIL_LIMIT = 4000
TIMEOUT = "TIMEOUT"
PROCESS_START_FAILED = "PROCESS_START_FAILED"
NONZERO_EXIT = "NONZERO_EXIT"
JSON_PARSE_FAILED = "JSON_PARSE_FAILED"


class ProcessFailure(RuntimeError):
    def __init__(self, kind: str, message: str, *, return_code: int | None = None,
                 stdout_tail: str = "", stderr_tail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.return_code = return_code
        self.stdout_tail = stdout_tail[-OUTPUT_TAIL_LIMIT:]
        self.stderr_tail = stderr_tail[-OUTPUT_TAIL_LIMIT:]

    def diagnostics(self) -> dict:
        return {"kind": self.kind, "return_code": self.return_code, "stdout_tail": self.stdout_tail,
                "stderr_tail": self.stderr_tail}


def _tail(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "")[-OUTPUT_TAIL_LIMIT:]


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


def run_json_process(argv: Sequence[str], cwd: Path, timeout: int, *, json_lines: bool = False) -> ProcessResult:
    """Run a fixed adapter-built argv; never interprets model-produced commands."""
    process = None
    try:
        process = subprocess.Popen(list(argv), cwd=cwd, text=True, encoding="utf-8", errors="replace",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout, stderr = process.communicate(timeout=timeout)
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
        raise ProcessFailure(NONZERO_EXIT,
            f"NONZERO_EXIT {process.returncode}: stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}",
            return_code=process.returncode, stdout_tail=stdout_tail, stderr_tail=stderr_tail)
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
