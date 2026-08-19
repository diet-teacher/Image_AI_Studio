from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ProcessFailure(RuntimeError):
    pass


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
        process = subprocess.Popen(list(argv), cwd=cwd, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False)
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            process.kill(); process.communicate()
        raise ProcessFailure(f"TIMEOUT after {timeout}s") from exc
    except KeyboardInterrupt:
        if process is not None:
            process.kill(); process.communicate()
        raise
    except OSError as exc:
        raise ProcessFailure(f"PROCESS_START_FAILED: {exc}") from exc
    if process.returncode != 0:
        raise ProcessFailure(f"NONZERO_EXIT {process.returncode}: {stderr[-1000:]}")
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
        raise ProcessFailure(f"JSON_PARSE_FAILED: {exc}") from exc
    metadata = {} if json_lines else envelope
    return ProcessResult(payload, stdout, stderr, metadata)
