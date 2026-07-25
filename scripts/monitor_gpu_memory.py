#!/usr/bin/env python
"""Poll nvidia-smi at a fixed interval and append rows to
results/gpu_memory_log.csv. On a machine without an NVIDIA GPU (e.g.
this Mac), it writes a single NOT_APPLICABLE row and exits -- it does
not fabricate readings.

Usage: python scripts/monitor_gpu_memory.py --runner run_aoti --model tiny_cnn
                                             --backend aoti --duration-sec 20 --interval-sec 0.5
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "results" / "gpu_memory_log.csv"
COLUMNS = [
    "timestamp", "runner", "process_id", "model", "backend",
    "gpu_index", "gpu_name", "used_memory_mb", "total_memory_mb",
]


def poll_once(runner: str, pid: str, model: str, backend: str) -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    rows = []
    for line in out.stdout.strip().splitlines():
        idx, name, used, total = [p.strip() for p in line.split(",")]
        rows.append({
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "runner": runner,
            "process_id": pid,
            "model": model,
            "backend": backend,
            "gpu_index": idx,
            "gpu_name": name,
            "used_memory_mb": used,
            "total_memory_mb": total,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default="n/a")
    parser.add_argument("--model", default="n/a")
    parser.add_argument("--backend", default="n/a")
    parser.add_argument("--pid", default="n/a")
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--interval-sec", type=float, default=0.5)
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()

    if not shutil.which("nvidia-smi"):
        with LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "runner": args.runner,
                "process_id": args.pid,
                "model": args.model,
                "backend": args.backend,
                "gpu_index": "N/A",
                "gpu_name": "N/A",
                "used_memory_mb": "NOT_APPLICABLE",
                "total_memory_mb": "NOT_APPLICABLE",
            })
        print("nvidia-smi not found on this machine -- wrote NOT_APPLICABLE row.", file=sys.stderr)
        return

    end_time = time.time() + args.duration_sec
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        while time.time() < end_time:
            for row in poll_once(args.runner, args.pid, args.model, args.backend):
                writer.writerow(row)
            f.flush()
            time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
