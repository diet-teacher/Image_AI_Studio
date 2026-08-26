from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import ClaudeCLIAdapter, CodexCLIAdapter
from .adapters import codex_exec_prefix
from .budget import BudgetManager
from .engine import LoopEngine
from .phase import PhaseManifestError, load_phase_manifest, resolve_phase_manifest_path
from .phase_engine import PhaseEngine
from .process import probe_process
from tools.project_harness.profiles import PROFILES


ROOT = Path.cwd()
RUNTIME = ROOT / ".loop"
PACKAGE = Path(__file__).parent


def load_config() -> dict:
    path = RUNTIME / "config.json"
    if not path.exists():
        return json.loads((PACKAGE / "example.config.json").read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def init() -> int:
    RUNTIME.mkdir(exist_ok=True)
    for name, source in [("config.json", "example.config.json"), ("budget.json", "example.budget.json")]:
        target = RUNTIME / name
        if not target.exists():
            target.write_text((PACKAGE / source).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Initialized {RUNTIME} (existing files preserved)")
    return 0


def doctor() -> int:
    config = load_config()
    def probe(argv):
        result = probe_process(argv, ROOT, 10)
        result["exit_code"] = result["return_code"]
        result["output"] = result["stdout_tail"].strip()[-2000:]
        result["stdout_tail"] = result["stdout_tail"][-2000:]
        result["stderr_tail"] = result["stderr_tail"][-2000:]
        return result
    git_ok = (ROOT / ".git").exists()
    git_status = probe(["git", "status", "--porcelain=v1", "--untracked-files=all"]) if git_ok else {"ok": False, "error": "not a Git repository"}
    dirty = bool(git_status.get("stdout_tail", "").strip())
    python_ok = sys.version_info >= (3, 10)
    claude = str(config.get("claude_executable", "claude")); codex = str(config.get("codex_executable", "codex"))
    checks = {"claude_version": probe([claude, "--version"]), "codex_version": probe([codex, "--version"]),
              "codex_exec_help": probe([codex, "exec", "--help"])}
    healthy = python_ok and git_ok and git_status.get("ok", False) and all(item["ok"] for item in checks.values())
    report = {"healthy": healthy, "python": {"version": sys.version.split()[0], "supported": python_ok},
              "git_repository": git_ok, "worktree_dirty": dirty, "worktree_status": git_status,
              "executables": {"claude": claude, "codex": codex}, "checks": checks,
              "selected_integrations": {"maker": "claude -p JSON", "verifier_planner": "codex exec --json read-only"}}
    print(json.dumps(report, indent=2)); return 0 if healthy else 1


def status() -> int:
    path = RUNTIME / "state.json"
    print(path.read_text(encoding="utf-8") if path.exists() else "No run state. Use init first.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.loop_orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor"); sub.add_parser("init"); sub.add_parser("status")
    run = sub.add_parser("run"); run.add_argument("--goal", required=True, type=Path); run.add_argument("--max-checkpoints", type=int, default=1)
    mode = run.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--execute", action="store_true")
    phase = sub.add_parser("run-phase")
    phase.add_argument("--manifest", required=True, type=Path)
    phase_mode = phase.add_mutually_exclusive_group()
    phase_mode.add_argument("--execute", action="store_true")
    phase_mode.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "doctor": return doctor()
    if args.command == "init": return init()
    if args.command == "status": return status()
    if args.command == "run-phase":
        config = load_config()
        try:
            manifest_path = resolve_phase_manifest_path(args.manifest, ROOT)
            manifest = load_phase_manifest(manifest_path, ROOT, config, set(PROFILES))
        except PhaseManifestError as exc:
            parser.error(str(exc))
        if not args.execute and not args.resume:
            state = PhaseEngine(ROOT, config, None, None, None).run(manifest)
            print(json.dumps(state, indent=2))
            return 0
        budget = BudgetManager(RUNTIME / "budget.json", config["soft_stop_percent"],
                               config["hard_stop_percent"], config["budget_validity_hours"])
        maker = ClaudeCLIAdapter(ROOT, config["process_timeout_seconds"], config["claude_max_budget_usd"],
                                 config.get("claude_model"))
        maker.executable = config.get("claude_executable", "claude")
        codex = CodexCLIAdapter(ROOT, config["process_timeout_seconds"], PACKAGE / "schemas",
                                config.get("codex_executable", "codex"))
        def preflight():
            timeout = min(int(config.get("process_timeout_seconds", 1800)), 10)
            probes = [
                ("claude", [str(config.get("claude_executable", "claude")), "--version"]),
                ("codex", [str(config.get("codex_executable", "codex")), "--version"]),
                ("codex_exec", [*codex_exec_prefix(str(config.get("codex_executable", "codex")), ROOT), "--help"]),
            ]
            for integration, probe_argv in probes:
                result = probe_process(probe_argv, ROOT, timeout)
                if not result["ok"]:
                    return {"integration": integration, "argv": probe_argv, **result}
            return None
        state = PhaseEngine(ROOT, config, maker, codex, budget, preflight=preflight).run(
            manifest, execute=True, resume=args.resume)
        print(json.dumps(state, indent=2))
        return 0 if state["state"] == "READY_TO_COMMIT" else 2
    if args.max_checkpoints < 1: parser.error("--max-checkpoints must be at least 1")
    if not args.goal.is_file(): parser.error(f"goal file not found: {args.goal}")
    try: goal = json.loads(args.goal.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: parser.error(f"invalid goal JSON: {exc}")
    config = load_config(); budget = BudgetManager(RUNTIME / "budget.json", config["soft_stop_percent"], config["hard_stop_percent"], config["budget_validity_hours"])
    maker = ClaudeCLIAdapter(ROOT, config["process_timeout_seconds"], config["claude_max_budget_usd"],
                             config.get("claude_model"))
    maker.executable = config.get("claude_executable", "claude")
    codex = CodexCLIAdapter(ROOT, config["process_timeout_seconds"], PACKAGE / "schemas", config.get("codex_executable", "codex"))
    state = LoopEngine(ROOT, config, maker, codex, budget).run(goal, args.max_checkpoints, execute=args.execute)
    print(json.dumps(state.to_dict(), indent=2)); return 0 if state.state.value == "ACCEPTED" else 2
