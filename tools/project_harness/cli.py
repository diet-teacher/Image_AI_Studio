"""Command-line entry point for the deterministic project harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .profiles import PROFILES
from .runner import doctor, dry_run, execute_profile


def _root(value: str) -> Path:
    root = Path(value).resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "tests").is_dir():
        raise argparse.ArgumentTypeError(f"not an Image AI Studio repository: {root}")
    return root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Image AI Studio deterministic quality harness")
    result.add_argument("--root", type=_root, default=Path.cwd().resolve())
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list fixed validation profiles")
    commands.add_parser("doctor", help="check the local validation environment")
    run = commands.add_parser("run", help="show or execute a fixed profile")
    run.add_argument("--profile", choices=sorted(PROFILES), required=True)
    run.add_argument("--execute", action="store_true", help="actually start the profile processes")
    run.add_argument("--require-clean", action="store_true", help="block execution when Git reports changes")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.root if isinstance(args.root, Path) else Path(args.root)
    if args.command == "list":
        payload = {
            name: {
                "description": profile.description,
                "steps": [step.name for step in profile.steps],
            }
            for name, profile in PROFILES.items()
        }
        exit_code = 0
    elif args.command == "doctor":
        payload = doctor(root)
        exit_code = 0 if payload["healthy"] else 2
    else:
        profile = PROFILES[args.profile]
        if args.execute:
            exit_code, payload = execute_profile(root, profile, require_clean=args.require_clean)
        else:
            payload = dry_run(root, profile, require_clean=args.require_clean)
            exit_code = 0
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return exit_code
