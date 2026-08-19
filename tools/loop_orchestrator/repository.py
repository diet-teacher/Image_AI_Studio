from __future__ import annotations

import subprocess
import hashlib
import os
from pathlib import Path


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, shell=False, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def base_commit(root: Path) -> str:
    return git(root, "rev-parse", "HEAD").strip()


def worktree_snapshot(root: Path) -> str:
    return git(root, "status", "--porcelain=v1", "--untracked-files=all") + "\n" + git(root, "diff", "--binary", "--no-ext-diff")


def changed_files(root: Path) -> list[str]:
    tracked = git(root, "diff", "--name-only", "HEAD", "--").splitlines()
    untracked = git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(tracked + untracked))


def file_snapshot(root: Path) -> dict[str, str]:
    snapshot = {}
    for relative in changed_files(root):
        path = root / relative
        if path.is_symlink(): snapshot[relative] = "SYMLINK:" + hashlib.sha256(os.readlink(path).encode()).hexdigest()
        elif path.is_file(): snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.exists(): snapshot[relative] = "NON_FILE"
        else: snapshot[relative] = "DELETED"
    return snapshot
