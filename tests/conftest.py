"""Make the src/ layout importable even without `pip install -e .` first,
matching the sys.path handling already used by scripts/*.py."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
