#!/usr/bin/env python
"""Thin cross-platform entry point -- see
src/image_ai_studio/tools/inspect_environment.py for the implementation.
Kept in scripts/ so `python scripts/inspect_environment.py` matches the
naming other Phase 0 docs use, without duplicating logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_ai_studio.tools.inspect_environment import main

if __name__ == "__main__":
    main()
