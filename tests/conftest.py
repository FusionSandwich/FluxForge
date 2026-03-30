"""Pytest configuration for FluxForge tests."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

collect_ignore = [
    "reference",
]

# Defensive guard in case a local developer creates a `tests/reference` tree.
collect_ignore_glob = [
    "reference/**/test_*.py",
    "reference/**/*_test.py",
]
