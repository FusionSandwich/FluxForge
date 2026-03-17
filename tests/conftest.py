"""Pytest configuration for FluxForge tests."""

collect_ignore = [
    "reference",
]

# Defensive guard in case a local developer creates a `tests/reference` tree.
collect_ignore_glob = [
    "reference/**/test_*.py",
    "reference/**/*_test.py",
]
