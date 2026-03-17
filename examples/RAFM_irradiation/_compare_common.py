"""Shared helpers for RAFM comparison scripts.

These helpers intentionally do not touch analysis behavior. They only unify
bootstrap/path and file-pairing logic that was duplicated across scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple


def ensure_fluxforge_src(caller_file: str) -> Path:
    """Ensure FluxForge `src/` is available on `sys.path` and return repo root."""
    root = Path(caller_file).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def pair_sample_files(
    raw_dir: Path,
    qg_dir: Path,
    *,
    raw_pattern: str = "*.ASC",
    qg_pattern: str = "*.txt",
) -> List[Tuple[str, Path, Path]]:
    """Return paired (stem, raw_path, qg_path) files present in both trees."""
    raw_files = {path.stem: path for path in raw_dir.glob(raw_pattern)}
    qg_files = {path.stem: path for path in qg_dir.glob(qg_pattern)}
    common = sorted(set(raw_files) & set(qg_files))
    return [(stem, raw_files[stem], qg_files[stem]) for stem in common]
