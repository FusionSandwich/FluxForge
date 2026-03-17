"""TensorFlow CUDA runtime helpers.

Ensures TensorFlow can discover NVIDIA shared libraries installed via pip
(`nvidia-*` wheels) before importing `tensorflow`.
"""

from __future__ import annotations

import os
import site
import sysconfig
from pathlib import Path
from typing import Iterable, Tuple


def _site_package_roots() -> Iterable[Path]:
    seen: set[Path] = set()
    for getter in (site.getsitepackages,):
        try:
            for raw in getter():
                path = Path(raw).resolve()
                if path not in seen:
                    seen.add(path)
                    yield path
        except Exception:
            continue

    try:
        user_site = Path(site.getusersitepackages()).resolve()
        if user_site not in seen:
            seen.add(user_site)
            yield user_site
    except Exception:
        pass

    for key in ("purelib", "platlib"):
        raw = sysconfig.get_paths().get(key)
        if not raw:
            continue
        path = Path(raw).resolve()
        if path not in seen:
            seen.add(path)
            yield path


def find_tensorflow_cuda_library_dirs() -> Tuple[str, ...]:
    """Return pip-installed NVIDIA library directories usable by TensorFlow."""

    lib_dirs: list[str] = []
    seen: set[str] = set()
    for root in _site_package_roots():
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib in nvidia_root.glob("*/lib/*.so*"):
            lib_dir = str(lib.parent.resolve())
            if lib_dir not in seen:
                seen.add(lib_dir)
                lib_dirs.append(lib_dir)
    return tuple(sorted(lib_dirs))


def configure_tensorflow_cuda_runtime() -> Tuple[str, ...]:
    """Prepend pip-installed NVIDIA library directories to `LD_LIBRARY_PATH`."""

    lib_dirs = find_tensorflow_cuda_library_dirs()
    if not lib_dirs:
        return ()

    existing = [
        entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry
    ]
    combined: list[str] = []
    for entry in (*lib_dirs, *existing):
        if entry and entry not in combined:
            combined.append(entry)
    os.environ["LD_LIBRARY_PATH"] = ":".join(combined)
    return lib_dirs


__all__ = [
    "configure_tensorflow_cuda_runtime",
    "find_tensorflow_cuda_library_dirs",
]
