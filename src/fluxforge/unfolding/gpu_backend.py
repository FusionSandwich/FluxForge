"""Optional GPU backend selection for Module 3 workflows."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np


@dataclass(frozen=True)
class ArrayBackendInfo:
    """Selected numerical backend for batch or ML workflows."""

    name: str
    device: str
    available: bool


def resolve_array_backend(*, prefer_gpu: bool = False):
    """Return the selected array backend and the imported module."""

    if prefer_gpu:
        try:
            cupy = importlib.import_module("cupy")
            return ArrayBackendInfo(name="cupy", device="gpu", available=True), cupy
        except Exception:
            pass
    return ArrayBackendInfo(name="numpy", device="cpu", available=True), np


__all__ = ["ArrayBackendInfo", "resolve_array_backend"]
