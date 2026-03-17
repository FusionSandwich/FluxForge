"""Runtime environment helpers for offline-safe FluxForge execution."""

from __future__ import annotations

import os
from urllib.parse import urlparse


_TRUE_VALUES = {"1", "true", "yes", "on"}
_REMOTE_SCHEMES = {"http", "https"}


def offline_mode_enabled() -> bool:
    """Return True when FluxForge should avoid all runtime network access."""

    raw = os.getenv("FLUXFORGE_OFFLINE", "")
    return raw.strip().lower() in _TRUE_VALUES


def is_remote_locator(locator: str) -> bool:
    """Return True when the locator points to a remote HTTP(S) resource."""

    return urlparse(str(locator).strip()).scheme.lower() in _REMOTE_SCHEMES


def require_network_access(feature: str, locator: str | None = None) -> None:
    """Raise a clear error when offline mode blocks a network-backed feature."""

    if not offline_mode_enabled():
        return

    detail = f" ({locator})" if locator else ""
    raise RuntimeError(
        f"{feature} is unavailable because FLUXFORGE_OFFLINE=1 blocks network access{detail}. "
        "Use bundled data, local files, sqlite:/// sources, or python:// plugins instead."
    )


__all__ = [
    "is_remote_locator",
    "offline_mode_enabled",
    "require_network_access",
]
