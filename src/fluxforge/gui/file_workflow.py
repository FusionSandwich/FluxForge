"""Recent-file and drag/drop helpers for the modern GUI shell."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class RecentFilesManager:
    """Small settings-backed MRU list for spectrum and session files."""

    SETTINGS_KEY = "io/recent_files"

    def __init__(self, settings=None, *, limit: int = 10) -> None:
        self.settings = settings
        self.limit = limit

    def files(self) -> tuple[str, ...]:
        if self.settings is None:
            return ()
        value = self.settings.value(self.SETTINGS_KEY, [])
        if isinstance(value, str):
            return (value,) if value else ()
        return tuple(str(item) for item in value or [])

    def record(self, path: str | Path) -> tuple[str, ...]:
        normalized = str(Path(path))
        entries = [item for item in self.files() if item != normalized]
        entries.insert(0, normalized)
        entries = entries[: self.limit]
        if self.settings is not None:
            self.settings.setValue(self.SETTINGS_KEY, entries)
            if hasattr(self.settings, "sync"):
                self.settings.sync()
        return tuple(entries)

    def record_many(self, paths: Iterable[str | Path]) -> tuple[str, ...]:
        entries = list(self.files())
        for path in paths:
            normalized = str(Path(path))
            entries = [item for item in entries if item != normalized]
            entries.insert(0, normalized)
        entries = entries[: self.limit]
        if self.settings is not None:
            self.settings.setValue(self.SETTINGS_KEY, entries)
            if hasattr(self.settings, "sync"):
                self.settings.sync()
        return tuple(entries)


def normalize_dropped_paths(items: Iterable[str | Path]) -> tuple[str, ...]:
    """Normalize dropped file paths and remove duplicates while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        path = str(Path(str(item).strip()))
        if not path or path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return tuple(normalized)


__all__ = ["RecentFilesManager", "normalize_dropped_paths"]
