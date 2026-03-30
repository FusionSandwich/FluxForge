"""Nuclide search controller for the modern Qt shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fluxforge.data.nuclide_library import (
    NuclideSearchHit,
    ensure_bundled_nuclide_database,
    reference_lines_for_nuclide,
    search_nuclides,
)
from fluxforge.gui.selection_bus import SelectionBus, SelectionState


@dataclass
class NuclideSearchController:
    """Bridge the SQLite nuclide library into the GUI selection model."""

    selection_bus: SelectionBus
    database_path: str | Path | None = None
    overlay_limit: int = 8

    def __post_init__(self) -> None:
        self.database_path = ensure_bundled_nuclide_database(self.database_path)

    def search(self, query: str, *, limit: int = 12) -> list[NuclideSearchHit]:
        return search_nuclides(self.database_path, query, limit=limit)

    def activate(self, hit: NuclideSearchHit | str) -> SelectionState:
        nuclide = hit.nuclide if isinstance(hit, NuclideSearchHit) else str(hit)
        lines = reference_lines_for_nuclide(
            self.database_path,
            nuclide,
            limit=self.overlay_limit,
        )
        return self.selection_bus.publish_nuclide(
            nuclide,
            reference_lines_keV=lines,
        )


__all__ = ["NuclideSearchController"]
