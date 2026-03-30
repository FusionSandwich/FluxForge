"""Nuclide search controller for the modern Qt shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fluxforge.data.gamma_database import GammaDatabase
from fluxforge.data.nuclear_data_sources import (
    get_nuclear_data_source,
    load_gamma_identification_source,
)
from fluxforge.data.nuclide_library import (
    NuclideSearchHit,
    ensure_bundled_nuclide_database,
    reference_lines_for_nuclide,
    search_gamma_lines_by_energy,
    search_nuclides,
)
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.selection_bus import SelectionBus, SelectionState


@dataclass(frozen=True)
class NuclideSearchResult:
    """Search result displayed in the Qt shell."""

    nuclide: str
    display_name: str
    strongest_lines_keV: tuple[float, ...] = ()
    source_id: str = "fluxforge_bundled_gamma"


@dataclass(frozen=True)
class GammaLineMatchResult:
    """One isotope-line candidate near a selected or typed centroid."""

    nuclide: str
    display_name: str
    line_energy_keV: float
    delta_keV: float
    intensity: float
    half_life_s: float
    source_id: str = "fluxforge_bundled_gamma"


@dataclass
class NuclideSearchController:
    """Bridge the SQLite nuclide library into the GUI selection model."""

    selection_bus: SelectionBus
    database_path: str | Path | None = None
    overlay_limit: int = 8
    library_manager: DataLibraryManager | None = None

    def __post_init__(self) -> None:
        self.database_path = ensure_bundled_nuclide_database(self.database_path)
        self._gamma_database: GammaDatabase | None = None
        self._source_id = "fluxforge_bundled_gamma"
        self._custom_path: str | None = None
        if self.library_manager is not None:
            self.library_manager.subscribe(self._on_library_state_changed)
            self._on_library_state_changed(self.library_manager.state)

    @property
    def source_id(self) -> str:
        return self._source_id

    def _on_library_state_changed(self, state) -> None:
        self.set_source(
            state.gamma_identification_source_id,
            custom_path=state.custom_gamma_path,
        )

    def set_source(self, source_id: str, *, custom_path: str | None = None) -> None:
        self._source_id = source_id
        self._custom_path = custom_path
        self._gamma_database = None
        if source_id != "fluxforge_bundled_gamma":
            if source_id == "custom_gamma_file" and not custom_path:
                self._gamma_database = GammaDatabase()
                return
            self._gamma_database = load_gamma_identification_source(
                source_id,
                custom_path=custom_path,
            )

    def source_label(self) -> str:
        return get_nuclear_data_source(
            self._source_id,
            custom_paths=[self._custom_path] if self._custom_path else (),
        ).label

    def search(self, query: str, *, limit: int = 12) -> list[NuclideSearchResult]:
        if self._source_id == "fluxforge_bundled_gamma":
            return [
                NuclideSearchResult(
                    nuclide=hit.nuclide,
                    display_name=hit.display_name,
                    strongest_lines_keV=tuple(hit.strongest_lines_keV),
                    source_id=self._source_id,
                )
                for hit in search_nuclides(self.database_path, query, limit=limit)
            ]

        database = self._gamma_database or GammaDatabase()
        token = self._normalize_query(query or "")
        matches: list[NuclideSearchResult] = []
        for nuclide in database.nuclides_with_gamma():
            normalized = self._normalize_query(nuclide)
            if token and token not in normalized:
                continue
            decay = database.get(nuclide)
            if decay is None:
                continue
            matches.append(
                NuclideSearchResult(
                    nuclide=nuclide,
                    display_name=nuclide,
                    strongest_lines_keV=tuple(
                        round(line.energy_keV, 3)
                        for line in decay.strongest_gamma_lines(n=4)
                    ),
                    source_id=self._source_id,
                )
            )
        matches.sort(
            key=lambda item: (
                not self._normalize_query(item.display_name).startswith(token),
                self._normalize_query(item.display_name),
            )
        )
        return matches[:limit]

    def reference_lines_for_nuclide(
        self,
        nuclide: str,
        *,
        limit: int | None = None,
    ) -> tuple[float, ...]:
        resolved_limit = limit if limit is not None else self.overlay_limit
        if self._source_id == "fluxforge_bundled_gamma":
            return reference_lines_for_nuclide(
                self.database_path,
                nuclide,
                limit=resolved_limit,
            )
        database = self._gamma_database or GammaDatabase()
        decay = database.get(nuclide)
        if decay is None:
            return ()
        return tuple(
            round(line.energy_keV, 3)
            for line in decay.strongest_gamma_lines(n=resolved_limit)
        )

    def activate(
        self,
        hit: NuclideSearchResult | NuclideSearchHit | str,
    ) -> SelectionState:
        nuclide = hit.nuclide if hasattr(hit, "nuclide") else str(hit)
        lines = self.reference_lines_for_nuclide(nuclide, limit=self.overlay_limit)
        return self.selection_bus.publish_nuclide(
            nuclide,
            reference_lines_keV=lines,
        )

    def line_matches_for_energy(
        self,
        energy_keV: float,
        *,
        tolerance_keV: float = 2.0,
        query: str = "",
        limit: int = 48,
        min_intensity: float = 0.0,
    ) -> list[GammaLineMatchResult]:
        if self._source_id == "fluxforge_bundled_gamma":
            return [
                GammaLineMatchResult(
                    nuclide=hit.nuclide,
                    display_name=hit.display_name,
                    line_energy_keV=hit.line_energy_keV,
                    delta_keV=hit.delta_keV,
                    intensity=hit.intensity,
                    half_life_s=hit.half_life_s,
                    source_id=self._source_id,
                )
                for hit in search_gamma_lines_by_energy(
                    self.database_path,
                    energy_keV,
                    tolerance_keV=tolerance_keV,
                    query=query,
                    limit=limit,
                    min_intensity=min_intensity,
                )
            ]

        token = self._normalize_query(query)
        database = self._gamma_database or GammaDatabase()
        matches: list[GammaLineMatchResult] = []
        for nuclide, line in database.find_matches(
            energy_keV,
            tolerance_keV=tolerance_keV,
            min_intensity=min_intensity,
        ):
            normalized = self._normalize_query(nuclide)
            if token and token not in normalized:
                continue
            matches.append(
                GammaLineMatchResult(
                    nuclide=nuclide,
                    display_name=nuclide,
                    line_energy_keV=round(line.energy_keV, 3),
                    delta_keV=round(line.energy_keV - float(energy_keV), 3),
                    intensity=float(line.intensity * line.norm),
                    half_life_s=float(
                        database.get(nuclide).halflife
                        if database.get(nuclide) is not None
                        else 0.0
                    ),
                    source_id=self._source_id,
                )
            )
        matches.sort(
            key=lambda item: (abs(item.delta_keV), -item.intensity, item.display_name.lower())
        )
        return matches[:limit]

    def _normalize_query(self, query: str) -> str:
        return "".join(character for character in query.lower() if character.isalnum())


__all__ = ["GammaLineMatchResult", "NuclideSearchController", "NuclideSearchResult"]
