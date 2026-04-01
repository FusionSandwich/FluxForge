"""Nuclide search controller for the modern Qt shell."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from fluxforge.data.gamma_database import GammaDatabase
from fluxforge.data.isotope_names import format_isotope_name, parse_gamma_nuclide_name
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
from fluxforge.physics.activation import radioisotope_specific_activity_bq_g
from fluxforge.physics.dose import GammaLine as DoseGammaLine
from fluxforge.physics.dose import isotope_dose_rate


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


@dataclass(frozen=True)
class NuclideRelative:
    """One parent or daughter nuclide relationship surfaced in the GUI."""

    nuclide: str
    display_name: str
    decay_mode: str = ""
    branching_ratio: float = 0.0


@dataclass(frozen=True)
class NuclideLineDetail:
    """Detailed line row shown in the modern reference workbench."""

    energy_keV: float
    intensity: float
    age_adjusted_intensity: float
    line_type: str = "gamma"


@dataclass(frozen=True)
class NuclideDetailResult:
    """Resolved nuclide details for the PeakEasy-style reference workflow."""

    nuclide: str
    display_name: str
    source_id: str
    half_life_s: float
    gamma_lines: tuple[NuclideLineDetail, ...] = ()
    xray_lines: tuple[NuclideLineDetail, ...] = ()
    parents: tuple[NuclideRelative, ...] = ()
    daughters: tuple[NuclideRelative, ...] = ()
    specific_activity_bq_g: float = 0.0
    dose_rate_uSv_h_per_uCi_at_1m: float = 0.0


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

    def line_details_for_nuclide(
        self,
        nuclide: str,
        *,
        limit: int = 8,
        age_s: float = 0.0,
    ) -> tuple[NuclideLineDetail, ...]:
        if self._source_id == "fluxforge_bundled_gamma":
            return self._sqlite_line_details_for_nuclide(
                nuclide,
                limit=limit,
                age_s=age_s,
            )
        return self._database_line_details_for_nuclide(
            nuclide,
            line_type="gamma",
            limit=limit,
            age_s=age_s,
        )

    def nuclide_details(
        self,
        nuclide: str,
        *,
        age_s: float = 0.0,
        limit: int = 8,
    ) -> NuclideDetailResult:
        if self._source_id == "fluxforge_bundled_gamma":
            resolved_nuclide, display_name, half_life_s = self._sqlite_nuclide_identity(nuclide)
            gamma_lines = self._sqlite_line_details_for_nuclide(
                resolved_nuclide,
                limit=limit,
                age_s=age_s,
            )
            xray_lines: tuple[NuclideLineDetail, ...] = ()
            parents, daughters = self._sqlite_decay_relatives(resolved_nuclide)
        else:
            decay = self._database_decay_data(nuclide)
            if decay is None:
                resolved_nuclide = nuclide
                try:
                    element, mass_number, metastable = parse_gamma_nuclide_name(nuclide)
                    display_name = format_isotope_name(element, mass_number, metastable)
                except ValueError:
                    display_name = nuclide
                half_life_s = 0.0
                gamma_lines = ()
                xray_lines = ()
            else:
                resolved_nuclide = decay.nuclide
                try:
                    element, mass_number, metastable = parse_gamma_nuclide_name(decay.nuclide)
                    display_name = format_isotope_name(element, mass_number, metastable)
                except ValueError:
                    display_name = decay.nuclide
                half_life_s = float(decay.halflife or 0.0)
                gamma_lines = self._database_line_details_for_nuclide(
                    resolved_nuclide,
                    line_type="gamma",
                    limit=limit,
                    age_s=age_s,
                )
                xray_lines = self._database_line_details_for_nuclide(
                    resolved_nuclide,
                    line_type="xray",
                    limit=limit,
                    age_s=age_s,
                )
            parents = ()
            daughters = ()
        return NuclideDetailResult(
            nuclide=resolved_nuclide,
            display_name=display_name,
            source_id=self._source_id,
            half_life_s=float(half_life_s or 0.0),
            gamma_lines=tuple(gamma_lines),
            xray_lines=tuple(xray_lines),
            parents=tuple(parents),
            daughters=tuple(daughters),
            specific_activity_bq_g=float(
                radioisotope_specific_activity_bq_g(
                    float(half_life_s or 0.0),
                    isotope=display_name,
                )
            ),
            dose_rate_uSv_h_per_uCi_at_1m=float(
                self._estimate_dose_rate_uSv_h_per_uCi_at_1m(
                    tuple(gamma_lines),
                    half_life_s=float(half_life_s or 0.0),
                )
            ),
        )

    def _normalize_query(self, query: str) -> str:
        return "".join(character for character in query.lower() if character.isalnum())

    def _database_decay_data(self, nuclide: str):
        database = self._gamma_database or GammaDatabase()
        direct = database.get(nuclide)
        if direct is not None:
            return direct
        normalized = self._normalize_query(nuclide)
        for key in database.nuclides:
            if self._normalize_query(key) == normalized:
                return database.get(key)
        return None

    def _database_line_details_for_nuclide(
        self,
        nuclide: str,
        *,
        line_type: str,
        limit: int,
        age_s: float,
    ) -> tuple[NuclideLineDetail, ...]:
        decay = self._database_decay_data(nuclide)
        if decay is None:
            return ()
        lines = list(decay.get_lines(line_type))
        lines.sort(key=lambda item: item.intensity * item.norm, reverse=True)
        age_factor = self._age_factor(float(decay.halflife or 0.0), age_s)
        return tuple(
            NuclideLineDetail(
                energy_keV=round(line.energy_keV, 3),
                intensity=float(line.intensity * line.norm),
                age_adjusted_intensity=float(line.intensity * line.norm * age_factor),
                line_type=line_type,
            )
            for line in lines[:limit]
        )

    def _sqlite_nuclide_identity(self, nuclide: str) -> tuple[str, str, float]:
        normalized = self._normalize_query(nuclide)
        with sqlite3.connect(Path(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT name, display_name, half_life_s
                FROM nuclides
                WHERE lower(replace(replace(name, '-', ''), ' ', '')) = ?
                   OR lower(replace(replace(display_name, '-', ''), ' ', '')) = ?
                LIMIT 1
                """,
                (normalized, normalized),
            ).fetchone()
        if row is None:
            return nuclide, nuclide, 0.0
        return str(row[0]), str(row[1]), float(row[2] or 0.0)

    def _sqlite_line_details_for_nuclide(
        self,
        nuclide: str,
        *,
        limit: int,
        age_s: float,
    ) -> tuple[NuclideLineDetail, ...]:
        resolved_nuclide, _display_name, half_life_s = self._sqlite_nuclide_identity(nuclide)
        age_factor = self._age_factor(half_life_s, age_s)
        with sqlite3.connect(Path(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT g.energy_keV, g.intensity * g.norm, g.line_type
                FROM gamma_lines g
                JOIN nuclides n ON n.id = g.nuclide_id
                WHERE n.name = ?
                ORDER BY g.intensity * g.norm DESC, g.energy_keV ASC
                LIMIT ?
                """,
                (resolved_nuclide, int(limit)),
            ).fetchall()
        return tuple(
            NuclideLineDetail(
                energy_keV=round(float(energy_keV), 3),
                intensity=float(intensity or 0.0),
                age_adjusted_intensity=float(float(intensity or 0.0) * age_factor),
                line_type=str(line_type or "gamma"),
            )
            for energy_keV, intensity, line_type in rows
        )

    def _sqlite_decay_relatives(
        self,
        nuclide: str,
    ) -> tuple[tuple[NuclideRelative, ...], tuple[NuclideRelative, ...]]:
        resolved_nuclide, _display_name, _half_life_s = self._sqlite_nuclide_identity(nuclide)
        with sqlite3.connect(Path(self.database_path)) as connection:
            parents = connection.execute(
                """
                SELECT parent.name, parent.display_name, dc.decay_mode, dc.branching_ratio
                FROM decay_chains dc
                JOIN nuclides child ON child.id = dc.daughter_id
                JOIN nuclides parent ON parent.id = dc.parent_id
                WHERE child.name = ?
                ORDER BY COALESCE(dc.branching_ratio, 0.0) DESC, parent.display_name ASC
                """,
                (resolved_nuclide,),
            ).fetchall()
            daughters = connection.execute(
                """
                SELECT daughter.name, daughter.display_name, dc.decay_mode, dc.branching_ratio
                FROM decay_chains dc
                JOIN nuclides parent ON parent.id = dc.parent_id
                JOIN nuclides daughter ON daughter.id = dc.daughter_id
                WHERE parent.name = ?
                ORDER BY COALESCE(dc.branching_ratio, 0.0) DESC, daughter.display_name ASC
                """,
                (resolved_nuclide,),
            ).fetchall()
        return (
            tuple(
                NuclideRelative(
                    nuclide=str(name),
                    display_name=str(display_name),
                    decay_mode=str(decay_mode or ""),
                    branching_ratio=float(branching_ratio or 0.0),
                )
                for name, display_name, decay_mode, branching_ratio in parents
            ),
            tuple(
                NuclideRelative(
                    nuclide=str(name),
                    display_name=str(display_name),
                    decay_mode=str(decay_mode or ""),
                    branching_ratio=float(branching_ratio or 0.0),
                )
                for name, display_name, decay_mode, branching_ratio in daughters
            ),
        )

    def _estimate_dose_rate_uSv_h_per_uCi_at_1m(
        self,
        gamma_lines: tuple[NuclideLineDetail, ...],
        *,
        half_life_s: float,
    ) -> float:
        del half_life_s
        if not gamma_lines:
            return 0.0
        dose_lines = [
            DoseGammaLine(
                energy_keV=float(line.energy_keV),
                intensity=float(line.intensity),
            )
            for line in gamma_lines[:8]
        ]
        result = isotope_dose_rate(
            dose_lines,
            activity_Bq=3.7e4,
            distance_cm=100.0,
        )
        return float(result.dose_rate_uSv_h)

    def _age_factor(self, half_life_s: float, age_s: float) -> float:
        if half_life_s <= 0.0 or age_s <= 0.0:
            return 1.0
        return math.exp(-(math.log(2.0) / float(half_life_s)) * float(age_s))


__all__ = [
    "GammaLineMatchResult",
    "NuclideDetailResult",
    "NuclideLineDetail",
    "NuclideRelative",
    "NuclideSearchController",
    "NuclideSearchResult",
]
