"""Core planning and masking models shared across CLI and GUI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LineMaskingResult:
    """Masking diagnostics for one target line versus one interfering line."""

    target_nuclide: str
    target_line_energy_keV: float
    masking_nuclide: str
    masking_line_energy_keV: float
    energy_delta_keV: float
    target_signal_counts: float
    masking_signal_counts: float
    background_counts: float
    interference_counts: float
    continuum_counts: float
    masking_score: float
    burial_ratio: float
    recommended_action: str

    def to_row(self, rank: int) -> dict[str, Any]:
        return {
            "rank": int(rank),
            "target_nuclide": self.target_nuclide,
            "target_line_energy_keV": float(self.target_line_energy_keV),
            "masking_nuclide": self.masking_nuclide,
            "masking_line_energy_keV": float(self.masking_line_energy_keV),
            "energy_delta_keV": float(self.energy_delta_keV),
            "target_signal_counts": float(self.target_signal_counts),
            "masking_signal_counts": float(self.masking_signal_counts),
            "background_counts": float(self.background_counts),
            "interference_counts": float(self.interference_counts),
            "continuum_counts": float(self.continuum_counts),
            "masking_score": float(self.masking_score),
            "burial_ratio": float(self.burial_ratio),
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class OptimizationScenario:
    """One ranked optimization scenario used by report and recommendation layers."""

    objective: str
    label: str
    rank: int
    irradiation_time_s: float
    cooldown_time_s: float
    count_time_s: float
    objective_score: float
    mask_isotope: str | None = None
    isotope_of_interest: str | None = None
    expected_dose_uSv: float | None = None

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "objective": self.objective,
            "label": self.label,
            "rank": int(self.rank),
            "irradiation_time_s": float(self.irradiation_time_s),
            "cooldown_time_s": float(self.cooldown_time_s),
            "count_time_s": float(self.count_time_s),
            "objective_score": float(self.objective_score),
            "mask_isotope": self.mask_isotope,
            "isotope_of_interest": self.isotope_of_interest,
        }
        if self.expected_dose_uSv is not None:
            row["expected_dose_uSv"] = float(self.expected_dose_uSv)
        return row
