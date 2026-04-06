"""Shared spectrum-level activation review helpers for GUI and CLI flows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fluxforge.core.analysis_workspace import PeakCandidate
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.data.gamma_database import FLUXFORGE_GAMMA_DATA, DecayData, GammaDatabase, GammaLine
from fluxforge.data.isotope_names import format_isotope_name, parse_nndc_isotope_name
from fluxforge.data.nuclear_data_sources import load_gamma_identification_source
from fluxforge.physics.activation import GammaLineMeasurement, activation_study_metrics
from fluxforge.physics.decay_chain import DecayChain


@dataclass(frozen=True)
class ActivityReviewLineResult:
    """One matched peak/gamma-line activity result."""

    peak_id: str
    nuclide: str
    peak_energy_keV: float
    line_energy_keV: float
    line_delta_keV: float
    net_counts: float
    net_counts_uncertainty: float
    efficiency: float
    efficiency_rel_uncertainty: float
    emission_probability: float
    emission_probability_uncertainty: float
    half_life_s: float
    count_time_activity_bq: float
    count_time_uncertainty_bq: float
    irradiation_time_activity_bq: float
    irradiation_time_uncertainty_bq: float
    cooling_time_s: float

    def to_row(self, *, sample_mass_g: float | None = None) -> dict[str, Any]:
        row = {
            "peak_id": self.peak_id,
            "nuclide": self.nuclide,
            "peak_energy_keV": self.peak_energy_keV,
            "matched_line_energy_keV": self.line_energy_keV,
            "line_delta_keV": self.line_delta_keV,
            "net_counts": self.net_counts,
            "net_counts_uncertainty": self.net_counts_uncertainty,
            "efficiency": self.efficiency,
            "efficiency_rel_uncertainty": self.efficiency_rel_uncertainty,
            "emission_probability": self.emission_probability,
            "emission_probability_uncertainty": self.emission_probability_uncertainty,
            "half_life_s": self.half_life_s,
            "cooling_time_s": self.cooling_time_s,
            "count_time_activity_Bq": self.count_time_activity_bq,
            "count_time_activity_unc_Bq": self.count_time_uncertainty_bq,
            "irradiation_time_activity_Bq": self.irradiation_time_activity_bq,
            "irradiation_time_activity_unc_Bq": self.irradiation_time_uncertainty_bq,
        }
        row.update(
            activation_study_metrics(
                activity_bq=self.irradiation_time_activity_bq,
                activity_unc_bq=self.irradiation_time_uncertainty_bq,
                half_life_s=self.half_life_s,
                isotope=self.nuclide,
                sample_mass_g=sample_mass_g,
            )
        )
        return row


@dataclass(frozen=True)
class ActivityReviewIsotopeSummary:
    """Aggregated isotope activity summary for one spectrum."""

    nuclide: str
    line_count: int
    peak_energies_keV: tuple[float, ...]
    matched_line_energies_keV: tuple[float, ...]
    total_net_counts: float
    half_life_s: float
    count_time_activity_bq: float
    count_time_uncertainty_bq: float
    irradiation_time_activity_bq: float
    irradiation_time_uncertainty_bq: float
    cooling_time_s: float
    chain_summary: str

    def to_row(self, *, sample_mass_g: float | None = None) -> dict[str, Any]:
        row = {
            "nuclide": self.nuclide,
            "line_count": self.line_count,
            "peak_energies_keV": ", ".join(f"{value:.3f}" for value in self.peak_energies_keV),
            "matched_line_energies_keV": ", ".join(
                f"{value:.3f}" for value in self.matched_line_energies_keV
            ),
            "total_net_counts": self.total_net_counts,
            "half_life_s": self.half_life_s,
            "cooling_time_s": self.cooling_time_s,
            "count_time_activity_Bq": self.count_time_activity_bq,
            "count_time_activity_unc_Bq": self.count_time_uncertainty_bq,
            "irradiation_time_activity_Bq": self.irradiation_time_activity_bq,
            "irradiation_time_activity_unc_Bq": self.irradiation_time_uncertainty_bq,
            "relative_uncertainty": (
                self.irradiation_time_uncertainty_bq
                / max(self.irradiation_time_activity_bq, 1e-30)
            ),
            "chain_summary": self.chain_summary,
        }
        row.update(
            activation_study_metrics(
                activity_bq=self.irradiation_time_activity_bq,
                activity_unc_bq=self.irradiation_time_uncertainty_bq,
                half_life_s=self.half_life_s,
                isotope=self.nuclide,
                sample_mass_g=sample_mass_g,
            )
        )
        return row


@dataclass(frozen=True)
class ActivityReviewResult:
    """Spectrum-level activation review with export-ready tables and plot series."""

    source_id: str
    custom_gamma_path: str | None
    live_time_s: float
    cooling_time_s: float
    plot_horizon_s: float
    line_results: tuple[ActivityReviewLineResult, ...]
    isotope_summaries: tuple[ActivityReviewIsotopeSummary, ...]
    decay_plot_data: dict[str, tuple[tuple[float, float, float], ...]]
    bateman_plot_data: dict[str, tuple[tuple[float, float, float], ...]]
    half_lives_s: dict[str, float]
    bateman_half_lives_s: dict[str, float]

    def line_rows(self, *, sample_mass_g: float | None = None) -> list[dict[str, Any]]:
        return [item.to_row(sample_mass_g=sample_mass_g) for item in self.line_results]

    def isotope_rows(self, *, sample_mass_g: float | None = None) -> list[dict[str, Any]]:
        return [item.to_row(sample_mass_g=sample_mass_g) for item in self.isotope_summaries]

    def to_payload(self, *, sample_mass_g: float | None = None) -> dict[str, Any]:
        return {
            "schema": "fluxforge.activity_review.v1",
            "source_id": self.source_id,
            "custom_gamma_path": self.custom_gamma_path,
            "live_time_s": self.live_time_s,
            "cooling_time_s": self.cooling_time_s,
            "irradiation_reference": "end_of_irradiation",
            "plot_horizon_s": self.plot_horizon_s,
            "notes": [
                "Irradiation-time activity rows are back-corrected to end of irradiation.",
                "Uncertainties include counting statistics, efficiency relative uncertainty, and emission-probability uncertainty when available.",
                "Bateman daughter curves are still simple parent-to-daughter EOI-equivalent inventory series; bundled decay-network and half-life-uncertainty libraries now exist separately, but this workflow does not consume them yet.",
            ],
            "line_results": self.line_rows(sample_mass_g=sample_mass_g),
            "isotope_summaries": self.isotope_rows(sample_mass_g=sample_mass_g),
        }


def review_spectrum_activation(
    peaks: Sequence[PeakCandidate],
    *,
    live_time_s: float,
    efficiency_curve: EfficiencyCurve,
    cooling_time_s: float = 0.0,
    source_id: str = "fluxforge_bundled_gamma",
    custom_gamma_path: str | None = None,
    energy_tolerance_keV: float = 2.0,
    dead_time_fraction: float = 0.0,
    sample_mass_g: float | None = None,
) -> ActivityReviewResult:
    """Resolve assigned peaks to isotope activities at count time and EOI."""

    database = _load_gamma_database(source_id, custom_gamma_path=custom_gamma_path)
    live_time = max(float(live_time_s), 1e-9)
    cooling_time = max(float(cooling_time_s), 0.0)
    dead_time = max(float(dead_time_fraction), 0.0)

    line_results: list[ActivityReviewLineResult] = []
    grouped: dict[str, list[ActivityReviewLineResult]] = {}

    for peak in peaks:
        if not peak.nuclide or peak.net_counts <= 0.0:
            continue
        resolved = _resolve_decay_record(database, peak.nuclide)
        if resolved is None:
            continue
        resolved_name, decay = resolved
        matched_line = _resolve_gamma_line(
            decay, peak.energy_keV, tolerance_keV=energy_tolerance_keV
        )
        if matched_line is None:
            continue

        efficiency = max(_as_float(efficiency_curve.efficiency(peak.energy_keV)), 1e-12)
        efficiency_rel_unc = max(
            _as_float(efficiency_curve.efficiency_uncertainty(peak.energy_keV)),
            0.0,
        )
        emission_probability = max(
            float(matched_line.intensity * matched_line.norm),
            1e-12,
        )
        emission_probability_unc = _effective_probability_uncertainty(matched_line)
        half_life_s = max(float(decay.halflife), 1e-12)
        net_counts = max(float(peak.net_counts), 0.0)
        net_counts_unc = math.sqrt(max(net_counts, 1.0))

        measurement = GammaLineMeasurement(
            net_counts=net_counts,
            live_time_s=live_time,
            efficiency=efficiency,
            gamma_intensity=emission_probability,
            half_life_s=half_life_s,
            cooling_time_s=cooling_time,
            dead_time_fraction=dead_time,
        )
        irradiation_activity = measurement.activity_at_reference()
        decay_constant = math.log(2.0) / half_life_s
        count_time_activity = irradiation_activity * math.exp(-decay_constant * cooling_time)

        rel_count_unc = net_counts_unc / max(net_counts, 1.0)
        rel_emission_unc = emission_probability_unc / max(emission_probability, 1e-12)
        combined_rel_unc = math.sqrt(
            rel_count_unc * rel_count_unc
            + efficiency_rel_unc * efficiency_rel_unc
            + rel_emission_unc * rel_emission_unc
        )

        result = ActivityReviewLineResult(
            peak_id=peak.peak_id,
            nuclide=resolved_name,
            peak_energy_keV=float(peak.energy_keV),
            line_energy_keV=float(matched_line.energy_keV),
            line_delta_keV=float(abs(matched_line.energy_keV - peak.energy_keV)),
            net_counts=net_counts,
            net_counts_uncertainty=float(net_counts_unc),
            efficiency=float(efficiency),
            efficiency_rel_uncertainty=float(efficiency_rel_unc),
            emission_probability=float(emission_probability),
            emission_probability_uncertainty=float(emission_probability_unc),
            half_life_s=float(half_life_s),
            count_time_activity_bq=float(count_time_activity),
            count_time_uncertainty_bq=float(count_time_activity * combined_rel_unc),
            irradiation_time_activity_bq=float(irradiation_activity),
            irradiation_time_uncertainty_bq=float(irradiation_activity * combined_rel_unc),
            cooling_time_s=float(cooling_time),
        )
        line_results.append(result)
        grouped.setdefault(resolved_name, []).append(result)

    if not line_results:
        raise ValueError(
            "No assigned peaks could be resolved to gamma-library lines for activity review."
        )

    isotope_summaries: list[ActivityReviewIsotopeSummary] = []
    half_lives_s: dict[str, float] = {}

    for nuclide in sorted(grouped):
        nuclide_lines = grouped[nuclide]
        count_time_activity, count_time_unc = _weighted_mean_and_uncertainty(
            [item.count_time_activity_bq for item in nuclide_lines],
            [item.count_time_uncertainty_bq for item in nuclide_lines],
        )
        irradiation_activity, irradiation_unc = _weighted_mean_and_uncertainty(
            [item.irradiation_time_activity_bq for item in nuclide_lines],
            [item.irradiation_time_uncertainty_bq for item in nuclide_lines],
        )
        half_life_s = max(
            max((item.half_life_s for item in nuclide_lines), default=0.0),
            1e-12,
        )
        half_lives_s[nuclide] = half_life_s
        isotope_summaries.append(
            ActivityReviewIsotopeSummary(
                nuclide=nuclide,
                line_count=len(nuclide_lines),
                peak_energies_keV=tuple(
                    sorted(float(item.peak_energy_keV) for item in nuclide_lines)
                ),
                matched_line_energies_keV=tuple(
                    sorted(float(item.line_energy_keV) for item in nuclide_lines)
                ),
                total_net_counts=float(sum(item.net_counts for item in nuclide_lines)),
                half_life_s=float(half_life_s),
                count_time_activity_bq=float(count_time_activity),
                count_time_uncertainty_bq=float(count_time_unc),
                irradiation_time_activity_bq=float(irradiation_activity),
                irradiation_time_uncertainty_bq=float(irradiation_unc),
                cooling_time_s=float(cooling_time),
                chain_summary=build_simple_bateman_summary(
                    nuclide,
                    half_life_s=half_life_s,
                    cooling_time_s=cooling_time,
                ),
            )
        )

    plot_horizon_s = _resolve_plot_horizon(
        [item.half_life_s for item in isotope_summaries],
        cooling_time_s=cooling_time,
    )
    time_points = tuple(float(value) for value in np.linspace(0.0, plot_horizon_s, 64))
    decay_plot_data: dict[str, tuple[tuple[float, float, float], ...]] = {}
    bateman_plot_data: dict[str, tuple[tuple[float, float, float], ...]] = {}
    bateman_half_lives: dict[str, float] = {}

    for summary in isotope_summaries:
        decay_plot_data[summary.nuclide] = tuple(
            (
                time_s,
                _decay_activity(summary.irradiation_time_activity_bq, summary.half_life_s, time_s),
                _decay_activity(
                    summary.irradiation_time_uncertainty_bq,
                    summary.half_life_s,
                    time_s,
                ),
            )
            for time_s in time_points
        )
        parent_label = f"{summary.nuclide} parent"
        daughter_label = f"{summary.nuclide} daughter eq"
        parent_series, daughter_series = _simple_bateman_series(
            summary.nuclide,
            activity_bq=summary.irradiation_time_activity_bq,
            uncertainty_bq=summary.irradiation_time_uncertainty_bq,
            half_life_s=summary.half_life_s,
            time_points=time_points,
        )
        bateman_plot_data[parent_label] = parent_series
        bateman_plot_data[daughter_label] = daughter_series
        bateman_half_lives[parent_label] = summary.half_life_s

    return ActivityReviewResult(
        source_id=source_id,
        custom_gamma_path=custom_gamma_path,
        live_time_s=live_time,
        cooling_time_s=cooling_time,
        plot_horizon_s=float(plot_horizon_s),
        line_results=tuple(sorted(line_results, key=lambda item: (item.nuclide, item.peak_energy_keV))),
        isotope_summaries=tuple(isotope_summaries),
        decay_plot_data=decay_plot_data,
        bateman_plot_data=bateman_plot_data,
        half_lives_s=half_lives_s,
        bateman_half_lives_s=bateman_half_lives,
    )


def build_simple_bateman_summary(
    nuclide: str,
    *,
    half_life_s: float,
    cooling_time_s: float,
) -> str:
    """Return a short parent/daughter Bateman review summary."""

    daughter = f"{nuclide} daughter"
    chain = DecayChain(
        nuclide,
        nuclide_data={
            nuclide: {"half_life_s": float(half_life_s), "decay_products": {daughter: 1.0}},
            daughter: {"half_life_s": float("inf"), "decay_products": {}},
        },
    )
    result = chain.decay(initial_atoms={nuclide: 1.0}, times=[0.0, float(cooling_time_s)])
    parent_fraction = float(result.atoms[nuclide][-1])
    daughter_fraction = float(result.atoms[daughter][-1])
    return (
        f"Bateman correction over {cooling_time_s / 3600.0:.2f} h: "
        f"{nuclide} retains {parent_fraction:.4f} of its EOI inventory and transfers "
        f"{daughter_fraction:.4f} to the daughter path."
    )


def _load_gamma_database(
    source_id: str,
    *,
    custom_gamma_path: str | None = None,
) -> GammaDatabase:
    if source_id == "fluxforge_bundled_gamma":
        return GammaDatabase(FLUXFORGE_GAMMA_DATA)
    return load_gamma_identification_source(source_id, custom_path=custom_gamma_path)


def _candidate_nuclide_names(nuclide: str) -> tuple[str, ...]:
    raw = str(nuclide).strip()
    candidates = [raw, raw.replace("-", ""), raw.replace("_", "-")]
    try:
        element, mass, meta = parse_nndc_isotope_name(raw)
        candidates.extend(
            [
                format_isotope_name(element, mass, meta, separator=""),
                format_isotope_name(element, mass, meta, separator="-"),
            ]
        )
    except ValueError:
        pass
    ordered: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = str(item).strip()
        if text and text.lower() not in seen:
            ordered.append(text)
            seen.add(text.lower())
    return tuple(ordered)


def _resolve_decay_record(
    database: GammaDatabase,
    nuclide: str,
) -> tuple[str, DecayData] | None:
    lookup = {name.lower(): name for name in database.nuclides}
    for candidate in _candidate_nuclide_names(nuclide):
        record = database.get(candidate)
        if record is not None:
            return candidate, record
        matched_name = lookup.get(candidate.lower())
        if matched_name is not None:
            record = database.get(matched_name)
            if record is not None:
                return matched_name, record
    return None


def _resolve_gamma_line(
    decay: DecayData,
    peak_energy_keV: float,
    *,
    tolerance_keV: float,
) -> GammaLine | None:
    candidates = [
        line
        for line in decay.gamma_lines
        if abs(line.energy_keV - float(peak_energy_keV)) <= max(float(tolerance_keV), 1e-6)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda line: (
            abs(line.energy_keV - float(peak_energy_keV)),
            -(line.intensity * line.norm),
        ),
    )


def _effective_probability_uncertainty(line: GammaLine) -> float:
    effective_probability = max(float(line.intensity * line.norm), 1e-12)
    rel_terms: list[float] = []
    if line.intensity > 0.0 and line.intensity_unc > 0.0:
        rel_terms.append(float(line.intensity_unc) / float(line.intensity))
    if line.norm > 0.0 and line.norm_unc > 0.0:
        rel_terms.append(float(line.norm_unc) / float(line.norm))
    if not rel_terms:
        return 0.0
    return effective_probability * math.sqrt(sum(term * term for term in rel_terms))


def _resolve_plot_horizon(
    half_lives_s: Sequence[float],
    *,
    cooling_time_s: float,
) -> float:
    candidates = [max(float(cooling_time_s), 0.0), 3600.0]
    candidates.extend(
        max(float(value), 0.0) * 5.0 for value in half_lives_s if float(value) > 0.0
    )
    return float(max(candidates))


def _decay_activity(activity_bq: float, half_life_s: float, time_s: float) -> float:
    if half_life_s <= 0.0:
        return float(activity_bq)
    return float(activity_bq) * math.exp(-(math.log(2.0) / float(half_life_s)) * float(time_s))


def _simple_bateman_series(
    nuclide: str,
    *,
    activity_bq: float,
    uncertainty_bq: float,
    half_life_s: float,
    time_points: Sequence[float],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    daughter = f"{nuclide} daughter"
    chain = DecayChain(
        nuclide,
        nuclide_data={
            nuclide: {"half_life_s": float(half_life_s), "decay_products": {daughter: 1.0}},
            daughter: {"half_life_s": float("inf"), "decay_products": {}},
        },
    )
    result = chain.decay(initial_atoms={nuclide: 1.0}, times=list(time_points))
    parent_atoms = np.asarray(result.atoms[nuclide], dtype=float)
    daughter_atoms = np.asarray(result.atoms[daughter], dtype=float)
    parent_series = tuple(
        (
            float(time_s),
            float(activity_bq) * float(parent_fraction),
            float(uncertainty_bq) * float(parent_fraction),
        )
        for time_s, parent_fraction in zip(time_points, parent_atoms)
    )
    daughter_series = tuple(
        (
            float(time_s),
            float(activity_bq) * float(daughter_fraction),
            float(uncertainty_bq) * float(daughter_fraction),
        )
        for time_s, daughter_fraction in zip(time_points, daughter_atoms)
    )
    return parent_series, daughter_series


def _weighted_mean_and_uncertainty(
    values: Sequence[float],
    uncertainties: Sequence[float],
) -> tuple[float, float]:
    valid_pairs = [
        (float(value), float(uncertainty))
        for value, uncertainty in zip(values, uncertainties)
        if float(uncertainty) > 0.0
    ]
    if valid_pairs:
        weights = [1.0 / (uncertainty * uncertainty) for _, uncertainty in valid_pairs]
        weighted = sum(
            value * weight for (value, _), weight in zip(valid_pairs, weights)
        ) / max(sum(weights), 1e-30)
        return float(weighted), float(math.sqrt(1.0 / max(sum(weights), 1e-30)))
    values_array = np.asarray(values, dtype=float)
    if values_array.size == 0:
        return 0.0, 0.0
    mean_value = float(np.mean(values_array))
    if values_array.size == 1:
        return mean_value, float(max(float(uncertainties[0]) if uncertainties else 0.0, 0.0))
    spread = float(np.std(values_array, ddof=0) / max(math.sqrt(values_array.size), 1.0))
    return mean_value, spread


def _as_float(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(array.reshape(-1)[0])


__all__ = [
    "ActivityReviewIsotopeSummary",
    "ActivityReviewLineResult",
    "ActivityReviewResult",
    "build_simple_bateman_summary",
    "review_spectrum_activation",
]
