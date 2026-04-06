"""Builders that derive optimization candidates from activity-review outputs.

These helpers connect measured isotope inventories to schedule candidates so the
CLI can optimize irradiation/cooldown/counting settings without requiring a
hand-authored candidate JSON payload.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_EPSILON = 1.0e-12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _nonnegative(value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        return 0.0
    return numeric


def _parse_energy_flux_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an energy/flux CSV into energy (eV) and flux arrays."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            field_map = {str(name).strip().lower(): str(name) for name in reader.fieldnames}
            energy_key = next(
                (
                    field_map[name]
                    for name in (
                        "energy_ev",
                        "energy_mev",
                        "energy_kev",
                        "energy",
                        "e_ev",
                        "e",
                    )
                    if name in field_map
                ),
                None,
            )
            flux_key = next(
                (
                    field_map[name]
                    for name in ("flux", "phi", "value", "spectral_flux")
                    if name in field_map
                ),
                None,
            )
            if energy_key and flux_key:
                energies: list[float] = []
                fluxes: list[float] = []
                energy_name_lower = energy_key.strip().lower()
                for row in reader:
                    energy = _safe_float(row.get(energy_key), default=float("nan"))
                    flux = _safe_float(row.get(flux_key), default=float("nan"))
                    if not np.isfinite(energy) or not np.isfinite(flux):
                        continue
                    if "mev" in energy_name_lower:
                        energy *= 1.0e6
                    elif "kev" in energy_name_lower:
                        energy *= 1.0e3
                    energies.append(float(energy))
                    fluxes.append(float(max(flux, 0.0)))
                if energies:
                    return np.asarray(energies, dtype=float), np.asarray(fluxes, dtype=float)

    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(
            f"Neutron spectrum CSV {path} must contain at least two columns (energy, flux)."
        )
    energy = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    return energy, flux


def summarize_neutron_spectrum_source(
    *,
    unfold_payload: Mapping[str, Any] | None = None,
    neutron_spectrum_csv: Path | None = None,
    base_scale: float = 1.0,
    reference_integral_flux: float = 0.0,
) -> dict[str, Any]:
    """Return an applied flux scale from unfolded or tabular neutron spectra."""

    integral_flux = 0.0
    high_energy_fraction = 0.5
    source = "none"

    if unfold_payload is not None:
        flux_values = np.asarray(unfold_payload.get("flux") or [], dtype=float)
        boundaries = np.asarray(
            unfold_payload.get("boundaries_eV")
            or unfold_payload.get("energy_edges_eV")
            or [],
            dtype=float,
        )
        if flux_values.size > 0:
            source = "unfold"
            integral_flux = float(np.sum(np.clip(flux_values, 0.0, None)))
            if boundaries.size == flux_values.size + 1:
                mids = np.sqrt(np.clip(boundaries[:-1], _EPSILON, None) * np.clip(boundaries[1:], _EPSILON, None))
                high_mask = mids >= 1.0e5
                total = np.sum(np.clip(flux_values, 0.0, None))
                if total > 0.0:
                    high_energy_fraction = float(
                        np.sum(np.clip(flux_values[high_mask], 0.0, None)) / total
                    )

    if neutron_spectrum_csv is not None:
        energies_eV, flux_values = _parse_energy_flux_rows(neutron_spectrum_csv)
        if energies_eV.size > 0:
            source = "csv"
            integral_flux = float(np.sum(np.clip(flux_values, 0.0, None)))
            total = np.sum(np.clip(flux_values, 0.0, None))
            if total > 0.0:
                high_energy_fraction = float(
                    np.sum(np.clip(flux_values[energies_eV >= 1.0e5], 0.0, None))
                    / total
                )

    hardness_scale = 0.5 + max(min(high_energy_fraction, 1.0), 0.0)
    applied_scale = max(float(base_scale), 0.0) * hardness_scale

    if reference_integral_flux > 0.0 and integral_flux > 0.0:
        ratio = integral_flux / max(float(reference_integral_flux), _EPSILON)
        applied_scale *= float(np.clip(ratio, 0.1, 10.0))

    return {
        "source": source,
        "integral_flux": float(integral_flux),
        "high_energy_fraction": float(high_energy_fraction),
        "hardness_scale": float(hardness_scale),
        "applied_flux_scale": float(applied_scale),
    }


def _isotope_rows_from_activity_review(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("isotope_summaries") or []
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if not rows:
        raise ValueError(
            "Activity-review payload does not contain isotope_summaries to build optimization candidates."
        )
    return [dict(row) for row in rows]


def _line_energy_map(payload: Mapping[str, Any]) -> dict[str, float]:
    energy_map: dict[str, float] = {}
    line_rows = payload.get("line_results") or []
    for row in line_rows:
        if not isinstance(row, Mapping):
            continue
        nuclide = str(row.get("nuclide") or "").strip()
        if not nuclide or nuclide in energy_map:
            continue
        energy = _safe_float(row.get("matched_line_energy_keV"), default=float("nan"))
        if np.isfinite(energy) and energy > 0.0:
            energy_map[nuclide] = float(energy)
    return energy_map


def build_difom_payload_from_activity_review(
    activity_review_payload: Mapping[str, Any],
    *,
    irradiation_grid_s: Sequence[float],
    cooldown_grid_s: Sequence[float],
    count_grid_s: Sequence[float],
    reference_irradiation_time_s: float,
    flux_scale: float = 1.0,
) -> dict[str, Any]:
    """Build a DI-FOM compatible candidate payload from activity review output."""

    isotope_rows = _isotope_rows_from_activity_review(activity_review_payload)
    line_energy_map = _line_energy_map(activity_review_payload)

    baseline_live_time_s = max(
        _safe_float(activity_review_payload.get("live_time_s"), default=1.0),
        _EPSILON,
    )
    baseline_cooling_s = max(
        _safe_float(activity_review_payload.get("cooling_time_s"), default=0.0),
        0.0,
    )
    reference_irr_s = max(float(reference_irradiation_time_s), _EPSILON)

    candidates: list[dict[str, Any]] = []
    isotope_weights: dict[str, float] = {}

    for t_irr in irradiation_grid_s:
        irradiation_time_s = max(float(t_irr), 1.0)
        for t_cool in cooldown_grid_s:
            cooldown_time_s = max(float(t_cool), 0.0)
            for t_count in count_grid_s:
                count_time_s = max(float(t_count), 1.0)
                lines: list[dict[str, Any]] = []

                # First pass: estimate signal/background per isotope line.
                for row in isotope_rows:
                    nuclide = str(row.get("nuclide") or "").strip()
                    if not nuclide:
                        continue
                    base_activity_eoi = _safe_float(
                        row.get("irradiation_time_activity_Bq"),
                        default=0.0,
                    )
                    if base_activity_eoi <= 0.0:
                        continue

                    half_life_s = max(_safe_float(row.get("half_life_s"), default=0.0), _EPSILON)
                    lam = math.log(2.0) / half_life_s
                    buildup_ref = max(1.0 - math.exp(-lam * reference_irr_s), _EPSILON)
                    buildup_new = max(1.0 - math.exp(-lam * irradiation_time_s), _EPSILON)

                    scaled_activity_eoi = base_activity_eoi * max(float(flux_scale), 0.0) * (buildup_new / buildup_ref)
                    scaled_activity_count = scaled_activity_eoi * math.exp(-lam * cooldown_time_s)

                    base_activity_count = base_activity_eoi * math.exp(-lam * baseline_cooling_s)
                    baseline_total_counts = max(
                        _safe_float(row.get("total_net_counts"), default=0.0),
                        0.0,
                    )
                    if base_activity_count > 0.0 and baseline_total_counts > 0.0:
                        counts_per_bq_s = baseline_total_counts / (base_activity_count * baseline_live_time_s)
                    else:
                        counts_per_bq_s = 1.0

                    signal_counts = max(scaled_activity_count * counts_per_bq_s * count_time_s, 0.0)

                    rel_unc = _safe_float(
                        row.get("irradiation_time_activity_unc_Bq"),
                        default=0.0,
                    ) / max(base_activity_eoi, _EPSILON)
                    background_counts = signal_counts * max(rel_unc, 0.05)

                    lines.append(
                        {
                            "nuclide": nuclide,
                            "line_energy_keV": float(
                                line_energy_map.get(
                                    nuclide,
                                    _safe_float(row.get("matched_line_energy_keV"), default=0.0),
                                )
                            ),
                            "signal_counts": float(signal_counts),
                            "background_counts": float(background_counts),
                            "interference_counts": 0.0,
                            "half_life_s": float(half_life_s),
                        }
                    )
                    isotope_weights.setdefault(nuclide, max(math.sqrt(base_activity_eoi), 1.0))

                if not lines:
                    continue

                # Second pass: estimate simple overlap interference burden.
                for idx, line in enumerate(lines):
                    interference = 0.0
                    for jdx, other in enumerate(lines):
                        if idx == jdx:
                            continue
                        if abs(_safe_float(line.get("line_energy_keV")) - _safe_float(other.get("line_energy_keV"))) <= 2.5:
                            interference += 0.2 * _nonnegative(_safe_float(other.get("signal_counts")))
                    line["interference_counts"] = float(interference)

                label = (
                    f"irr_{int(round(irradiation_time_s))}s_"
                    f"cool_{int(round(cooldown_time_s))}s_"
                    f"count_{int(round(count_time_s))}s"
                )
                candidates.append(
                    {
                        "label": label,
                        "irradiation_time_s": float(irradiation_time_s),
                        "cooldown_time_s": float(cooldown_time_s),
                        "count_time_s": float(count_time_s),
                        "lines": lines,
                    }
                )

    if not candidates:
        raise ValueError(
            "Could not construct optimization candidates from activity-review isotope summaries."
        )

    return {
        "schema": "fluxforge.optimization_candidates.activity_review.v1",
        "source": "activity_review",
        "activity_review_schema": str(activity_review_payload.get("schema") or ""),
        "candidates": candidates,
        "isotope_weights": {key: float(value) for key, value in isotope_weights.items()},
    }


__all__ = [
    "build_difom_payload_from_activity_review",
    "summarize_neutron_spectrum_source",
]
