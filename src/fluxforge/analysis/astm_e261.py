"""ASTM E261 reactor dosimetry workflow helpers.

This module packages the core ASTM E261 activation workflow around FluxForge's
existing activity and reaction-rate primitives. The workflow is intentionally
data-driven: users provide a JSON plan describing one or more activated
monitors, the irradiation history, and the effective spectrum-averaged cross
sections that should be used to convert reaction rate into fluence and fluence
rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from fluxforge.physics.activation import (
    AVOGADRO,
    GammaLineMeasurement,
    IrradiationSegment,
    activation_study_metrics,
    irradiation_buildup_factor,
)


BARN_TO_CM2 = 1.0e-24


@dataclass(frozen=True)
class AstmE261MonitorResult:
    """Per-monitor ASTM E261 result payload."""

    measurement_id: str
    reaction_id: str
    payload: dict[str, Any]


def target_atom_count(
    *,
    sample_mass_g: float,
    atomic_mass_g_mol: float,
    isotopic_abundance: float = 1.0,
    mass_fraction: float = 1.0,
    sample_purity: float = 1.0,
    atoms_per_formula_unit: float = 1.0,
) -> float:
    r"""Return the number of target nuclei in the monitor sample.

    This implements the standard monitor inventory relationship

    $$N = \frac{m\,w\,p\,\theta}{M} N_A n$$

    where $m$ is sample mass, $w$ is analyte mass fraction, $p$ is purity,
    $\theta$ is isotopic abundance, $M$ is molar mass, and $n$ is the number of
    target atoms per formula unit.
    """

    if sample_mass_g <= 0.0 or atomic_mass_g_mol <= 0.0:
        return 0.0
    effective_mass = sample_mass_g * mass_fraction * sample_purity * isotopic_abundance
    return float(
        (effective_mass / atomic_mass_g_mol) * AVOGADRO * atoms_per_formula_unit
    )


def equivalent_irradiation_duration_s(segments: Sequence[IrradiationSegment]) -> float:
    """Return the power-weighted irradiation duration used for fluence totals."""

    return float(
        sum(segment.duration_s * segment.relative_power for segment in segments)
    )


def _relative_uncertainty(value: float, uncertainty: float) -> float:
    if value <= 0.0 or uncertainty <= 0.0:
        return 0.0
    return float(uncertainty / value)


def _coerce_segments(payload: dict[str, Any]) -> list[IrradiationSegment]:
    irradiation = payload.get("irradiation") or {}
    raw_segments = irradiation.get("segments")
    if raw_segments:
        return [
            IrradiationSegment(
                duration_s=float(row.get("duration_s", 0.0)),
                relative_power=float(row.get("relative_power", 1.0)),
            )
            for row in raw_segments
        ]
    duration_s = float(
        irradiation.get("duration_s", payload.get("duration_s", 0.0)) or 0.0
    )
    return (
        [IrradiationSegment(duration_s=duration_s, relative_power=1.0)]
        if duration_s > 0.0
        else []
    )


def _build_measurement(row: dict[str, Any]) -> GammaLineMeasurement:
    return GammaLineMeasurement(
        net_counts=float(row.get("net_counts", 0.0)),
        live_time_s=float(row.get("live_time_s", 0.0)),
        efficiency=float(row.get("efficiency", 0.0)),
        gamma_intensity=float(
            row.get("gamma_intensity", row.get("emission_probability", 0.0))
        ),
        half_life_s=float(row.get("half_life_s", 0.0)),
        cooling_time_s=float(
            row.get("cooling_time_s", row.get("decay_time_s", 0.0)) or 0.0
        ),
        dead_time_fraction=float(row.get("dead_time_fraction", 0.0) or 0.0),
    )


def analyze_astm_e261_plan(plan: dict[str, Any]) -> dict[str, Any]:
    r"""Execute an ASTM E261 monitor plan.

    Expected input schema:

    - `irradiation.segments`: list of `{duration_s, relative_power}` entries
    - `measurements`: list of monitor rows containing count, efficiency,
      half-life, target inventory, and effective cross-section data

    Required per-measurement fields:

    - `net_counts`, `live_time_s`, `efficiency`, `gamma_intensity`, `half_life_s`
    - `sample_mass_g`, `atomic_mass_g_mol`
    - `effective_cross_section_barn`

    Optional correction factors are applied multiplicatively in the denominator:

    $$\phi = \frac{R}{N\,\bar{\sigma}\,C}$$

    where $C$ is the product of self-shielding, cover, geometry, and any user
    supplied ASTM correction factor.
    """

    segments = _coerce_segments(plan)
    if not segments:
        raise ValueError(
            "ASTM E261 workflow requires a positive irradiation duration or segment list."
        )

    measurement_rows = list(plan.get("measurements") or [])
    if not measurement_rows:
        raise ValueError("ASTM E261 workflow requires at least one measurement entry.")

    fluence_duration_s = float(
        (plan.get("irradiation") or {}).get(
            "fluence_duration_s", equivalent_irradiation_duration_s(segments)
        )
    )
    results: list[dict[str, Any]] = []

    for index, row in enumerate(measurement_rows, start=1):
        measurement = _build_measurement(row)
        activity_eoi_bq = measurement.activity_at_reference()
        activity_unc_bq = float(
            activity_eoi_bq / math.sqrt(max(measurement.net_counts, 1.0))
        )
        buildup_factor = irradiation_buildup_factor(segments, measurement.half_life_s)
        if buildup_factor <= 0.0:
            raise ValueError(
                "Irradiation buildup factor must be positive for ASTM E261 analysis."
            )
        reaction_rate_s = float(activity_eoi_bq / buildup_factor)
        reaction_rate_unc_s = float(activity_unc_bq / buildup_factor)

        target_atoms = target_atom_count(
            sample_mass_g=float(row.get("sample_mass_g", 0.0)),
            atomic_mass_g_mol=float(row.get("atomic_mass_g_mol", 0.0)),
            isotopic_abundance=float(row.get("isotopic_abundance", 1.0) or 1.0),
            mass_fraction=float(row.get("mass_fraction", 1.0) or 1.0),
            sample_purity=float(row.get("sample_purity", 1.0) or 1.0),
            atoms_per_formula_unit=float(row.get("atoms_per_formula_unit", 1.0) or 1.0),
        )
        effective_cross_section_barn = float(
            row.get("effective_cross_section_barn", 0.0)
        )
        effective_cross_section_unc_barn = float(
            row.get("effective_cross_section_unc_barn", 0.0) or 0.0
        )
        correction_factor = (
            float(row.get("astm_correction_factor", 1.0) or 1.0)
            * float(row.get("self_shielding_factor", 1.0) or 1.0)
            * float(row.get("cover_correction_factor", 1.0) or 1.0)
            * float(row.get("geometry_factor", 1.0) or 1.0)
        )
        denominator = (
            target_atoms
            * effective_cross_section_barn
            * BARN_TO_CM2
            * correction_factor
        )
        if denominator <= 0.0:
            raise ValueError(
                "ASTM E261 workflow requires positive target atoms, cross section, and correction factor."
            )

        fluence_rate_cm2_s = float(reaction_rate_s / denominator)
        rel_rate_unc = _relative_uncertainty(reaction_rate_s, reaction_rate_unc_s)
        rel_sigma_unc = _relative_uncertainty(
            effective_cross_section_barn, effective_cross_section_unc_barn
        )
        fluence_rate_unc_cm2_s = float(
            fluence_rate_cm2_s
            * math.sqrt(rel_rate_unc * rel_rate_unc + rel_sigma_unc * rel_sigma_unc)
        )
        fluence_cm2 = float(fluence_rate_cm2_s * fluence_duration_s)
        fluence_unc_cm2 = float(fluence_rate_unc_cm2_s * fluence_duration_s)

        result = {
            "measurement_id": str(row.get("measurement_id") or f"measurement_{index}"),
            "reaction_id": str(
                row.get("reaction_id")
                or row.get("monitor_id")
                or row.get("product_isotope")
                or f"reaction_{index}"
            ),
            "monitor_id": row.get("monitor_id"),
            "product_isotope": row.get("product_isotope"),
            "target_isotope": row.get("target_isotope"),
            "line_energy_keV": row.get("line_energy_keV"),
            "activity_eoi_Bq": float(activity_eoi_bq),
            "activity_eoi_unc_Bq": float(activity_unc_bq),
            "reaction_rate_s": float(reaction_rate_s),
            "reaction_rate_unc_s": float(reaction_rate_unc_s),
            "target_atoms": float(target_atoms),
            "effective_cross_section_barn": float(effective_cross_section_barn),
            "effective_cross_section_unc_barn": float(effective_cross_section_unc_barn),
            "combined_correction_factor": float(correction_factor),
            "fluence_rate_cm2_s": float(fluence_rate_cm2_s),
            "fluence_rate_unc_cm2_s": float(fluence_rate_unc_cm2_s),
            "fluence_cm2": float(fluence_cm2),
            "fluence_unc_cm2": float(fluence_unc_cm2),
            "fluence_duration_s": float(fluence_duration_s),
            "buildup_factor": float(buildup_factor),
        }
        result.update(
            activation_study_metrics(
                activity_bq=float(activity_eoi_bq),
                activity_unc_bq=float(activity_unc_bq),
                half_life_s=float(measurement.half_life_s),
                isotope=str(row.get("product_isotope") or "") or None,
                sample_mass_g=(
                    float(row.get("sample_mass_g"))
                    if row.get("sample_mass_g") is not None
                    else None
                ),
            )
        )
        results.append(result)

    summary = {
        "measurement_count": len(results),
        "fluence_duration_s": float(fluence_duration_s),
        "max_fluence_cm2": float(
            max((row["fluence_cm2"] for row in results), default=0.0)
        ),
        "max_fluence_rate_cm2_s": float(
            max((row["fluence_rate_cm2_s"] for row in results), default=0.0)
        ),
        "reaction_ids": [str(row["reaction_id"]) for row in results],
    }

    return {
        "schema": "fluxforge.astm_e261_result.v1",
        "standard": "ASTM E261",
        "title": plan.get("title") or "ASTM E261 reactor dosimetry workflow",
        "reference": (
            "ASTM E261 Standard Practice for Determining Neutron Fluence, Fluence Rate, and Spectra by Radioactivation Techniques"
        ),
        "inputs": {
            "irradiation": {
                "segments": [
                    {
                        "duration_s": float(segment.duration_s),
                        "relative_power": float(segment.relative_power),
                    }
                    for segment in segments
                ],
                "fluence_duration_s": float(fluence_duration_s),
            },
            "measurement_count": len(measurement_rows),
        },
        "summary": summary,
        "measurements": results,
    }


__all__ = [
    "AstmE261MonitorResult",
    "analyze_astm_e261_plan",
    "equivalent_irradiation_duration_s",
    "target_atom_count",
]
