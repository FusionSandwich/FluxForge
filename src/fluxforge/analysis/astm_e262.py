"""ASTM E262 thermal neutron fluence-rate workflow helpers.

This module implements a practical ASTM E262-oriented workflow for
thermal-neutron fluence-rate determination with activation monitors.
The implementation supports two common measurement modes:

1. Radiometric mode (counting-based), optionally using cadmium-covered
   companion measurements to remove epithermal contribution.
2. Standard-comparison mode, where an unknown field is compared against a
   known reference fluence-rate field.

The reported primary quantity is the equivalent 2200 m/s thermal fluence
rate, following the Stoughton and Halperin convention used by ASTM E262.
"""

from __future__ import annotations

import math
from fluxforge.data.k0_library import get_k0_library_record
from typing import Any, Sequence

from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    activation_study_metrics,
    irradiation_buildup_factor,
)
from fluxforge.physics.neutron_corrections import (
    calculate_cd_ratio,
    extract_thermal_epithermal_components,
)
from fluxforge.data.k0_library import get_k0_library_record


BARN_TO_CM2 = 1.0e-24
T0_K = 293.4


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


def equivalent_irradiation_duration_s(segments: Sequence[IrradiationSegment]) -> float:
    """Return the power-weighted irradiation duration used for fluence totals."""

    return float(
        sum(segment.duration_s * segment.relative_power for segment in segments)
    )


def _measurement_from_row(
    row: dict[str, Any], prefix: str = ""
) -> GammaLineMeasurement:
    key = lambda name: f"{prefix}{name}" if prefix else name
    return GammaLineMeasurement(
        net_counts=float(row.get(key("net_counts"), 0.0)),
        live_time_s=float(row.get(key("live_time_s"), 0.0)),
        efficiency=float(row.get(key("efficiency"), row.get("efficiency", 0.0))),
        gamma_intensity=float(
            row.get(
                key("gamma_intensity"),
                row.get(
                    key("emission_probability"),
                    row.get("gamma_intensity", row.get("emission_probability", 0.0)),
                ),
            )
        ),
        half_life_s=float(row.get(key("half_life_s"), row.get("half_life_s", 0.0))),
        cooling_time_s=float(
            row.get(
                key("cooling_time_s"),
                row.get(key("decay_time_s"), row.get("cooling_time_s", 0.0)),
            )
            or 0.0
        ),
        dead_time_fraction=float(
            row.get(key("dead_time_fraction"), row.get("dead_time_fraction", 0.0))
            or 0.0
        ),
    )


def _reaction_rate_from_measurement(
    measurement: GammaLineMeasurement, segments: Sequence[IrradiationSegment]
) -> tuple[float, float, float, float]:
    activity_eoi_bq = measurement.activity_at_reference()
    activity_unc_bq = float(
        activity_eoi_bq / math.sqrt(max(measurement.net_counts, 1.0))
    )
    buildup_factor = irradiation_buildup_factor(segments, measurement.half_life_s)
    if buildup_factor <= 0.0:
        raise ValueError(
            "Irradiation buildup factor must be positive for ASTM E262 analysis."
        )
    reaction_rate_s = float(activity_eoi_bq / buildup_factor)
    reaction_rate_unc_s = float(activity_unc_bq / buildup_factor)
    return reaction_rate_s, reaction_rate_unc_s, activity_eoi_bq, activity_unc_bq


def _true_thermal_from_equivalent(
    phi0_cm2_s: float, neutron_temperature_K: float
) -> float:
    if phi0_cm2_s <= 0.0:
        return 0.0
    if neutron_temperature_K <= 0.0:
        return phi0_cm2_s
    return float(phi0_cm2_s * math.sqrt(neutron_temperature_K / T0_K))


def _analyze_radiometric_measurement(
    row: dict[str, Any],
    *,
    segments: Sequence[IrradiationSegment],
    fluence_duration_s: float,
    default_neutron_temperature_K: float,
) -> dict[str, Any]:
    measurement = _measurement_from_row(row)
    reaction_rate_s, reaction_rate_unc_s, activity_eoi_bq, activity_unc_bq = (
        _reaction_rate_from_measurement(measurement, segments)
    )

    sigma_0_barn = float(
        row.get("sigma_0_barn", row.get("thermal_cross_section_barn", 0.0))
    )
    sigma_0_unc_barn = float(
        row.get("sigma_0_unc_barn", row.get("thermal_cross_section_unc_barn", 0.0))
        or 0.0
    )

    # Fallback to governed k0 library if cross-section is not manually specified
    if sigma_0_barn <= 0.0:
        iso_raw = row.get("measurement_id", "")
        if "reaction_id" in row and ")" in row["reaction_id"]:
            iso_raw = row["reaction_id"].split(")")[-1]

        iso_fmt = (
            iso_raw[:-2] + "-" + iso_raw[-2:]
            if len(iso_raw) > 2 and iso_raw[-2:].isdigit() and "-" not in iso_raw
            else iso_raw
        )
        record = get_k0_library_record(iso_fmt)
        if record and record.sigma_0_barn > 0.0:
            sigma_0_barn = float(record.sigma_0_barn)
            if float(record.k0_unc_percent) > 0.0:
                sigma_0_unc_barn = sigma_0_barn * (record.k0_unc_percent / 100.0)

    westcott_g = float(row.get("westcott_g", 1.0) or 1.0)
    thermal_self_shielding_factor = float(
        row.get("thermal_self_shielding_factor", row.get("G_th", 1.0)) or 1.0
    )

    denominator = (
        sigma_0_barn * BARN_TO_CM2 * westcott_g * thermal_self_shielding_factor
    )
    if denominator <= 0.0:
        raise ValueError(
            "ASTM E262 radiometric mode requires positive sigma_0_barn, Westcott g, and thermal self-shielding factor."
        )

    thermal_rate_s = reaction_rate_s
    thermal_rate_unc_s = reaction_rate_unc_s
    cadmium_payload: dict[str, Any] = {}

    has_cd_pair = any(
        key in row
        for key in (
            "cd_net_counts",
            "cd_live_time_s",
            "cd_efficiency",
            "cd_gamma_intensity",
        )
    )
    if has_cd_pair:
        cd_measurement = _measurement_from_row(row, prefix="cd_")
        (
            cd_reaction_rate_s,
            cd_reaction_rate_unc_s,
            cd_activity_bq,
            cd_activity_unc_bq,
        ) = _reaction_rate_from_measurement(cd_measurement, segments)
        cd_transmission_factor = float(
            row.get(
                "cd_transmission_factor", row.get("cadmium_transmission_factor", 1.0)
            )
            or 1.0
        )
        thermal_rate_s = max(
            reaction_rate_s - cd_transmission_factor * cd_reaction_rate_s, 0.0
        )
        thermal_rate_unc_s = float(
            math.sqrt(
                reaction_rate_unc_s * reaction_rate_unc_s
                + (cd_transmission_factor * cd_reaction_rate_unc_s)
                * (cd_transmission_factor * cd_reaction_rate_unc_s)
            )
        )

        cd_ratio, cd_ratio_unc = calculate_cd_ratio(
            activity_bare=activity_eoi_bq,
            activity_cd_covered=cd_activity_bq,
            uncertainty_bare=activity_unc_bq,
            uncertainty_covered=cd_activity_unc_bq,
        )
        cd_components = extract_thermal_epithermal_components(
            activity_bare=activity_eoi_bq,
            activity_cd_covered=cd_activity_bq,
            reaction=str(row.get("reaction_id") or ""),
            cd_thickness=float(row.get("cd_thickness_mm", 1.0) or 1.0),
        )
        cadmium_payload = {
            "cd_activity_eoi_Bq": float(cd_activity_bq),
            "cd_activity_eoi_unc_Bq": float(cd_activity_unc_bq),
            "cd_reaction_rate_s": float(cd_reaction_rate_s),
            "cd_reaction_rate_unc_s": float(cd_reaction_rate_unc_s),
            "cd_transmission_factor": float(cd_transmission_factor),
            "cadmium_ratio": float(cd_ratio),
            "cadmium_ratio_unc": float(cd_ratio_unc),
            "cd_correction_factor": float(cd_components.F_Cd),
            "thermal_fraction": float(cd_components.thermal_fraction),
            "epithermal_fraction": float(cd_components.epithermal_fraction),
            "effective_resonance_integral_barn": float(cd_components.I_eff),
        }

    phi0_eq_cm2_s = float(thermal_rate_s / denominator)
    rel_rate_unc = _relative_uncertainty(thermal_rate_s, thermal_rate_unc_s)
    rel_sigma_unc = _relative_uncertainty(sigma_0_barn, sigma_0_unc_barn)
    phi0_eq_unc_cm2_s = float(
        phi0_eq_cm2_s
        * math.sqrt(rel_rate_unc * rel_rate_unc + rel_sigma_unc * rel_sigma_unc)
    )

    neutron_temperature_K = float(
        row.get("neutron_temperature_K", default_neutron_temperature_K)
        or default_neutron_temperature_K
    )
    true_thermal_fluence_rate_cm2_s = _true_thermal_from_equivalent(
        phi0_eq_cm2_s, neutron_temperature_K
    )
    true_thermal_fluence_rate_unc_cm2_s = _true_thermal_from_equivalent(
        phi0_eq_unc_cm2_s, neutron_temperature_K
    )

    result = {
        "mode": "radiometric",
        "activity_eoi_Bq": float(activity_eoi_bq),
        "activity_eoi_unc_Bq": float(activity_unc_bq),
        "reaction_rate_s": float(reaction_rate_s),
        "reaction_rate_unc_s": float(reaction_rate_unc_s),
        "thermal_reaction_rate_s": float(thermal_rate_s),
        "thermal_reaction_rate_unc_s": float(thermal_rate_unc_s),
        "sigma_0_barn": float(sigma_0_barn),
        "sigma_0_unc_barn": float(sigma_0_unc_barn),
        "westcott_g": float(westcott_g),
        "thermal_self_shielding_factor": float(thermal_self_shielding_factor),
        "equivalent_2200ms_fluence_rate_cm2_s": float(phi0_eq_cm2_s),
        "equivalent_2200ms_fluence_rate_unc_cm2_s": float(phi0_eq_unc_cm2_s),
        "equivalent_2200ms_fluence_cm2": float(phi0_eq_cm2_s * fluence_duration_s),
        "equivalent_2200ms_fluence_unc_cm2": float(
            phi0_eq_unc_cm2_s * fluence_duration_s
        ),
        "true_thermal_fluence_rate_cm2_s": float(true_thermal_fluence_rate_cm2_s),
        "true_thermal_fluence_rate_unc_cm2_s": float(
            true_thermal_fluence_rate_unc_cm2_s
        ),
        "true_thermal_fluence_cm2": float(
            true_thermal_fluence_rate_cm2_s * fluence_duration_s
        ),
        "true_thermal_fluence_unc_cm2": float(
            true_thermal_fluence_rate_unc_cm2_s * fluence_duration_s
        ),
        "neutron_temperature_K": float(neutron_temperature_K),
        "fluence_duration_s": float(fluence_duration_s),
    }
    result.update(cadmium_payload)
    return result


def _analyze_standard_comparison_measurement(
    row: dict[str, Any],
    *,
    segments: Sequence[IrradiationSegment],
    fluence_duration_s: float,
    default_neutron_temperature_K: float,
) -> dict[str, Any]:
    unknown_measurement = _measurement_from_row(row, prefix="unknown_")
    standard_measurement = _measurement_from_row(row, prefix="standard_")
    unknown_rate_s, unknown_rate_unc_s, unknown_activity_bq, unknown_activity_unc_bq = (
        _reaction_rate_from_measurement(unknown_measurement, segments)
    )
    (
        standard_rate_s,
        standard_rate_unc_s,
        standard_activity_bq,
        standard_activity_unc_bq,
    ) = _reaction_rate_from_measurement(standard_measurement, segments)

    known_phi0_cm2_s = float(
        row.get(
            "known_reference_fluence_rate_cm2_s",
            row.get("known_fluence_rate_cm2_s", 0.0),
        )
    )
    known_phi0_unc_cm2_s = float(
        row.get(
            "known_reference_fluence_rate_unc_cm2_s",
            row.get("known_fluence_rate_unc_cm2_s", 0.0),
        )
        or 0.0
    )
    if known_phi0_cm2_s <= 0.0:
        raise ValueError(
            "ASTM E262 standard-comparison mode requires known_reference_fluence_rate_cm2_s > 0."
        )
    if standard_rate_s <= 0.0:
        raise ValueError(
            "ASTM E262 standard-comparison mode requires a positive standard reaction rate."
        )

    spectral_correction_factor = float(
        row.get("spectral_correction_factor", 1.0) or 1.0
    )
    if spectral_correction_factor <= 0.0:
        raise ValueError(
            "ASTM E262 standard-comparison mode requires spectral_correction_factor > 0."
        )

    phi0_eq_cm2_s = float(
        known_phi0_cm2_s
        * (unknown_rate_s / standard_rate_s)
        * spectral_correction_factor
    )
    rel_unc = math.sqrt(
        _relative_uncertainty(known_phi0_cm2_s, known_phi0_unc_cm2_s) ** 2
        + _relative_uncertainty(unknown_rate_s, unknown_rate_unc_s) ** 2
        + _relative_uncertainty(standard_rate_s, standard_rate_unc_s) ** 2
    )
    phi0_eq_unc_cm2_s = float(abs(phi0_eq_cm2_s) * rel_unc)

    neutron_temperature_K = float(
        row.get("neutron_temperature_K", default_neutron_temperature_K)
        or default_neutron_temperature_K
    )
    true_thermal_fluence_rate_cm2_s = _true_thermal_from_equivalent(
        phi0_eq_cm2_s, neutron_temperature_K
    )
    true_thermal_fluence_rate_unc_cm2_s = _true_thermal_from_equivalent(
        phi0_eq_unc_cm2_s, neutron_temperature_K
    )

    return {
        "mode": "standard_comparison",
        "known_reference_fluence_rate_cm2_s": float(known_phi0_cm2_s),
        "known_reference_fluence_rate_unc_cm2_s": float(known_phi0_unc_cm2_s),
        "spectral_correction_factor": float(spectral_correction_factor),
        "unknown_activity_eoi_Bq": float(unknown_activity_bq),
        "unknown_activity_eoi_unc_Bq": float(unknown_activity_unc_bq),
        "standard_activity_eoi_Bq": float(standard_activity_bq),
        "standard_activity_eoi_unc_Bq": float(standard_activity_unc_bq),
        "unknown_reaction_rate_s": float(unknown_rate_s),
        "unknown_reaction_rate_unc_s": float(unknown_rate_unc_s),
        "standard_reaction_rate_s": float(standard_rate_s),
        "standard_reaction_rate_unc_s": float(standard_rate_unc_s),
        "equivalent_2200ms_fluence_rate_cm2_s": float(phi0_eq_cm2_s),
        "equivalent_2200ms_fluence_rate_unc_cm2_s": float(phi0_eq_unc_cm2_s),
        "equivalent_2200ms_fluence_cm2": float(phi0_eq_cm2_s * fluence_duration_s),
        "equivalent_2200ms_fluence_unc_cm2": float(
            phi0_eq_unc_cm2_s * fluence_duration_s
        ),
        "true_thermal_fluence_rate_cm2_s": float(true_thermal_fluence_rate_cm2_s),
        "true_thermal_fluence_rate_unc_cm2_s": float(
            true_thermal_fluence_rate_unc_cm2_s
        ),
        "true_thermal_fluence_cm2": float(
            true_thermal_fluence_rate_cm2_s * fluence_duration_s
        ),
        "true_thermal_fluence_unc_cm2": float(
            true_thermal_fluence_rate_unc_cm2_s * fluence_duration_s
        ),
        "neutron_temperature_K": float(neutron_temperature_K),
        "fluence_duration_s": float(fluence_duration_s),
    }


def analyze_astm_e262_plan(plan: dict[str, Any]) -> dict[str, Any]:
    r"""Execute an ASTM E262 thermal-neutron fluence-rate plan.

    Expected top-level fields:

    - `irradiation.segments` or `irradiation.duration_s`
    - `measurements`: list of monitor entries

    Per measurement, set `mode` to either `radiometric` or
    `standard_comparison` (default: `radiometric`).
    """

    segments = _coerce_segments(plan)
    if not segments:
        raise ValueError(
            "ASTM E262 workflow requires a positive irradiation duration or segment list."
        )

    measurement_rows = list(plan.get("measurements") or [])
    if not measurement_rows:
        raise ValueError("ASTM E262 workflow requires at least one measurement entry.")

    fluence_duration_s = float(
        (plan.get("irradiation") or {}).get(
            "fluence_duration_s", equivalent_irradiation_duration_s(segments)
        )
    )
    default_neutron_temperature_K = float(
        plan.get("neutron_temperature_K", T0_K) or T0_K
    )

    results: list[dict[str, Any]] = []
    for index, row in enumerate(measurement_rows, start=1):
        mode = str(row.get("mode", "radiometric")).strip().lower()
        if mode == "standard_comparison":
            calc = _analyze_standard_comparison_measurement(
                row,
                segments=segments,
                fluence_duration_s=fluence_duration_s,
                default_neutron_temperature_K=default_neutron_temperature_K,
            )
        else:
            calc = _analyze_radiometric_measurement(
                row,
                segments=segments,
                fluence_duration_s=fluence_duration_s,
                default_neutron_temperature_K=default_neutron_temperature_K,
            )

        result = {
            "measurement_id": str(row.get("measurement_id") or f"measurement_{index}"),
            "reaction_id": str(
                row.get("reaction_id")
                or row.get("monitor_id")
                or row.get("product_isotope")
                or f"reaction_{index}"
            ),
            "monitor_id": row.get("monitor_id"),
            "target_isotope": row.get("target_isotope"),
            "product_isotope": row.get("product_isotope"),
            "line_energy_keV": row.get("line_energy_keV"),
        }
        result.update(calc)
        if (
            "activity_eoi_Bq" in result
            and "activity_eoi_unc_Bq" in result
            and row.get("half_life_s") is not None
        ):
            result.update(
                activation_study_metrics(
                    activity_bq=float(result["activity_eoi_Bq"]),
                    activity_unc_bq=float(result["activity_eoi_unc_Bq"]),
                    half_life_s=float(row.get("half_life_s") or 0.0),
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
        "max_equivalent_2200ms_fluence_cm2": float(
            max((row["equivalent_2200ms_fluence_cm2"] for row in results), default=0.0)
        ),
        "max_equivalent_2200ms_fluence_rate_cm2_s": float(
            max(
                (row["equivalent_2200ms_fluence_rate_cm2_s"] for row in results),
                default=0.0,
            )
        ),
        "reaction_ids": [str(row["reaction_id"]) for row in results],
    }

    return {
        "schema": "fluxforge.astm_e262_result.v1",
        "standard": "ASTM E262",
        "title": plan.get("title") or "ASTM E262 thermal neutron fluence-rate workflow",
        "reference": "ASTM E262 Test Method for Determining Thermal Neutron Reaction Rates and Equivalent 2200 m/s Fluence Rate by Radioactivation Techniques",
        "convention": "Stoughton and Halperin",
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
            "neutron_temperature_K": float(default_neutron_temperature_K),
            "measurement_count": len(measurement_rows),
        },
        "summary": summary,
        "measurements": results,
    }


__all__ = [
    "analyze_astm_e262_plan",
    "equivalent_irradiation_duration_s",
]
