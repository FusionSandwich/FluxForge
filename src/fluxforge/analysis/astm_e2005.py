"""ASTM E2005 reactor dosimetry benchmark testing workflow.

This module implements the ASTM E2005 standard guide for benchmarking
neutron measurements and calculations. It focuses on the primary 
analytical derivations defined in the standard:
1. Fluence Rate Transfer (Eq. 3)
2. Spectral Indexes (Eq. 4)
"""

from __future__ import annotations

import math
from typing import Any


def _relative_uncertainty(value: float, uncertainty: float) -> float:
    if value == 0.0 or uncertainty <= 0.0:
        return 0.0
    return float(uncertainty / abs(value))


def calculate_fluence_rate_transfer(
    *,
    phi_b: float,
    rate_a: float,
    rate_b: float,
    sigma_a: float,
    sigma_b: float,
    phi_b_unc: float = 0.0,
    rate_a_unc: float = 0.0,
    rate_b_unc: float = 0.0,
    sigma_a_unc: float = 0.0,
    sigma_b_unc: float = 0.0,
) -> tuple[float, float]:
    r"""Calculate the transferred fluence rate using a standard field.

    Implements Equation 3:
    $$\phi_A = \phi_B \left(\frac{R_A}{R_B}\right)\left(\frac{\bar{\sigma}_B}{\bar{\sigma}_A}\right)$$
    """
    if rate_b == 0.0 or sigma_a == 0.0:
        raise ValueError("Benchmark reaction rate and Field A cross section must be non-zero.")

    phi_a = phi_b * (rate_a / rate_b) * (sigma_b / sigma_a)

    rel_unc_sq = (
        _relative_uncertainty(phi_b, phi_b_unc) ** 2
        + _relative_uncertainty(rate_a, rate_a_unc) ** 2
        + _relative_uncertainty(rate_b, rate_b_unc) ** 2
        + _relative_uncertainty(sigma_a, sigma_a_unc) ** 2
        + _relative_uncertainty(sigma_b, sigma_b_unc) ** 2
    )

    phi_a_unc = abs(phi_a) * math.sqrt(rel_unc_sq)
    return float(phi_a), float(phi_a_unc)


def calculate_spectral_index(
    rate_a: float, 
    rate_b: float, 
    rate_a_unc: float = 0.0, 
    rate_b_unc: float = 0.0
) -> tuple[float, float]:
    r"""Calculate spectral index (ratio of reaction rates)."""
    if rate_b == 0.0:
        raise ValueError("Denominator reaction rate (rate_b) must be non-zero.")

    index = rate_a / rate_b
    rel_unc_sq = (
        _relative_uncertainty(rate_a, rate_a_unc) ** 2
        + _relative_uncertainty(rate_b, rate_b_unc) ** 2
    )
    index_unc = abs(index) * math.sqrt(rel_unc_sq)
    return float(index), float(index_unc)


def evaluate_spectral_index_double_ratio(
    si_cal: float, 
    si_meas: float, 
    si_cal_unc: float = 0.0, 
    si_meas_unc: float = 0.0
) -> tuple[float, float]:
    r"""Calculate the double ratio (C/E) for a spectral index."""
    if si_meas == 0.0:
        raise ValueError("Measured spectral index must be non-zero.")

    ratio = si_cal / si_meas
    rel_unc_sq = (
        _relative_uncertainty(si_cal, si_cal_unc) ** 2
        + _relative_uncertainty(si_meas, si_meas_unc) ** 2
    )
    ratio_unc = abs(ratio) * math.sqrt(rel_unc_sq)
    return float(ratio), float(ratio_unc)


def analyze_astm_e2005_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute an ASTM E2005 benchmark plan.

    Processes two optional lists:
    - `fluence_transfers`: Calculates target fluence rates based on standard field calibrations.
    - `spectral_indices`: Calculates measured and calculated spectral indexes, and their double ratio.
    """
    results: dict[str, Any] = {
        "title": plan.get("title", "ASTM E2005 Benchmark Analysis"),
        "fluence_transfers": [],
        "spectral_indices": [],
    }

    # Process Fluence Transfers
    for i, ft in enumerate(plan.get("fluence_transfers", [])):
        field_a = ft.get("field_a", {})
        field_b = ft.get("field_b", {})

        phi_b = float(field_b.get("fluence_rate_cm2_s", 0.0))
        phi_b_unc = float(field_b.get("fluence_rate_unc_cm2_s", 0.0))

        ra = float(field_a.get("reaction_rate_s", 0.0))
        ra_unc = float(field_a.get("reaction_rate_unc_s", 0.0))
        sa = float(field_a.get("cross_section_barn", 0.0))
        sa_unc = float(field_a.get("cross_section_unc_barn", 0.0))

        rb = float(field_b.get("reaction_rate_s", 0.0))
        rb_unc = float(field_b.get("reaction_rate_unc_s", 0.0))
        sb = float(field_b.get("cross_section_barn", 0.0))
        sb_unc = float(field_b.get("cross_section_unc_barn", 0.0))

        phi_a, phi_a_unc = calculate_fluence_rate_transfer(
            phi_b=phi_b, phi_b_unc=phi_b_unc,
            rate_a=ra, rate_a_unc=ra_unc,
            rate_b=rb, rate_b_unc=rb_unc,
            sigma_a=sa, sigma_a_unc=sa_unc,
            sigma_b=sb, sigma_b_unc=sb_unc,
        )

        results["fluence_transfers"].append({
            "transfer_id": str(ft.get("transfer_id", f"transfer_{i+1}")),
            "fluence_rate_cm2_s": phi_a,
            "fluence_rate_unc_cm2_s": phi_a_unc,
        })

    # Process Spectral Indices
    for i, si in enumerate(plan.get("spectral_indices", [])):
        meas = si.get("measured", {})
        cal = si.get("calculated", {})

        ra_meas = float(meas.get("reaction_rate_a_s", 0.0))
        ra_meas_unc = float(meas.get("reaction_rate_a_unc_s", 0.0))
        # Default to None or check if present properly. Using 1.0 safely since we'll check later.
        rb_meas = float(meas.get("reaction_rate_b_s", 1.0))
        rb_meas_unc = float(meas.get("reaction_rate_b_unc_s", 0.0))

        ra_cal = float(cal.get("reaction_rate_a_s", 0.0))
        ra_cal_unc = float(cal.get("reaction_rate_a_unc_s", 0.0))
        rb_cal = float(cal.get("reaction_rate_b_s", 1.0))
        rb_cal_unc = float(cal.get("reaction_rate_b_unc_s", 0.0))

        # Prefer direct spectral index values if provided, otherwise compute from reaction rates
        if "index" in meas:
            si_meas = float(meas["index"])
            si_meas_unc = float(meas.get("index_unc", 0.0))
        else:
            if "reaction_rate_b_s" not in meas:
                raise ValueError("Spectral index requires 'index' or 'reaction_rate_b_s'.")
            si_meas, si_meas_unc = calculate_spectral_index(
                ra_meas, rb_meas, ra_meas_unc, rb_meas_unc
            )

        if "index" in cal:
            si_cal = float(cal["index"])
            si_cal_unc = float(cal.get("index_unc", 0.0))
        else:
            if "reaction_rate_b_s" not in cal:
                raise ValueError("Spectral index requires 'index' or 'reaction_rate_b_s'.")
            si_cal, si_cal_unc = calculate_spectral_index(
                ra_cal, rb_cal, ra_cal_unc, rb_cal_unc
            )

        c_e_ratio, c_e_unc = evaluate_spectral_index_double_ratio(
            si_cal, si_meas, si_cal_unc, si_meas_unc
        )

        results["spectral_indices"].append({
            "index_id": str(si.get("index_id", f"index_{i+1}")),
            "measured_index": si_meas,
            "measured_index_unc": si_meas_unc,
            "calculated_index": si_cal,
            "calculated_index_unc": si_cal_unc,
            "c_e_ratio": c_e_ratio,
            "c_e_ratio_unc": c_e_unc,
        })

    return results