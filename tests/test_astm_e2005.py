"""Tests for ASTM E2005 benchmark dosimetry standard."""

import pytest
import math

from fluxforge.analysis.astm_e2005 import (
    calculate_fluence_rate_transfer,
    calculate_spectral_index,
    evaluate_spectral_index_double_ratio,
    analyze_astm_e2005_plan,
)


def test_calculate_fluence_rate_transfer():
    # phi_a = phi_b * (r_a / r_b) * (sigma_b / sigma_a)
    # phi_b = 100, ra = 10, rb = 5, sig_a = 2, sig_b = 4
    # phi_a = 100 * (10 / 5) * (4 / 2) = 100 * 2 * 2 = 400
    phi_a, phi_a_unc = calculate_fluence_rate_transfer(
        phi_b=100.0,
        phi_b_unc=10.0,  # 10%
        rate_a=10.0,
        rate_a_unc=0.0,
        rate_b=5.0,
        rate_b_unc=0.0,
        sigma_a=2.0,
        sigma_a_unc=0.0,
        sigma_b=4.0,
        sigma_b_unc=0.0,
    )
    assert math.isclose(phi_a, 400.0)
    assert math.isclose(
        phi_a_unc, 40.0
    )  # 10% uncertainty propagates directly if others 0


def test_calculate_spectral_index():
    # SI = ra / rb
    si, si_unc = calculate_spectral_index(
        rate_a=10.0, rate_a_unc=1.0, rate_b=5.0, rate_b_unc=0.0  # 10%  # 0%
    )
    assert math.isclose(si, 2.0)
    # 2.0 * sqrt(0.1^2 + 0) = 0.2
    assert math.isclose(si_unc, 0.2)


def test_analyze_astm_e2005_plan():
    plan = {
        "title": "Test Plan",
        "fluence_transfers": [
            {
                "transfer_id": "ni_transfer",
                "field_a": {"reaction_rate_s": 10.0, "cross_section_barn": 2.0},
                "field_b": {
                    "fluence_rate_cm2_s": 100.0,
                    "reaction_rate_s": 5.0,
                    "cross_section_barn": 4.0,
                },
            }
        ],
        "spectral_indices": [
            {
                "index_id": "si_1",
                "measured": {"reaction_rate_a_s": 10.0, "reaction_rate_b_s": 5.0},
                "calculated": {"index": 2.5},
            }
        ],
    }
    res = analyze_astm_e2005_plan(plan)
    assert len(res["fluence_transfers"]) == 1
    assert res["fluence_transfers"][0]["fluence_rate_cm2_s"] == 400.0

    assert len(res["spectral_indices"]) == 1
    si = res["spectral_indices"][0]
    assert si["measured_index"] == 2.0
    assert si["calculated_index"] == 2.5
    assert si["c_e_ratio"] == 1.25


def test_analyze_astm_e2005_plan_uses_direct_indices_and_uncertainties():
    plan = {
        "spectral_indices": [
            {
                "index_id": "direct_idx",
                "measured": {"index": 2.0, "index_unc": 0.1},
                "calculated": {"index": 2.2, "index_unc": 0.11},
            }
        ]
    }

    res = analyze_astm_e2005_plan(plan)
    row = res["spectral_indices"][0]
    assert math.isclose(row["measured_index"], 2.0)
    assert math.isclose(row["calculated_index"], 2.2)
    assert math.isclose(row["c_e_ratio"], 1.1)
    expected_unc = 1.1 * math.sqrt((0.11 / 2.2) ** 2 + (0.1 / 2.0) ** 2)
    assert math.isclose(row["c_e_ratio_unc"], expected_unc)


def test_fluence_rate_transfer_rejects_zero_benchmark_rate():
    with pytest.raises(ValueError):
        calculate_fluence_rate_transfer(
            phi_b=100.0,
            rate_a=10.0,
            rate_b=0.0,
            sigma_a=1.0,
            sigma_b=1.0,
        )


def test_double_ratio_rejects_zero_measured_index():
    with pytest.raises(ValueError):
        evaluate_spectral_index_double_ratio(si_cal=1.0, si_meas=0.0)
