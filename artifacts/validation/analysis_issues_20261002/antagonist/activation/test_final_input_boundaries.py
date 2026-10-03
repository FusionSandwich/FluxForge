"""Independent explicit factor and cross-section validation probes."""

import pytest

from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan
from fluxforge.analysis.astm_e262 import analyze_astm_e262_plan


def _plan(**overrides):
    row = {
        "reaction_id": "Co-59(n,g)Co-60", "product_isotope": "Co60",
        "net_counts": 1000.0, "live_time_s": 1000.0, "half_life_s": 166337000.0,
        "efficiency": 0.05, "gamma_intensity": 0.9998,
        "sample_mass_g": 0.010, "atomic_mass_g_mol": 58.9332,
        "sigma_0_barn": 37.18, "effective_cross_section_barn": 37.18,
        **overrides,
    }
    return {"irradiation": {"duration_s": 7200.0}, "measurements": [row]}


@pytest.mark.parametrize("field", ["astm_correction_factor", "self_shielding_factor", "cover_correction_factor", "geometry_factor", "effective_cross_section_barn"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_e261_invalid_explicit_correction_or_cross_section_rejected(field, value):
    with pytest.raises(ValueError):
        analyze_astm_e261_plan(_plan(**{field: value}))


@pytest.mark.parametrize("field", ["westcott_g", "thermal_self_shielding_factor", "sigma_0_barn"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_e262_invalid_explicit_correction_or_cross_section_rejected(field, value):
    with pytest.raises(ValueError):
        analyze_astm_e262_plan(_plan(**{field: value}))
