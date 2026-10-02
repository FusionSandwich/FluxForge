"""Material inventory must not manufacture target atoms (issue #204)."""

import pytest

from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan, target_atom_count
from fluxforge.analysis.flux_unfold import calculate_n_atoms


@pytest.mark.parametrize("element", ["Unobtainium", "coo", "Co-Al"])
def test_unknown_element_has_clear_inventory_error(element):
    with pytest.raises(ValueError, match="atomic mass|element"):
        calculate_n_atoms(element, 10.0)


@pytest.mark.parametrize(
    "field",
    [
        "sample_mass_g",
        "atomic_mass_g_mol",
        "isotopic_abundance",
        "mass_fraction",
        "sample_purity",
        "atoms_per_formula_unit",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_target_inventory_rejects_invalid_input(field, value):
    args = {"sample_mass_g": 0.01, "atomic_mass_g_mol": 58.933}
    args[field] = value
    with pytest.raises(ValueError, match=field):
        target_atom_count(**args)


@pytest.mark.parametrize(
    "field", ["isotopic_abundance", "mass_fraction", "sample_purity"]
)
def test_zero_plan_fraction_is_not_replaced_with_pure_material(field):
    row = {
        "net_counts": 1000.0,
        "live_time_s": 100.0,
        "efficiency": 0.05,
        "gamma_intensity": 0.8,
        "half_life_s": 1000.0,
        "sample_mass_g": 0.01,
        "atomic_mass_g_mol": 58.933,
        "effective_cross_section_barn": 1.0,
        field: 0.0,
    }
    with pytest.raises(ValueError, match=field):
        analyze_astm_e261_plan(
            {"irradiation": {"duration_s": 7200.0}, "measurements": [row]}
        )
