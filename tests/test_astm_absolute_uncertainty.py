"""Independent count integral derivatives; deliberately no production helpers."""

import math
from types import SimpleNamespace
import pytest
from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan
from fluxforge.analysis.astm_e262 import analyze_astm_e262_plan

T_IRR = 7200.0


def row(prefix="", counts=100.0, sigma=31.0, atoms=2e20):
    return {
        prefix + k: v
        for k, v in dict(
            net_counts=counts,
            net_counts_unc=sigma,
            live_time_s=480.0,
            real_time_s=600.0,
            cooling_time_s=900.0,
            half_life_s=232848.0,
            efficiency=0.04,
            gamma_intensity=0.955,
            target_atoms=atoms,
        ).items()
    }


def activity_gain(data, prefix=""):
    lam = math.log(2) / data[prefix + "half_life_s"]
    real = data[prefix + "real_time_s"]
    integrated = -math.expm1(-lam * real) / (lam * real)
    return math.exp(lam * data[prefix + "cooling_time_s"]) / (
        data[prefix + "efficiency"]
        * data[prefix + "gamma_intensity"]
        * data[prefix + "live_time_s"]
        * integrated
    )


def rate_gain(data, prefix=""):
    buildup = -math.expm1(-math.log(2) * T_IRR / data[prefix + "half_life_s"])
    return activity_gain(data, prefix) / buildup


def plan(data):
    return {"irradiation": {"duration_s": T_IRR}, "measurements": [data]}


@pytest.mark.parametrize("counts", [0.0, 100.0])
def test_e261_declared_uncertainty_absolute_derivative(counts):
    data = row(counts=counts)
    data.pop("target_atoms")
    data.update(
        sample_mass_g=0.02,
        atomic_mass_g_mol=196.96657,
        effective_cross_section_barn=98.65,
    )
    result = analyze_astm_e261_plan(plan(data))["measurements"][0]
    assert result["activity_eoi_unc_Bq"] == pytest.approx(activity_gain(data) * 31)
    denominator = result["target_atoms"] * 98.65e-24
    assert result["fluence_rate_unc_cm2_s"] == pytest.approx(
        rate_gain(data) * 31 / denominator
    )


@pytest.mark.parametrize("cd_counts", [100.0, 600.0])
def test_e262_cd_equal_or_negative_difference_retains_sigma(cd_counts):
    data = row(counts=100.0, atoms=2e20)
    data.update(
        row(
            "cd_",
            counts=cd_counts,
            sigma=53.0,
            atoms=2e20 if cd_counts == 100 else 1e21,
        )
    )
    data.update(sigma_0_barn=98.65)
    result = analyze_astm_e262_plan(plan(data))["measurements"][0]
    expected = math.hypot(
        rate_gain(data) * 31,
        rate_gain(data, "cd_") * 53 * data["target_atoms"] / data["cd_target_atoms"],
    ) / (2e20 * 98.65e-24)
    assert result["equivalent_2200ms_fluence_rate_cm2_s"] == 0
    assert result["equivalent_2200ms_fluence_rate_unc_cm2_s"] == pytest.approx(expected)


def test_e262_zero_unknown_retains_absolute_ratio_sigma():
    data = row("unknown_", counts=0, sigma=31, atoms=2e20)
    data.update(row("standard_", counts=100, sigma=53, atoms=1e21))
    data.update(mode="standard_comparison", known_reference_fluence_rate_cm2_s=1e10)
    result = analyze_astm_e262_plan(plan(data))["measurements"][0]
    assert result["equivalent_2200ms_fluence_rate_cm2_s"] == 0
    assert result["equivalent_2200ms_fluence_rate_unc_cm2_s"] == pytest.approx(
        1e10 * 31 / 100 * 5
    )


def test_e262_standard_nonpoisson_all_three_declared_sources():
    data = row("unknown_", counts=75, sigma=31, atoms=2e20)
    data.update(row("standard_", counts=100, sigma=53, atoms=1e21))
    data.update(
        mode="standard_comparison",
        known_reference_fluence_rate_cm2_s=1e10,
        known_reference_fluence_rate_unc_cm2_s=2e9,
    )
    result = analyze_astm_e262_plan(plan(data))["measurements"][0]
    value = 1e10 * 75 / 100 * 5
    expected = math.hypot(value * 31 / 75, value * 53 / 100, value * 0.2)
    assert result["equivalent_2200ms_fluence_rate_cm2_s"] == pytest.approx(value)
    assert result["equivalent_2200ms_fluence_rate_unc_cm2_s"] == pytest.approx(expected)


def test_e262_radiometric_zero_signal_declared_sigma():
    data = row(counts=0)
    data.update(sigma_0_barn=98.65, sigma_0_unc_barn=10)
    result = analyze_astm_e262_plan(plan(data))["measurements"][0]
    assert result["equivalent_2200ms_fluence_rate_unc_cm2_s"] == pytest.approx(
        rate_gain(data) * 31 / (2e20 * 98.65e-24)
    )


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["net_counts_unc", "sigma_0_unc_barn"])
def test_e262_malformed_uncertainty_rejected(field, invalid):
    data = row()
    data.update(sigma_0_barn=98.65)
    data[field] = invalid
    with pytest.raises(ValueError):
        analyze_astm_e262_plan(plan(data))


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_e261_malformed_uncertainty_rejected(invalid):
    data = row()
    data.pop("target_atoms")
    data.update(
        sample_mass_g=0.02,
        atomic_mass_g_mol=196.96657,
        effective_cross_section_barn=98.65,
        effective_cross_section_unc_barn=invalid,
    )
    with pytest.raises(ValueError):
        analyze_astm_e261_plan(plan(data))


@pytest.mark.parametrize("standard", ["E261", "E262"])
@pytest.mark.parametrize("supplied", [False, True])
def test_count_source_and_conditional_budget_are_explicit(standard, supplied):
    data = row()
    if not supplied:
        data.pop("net_counts_unc")
    if standard == "E261":
        data.pop("target_atoms")
        data.update(
            sample_mass_g=0.02,
            atomic_mass_g_mol=196.96657,
            effective_cross_section_barn=98.65,
        )
        analyze = analyze_astm_e261_plan
    else:
        data.update(sigma_0_barn=98.65)
        analyze = analyze_astm_e262_plan
    result = analyze(plan(data))["measurements"][0]
    budget = result["uncertainty_budget"]
    assert budget["uncertainty_scope"] == "conditional"
    assert budget["scientific_admission"] is False
    assert budget["cross_section_uncertainty_supplied"] is False
    assert "target_inventory" in budget["omitted_components"]
    assert "efficiency" in budget["omitted_components"]
    count = budget["count"]
    assert count["assumed"] is (not supplied)
    assert count["scientific_admission"] is False
    if not supplied:
        assert "background uncertainty unavailable" in count["source"]
        assert result["activity_eoi_unc_Bq"] == pytest.approx(activity_gain(data) * 10)


def test_cd_negative_unconstrained_estimate_not_hidden():
    data = row(counts=100, atoms=2e20)
    data.update(row("cd_", counts=600, sigma=53, atoms=1e21))
    data.update(sigma_0_barn=98.65)
    result = analyze_astm_e262_plan(plan(data))["measurements"][0]
    assert result["thermal_rate_clipped"] is True
    assert result["unconstrained_thermal_reaction_rate_s"] == pytest.approx(
        -20 * rate_gain(data)
    )
    assert result["cd_count_uncertainty"]["assumed"] is False
    assert (
        "not a censored posterior"
        in result["uncertainty_budget"]["clipping_interpretation"]
    )


def test_standard_counts_do_not_inherit_unknown_sigma():
    data = row("unknown_", counts=75, sigma=31, atoms=2e20)
    data.update(row("standard_", counts=100, sigma=53, atoms=1e21))
    data.pop("standard_net_counts_unc")
    data.update(mode="standard_comparison", known_reference_fluence_rate_cm2_s=1e10)
    result = analyze_astm_e262_plan(plan(data))["measurements"][0]
    budget = result["uncertainty_budget"]
    assert budget["unknown_count"]["assumed"] is False
    assert budget["standard_count"]["assumed"] is True
    assert budget["reference_uncertainty_supplied"] is False
    value = 1e10 * 75 / 100 * 5
    assert result["equivalent_2200ms_fluence_rate_unc_cm2_s"] == pytest.approx(
        math.hypot(value * 31 / 75, value * 10 / 100)
    )


@pytest.mark.parametrize("declared", [False, True])
def test_library_cross_section_uncertainty_is_proxy_or_declared(monkeypatch, declared):
    import fluxforge.analysis.astm_e262 as module

    monkeypatch.setattr(
        module,
        "get_k0_library_record",
        lambda _: SimpleNamespace(sigma_0_barn=98.65, k0_unc_percent=5.0),
    )
    data = row()
    if declared:
        data["sigma_0_unc_barn"] = 7.0
    result = analyze_astm_e262_plan(plan(data))["measurements"][0]
    assert result["sigma_0_unc_barn"] == pytest.approx(
        7.0 if declared else 98.65 * 0.05
    )
    source = result["uncertainty_budget"]["cross_section_uncertainty_source"]
    if declared:
        assert source == "supplied"
    else:
        assert "proxy" in source and "not a qualified" in source
    assert result["uncertainty_budget"]["scientific_admission"] is False
