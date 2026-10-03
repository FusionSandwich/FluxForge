"""Known-truth count-clock regressions for issue #198 across shared consumers."""

import math

import pytest

from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan
from fluxforge.physics.activation import GammaLineMeasurement


@pytest.mark.parametrize("dead_fraction", [0.0, 0.1, 0.25, 0.3])
def test_shared_activity_recovers_truth_with_dead_time(dead_fraction):
    activity, real, half_life = 1000.0, 3600.0, 54.29 * 60.0
    live = real * (1.0 - dead_fraction)
    lam = math.log(2) / half_life
    counts = 0.05 * 0.8 * (live / real) * activity * -math.expm1(-lam * real) / lam
    measurement = GammaLineMeasurement(
        counts, live, 0.05, 0.8, half_life, dead_time_fraction=dead_fraction
    )
    assert measurement.activity_at_reference() == pytest.approx(activity, rel=1e-12)


def test_shared_activity_preserves_long_lived_limit():
    line = GammaLineMeasurement(1000.0, 10.0, 0.1, 0.5, 1e25)
    assert line.activity_at_reference() == pytest.approx(2000.0, rel=1e-12)


def test_astm_plan_passes_real_count_time_once():
    activity, real, live, half_life = 1000.0, 3600.0, 2700.0, 3257.4
    lam = math.log(2) / half_life
    counts = 0.05 * 0.8 * live / real * activity * -math.expm1(-lam * real) / lam
    result = analyze_astm_e261_plan(
        {
            "irradiation": {"duration_s": 7200.0},
            "measurements": [
                {
                    "net_counts": counts,
                    "live_time_s": live,
                    "real_time_s": real,
                    "efficiency": 0.05,
                    "gamma_intensity": 0.8,
                    "half_life_s": half_life,
                    "sample_mass_g": 0.01,
                    "atomic_mass_g_mol": 115.0,
                    "effective_cross_section_barn": 1.0,
                }
            ],
        }
    )
    assert result["measurements"][0]["activity_eoi_Bq"] == pytest.approx(
        activity, rel=1e-12
    )


@pytest.mark.parametrize("clock", [0.0, -1.0, float("nan"), float("inf")])
def test_uncorrected_report_cannot_skip_unknown_count_duration(clock):
    from fluxforge.analysis.flux_unfold import _report_count_time
    from fluxforge.examples.rafm_workflow import report_count_real_time_s
    from fluxforge.io.flux_wire import FluxWireData

    report = FluxWireData(sample_id="Co-RAFM-1", real_time=clock)
    with pytest.raises(ValueError, match="real_time"):
        _report_count_time(False, report)
    with pytest.raises(ValueError, match="real_time"):
        report_count_real_time_s(
            {"qg_report_activity_includes_count_decay": False}, report
        )
    assert _report_count_time(True, report) == 0.0


def test_explicit_clock_and_acceptance_must_agree():
    line = GammaLineMeasurement(
        1000.0, 90.0, 0.1, 0.5, 1000.0, dead_time_fraction=0.25, real_time_s=100.0
    )
    with pytest.raises(ValueError, match="inconsistent"):
        line.activity_at_reference()
