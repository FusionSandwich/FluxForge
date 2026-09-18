from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fluxforge.core.activity_review import review_spectrum_activation
from fluxforge.core.analysis_workspace import PeakCandidate
from fluxforge.core.inventory_timeline import build_inventory_state_from_payload
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    irradiation_buildup_factor,
    reaction_rate_from_activity,
)


def _constant_efficiency(value: float, relative_uncertainty: float) -> EfficiencyCurve:
    return EfficiencyCurve(
        model_type="empirical",
        parameters={
            "energies": [1.0, 10000.0],
            "efficiencies": [value, value],
            "interpolation": "linear",
        },
        uncertainty_model={"type": "constant", "value": relative_uncertainty},
    )


def _gamma_file(tmp_path) -> str:
    path = tmp_path / "lines.json"
    path.write_text(
        json.dumps(
            [
                {
                    "nuclide": "Co-60",
                    "energy_keV": 100.0,
                    "intensity": 1.0,
                    "intensity_unc": 0.0,
                    "half_life_s": 1.0e12,
                },
                {
                    "nuclide": "Co-60",
                    "energy_keV": 200.0,
                    "intensity": 1.0,
                    "intensity_unc": 0.0,
                    "half_life_s": 1.0e12,
                },
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def _peak(peak_id: str, energy_keV: float, counts: float, uncertainty: float) -> PeakCandidate:
    return PeakCandidate(
        peak_id=peak_id,
        channel=energy_keV,
        energy_keV=energy_keV,
        significance=abs(counts) / uncertainty,
        roi_bounds_keV=(energy_keV - 1.0, energy_keV + 1.0),
        net_counts=counts,
        net_counts_uncertainty=uncertainty,
        fit_quality=1.0,
        status="matched",
        nuclide="Co-60",
    )


def test_dead_time_has_one_constant_rate_correction() -> None:
    measurement = GammaLineMeasurement(
        net_counts=800.0,
        live_time_s=80.0,
        real_time_s=100.0,
        dead_time_fraction=0.2,
        efficiency=0.5,
        gamma_intensity=0.5,
        half_life_s=1.0e30,
    )

    assert measurement.activity_at_reference() == pytest.approx(40.0, rel=1e-12)


def test_decay_during_count_matches_numerical_quadrature() -> None:
    activity_at_reference = 125.0
    cooling_time_s = 4.0
    real_time_s = 8.0
    live_fraction = 0.7
    half_life_s = 6.0
    efficiency = 0.23
    intensity = 0.61
    decay_constant = math.log(2.0) / half_life_s

    grid = np.linspace(0.0, real_time_s, 200_001)
    rate = (
        live_fraction
        * efficiency
        * intensity
        * activity_at_reference
        * np.exp(-decay_constant * (cooling_time_s + grid))
    )
    step = float(grid[1] - grid[0])
    counts = step * float(0.5 * rate[0] + np.sum(rate[1:-1]) + 0.5 * rate[-1])
    measurement = GammaLineMeasurement(
        net_counts=counts,
        live_time_s=live_fraction * real_time_s,
        real_time_s=real_time_s,
        efficiency=efficiency,
        gamma_intensity=intensity,
        half_life_s=half_life_s,
        cooling_time_s=cooling_time_s,
    )

    assert measurement.activity_at_reference() == pytest.approx(
        activity_at_reference, rel=2e-11
    )


def test_timing_inputs_must_be_consistent() -> None:
    measurement = GammaLineMeasurement(
        net_counts=10.0,
        live_time_s=80.0,
        real_time_s=100.0,
        dead_time_fraction=0.1,
        efficiency=1.0,
        gamma_intensity=1.0,
        half_life_s=10.0,
    )

    with pytest.raises(ValueError, match="inconsistent"):
        measurement.activity_at_reference()


def test_activity_review_preserves_signed_counts_and_uncertainty_source(tmp_path) -> None:
    review = review_spectrum_activation(
        (_peak("negative", 100.0, -20.0, 7.0),),
        live_time_s=10.0,
        efficiency_curve=_constant_efficiency(0.5, 0.02),
        source_id="custom_gamma_file",
        custom_gamma_path=_gamma_file(tmp_path),
    )

    line = review.line_results[0]
    assert line.net_counts == -20.0
    assert line.irradiation_time_activity_bq < 0.0
    assert line.irradiation_time_uncertainty_bq > 0.0
    assert line.net_counts_uncertainty_source == "provided"


def test_activity_review_does_not_average_shared_efficiency_error(tmp_path) -> None:
    review = review_spectrum_activation(
        (
            _peak("one", 100.0, 100.0, 10.0),
            _peak("two", 200.0, 100.0, 10.0),
        ),
        live_time_s=100.0,
        efficiency_curve=_constant_efficiency(0.2, 0.1),
        source_id="custom_gamma_file",
        custom_gamma_path=_gamma_file(tmp_path),
    )

    line = review.line_results[0]
    summary = review.isotope_summaries[0]
    assert summary.irradiation_time_shared_efficiency_uncertainty_bq == pytest.approx(
        line.efficiency_uncertainty_bq, rel=1e-12
    )
    assert summary.irradiation_time_independent_uncertainty_bq == pytest.approx(
        line.counting_uncertainty_bq / math.sqrt(2.0), rel=1e-12
    )
    assert summary.irradiation_time_uncertainty_bq == pytest.approx(
        math.hypot(
            summary.irradiation_time_independent_uncertainty_bq,
            summary.irradiation_time_shared_efficiency_uncertainty_bq,
        )
    )


def test_shared_efficiency_covariance_keeps_activity_sign(tmp_path) -> None:
    review = review_spectrum_activation(
        (
            _peak("positive", 100.0, 100.0, 10.0),
            _peak("negative", 200.0, -100.0, 10.0),
        ),
        live_time_s=100.0,
        efficiency_curve=_constant_efficiency(0.2, 0.1),
        source_id="custom_gamma_file",
        custom_gamma_path=_gamma_file(tmp_path),
    )

    summary = review.isotope_summaries[0]
    assert summary.irradiation_time_activity_bq == pytest.approx(0.0, abs=1e-12)
    assert summary.irradiation_time_shared_efficiency_uncertainty_bq == pytest.approx(
        0.0, abs=1e-12
    )


def test_activity_review_rejects_missing_net_area_uncertainty(tmp_path) -> None:
    peak = _peak("missing", 100.0, 20.0, 2.0)
    peak = PeakCandidate(
        **{
            **peak.__dict__,
            "significance": 0.0,
            "net_counts_uncertainty": None,
        }
    )

    with pytest.raises(ValueError, match="needs net-count uncertainty"):
        review_spectrum_activation(
            (peak,),
            live_time_s=10.0,
            efficiency_curve=_constant_efficiency(0.5, 0.02),
            source_id="custom_gamma_file",
            custom_gamma_path=_gamma_file(tmp_path),
        )


def test_inventory_count_end_uses_real_elapsed_time() -> None:
    state = build_inventory_state_from_payload(
        {
            "cooling_time_s": 50.0,
            "live_time_s": 80.0,
            "real_time_s": 100.0,
            "isotope_summaries": [],
        }
    )

    assert state.schedule.count_end_time_s == pytest.approx(150.0)


def test_reaction_rate_uncertainty_requires_activity_uncertainty() -> None:
    segments = (IrradiationSegment(duration_s=30.0),)
    factor = irradiation_buildup_factor(segments, 60.0)

    without_uncertainty = reaction_rate_from_activity(5.0, segments, 60.0)
    with_uncertainty = reaction_rate_from_activity(
        5.0, segments, 60.0, activity_uncertainty=0.4
    )

    assert without_uncertainty.uncertainty is None
    assert with_uncertainty.uncertainty == pytest.approx(0.4 / factor)
