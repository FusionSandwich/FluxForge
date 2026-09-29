"""Decay during counting uses clock time and is never applied twice (issue #198)."""

from __future__ import annotations

import math

import pytest

from fluxforge.analysis.flux_unfold import activity_to_reaction_rate, extract_reactions_from_processed
from fluxforge.examples.rafm_workflow import decay_correction_factor, report_count_real_time_s
from fluxforge.io.flux_wire import FluxWireData, NuclideResult

IN116M_HALF_LIFE_S = 54.29 * 60.0
IN115M_HALF_LIFE_S = 16149.6  # FLUX_WIRE_NUCLIDES value


def test_count_average_activity_recovers_count_start_with_real_time() -> None:
    lam = math.log(2) / IN116M_HALF_LIFE_S
    a0 = 1000.0
    real, live = 3600.0, 2700.0  # 25% dead time
    # Uniform acceptance: counts = eps*y*(live/real) * a0 * (1-exp(-lam*real))/lam;
    # dividing by eps*y*live gives the count-averaged activity over real time.
    count_average = a0 * (-math.expm1(-lam * real)) / (lam * real)
    assert decay_correction_factor(IN116M_HALF_LIFE_S, real, 0.0) * count_average == pytest.approx(a0)
    biased = decay_correction_factor(IN116M_HALF_LIFE_S, live, 0.0) * count_average
    assert abs(biased / a0 - 1) > 0.05  # ~8% low with live time


def test_rate_conversion_rejects_live_time_and_uses_real_time() -> None:
    with pytest.raises(ValueError, match="count_real_time_s"):
        activity_to_reaction_rate(100.0, 1e18, IN116M_HALF_LIFE_S, 3600.0, live_time_s=2700.0)
    lam = math.log(2) / IN116M_HALF_LIFE_S
    start = activity_to_reaction_rate(100.0, 1e18, IN116M_HALF_LIFE_S, 3600.0)
    averaged = activity_to_reaction_rate(
        100.0, 1e18, IN116M_HALF_LIFE_S, 3600.0, count_real_time_s=3600.0
    )
    assert averaged / start == pytest.approx(lam * 3600.0 / -math.expm1(-lam * 3600.0))


def _report() -> FluxWireData:
    data = FluxWireData(sample_id="In-Cd-RAFM-1")
    data.live_time, data.real_time = 2700.0, 3600.0
    data.nuclides.append(NuclideResult("In115m", 4.486, "h", 0.01, 0.0005, "uCi"))
    return data


def test_report_activity_requires_declaration() -> None:
    with pytest.raises(ValueError, match="report_includes_count_decay"):
        extract_reactions_from_processed(_report(), sample_mass_mg=17.364, irradiation_time_s=7200.0)
    corrected = extract_reactions_from_processed(
        _report(), sample_mass_mg=17.364, irradiation_time_s=7200.0,
        report_includes_count_decay=True,
    )[0]
    uncorrected = extract_reactions_from_processed(
        _report(), sample_mass_mg=17.364, irradiation_time_s=7200.0,
        report_includes_count_decay=False,
    )[0]
    lam = math.log(2) / IN115M_HALF_LIFE_S
    assert corrected.reaction_rate > 0
    assert uncorrected.reaction_rate / corrected.reaction_rate == pytest.approx(
        lam * 3600.0 / -math.expm1(-lam * 3600.0), rel=1e-9
    )


def test_workflow_report_count_time_is_declared() -> None:
    report = _report()
    with pytest.raises(ValueError, match="qg_report_activity_includes_count_decay"):
        report_count_real_time_s({}, report)
    assert report_count_real_time_s({"qg_report_activity_includes_count_decay": True}, report) == 0.0
    assert report_count_real_time_s({"qg_report_activity_includes_count_decay": False}, report) == 3600.0
