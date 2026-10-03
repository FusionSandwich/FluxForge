"""Independent analytic regression probes for unresolved issues #197/#204.

These intentionally fail against baseline 1148e18 and are scratch-only.
"""

import math

import pytest

from fluxforge.analysis.flux_unfold import (
    calculate_n_atoms,
    extract_reactions_from_processed,
)
from fluxforge.io.flux_wire import FluxWireData, NuclideResult
from fluxforge.physics.sigphi import (
    IrradiationHistory,
    MonitorMeasurement,
    calculate_saturation_rate,
)


def test_sigphi_single_reduced_power_matches_known_production_rate():
    """A constant half-power exposure must include its relative power."""
    rate_at_reference_power = 2.5e-13
    n_atoms = 1e19
    half_life_s, duration_s, relative_power = 12.701 * 3600, 7200.0, 0.5
    buildup = relative_power * -math.expm1(-math.log(2) * duration_s / half_life_s)
    activity_eoi = n_atoms * rate_at_reference_power * buildup
    result = calculate_saturation_rate(
        MonitorMeasurement(
            "Cu-63(n,g)Cu-64",
            activity_eoi,
            activity_eoi * 0.05,
            half_life_s,
            n_atoms,
            IrradiationHistory([(duration_s, relative_power)]),
        )
    )
    assert result.R_sat == pytest.approx(rate_at_reference_power, rel=1e-12, abs=0)
    assert result.uncertainty == pytest.approx(
        rate_at_reference_power * 0.05, rel=1e-12, abs=0
    )


def test_unknown_element_is_a_descriptive_validation_error():
    """Unknown atomic weights must fail validation before division."""
    with pytest.raises(ValueError, match="atomic mass|element"):
        calculate_n_atoms("Xx", mass_mg=4.0)


def test_processed_report_cannot_normalize_foreign_product_to_monitor_atoms():
    """Sc-46 in a Co monitor cannot silently become an Sc-45 monitor rate."""
    report = FluxWireData(sample_id="Co-RAFM-1")
    report.real_time, report.live_time = 3600.0, 3500.0
    report.nuclides.append(NuclideResult("Sc46", 83.79, "d", 100.0, 5.0, "Bq"))
    with pytest.raises(ValueError, match="reaction|product|element"):
        extract_reactions_from_processed(
            report,
            sample_mass_mg=4.0661,
            irradiation_time_s=7200.0,
            report_includes_count_decay=True,
        )
