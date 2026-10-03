"""Line emission probabilities do not determine inter-isotope count ratios."""

import pytest

from fluxforge.analysis.flux_wire_analysis import GammaLine
from fluxforge.examples.rafm_workflow import prune_generic_targeted_lines


@pytest.mark.parametrize(
    "first_isotope,second_isotope", [("A", "B"), ("", ""), (None, None), (" ", " ")]
)
def test_independent_activities_can_reverse_emission_probability_order(
    first_isotope, second_isotope
):
    # With common live time and efficiency, expected counts scale as activity*I.
    # A larger branching probability cannot suppress a more active species.
    weak_probability = GammaLine(100.0, 0.002, first_isotope)
    strong_probability = GammaLine(104.0, 0.9, second_isotope)
    expected_counts = [
        10000.0 * weak_probability.intensity,
        strong_probability.intensity,
    ]
    assert expected_counts[0] > expected_counts[1]
    assert prune_generic_targeted_lines(
        [weak_probability, strong_probability],
        neighbor_window_keV=5.0,
        intensity_ratio=20.0,
    ) == [weak_probability, strong_probability]


def test_same_known_isotope_still_uses_configured_candidate_heuristic():
    weak = GammaLine(100.0, 0.002, "A")
    strong = GammaLine(104.0, 0.9, "A")
    assert prune_generic_targeted_lines(
        [weak, strong], neighbor_window_keV=5.0, intensity_ratio=20.0
    ) == [strong]
