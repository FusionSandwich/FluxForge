"""Synthetic blanks/injections constrain any proposed sideband tuning."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def study():
    path = Path(__file__).resolve().parents[1] / "tools/check_roi_sensitivity.py"
    spec = importlib.util.spec_from_file_location("roi_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate(trials=4000)


@pytest.mark.parametrize(
    "variant",
    ["baseline", "wider_sidebands", "narrower_roi", "narrower_roi_wider_sidebands"],
)
def test_fixed_window_uncertainties_cover_flat_poisson_blanks(study, variant):
    result = study["results"]["0.0"][variant]
    assert result["detection_fraction"] < 0.04
    assert (
        0.90
        < result["average_uncertainty"] / result["empirical_standard_deviation"]
        < 1.10
    )
    assert 0.62 < result["coverage_within_1_sigma"] < 0.74


def test_wider_clean_sidebands_improve_injection_recovery_without_dropping_variance(
    study,
):
    weak = study["results"]["200.0"]
    assert (
        weak["narrower_roi_wider_sidebands"]["detection_fraction"]
        > weak["baseline"]["detection_fraction"] + 0.3
    )
    assert weak["narrower_roi_wider_sidebands"]["peak_capture_fraction"] > 0.99
    for variant in (
        "baseline",
        "wider_sidebands",
        "narrower_roi",
        "narrower_roi_wider_sidebands",
    ):
        assert study["results"]["10000.0"][variant]["detection_fraction"] == 1.0


def test_best_window_selection_is_not_a_calibrated_two_sigma_detection(study):
    blank = study["results"]["0.0"]
    assert blank["choose_best_of_four"]["detection_fraction"] > max(
        blank[name]["detection_fraction"]
        for name in (
            "baseline",
            "wider_sidebands",
            "narrower_roi",
            "narrower_roi_wider_sidebands",
        )
    )
    assert study["scientific_admission"] is False
