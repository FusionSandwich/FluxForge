"""Weak associations remain review candidates, separate from detected activity."""

from pathlib import Path

import pytest

from fluxforge.analysis.flux_wire_analysis import analyze_raw_spectrum_targeted
from fluxforge.examples.rafm_workflow import (
    build_generic_gamma_library,
    default_paths,
    load_rafm_example_metadata,
    workflow_profile_energy_calibration,
)
from fluxforge.io.flux_wire import read_raw_asc


ROOT = Path(__file__).resolve().parents[1] / "examples/RAFM_irradiation"


@pytest.fixture(scope="module")
def inputs():
    metadata = load_rafm_example_metadata(ROOT)
    paths = default_paths(ROOT)
    calibration = workflow_profile_energy_calibration(metadata.config)
    library, _ = build_generic_gamma_library(metadata)
    background = read_raw_asc(
        paths.background_path,
        energy_calibration_override=calibration,
        profile_name=metadata.config["profile_name"],
    ).spectrum
    return metadata, paths, calibration, library, background


def extract(inputs, group, sample, isotope, energy, threshold=2):
    metadata, paths, calibration, library, background = inputs
    data = read_raw_asc(
        paths.raw_root / group / (sample + ".ASC"),
        energy_calibration_override=calibration,
        profile_name=metadata.config["profile_name"],
    )
    line = min(
        (line for line in library if line.isotope == isotope),
        key=lambda line: abs(line.energy_keV - energy),
    )
    candidates = []
    detected = analyze_raw_spectrum_targeted(
        data,
        [line],
        peak_threshold=threshold,
        min_energy_keV=80,
        background_spectrum=background,
        counting_method="iec_tiered",
        profile_name=metadata.config["profile_name"],
        low_significance_candidates=candidates,
    )
    return detected, candidates, line


@pytest.mark.parametrize(
    "sample,isotope,energy",
    [
        ("RAFM3-C_300sEOI", "W187", 772.91),
        ("RAFM3-C_4dEOI", "Fe59", 1099.25),
        ("RAFM3-N_24hrEOI", "Mn56", 2113.09),
        ("RAFM3-N_4dEOI", "Fe59", 1099.25),
    ],
)
def test_bundled_weak_candidates_do_not_enter_detected_activity(
    inputs, sample, isotope, energy
):
    detected, candidates, line = extract(inputs, "RAFM3", sample, isotope, energy)
    assert detected == []
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.isotope == isotope
    assert abs(candidate.energy_keV - line.energy_keV) <= 2
    assert 0 < candidate.significance < 2
    assert candidate.net_counts_unc > candidate.net_counts / 2
    assert candidate.activity_bq == 0  # Activity conversion only sees detections.


def test_strong_peak_is_not_reclassified_as_tentative(inputs):
    detected, candidates, _ = extract(
        inputs, "RAFM4", "RAFM4-A_15dEOI", "Cr51", 320.0835
    )
    assert len(detected) == 1
    assert detected[0].significance >= 2
    assert detected[0].activity_bq > 0
    assert candidates == []


@pytest.mark.parametrize(
    "sample,isotope,energy",
    [
        ("RAFM4-N_15dEOI", "Tb154m", 247.94),
        ("RAFM4-C_15dEOI", "W187", 206.25),
    ],
)
def test_tentative_collection_does_not_bypass_energy_support(
    inputs, sample, isotope, energy
):
    detected, candidates, _ = extract(inputs, "RAFM4", sample, isotope, energy)
    assert detected == []
    assert candidates == []
