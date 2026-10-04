"""Diagnostic tooling must preserve independent energy and detector behavior."""

import importlib.util
import sys
from pathlib import Path

import pytest

from fluxforge.analysis.flux_wire_analysis import (
    GammaLine,
    analyze_raw_spectrum_targeted,
)
from fluxforge.examples.rafm_workflow import (
    build_generic_gamma_library,
    default_paths,
    load_rafm_example_metadata,
    workflow_profile_energy_calibration,
)
from fluxforge.io.flux_wire import read_raw_asc


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def review():
    spec = importlib.util.spec_from_file_location(
        "sensitivity_review", ROOT / "tools/review_qg_sensitivity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_centroid_cannot_shift_the_independent_library_target(review):
    row = {"reported_isotope": "Mn56", "reported_energy_keV": 2532.86}
    library = [GammaLine(2523.06, 0.0103, "Mn56"), GammaLine(2657.56, 0.0066, "Mn56")]
    line = review.nearest_library_line(row, library)
    assert line.energy_keV == 2523.06
    assert row["reported_energy_keV"] == 2532.86


def test_missing_library_identity_is_not_fabricated(review):
    with pytest.raises(ValueError, match="No independent library isotope"):
        review.nearest_library_line(
            {"reported_isotope": "fictional", "reported_energy_keV": 100}, []
        )


def test_diagnostic_tracing_does_not_change_detector_results_or_trace_state(review):
    example = ROOT / "examples/RAFM_irradiation"
    metadata = load_rafm_example_metadata(example)
    paths = default_paths(example)
    calibration = workflow_profile_energy_calibration(metadata.config)
    data = read_raw_asc(
        paths.raw_root / "RAFM4/RAFM4-A_15dEOI.ASC",
        energy_calibration_override=calibration,
        profile_name=metadata.config["profile_name"],
    )
    background = read_raw_asc(
        paths.background_path,
        energy_calibration_override=calibration,
        profile_name=metadata.config["profile_name"],
    ).spectrum
    library, _ = build_generic_gamma_library(metadata)
    line = next(line for line in library if line.isotope == "Cr51")
    options = dict(
        peak_threshold=2.0,
        background_spectrum=background,
        profile_name=metadata.config["profile_name"],
        counting_method="iec_tiered",
        max_assignment_energy_delta_fwhm=1.0,
    )
    before = sys.gettrace()
    detected, candidates, diagnostics = review.trace_targeted_call(
        data, [line], options
    )
    assert sys.gettrace() is before
    independent = analyze_raw_spectrum_targeted(data, [line], **options)
    assert len(detected) == len(independent) == 1
    for field in ("channel", "energy_keV", "net_counts", "net_counts_unc", "isotope"):
        assert getattr(detected[0], field) == getattr(independent[0], field)
    assert not candidates
    assert len(diagnostics) == 1
    assert diagnostics[0]["net_unc"] >= diagnostics[0]["roi_unc"]
    assert diagnostics[0]["net_unc"] >= diagnostics[0]["fit_unc"]
