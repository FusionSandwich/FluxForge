"""Independent histogram/validation oracles, deliberately expose audit defects."""

import numpy as np
import pytest

from fluxforge.io.spe import GammaSpectrum
from fluxforge.analysis.spectrum_math import subtract_measured_background


def spectrum(counts, energies, gain):
    counts = np.asarray(counts, dtype=float)
    return GammaSpectrum(
        counts=counts,
        counts_uncertainty=np.sqrt(counts),
        channels=np.arange(counts.size),
        energies=np.asarray(energies),
        calibration={"energy": [float(energies[0]), gain]},
        live_time=10.0,
        real_time=10.0,
    )


def split_case():
    # Source bins [0,1), [1,2), [2,3); target bins split each in half.
    sample = spectrum(np.zeros(6), np.arange(6) * 0.5 + 0.25, 0.5)
    background = spectrum([10, 20, 30], [0.5, 1.5, 2.5], 1.0)
    return subtract_measured_background(
        sample, background, mode="manual", manual_scale=1.0, negative_policy="preserve"
    )


def test_rebin_preserves_integrated_histogram_counts():
    net = split_case()
    assert np.sum(net.counts) == pytest.approx(-60.0)


def test_rebin_matches_independent_bin_overlap_oracle():
    np.testing.assert_allclose(split_case().counts, [-5, -5, -10, -10, -15, -15])


def test_rebin_preserves_full_roi_variance_including_shared_source_bins():
    net = split_case()
    covariance = (
        net.count_covariance_matrix().toarray()
        if hasattr(net, "count_covariance_matrix")
        else np.diag(net.counts_uncertainty**2)
    )
    # Each source count is used exactly once over the union of target bins.
    assert covariance.sum() == pytest.approx(60.0)


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_invalid_background_scale_is_rejected(scale):
    sample = spectrum([5, 8], [0.5, 1.5], 1.0)
    background = spectrum([1, 2], [0.5, 1.5], 1.0)
    with pytest.raises(ValueError):
        subtract_measured_background(
            sample, background, mode="manual", manual_scale=scale
        )


def test_background_requires_measured_positive_normalization_time():
    sample = spectrum([5, 8], [0.5, 1.5], 1.0)
    background = spectrum([1, 2], [0.5, 1.5], 1.0)
    background.live_time = 0.0
    with pytest.raises(ValueError):
        subtract_measured_background(sample, background, mode="live")


def test_background_preserves_explicit_sample_energy_axis():
    sample = spectrum([5, 8], [2.0, 4.0], 2.0)
    background = spectrum([1, 2], [2.0, 4.0], 2.0)
    sample.calibration = {}
    background.calibration = {}
    net = subtract_measured_background(
        sample, background, mode="manual", manual_scale=1.0
    )
    np.testing.assert_array_equal(net.energies, sample.energies)


def test_rebin_keeps_source_covariance_and_full_coverage_at_outer_half_bins():
    sample = spectrum(np.zeros(4), np.arange(4), 1.0)
    background = spectrum([10, 20, 30], [0.5, 1.5, 2.5], 1.0)
    background.counts_covariance = __import__(
        "scipy.sparse", fromlist=["csr_matrix"]
    ).csr_matrix([[10, 2, 0], [2, 20, 3], [0, 3, 30]])
    net = subtract_measured_background(
        sample, background, mode="manual", manual_scale=1
    )
    np.testing.assert_allclose(net.counts, [-5, -15, -25, -15])
    assert net.weighted_counts_variance(np.ones(4)) == pytest.approx(70)


@pytest.mark.parametrize("axis", [[1, 0], [1, 1], [0, float("nan")]])
def test_invalid_target_energy_grid_is_rejected(axis):
    sample = spectrum([1, 2], axis, 1)
    background = spectrum([1, 2], [0, 1], 1)
    with pytest.raises(ValueError, match="increasing"):
        subtract_measured_background(sample, background, mode="manual", manual_scale=1)


def test_signed_rebinned_covariance_survives_workspace_save_and_schema(tmp_path):
    import json
    from importlib.resources import files
    import jsonschema
    from fluxforge.core.workspace_document import WorkspaceDocument, WorkspaceSpectrum

    net = split_case()
    document = WorkspaceDocument(
        spectra=(WorkspaceSpectrum("net", net),), active_spectrum_id="net"
    )
    path = tmp_path / "net_session.json"
    path.write_text(json.dumps(document.to_dict()))
    payload = json.loads(path.read_text())
    schema = json.loads(
        files("fluxforge.resources.schemas")
        .joinpath("workspace_document_v2.schema.json")
        .read_text()
    )
    jsonschema.validate(payload, schema)
    loaded = WorkspaceDocument.from_dict(payload).spectrum_by_id("net").spectrum
    np.testing.assert_array_equal(loaded.counts, net.counts)
    assert loaded.weighted_counts_variance(np.ones(6)) == pytest.approx(60)


def test_workspace_roi_uses_shared_covariance_and_sideband_cross_terms():
    from fluxforge.core.analysis_workspace import analyze_roi_region

    counts = np.array([10, 10, 20, 40, 20, 10, 10], dtype=float)
    covariance = np.eye(7) * 4 + np.ones((7, 7)) * 9
    sample = GammaSpectrum(
        counts=counts,
        counts_covariance=covariance,
        calibration={"energy": [0, 1]},
        live_time=10,
    )
    result = analyze_roi_region(sample, roi_bounds_keV=(2, 4), sideband_width_keV=1)
    # The sidebands are [1,2] and [4,5]. The shared common mode cancels
    # in the net sum, while it remains present in the gross ROI uncertainty.
    weights = np.array([0, -0.75, 0.25, 1, 0.25, -0.75, 0])
    assert result.gross_counts_uncertainty**2 == pytest.approx(93)
    assert result.net_counts_uncertainty**2 == pytest.approx(
        weights @ covariance @ weights
    )
