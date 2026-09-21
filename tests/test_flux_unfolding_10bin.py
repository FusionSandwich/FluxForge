"""Integration test for 10-bin flux unfolding regression (raw vs processed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pytest

from fluxforge.io.reader_factory import read_spectrum_any
from fluxforge.analysis.spectrum_math import subtract_measured_background


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_ROOT = Path(__file__).resolve().parent / "data"

PROC_DIR = TEST_DATA_ROOT / "flux_wires" / "processed"
RAW_DIR = TEST_DATA_ROOT / "flux_wires" / "raw"
MODEL_PATH = RAW_DIR / "spectrum_vit_j.csv"


@pytest.mark.parametrize('sample_path', sorted(p for p in RAW_DIR.glob('*.ASC') if p.name.lower() != 'background.asc'), ids=lambda p:p.stem)
def test_inl_background_counts_and_integral_variance_independent_oracle(sample_path):
    sample = read_spectrum_any(sample_path)
    background = read_spectrum_any(RAW_DIR/'background.ASC')
    before = sample.to_dict(), background.to_dict()
    def boundaries(centers):
        return np.concatenate(([centers[0]-(centers[1]-centers[0])/2],
                               (centers[:-1]+centers[1:])/2,
                               [centers[-1]+(centers[-1]-centers[-2])/2]))
    src, dst = boundaries(background.energies), boundaries(sample.energies)
    assert src[0] <= dst[0] and src[-1] >= dst[-1]
    # Integrate piecewise-uniform density using an independently interpolated
    # cumulative histogram, not the production overlap-matrix implementation.
    cumulative = np.r_[0., np.cumsum(background.counts)]
    expected_background = np.diff(np.interp(dst,src,cumulative))
    scale = sample.live_time/background.live_time
    actual = subtract_measured_background(sample,background,negative_policy='preserve')
    np.testing.assert_allclose(actual.counts,sample.counts-scale*expected_background,atol=1e-7,rtol=1e-10)
    retained = np.maximum(0,np.minimum(src[1:],dst[-1])-np.maximum(src[:-1],dst[0])) / np.diff(src)
    expected_variance = np.sum(sample.counts_uncertainty**2) + scale**2*np.sum((retained*background.counts_uncertainty)**2)
    assert actual.linear_variance(np.ones(len(sample.counts))) == pytest.approx(expected_variance,rel=1e-10)
    assert actual.metadata['background_subtraction']['rebin']['discarded_source_counts'] == pytest.approx(background.counts @ (1-retained),abs=1e-7)
    assert actual.counts_covariance.nnz > len(actual.counts)
    # A named Co-60-region adjacent-bin covariance oracle.
    index = int(np.argmin(abs(sample.energies-1173.2)))
    def overlap_weights(i):
        return np.maximum(0,np.minimum(src[1:],dst[i+1])-np.maximum(src[:-1],dst[i]))/np.diff(src)
    expected_cross = scale**2*np.sum(overlap_weights(index)*overlap_weights(index+1)*background.counts_uncertainty**2)
    assert actual.counts_covariance[index,index+1] == pytest.approx(expected_cross,abs=1e-10)
    assert (sample.to_dict(),background.to_dict()) == before


def load_regression_module():
    script_path = (
        REPO_ROOT / "examples" / "validation" / "flux_unfolding_10bin_regression.py"
    )
    spec = importlib.util.spec_from_file_location(
        "flux_unfolding_10bin_regression", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flux_unfolding_10bin_regression(tmp_path):
    artifact_dir = tmp_path / "flux_unfolding_10bin"
    assert PROC_DIR.exists(), f"Processed directory missing: {PROC_DIR}"
    assert RAW_DIR.exists(), f"Raw directory missing: {RAW_DIR}"

    module = load_regression_module()
    results = module.run_unfolding_regression(PROC_DIR, RAW_DIR, MODEL_PATH)

    assert len(results.processed_discrete.flux) == 10
    assert len(results.raw_discrete.flux) == 10
    assert len(results.processed_gls.flux) == 50
    assert len(results.raw_gls.flux) == 50

    artifact_dir.mkdir(parents=True, exist_ok=True)
    module.save_spectrum_csv(
        artifact_dir / "processed_discrete.csv",
        results.processed_discrete.energy_bounds_eV,
        results.processed_discrete.flux,
        results.processed_discrete.flux_unc,
    )
    module.save_spectrum_csv(
        artifact_dir / "raw_discrete.csv",
        results.raw_discrete.energy_bounds_eV,
        results.raw_discrete.flux,
        results.raw_discrete.flux_unc,
    )
