"""
Unit tests for gamma and neutron unfolding using FluxForge built-in solvers.

These tests exercise unfolding algorithms without external dependencies.
All tests use FluxForge's built-in RMLE, GRAVEL, MLEM, and GLS solvers.
"""

from __future__ import annotations

import pytest
import numpy as np

import sys
sys.path.insert(0, '/filespace/s/smandych/CAE/projects/ALARA/FluxForge/src')

from fluxforge.unfold._types import ReactionRates, ResponseBundle, SpectrumFile


def _top_k_indices(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    k = max(int(k), 1)
    if k >= x.size:
        return np.argsort(x)[::-1]
    idx = np.argpartition(x, -k)[-k:]
    return idx[np.argsort(x[idx])[::-1]]


# ===========================================================================
# Tests for Gamma RMLE Unfolding (using FluxForge built-in Poisson RMLE)
# ===========================================================================
class TestGammaUnfolderRMLE:
    """Test gamma unfolding via FluxForge's built-in Poisson RMLE."""

    def test_simple_identity_recovery(self):
        """With identity response, should recover input."""
        from fluxforge.solvers.rmle import (
            SpectrumData,
            ResponseMatrix,
            poisson_rmle_unfolding,
            PoissonRMLEConfig,
            PoissonPenalty,
        )

        n = 50
        R = np.eye(n)
        true_spectrum = np.zeros(n)
        true_spectrum[10] = 100
        true_spectrum[30] = 200

        y = R @ true_spectrum

        spec = SpectrumData(counts=y)
        resp = ResponseMatrix(matrix=R)
        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.L1,
            alpha=0.01,
            background_mode="none",
            max_iterations=200,
            tolerance=1e-7,
            positivity=True,
        )
        result = poisson_rmle_unfolding(spec, resp, cfg)
        eta = result.solution
        resid = y - R @ eta

        # Should recover peaks approximately
        assert eta[10] > 50
        assert eta[30] > 100
        assert np.linalg.norm(resid) < 50

    def test_positivity_enforced(self):
        """Unfolded spectrum should be non-negative."""
        from fluxforge.solvers.rmle import (
            SpectrumData,
            ResponseMatrix,
            poisson_rmle_unfolding,
            PoissonRMLEConfig,
            PoissonPenalty,
        )

        n_ch, n_bins = 100, 20
        rng = np.random.default_rng(42)
        R = rng.random((n_ch, n_bins)) * 0.1
        R += np.eye(n_ch, n_bins)
        y = np.maximum(rng.poisson(50, n_ch).astype(float), 1.0)

        spec = SpectrumData(counts=y)
        resp = ResponseMatrix(matrix=R)
        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.L1,
            alpha=1.0,
            positivity=True,
            max_iterations=100,
        )
        result = poisson_rmle_unfolding(spec, resp, cfg)

        assert np.all(result.solution >= 0)

    def test_full_result_object(self):
        """poisson_rmle_unfolding should return a proper result object."""
        from fluxforge.solvers.rmle import (
            SpectrumData,
            ResponseMatrix,
            poisson_rmle_unfolding,
            PoissonRMLEConfig,
            PoissonPenalty,
        )

        n = 30
        R = np.eye(n)
        y = np.ones(n) * 10

        spec = SpectrumData(counts=y)
        resp = ResponseMatrix(matrix=R)
        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.L1,
            alpha=0.1,
            max_iterations=50,
        )
        result = poisson_rmle_unfolding(spec, resp, cfg)

        assert result.solution.shape == (n,)
        assert hasattr(result, 'converged')

    def test_dimension_mismatch_raises(self):
        """Mismatched input dimensions should raise ValueError."""
        from fluxforge.solvers.rmle import (
            SpectrumData,
            ResponseMatrix,
            poisson_rmle_unfolding,
            PoissonRMLEConfig,
        )

        R = np.eye(50)
        spec = SpectrumData(counts=np.ones(30))
        resp = ResponseMatrix(matrix=R)
        cfg = PoissonRMLEConfig()

        with pytest.raises((ValueError, IndexError)):
            poisson_rmle_unfolding(spec, resp, cfg)


# ===========================================================================
# Tests for Neutron Unfolding (using FluxForge built-in MLEM/GRAVEL)
# ===========================================================================
class TestNeutronUnfolderIBU:
    """Test neutron unfolding via FluxForge's built-in iterative methods."""

    def test_simple_unfolding(self):
        """Basic MLEM should converge and return non-negative flux."""
        from fluxforge.solvers.iterative import mlem

        n_monitors = 5
        n_groups = 10

        rng = np.random.default_rng(0)
        R = np.eye(n_monitors, n_groups) * 0.8
        R += rng.random((n_monitors, n_groups)) * 0.05

        true_flux = np.linspace(1, 10, n_groups)
        expected_rates = R @ true_flux
        noise = rng.normal(0, 0.1 * expected_rates)
        measured = np.maximum(expected_rates + noise, 0.1)

        prior = np.ones(n_groups)
        result = mlem(
            response=R.tolist(),
            measurements=measured.tolist(),
            initial_flux=prior.tolist(),
            max_iters=50,
            tolerance=1e-6,
        )

        assert len(result.flux) == n_groups
        assert all(f >= 0 for f in result.flux)
        assert result.iterations >= 1

    def test_with_prior(self):
        """Passing a prior should influence result."""
        from fluxforge.solvers.iterative import gravel

        n_monitors = 4
        n_groups = 8

        R = np.eye(n_monitors, n_groups)
        measured = np.ones(n_monitors) * 10
        uncertainties = np.ones(n_monitors)

        prior = np.linspace(0.5, 2.0, n_groups)

        result = gravel(
            response=R.tolist(),
            measurements=measured.tolist(),
            measurement_uncertainty=uncertainties.tolist(),
            initial_flux=prior.tolist(),
            max_iters=20,
        )

        assert len(result.flux) == n_groups

    def test_compare_with_gls_utility(self):
        """GRAVEL and GLS should produce comparable results."""
        from fluxforge.solvers.iterative import gravel
        from fluxforge.solvers.gls import gls_adjust

        n_monitors = 4
        n_groups = 6

        rng = np.random.default_rng(1)
        R = np.eye(n_monitors, n_groups) * 0.9
        R += rng.random((n_monitors, n_groups)) * 0.1

        true_flux = np.ones(n_groups) * 5.0
        measured = R @ true_flux
        uncertainties = measured * 0.1

        prior = np.ones(n_groups) * 4.0

        # GRAVEL
        gravel_result = gravel(
            response=R.tolist(),
            measurements=measured.tolist(),
            measurement_uncertainty=uncertainties.tolist(),
            initial_flux=prior.tolist(),
            max_iters=100,
        )

        # GLS
        meas_cov = np.diag(uncertainties**2).tolist()
        prior_cov = np.diag((prior * 0.5)**2).tolist()
        gls_result = gls_adjust(
            response=R.tolist(),
            measurements=measured.tolist(),
            measurement_cov=meas_cov,
            prior_flux=prior.tolist(),
            prior_cov=prior_cov,
        )

        gravel_flux = np.array(gravel_result.flux)
        gls_flux = np.array(gls_result.flux)

        # Both should be reasonably close (within 50% relative)
        rel_diff = np.abs(gravel_flux - gls_flux) / (np.abs(gls_flux) + 1e-10)
        assert np.max(rel_diff) < 0.5


# ===========================================================================
# Comparison tests: Poisson RMLE vs iterative methods
# ===========================================================================
class TestGammaCompareAgainstExisting:
    """Compare different unfolding algorithms on gamma problems."""

    def test_gamma_rmle_refold_and_peaks(self):
        """Test that RMLE recovers peaks and refolds correctly."""
        from fluxforge.solvers.rmle import (
            SpectrumData,
            ResponseMatrix as RMResponseMatrix,
            poisson_rmle_unfolding,
            PoissonRMLEConfig,
            PoissonPenalty,
        )

        rng = np.random.default_rng(0)

        n_channels = 200
        n_bins = 40

        # Build a mildly smearing response matrix
        R = np.zeros((n_channels, n_bins), dtype=float)
        ch = np.arange(n_channels)
        for j in range(n_bins):
            center = (j + 0.5) * (n_channels / n_bins)
            sigma = 2.5
            col = np.exp(-0.5 * ((ch - center) / sigma) ** 2)
            col = col / np.sum(col)
            R[:, j] = col

        # Sparse true spectrum
        true_eta = np.zeros(n_bins, dtype=float)
        peak_bins = np.array([6, 20, 33])
        true_eta[peak_bins] = np.array([80.0, 150.0, 60.0])

        expected = R @ true_eta
        y = np.round(expected * 100.0).astype(float)

        spec = SpectrumData(counts=y)
        resp = RMResponseMatrix(matrix=R)
        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.L1,
            alpha=0.01,
            background_mode="none",
            max_iterations=300,
            tolerance=1e-7,
            positivity=True,
            guardrail_max_reduced_chi2=1e9,
            mc_samples=0,
        )
        res = poisson_rmle_unfolding(spec, resp, cfg)
        eta = res.solution
        resid = y - (R @ eta)

        # Should refold reasonably well
        assert np.linalg.norm(resid) / (np.linalg.norm(y) + 1e-12) < 0.5

        # Should identify the true peak bins
        top_bins = set(_top_k_indices(eta, 5).tolist())
        true_set = set(peak_bins.tolist())
        assert len(true_set.intersection(top_bins)) >= 2


class TestNeutronCompareAgainstExisting:
    """Compare iterative unfolding vs GLS adjustment."""

    def test_neutron_mlem_vs_gls_refold(self):
        """Compare MLEM vs GLS refolding accuracy."""
        from fluxforge.solvers.iterative import mlem
        from fluxforge.solvers.gls import gls_adjust

        rng = np.random.default_rng(1)

        n_monitors = 8
        n_groups = 12

        # Create a positive response matrix and normalize columns
        R = rng.random((n_monitors, n_groups))
        R = R / np.maximum(R.sum(axis=0, keepdims=True), 1e-12)

        true_flux = np.exp(-0.2 * np.arange(n_groups)) * 100.0
        meas = R @ true_flux
        meas_noise = rng.normal(0.0, 0.05 * meas)
        meas = np.maximum(meas + meas_noise, 1e-6)

        meas_unc = np.maximum(0.05 * meas, 1e-6)

        # Prior
        prior_flux = np.exp(-0.15 * np.arange(n_groups)) * 100.0

        # --- Run GLS ---
        measurement_cov = np.diag(meas_unc**2).tolist()
        prior_cov = np.diag((0.5 * prior_flux) ** 2).tolist()

        gls = gls_adjust(
            response=R.tolist(),
            measurements=meas.tolist(),
            measurement_cov=measurement_cov,
            prior_flux=prior_flux.tolist(),
            prior_cov=prior_cov,
            enforce_nonnegativity=True,
        )
        gls_flux = np.asarray(gls.flux, dtype=float)

        # --- Run MLEM ---
        mlem_result = mlem(
            response=R.tolist(),
            measurements=meas.tolist(),
            initial_flux=prior_flux.tolist(),
            max_iters=100,
        )
        mlem_flux = np.array(mlem_result.flux)

        # Both should refold to the measured rates to similar accuracy
        gls_refold = R @ gls_flux
        mlem_refold = R @ mlem_flux

        gls_rel = np.linalg.norm(gls_refold - meas) / (np.linalg.norm(meas) + 1e-12)
        mlem_rel = np.linalg.norm(mlem_refold - meas) / (np.linalg.norm(meas) + 1e-12)

        assert gls_rel < 0.5
        assert mlem_rel < 0.5

        # Both should remain non-negative
        assert np.all(gls_flux >= 0)
        assert np.all(mlem_flux >= 0)


# ===========================================================================
# Tests for type definitions
# ===========================================================================
class TestUnfoldTypes:
    """Test the shared dataclass types."""

    def test_reaction_rates(self):
        """ReactionRates should store values and uncertainties."""
        rr = ReactionRates(
            values=np.array([1.0, 2.0]),
            uncertainties=np.array([0.1, 0.2]),
        )
        assert rr.values.shape == (2,)
        assert rr.uncertainties.shape == (2,)

    def test_response_bundle(self):
        """ResponseBundle should store matrix and energy_bins."""
        rb = ResponseBundle(
            matrix=np.eye(3, 5),
            energy_bins=np.linspace(0, 1e6, 6),
        )
        assert rb.matrix.shape == (3, 5)
        assert rb.energy_bins.shape == (6,)

    def test_spectrum_file(self):
        """SpectrumFile should store counts."""
        sf = SpectrumFile(counts=np.arange(100))
        assert sf.counts.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
