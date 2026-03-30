from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import numpy as np

import fluxforge.cli.app as app
from fluxforge.core.unfolding_diagnostics import summarize_flux_bins
from fluxforge.solvers.advanced_unfolding import AdvancedIterativeSolution
from fluxforge.solvers.gls import GLSSolution
from fluxforge.solvers.iterative import IterativeSolution
from fluxforge.solvers.mcmc import MCMCSolution
from fluxforge.solvers.regularized import RegularizedSolution
from fluxforge.solvers.rmle import UnfoldingResult as RMLEUnfoldingResult
from fluxforge.unfold.gamma_rmle import GammaUnfoldResult
from fluxforge.unfold.neutron_ibu import NeutronIBUResult
from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement, SpectrumUnfolder


def test_summarize_flux_bins_reports_negative_bin_summary():
    summary = summarize_flux_bins(
        [1.5, -0.25, 0.0, -0.5],
        negative_tolerance=1e-6,
        negative_policy="preserved_for_review",
        nonnegativity_enforced=False,
    )

    assert summary["negative_bin_count"] == 2
    assert summary["has_negative_bins"] is True
    assert summary["negative_bin_index_preview"] == [1, 3]
    assert summary["negative_bin_min_flux"] == -0.5
    assert summary["negative_policy"] == "preserved_for_review"
    assert summary["nonnegativity_enforced"] is False


def test_result_objects_attach_negative_bin_summaries():
    gls = GLSSolution(
        flux=[1.0, -0.2],
        covariance=[[1.0, 0.0], [0.0, 1.0]],
        residuals=[0.0],
        chi2=1.0,
    )
    iterative = IterativeSolution(
        flux=[1.0, -0.2],
        history=[[1.0, -0.2]],
        iterations=2,
        converged=False,
    )
    rmle = RMLEUnfoldingResult(
        solution=np.array([1.0, -0.2]),
        uncertainty=np.array([0.1, 0.1]),
    )
    regularized = RegularizedSolution(
        spectrum=np.array([1.0, -0.2]),
        uncertainties=np.array([0.1, 0.1]),
        residuals=np.array([0.0, 0.0]),
        chi_squared=1.0,
        regularization_param=0.1,
        n_iterations=3,
        converged=True,
    )
    advanced = AdvancedIterativeSolution(flux=[1.0, -0.2])
    mcmc = MCMCSolution(
        flux=[1.0, -0.2],
        samples=[[1.0, -0.2]],
        credible_lower=[0.8, -0.3],
        credible_upper=[1.2, -0.1],
        credible_median=[1.0, -0.2],
        acceptance_rate=0.5,
    )
    gamma = GammaUnfoldResult(
        unfolded_spectrum=np.array([1.0, -0.2]),
        residuals=np.array([0.0, 0.0]),
    )
    ibu = NeutronIBUResult(
        unfolded_flux=np.array([1.0, -0.2]),
        statistical_uncertainty=np.array([0.1, 0.1]),
        systematic_uncertainty=np.array([0.1, 0.1]),
    )

    assert gls.diagnostics["negative_bin_count"] == 1
    assert iterative.diagnostics["negative_bin_count"] == 1
    assert rmle.diagnostics["negative_bin_count"] == 1
    assert regularized.details["negative_bin_count"] == 1
    assert advanced.diagnostics["negative_bin_count"] == 1
    assert mcmc.diagnostics["negative_bin_count"] == 1
    assert gamma.diagnostics["negative_bin_count"] == 1
    assert ibu.diagnostics["negative_bin_count"] == 1


def test_cmd_unfold_reports_negative_bins_in_artifact_payload(monkeypatch, tmp_path):
    unfold_written: dict[str, object] = {}

    monkeypatch.setattr(
        app,
        "read_response_bundle",
        lambda _: {
            "matrix": [[1.0, 0.0], [0.0, 1.0]],
            "boundaries_eV": [1e-5, 1.0, 1e3],
            "reactions": ["r1", "r2"],
        },
    )
    monkeypatch.setattr(
        app,
        "read_reaction_rates",
        lambda _: {
            "rates": [
                {"rate": 1.0, "uncertainty": 0.1},
                {"rate": 0.5, "uncertainty": 0.1},
            ]
        },
    )
    monkeypatch.setattr(
        app,
        "gls_adjust",
        lambda *args, **kwargs: SimpleNamespace(
            flux=[1.0, -0.25],
            covariance=[[0.0, 0.0], [0.0, 0.0]],
            chi2=0.5,
            reduced_chi2=0.5,
            n_dof=1,
            diagnostics={},
            pull=None,
            prior_posterior_change=None,
        ),
    )
    monkeypatch.setattr(
        app,
        "write_unfold_result",
        lambda output, **kwargs: unfold_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    app.cmd_unfold(
        Namespace(
            rates_file=tmp_path / "rates.json",
            response_file=tmp_path / "response.json",
            prior_flux_file=None,
            method="gls",
            prior_uncertainty=0.25,
            prior_cov_model="diagonal",
            prior_correlation_length=1.0,
            max_iters=25,
            tolerance=1e-4,
            chi2_tolerance=0.01,
            relaxation=0.7,
            floor=1e-20,
            convergence_mode="relative",
            enforce_nonnegativity=False,
            verbose_solver=False,
            output=tmp_path / "unfold_gls.json",
            validate=False,
        )
    )

    diagnostics = unfold_written["payload"]["diagnostics"]
    assert diagnostics["negative_bin_count"] == 1
    assert diagnostics["has_negative_bins"] is True
    assert diagnostics["min_flux"] == -0.25
    assert diagnostics["predicted_rates"] == [1.0, -0.25]


def test_spectrum_unfolder_metadata_carries_negative_bin_summary(monkeypatch):
    def fake_build_response_matrix(self):
        return np.array([[1.0, 0.0], [0.0, 1.0]]), ["rx1", "rx2"], np.zeros(2)

    def fake_gravel(*args, **kwargs):
        return IterativeSolution(
            flux=[1.0, -0.15],
            history=[[1.0, -0.15]],
            iterations=4,
            converged=False,
            chi_squared=0.2,
            chi_squared_history=[0.2],
            final_residuals=[0.0, 0.0],
            diagnostics={
                "negative_policy": "preserved_for_review",
                "nonnegativity_enforced": False,
            },
        )

    monkeypatch.setattr(SpectrumUnfolder, "_build_response_matrix", fake_build_response_matrix)
    monkeypatch.setattr(
        "fluxforge.workflows.spectrum_unfolding.IRDFFDatabase",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "fluxforge.workflows.spectrum_unfolding.gravel",
        fake_gravel,
    )

    unfolder = SpectrumUnfolder(custom_energy_edges=np.array([1.0, 10.0, 100.0]), verbose=False)
    unfolder.measurements = [
        FluxWireMeasurement("rx1", activity_Bq=1.0, uncertainty_Bq=0.1),
        FluxWireMeasurement("rx2", activity_Bq=0.8, uncertainty_Bq=0.1),
    ]

    result = unfolder.unfold(method="GRAVEL", max_iterations=10)

    assert result.metadata["negative_bin_count"] == 1
    assert result.metadata["has_negative_bins"] is True
    assert result.metadata["negative_policy"] == "preserved_for_review"
    assert result.metadata["nonnegativity_enforced"] is False
