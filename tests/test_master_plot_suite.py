from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

from fluxforge.io.artifacts import (
    write_reaction_rates,
    write_response_bundle,
    write_unfold_result,
)
from fluxforge.plots.master_suite import (
    generate_master_plan_plots,
    load_example_plot_inputs,
    load_plot_inputs_from_artifacts,
)


def test_master_plot_suite_example_data(tmp_path):
    inputs = load_example_plot_inputs()
    out_dir = tmp_path / "example_plots"
    produced = generate_master_plan_plots(inputs, out_dir, formats=("png",))

    expected_keys = {
        "g1_spectrum_uncertainty",
        "g1_prior_posterior_overlay",
        "g1_residuals_pulls",
        "g1_covariance_correlation",
        "g1_parity",
        "response_matrix",
    }
    assert expected_keys.issubset(produced.keys())
    for paths in produced.values():
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 0


def test_master_plot_suite_artifact_inputs(tmp_path):
    boundaries = [1e-5, 1e2, 1e5]
    reactions = ["r1", "r2"]
    response_matrix = [[0.01, 0.02], [0.03, 0.01]]
    flux = [10.0, 5.0]
    covariance = [[1.0, 0.1], [0.1, 0.5]]

    response_file = tmp_path / "response.json"
    rates_file = tmp_path / "rates.json"
    unfold_file = tmp_path / "unfold.json"
    prior_file = tmp_path / "prior_flux.json"

    write_response_bundle(
        response_file,
        matrix=response_matrix,
        reactions=reactions,
        boundaries_eV=boundaries,
    )
    write_reaction_rates(
        rates_file,
        rates=[
            {
                "reaction_id": "r1",
                "rate": 0.22,
                "uncertainty": 0.01,
                "half_life_s": 1.0,
            },
            {
                "reaction_id": "r2",
                "rate": 0.34,
                "uncertainty": 0.02,
                "half_life_s": 1.0,
            },
        ],
        segments=[{"duration_s": 1.0, "relative_power": 1.0}],
    )
    write_unfold_result(
        unfold_file,
        boundaries_eV=boundaries,
        reactions=reactions,
        flux=flux,
        covariance=covariance,
        chi2=1.2,
        method="gls",
    )
    prior_file.write_text(json.dumps([8.0, 6.0]), encoding="utf-8")

    inputs = load_plot_inputs_from_artifacts(
        unfold_file=unfold_file,
        response_file=response_file,
        rates_file=rates_file,
        prior_flux_file=prior_file,
        validate=True,
    )

    assert np.allclose(inputs.posterior_flux, np.array(flux))
    assert np.allclose(inputs.boundaries_eV, np.array(boundaries))
    assert inputs.reactions == reactions

    out_dir = tmp_path / "artifact_plots"
    produced = generate_master_plan_plots(
        inputs,
        out_dir,
        formats=("png",),
        include_response_plot=False,
    )

    assert "response_matrix" not in produced
    assert len(produced["g1_parity"]) == 1
    assert (out_dir / "g1_parity.png").exists()
