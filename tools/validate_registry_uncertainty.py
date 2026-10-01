"""Offline, hash-bound registry replay; unchanged rates/estimates, unavailable U."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np

from fluxforge.cli.app import _solve_unfold_method, _build_unfold_diagnostics
from fluxforge.io.artifacts import write_unfold_result, read_unfold_result
from fluxforge.core.schemas import validate_or_raise
from fluxforge.unfolding import GravelUnfolder, MaxedUnfolder
from fluxforge.unfolding.base import estimate_unfolding_uncertainties


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert (
        sha(args.input)
        == "3beb7b2df86f21597f78e3d8e49be6a2d571eba3099410173bd76a6e81e85b58"
    )
    assert (
        sha(args.baseline)
        == "9dc669432375af41049b8aad47e91edc7ecfb4cb1b0a2869ed6a094883072609"
    )
    inputs = {str(p.resolve()): sha(p) for p in (args.input, args.baseline)}
    source_paths = [
        "src/fluxforge/" + p
        for p in [
            "unfolding/base.py",
            "unfolding/gravel.py",
            "unfolding/maxed.py",
            "unfolding/ml_seed.py",
            "unfolding/rmle.py",
            "solvers/rmle.py",
            "gui/dialogs/unfolding_dialog.py",
            "cli/app.py",
            "io/artifacts.py",
            "core/schemas.py",
            "solvers/iterative.py",
            "core/unfolding_inputs.py",
        ]
    ] + [
        "tools/validate_registry_uncertainty.py",
        "tests/test_registry_uncertainty_qualification.py",
        "tests/test_unfolding_registry.py",
        "tests/test_unfolding_workflows.py",
        "tests/test_unfolding_workspace_qt.py",
        "tests/test_artifacts_io.py",
        "tests/test_cli_app.py",
        "tests/test_rmle.py",
    ]
    sources = {p: sha(p) for p in source_paths}
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for path, digest in data["input_sha256"].items():
        assert sha(path) == digest, path
        inputs[path] = digest
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    rows = data["sources"]
    response = np.asarray(data["row_cross_sections_barn"]) * 1e-24
    rates = np.array([r["replayed_rate"] for r in rows])
    sigma = np.array([r["replayed_rate_unc"] for r in rows])
    flux_scale = 1e11
    whitened = response * flux_scale / sigma[:, None]
    measured = rates / sigma
    assert response.shape == (13, 20)
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for old in baseline["methods"]:
        method = "gravel" if old["method"] == "FluxForge_GRAVEL" else "maxed"
        assert old["method"] in {"FluxForge_GRAVEL", "FluxForge_MAXED"}
        instance = GravelUnfolder() if method == "gravel" else MaxedUnfolder()
        prior = np.asarray(old["initial_u"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = instance.unfold(
                measured,
                whitened,
                initial_flux=prior,
                measurement_uncertainty=np.ones(13),
                **old["requested_parameters"],
            )
        flux = result.flux * flux_scale
        predictions = response @ flux
        residuals = (predictions - rates) / sigma
        assert np.array_equal(flux, old["group_integral_flux"])
        assert np.array_equal(predictions, old["predictions"])
        assert np.array_equal(residuals, old["standardized_residuals"])
        assert result.converged == old["numerical_converged"]
        assert result.uncertainties is None
        assert result.parameters_used["uncertainty_qualified"] is False
        assert result.parameters_used["uncertainty_unavailable_reason"]
        cases.append(
            dict(
                method=method,
                prior_and_start_name=old["prior_name"],
                group_integral_flux=flux.tolist(),
                predictions=predictions.tolist(),
                standardized_residuals=residuals.tolist(),
                weighted_squared_residual=float(residuals @ residuals),
                numerical_converged=result.converged,
                iterations=result.iterations,
                parameters_used=result.parameters_used,
                reported_uncertainties=None,
                estimates_predictions_residuals_unchanged=True,
                warnings=list(dict.fromkeys(str(w.message) for w in caught)),
            )
        )
    unit_cases = []
    for scale in [1.0, 1e-12, 1e12]:
        u = estimate_unfolding_uncertainties(
            np.eye(2) * scale, measurement_uncertainty=np.array([0.1, 0.2]) * scale
        )
        np.testing.assert_allclose(u, [0.1, 0.2], rtol=1e-14, atol=0)
        unit_cases.append(
            dict(row_scale=scale, sigma=u.tolist(), analytic_sigma=[0.1, 0.2])
        )
    try:
        estimate_unfolding_uncertainties(whitened, measurement_uncertainty=np.ones(13))
    except ValueError as exc:
        rank_rejection = str(exc)
    else:
        raise AssertionError("Rank-deficient actual response passed linear utility")
    exports = []
    for method in ["gravel", "maxed"]:
        prior = baseline["methods"][0]["initial_u"]
        flux, covariance, chi2, _, diagnostics = _solve_unfold_method(
            method=method,
            response_matrix=whitened.tolist(),
            measured_rates=measured.tolist(),
            rate_uncertainties=np.ones(13).tolist(),
            measurement_cov=np.eye(13).tolist(),
            prior_flux=prior,
            prior_cov=np.eye(20).tolist(),
            args=argparse.Namespace(max_iters=20),
        )
        assert covariance is None
        diagnostics = _build_unfold_diagnostics(
            reactions=[r["observation_id"] for r in rows],
            response_matrix=whitened.tolist(),
            measured_rates=measured.tolist(),
            rate_uncertainties=np.ones(13).tolist(),
            prior_flux=prior,
            prior_cov=np.eye(20).tolist(),
            flux=flux,
            covariance=covariance,
            diagnostics=diagnostics,
        )
        assert diagnostics["flux_uncertainty"] is None
        assert diagnostics["predicted_rate_uncertainties"] is None
        diagnostics["flux_representation"] = "u=group-integral flux/1e11"
        path = args.out / (method + "_unavailable_export.json")
        write_unfold_result(
            path,
            boundaries_eV=data["edges_eV"],
            reactions=[r["observation_id"] for r in rows],
            flux=flux,
            covariance=None,
            chi2=chi2,
            method=method,
            diagnostics=diagnostics,
        )
        validate_or_raise(read_unfold_result(path))
        exports.append(
            dict(
                method=method,
                path=path.name,
                sha256=sha(path),
                covariance=None,
                unknown_basis="u=group-integral flux/1e11; source energy edges retained",
            )
        )
    assert all(sha(p) == h for p, h in inputs.items())
    assert all(sha(p) == h for p, h in sources.items())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_blobs = {
        p: hashlib.sha256(
            subprocess.check_output(["git", "show", commit + ":" + p])
        ).hexdigest()
        for p in source_paths
    }
    receipt = dict(
        schema="registry-uncertainty-repair-v1",
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        input_sha256=inputs,
        source_sha256=sources,
        git_blob_sha256=git_blobs,
        rows=13,
        groups=20,
        rank=int(np.linalg.matrix_rank(whitened)),
        observation_ids=[r["observation_id"] for r in rows],
        replay_rates=rates.tolist(),
        replay_rate_uncertainties=sigma.tolist(),
        row_cross_sections_barn=data["row_cross_sections_barn"],
        normalization="u=group-integral flux/1e11; response*1e11/sigma and rates/sigma; explicit whitened sigma=1",
        cases=cases,
        row_unit_cases=unit_cases,
        rank_deficient_linear_rejection=rank_rejection,
        unavailable_exports=exports,
        source_and_input_unchanged=True,
        measured_inputs_corrected=False,
        rmle_scope="Count-domain identity/default/fallback/MC checks only; no new activation-rate Poisson replay",
        scientific_admission=False,
        limits=[
            "No estimator-specific registry uncertainty is implemented; no linear proxy is published as nonlinear uncertainty.",
            "The linear utility describes independent measurement errors, fixed response, unconstrained full-rank weighted least squares only.",
            "MAXED prior cases change both entropy prior and initial iterate; not pure optimizer-start variation.",
            "Calibration/history/covariance/event/geometry remain unqualified; coarse Sc/Co group bounds do not establish a general pointwise incompatibility.",
        ],
    )
    path = args.out / "receipt.json"
    path.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                receipt=str(path),
                sha256=sha(path),
                cases=len(cases),
                scientific_admission=False,
            )
        )
    )


if __name__ == "__main__":
    main()
