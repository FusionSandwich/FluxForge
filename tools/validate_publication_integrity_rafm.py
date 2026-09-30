"""Additive continuation replay; consumes the source-joined operator receipt.

No downloads or scientific admission. All actual error budgets remain incomplete.
Synthetic prior/response-error terms exercise the GLS interface only.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from fluxforge.analysis.physical_gls import (
    MonitorRow,
    SourceBinding,
    unfold_gls_physical,
)
from fluxforge.data.irdff import (
    IRDFFDatabase,
    DEFAULT_CACHE_DIR,
    IRDFF_TAB_ARCHIVE_NAME,
)
from fluxforge.physics.monitor_response import CoverLayer, MonitorResponseSpec
from fluxforge.uncertainty.reaction_rate_budget import (
    RateUncertaintyBudget,
    UncertaintyComponent,
    rate_covariance,
)
from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    after = json.loads(args.operator.read_text(encoding="utf-8"))
    before = json.loads(args.before.read_text(encoding="utf-8"))
    # Equality excludes newly explicit budget coverage/sign fields, not data values.
    exact = [
        "edges_eV",
        "row_cross_sections_barn",
        "row_uncertainties_barn",
        "group_integral_prior",
        "forward_predictions",
    ]
    for name in exact:
        assert before[name] == after[name], name
    for a, b in zip(before["sources"], after["sources"]):
        assert {k: v for k, v in a.items() if k != "rate_uncertainty_budget"} == {
            k: v for k, v in b.items() if k != "rate_uncertainty_budget"
        }
    for name in before["workflows"]:
        for field in ("response", "rates", "flux", "predictions"):
            np.testing.assert_allclose(
                before["workflows"][name][field],
                after["workflows"][name][field],
                rtol=1e-12,
                atol=0.0,
            )
    assert all(sha(p) == h for p, h in after["input_sha256"].items())
    edges = np.array(after["edges_eV"])
    sources = after["sources"]
    rates = np.array([s["replayed_rate"] for s in sources])
    errors = np.array([s["replayed_rate_unc"] for s in sources])
    budgets = []
    for s in sources:
        b = dict(s["rate_uncertainty_budget"])
        b["components"] = [UncertaintyComponent(**c) for c in b["components"]]
        budgets.append(RateUncertaintyBudget(**b))
    covariance = rate_covariance(budgets)
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)), errors, rtol=1e-12, atol=0.0
    )
    assert all(not b.complete and len(b.missing) == 6 for b in budgets)
    try:
        rate_covariance(budgets, require_complete=True)
    except ValueError as exc:
        strict_error = str(exc)
    else:
        raise AssertionError("Incomplete actual budgets passed strict gate")
    archive = DEFAULT_CACHE_DIR / "tab" / IRDFF_TAB_ARCHIVE_NAME
    db = IRDFFDatabase(
        auto_download=False, archive_path=archive, expected_archive_sha256=sha(archive)
    )
    specs = [
        MonitorResponseSpec(
            s["observation_id"],
            s["sample_id"],
            s["reaction"],
            cover=CoverLayer("Cd", 0.0508) if s["covered"] else None,
        )
        for s in sources
    ]

    def make(current_specs=specs):
        u = SpectrumUnfolder(custom_energy_edges=edges, verbose=False)
        u.irdff_db = db
        for spec, rate, error, budget in zip(current_specs, rates, errors, budgets):
            u.add_reaction(
                spec.reaction,
                rate,
                error,
                rate_per_atom=rate,
                sample_id=spec.sample_id,
                cover="Cd" if spec.cover else None,
                response_spec=spec,
                rate_uncertainty_budget=budget,
            )
        u.set_initial_guess(
            np.array(after["group_integral_prior"]) / np.diff(edges),
            source="synthetic equal integral group prior; diagnostic",
        )
        return u

    u = make()
    original = u._build_response_matrix()[0]
    u.measurements[0].activity_Bq *= 2
    u.measurements[0].uncertainty_Bq *= 2
    assert u._build_response_matrix()[0] is original
    changed = replace(specs[0], cover=replace(specs[0].cover, density_g_cm3=4.325))
    u.measurements[0].response_spec = changed
    warm = u._build_response_matrix()[0]
    fresh = make([changed, *specs[1:]])._build_response_matrix()[0]
    np.testing.assert_array_equal(warm, fresh)
    assert not np.array_equal(warm[0], original[0])
    u.measurements[0].response_spec = None
    try:
        u.unfold(method="MLEM", max_iterations=1)
    except ValueError as exc:
        missing_cover_error = str(exc)
    else:
        raise AssertionError("Warm cache admitted missing covered response")

    u = make()
    mc = {}
    for estimator in ("converged", "capped"):
        r = u.unfold(
            method="MLEM",
            max_iterations=1,
            tolerance=1e-10,
            uncertainty_method="monte_carlo",
            uncertainty_estimator=estimator,
            n_uncertainty_samples=8,
            uncertainty_seed=42,
        )
        meta = {
            k: v
            for k, v in r.metadata.items()
            if k.startswith("flux_uncertainty")
            or k in ("rate_covariance_qualification", "iterative_fit_covariance")
        }
        meta["all_flux_uncertainties_finite"] = bool(
            np.all(np.isfinite(r.flux_uncertainty))
        )
        mc[estimator] = meta
    assert mc["converged"]["flux_uncertainty_qualification"] == "unavailable"
    assert mc["capped"]["flux_uncertainty_qualification"] == "capped_diagnostic"
    try:
        u.unfold(method="MLEM", max_iterations=1, require_complete_rate_budget=True)
    except ValueError:
        strict_workflow_rejected = True
    else:
        raise AssertionError("Iterative workflow bypassed actual source budget gate")

    # Primary physical GLS interface. Prior and response-error covariance below
    # are explicitly synthetic; measured component completeness was rejected.
    response = np.array(after["row_cross_sections_barn"]) * 1e-24
    prior = np.array(after["group_integral_prior"])
    prior_cov = np.diag((5 * prior) ** 2)
    response_cov = np.diag((0.01 * (response @ prior)) ** 2)
    identities = [
        MonitorRow(
            s["observation_id"],
            s["sample_id"],
            "Cd" if s["covered"] else "bare",
            s["reaction"],
            s["isotope"],
        )
        for s in sources
    ]
    inputs = dict(
        row_identities=[vars(r) for r in identities],
        energy_edges=edges.tolist(),
        rates=rates.tolist(),
        response=response.tolist(),
        prior=prior.tolist(),
        prior_covariance=prior_cov.tolist(),
        observation_covariance=covariance.tolist(),
        response_error_covariance=response_cov.tolist(),
    )
    units = dict(
        row_identities="identity",
        energy_edges="eV",
        rates="reactions/target_atom/s",
        response="cm2",
        prior="n/cm2/s",
        prior_covariance="(n/cm2/s)^2",
        observation_covariance="(reactions/target_atom/s)^2",
        response_error_covariance="(reactions/target_atom/s)^2",
    )
    bindings = {
        name: SourceBinding(
            "diagnostic-input://" + name,
            hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
            units[name],
        )
        for name, value in inputs.items()
    }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    holdout = next(
        s["observation_id"]
        for s in sources
        if s["covered"] and s["reaction"].startswith("Sc-")
    )
    gls = unfold_gls_physical(
        rows=identities,
        energy_edges_eV=edges,
        measured_rates=rates,
        response_matrix=response,
        prior_flux=prior,
        prior_covariance=prior_cov,
        observation_covariance=covariance,
        response_error_covariance=response_cov,
        sources=bindings,
        source_commit=head,
        holdout_ids=[holdout],
    )
    ids = [i for i, s in enumerate(sources) if s["observation_id"] != holdout]
    total_fit_cov = (covariance + response_cov)[np.ix_(ids, ids)]
    expected = gls.fit_residuals @ np.linalg.solve(total_fit_cov, gls.fit_residuals)
    np.testing.assert_allclose(
        gls.receipt()["postfit_total_error_chi2"], expected, rtol=1e-12
    )
    assert gls.receipt()["scientific_admission"] is False
    assert all(sha(p) == h for p, h in after["input_sha256"].items())
    result = dict(
        code_commit=head,
        validation_script_sha256=sha(__file__),
        source_files_sha256={
            str(p): sha(p)
            for p in Path("src/fluxforge").rglob("*.py")
            if p.name
            in {
                "spectrum_unfolding.py",
                "reaction_rate_budget.py",
                "covariance.py",
                "holdout_validation.py",
                "physical_gls.py",
                "rafm_workflow.py",
            }
        },
        operator_receipt_sha256=sha(args.operator),
        before_receipt_sha256=sha(args.before),
        exact_operator_fields=exact,
        unchanged_valid_workflows_rtol=1e-12,
        source_bytes_unchanged=True,
        input_sha256=after["input_sha256"],
        observations=13,
        group_count=len(edges) - 1,
        common_operator_rank=int(np.linalg.matrix_rank(response)),
        cache=dict(
            changed_density_matches_fresh=True,
            rate_only_hit=True,
            missing_cover_rejected=missing_cover_error,
        ),
        actual_budgets=[b.as_row() for b in budgets],
        strict_error=strict_error,
        strict_workflow_rejected=strict_workflow_rejected,
        monte_carlo=mc,
        gls_inputs=inputs,
        physical_gls=gls.receipt(),
        scientific_admission=False,
        limitations=[
            "All 13 actual budgets lack six measured components; opaque activity composition retained",
            "Synthetic prior and 1% response-error term exercise the GLS interface only",
            "Iterative objectives remain diagonal; full covariance is used for resampling and aggregation",
            "SpecKit original settings/quasi-single benchmark unresolved; comparator agreement is not admission",
            "Measured history, cover/body geometry and covariance qualification remain with source audit owner",
        ],
    )
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.out),
                sha256=sha(args.out),
                observations=13,
                strict_budget_rejected=True,
                mc_qualification={
                    k: v["flux_uncertainty_qualification"] for k, v in mc.items()
                },
                common_operator_rank=result["common_operator_rank"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
