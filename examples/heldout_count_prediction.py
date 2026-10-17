"""Reproducible synthetic reaction-family holdout; no measured-data claim.

Usage: PYTHONPATH=src python examples/heldout_count_prediction.py NEW_DIRECTORY
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from fluxforge.physics.activation import IrradiationSegment
from fluxforge.physics.count_prediction import (
    activation_count_operator,
    expected_count_draws,
    gaussian_bin_probabilities,
)
from fluxforge.solvers.parametric import (
    GaussianResponseModel,
    Parameter,
    ParametricProblem,
    parametric_unfold,
)
from fluxforge.validation.count_benchmark import (
    freeze_count_prediction,
    score_frozen_prediction,
)


def run(output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "inputs"
    source.mkdir()
    response = np.array([[1, 0.2], [0.1, 1], [0.3, 1.5]]) * 1e-24
    true_flux = np.array([1, 2]) * 1e10
    rates = response[:2] @ true_flux
    covariance = np.diag((rates * 0.1) ** 2)
    # Only the first two reactions enter inference. The final family is withheld.
    model = GaussianResponseModel(response[:2], rates, covariance)
    problem = ParametricProblem(
        model,
        lambda t: t * 1e10,
        [Parameter("thermal", 0.1, 5, 0.1), Parameter("fast", 0.1, 5, 0.1)],
        "synthetic_two_group_v1",
    )
    posterior = parametric_unfold(
        problem, [[0.5, 0.5], [3, 4]], draws=1000, warmup=1000, seed=17
    )
    edges = np.linspace(997, 1003, 13)
    count_config = dict(
        target_atoms=1e18,
        half_life_s=21600,
        cooling_time_s=3600,
        real_time_s=600,
        live_time_s=540,
        emission_probabilities=[0.7],
        efficiencies=[0.1],
    )
    operator = activation_count_operator(
        response[2],
        **count_config,
        irradiation_segments=[IrradiationSegment(21600)],
        line_bin_probabilities=gaussian_bin_probabilities(edges, [1000], [0.7]),
    )
    posterior_means = expected_count_draws(
        posterior.flux_samples.reshape(-1, 2), operator, additive_counts=5
    )
    prior_means = expected_count_draws(
        np.random.default_rng(7).uniform(0.1, 5, (2000, 2)) * 1e10,
        operator,
        additive_counts=5,
    )
    heldout = np.random.default_rng(18).poisson(operator @ true_flux + 5).tolist()
    split = {
        "observations": [
            {
                "observation_id": f"r{i}",
                "reaction_family": f"family{i}",
                "acquisition_id": f"acq{i}",
            }
            for i in range(3)
        ],
        "training_ids": ["r0", "r1"],
        "holdout_ids": ["r2"],
    }
    inputs = {
        "training_data": {"rates": rates.tolist(), "covariance": covariance.tolist()},
        "heldout_data": {"native_counts": heldout},
        "calibration": {
            **count_config,
            "edges_keV": edges.tolist(),
            "assumption": "synthetic fixed response and detector",
        },
        "irradiation_history": {
            "segments": [{"duration_s": 21600, "relative_power": 1}]
        },
        "responses": {"group_averaged_cm2": response.tolist()},
        "analysis_config": {
            "split": split,
            "posterior": posterior.metadata,
            "baseline": "unadjusted uniform parameter prior",
        },
    }
    artifacts = []
    for role, value in inputs.items():
        path = source / f"{role}.json"
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        artifacts.append(
            {
                "path": path.name,
                "role": role,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    root = Path(__file__).resolve().parents[1]
    code_paths = [
        Path(__file__).resolve(),
        root / "src/fluxforge/solvers/parametric.py",
        root / "src/fluxforge/core/probability.py",
        root / "src/fluxforge/physics/activation.py",
        root / "src/fluxforge/physics/count_prediction.py",
        root / "src/fluxforge/validation/count_benchmark.py",
        root / "src/fluxforge/validation/run_readiness.py",
    ]
    for original in code_paths:
        path = source / original.name
        shutil.copyfile(original, path)
        artifacts.append(
            {
                "path": path.name,
                "role": "analysis_code",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "fluxforge.run_readiness_manifest.v1",
                "metadata": {
                    "manifest_id": "synthetic_family_holdout_v1",
                    "purpose": "verify frozen absolute-count prediction",
                },
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n"
    )
    scores = {}
    for name, means in (
        ("parametric", posterior_means),
        ("unadjusted_prior", prior_means),
    ):
        path = output / f"{name}.prediction.json"
        digest = freeze_count_prediction(
            path,
            manifest_path=manifest,
            split=split,
            bin_observation_ids=["r2"] * len(heldout),
            expected_counts=means,
            model_id=name,
            uncertainty_scope="flux only; fixed synthetic nuclear and detector inputs",
            study_type="synthetic",
            seed=19,
        )
        # Evaluator reads withheld count values only after the packet is frozen.
        evaluation_counts = json.loads((source / "heldout_data.json").read_text())[
            "native_counts"
        ]
        scores[name] = score_frozen_prediction(
            path, evaluation_counts, expected_sha256=digest
        )
    result = {
        "scores": scores,
        "split_rhat": posterior.rhat.tolist(),
        "physical_qualified": False,
        "evidence_scope": "synthetic_only",
    }
    (output / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=2, allow_nan=False))
