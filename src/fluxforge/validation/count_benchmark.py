"""Freeze source-bound count predictions, then score unreduced integer counts.

Hash bindings and family splits make a benchmark inspectable. They cannot prove
that an analyst has never seen test data or establish physical source admission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

from fluxforge.validation.run_readiness import check_run_readiness_manifest


def _mean_draws(values):
    draws = np.array(values, dtype=float, copy=True)
    if (
        draws.ndim != 2
        or 0 in draws.shape
        or not np.isfinite(draws).all()
        or np.any(draws < 0)
    ):
        raise ValueError(
            "expected counts must be a finite nonnegative draws x bins array"
        )
    return draws


def validate_family_split(observations, training_ids, holdout_ids):
    """Require an exhaustive, disjoint split of complete declared reaction families."""
    records = [dict(item) for item in observations]
    ids, families = [], {}
    for item in records:
        if set(item) != {"observation_id", "reaction_family", "acquisition_id"}:
            raise ValueError("observation records must contain identity metadata only")
        for key in ("observation_id", "reaction_family", "acquisition_id"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"each observation requires nonempty {key}")
        ids.append(item["observation_id"])
        families[item["observation_id"]] = item["reaction_family"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("observation IDs must be nonempty and unique")
    train, holdout = list(training_ids), list(holdout_ids)
    if (
        not train
        or not holdout
        or len(set(train)) != len(train)
        or len(set(holdout)) != len(holdout)
    ):
        raise ValueError("training and holdout IDs must be nonempty and unique")
    if set(train) & set(holdout) or set(train) | set(holdout) != set(ids):
        raise ValueError(
            "training and holdout IDs must form an exhaustive disjoint split"
        )
    if {families[i] for i in train} & {families[i] for i in holdout}:
        raise ValueError("a reaction family cannot occur in both training and holdout")
    return {"observations": records, "training_ids": train, "holdout_ids": holdout}


def predictive_count_summary(expected_counts, *, seed, predictive_draws=20000):
    """Moments and Monte Carlo intervals of the Poisson posterior mixture."""
    means = _mean_draws(expected_counts)
    if (
        isinstance(predictive_draws, bool)
        or not isinstance(predictive_draws, int)
        or predictive_draws < 1000
    ):
        raise ValueError("predictive_draws must be an integer >= 1000")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    rng = np.random.default_rng(seed)
    # Selecting one shared row per replicate retains nuisance dependence.
    indices = rng.integers(means.shape[0], size=predictive_draws)
    simulated = rng.poisson(means[indices])
    intervals = {}
    for level in (0.5, 0.8, 0.95):
        tail = (1 - level) / 2
        intervals[str(level)] = np.quantile(
            simulated, [tail, 1 - tail], axis=0, method="inverted_cdf"
        ).tolist()
    return {
        "mean": means.mean(axis=0).tolist(),
        "variance": (means.mean(axis=0) + means.var(axis=0)).tolist(),
        "intervals": intervals,
        "predictive_draws": predictive_draws,
        "seed": seed,
        "interval_method": "Monte_Carlo_Poisson_mixture",
    }


def score_count_draws(observed_counts, expected_counts):
    """Log of the average JOINT likelihood, not a sum of marginal log scores."""
    means = _mean_draws(expected_counts)
    counts = np.asarray(observed_counts)
    if counts.dtype.kind == "b":
        raise ValueError("observed_counts must be integer counts, not booleans")
    counts = np.asarray(counts, dtype=float)
    if (
        counts.shape != (means.shape[1],)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError("observed_counts must be a nonnegative integer bin vector")
    log_likelihood = (xlogy(counts, means) - means - gammaln(counts + 1)).sum(axis=1)
    joint_score = float(logsumexp(log_likelihood) - np.log(len(means)))
    mean = means.mean(axis=0)
    return {
        "joint_log_predictive_density": (
            joint_score if np.isfinite(joint_score) else None
        ),
        "status": (
            "scored" if np.isfinite(joint_score) else "zero_predictive_probability"
        ),
        "observed_counts": counts.astype(int).tolist(),
        "predicted_mean_counts": mean.tolist(),
        "residual_counts": (counts - mean).tolist(),
        "physical_qualified": False,
    }


def freeze_count_prediction(
    path,
    *,
    manifest_path,
    split,
    bin_observation_ids,
    expected_counts,
    model_id,
    uncertainty_scope,
    study_type,
    seed,
    predictive_draws=20000,
):
    """Create an immutable-by-convention packet before any scoring.

    The caller must archive/commit its returned SHA before releasing test values.
    This function accepts no observed test count values and refuses overwrite.
    `study_type` explicitly distinguishes synthetic/retrospective/prospective.
    """
    split = validate_family_split(**split)
    means = _mean_draws(expected_counts)
    labels = list(bin_observation_ids)
    if len(labels) != means.shape[1] or set(labels) != set(split["holdout_ids"]):
        raise ValueError(
            "each bin must identify a holdout observation; all must appear"
        )
    if study_type not in ("synthetic", "retrospective", "prospective"):
        raise ValueError("study_type must be synthetic, retrospective or prospective")
    for name, value in (
        ("model_id", model_id),
        ("uncertainty_scope", uncertainty_scope),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be explicitly declared")
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    sources = check_run_readiness_manifest(manifest_path)
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("source manifest changed during validation")
    required = {
        "training_data",
        "heldout_data",
        "calibration",
        "irradiation_history",
        "responses",
        "analysis_config",
        "analysis_code",
    }
    roles = {item["role"] for item in sources["artifacts"]}
    if not required.issubset(roles):
        raise ValueError(f"manifest lacks required roles: {sorted(required - roles)}")
    packet = {
        "schema": "fluxforge.frozen_count_prediction.v1",
        "model_id": model_id,
        "uncertainty_scope": uncertainty_scope,
        "study_type": study_type,
        "split": split,
        "bin_observation_ids": labels,
        "expected_count_draws": means.tolist(),
        "count_model": "conditionally_independent_native_bin_Poisson",
        "predictive_summary": predictive_count_summary(
            means, seed=seed, predictive_draws=predictive_draws
        ),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_bindings": sources,
        "physical_qualified": False,
    }
    data = (
        json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with Path(path).open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def score_frozen_prediction(path, observed_counts, *, expected_sha256):
    """Score the exact archived means without running or refitting a model."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("frozen prediction does not match the committed SHA-256")
    packet = json.loads(data)
    if packet.get("schema") != "fluxforge.frozen_count_prediction.v1":
        raise ValueError("unsupported frozen prediction schema")
    score = score_count_draws(observed_counts, packet["expected_count_draws"])
    score["prediction_sha256"] = hashlib.sha256(data).hexdigest()
    score["source_manifest_sha256"] = packet["source_manifest_sha256"]
    score["model_id"] = packet["model_id"]
    score["study_type"] = packet["study_type"]
    summary = packet["predictive_summary"]
    counts = np.asarray(observed_counts)
    score["interval_contains_observation"] = {
        level: (
            (np.asarray(bounds[0]) <= counts) & (counts <= np.asarray(bounds[1]))
        ).tolist()
        for level, bounds in summary["intervals"].items()
    }
    score["coverage_claim"] = (
        "single-case containment only; no calibrated coverage claim"
    )
    return score
