"""Count-clock closure, shared uncertainty, split and frozen-score regressions."""

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import poisson

from fluxforge.analysis.prediction_comparison import compare_calculated_experimental
from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    irradiation_buildup_factor,
)
from fluxforge.physics.count_prediction import (
    activation_count_operator,
    expected_count_draws,
    gaussian_bin_probabilities,
)
from fluxforge.validation.count_benchmark import (
    freeze_count_prediction,
    predictive_count_summary,
    score_count_draws,
    score_frozen_prediction,
    validate_family_split,
)


def count_arguments():
    return dict(
        target_atoms=100,
        half_life_s=10,
        irradiation_segments=[IrradiationSegment(10, 1), IrradiationSegment(5, 0)],
        cooling_time_s=3,
        real_time_s=8,
        live_time_s=4,
        emission_probabilities=[0.8],
        efficiencies=[0.2],
        line_bin_probabilities=[[1]],
    )


def test_absolute_count_operator_against_time_integrals_and_inverse_activity():
    args = count_arguments()
    matrix = activation_count_operator([0.01, 0.02], **args)
    flux = np.array([3, 5])
    decay = math.log(2) / 10
    production = 100 * (np.array([0.01, 0.02]) @ flux)
    eoi_atoms = quad(lambda t: production * math.exp(-decay * (15 - t)), 0, 10)[0]
    expected = quad(
        lambda t: 0.5 * 0.8 * 0.2 * decay * eoi_atoms * math.exp(-decay * t), 3, 11
    )[0]
    counts = (matrix @ flux).item()
    assert counts == pytest.approx(expected)
    measured = GammaLineMeasurement(
        counts, 4, 0.2, 0.8, 10, cooling_time_s=3, real_time_s=8
    )
    assert measured.activity_at_reference() == pytest.approx(decay * eoi_atoms)
    expected_activity = production * irradiation_buildup_factor(
        args["irradiation_segments"], 10
    )
    assert expected_activity == pytest.approx(decay * eoi_atoms)


def test_native_bins_integrate_probability_and_keep_outside_tails():
    probabilities = gaussian_bin_probabilities([0, 1, 1.5, 3], [1], [1])
    expected = [
        quad(lambda e: math.exp(-((e - 1) ** 2) / 2) / math.sqrt(2 * math.pi), a, b)[0]
        for a, b in zip([0, 1, 1.5], [1, 1.5, 3])
    ]
    np.testing.assert_allclose(probabilities[0], expected)
    assert probabilities.sum() < 1


@pytest.mark.parametrize(
    "change",
    [
        {"live_time_s": 9},
        {"cooling_time_s": -1},
        {"efficiencies": [20]},
        {"line_bin_probabilities": [[0.8, 0.8]]},
        {"irradiation_segments": []},
    ],
)
def test_invalid_count_physics_rejected(change):
    args = count_arguments()
    args.update(change)
    with pytest.raises(ValueError):
        activation_count_operator([1], **args)


def test_joint_draw_alignment_is_preserved():
    flux = [[1], [2]]
    operators = [[[2], [3]], [[4], [5]]]
    actual = expected_count_draws(flux, operators, additive_counts=[[1, 2], [3, 4]])
    np.testing.assert_allclose(actual, [[3, 5], [11, 14]])


def test_joint_score_matches_mixture_not_product_of_marginals():
    means = np.array([[1, 10], [10, 1]])
    counts = np.array([1, 1])
    reference = logsumexp(poisson.logpmf(counts, means).sum(axis=1)) - math.log(2)
    wrong = np.sum(logsumexp(poisson.logpmf(counts, means), axis=0) - math.log(2))
    score = score_count_draws(counts, means)
    assert score["joint_log_predictive_density"] == pytest.approx(reference)
    assert abs(reference - wrong) > 1


@pytest.mark.parametrize("counts", [[-1], [0.2], [np.nan], [True]])
def test_only_unreduced_integer_counts_can_be_scored(counts):
    with pytest.raises(ValueError):
        score_count_draws(counts, [[1]])


def test_zero_count_and_impossible_prediction_are_explicit():
    assert score_count_draws([0], [[0]])["joint_log_predictive_density"] == 0
    result = score_count_draws([1], [[0]])
    assert result["joint_log_predictive_density"] is None
    assert result["status"] == "zero_predictive_probability"


def test_predictive_intervals_include_count_noise():
    result = predictive_count_summary([[10]], seed=1, predictive_draws=100000)
    assert result["variance"] == [10]
    assert result["intervals"]["0.95"][0][0] < 10
    assert result["intervals"]["0.95"][1][0] > 10
    np.testing.assert_allclose(
        result["intervals"]["0.95"],
        poisson.ppf([0.025, 0.975], 10).reshape(2, 1),
        atol=1,
    )


def split():
    return {
        "observations": [
            {
                "observation_id": "co",
                "reaction_family": "Co59_capture",
                "acquisition_id": "a",
            },
            {
                "observation_id": "sc",
                "reaction_family": "Sc45_capture",
                "acquisition_id": "b",
            },
        ],
        "training_ids": ["co"],
        "holdout_ids": ["sc"],
    }


def test_shared_family_cannot_leak_across_split():
    args = split()
    args["observations"][1]["reaction_family"] = "Co59_capture"
    with pytest.raises(ValueError, match="reaction family"):
        validate_family_split(**args)


def test_holdout_values_cannot_be_smuggled_in_metadata():
    args = split()
    args["observations"][1]["counts"] = 99
    with pytest.raises(ValueError, match="identity metadata"):
        validate_family_split(**args)


def manifest(tmp_path):
    roles = [
        "training_data",
        "heldout_data",
        "calibration",
        "irradiation_history",
        "responses",
        "analysis_config",
        "analysis_code",
    ]
    artifacts = []
    for role in roles:
        path = tmp_path / f"{role}.txt"
        path.write_text(f"synthetic {role}")
        artifacts.append(
            {
                "path": path.name,
                "role": role,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "fluxforge.run_readiness_manifest.v1",
                "metadata": {
                    "manifest_id": "synthetic",
                    "purpose": "count prediction test",
                },
                "artifacts": artifacts,
            }
        )
    )
    return path


def test_frozen_packet_hash_scoring_overwrite_and_source_mutation(tmp_path):
    source = manifest(tmp_path)
    path = tmp_path / "prediction.json"
    args = dict(
        manifest_path=source,
        split=split(),
        bin_observation_ids=["sc"],
        expected_counts=[[8], [10], [12]],
        model_id="fixed_v1",
        uncertainty_scope="synthetic flux only",
        study_type="synthetic",
        seed=11,
        predictive_draws=1000,
    )
    digest = freeze_count_prediction(path, **args)
    score = score_frozen_prediction(path, [11], expected_sha256=digest)
    assert score["prediction_sha256"] == digest
    assert score["physical_qualified"] is False
    with pytest.raises(FileExistsError):
        freeze_count_prediction(path, **args)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="committed SHA"):
        score_frozen_prediction(path, [11], expected_sha256=digest)
    (tmp_path / "calibration.txt").write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        freeze_count_prediction(tmp_path / "new.json", **args)


def test_shared_calibration_cancels_in_ratio_covariance():
    values = np.array([10.0, 20.0, 10.0, 20.0])
    shared = 0.1 * values
    result = compare_calculated_experimental(
        [10, 20], [10, 20], np.outer(shared, shared)
    )
    np.testing.assert_allclose(result["ratio_covariance"], 0, atol=1e-16)
    assert result["joint_gaussian_log_density"] is None


def test_ce_full_covariance_against_independent_jacobian():
    c, e = np.array([9.0, 22.0]), np.array([10.0, 20.0])
    covariance = np.diag([1.0, 2.0, 3.0, 4.0])
    result = compare_calculated_experimental(c, e, covariance)
    expected = np.diag([1 / 100 + 81 * 3 / 10000, 2 / 400 + 484 * 4 / 160000])
    np.testing.assert_allclose(result["ratio_covariance"], expected)
    assert result["chi_squared"] == pytest.approx(1 / 4 + 4 / 6)


def test_synthetic_family_holdout_runs_end_to_end(tmp_path):
    import runpy
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "examples/heldout_count_prediction.py"
    )
    result = runpy.run_path(str(script))["run"](tmp_path / "benchmark")
    assert result["evidence_scope"] == "synthetic_only"
    assert set(result["scores"]) == {"parametric", "unadjusted_prior"}
    for score in result["scores"].values():
        assert score["status"] == "scored"
        assert score["physical_qualified"] is False
        assert len(score["observed_counts"]) == 12
