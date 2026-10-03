"""Analytical covariance display contracts, including singular/unknown cases."""

import json

import numpy as np
import pytest

from fluxforge.gui.covariance_view import prepare_covariance_view, read_covariance_view


@pytest.mark.parametrize("scale", [1e-300, 1, 1e300])
def test_correlation_is_invariant_to_units_and_covariance_is_not_changed(scale):
    matrix = np.array([[4.0, -3.0], [-3.0, 9.0]]) * scale
    view = prepare_covariance_view(matrix, ["Activity A", "Activity B"])
    np.testing.assert_allclose(view.correlation, [[1, -0.5], [-0.5, 1]])
    np.testing.assert_array_equal(view.covariance, matrix)
    matrix[:] = 0
    assert view.covariance[0, 0] == 4 * scale
    assert not view.covariance.flags.writeable


def test_mixed_units_do_not_hide_correlation():
    view = prepare_covariance_view([[1e-300, 0.5], [0.5, 1e300]])
    np.testing.assert_allclose(view.correlation, [[1, 0.5], [0.5, 1]])


def test_singular_covariance_is_valid_and_zero_variance_stays_undefined():
    view = prepare_covariance_view([[1, -2, 0], [-2, 4, 0], [0, 0, 0]])
    np.testing.assert_allclose(view.correlation[:2, :2], [[1, -1], [-1, 1]])
    assert np.isnan(view.correlation[2]).all()
    assert np.isnan(view.correlation[:, 2]).all()
    assert view.covariance[2, 2] == 0


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        [],
        [[1, 2]],
        [[1, float("nan")], [0, 1]],
        [[-1]],
        [[0, 1e-300], [1e-300, 1]],
        [[1, 0.5], [0.3, 1]],
        [[1e-40, 2e-40], [2e-40, 1e-40]],
        [[1, 1e308], [1e308, 1]],
        [[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]],
    ],
)
def test_invalid_covariance_cannot_be_displayed_as_valid(matrix):
    with pytest.raises(ValueError):
        prepare_covariance_view(matrix)


@pytest.mark.parametrize("labels", ["ab", ["a"], ["a", " "]])
def test_labels_must_match_matrix(labels):
    with pytest.raises(ValueError, match="labels"):
        prepare_covariance_view(np.eye(2), labels)


def test_artifact_preserves_rate_identities_and_unavailable_reason(tmp_path):
    path = tmp_path / "rates.json"
    path.write_text(
        json.dumps(
            {
                "measurement_covariance": [[4]],
                "rates": [{"observation_id": "wire-1", "reaction_id": "Co59"}],
            }
        )
    )
    assert read_covariance_view(path).labels == ("wire-1",)
    path.write_text(
        json.dumps(
            {
                "covariance": None,
                "diagnostics": {"uncertainty_unavailable_reason": "No qualified sigma"},
            }
        )
    )
    with pytest.raises(ValueError, match="No qualified sigma"):
        read_covariance_view(path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"covariance": [[1]], "rate_covariance": [[1]]},
        {"covariance": None, "diagnostics": None},
        {"covariance": [[1]], "rates": [None]},
    ],
)
def test_malformed_or_ambiguous_artifacts_fail_clearly(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_covariance_view(path)
