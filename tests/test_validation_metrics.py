import numpy as np
import pytest

from fluxforge.validation import spectrum_comparison_metrics


def test_spectrum_comparison_metrics_basic():
    truth = np.array([1.0, 2.0, 4.0])
    predicted = np.array([1.1, 1.8, 4.2])
    metrics = spectrum_comparison_metrics(truth, predicted)

    assert metrics["mae"] == pytest.approx((0.1 + 0.2 + 0.2) / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.1**2 + 0.2**2 + 0.2**2) / 3.0))
    assert metrics["mean_ratio"] == pytest.approx(np.mean(predicted / truth))
    assert metrics["std_ratio"] == pytest.approx(np.std(predicted / truth))
    assert metrics["max_abs_residual"] == pytest.approx(0.2)
    assert metrics["corrcoef"] is not None


def test_spectrum_comparison_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spectrum_comparison_metrics(np.array([1.0, 2.0]), np.array([1.0]))
