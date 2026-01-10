import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from fluxforge.analysis.peakfit import estimate_background, five_point_smooth


def test_five_point_smooth_constant():
    counts = np.full(10, 10.0)
    smoothed = five_point_smooth(counts)
    assert np.allclose(smoothed, counts)


def test_five_point_smooth_short_raises():
    with pytest.raises(ValueError):
        five_point_smooth(np.array([1.0, 2.0, 3.0, 4.0]))


def test_snip_background_constant():
    counts = np.full(50, 10.0)
    background = estimate_background(np.arange(len(counts)), counts, method="snip", iterations=10)
    assert np.allclose(background, counts, atol=1e-6)
