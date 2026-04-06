from __future__ import annotations

import numpy as np
import pytest

from fluxforge.analysis.flux_wire_analysis import _window_metrics_local_background


def test_window_metrics_linear_background_tracks_linear_continuum() -> None:
    counts = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    net, net_unc, gross, background_sum = _window_metrics_local_background(
        counts,
        2,
        7,
        background_width_channels=2,
        background_model="linear",
    )

    assert gross == pytest.approx(np.sum(counts[2:8]))
    assert background_sum == pytest.approx(np.sum(counts[2:8]))
    assert net == pytest.approx(0.0)
    assert net_unc >= 0.0


def test_window_metrics_linear_background_matches_constant_sum_on_linear_background() -> (
    None
):
    counts = np.array([10.0, 11.0, 12.0, 20.0, 25.0, 24.0, 18.0, 17.0, 18.0, 19.0])
    const_net, _, _, _ = _window_metrics_local_background(
        counts,
        3,
        6,
        background_width_channels=2,
        background_model="constant",
    )
    linear_net, _, _, _ = _window_metrics_local_background(
        counts,
        3,
        6,
        background_width_channels=2,
        background_model="linear",
    )

    assert linear_net == pytest.approx(const_net)
