"""Pytest configuration for FluxForge tests."""

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

collect_ignore = [
    "reference",
]

# Defensive guard in case a local developer creates a `tests/reference` tree.
collect_ignore_glob = [
    "reference/**/test_*.py",
    "reference/**/*_test.py",
]


@pytest.fixture
def independent_background_sum_variance():
    """Direct source-bin overlap oracle, independent of the production sparse W."""

    def variance(sample, background, weights, scale):
        active = np.flatnonzero(weights)
        sample_variance = np.sum((weights * sample.counts_uncertainty) ** 2)

        def edges(centers):
            return np.r_[
                centers[0] - (centers[1] - centers[0]) / 2,
                (centers[:-1] + centers[1:]) / 2,
                centers[-1] + (centers[-1] - centers[-2]) / 2,
            ]

        target_edges = edges(sample.energies)
        source_edges = edges(background.energies)
        background_variance = 0.0
        for source_channel, uncertainty in enumerate(background.counts_uncertainty):
            mapped_basis = np.maximum(
                0.0,
                np.minimum(target_edges[active + 1], source_edges[source_channel + 1])
                - np.maximum(target_edges[active], source_edges[source_channel]),
            ) / (source_edges[source_channel + 1] - source_edges[source_channel])
            coefficient = weights[active] @ mapped_basis
            background_variance += (coefficient * uncertainty) ** 2
        return float(sample_variance + scale**2 * background_variance)

    return variance
