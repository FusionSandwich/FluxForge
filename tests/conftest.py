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
    """Oracle based on np.interp of source basis vectors, without sparse W."""

    def variance(sample, background, weights, scale):
        active = np.flatnonzero(weights)
        sample_variance = np.sum((weights * sample.counts_uncertainty) ** 2)
        source_basis = np.zeros(len(background.counts))
        background_variance = 0.0
        for source_channel, uncertainty in enumerate(background.counts_uncertainty):
            source_basis[source_channel] = 1.0
            mapped_basis = np.interp(
                sample.energies[active],
                background.energies,
                source_basis,
                left=0.0,
                right=0.0,
            )
            coefficient = weights[active] @ mapped_basis
            background_variance += (coefficient * uncertainty) ** 2
            source_basis[source_channel] = 0.0
        return float(sample_variance + scale**2 * background_variance)

    return variance
