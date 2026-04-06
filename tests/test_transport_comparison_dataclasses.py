from __future__ import annotations

import numpy as np

from fluxforge.validation.transport_comparison import (
    ReactionRateComparison,
    SpectrumComparison,
    TransportCode,
)


def test_spectrum_comparison_metrics_are_computed():
    energy = np.array([1e-5, 1e-2, 1.0, 10.0], dtype=float)
    unfolded = np.array([10.0, 20.0, 30.0], dtype=float)
    transport = np.array([11.0, 19.0, 29.0], dtype=float)
    unc = np.array([1.0, 1.0, 1.5], dtype=float)

    comp = SpectrumComparison(
        energy_grid=energy,
        unfolded=unfolded,
        transport=transport,
        unfolded_unc=unc,
        transport_unc=unc * 0.5,
        transport_code=TransportCode.OPENMC,
    )

    assert comp.n_groups == 3
    assert np.isfinite(comp.mean_c_over_e)
    assert np.isfinite(comp.chi2)
    assert comp.max_deviation[1] >= 0
    assert "Spectrum Comparison" in comp.summary()
    assert comp.to_dict()["transport_code"] == "openmc"


def test_reaction_rate_comparison_chi2_and_uncertainties():
    comp = ReactionRateComparison(
        reactions=["Au-197(n,g)", "Ni-58(n,p)"],
        measured=np.array([100.0, 40.0], dtype=float),
        measured_unc=np.array([5.0, 2.0], dtype=float),
        calculated=np.array([98.0, 44.0], dtype=float),
        calculated_unc=np.array([3.0, 2.5], dtype=float),
    )

    assert comp.c_over_e.shape == (2,)
    assert np.all(np.isfinite(comp.c_over_e))
    assert np.isfinite(comp.chi2)
