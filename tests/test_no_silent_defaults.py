"""No silent physical defaults in the flux-wire reaction path (issue #204)."""

from __future__ import annotations

import numpy as np
import pytest

from fluxforge.analysis.flux_unfold import AVOGADRO, calculate_n_atoms
from fluxforge.io.flux_wire import NuclideResult
from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement


def test_n_atoms_requires_element_and_mass() -> None:
    with pytest.raises(ValueError, match="element"):
        calculate_n_atoms(None, mass_mg=4.0)
    with pytest.raises(ValueError, match="mass"):
        calculate_n_atoms("Co")
    assert calculate_n_atoms("Co", allow_default_mass=True) > 0


def test_alloy_mass_fraction() -> None:
    pure = calculate_n_atoms("Co", mass_mg=50.0, element_mass_fraction=1.0)
    dilute = calculate_n_atoms("Co", mass_mg=50.0, element_mass_fraction=0.001)
    assert dilute == pytest.approx(pure * 0.001)
    assert pure == pytest.approx(0.050 * AVOGADRO / 58.9332)


def test_uncertainty_uses_same_unit_conversion_as_activity() -> None:
    nci = NuclideResult("Co60", 5.27, "y", 200.0, 10.0, "nCi")
    assert nci.activity_bq == pytest.approx(7400.0)
    assert nci.activity_unc_bq == pytest.approx(370.0)
    with pytest.raises(ValueError, match="Unknown activity unit"):
        NuclideResult("Co60", 5.27, "y", 1.0, 0.1, "dpm").activity_bq


def test_measurement_needs_mass_and_timing_to_become_a_rate() -> None:
    no_mass = FluxWireMeasurement("Co-59(n,g)Co-60", activity_Bq=100.0, irradiation_time=7200.0)
    with pytest.raises(ValueError, match="sample_mass_g"):
        no_mass.reaction_rate_per_atom
    no_timing = FluxWireMeasurement("Co-59(n,g)Co-60", activity_Bq=100.0, sample_mass_g=0.004)
    with pytest.raises(ValueError, match="saturation is never assumed"):
        no_timing.reaction_rate_per_atom
    ok = FluxWireMeasurement(
        "Co-59(n,g)Co-60", activity_Bq=100.0, sample_mass_g=0.004, irradiation_time=7200.0
    )
    assert np.isfinite(ok.reaction_rate_per_atom) and ok.reaction_rate_per_atom > 0
