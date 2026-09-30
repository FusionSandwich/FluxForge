"""No silent physical defaults in the flux-wire reaction path (issue #204)."""

from __future__ import annotations

import numpy as np
import pytest

from fluxforge.analysis.flux_unfold import AVOGADRO, activity_to_reaction_rate, calculate_n_atoms
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


@pytest.mark.parametrize("fraction", [-0.1, 0.0, 1.1, float("nan"), float("inf")])
def test_invalid_composition_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="element_mass_fraction"):
        calculate_n_atoms("Co", mass_mg=50.0, element_mass_fraction=fraction)


@pytest.mark.parametrize("atoms,half_life", [(-1.0, 10.0), (0.0, 10.0), (float("nan"), 10.0), (1.0, 0.0)])
def test_invalid_rate_normalization_is_rejected(atoms: float, half_life: float) -> None:
    with pytest.raises(ValueError):
        activity_to_reaction_rate(
            activity_bq=1.0, n_atoms=atoms, half_life_s=half_life,
            irradiation_time_s=10.0,
        )


def test_nonfinite_activity_is_rejected() -> None:
    with pytest.raises(ValueError, match="activity_bq"):
        activity_to_reaction_rate(float("nan"), 1e20, 1e5, 10.0)


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


def test_measurement_alloy_fraction_changes_rate_and_rejects_invalid_fraction() -> None:
    common = dict(activity_Bq=100.0, sample_mass_g=0.004, irradiation_time=7200.0)
    pure = FluxWireMeasurement("Co-59(n,g)Co-60", **common, element_mass_fraction=1.0)
    alloy = FluxWireMeasurement("Co-59(n,g)Co-60", **common, element_mass_fraction=0.001)
    assert alloy.target_atom_count == pytest.approx(pure.target_atom_count * 0.001)
    assert alloy.reaction_rate_per_atom == pytest.approx(pure.reaction_rate_per_atom * 1000)
    bad = FluxWireMeasurement("Co-59(n,g)Co-60", **common, element_mass_fraction=float("nan"))
    with pytest.raises(ValueError, match="element_mass_fraction"):
        bad.reaction_rate_per_atom


def test_measurement_invalid_normalization_does_not_become_zero_or_nan() -> None:
    common = dict(activity_Bq=100.0, sample_mass_g=0.004, saturation_factor=0.5)
    with pytest.raises(ValueError, match="target isotope"):
        FluxWireMeasurement("unknown", **common).reaction_rate_per_atom
    with pytest.raises(ValueError, match="rate normalization"):
        FluxWireMeasurement(
            "Co-59(n,g)Co-60", **common, decay_factor=float("nan")
        ).reaction_rate_per_atom
    with pytest.raises(ValueError, match="rate_per_atom"):
        FluxWireMeasurement(
            "Co-59(n,g)Co-60", **common, rate_per_atom=float("nan")
        ).reaction_rate_per_atom
    for abundance in (0.0, float("nan"), 2.0):
        with pytest.raises(ValueError, match="isotope_abundance"):
            FluxWireMeasurement(
                "Co-59(n,g)Co-60", **common, isotope_abundance=abundance
            ).reaction_rate_per_atom
    with pytest.raises(ValueError, match="activity_Bq"):
        FluxWireMeasurement(
            "Co-59(n,g)Co-60", activity_Bq=float("nan"),
            sample_mass_g=0.004, saturation_factor=0.5,
        ).reaction_rate_per_atom
