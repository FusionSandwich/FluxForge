"""Reference and example sample separation, preserving explicit mass gates (#31)."""

from importlib import resources
import json

import pytest

from fluxforge.analysis.flux_unfold import AVOGADRO, calculate_n_atoms
from fluxforge.data.flux_wire_unfolding import (
    load_flux_wire_nominal_samples,
    load_flux_wire_sample_defaults,
    load_flux_wire_sample_reference,
    load_flux_wire_unfolding_defaults,
)


def test_reference_resources_do_not_contain_nominal_geometry_or_masses():
    refs = load_flux_wire_sample_reference()
    assert all(
        "mass_mg" not in row and "diameter_mm" not in row for row in refs.values()
    )
    examples = load_flux_wire_nominal_samples()
    assert set(refs) == set(examples)
    assert all("reaction_target_fractions" not in row for row in examples.values())
    with (
        resources.files("fluxforge.data")
        .joinpath("flux_wire_unfolding_defaults.json")
        .open() as handle
    ):
        raw = json.load(handle)
    assert "sample_defaults" not in raw
    assert "product_reactions" not in raw


@pytest.mark.parametrize(
    "element,mass,atomic,purity",
    [
        ("Co", 10, 58.9332, 0.9999),
        ("Cu", 20, 63.546, 0.9999),
        ("Sc", 5, 44.9559, 0.999),
        ("In", 20, 114.818, 0.9999),
        ("Ti", 15, 47.867, 0.9999),
        ("Ni", 15, 58.693, 0.9999),
        ("Fe", 20, 55.845, 0.9999),
    ],
)
def test_compatibility_values_preserve_target_atoms_without_implicit_mass(
    element, mass, atomic, purity
):
    row = load_flux_wire_sample_defaults()[element]
    assert (row["mass_mg"], row["atomic_mass"], row["purity"]) == (mass, atomic, purity)
    with pytest.raises(ValueError, match="Monitor mass is required"):
        calculate_n_atoms(element)
    assert calculate_n_atoms(element, allow_default_mass=True) == pytest.approx(
        mass / 1000 * AVOGADRO / atomic * purity
    )
    assert calculate_n_atoms(
        element, mass_mg=2, element_mass_fraction=1
    ) == pytest.approx(0.002 * AVOGADRO / atomic)


def test_composite_payload_is_fresh_and_preserves_public_shape():
    payload = load_flux_wire_unfolding_defaults()
    assert payload["sample_defaults"] == load_flux_wire_sample_defaults()
    payload["sample_defaults"]["Ti"]["reaction_target_fractions"]["Ti-48(n,p)Sc-48"] = 1
    assert (
        load_flux_wire_sample_reference()["Ti"]["reaction_target_fractions"][
            "Ti-48(n,p)Sc-48"
        ]
        == 0.7372
    )


def test_nominal_sample_metadata_cannot_claim_measured_admission():
    with (
        resources.files("fluxforge.data")
        .joinpath("flux_wire_nominal_samples.json")
        .open() as handle
    ):
        payload = json.load(handle)
    assert payload["_metadata"]["scientific_admission"] is False
    assert "not measured" in payload["_metadata"]["description"]


def test_explicit_element_mass_does_not_inherit_example_purity():
    reference = load_flux_wire_sample_reference()["Sc"]
    expected = 0.002 * AVOGADRO / reference["atomic_mass"]
    assert calculate_n_atoms("Sc", mass_mg=2) == pytest.approx(expected)
    assert calculate_n_atoms(
        "Sc", mass_mg=2, element_mass_fraction=0.25
    ) == pytest.approx(expected * 0.25)
