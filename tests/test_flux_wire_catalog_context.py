"""Regression coverage for shared products in distinct wire contexts (#29)."""

import pytest

from fluxforge.data.flux_wire_catalog import (
    get_flux_wire_catalog_entry,
    get_flux_wire_isotopes_for_element,
    load_flux_wire_catalog,
)
from fluxforge.data.flux_wire_unfolding import (
    get_flux_wire_reaction_id,
    get_flux_wire_isotope_fraction,
    load_flux_wire_product_reactions,
)
from fluxforge.data.rafm_decay import get_rafm_decay_entry


def test_sc46_context_is_explicit_and_does_not_fall_back():
    sc = get_flux_wire_catalog_entry("Sc46", "Sc")
    ti = get_flux_wire_catalog_entry("Sc46", " ti ")
    assert (sc.parent_element, sc.reaction) == ("Sc", "Sc45(n,g)")
    assert (ti.parent_element, ti.reaction) == ("Ti", "Ti46(n,p)")
    assert get_flux_wire_catalog_entry("Sc46", "Co") is None
    assert get_flux_wire_reaction_id("Sc46", "Co") == "Unknown(Sc46)"
    with pytest.raises(ValueError, match="Wire element is required"):
        get_flux_wire_reaction_id("Sc46")


@pytest.mark.parametrize(
    "element,product,reaction",
    [
        ("Co", "Co60", "Co-59(n,g)Co-60"),
        ("Sc", "Sc46", "Sc-45(n,g)Sc-46"),
        ("Ti", "Sc46", "Ti-46(n,p)Sc-46"),
        ("Ti", "Sc47", "Ti-47(n,p)Sc-47"),
        ("Ti", "Sc48", "Ti-48(n,p)Sc-48"),
        ("Ti", "Ti51", "Ti-50(n,g)Ti-51"),
        ("Ni", "Co58", "Ni-58(n,p)Co-58"),
        ("Ni", "Ni57", "Ni-58(n,2n)Ni-57"),
        ("Cu", "Cu64", "Cu-63(n,g)Cu-64"),
        ("In", "In114m", "In-113(n,g)In-114m"),
        ("In", "In115m", "In-115(n,n')In-115m"),
        ("Fe", "Fe59", "Fe-58(n,g)Fe-59"),
        ("Fe", "Mn54", "Fe-54(n,p)Mn-54"),
    ],
)
def test_each_bundled_wire_product_has_consistent_reaction_and_decay(
    element, product, reaction
):
    assert product in get_flux_wire_isotopes_for_element(element)
    entry = get_flux_wire_catalog_entry(product, element)
    assert entry.reaction_ids_by_element[element] == reaction
    assert get_flux_wire_reaction_id(product, element) == reaction
    decay = get_rafm_decay_entry(product)
    assert decay is not None
    for target in entry.target_lines_keV:
        assert any(
            abs(line["energy_keV"] - target) <= 2 for line in decay["gamma_lines"]
        )


def test_catalog_maps_cover_every_expected_element_and_are_fresh():
    mappings = load_flux_wire_product_reactions()
    for product, entry in load_flux_wire_catalog().items():
        assert set(entry.expected_elements) == set(entry.reactions_by_element)
        assert set(entry.expected_elements) == set(entry.reaction_ids_by_element)
        for element in entry.expected_elements:
            assert mappings[element][product] == entry.reaction_ids_by_element[element]
    mappings["Ti"]["Sc46"] = "wrong"
    assert get_flux_wire_reaction_id("Sc46", "Ti") == "Ti-46(n,p)Sc-46"


def test_spectroscopy_metadata_preserves_both_sc46_contexts():
    from fluxforge.analysis.flux_wire_analysis import FLUX_WIRE_NUCLIDES

    assert FLUX_WIRE_NUCLIDES["Sc46"]["reactions_by_element"] == {
        "Sc": "Sc45(n,g)",
        "Ti": "Ti46(n,p)",
    }


def test_ti51_uses_natural_ti50_fraction_and_unknown_fraction_is_unavailable():
    assert get_flux_wire_isotope_fraction("Ti-50(n,g)Ti-51", "Ti") == 0.0518
    with pytest.raises(ValueError, match="No target-isotope fraction"):
        get_flux_wire_isotope_fraction("Sc-45(n,g)Sc-46", "Ti")
