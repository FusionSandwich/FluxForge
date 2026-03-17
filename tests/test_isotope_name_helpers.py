"""Regression tests for shared isotope/nuclide name parsing helpers."""

from __future__ import annotations

import pytest

from fluxforge.data.elements import parse_isotope as parse_elements_isotope
from fluxforge.data.gamma_database import parse_nuclide_name, zai_to_name
from fluxforge.data.nndc import format_isotope, parse_isotope as parse_nndc_isotope


def test_elements_parse_isotope_preserves_compact_rules() -> None:
    assert parse_elements_isotope("U235m2") == ("U", 235, 2)

    with pytest.raises(ValueError):
        parse_elements_isotope("co60")
    with pytest.raises(ValueError):
        parse_elements_isotope("Co-60")


def test_nndc_parse_and_format_flexible_rules() -> None:
    assert parse_nndc_isotope("co-60") == ("Co", 60, 0)
    assert parse_nndc_isotope("TC99m") == ("Tc", 99, 1)
    assert format_isotope("tc", 99, 2) == "Tc-99m2"


def test_gamma_parse_nuclide_name_remains_strict() -> None:
    assert parse_nuclide_name("Co60") == ("Co", 60, 0)
    assert parse_nuclide_name("Tc99m") == ("Tc", 99, 1)

    with pytest.raises(ValueError):
        parse_nuclide_name("co60")
    with pytest.raises(ValueError):
        parse_nuclide_name("Tc99m2")


def test_gamma_zai_to_name_uses_compact_suffixes() -> None:
    assert zai_to_name(270601) == "Co60m"
    assert zai_to_name(270602) == "Co60m2"
