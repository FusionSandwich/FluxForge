"""Shared isotope/nuclide name parsing and formatting helpers."""

from __future__ import annotations

import re
from typing import Pattern, Tuple


_ELEMENTS_ISOTOPE_PATTERN = re.compile(r"^([A-Z][a-z]?)(\d+)(m\d*)?$")
_NNDC_ISOTOPE_PATTERN = re.compile(r"^([A-Za-z]{1,2})-?(\d+)(m?\d?)$")
_GAMMA_NUCLIDE_PATTERN = re.compile(r"^([A-Z][a-z]?)(\d+)(m?)$")


def _parse_metastable_suffix(suffix: str, *, allow_index: bool) -> int:
    """Convert metastable suffix text into an integer state."""
    meta = suffix.lower()
    if not meta:
        return 0
    if meta == "m" or meta == "m1":
        return 1
    if allow_index and meta.startswith("m") and meta[1:].isdigit():
        return int(meta[1:])
    raise ValueError(f"Invalid metastable suffix: {suffix}")


def _parse_with_pattern(
    name: str,
    *,
    pattern: Pattern[str],
    normalize_element: bool,
    allow_meta_index: bool,
    error_message: str,
) -> Tuple[str, int, int]:
    """Parse isotope-like names with configurable matching/normalization."""
    match = pattern.match(name)
    if not match:
        raise ValueError(error_message)

    element = match.group(1)
    if normalize_element:
        element = element.capitalize()

    mass = int(match.group(2))
    metastable = _parse_metastable_suffix(
        match.group(3) or "", allow_index=allow_meta_index
    )
    return element, mass, metastable


def parse_elements_isotope_name(name: str) -> Tuple[str, int, int]:
    """Parse compact names used by ``elements.parse_isotope`` (e.g., ``U235m``)."""
    return _parse_with_pattern(
        name,
        pattern=_ELEMENTS_ISOTOPE_PATTERN,
        normalize_element=False,
        allow_meta_index=True,
        error_message=f"Cannot parse isotope name: {name}",
    )


def parse_nndc_isotope_name(name: str) -> Tuple[str, int, int]:
    """Parse flexible NNDC-style names (e.g., ``Co-60``, ``co60``, ``Tc99m``)."""
    return _parse_with_pattern(
        name.strip(),
        pattern=_NNDC_ISOTOPE_PATTERN,
        normalize_element=True,
        allow_meta_index=True,
        error_message=f"Cannot parse isotope: {name.strip()}",
    )


def parse_gamma_nuclide_name(name: str) -> Tuple[str, int, int]:
    """Parse compact nuclide names used by gamma database helpers."""
    return _parse_with_pattern(
        name,
        pattern=_GAMMA_NUCLIDE_PATTERN,
        normalize_element=False,
        allow_meta_index=False,
        error_message=f"Cannot parse nuclide name: {name}",
    )


def format_isotope_name(
    element: str, mass: int, metastable: int = 0, separator: str = "-"
) -> str:
    """Format an isotope name with optional separator and metastable suffix."""
    base = f"{element.capitalize()}{separator}{mass}"
    if metastable <= 0:
        return base
    if metastable == 1:
        return f"{base}m"
    return f"{base}m{metastable}"
