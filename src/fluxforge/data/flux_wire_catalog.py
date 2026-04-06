"""Bundled flux-wire reaction metadata."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FluxWireCatalogEntry:
    """Committed flux-wire reaction metadata for one activation product."""

    isotope: str
    parent_element: str
    reaction: str
    target_lines_keV: List[float]
    expected_elements: List[str]


def _load_catalog_payload() -> Dict[str, Any]:
    with (
        resources.files("fluxforge.data")
        .joinpath("flux_wire_catalog.json")
        .open(
            "r",
            encoding="utf-8",
        ) as handle
    ):
        return json.load(handle)


def load_flux_wire_catalog() -> Dict[str, FluxWireCatalogEntry]:
    """Load bundled flux-wire reaction metadata."""
    payload = _load_catalog_payload().get("isotopes", {})
    return {
        isotope: FluxWireCatalogEntry(
            isotope=isotope,
            parent_element=str(entry["parent_element"]),
            reaction=str(entry["reaction"]),
            target_lines_keV=[
                float(value) for value in entry.get("target_lines_keV", [])
            ],
            expected_elements=[
                str(value)
                for value in entry.get("expected_elements", [entry["parent_element"]])
            ],
        )
        for isotope, entry in payload.items()
    }


def list_flux_wire_isotopes() -> List[str]:
    """Return bundled flux-wire isotopes in catalog order."""
    return list(load_flux_wire_catalog().keys())


def get_flux_wire_catalog_entry(isotope: str) -> Optional[FluxWireCatalogEntry]:
    """Return reaction metadata for one flux-wire isotope."""
    return load_flux_wire_catalog().get(isotope)


def get_flux_wire_isotopes_for_element(element: str) -> List[str]:
    """Return expected activation products for a parent wire element."""
    normalized = element[:1].upper() + element[1:].lower() if element else ""
    return [
        isotope
        for isotope, entry in load_flux_wire_catalog().items()
        if normalized in entry.expected_elements
    ]


def list_flux_wire_elements() -> List[str]:
    """Return sorted parent elements present in the bundled flux-wire catalog."""
    return sorted(
        {
            element
            for entry in load_flux_wire_catalog().values()
            for element in entry.expected_elements
        }
    )
