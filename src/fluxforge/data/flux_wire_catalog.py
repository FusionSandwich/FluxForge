"""Bundled flux-wire reaction metadata."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    reactions_by_element: Dict[str, str] = field(default_factory=dict)
    reaction_ids_by_element: Dict[str, str] = field(default_factory=dict)


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
            reactions_by_element=dict(entry.get("reactions_by_element", {})),
            reaction_ids_by_element=dict(entry.get("reaction_ids_by_element", {})),
        )
        for isotope, entry in payload.items()
    }


def list_flux_wire_isotopes() -> List[str]:
    """Return bundled flux-wire isotopes in catalog order."""
    return list(load_flux_wire_catalog().keys())


def get_flux_wire_catalog_entry(
    isotope: str, sample_element: Optional[str] = None
) -> Optional[FluxWireCatalogEntry]:
    """Return product metadata, optionally resolved for a specific wire element.

    Without a context, the historical primary parent/reaction fields remain
    available. The per-element maps carry all supported production reactions.
    An explicit incompatible element never falls back to the primary parent.
    """
    entry = load_flux_wire_catalog().get(isotope)
    if entry is None or sample_element is None:
        return entry
    element = sample_element.strip().capitalize()
    reaction = entry.reactions_by_element.get(element)
    if reaction is None:
        return None
    return replace(entry, parent_element=element, reaction=reaction)


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
