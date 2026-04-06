"""Bundled defaults for flux-wire unfolding helpers."""

from __future__ import annotations

from importlib import resources
import json
from typing import Any, Dict, Optional, Tuple


def _load_payload() -> Dict[str, Any]:
    with (
        resources.files("fluxforge.data")
        .joinpath("flux_wire_unfolding_defaults.json")
        .open(
            "r",
            encoding="utf-8",
        ) as handle
    ):
        return json.load(handle)


def load_flux_wire_unfolding_defaults() -> Dict[str, Any]:
    """Load the bundled flux-wire unfolding defaults payload."""
    return _load_payload()


def load_flux_wire_sample_defaults() -> Dict[str, Dict[str, Any]]:
    """Return bundled sample-property defaults keyed by wire element."""
    return dict(_load_payload().get("sample_defaults", {}))


def load_flux_wire_reaction_defaults() -> Dict[str, Dict[str, Any]]:
    """Return bundled reaction defaults keyed by reaction id."""
    return dict(_load_payload().get("reaction_defaults", {}))


def load_flux_wire_product_reactions() -> Dict[str, Dict[str, str]]:
    """Return bundled element/isotope -> reaction-id mappings."""
    return dict(_load_payload().get("product_reactions", {}))


def get_flux_wire_reaction_id(
    isotope: str, sample_element: Optional[str] = None
) -> str:
    """Resolve the default reaction id for a product isotope in one wire context."""
    product_reactions = load_flux_wire_product_reactions()
    if sample_element:
        mapping = product_reactions.get(sample_element, {})
        if isotope in mapping:
            return str(mapping[isotope])
    for mapping in product_reactions.values():
        if isotope in mapping:
            return str(mapping[isotope])
    return f"Unknown({isotope})"


def get_flux_wire_isotope_fraction(reaction_id: str, element: str) -> float:
    """Return the bundled target-isotope fraction for a reaction in one wire element."""
    sample_defaults = load_flux_wire_sample_defaults()
    element_defaults = sample_defaults.get(element, {})
    fractions = element_defaults.get("reaction_target_fractions", {})
    return float(fractions.get(reaction_id, 1.0))


def get_flux_wire_reaction_cross_section_defaults() -> Dict[str, Dict[str, float]]:
    """Return bundled simplified cross-section defaults keyed by reaction id."""
    return {
        reaction_id: dict(entry.get("cross_section", {}))
        for reaction_id, entry in load_flux_wire_reaction_defaults().items()
    }


def get_flux_wire_reaction_characteristic_energies() -> Dict[str, float]:
    """Return bundled characteristic energies keyed by reaction id."""
    return {
        reaction_id: float(entry.get("characteristic_energy_eV", 1.0e6))
        for reaction_id, entry in load_flux_wire_reaction_defaults().items()
    }


def get_flux_wire_response_parameters() -> Dict[str, Tuple[float, float]]:
    """Return bundled simplified response-curve parameters keyed by reaction id."""
    return {
        reaction_id: (
            float(entry.get("response_center_eV", 1.0e6)),
            float(entry.get("response_log_width", 1.0)),
        )
        for reaction_id, entry in load_flux_wire_reaction_defaults().items()
    }
