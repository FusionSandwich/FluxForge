"""Explicit EOI activity and specimen constraints for the reaction-rate UI."""

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from fluxforge.analysis.flux_unfold import calculate_n_atoms
from fluxforge.data.flux_wire_catalog import get_flux_wire_catalog_entry
from fluxforge.data.isotope_names import format_isotope_name, parse_nndc_isotope_name
from fluxforge.physics.activation import IrradiationSegment, reaction_rate_from_activity


@dataclass(frozen=True)
class ReactionRateInput:
    product: str
    element: str
    mass_mg: float
    element_mass_fraction: float
    isotope_fraction: float
    half_life_s: float
    activity_eoi_bq: float
    activity_sigma_bq: float | None = None


@dataclass(frozen=True)
class ReactionRateRow:
    product: str
    reaction_id: str
    target_atoms: float
    half_life_s: float
    saturation_rate_per_s: float
    sigphi_per_atom_s: float
    sigphi_activity_sigma: float | None
    uncertainty_scope: str


def _finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return number


def translate_reaction_rate(
    observation: ReactionRateInput, segments: Sequence[IrradiationSegment]
) -> ReactionRateRow:
    """Use the shared activation engine; sigma remains activity-only conditional.

    The caller must declare EOI activity and measured specimen constraints.
    Nothing infers saturation, a nominal mass, natural enrichment or purity.
    """
    if not observation.element or not observation.product:
        raise ValueError("Product and target element are required.")
    element = observation.element.strip().capitalize()
    product = format_isotope_name(
        *parse_nndc_isotope_name(observation.product), separator=""
    )
    entry = get_flux_wire_catalog_entry(product, element)
    if entry is None:
        raise ValueError(
            "Product has no catalog reaction for the supplied target element."
        )
    mass = _finite(observation.mass_mg, "Monitor mass", positive=True)
    fraction = _finite(
        observation.element_mass_fraction, "Element mass fraction", positive=True
    )
    enrichment = _finite(
        observation.isotope_fraction, "Target isotope fraction", positive=True
    )
    half_life = _finite(observation.half_life_s, "Half-life", positive=True)
    activity = _finite(observation.activity_eoi_bq, "EOI activity")
    sigma = (
        None
        if observation.activity_sigma_bq is None
        else _finite(observation.activity_sigma_bq, "Activity sigma")
    )
    if not segments:
        raise ValueError("Apply an irradiation history before converting activities.")
    atoms = calculate_n_atoms(
        element,
        mass,
        enrichment,
        element_mass_fraction=fraction,
        allow_default_mass=False,
    )
    estimate = reaction_rate_from_activity(activity, segments, half_life, sigma)
    sigphi = estimate.rate / atoms
    sigphi_sigma = (
        None if estimate.uncertainty is None else estimate.uncertainty / atoms
    )
    if (
        not math.isfinite(atoms)
        or atoms <= 0
        or not math.isfinite(sigphi)
        or (sigphi_sigma is not None and not math.isfinite(sigphi_sigma))
    ):
        raise ValueError("Target atoms and converted rates must be finite.")
    return ReactionRateRow(
        product=entry.isotope,
        reaction_id=entry.reaction_ids_by_element[element],
        target_atoms=atoms,
        half_life_s=half_life,
        saturation_rate_per_s=estimate.rate,
        sigphi_per_atom_s=sigphi,
        sigphi_activity_sigma=sigphi_sigma,
        uncertainty_scope=estimate.uncertainty_scope,
    )


def reaction_rate_payload(inputs, segments):
    if not inputs:
        raise ValueError("Add at least one activity row.")
    results = [translate_reaction_rate(row, segments) for row in inputs]
    return {
        "schema": "fluxforge.reaction_rate_editor.v1",
        "activity_reference": "end_of_irradiation",
        "rate_unit": "reactions_per_target_atom_per_s",
        "scientific_admission": False,
        "complete_uncertainty_budget": False,
        "inputs": [asdict(row) for row in inputs],
        "irradiation": {"segments": [asdict(segment) for segment in segments]},
        "rows": [asdict(row) for row in results],
    }
