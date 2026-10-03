"""Independent target-atom/rate checks for the activity translator UI."""

from dataclasses import replace

import pytest

from fluxforge.gui.reaction_rate_view import (
    ReactionRateInput,
    reaction_rate_payload,
    translate_reaction_rate,
)
from fluxforge.physics.activation import IrradiationSegment


BASE = ReactionRateInput("Co60", "Co", 100, 0.5, 0.25, 10, 12, 3)
HISTORY = [IrradiationSegment(10, 0.25)]


def test_rate_and_sigma_use_explicit_mass_composition_and_relative_power():
    result = translate_reaction_rate(BASE, HISTORY)
    atoms = (0.1 / 58.933194) * 6.02214076e23 * 0.5 * 0.25
    assert result.target_atoms == pytest.approx(atoms)
    assert result.saturation_rate_per_s == pytest.approx(12 / (0.25 * 0.5))
    assert result.sigphi_per_atom_s == pytest.approx(96 / atoms)
    assert result.sigphi_activity_sigma == pytest.approx(24 / atoms)
    assert result.uncertainty_scope == "activity_only_conditional"
    payload = reaction_rate_payload([BASE], HISTORY)
    assert payload["complete_uncertainty_budget"] is False
    assert payload["activity_reference"] == "end_of_irradiation"


def test_zero_activity_retains_positive_supplied_sigma():
    result = translate_reaction_rate(replace(BASE, activity_eoi_bq=0), HISTORY)
    assert result.sigphi_per_atom_s == 0
    assert result.sigphi_activity_sigma > 0
    result = translate_reaction_rate(replace(BASE, activity_sigma_bq=None), HISTORY)
    assert result.sigphi_activity_sigma is None
    assert result.uncertainty_scope == "unavailable"
    assert (
        translate_reaction_rate(replace(BASE, product="co-60"), HISTORY).product
        == "Co60"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"mass_mg": ""},
        {"mass_mg": 0},
        {"element_mass_fraction": None},
        {"element_mass_fraction": 1.1},
        {"isotope_fraction": 0},
        {"isotope_fraction": float("nan")},
        {"half_life_s": 0},
        {"activity_eoi_bq": -1},
        {"activity_sigma_bq": -1},
        {"mass_mg": True},
        {"element": ""},
        {"product": "Sc46"},
    ],
)
def test_incomplete_or_incompatible_constraints_are_rejected(change):
    with pytest.raises(ValueError):
        translate_reaction_rate(replace(BASE, **change), HISTORY)


def test_history_is_required_and_catalog_resolves_titanium_scandium_product():
    with pytest.raises(ValueError, match="history"):
        translate_reaction_rate(BASE, [])
    with pytest.raises(ValueError, match="positive"):
        translate_reaction_rate(BASE, [IrradiationSegment(10, 0)])
    row = translate_reaction_rate(replace(BASE, product="Sc46", element="Ti"), HISTORY)
    assert row.reaction_id == "Ti-46(n,p)Sc-46"
