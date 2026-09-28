"""Flux and reaction-rate quantities are never mixed or relabeled (issue #200)."""

from __future__ import annotations

import numpy as np
import pytest

from fluxforge.analysis.flux_unfold import (
    FluxWireReaction,
    extract_reactions_from_processed,
    reaction_rate_to_flux,
    unfold_discrete_bins,
    unfold_gls,
)
from fluxforge.io.flux_wire import FluxWireData, NuclideResult


def _rxn(reaction_id: str, rate: float, flux: float | None = None) -> FluxWireReaction:
    rxn = FluxWireReaction("s", reaction_id, "x", 1.0, 0.1, rate, 0.1 * rate)
    if flux is not None:
        rxn.flux = flux
    return rxn


def test_equivalent_flux_labels_are_explicit() -> None:
    _, capture = reaction_rate_to_flux(1e-12, "Co-59(n,g)Co-60")
    _, threshold = reaction_rate_to_flux(1e-15, "Ni-58(n,p)Co-58")
    assert capture == "2200m_s_equivalent"
    assert threshold == "spectrum_averaged_equivalent"


def test_cd_covered_capture_gets_no_equivalent_flux() -> None:
    data = FluxWireData(sample_id="Co-Cd-RAFM-1")
    data.real_time = data.live_time = 3600.0
    data.nuclides.append(NuclideResult("Co60", 5.2714, "y", 0.006, 0.0006, "uCi"))
    rxn = extract_reactions_from_processed(
        data, sample_mass_mg=3.67, irradiation_time_s=7200.0, report_includes_count_decay=True
    )[0]
    assert rxn.reaction_rate > 0
    assert rxn.flux == 0.0 and rxn.flux_type == "not_computed_cd_covered_capture"


def test_discrete_rates_are_an_indicator_not_flux() -> None:
    result = unfold_discrete_bins([_rxn("Co-59(n,g)Co-60", 2e-12), _rxn("Ni-58(n,p)Co-58", 3e-15)])
    assert result.quantity == "reaction_rate_indicator"
    assert np.all(np.isnan(result.flux))
    assert np.nansum(result.values) == pytest.approx(2e-12 + 3e-15)


def test_discrete_refuses_mixed_quantities() -> None:
    with pytest.raises(ValueError, match="mix"):
        unfold_discrete_bins([_rxn("Co-59(n,g)Co-60", 2e-12, flux=5e10), _rxn("Ni-58(n,p)Co-58", 3e-15)])


def test_gls_observable_is_always_reaction_rate() -> None:
    plain = [_rxn("Co-59(n,g)Co-60", 2e-12), _rxn("Ni-58(n,p)Co-58", 3e-15)]
    with_flux = [_rxn("Co-59(n,g)Co-60", 2e-12, flux=5e10), _rxn("Ni-58(n,p)Co-58", 3e-15)]
    a = unfold_gls(plain, n_groups=8)
    b = unfold_gls(with_flux, n_groups=8)
    np.testing.assert_allclose(a.flux, b.flux)
