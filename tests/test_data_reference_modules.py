"""Tests for data helper modules with previously low/zero coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluxforge.data.materials import (
    Material,
    effective_atomic_number,
    get_element_data,
    get_material,
    list_elements,
    list_materials,
    search_materials,
)
from fluxforge.data.flux_wire_catalog import (
    FluxWireCatalogEntry,
    get_flux_wire_catalog_entry,
    get_flux_wire_isotopes_for_element,
    list_flux_wire_elements,
    list_flux_wire_isotopes,
    load_flux_wire_catalog,
)
from fluxforge.data.flux_wire_unfolding import (
    get_flux_wire_isotope_fraction,
    get_flux_wire_reaction_characteristic_energies,
    get_flux_wire_reaction_cross_section_defaults,
    get_flux_wire_reaction_id,
    get_flux_wire_response_parameters,
    load_flux_wire_product_reactions,
    load_flux_wire_sample_defaults,
)
from fluxforge.data.nndc import (
    Isotope,
    IsotopeQuantity,
    canonical_isotope,
    format_isotope,
    get_nuclear_data,
    list_isotopes_with_gammas,
    parse_isotope,
)
from fluxforge.data.thermal_scattering import (
    ThermalScatteringConfig,
    ThermalScatteringData,
    TSLLibrary,
    get_mcnp_sab_card,
    get_openmc_tsl_name,
    get_tsl_for_material,
    requires_thermal_scattering,
)
from fluxforge.data.rafm_decay import get_rafm_decay_entry, get_rafm_gamma_lines


def test_material_lookup_and_utilities() -> None:
    mat = get_material("Concrete, Ordinary")
    assert isinstance(mat, Material)
    assert mat.validate()
    assert "Si" in mat.elements
    assert mat.weight_fractions.ndim == 1

    # Case-insensitive lookup and elemental fallback path.
    mat_lower = get_material("concrete, ordinary")
    assert mat_lower.name == mat.name
    fe = get_material("Fe")
    assert fe.composition == {"Fe": 1.0}
    assert fe.validate()

    materials = list_materials()
    assert "Concrete, Ordinary" in materials
    assert "Fe" in list_elements()

    elem = get_element_data("Fe")
    assert elem["Z"] == 26
    assert search_materials("concrete")

    with pytest.raises(ValueError):
        get_element_data("Xx")
    with pytest.raises(ValueError):
        get_material("this-does-not-exist")


def test_effective_atomic_number() -> None:
    # Pure element should reproduce its Z.
    assert abs(effective_atomic_number({"Fe": 1.0}) - 26.0) < 1e-12
    # Mixed composition remains within bounds.
    zeff = effective_atomic_number({"H": 0.5, "O": 0.5})
    assert 1.0 < zeff < 8.0


def test_flux_wire_catalog_and_decay_subset_are_data_backed() -> None:
    catalog = load_flux_wire_catalog()
    assert catalog
    assert isinstance(catalog["Co60"], FluxWireCatalogEntry)
    assert get_flux_wire_catalog_entry("In114m") is not None
    assert get_flux_wire_catalog_entry("not-a-wire") is None

    assert "Ti" in list_flux_wire_elements()
    assert get_flux_wire_isotopes_for_element("Ti") == ["Sc46", "Sc47", "Sc48", "Ti51"]
    assert "Sc46" in get_flux_wire_isotopes_for_element("Sc")
    assert "Co60" in list_flux_wire_isotopes()

    in114m = get_rafm_decay_entry("In114m")
    assert in114m is not None
    lines = get_rafm_gamma_lines("In114m")
    assert lines
    assert any(line["energy_keV"] == pytest.approx(190.34, abs=1e-3) for line in lines)
    assert all("intensity_uncertainty" in line for line in lines)
    assert all(line["intensity"] > 0.0 for line in lines)


def test_flux_wire_unfolding_defaults_are_data_backed() -> None:
    sample_defaults = load_flux_wire_sample_defaults()
    assert sample_defaults["Ti"]["reaction_target_fractions"][
        "Ti-48(n,p)Sc-48"
    ] == pytest.approx(0.7372)

    product_reactions = load_flux_wire_product_reactions()
    assert product_reactions["Ti"]["Sc46"] == "Ti-46(n,p)Sc-46"

    assert get_flux_wire_reaction_id("Sc46", "Ti") == "Ti-46(n,p)Sc-46"
    assert get_flux_wire_reaction_id("Sc46", "Sc") == "Sc-45(n,g)Sc-46"
    assert get_flux_wire_reaction_id("missing", "Ti").startswith("Unknown(")

    assert get_flux_wire_isotope_fraction("Ti-46(n,p)Sc-46", "Ti") == pytest.approx(
        0.0825
    )
    assert get_flux_wire_isotope_fraction("Co-59(n,g)Co-60", "Co") == pytest.approx(1.0)

    xs = get_flux_wire_reaction_cross_section_defaults()
    assert xs["Co-59(n,g)Co-60"]["sigma_thermal"] == pytest.approx(37.2)

    e_char = get_flux_wire_reaction_characteristic_energies()
    assert e_char["In-113(n,g)In-114m"] == pytest.approx(1.45)

    response_params = get_flux_wire_response_parameters()
    assert response_params["Ni-58(n,2n)Ni-57"] == pytest.approx((1.2e7, 0.5))


def test_thermal_scattering_functions_and_config() -> None:
    tsl = get_tsl_for_material("water")
    assert tsl is not None
    assert tsl.material == "H2O"
    assert tsl.bound_nuclide == "H"
    assert tsl.nearest_temperature(300.0) == 293.6
    low, high = tsl.interpolation_temperatures(375.0)
    assert low is not None and high is not None and low <= 375.0 <= high

    # Material alias and element/context behavior.
    assert requires_thermal_scattering("H1", "H2O")
    assert not requires_thermal_scattering("Na23")
    assert requires_thermal_scattering("C12")

    sab = get_mcnp_sab_card("H2O", 293.6)
    assert sab is not None and sab.endswith(".10t")
    assert get_openmc_tsl_name("graphite") == "c_C_in_graphite"
    assert get_tsl_for_material("not-a-material") is None

    cfg = ThermalScatteringConfig.default()
    assert cfg.enabled
    assert "H2O" in cfg.materials
    cfg.add_material("graphite")
    assert "graphite" in cfg.materials
    cfg.add_material("custom_mat", tsl_id="x_custom")
    assert cfg.materials["custom_mat"].library == TSLLibrary.CUSTOM
    assert cfg.get_njoy_thermr_inputs("H2O")["ntemp"] == 1
    assert cfg.get_njoy_thermr_inputs("unknown") == {}

    # Empty temperature list edge behavior.
    custom = ThermalScatteringData(
        material="dummy",
        bound_nuclide="H",
        tsl_id="d",
        library=TSLLibrary.CUSTOM,
        temperatures_K=[],
    )
    assert custom.nearest_temperature(350.0) == 350.0
    assert custom.interpolation_temperatures(350.0) == (None, None)


def test_nndc_isotope_parsing_and_properties() -> None:
    assert parse_isotope("co-60") == ("Co", 60, 0)
    assert parse_isotope("Tc99m") == ("Tc", 99, 1)
    assert format_isotope("co", 60) == "Co-60"
    assert canonical_isotope("tc99m") == "Tc-99m"

    co60 = Isotope.from_string("Co-60")
    assert co60.name == "Co-60"
    assert not co60.is_stable
    assert co60.half_life_s is not None and co60.half_life_s > 0
    assert co60.main_gamma_keV is not None
    assert co60.specific_activity_Bq_g is not None and co60.specific_activity_Bq_g > 0
    assert "Co-60" in repr(co60)

    ni58 = Isotope.from_string("Ni-58")
    assert ni58.is_stable
    assert ni58.decay_constant == 0.0

    with pytest.raises(ValueError):
        parse_isotope("bad-input")


def test_nndc_isotope_quantity_and_queries() -> None:
    co60 = Isotope.from_string("Co-60")
    q = IsotopeQuantity(isotope=co60, activity_Bq=1000.0, reference_time=0.0)
    assert q.atoms > 0.0

    # Decay over time must reduce activity for unstable isotope.
    a_later = q.activity_at(3600.0)
    assert 0.0 < a_later < q.activity_Bq
    assert q.atoms_at(3600.0) < q.atoms
    assert q.decays_in_interval(0.0, 3600.0) > 0.0
    assert q.average_activity(0.0, 3600.0) > 0.0
    assert q.time_when(q.activity_Bq * 0.5) > 0.0

    # Stable isotope edge behavior.
    ni58 = Isotope.from_string("Ni-58")
    stable_q = IsotopeQuantity(isotope=ni58, atoms=1e6, reference_time=10.0)
    assert stable_q.decays_in_interval(10.0, 20.0) == 0.0
    assert stable_q.time_when(1.0) == stable_q.reference_time
    assert math.isinf(stable_q.time_when(0.0))
    assert stable_q.average_activity(10.0, 10.0) == 0.0

    # from_decays constructor.
    q_from_decays = IsotopeQuantity.from_decays(
        "Co-60", decays=1000.0, start_time=0.0, end_time=100.0
    )
    assert q_from_decays.activity_Bq > 0.0

    data = get_nuclear_data("Co-60")
    assert data["isotope"] == "Co-60"
    assert "gamma_lines" in data
    assert "decay_modes" in data

    gamma_list = list_isotopes_with_gammas(e_min=800, e_max=900, i_min=0.5)
    assert any(item["isotope"] == "Mn-56" for item in gamma_list)
