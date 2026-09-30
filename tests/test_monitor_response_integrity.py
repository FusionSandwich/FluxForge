"""Regression contracts for physical row identity and validation (#208/#209)."""

from dataclasses import replace

import numpy as np
import pytest

from tests.test_monitor_response import EDGES, mini_db, _unfolder
from fluxforge.physics.monitor_response import (
    CoverLayer,
    MonitorShielding,
    MonitorResponseSpec,
    build_monitor_response,
    cover_transmission,
    self_shielding_factor,
)
from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder


def body(**kwargs):
    fields = dict(
        geometry="slab",
        dimension_cm=0.05,
        number_density_per_cm3=1e22,
        total_energies_eV=(1e-5, 2e7),
        total_cross_section_barn=(1.0, 2.0),
        total_source="synthetic regression",
    )
    fields.update(kwargs)
    return MonitorShielding(**fields)


@pytest.mark.parametrize(
    "change",
    [
        dict(density_g_cm3=4.325),
        dict(atomic_mass=120.0),
        dict(thickness_cm=np.nextafter(0.0508, 1.0)),
        dict(thickness_unc_cm=0.001),
    ],
)
def test_complete_cover_identity(change):
    original = CoverLayer("Cd", 0.0508)
    assert original.key != replace(original, **change).key
    assert (
        original.key
        == CoverLayer("Cd", 0.0508, density_g_cm3=8.65, atomic_mass=112.414).key
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(number_density_per_cm3=2e22),
        dict(total_energies_eV=(1e-5, 1e7)),
        dict(dimension_cm=np.nextafter(0.05, 1.0)),
        dict(dimension_unc_cm=0.001),
        dict(total_cross_section_barn=(1.0, 3.0)),
        dict(total_source="different provenance"),
    ],
)
def test_complete_body_identity(change):
    assert body().key != replace(body(), **change).key


@pytest.mark.parametrize(
    "change",
    [dict(density_g_cm3=4.325), dict(atomic_mass=120.0), dict(thickness_unc_cm=0.001)],
)
def test_grouped_cover_rows_remain_separate(mini_db, change):
    cover = CoverLayer("Cd", 0.0508)
    specs = [
        MonitorResponseSpec(str(i), str(i), "Co-59(n,g)Co-60", cover=c)
        for i, c in enumerate([cover, replace(cover, **change)])
    ]
    rows = [build_monitor_response(s, EDGES, mini_db) for s in specs]
    result = SpectrumUnfolder.__new__(
        SpectrumUnfolder
    )._aggregate_duplicate_reaction_rows(
        np.array([r.group_cross_section_barn for r in rows]),
        [s.reaction for s in specs],
        np.array([1.0, 2.0]),
        np.ones(2),
        np.array([r.group_uncertainty_barn for r in rows]),
        row_keys=[s.physics_key for s in specs],
    )
    assert result["response"].shape[0] == 2
    np.testing.assert_array_equal(result["measurements"], [1.0, 2.0])


@pytest.mark.parametrize("different", ["response", "uncertainty"])
def test_stale_key_cannot_merge_incompatible_rows(different):
    rows = np.array([[1e-24, 2e-24], [1e-24, 2e-24]])
    unc = rows * 0.1
    (rows if different == "response" else unc)[1, 0] = np.nextafter(
        (rows if different == "response" else unc)[1, 0], 1.0
    )
    result = SpectrumUnfolder.__new__(
        SpectrumUnfolder
    )._aggregate_duplicate_reaction_rows(
        rows,
        ["r", "r"],
        np.array([1.0, 2.0]),
        np.ones(2),
        unc,
        row_keys=["stale", "stale"],
    )
    assert result["response"].shape[0] == 2
    np.testing.assert_array_equal(result["response_uncertainties"], unc)


def test_valid_replicates_preserve_membership():
    rows = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]])
    result = SpectrumUnfolder.__new__(
        SpectrumUnfolder
    )._aggregate_duplicate_reaction_rows(
        rows,
        ["r"] * 3,
        np.array([2.0, 4.0, 9.0]),
        np.array([1.0, 2.0, 1.0]),
        rows * 0.1,
        row_keys=["replicate"] * 3,
        observation_ids=["a", "b", "c"],
    )
    np.testing.assert_array_equal(result["response"], rows[[0, 2]])
    np.testing.assert_allclose(result["measurements"], [2.4, 9.0])
    np.testing.assert_allclose(result["uncertainties"], [np.sqrt(0.8), 1.0])
    assert result["metadata"]["memberships"] == [
        {"row_indices": [0, 1], "observation_ids": ["a", "b"]},
        {"row_indices": [2], "observation_ids": ["c"]},
    ]


@pytest.mark.parametrize("field", ["thickness_cm", "density_g_cm3", "atomic_mass"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_cover_through_grouped_response(mini_db, field, value):
    with pytest.raises(ValueError):
        cover = replace(CoverLayer("Cd", 0.0508), **{field: value})
        build_monitor_response(
            MonitorResponseSpec("x", "x", "Co-59(n,g)Co-60", cover=cover),
            EDGES,
            mini_db,
        )


@pytest.mark.parametrize("field", ["dimension_cm", "number_density_per_cm3"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_body_through_grouped_response(mini_db, field, value):
    with pytest.raises(ValueError):
        shielding = body(**{field: value})
        build_monitor_response(
            MonitorResponseSpec("x", "x", "Co-59(n,g)Co-60", shielding=shielding),
            EDGES,
            mini_db,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_uncertainty_bounds(mini_db, value):
    for constructor in [
        lambda: CoverLayer("Cd", 0.0508, thickness_unc_cm=value),
        lambda: body(dimension_unc_cm=value),
    ]:
        with pytest.raises(ValueError):
            constructor()


@pytest.mark.parametrize(
    "change",
    [
        dict(total_energies_eV=(1.0, 1.0)),
        dict(total_energies_eV=(2.0, 1.0)),
        dict(total_energies_eV=(0.0, 1.0)),
        dict(total_energies_eV=(-1.0, 1.0)),
        dict(total_energies_eV=(float("nan"), 1.0)),
        dict(total_energies_eV=(1.0, float("inf"))),
        dict(total_energies_eV=((1.0, 2.0), (3.0, 4.0))),
        dict(total_energies_eV=(1.0,)),
        dict(total_cross_section_barn=(-1.0, 0.0)),
        dict(total_cross_section_barn=(float("nan"), 0.0)),
        dict(total_cross_section_barn=(0.0, float("inf"))),
        dict(total_cross_section_barn=((1.0, 2.0), (3.0, 4.0))),
    ],
)
def test_invalid_total_table(mini_db, change):
    with pytest.raises(ValueError):
        shielding = body(**change)
        build_monitor_response(
            MonitorResponseSpec("x", "x", "Co-59(n,g)Co-60", shielding=shielding),
            EDGES,
            mini_db,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_kernels_reject_invalid_inputs(value):
    with pytest.raises(ValueError):
        cover_transmission(np.array([0.0, value]))
    with pytest.raises(ValueError):
        self_shielding_factor(np.array([0.0, value]), body())


def test_zero_kernel_limits_and_invalid_model():
    np.testing.assert_array_equal(cover_transmission(np.zeros(3)), np.ones(3))
    np.testing.assert_array_equal(
        self_shielding_factor(np.zeros(3), body()), np.ones(3)
    )
    with pytest.raises(ValueError):
        cover_transmission(np.ones(2), "typo")


@pytest.mark.parametrize(
    "edges", [[1.0, float("nan"), 3.0], [1.0, float("inf")], [-1.0, 2.0], [0.0, 2.0]]
)
def test_invalid_group_edges(mini_db, edges):
    with pytest.raises(ValueError):
        build_monitor_response(
            MonitorResponseSpec("x", "x", "Co-59(n,g)Co-60"), edges, mini_db
        )


def test_bad_evaluated_cover_rejected_in_workflow(mini_db):
    cover_xs = mini_db.get_cover_cross_section("Cd", "disap")
    cover_xs.cross_sections[0] = -1.0
    unfolder = _unfolder(mini_db)
    spec = MonitorResponseSpec(
        "x", "x", "Co-59(n,g)Co-60", cover=CoverLayer("Cd", 0.0508)
    )
    unfolder.add_reaction(
        spec.reaction, 1.0, 0.05, rate_per_atom=1e-12, response_spec=spec
    )
    with pytest.raises(ValueError):
        unfolder.unfold(method="MLEM", max_iterations=2)


@pytest.mark.parametrize(
    "change",
    [
        dict(number_density_per_cm3=2e22),
        dict(total_energies_eV=(1e-5, 1e7)),
        dict(dimension_unc_cm=0.001),
    ],
)
def test_grouped_body_models_remain_distinct(mini_db, change):
    specs = [
        MonitorResponseSpec(str(i), str(i), "Co-59(n,g)Co-60", shielding=s)
        for i, s in enumerate([body(), replace(body(), **change)])
    ]
    unfolder = _unfolder(mini_db)
    for spec in specs:
        unfolder.add_reaction(
            spec.reaction, 1.0, 0.05, rate_per_atom=1e-12, response_spec=spec
        )
    result = unfolder.unfold(
        method="MLEM", max_iterations=2, aggregate_duplicate_reactions=True
    )
    assert result.response_matrix.shape[0] == 2
    assert [
        m["observation_ids"] for m in result.metadata["duplicate_reaction_memberships"]
    ] == [["0"], ["1"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dimension_cm", float("nan")),
        ("number_density_per_cm3", float("inf")),
        ("dimension_unc_cm", -1.0),
        ("total_cross_section_barn", (-1.0, 1.0)),
        ("total_energies_eV", (2.0, 1.0)),
    ],
)
def test_mutated_body_rejected_at_workflow_boundary(mini_db, field, value):
    shielding = body()
    object.__setattr__(shielding, field, value)
    spec = MonitorResponseSpec("x", "x", "Co-59(n,g)Co-60", shielding=shielding)
    unfolder = _unfolder(mini_db)
    unfolder.add_reaction(
        spec.reaction, 1.0, 0.05, rate_per_atom=1e-12, response_spec=spec
    )
    with pytest.raises(ValueError):
        unfolder.unfold(method="GRAVEL", max_iterations=2)


def test_zero_total_table_preserves_bare_row(mini_db):
    bare = MonitorResponseSpec("a", "a", "Co-59(n,g)Co-60")
    shielded = replace(bare, shielding=body(total_cross_section_barn=(0.0, 0.0)))
    np.testing.assert_array_equal(
        build_monitor_response(bare, EDGES, mini_db).group_cross_section_barn,
        build_monitor_response(shielded, EDGES, mini_db).group_cross_section_barn,
    )


def test_equal_total_uncertainty_different_models_do_not_merge():
    rows = np.ones((2, 2))
    result = SpectrumUnfolder.__new__(
        SpectrumUnfolder
    )._aggregate_duplicate_reaction_rows(
        rows,
        ["r"] * 2,
        np.ones(2),
        np.ones(2),
        rows * 0.1,
        uncertainty_models=[{"cover": [0.1, 0.1]}, {"reaction": [0.1, 0.1]}],
    )
    assert result["response"].shape[0] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("cross_sections", -1.0),
        ("uncertainties", float("nan")),
        ("uncertainties", -1.0),
        ("energies", float("inf")),
    ],
)
def test_invalid_reaction_table_rejected_at_build(mini_db, field, value):
    xs = mini_db.get_cross_section("Co-59(n,g)Co-60")
    getattr(xs, field)[0] = value
    with pytest.raises(ValueError):
        build_monitor_response(
            MonitorResponseSpec("x", "x", xs.reaction), EDGES, mini_db
        )


def test_body_owns_input_arrays():
    energies, values = np.array([1e-5, 2e7]), np.array([1.0, 2.0])
    shielding = body(total_energies_eV=energies, total_cross_section_barn=values)
    key = shielding.key
    values[0] = -1.0
    energies[0] = float("nan")
    assert shielding.key == key
    assert shielding.total_cross_section_barn == (1.0, 2.0)
