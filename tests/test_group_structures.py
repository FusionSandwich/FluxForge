"""Exact standard energy group structures (issue #202)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from fluxforge.analysis.flux_unfold import make_vitamin_j_175_groups
from fluxforge.data.group_structures import (
    get_group_structure,
    group_structure_info,
    list_group_structures,
)
from fluxforge.data.irdff import IRDFFDatabase
from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

OPENMC_NAMES = [
    "CASMO-2", "CASMO-4", "CASMO-8", "CASMO-16", "CASMO-25", "ECCO-33", "CASMO-40",
    "VITAMIN-J-42", "SCALE-44", "MPACT-51", "MPACT-60", "MPACT-69", "CASMO-70",
    "XMAS-172", "VITAMIN-J-175", "SCALE-252", "TRIPOLI-315", "SHEM-361", "LLNL-616",
    "CCFE-709", "SCALE-999", "UKAEA-1102", "ECCO-1968",
]


def test_all_openmc_default_structures_present_with_named_group_counts() -> None:
    names = list_group_structures()
    for name in OPENMC_NAMES:
        assert name in names
        edges = get_group_structure(name)
        assert len(edges) - 1 == int(re.search(r"(\d+)$", name).group(1))
        assert np.all(np.diff(edges) > 0)
        assert group_structure_info(name)["source"]["commit"].startswith("3cded0f")


def test_vitamin_j_alara_structure_is_exact_njoy() -> None:
    edges = get_group_structure("VITAMIN-J")
    np.testing.assert_array_equal(edges, get_group_structure("ALARA-175"))
    assert len(edges) == 176
    assert edges[0] == 1.0e-5
    assert edges[-1] == pytest.approx(1.9640e7, rel=1e-4)
    for value in (4.1399e-1, 5.3158e-1, 7.2e4):
        assert np.any(edges == value)
    # OpenMC's copy is rounded and differs at 72 keV
    openmc = get_group_structure("VITAMIN-J-175")
    assert np.max(np.abs(openmc / edges - 1)) == pytest.approx(24 / 72000, rel=1e-6)
    np.testing.assert_array_equal(
        np.array(make_vitamin_j_175_groups().boundaries_eV), edges
    )


def test_irdff_structures_are_published_grids() -> None:
    sand = get_group_structure("SAND-II-725")
    assert len(sand) == 726 and sand[0] == 1e-5 and sand[1] == 1.05e-5 and sand[-1] == 6e7
    irdff640 = get_group_structure("IRDFF-640")
    assert len(irdff640) == 641 and irdff640[-1] == pytest.approx(2e7)
    db = IRDFFDatabase(auto_download=False)
    np.testing.assert_array_equal(db.get_energy_grid("sand725"), sand)
    np.testing.assert_array_equal(db.get_energy_grid("mcnp640"), irdff640)
    # No longer an equal-lethargy grid
    assert not np.allclose(np.diff(np.log(sand)), np.diff(np.log(sand))[0])


def test_descending_and_unknown_names() -> None:
    edges = get_group_structure("CASMO-4")
    np.testing.assert_array_equal(get_group_structure("casmo_4", descending=True), edges[::-1])
    with pytest.raises(KeyError, match="available"):
        get_group_structure("VITAMIN-Q")


def test_spectrum_unfolder_accepts_registry_names() -> None:
    assert SpectrumUnfolder(energy_structure="CCFE-709", verbose=False).n_groups == 709
    assert SpectrumUnfolder(energy_structure="VITAMIN-J", verbose=False).n_groups == 175
