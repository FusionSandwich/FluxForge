"""Physical monitor response rows: cover transmission and self-shielding (issue #196)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.special import expn

from fluxforge.data.irdff import (
    DEFAULT_CACHE_DIR,
    IRDFF_ABS_ARCHIVE_NAME,
    IRDFF_TAB_ARCHIVE_NAME,
    IRDFFDatabase,
)
from fluxforge.physics.monitor_response import (
    CoverLayer,
    MonitorResponseSpec,
    MonitorShielding,
    build_monitor_response,
    build_monitor_response_matrix,
    cover_transmission,
    self_shielding_factor,
)
from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

TAB = """\
Co-59(n,g)
 1.000000-5 2.0000E+03 2.0000E+01                                        1.00
 2.530000-2 3.7000E+01 3.7000E-01                                        1.00
 1.000000+0 6.0000E+00 6.0000E-02                                        1.00
 1.000000+2 1.0000E+00 1.0000E-02                                        1.00
 1.300000+2 8.0000E+02 8.0000E+00                                        1.00
 1.400000+2 1.0000E+00 1.0000E-02                                        1.00
 2.000000+7 1.0000E-04 1.0000E-06                                        1.00
"""
# Crude Cd absorber: large below 0.5 eV, small above
ABS = """\
Cd(n,disap)
 1.000000-5 1.0000E+05
 4.000000-1 3.0000E+03
 6.000000-1 1.0000E+00
 2.000000+7 1.0000E-03
"""


@pytest.fixture()
def mini_db(tmp_path: Path) -> IRDFFDatabase:
    (tmp_path / IRDFF_TAB_ARCHIVE_NAME).write_text(TAB, encoding="ascii")
    (tmp_path / IRDFF_ABS_ARCHIVE_NAME).write_text(ABS, encoding="ascii")
    return IRDFFDatabase(
        cache_dir=tmp_path, auto_download=False, archive_path=tmp_path / IRDFF_TAB_ARCHIVE_NAME
    )


EDGES = np.array([1e-5, 0.1, 0.5, 1.0, 100.0, 1e4, 2e7])


def test_cover_transmission_isotropic_and_beam() -> None:
    tau = np.array([0.0, 0.1, 1.0, 5.0])
    np.testing.assert_allclose(cover_transmission(tau, "beam"), np.exp(-tau))
    np.testing.assert_allclose(cover_transmission(tau), [1.0, *expn(2, tau[1:])])


@pytest.mark.parametrize("geometry", ["slab", "cylinder", "sphere"])
def test_self_shielding_limits(geometry: str) -> None:
    shielding = MonitorShielding(geometry, 0.05, 1.0, (1e-5, 2e7), (1.0, 1.0), total_source="test")
    x = np.array([1e-6, 1e3])
    g = self_shielding_factor(x / shielding.mean_chord_cm, shielding)
    assert g[0] == pytest.approx(1.0, abs=1e-5)
    assert g[1] * 1e3 == pytest.approx(1.0, rel=1e-3)
    grid = np.geomspace(1e-3, 1e2, 50) / shielding.mean_chord_cm
    assert np.all(np.diff(self_shielding_factor(grid, shielding)) < 0)


def test_slab_self_shielding_matches_analytic() -> None:
    shielding = MonitorShielding("slab", 0.02, 1.0, (1e-5, 2e7), (1.0, 1.0), total_source="test")
    sigma = 15.0  # 1/cm, tau = 0.3
    tau = sigma * 0.02
    expected = (1.0 - 2.0 * expn(3, tau)) / (2.0 * tau)
    assert self_shielding_factor(np.array([sigma]), shielding)[0] == pytest.approx(expected, rel=1e-12)


def test_cylinder_self_shielding_small_x_slope() -> None:
    # First-order expansion: G = 1 - x <l^2>/(2 <l>^2) with <l^2> = 3/2 * (2R)^2 * ... ;
    # compare to a direct Monte Carlo chord sample for an infinite cylinder.
    rng = np.random.default_rng(3)
    n = 400_000
    b = rng.uniform(0.0, 1.0, n)
    theta = np.arccos(rng.uniform(-1, 1, n))
    keep = rng.uniform(0, 1, n) < np.sin(theta)  # sin^2 weighting via rejection
    chord = np.sqrt(1 - b[keep] ** 2) / np.sin(theta[keep])  # unit diameter
    sigma = 0.8
    g_mc = (1 - np.exp(-sigma * chord).mean()) / (sigma * 1.0)
    shielding = MonitorShielding("cylinder", 1.0, 1.0, (1e-5, 2e7), (1.0, 1.0), total_source="test")
    assert self_shielding_factor(np.array([sigma]), shielding)[0] == pytest.approx(g_mc, rel=5e-3)


def test_cd_row_differs_from_bare_and_removes_thermal(mini_db: IRDFFDatabase) -> None:
    cover = CoverLayer("Cd", 0.0508, thickness_unc_cm=0.002)
    rows, unc, meta = build_monitor_response_matrix(
        [
            MonitorResponseSpec("co", "Co-1", "Co-59(n,g)Co-60"),
            MonitorResponseSpec("cocd", "Co-Cd-1", "Co-59(n,g)Co-60", cover=cover),
        ],
        EDGES,
        mini_db,
    )
    bare, covered = rows
    assert covered[0] < 1e-6 * bare[0]  # thermal group removed
    # ~1 b of residual Cd absorption above the cutoff removes ~1.5%
    assert covered[4] == pytest.approx(bare[4], rel=0.03)  # 100 eV - 10 keV kept
    assert covered[4] < bare[4]
    assert meta[1].metadata["cover"]["data"] == "Cd(n,disap)"
    assert "cover_thickness" in meta[1].metadata["uncertainty_components_barn"]
    assert np.all(unc[1] >= 0)


def test_duplicate_observation_ids_rejected(mini_db: IRDFFDatabase) -> None:
    spec = MonitorResponseSpec("same", "Co-1", "Co-59(n,g)Co-60")
    with pytest.raises(ValueError, match="unique"):
        build_monitor_response_matrix([spec, spec], EDGES, mini_db)


def test_builtin_approximation_rejected_for_physical_rows(tmp_path: Path) -> None:
    db = IRDFFDatabase(
        cache_dir=tmp_path, auto_download=False, archive_path=tmp_path / "absent.txt",
        allow_builtin_approximations=True,
    )
    with pytest.raises(ValueError, match="approximation"):
        build_monitor_response(MonitorResponseSpec("x", "Ti", "Ti-48(n,p)Sc-48"), EDGES, db)


def _unfolder(mini_db: IRDFFDatabase) -> SpectrumUnfolder:
    unfolder = SpectrumUnfolder(custom_energy_edges=EDGES, verbose=False)
    unfolder.irdff_db = mini_db
    return unfolder


def test_covered_label_without_cover_spec_is_refused(mini_db: IRDFFDatabase) -> None:
    unfolder = _unfolder(mini_db)
    unfolder.add_reaction("Co-59(n,g)Co-60", 1.0, 0.05, rate_per_atom=1e-12, cover="Cd")
    with pytest.raises(ValueError, match="CoverLayer"):
        unfolder.unfold(method="MLEM", max_iterations=5)


def test_bare_and_cd_rows_are_never_aggregated(mini_db: IRDFFDatabase) -> None:
    unfolder = _unfolder(mini_db)
    cover = CoverLayer("Cd", 0.0508)
    unfolder.add_reaction("Co-59(n,g)Co-60", 1.0, 0.05, rate_per_atom=1.6e-12, sample_id="Co-1")
    unfolder.add_reaction(
        "Co-59(n,g)Co-60", 1.0, 0.05, rate_per_atom=2.1e-13, sample_id="Co-Cd-1", cover="Cd",
        response_spec=MonitorResponseSpec("cocd", "Co-Cd-1", "Co-59(n,g)Co-60", cover=cover),
    )
    result = unfolder.unfold(method="MLEM", max_iterations=20, aggregate_duplicate_reactions=True)
    assert result.metadata["duplicate_reaction_aggregated_rows"] == 2
    assert result.response_matrix.shape[0] == 2
    assert not np.allclose(result.response_matrix[0], result.response_matrix[1], atol=0.0)
    assert result.metadata["response_rows"][1]["cover"]["thickness_cm"] == 0.0508


REAL_TAB = DEFAULT_CACHE_DIR / "tab" / IRDFF_TAB_ARCHIVE_NAME


@pytest.mark.skipif(
    not (REAL_TAB.exists() and REAL_TAB.with_name(IRDFF_ABS_ARCHIVE_NAME).exists()),
    reason="IAEA IRDFF-II archives not cached",
)
def test_real_cd_cover_on_co59() -> None:
    db = IRDFFDatabase(auto_download=False, archive_path=REAL_TAB)
    cd = db.get_cover_cross_section("Cd", "disap")
    cover = CoverLayer("Cd", 0.0508)
    tau = lambda e: cover.number_density_per_cm3 * cd.evaluate(e) * 1e-24 * 0.0508  # noqa: E731
    # Evaluated Cd: E2 tail leaves ~4e-4 at thermal, nearly transparent at 1-10 eV
    assert cover_transmission(np.atleast_1d(tau(0.0253)))[0] < 1e-3
    assert cover_transmission(np.atleast_1d(tau(1.0)))[0] > 0.8
    edges = np.array([1e-5, 0.55, 2e6, 2e7])
    bare = build_monitor_response(MonitorResponseSpec("a", "Co", "Co-59(n,g)Co-60"), edges, db)
    covered = build_monitor_response(
        MonitorResponseSpec("b", "CoCd", "Co-59(n,g)Co-60", cover=cover), edges, db
    )
    assert covered.group_cross_section_barn[0] < 1e-3 * bare.group_cross_section_barn[0]
    ratio = covered.group_cross_section_barn[1] / bare.group_cross_section_barn[1]
    assert 0.9 < ratio < 1.0
