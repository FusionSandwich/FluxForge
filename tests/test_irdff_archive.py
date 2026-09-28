"""Evaluated IRDFF-II archive access (issue #193)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fluxforge.data.irdff import (
    BUILTIN_APPROXIMATION_SOURCE,
    DEFAULT_CACHE_DIR,
    IRDFF_TAB_ARCHIVE_NAME,
    IRDFFCrossSection,
    IRDFFDatabase,
    MissingCrossSectionError,
    build_response_matrix,
    match_irdff_archive_key,
    parse_irdff_tab_archive,
)

try:
    _trapezoid = np.trapezoid
except AttributeError:  # NumPy < 2
    _trapezoid = np.trapz

MINI_ARCHIVE = """\
Co-59(n,g)
 1.000000-5 1.8691E+03 1.2539E+01                                        0.67
 1.000000+0 4.0000E+00 4.0000E-02                                        1.00
 1.000000+2 1.0000E+00 1.0000E-02                                        1.00
 1.300000+2 8.0000E+02 8.0000E+00                                        1.00
 1.400000+2 1.0000E+00 1.0000E-02                                        1.00
 2.000000+7 1.0000E-04 1.0000E-06                                        1.00
Ni-58(n,2n)
 1.243000+7 0.0000E+00 0.0000E+00                                        0.00
 1.400000+7 3.0000E-02 3.0000E-03                                       10.00
 2.000000+7 5.0000E-02 5.0000E-03                                       10.00
In-113(n,g)In-114m
 1.000000-5 1.0000E+02 1.0000E+00                                        1.00
 2.000000+7 1.0000E-04 1.0000E-06                                        1.00
"""


@pytest.fixture()
def mini_db(tmp_path: Path) -> IRDFFDatabase:
    archive = tmp_path / IRDFF_TAB_ARCHIVE_NAME
    archive.write_text(MINI_ARCHIVE, encoding="ascii")
    return IRDFFDatabase(cache_dir=tmp_path, auto_download=False, archive_path=archive)


def test_parse_archive_reads_endf_floats_and_blocks(tmp_path: Path) -> None:
    archive = tmp_path / IRDFF_TAB_ARCHIVE_NAME
    archive.write_text(MINI_ARCHIVE, encoding="ascii")
    blocks = parse_irdff_tab_archive(archive)
    assert set(blocks) == {"Co-59(n,g)", "Ni-58(n,2n)", "In-113(n,g)In-114m"}
    assert blocks["Co-59(n,g)"].shape == (6, 4)
    assert blocks["Co-59(n,g)"][0, 0] == pytest.approx(1e-5)
    assert blocks["Co-59(n,g)"][3, 1] == pytest.approx(800.0)


@pytest.mark.parametrize(
    "reaction, expected",
    [
        ("Co-59(n,g)Co-60", "Co-59(n,g)"),
        ("Ni-58(n,2n)Ni-57", "Ni-58(n,2n)"),
        ("In-113(n,g)In-114m", "In-113(n,g)In-114m"),
        ("Co-59(n,g)Co-60m", None),  # metastable must be named explicitly
        ("Co-59(n,g)Co-61", None),  # wrong residual
        ("Ni-58(n,2n)Co-57", None),  # wrong element
        ("Unknown(Ti51)", None),
    ],
)
def test_archive_key_matching_checks_product(reaction: str, expected) -> None:
    keys = ["Co-59(n,g)", "Ni-58(n,2n)", "In-113(n,g)In-114m"]
    assert match_irdff_archive_key(reaction, keys) == expected


def test_archive_data_preferred_and_labeled(mini_db: IRDFFDatabase) -> None:
    xs = mini_db.get_cross_section("Co-59(n,g)Co-60")
    assert xs.source == "IRDFF-II"
    assert xs.evaluation_key == "Co-59(n,g)"
    assert xs.interpolation == "lin-lin"
    assert len(xs.source_sha256) == 64
    assert not xs.is_approximation
    # Lin-lin between tabulated points (not log-log)
    assert float(xs.evaluate(115.0)) == pytest.approx(1.0 + 0.5 * 799.0)


def test_threshold_keeps_linear_ramp(mini_db: IRDFFDatabase) -> None:
    xs = mini_db.get_cross_section("Ni-58(n,2n)Ni-57")
    assert xs.threshold_eV == pytest.approx(1.243e7)
    assert float(xs.evaluate(1.3e7)) > 0.0
    assert float(xs.evaluate(1.2e7)) == 0.0


def test_missing_reaction_raises_instead_of_dropping_row(mini_db: IRDFFDatabase) -> None:
    edges = np.geomspace(1e-5, 2e7, 11)
    with pytest.raises(MissingCrossSectionError, match="Ti-48"):
        build_response_matrix(
            ["Co-59(n,g)Co-60", "Ti-48(n,p)Sc-48"], edges, db=mini_db
        )
    response, valid, _ = build_response_matrix(
        ["Co-59(n,g)Co-60", "Ti-48(n,p)Sc-48"], edges, db=mini_db, on_missing="skip"
    )
    assert valid == ["Co-59(n,g)Co-60"] and response.shape == (1, 10)


def test_builtin_approximations_are_opt_in_and_labeled(tmp_path: Path) -> None:
    empty = IRDFFDatabase(
        cache_dir=tmp_path, auto_download=False, archive_path=tmp_path / "absent.txt"
    )
    assert empty.get_cross_section("Ti-48(n,p)Sc-48") is None
    opted = IRDFFDatabase(
        cache_dir=tmp_path,
        auto_download=False,
        archive_path=tmp_path / "absent.txt",
        allow_builtin_approximations=True,
    )
    xs = opted.get_cross_section("Ti-48(n,p)Sc-48")
    assert xs.source == BUILTIN_APPROXIMATION_SOURCE and xs.is_approximation


def test_group_collapse_captures_narrow_resonance_and_has_barn_units() -> None:
    energies = np.array([1.0, 100.0, 130.0, 130.1, 130.2, 1000.0])
    sigma = np.array([1.0, 1.0, 1.0, 1000.0, 1.0, 1.0])
    xs = IRDFFCrossSection(
        reaction="X-1(n,g)X-2", target="X-1", product="X-2", mt_number=102,
        energies=energies, cross_sections=sigma, uncertainties=0.1 * sigma,
        relative_unc=np.full(6, 10.0), interpolation="lin-lin",
    )
    edges = np.array([1.0, 1000.0])
    group_xs, group_unc = xs.to_group_structure(edges)
    fine = np.unique(np.concatenate([np.geomspace(1, 1000, 200001), energies]))
    reference = _trapezoid(xs.evaluate(fine) / fine, fine) / _trapezoid(1 / fine, fine)
    assert group_xs[0] == pytest.approx(reference, rel=1e-3)
    # Fully correlated within group: uncertainty is 10% of the group value.
    assert group_unc[0] == pytest.approx(0.1 * group_xs[0], rel=1e-6)


REAL_ARCHIVE = DEFAULT_CACHE_DIR / "tab" / IRDFF_TAB_ARCHIVE_NAME


@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="IAEA IRDFF-II archive not cached")
@pytest.mark.parametrize(
    "reaction, i0_barn",
    [
        # IRDFF-II resonance integrals above 0.5 eV (Trkov et al. 2020 values
        # agree within a few percent); built-in approximations were 2-4.5x low.
        ("Co-59(n,g)Co-60", 75.5),
        ("Cu-63(n,g)Cu-64", 4.95),
        ("Sc-45(n,g)Sc-46", 11.7),
        ("Fe-58(n,g)Fe-59", 1.26),
    ],
)
def test_real_archive_resonance_integrals(reaction: str, i0_barn: float) -> None:
    db = IRDFFDatabase(auto_download=False, archive_path=REAL_ARCHIVE)
    xs = db.get_cross_section(reaction)
    energy = np.geomspace(0.5, 2e6, 400001)
    assert _trapezoid(xs.evaluate(energy) / energy, energy) == pytest.approx(i0_barn, rel=0.02)


@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="IAEA IRDFF-II archive not cached")
def test_real_archive_resolves_all_rafm_monitor_reactions() -> None:
    db = IRDFFDatabase(auto_download=False, archive_path=REAL_ARCHIVE)
    reactions = [
        "Co-59(n,g)Co-60", "Sc-45(n,g)Sc-46", "Cu-63(n,g)Cu-64", "Fe-58(n,g)Fe-59",
        "Ti-46(n,p)Sc-46", "Ti-47(n,p)Sc-47", "Ti-48(n,p)Sc-48", "Ni-58(n,p)Co-58",
        "Ni-58(n,2n)Ni-57", "In-113(n,g)In-114m", "In-115(n,n')In-115m",
    ]
    for reaction in reactions:
        xs = db.get_cross_section(reaction)
        assert xs is not None and xs.source == "IRDFF-II", reaction
