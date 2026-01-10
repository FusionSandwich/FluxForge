"""
Unit tests for IRDFF-II data access.

Tests reaction catalog, cross section access, and library interface.
"""

import math
import pytest
import numpy as np

import sys
sys.path.insert(0, '/filespace/s/smandych/CAE/projects/ALARA/FluxForge/src')

from fluxforge.data.irdff_access import (
    ReactionCategory,
    DataStatus,
    IRDFFReaction,
    CrossSectionData,
    IRDFF_CATALOG,
    IRDFFLibrary,
    get_default_library,
)


class TestReactionCategories:
    """Test reaction categorization."""
    
    def test_categories_exist(self):
        """All categories should exist."""
        assert ReactionCategory.THRESHOLD
        assert ReactionCategory.RADIATIVE_CAPTURE
        assert ReactionCategory.FISSION
        assert ReactionCategory.INELASTIC
        assert ReactionCategory.COVER
    
    def test_status_levels(self):
        """All status levels should exist."""
        assert DataStatus.RECOMMENDED
        assert DataStatus.SECONDARY
        assert DataStatus.MONITORING
        assert DataStatus.RESEARCH
        assert DataStatus.DEPRECATED


class TestIRDFFCatalog:
    """Test IRDFF reaction catalog."""
    
    def test_catalog_populated(self):
        """Catalog should have reactions."""
        assert len(IRDFF_CATALOG) > 10
    
    def test_gold_capture(self):
        """Au-197(n,g) should be in catalog."""
        assert "Au-197(n,g)" in IRDFF_CATALOG
        
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        assert rxn.target == "Au-197"
        assert rxn.product == "Au-198"
        assert rxn.mt == 102
        assert rxn.za == 79197
        assert rxn.threshold_eV == 0  # Exothermic
        assert rxn.half_life_s > 0
        assert rxn.status == DataStatus.RECOMMENDED
    
    def test_nickel_threshold(self):
        """Ni-58(n,p) should have threshold."""
        rxn = IRDFF_CATALOG["Ni-58(n,p)"]
        
        assert rxn.threshold_eV > 0
        assert rxn.category == ReactionCategory.THRESHOLD
    
    def test_aluminum_alpha(self):
        """Al-27(n,a) should be present."""
        rxn = IRDFF_CATALOG["Al-27(n,a)"]
        
        assert rxn.mt == 107
        assert rxn.threshold_eV > 3e6  # ~3.25 MeV
    
    def test_inelastic_reaction(self):
        """In-115(n,n') should be inelastic."""
        rxn = IRDFF_CATALOG["In-115(n,n')"]
        
        assert rxn.category == ReactionCategory.INELASTIC
        assert rxn.mt == 4


class TestIRDFFReaction:
    """Test reaction dataclass."""
    
    def test_full_name(self):
        """Test full reaction name."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        assert rxn.full_name == "Au-197(n,g)Au-198"
    
    def test_short_name(self):
        """Test short reaction name."""
        rxn = IRDFF_CATALOG["Ni-58(n,p)"]
        
        assert rxn.short_name == "Ni-58(n,p)"
    
    def test_threshold_MeV(self):
        """Test threshold in MeV."""
        rxn = IRDFF_CATALOG["Al-27(n,a)"]
        
        assert rxn.threshold_MeV == pytest.approx(3.25, rel=0.01)
    
    def test_half_life_conversions(self):
        """Test half-life unit conversions."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        assert rxn.half_life_days == pytest.approx(2.6943, rel=0.01)
        assert rxn.half_life_hours == pytest.approx(2.6943 * 24, rel=0.01)
    
    def test_decay_constant(self):
        """Test decay constant calculation."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        expected_lambda = np.log(2) / rxn.half_life_s
        assert rxn.decay_constant == pytest.approx(expected_lambda)
    
    def test_gamma_lines(self):
        """Test gamma line data."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        assert len(rxn.gamma_lines_keV) >= 1
        assert 411.8 in rxn.gamma_lines_keV
        assert len(rxn.gamma_intensities) == len(rxn.gamma_lines_keV)


class TestCrossSectionData:
    """Test cross section data class."""
    
    def test_data_creation(self):
        """Test cross section data creation."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        energies = np.logspace(-5, 7, 100)
        xs = np.random.rand(100)
        unc = xs * 0.05
        
        data = CrossSectionData(
            reaction=rxn,
            energies_eV=energies,
            cross_section_b=xs,
            uncertainty_b=unc,
        )
        
        assert data.n_points == 100
    
    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        energies = np.array([1, 10, 100])
        xs = np.array([100, 50, 10])
        unc = np.array([5, 5, 2])
        
        data = CrossSectionData(
            reaction=rxn,
            energies_eV=energies,
            cross_section_b=xs,
            uncertainty_b=unc,
        )
        
        rel = data.relative_uncertainty
        
        assert rel[0] == pytest.approx(0.05)
        assert rel[1] == pytest.approx(0.10)
        assert rel[2] == pytest.approx(0.20)
    
    def test_interpolation(self):
        """Test cross section interpolation."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        energies = np.array([1, 10, 100, 1000])
        xs = np.array([1000, 100, 10, 1])
        unc = xs * 0.05
        
        data = CrossSectionData(
            reaction=rxn,
            energies_eV=energies,
            cross_section_b=xs,
            uncertainty_b=unc,
        )
        
        # Interpolate at 30 eV (between 10 and 100)
        xs_interp, unc_interp = data.get_cross_section(30)
        
        assert xs_interp > 10 and xs_interp < 100
    
    def test_group_averaging(self):
        """Test group-averaged cross sections."""
        rxn = IRDFF_CATALOG["Au-197(n,g)"]
        
        energies = np.logspace(-3, 4, 100)
        xs = 100 / np.sqrt(energies)  # 1/v shape
        unc = xs * 0.05
        
        data = CrossSectionData(
            reaction=rxn,
            energies_eV=energies,
            cross_section_b=xs,
            uncertainty_b=unc,
        )
        
        group_bounds = np.array([0.001, 0.1, 10, 1000, 10000])
        group_xs, group_unc = data.get_group_averaged(group_bounds)
        
        assert len(group_xs) == 4
        assert len(group_unc) == 4
        
        # Higher energy should have lower cross section
        assert group_xs[0] > group_xs[-1]


class TestIRDFFLibrary:
    """Test IRDFF library interface."""
    
    def test_library_creation(self):
        """Test library creation."""
        library = IRDFFLibrary()
        
        assert library.data_path is None
    
    def test_list_reactions(self):
        """Test listing reactions."""
        library = IRDFFLibrary()
        
        all_rxns = library.list_reactions()
        assert len(all_rxns) > 10
        
        threshold_rxns = library.list_reactions(category=ReactionCategory.THRESHOLD)
        assert len(threshold_rxns) > 0
        assert all(r.category == ReactionCategory.THRESHOLD for r in threshold_rxns)
        
        recommended = library.list_reactions(status=DataStatus.RECOMMENDED)
        assert len(recommended) > 0
        assert all(r.status == DataStatus.RECOMMENDED for r in recommended)
    
    def test_get_reaction(self):
        """Test getting reaction by name."""
        library = IRDFFLibrary()
        
        rxn = library.get_reaction("Au-197(n,g)")
        
        assert rxn is not None
        assert rxn.target == "Au-197"
    
    def test_get_by_za_mt(self):
        """Test getting reaction by ZA and MT."""
        library = IRDFFLibrary()
        
        rxn = library.get_by_za_mt(79197, 102)
        
        assert rxn is not None
        assert rxn.target == "Au-197"
        
        # Non-existent should return None
        assert library.get_by_za_mt(99999, 999) is None

    def test_load_g4_two_column_mev_heuristic(self, tmp_path):
        """Library should load IRDFF-style .G4 two-column files and convert MeV->eV."""
        rxn_name = "Au-197(n,g)"
        rxn = IRDFF_CATALOG[rxn_name]

        # Name includes target token + MT for heuristic matching
        p = tmp_path / "au197_mt102.G4"
        p.write_text(
            "# E(MeV)  XS(b)\n"
            "0.1  100\n"
            "1.0  10\n"
            "10.0 1\n"
        )

        lib = IRDFFLibrary(data_path=tmp_path)
        xs = lib.get_cross_section(rxn_name)
        assert xs is not None
        assert xs.reaction.za == rxn.za
        assert xs.reaction.mt == rxn.mt
        assert xs.energies_eV[0] == pytest.approx(0.1 * 1e6)
        assert xs.energies_eV[-1] == pytest.approx(10.0 * 1e6)

    def test_load_dat_three_column_and_clamp_negative(self, tmp_path):
        """Library should load .DAT and clamp negative cross sections to zero."""
        rxn_name = "Au-197(n,g)"

        p = tmp_path / "Au197_MT102.DAT"
        p.write_text(
            "! E(MeV), XS(b), UNC(b)\n"
            "0.5, -5.0, 0.2\n"
            "1.0,  2.0, 0.1\n"
        )

        lib = IRDFFLibrary(data_path=tmp_path)
        xs = lib.get_cross_section(rxn_name)
        assert xs is not None
        assert xs.cross_section_b[0] == pytest.approx(0.0)
        assert xs.cross_section_b[1] == pytest.approx(2.0)
        assert np.all(xs.uncertainty_b > 0)
    
    def test_get_cross_section(self):
        """Test getting cross section data."""
        library = IRDFFLibrary()
        
        xs_data = library.get_cross_section("Au-197(n,g)")
        
        assert xs_data is not None
        assert xs_data.n_points > 0
        assert xs_data.reaction.target == "Au-197"
    
    def test_threshold_reactions_filter(self):
        """Test filtering by threshold energy."""
        library = IRDFFLibrary()
        
        # Get reactions with threshold between 1-5 MeV
        rxns = library.get_threshold_reactions(E_min_MeV=1.0, E_max_MeV=5.0)
        
        for rxn in rxns:
            assert 1e6 <= rxn.threshold_eV <= 5e6
    
    def test_library_serialization(self):
        """Test library export to dict."""
        library = IRDFFLibrary()
        
        data = library.to_dict()
        
        assert "schema" in data
        assert data["n_reactions"] > 10


class TestDefaultLibrary:
    """Test default library getter."""
    
    def test_get_default(self):
        """Test getting default library."""
        library = get_default_library()
        
        assert isinstance(library, IRDFFLibrary)
        assert len(library.list_reactions()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
