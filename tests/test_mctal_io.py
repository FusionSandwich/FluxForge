"""
Tests for MCNP MCTAL file reading.

Tests parsing of MCTAL text files for tally extraction.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from fluxforge.io.mcnp import (
    MCTALTally,
    MCTALFile,
    read_mctal,
    read_mcnp_flux_tally,
)


# Sample MCTAL content for testing
SAMPLE_MCTAL = """mcnp     6    02/15/23 10:30:45     
message: test problem
nps      1000000
tally        4     1
f       1       2       3
e      1.00000E-08  1.00000E-06  1.00000E-04  1.00000E-02  1.00000E+00  2.00000E+01
vals 
  1.23456E-04  0.0123
  2.34567E-04  0.0234
  3.45678E-04  0.0345
  4.56789E-04  0.0456
  5.67890E-04  0.0567
tfc     5
  1  2  3  4  5
tally       14     1
f       5
e      1.00000E-08  1.00000E-02  2.00000E+01
vals 
  9.87654E-05  0.0098
  8.76543E-05  0.0087
tfc     2
  1  2
"""

SIMPLE_MCTAL = """mcnp6     6.2    01/01/24 12:00:00
nps      500000
tally        4     1
e      0.0  0.5  1.0  5.0  10.0
vals 
  1.0E-04  0.01  2.0E-04  0.02  3.0E-04  0.03  4.0E-04  0.04
tfc     4
"""


class TestMCTALTally:
    """Tests for MCTALTally dataclass."""
    
    def test_tally_creation(self):
        """Test basic tally creation."""
        tally = MCTALTally(
            tally_id=4,
            particle_type=1,
            energy_bins=np.array([0.0, 0.5, 1.0, 5.0]),
            flux=np.array([1e-4, 2e-4, 3e-4]),
            uncertainty=np.array([0.01, 0.02, 0.03]),
            cells=[1, 2, 3],
        )
        
        assert tally.tally_id == 4
        assert tally.particle_type == 1
        assert len(tally.energy_bins) == 4
        assert len(tally.flux) == 3
        assert tally.cells == [1, 2, 3]
    
    def test_to_spectrum(self):
        """Test conversion to MCNPSpectrum."""
        tally = MCTALTally(
            tally_id=14,
            particle_type=1,
            energy_bins=np.array([0.0, 0.5, 1.0, 5.0]),
            flux=np.array([1e-4, 2e-4, 3e-4]),
            uncertainty=np.array([0.01, 0.02, 0.03]),
        )
        
        spectrum = tally.to_spectrum()
        
        assert spectrum.n_groups == 3
        np.testing.assert_array_equal(spectrum.energy_low, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(spectrum.energy_high, [0.5, 1.0, 5.0])
        np.testing.assert_array_equal(spectrum.flux, [1e-4, 2e-4, 3e-4])
        assert spectrum.tally_id == "14"
    
    def test_to_spectrum_insufficient_bins(self):
        """Test that insufficient bins raises error."""
        tally = MCTALTally(
            tally_id=4,
            energy_bins=np.array([1.0]),  # Only 1 bin, need at least 2
            flux=np.array([]),
            uncertainty=np.array([]),
        )
        
        with pytest.raises(ValueError, match="insufficient energy bins"):
            tally.to_spectrum()


class TestMCTALFile:
    """Tests for MCTALFile dataclass."""
    
    def test_file_creation(self):
        """Test basic file creation."""
        tally = MCTALTally(tally_id=4)
        
        mctal = MCTALFile(
            code="mcnp6",
            version="6.2",
            nps=1000000,
            tallies={4: tally},
        )
        
        assert mctal.code == "mcnp6"
        assert mctal.nps == 1000000
        assert 4 in mctal.tallies
    
    def test_get_tally(self):
        """Test get_tally method."""
        tally = MCTALTally(tally_id=14)
        mctal = MCTALFile(tallies={14: tally})
        
        result = mctal.get_tally(14)
        assert result.tally_id == 14
    
    def test_get_tally_not_found(self):
        """Test get_tally raises for missing tally."""
        mctal = MCTALFile(tallies={4: MCTALTally(tally_id=4)})
        
        with pytest.raises(KeyError, match="Tally 99 not found"):
            mctal.get_tally(99)
    
    def test_list_tallies(self):
        """Test list_tallies method."""
        mctal = MCTALFile(tallies={
            4: MCTALTally(tally_id=4),
            14: MCTALTally(tally_id=14),
            24: MCTALTally(tally_id=24),
        })
        
        tally_ids = mctal.list_tallies()
        assert set(tally_ids) == {4, 14, 24}


class TestReadMCTAL:
    """Tests for read_mctal function."""
    
    def test_read_sample_mctal(self):
        """Test reading sample MCTAL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SAMPLE_MCTAL)
            
            mctal = read_mctal(mctal_path)
            
            assert mctal.code == "mcnp"
            assert mctal.nps == 1000000
            assert 4 in mctal.tallies
            assert 14 in mctal.tallies
    
    def test_read_tally_energy_bins(self):
        """Test that energy bins are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SAMPLE_MCTAL)
            
            mctal = read_mctal(mctal_path)
            tally4 = mctal.get_tally(4)
            
            # Should have 6 energy bin boundaries
            assert len(tally4.energy_bins) == 6
            np.testing.assert_allclose(tally4.energy_bins[0], 1e-8, rtol=1e-5)
            np.testing.assert_allclose(tally4.energy_bins[-1], 20.0, rtol=1e-5)
    
    def test_read_tally_flux_values(self):
        """Test that flux values are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SAMPLE_MCTAL)
            
            mctal = read_mctal(mctal_path)
            tally4 = mctal.get_tally(4)
            
            # Should have 5 flux values (one per energy group)
            assert len(tally4.flux) == 5
            np.testing.assert_allclose(tally4.flux[0], 1.23456e-4, rtol=1e-5)
    
    def test_read_tally_uncertainties(self):
        """Test that uncertainties are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SAMPLE_MCTAL)
            
            mctal = read_mctal(mctal_path)
            tally4 = mctal.get_tally(4)
            
            np.testing.assert_allclose(tally4.uncertainty[0], 0.0123, rtol=1e-5)
    
    def test_read_tally_cells(self):
        """Test that cell list is parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SAMPLE_MCTAL)
            
            mctal = read_mctal(mctal_path)
            tally4 = mctal.get_tally(4)
            
            assert tally4.cells == [1, 2, 3]
    
    def test_read_simple_mctal(self):
        """Test reading simpler MCTAL format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SIMPLE_MCTAL)
            
            mctal = read_mctal(mctal_path)
            
            assert mctal.nps == 500000
            assert 4 in mctal.tallies
            
            tally = mctal.get_tally(4)
            assert len(tally.flux) == 4
    
    def test_missing_file(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            read_mctal("/nonexistent/mctal")


class TestReadMCNPFluxTally:
    """Tests for convenience function read_mcnp_flux_tally."""
    
    def test_read_flux_tally(self):
        """Test reading flux tally as spectrum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SIMPLE_MCTAL)
            
            spectrum = read_mcnp_flux_tally(mctal_path, 4)
            
            assert spectrum.n_groups == 4
            assert spectrum.tally_id == "4"
            assert spectrum.energy_units == "MeV"
    
    def test_read_flux_tally_not_found(self):
        """Test that missing tally raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SIMPLE_MCTAL)
            
            with pytest.raises(KeyError):
                read_mcnp_flux_tally(mctal_path, 99)


class TestMCTALEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_file(self):
        """Test handling of empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text("")
            
            mctal = read_mctal(mctal_path)
            assert len(mctal.tallies) == 0
    
    def test_header_only(self):
        """Test file with only header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text("mcnp6 6.2 01/01/24 12:00:00\nnps 100000\n")
            
            mctal = read_mctal(mctal_path)
            assert mctal.code == "mcnp6"
            assert mctal.nps == 100000
    
    def test_multiple_tallies(self):
        """Test file with multiple tallies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mctal_path = Path(tmpdir) / "mctal"
            mctal_path.write_text(SAMPLE_MCTAL)
            
            mctal = read_mctal(mctal_path)
            
            assert len(mctal.tallies) == 2
            assert 4 in mctal.tallies
            assert 14 in mctal.tallies
            
            # Check they have different values
            tally4 = mctal.get_tally(4)
            tally14 = mctal.get_tally(14)
            
            assert len(tally4.flux) != len(tally14.flux)
