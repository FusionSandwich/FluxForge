"""
Tests for OpenMC and MCNP I/O modules.

Tests extraction of flux tallies from:
- OpenMC statepoint files (statepoint.0250.h5)
- MCNP HDF5 output (runtpe.h5)
"""

import pytest
import numpy as np
from pathlib import Path


# Test data paths
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
STATEPOINT_FILE = EXAMPLES_DIR / "statepoint.0250.h5"
RUNTPE_FILE = EXAMPLES_DIR / "runtpe.h5"
MCNP_INPUT_FILE = EXAMPLES_DIR / "whale_J_core_clean_loc.i"


class TestOpenMCStatepoint:
    """Tests for OpenMC statepoint reading."""
    
    @pytest.fixture
    def h5py_available(self):
        """Check if h5py is available."""
        try:
            import h5py
            return True
        except ImportError:
            pytest.skip("h5py not available")
    
    def test_statepoint_exists(self):
        """Verify test statepoint file exists."""
        assert STATEPOINT_FILE.exists(), f"Test file not found: {STATEPOINT_FILE}"
    
    def test_read_statepoint_info(self, h5py_available):
        """Test reading basic statepoint info."""
        from fluxforge.io.openmc import read_statepoint_info
        
        info = read_statepoint_info(STATEPOINT_FILE)
        
        # Version should be detected
        assert info.version != ""
        
        # Should have tally IDs
        assert len(info.tally_ids) > 0
        
        # Verify our target tallies are present
        assert 7114 in info.tally_ids
        assert 7124 in info.tally_ids
        assert 7134 in info.tally_ids
        
        # k_eff should be available for criticality calculation
        assert info.k_eff is not None
    
    def test_read_tally_7114_flux(self, h5py_available):
        """Test extracting tally 7114 (whale_E8_flux)."""
        from fluxforge.io.openmc import read_openmc_tally
        
        result = read_openmc_tally(STATEPOINT_FILE, 7114)
        
        assert 'data' in result
        # Shape should be (90, 1, 2) = 30 mesh * 3 energy * 1 score * 2 (sum, sum_sq)
        assert result['data'].shape == (90, 1, 2)
    
    def test_read_tally_7124_photon(self, h5py_available):
        """Test extracting tally 7124 (whale_E8_photon)."""
        from fluxforge.io.openmc import read_openmc_tally
        
        result = read_openmc_tally(STATEPOINT_FILE, 7124)
        
        assert 'data' in result
        assert result['data'].shape == (30, 1, 2)  # 30 mesh cells, no energy filter
    
    def test_read_tally_7134_heating(self, h5py_available):
        """Test extracting tally 7134 (whale_E8_heating)."""
        from fluxforge.io.openmc import read_openmc_tally
        
        result = read_openmc_tally(STATEPOINT_FILE, 7134)
        
        assert 'data' in result
        assert result['data'].shape == (30, 1, 2)
    
    def test_extract_flux_spectrum(self, h5py_available):
        """Test extracting full flux spectrum from tally 7114."""
        import h5py
        
        with h5py.File(STATEPOINT_FILE, 'r') as f:
            n_realizations = int(f['n_realizations'][()])
            
            # Get energy bins from filter 1
            energy_bins = f['tallies/filters/filter 1/bins'][()]
            n_energy = int(f['tallies/filters/filter 1/n_bins'][()])
            
            # Get mesh size from filter 23
            n_mesh = int(f['tallies/filters/filter 23/n_bins'][()])
            
            # Extract tally data
            results = f['tallies/tally 7114/results'][()]
            
            # Reshape: filter order [23, 1] means mesh is outer loop
            sum_val = results[:, 0, 0].reshape(n_mesh, n_energy)
            mean = sum_val / n_realizations
            
            # Sum over mesh to get spectrum
            spectrum = np.sum(mean, axis=0)
            
            assert len(spectrum) == n_energy
            assert len(energy_bins) == n_energy + 1
            assert np.sum(spectrum) > 0  # Non-zero flux


class TestMCNPHDF5:
    """Tests for MCNP HDF5 output reading."""
    
    @pytest.fixture
    def h5py_available(self):
        """Check if h5py is available."""
        try:
            import h5py
            return True
        except ImportError:
            pytest.skip("h5py not available")
    
    def test_runtpe_exists(self):
        """Verify test MCNP file exists."""
        assert RUNTPE_FILE.exists(), f"Test file not found: {RUNTPE_FILE}"
    
    def test_read_mesh_tally_structure(self, h5py_available):
        """Test reading mesh tally structure from MCNP runtpe."""
        import h5py
        
        with h5py.File(RUNTPE_FILE, 'r') as f:
            assert 'results/mesh_tally' in f
            
            mesh_tallies = list(f['results/mesh_tally'].keys())
            assert len(mesh_tallies) > 0
            
            # Check for our expected tallies
            expected = ['mesh_tally_85114', 'mesh_tally_85214']
            for name in expected:
                assert name in mesh_tallies, f"Expected {name} in mesh tallies"
    
    def test_extract_mesh_tally_85114(self, h5py_available):
        """Test extracting mesh tally 85114."""
        import h5py
        
        with h5py.File(RUNTPE_FILE, 'r') as f:
            mt = f['results/mesh_tally/mesh_tally_85114']
            
            # Check expected datasets
            assert 'mean' in mt
            assert 'relative_standard_error' in mt
            assert 'grid_energy' in mt
            
            mean = mt['mean'][()]
            energy = mt['grid_energy'][()]
            
            # Shape: (177, 1, 4, 4, 4) = (n_energy, n_time, nx, ny, nz)
            assert mean.shape[0] == 177  # VITAMIN-J 175 groups + 2 extra
            assert len(energy) == 177
            
            # Energy should be in increasing order
            assert np.all(np.diff(energy) > 0)
    
    def test_compute_volume_averaged_spectrum(self, h5py_available):
        """Test computing volume-averaged spectrum from mesh tally."""
        import h5py
        
        with h5py.File(RUNTPE_FILE, 'r') as f:
            mt = f['results/mesh_tally/mesh_tally_85114']
            
            mean = mt['mean'][()]
            error = mt['relative_standard_error'][()]
            energy = mt['grid_energy'][()]
            
            # Average over spatial dimensions (last 3)
            # Shape: (177, 1, 4, 4, 4) -> (177,)
            vol_avg_flux = np.mean(mean[:, 0, :, :, :], axis=(1, 2, 3))
            vol_avg_err = np.sqrt(np.mean(error[:, 0, :, :, :] ** 2, axis=(1, 2, 3)))
            
            assert len(vol_avg_flux) == 177
            assert np.sum(vol_avg_flux) > 0
            
            # Most flux should be above thermal
            thermal_idx = np.searchsorted(energy, 0.5)  # 0.5 eV
            fast_flux = np.sum(vol_avg_flux[thermal_idx:])
            total_flux = np.sum(vol_avg_flux)
            assert fast_flux / total_flux > 0.1  # At least 10% above thermal


class TestMCNPInputParser:
    """Tests for MCNP input file parsing."""
    
    def test_mcnp_input_exists(self):
        """Verify test MCNP input file exists."""
        assert MCNP_INPUT_FILE.exists(), f"Test file not found: {MCNP_INPUT_FILE}"
    
    def test_parse_mcnp_input(self):
        """Test parsing MCNP input file for materials."""
        from fluxforge.io.mcnp import parse_mcnp_input
        
        result = parse_mcnp_input(MCNP_INPUT_FILE)
        
        assert 'materials' in result
        # Should find some materials
        assert len(result['materials']) > 0
    
    def test_material_density_extraction(self):
        """Test that we can extract material densities from cells."""
        # Read first few hundred lines looking for density values
        with open(MCNP_INPUT_FILE, 'r') as f:
            lines = f.readlines()[:500]
        
        densities = []
        for line in lines:
            # Look for negative density (mass density in g/cc)
            if '-' in line and any(c.isdigit() for c in line):
                parts = line.split()
                for p in parts:
                    try:
                        val = float(p)
                        if -15 < val < 0:  # Reasonable mass density range
                            densities.append(val)
                    except ValueError:
                        continue
        
        # Should find multiple negative densities
        assert len(densities) > 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
