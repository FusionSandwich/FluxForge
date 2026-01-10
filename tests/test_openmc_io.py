"""
Tests for OpenMC I/O module.

Tests statepoint reading, flux spectrum extraction, and data conversion.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from fluxforge.io.openmc import (
    OpenMCSpectrum,
    StatepointInfo,
    read_statepoint_info,
    read_openmc_tally,
    read_openmc_flux_spectrum,
    HAS_H5PY,
)


class TestOpenMCSpectrum:
    """Tests for OpenMCSpectrum dataclass."""
    
    def test_spectrum_creation(self):
        """Test basic spectrum creation."""
        energy_bins = np.array([1e-5, 0.1, 1.0, 10.0, 20.0])  # eV
        flux = np.array([1e10, 1e11, 5e10, 1e9])
        uncertainty = np.array([1e9, 1e10, 5e9, 1e8])
        
        spectrum = OpenMCSpectrum(
            energy_bins_ev=energy_bins,
            flux=flux,
            uncertainty=uncertainty,
        )
        
        assert spectrum.n_groups == 4
        assert len(spectrum.energy_bins_ev) == 5
    
    def test_group_centers(self):
        """Test geometric mean calculation for group centers."""
        energy_bins = np.array([1.0, 10.0, 100.0])  # eV
        flux = np.array([1.0, 1.0])
        
        spectrum = OpenMCSpectrum(
            energy_bins_ev=energy_bins,
            flux=flux,
            uncertainty=np.zeros(2),
        )
        
        centers = spectrum.group_centers_ev
        np.testing.assert_allclose(centers, [np.sqrt(10), np.sqrt(1000)])
    
    def test_lethargy_widths(self):
        """Test lethargy width calculation."""
        energy_bins = np.array([1.0, 10.0, 100.0])  # e-fold
        flux = np.array([1.0, 1.0])
        
        spectrum = OpenMCSpectrum(
            energy_bins_ev=energy_bins,
            flux=flux,
            uncertainty=np.zeros(2),
        )
        
        du = spectrum.lethargy_widths
        # ln(10) ≈ 2.303
        np.testing.assert_allclose(du, [np.log(10), np.log(10)], rtol=1e-10)
    
    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        flux = np.array([100.0, 50.0, 0.0])  # Third group has zero flux
        uncertainty = np.array([10.0, 10.0, 5.0])
        
        spectrum = OpenMCSpectrum(
            energy_bins_ev=np.array([1, 2, 3, 4]),
            flux=flux,
            uncertainty=uncertainty,
        )
        
        rel_unc = spectrum.relative_uncertainty
        np.testing.assert_allclose(rel_unc[0], 0.1)
        np.testing.assert_allclose(rel_unc[1], 0.2)
        np.testing.assert_allclose(rel_unc[2], 0.0)  # Zero flux -> zero relative
    
    def test_to_per_unit_lethargy(self):
        """Test conversion to per-unit-lethargy form."""
        energy_bins = np.array([1.0, 10.0, 100.0])
        flux = np.array([10.0, 20.0])
        uncertainty = np.array([1.0, 2.0])
        
        spectrum = OpenMCSpectrum(
            energy_bins_ev=energy_bins,
            flux=flux,
            uncertainty=uncertainty,
        )
        
        per_u = spectrum.to_per_unit_lethargy()
        du = np.log(10)
        
        np.testing.assert_allclose(per_u.flux, [10.0/du, 20.0/du], rtol=1e-10)
        np.testing.assert_allclose(per_u.uncertainty, [1.0/du, 2.0/du], rtol=1e-10)
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        spectrum = OpenMCSpectrum(
            energy_bins_ev=np.array([1, 10, 100]),
            flux=np.array([1e10, 2e10]),
            uncertainty=np.array([1e9, 2e9]),
            cell_id=42,
            score="flux",
            n_particles=1000000,
            metadata={"test": "value"},
        )
        
        d = spectrum.to_dict()
        restored = OpenMCSpectrum.from_dict(d)
        
        np.testing.assert_array_equal(restored.energy_bins_ev, spectrum.energy_bins_ev)
        np.testing.assert_array_equal(restored.flux, spectrum.flux)
        assert restored.cell_id == 42
        assert restored.n_particles == 1000000
        assert restored.metadata["test"] == "value"


class TestStatepointInfo:
    """Tests for StatepointInfo dataclass."""
    
    def test_info_creation(self):
        """Test basic info creation."""
        info = StatepointInfo(
            version="0.13.3",
            n_particles=1000000,
            n_batches=100,
            n_inactive=10,
            k_eff=(1.00023, 0.00012),
            tally_ids=[1, 2, 3],
        )
        
        assert info.version == "0.13.3"
        assert info.n_particles == 1000000
        assert info.k_eff[0] == pytest.approx(1.00023)
        assert 3 in info.tally_ids


class TestOpenMCReaders:
    """Tests for OpenMC file readers."""
    
    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
    def test_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_statepoint_info("/nonexistent/statepoint.h5")
    
    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
    def test_missing_tally_file_raises(self):
        """Test that missing file raises FileNotFoundError for tally reader."""
        with pytest.raises(FileNotFoundError):
            read_openmc_tally("/nonexistent/statepoint.h5", 1)
    
    def test_import_error_without_h5py(self):
        """Test behavior when h5py is not available (mocked)."""
        # This is more of a documentation test - actual behavior depends on import
        import fluxforge.io.openmc as openmc_io
        
        # The module should define HAS_H5PY
        assert hasattr(openmc_io, 'HAS_H5PY')


class TestOpenMCIntegration:
    """Integration tests using mock HDF5 data."""
    
    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
    def test_create_mock_statepoint(self):
        """Test reading a mock statepoint file."""
        import h5py
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sp_path = Path(tmpdir) / "statepoint.h5"
            
            # Create mock statepoint
            with h5py.File(sp_path, 'w') as f:
                f.attrs['version'] = [0, 13, 3]
                f.attrs['n_particles'] = 1000000
                f.attrs['n_batches'] = 100
                f.attrs['n_inactive'] = 10
                
                # Create tally group
                tallies = f.create_group('tallies')
                tally1 = tallies.create_group('tally 1')
                
                # Results: shape (n_bins, n_scores, 2) = (sum, sum_sq)
                results = np.array([[[10.0, 100.0]], [[20.0, 400.0]]])
                tally1.create_dataset('results', data=results)
                tally1.create_dataset('n_realizations', data=100)
            
            # Read statepoint info
            info = read_statepoint_info(sp_path)
            
            assert info.version == "0.13.3"
            assert info.n_particles == 1000000
            assert 1 in info.tally_ids
    
    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
    def test_read_mock_tally(self):
        """Test reading a mock tally."""
        import h5py
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sp_path = Path(tmpdir) / "statepoint.h5"
            
            with h5py.File(sp_path, 'w') as f:
                f.attrs['n_particles'] = 100000
                
                tallies = f.create_group('tallies')
                tally1 = tallies.create_group('tally 1')
                
                # 4 energy bins with results
                results = np.array([
                    [[100.0, 10000.0]],
                    [[200.0, 40000.0]],
                    [[150.0, 22500.0]],
                    [[50.0, 2500.0]],
                ])
                tally1.create_dataset('results', data=results)
                tally1.create_dataset('n_realizations', data=50)
            
            # Read tally
            result = read_openmc_tally(sp_path, 1)
            
            assert 'data' in result
            assert result['data'].shape[0] == 4
