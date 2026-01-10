"""Tests for MCNP I/O module."""

import numpy as np
import pytest
from fluxforge.io.mcnp import parse_mcnp_input, MCNPSpectrum, read_mcnp_spectrum_csv


class TestMCNPSpectrum:
    """Tests for MCNPSpectrum dataclass."""

    def test_spectrum_creation(self):
        spectrum = MCNPSpectrum(
            energy_low=np.array([0.001, 0.01, 0.1]),
            energy_high=np.array([0.01, 0.1, 1.0]),
            flux=np.array([1e10, 5e9, 1e8]),
            uncertainty=np.array([0.05, 0.03, 0.10]),
            tally_id="test_tally",
            energy_units="MeV",
        )
        assert spectrum.n_groups == 3
        assert spectrum.tally_id == "test_tally"
        assert spectrum.energy_units == "MeV"

    def test_spectrum_energy_mid(self):
        spectrum = MCNPSpectrum(
            energy_low=np.array([1.0, 10.0]),
            energy_high=np.array([10.0, 100.0]),
            flux=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.1]),
        )
        expected_mid = np.sqrt(np.array([1.0, 10.0]) * np.array([10.0, 100.0]))
        np.testing.assert_array_almost_equal(spectrum.energy_mid, expected_mid)

    def test_spectrum_to_eV(self):
        spectrum = MCNPSpectrum(
            energy_low=np.array([0.001]),
            energy_high=np.array([1.0]),
            flux=np.array([1e10]),
            uncertainty=np.array([0.05]),
            energy_units="MeV",
        )
        converted = spectrum.to_eV()
        assert converted.energy_units == "eV"
        assert converted.energy_low[0] == pytest.approx(1000.0)
        assert converted.energy_high[0] == pytest.approx(1e6)

    def test_spectrum_to_dict(self):
        spectrum = MCNPSpectrum(
            energy_low=np.array([0.1]),
            energy_high=np.array([1.0]),
            flux=np.array([1e9]),
            uncertainty=np.array([0.1]),
            tally_id="F4",
        )
        d = spectrum.to_dict()
        assert "energy_low" in d
        assert "flux" in d
        assert d["tally_id"] == "F4"


class TestReadMCNPSpectrumCSV:
    """Tests for read_mcnp_spectrum_csv function."""

    def test_read_simple_csv(self, tmp_path):
        """Test reading a 4-column CSV (E_low, E_high, flux, unc)."""
        csv_content = """0.001,0.01,1e10,0.05
0.01,0.1,5e9,0.03
0.1,1.0,1e8,0.10
"""
        csv_file = tmp_path / "test_spectrum.csv"
        csv_file.write_text(csv_content)

        spectrum = read_mcnp_spectrum_csv(csv_file)
        assert spectrum.n_groups == 3
        assert spectrum.flux[0] == pytest.approx(1e10)
        assert spectrum.uncertainty[1] == pytest.approx(0.03)
        assert spectrum.energy_units == "MeV"

    def test_read_multizone_csv(self, tmp_path):
        """Test reading a multi-zone CSV with zone-averaging."""
        # Format: E_low, E_high, flux1, unc1, flux2, unc2
        csv_content = """0.001,0.01,1e10,0.05,2e10,0.04
0.01,0.1,5e9,0.03,6e9,0.02
"""
        csv_file = tmp_path / "multizone.csv"
        csv_file.write_text(csv_content)

        spectrum = read_mcnp_spectrum_csv(csv_file)
        assert spectrum.n_groups == 2
        # Flux should be average of zones
        assert spectrum.flux[0] == pytest.approx(1.5e10)

    def test_read_with_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        csv_content = """
0.001,0.01,1e10,0.05

0.01,0.1,5e9,0.03

"""
        csv_file = tmp_path / "with_blanks.csv"
        csv_file.write_text(csv_content)

        spectrum = read_mcnp_spectrum_csv(csv_file)
        assert spectrum.n_groups == 2

    def test_empty_file_raises(self, tmp_path):
        """Test that empty file raises ValueError."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        with pytest.raises(ValueError, match="No valid data"):
            read_mcnp_spectrum_csv(csv_file)


def test_parse_mcnp_input(tmp_path):
    """Test parsing of MCNP input file."""
    input_content = """
    MCNP Test Input
    c Comment
    1 1 -7.8 10 -20 imp:n=1
    
    m1 26056.70c 0.9 24052.70c 0.1 $ Steel
    m2 1001.70c 2.0 8016.70c 1.0 $ Water
       6012.70c 0.5
    """
    
    input_path = tmp_path / "test.i"
    input_path.write_text(input_content)
    
    data = parse_mcnp_input(str(input_path))
    
    assert 1 in data['materials']
    assert 2 in data['materials']
    
    mat1 = data['materials'][1]
    assert len(mat1['components']) == 2
    assert mat1['components'][0] == ('26056.70c', '0.9')
    
    mat2 = data['materials'][2]
    assert len(mat2['components']) == 3
    assert mat2['components'][2] == ('6012.70c', '0.5')

def test_read_meshtal_hdf5_missing_file():
    """Test error handling for missing HDF5 file."""
    from fluxforge.io.mcnp import read_meshtal_hdf5, HAS_H5PY
    
    if not HAS_H5PY:
        pytest.skip("h5py not installed")
        
    with pytest.raises(FileNotFoundError):
        read_meshtal_hdf5("nonexistent.h5", 1)
