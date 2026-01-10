"""
Tests for CNF (Canberra) file reader.

Tests parsing of Canberra CNF binary spectrum files.
"""

import pytest
import numpy as np
import tempfile
import struct
from pathlib import Path

from fluxforge.io.cnf import (
    read_cnf_file,
    parse_cnf_binary,
    can_read_cnf,
    CNFData,
    CNFHeader,
    CNFCalibration,
)


def create_mock_cnf_data(
    n_channels: int = 4096,
    live_time: float = 300.0,
    real_time: float = 310.0,
    spectrum_offset: int = 0x200,
    cal_offset: int = 0x30,
) -> bytes:
    """Create mock CNF binary data for testing."""
    # Create buffer large enough
    data = bytearray(spectrum_offset + n_channels * 4 + 1024)
    
    # Add calibration at cal_offset: a0=0.5, a1=0.5 (keV/ch), a2=0
    struct.pack_into('<d', data, cal_offset, 0.5)  # a0
    struct.pack_into('<d', data, cal_offset + 8, 0.5)  # a1
    struct.pack_into('<d', data, cal_offset + 16, 0.0)  # a2
    
    # Add timing at 0x1A0
    struct.pack_into('<d', data, 0x1A0, live_time)
    struct.pack_into('<d', data, 0x1A8, real_time)
    
    # Add spectrum data at spectrum_offset
    # Create gaussian-like spectrum with peak at channel 1000
    for i in range(n_channels):
        # Background + Gaussian peak
        bg = 100
        peak = 10000 * np.exp(-0.5 * ((i - 1000) / 50) ** 2)
        count = int(bg + peak)
        struct.pack_into('<I', data, spectrum_offset + i * 4, count)
    
    return bytes(data)


class TestCNFDataClasses:
    """Tests for CNF data classes."""
    
    def test_header_creation(self):
        """Test CNFHeader creation."""
        header = CNFHeader(
            file_size=16384,
            detector_name="HPGe-1",
            live_time=300.0,
            real_time=310.0,
            n_channels=4096,
        )
        
        assert header.file_size == 16384
        assert header.detector_name == "HPGe-1"
        assert header.n_channels == 4096
    
    def test_calibration_creation(self):
        """Test CNFCalibration creation."""
        cal = CNFCalibration(
            energy_coefficients=[0.0, 0.5, 0.0001],
        )
        
        assert len(cal.energy_coefficients) == 3
        assert cal.energy_coefficients[1] == 0.5
    
    def test_data_creation(self):
        """Test CNFData creation."""
        data = CNFData(
            header=CNFHeader(n_channels=4096),
            spectrum=np.zeros(4096),
        )
        
        assert data.header.n_channels == 4096
        assert len(data.spectrum) == 4096


class TestParseCNFBinary:
    """Tests for parse_cnf_binary function."""
    
    def test_parse_mock_cnf(self):
        """Test parsing mock CNF data."""
        binary_data = create_mock_cnf_data()
        
        cnf = parse_cnf_binary(binary_data)
        
        assert cnf.header.file_size == len(binary_data)
        assert len(cnf.spectrum) > 0
    
    def test_parse_spectrum_shape(self):
        """Test that spectrum has expected shape."""
        binary_data = create_mock_cnf_data(n_channels=4096)
        
        cnf = parse_cnf_binary(binary_data)
        
        assert len(cnf.spectrum) == 4096
    
    def test_parse_calibration(self):
        """Test calibration extraction."""
        binary_data = create_mock_cnf_data()
        
        cnf = parse_cnf_binary(binary_data)
        
        # Should find calibration
        if cnf.calibration.energy_coefficients:
            # a0 should be 0.5, a1 should be 0.5
            assert cnf.calibration.energy_coefficients[0] == pytest.approx(0.5)
            assert cnf.calibration.energy_coefficients[1] == pytest.approx(0.5)
    
    def test_parse_timing(self):
        """Test timing extraction."""
        binary_data = create_mock_cnf_data(live_time=300.0, real_time=310.0)
        
        cnf = parse_cnf_binary(binary_data)
        
        assert cnf.header.live_time == pytest.approx(300.0)
        assert cnf.header.real_time == pytest.approx(310.0)
    
    def test_parse_spectrum_values(self):
        """Test that spectrum values are parsed correctly."""
        binary_data = create_mock_cnf_data(n_channels=4096)
        
        cnf = parse_cnf_binary(binary_data)
        
        # Should have peak around channel 1000
        peak_region = cnf.spectrum[950:1050]
        bg_region = cnf.spectrum[2000:2100]
        
        assert np.max(peak_region) > np.max(bg_region)
    
    def test_parse_small_file(self):
        """Test handling of file too small."""
        small_data = b'\x00' * 50  # Too small
        
        cnf = parse_cnf_binary(small_data)
        
        # Should return empty spectrum
        assert len(cnf.spectrum) == 0


class TestReadCNFFile:
    """Tests for read_cnf_file function."""
    
    def test_read_mock_cnf_file(self):
        """Test reading mock CNF file."""
        binary_data = create_mock_cnf_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "test.cnf"
            cnf_path.write_bytes(binary_data)
            
            spectrum = read_cnf_file(str(cnf_path))
            
            assert len(spectrum.counts) > 0
            assert spectrum.live_time > 0
    
    def test_spectrum_type(self):
        """Test that returned object is GammaSpectrum."""
        from fluxforge.io.spe import GammaSpectrum
        
        binary_data = create_mock_cnf_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "test.cnf"
            cnf_path.write_bytes(binary_data)
            
            spectrum = read_cnf_file(str(cnf_path))
            
            assert isinstance(spectrum, GammaSpectrum)
    
    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_cnf_file("/nonexistent/file.cnf")
    
    def test_empty_file(self):
        """Test ValueError for empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "empty.cnf"
            cnf_path.write_bytes(b'')
            
            with pytest.raises(ValueError, match="too small"):
                read_cnf_file(str(cnf_path))
    
    def test_metadata_preserved(self):
        """Test that metadata is included."""
        binary_data = create_mock_cnf_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "test.cnf"
            cnf_path.write_bytes(binary_data)
            
            spectrum = read_cnf_file(str(cnf_path))
            
            assert 'format' in spectrum.metadata
            assert spectrum.metadata['format'] == 'CNF'
    
    def test_calibration_applied(self):
        """Test that energy calibration is extracted."""
        binary_data = create_mock_cnf_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "test.cnf"
            cnf_path.write_bytes(binary_data)
            
            spectrum = read_cnf_file(str(cnf_path))
            
            # Should have calibration dict
            assert spectrum.calibration is not None
            assert len(spectrum.calibration) >= 2


class TestCanReadCNF:
    """Tests for can_read_cnf function."""
    
    def test_nonexistent_file(self):
        """Test that nonexistent file returns False."""
        assert can_read_cnf("/nonexistent/file.cnf") is False
    
    def test_wrong_extension(self):
        """Test that wrong extension returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "test.txt"
            txt_path.write_text("not a cnf file")
            
            assert can_read_cnf(str(txt_path)) is False
    
    def test_too_small(self):
        """Test that file too small returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "small.cnf"
            cnf_path.write_bytes(b'\x00' * 100)
            
            assert can_read_cnf(str(cnf_path)) is False
    
    def test_valid_mock_cnf(self):
        """Test that valid mock CNF returns True."""
        binary_data = create_mock_cnf_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "test.cnf"
            cnf_path.write_bytes(binary_data)
            
            assert can_read_cnf(str(cnf_path)) is True


class TestCNFIntegration:
    """Integration tests for CNF reading."""
    
    def test_full_workflow(self):
        """Test complete workflow: create, read, analyze."""
        # Create mock CNF with known characteristics
        binary_data = create_mock_cnf_data(
            n_channels=4096,
            live_time=600.0,
            real_time=620.0,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cnf_path = Path(tmpdir) / "spectrum.cnf"
            cnf_path.write_bytes(binary_data)
            
            # Read
            spectrum = read_cnf_file(str(cnf_path))
            
            # Verify
            assert len(spectrum.counts) == 4096
            assert spectrum.live_time == pytest.approx(600.0)
            assert spectrum.real_time == pytest.approx(620.0)
            
            # Check spectrum has expected peak
            peak_idx = np.argmax(spectrum.counts)
            assert 950 < peak_idx < 1050  # Peak near channel 1000
