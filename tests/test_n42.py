"""
Tests for N42 Format Reader

Tests the ANSI N42.42 XML format reader for gamma spectrum data.
"""

import pytest
import tempfile
import os
from pathlib import Path
import numpy as np

from fluxforge.io.n42 import (
    N42Measurement,
    N42Document,
    read_n42_file,
    read_n42_spectrum,
    write_n42_file,
    parse_iso8601_duration,
)


# =============================================================================
# Test ISO 8601 Duration Parser
# =============================================================================

class TestISO8601Duration:
    """Tests for ISO 8601 duration parsing."""
    
    def test_parse_seconds(self):
        """Test parsing seconds only."""
        assert parse_iso8601_duration("PT60S") == pytest.approx(60.0)
        assert parse_iso8601_duration("PT100.5S") == pytest.approx(100.5)
    
    def test_parse_minutes(self):
        """Test parsing minutes."""
        assert parse_iso8601_duration("PT5M") == pytest.approx(300.0)
        assert parse_iso8601_duration("PT5M30S") == pytest.approx(330.0)
    
    def test_parse_hours(self):
        """Test parsing hours."""
        assert parse_iso8601_duration("PT2H") == pytest.approx(7200.0)
        assert parse_iso8601_duration("PT1H30M") == pytest.approx(5400.0)
    
    def test_parse_days(self):
        """Test parsing days."""
        assert parse_iso8601_duration("P1D") == pytest.approx(86400.0)
        assert parse_iso8601_duration("P2DT3H") == pytest.approx(2 * 86400 + 3 * 3600)
    
    def test_invalid_format(self):
        """Test that invalid format returns None."""
        assert parse_iso8601_duration("invalid") is None
        assert parse_iso8601_duration("60 seconds") is None


# =============================================================================
# Test N42Measurement Dataclass
# =============================================================================

class TestN42Measurement:
    """Tests for N42Measurement dataclass."""
    
    def test_create_measurement(self):
        """Test creating a measurement."""
        meas = N42Measurement(
            counts=np.array([1, 2, 3, 4, 5]),
            live_time=100.0,
            real_time=105.0,
        )
        
        assert len(meas.counts) == 5
        assert meas.live_time == 100.0
        assert meas.real_time == 105.0
        # dead_time_fraction = 1 - (live_time / real_time) = 1 - (100/105) = 5/105
        assert meas.dead_time_fraction == pytest.approx(1.0 - 100.0/105.0)
    
    def test_energy_calibration(self):
        """Test energy axis generation from calibration."""
        meas = N42Measurement(
            counts=np.array([1, 2, 3]),
            live_time=100.0,
            energy_calibration=(0.0, 1.0, 0.0),  # linear: E = 1*ch
        )
        
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(meas.energy_axis(), expected)
    
    def test_quadratic_calibration(self):
        """Test quadratic energy calibration."""
        meas = N42Measurement(
            counts=np.zeros(10),
            live_time=100.0,
            energy_calibration=(0.5, 0.3, 0.001),  # E = 0.5 + 0.3*ch + 0.001*ch^2
        )
        
        channels = np.arange(10)
        expected = 0.5 + 0.3 * channels + 0.001 * channels**2
        np.testing.assert_array_almost_equal(meas.energy_axis(), expected)


# =============================================================================
# Test N42 File Writing and Reading
# =============================================================================

class TestN42FileIO:
    """Tests for N42 file I/O."""
    
    @pytest.fixture
    def sample_measurement(self):
        """Create a sample measurement for testing."""
        return N42Measurement(
            counts=np.array([10, 20, 100, 500, 200, 50, 10]),
            live_time=300.0,
            real_time=310.0,
            energy_calibration=(0.0, 0.5, 0.0),
            detector_type="HPGe",
            detector_description="HPGe-1 test detector",
        )
    
    def test_write_and_read_n42(self, sample_measurement, tmp_path):
        """Test writing and reading N42 file."""
        output_file = tmp_path / "test.n42"
        
        # Write
        write_n42_file(output_file, sample_measurement)
        
        assert output_file.exists()
        
        # Read back
        doc = read_n42_file(output_file)
        
        assert doc is not None
        assert len(doc.measurements) > 0
        
        meas = doc.measurements[0]
        assert len(meas.counts) == len(sample_measurement.counts)
        np.testing.assert_array_equal(meas.counts, sample_measurement.counts)
        assert meas.live_time == pytest.approx(sample_measurement.live_time)
    
    def test_read_n42_spectrum_simple(self, sample_measurement, tmp_path):
        """Test the simple spectrum reader."""
        output_file = tmp_path / "test_simple.n42"
        
        write_n42_file(output_file, sample_measurement)
        
        # read_n42_spectrum returns N42Measurement, not tuple
        meas = read_n42_spectrum(output_file)
        
        assert len(meas.counts) == len(sample_measurement.counts)
        np.testing.assert_array_equal(meas.counts, sample_measurement.counts)
        # Energy axis should be generated from calibration
        energy = meas.energy_axis()
        assert len(energy) == len(meas.counts)
    
    def test_read_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_n42_file("/nonexistent/path/file.n42")


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
