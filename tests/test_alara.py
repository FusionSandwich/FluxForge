"""
Tests for ALARA Integration Module

Tests for ALARA input/output file handling and conversion functions.
"""

import pytest
import tempfile
import os
from pathlib import Path
import numpy as np

from fluxforge.io.alara import (
    ALARAMixture,
    ALARASchedule,
    ALARAPulseHistory,
    ALARAFlux,
    ALARAZoneResult,
    ALARAOutput,
    ALARASettings,
    ALARAInputGenerator,
    read_alara_input,
    read_alara_output,
    write_alara_flux,
    read_alara_flux,
    fluxforge_spectrum_to_alara,
    create_alara_activation_input,
    parse_alara_output,
)


# =============================================================================
# Test ALARA Data Classes
# =============================================================================

class TestALARAMixture:
    """Tests for ALARAMixture dataclass."""
    
    def test_create_mixture(self):
        """Test creating a mixture."""
        mix = ALARAMixture(name="steel")
        mix.add_element("fe", 7.86, 0.90)
        mix.add_element("cr", 7.19, 0.09)
        mix.add_element("c", 2.62, 0.01)
        
        assert len(mix.constituents) == 3
        assert mix.name == "steel"
    
    def test_mixture_to_alara(self):
        """Test converting mixture to ALARA format."""
        mix = ALARAMixture(name="eurofer")
        mix.add_element("fe", 7.87, 0.89)
        mix.add_element("cr", 7.19, 0.09)
        
        output = mix.to_alara()
        
        assert "mixture eurofer" in output
        assert "element  fe" in output
        assert "element  cr" in output
        assert "end" in output
    
    def test_mixture_add_material(self):
        """Test adding a material to mixture."""
        mix = ALARAMixture(name="alloy")
        mix.add_material("C1020", 7.86, 1.0)
        
        assert len(mix.constituents) == 1
        assert mix.constituents[0]['type'] == 'material'


class TestALARASchedule:
    """Tests for ALARASchedule dataclass."""
    
    def test_create_schedule(self):
        """Test creating a schedule."""
        sched = ALARASchedule(name="irradiation")
        sched.add_item("2 h", "test_flux", "steady_state", "0 s")
        
        assert len(sched.items) == 1
        assert sched.name == "irradiation"
    
    def test_schedule_to_alara(self):
        """Test converting schedule to ALARA format."""
        sched = ALARASchedule(name="irrad_2hr")
        sched.add_item("2 h", "flux_1", "pulse_once")
        sched.add_item("1 d", "flux_2", "continuous", "1 h")
        
        output = sched.to_alara()
        
        assert "schedule irrad_2hr" in output
        assert "2 h  flux_1" in output
        assert "1 d  flux_2" in output
        assert "end" in output


class TestALARAFlux:
    """Tests for ALARAFlux dataclass."""
    
    def test_create_flux(self):
        """Test creating a flux definition."""
        flux = ALARAFlux(
            name="test_flux",
            file_path="/path/to/flux.txt",
            normalization=1e6,
            skip=0,
            fmt="default",
        )
        
        assert flux.name == "test_flux"
        assert flux.normalization == 1e6
    
    def test_flux_to_alara(self):
        """Test converting flux to ALARA format."""
        flux = ALARAFlux(
            name="sample_flux",
            file_path="/data/flux.flx",
            normalization=1.0,
            skip=10,
            fmt="default",
        )
        
        output = flux.to_alara()
        
        assert "flux  sample_flux" in output
        assert "/data/flux.flx" in output
        assert "10" in output


# =============================================================================
# Test ALARA File I/O
# =============================================================================

class TestALARAFluxIO:
    """Tests for ALARA flux file I/O."""
    
    def test_write_flux_file(self, tmp_path):
        """Test writing a flux file."""
        flux = np.array([1e10, 2e10, 5e10, 8e10, 3e10])
        energy = np.array([0, 1e6, 5e6, 10e6, 15e6, 20e6])
        
        output_file = tmp_path / "flux.txt"
        write_alara_flux(flux, energy, output_file, n_intervals=1, normalize=False)
        
        assert output_file.exists()
        
        # Read back
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # First line should be number of groups
        assert int(lines[0].strip()) == 5
    
    def test_read_flux_file(self, tmp_path):
        """Test reading a flux file."""
        # Create a flux file
        flux_data = np.array([1e10, 2e10, 5e10])
        output_file = tmp_path / "flux_read.txt"
        
        with open(output_file, 'w') as f:
            f.write("3\n")
            for val in flux_data:
                f.write(f"{val:.6e}\n")
        
        n_groups, flux = read_alara_flux(output_file)
        
        assert n_groups == 3
        assert len(flux) == 3
        np.testing.assert_array_almost_equal(flux, flux_data)
    
    def test_flux_normalization(self, tmp_path):
        """Test that flux normalization works."""
        flux = np.array([1.0, 2.0, 3.0, 4.0])
        energy = np.array([0, 5, 10, 15, 20])
        
        output_file = tmp_path / "flux_norm.txt"
        write_alara_flux(flux, energy, output_file, normalize=True)
        
        n_groups, flux_read = read_alara_flux(output_file)
        
        # Normalized flux should sum to 1
        assert np.sum(flux_read) == pytest.approx(1.0, rel=0.01)
    
    def test_fluxforge_spectrum_to_alara(self, tmp_path):
        """Test converting FluxForge spectrum to ALARA format."""
        flux = np.array([1e10, 5e10, 2e10])
        energy = np.array([1e6, 5e6, 10e6, 15e6])
        
        output_file = tmp_path / "converted.txt"
        result_path = fluxforge_spectrum_to_alara(flux, energy, output_file)
        
        assert result_path == str(output_file)
        assert Path(output_file).exists()


# =============================================================================
# Test ALARA Input File Creation
# =============================================================================

class TestALARAInputCreation:
    """Tests for ALARA input file creation."""
    
    def test_create_activation_input(self, tmp_path):
        """Test creating an activation input file."""
        elements = {"fe": 0.90, "cr": 0.09, "c": 0.01}
        
        output_file = tmp_path / "activation.alara"
        
        result = create_alara_activation_input(
            mixture_name="steel_test",
            elements=elements,
            flux_file="/path/to/flux.txt",
            irradiation_time="2 h",
            cooling_times=["0 s", "3 d", "7 d"],
            output_path=output_file,
            density=7.86,
        )
        
        assert output_file.exists()
        
        # Check content
        content = output_file.read_text()
        
        assert "geometry point" in content
        assert "mixture steel_test" in content
        assert "element  fe" in content
        assert "schedule irradiation" in content
        assert "2 h" in content
        assert "cooling" in content
        assert "3 d" in content
    
    def test_legacy_input_generator(self, tmp_path):
        """Test legacy ALARAInputGenerator."""
        settings = ALARASettings(
            material_name="Test Steel",
            density=7.86,
            flux_file="/path/to/flux",
            material_lib="/path/to/matlib",
            element_lib="/path/to/elelib",
            data_library="/path/to/datalib",
            cooling_times=["0 s", "1 h", "1 d"],
        )
        
        generator = ALARAInputGenerator(settings)
        
        output_file = tmp_path / "legacy.inp"
        generator.write(str(output_file))
        
        assert output_file.exists()
        
        content = output_file.read_text()
        assert "ALARA Input generated by FluxForge" in content


# =============================================================================
# Test ALARA Input File Reading
# =============================================================================

class TestALARAInputReading:
    """Tests for reading ALARA input files."""
    
    @pytest.fixture
    def sample_alara_input(self, tmp_path):
        """Create a sample ALARA input file."""
        content = """
## Test ALARA input file

geometry point

volumes
    1.0e-03    sample_zone
end

mat_loading
    sample_zone    steel_mix
end

material_lib  /path/to/matlib
element_lib   /path/to/elelib

mixture steel_mix
    element  fe   7.86    0.90
    element  cr   7.19    0.09
end

flux  test_flux  /path/to/flux.txt  1.0  0  default

schedule irradiation
    2 h  test_flux  steady_state  0 s
end

pulsehistory steady_state
    1   0 s
end

data_library alaralib /path/to/fendl2bin

cooling
    0 s
    3 d
    7 d
end

truncation 1e-7
"""
        input_file = tmp_path / "test.alara"
        input_file.write_text(content)
        return input_file
    
    def test_read_alara_input(self, sample_alara_input):
        """Test reading ALARA input file."""
        result = read_alara_input(sample_alara_input)
        
        assert result['geometry_type'] == 'point'
        assert len(result['volumes']) == 1
        assert result['volumes'][0][1] == 'sample_zone'
        assert 'sample_zone' in result['mat_loading']
        assert 'steel_mix' in result['mixtures']
        assert 'test_flux' in result['fluxes']
        assert len(result['cooling_times']) == 3
        assert result['truncation'] == pytest.approx(1e-7)
    
    def test_read_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_alara_input("/nonexistent/path.alara")


# =============================================================================
# Test ALARA Output Parsing
# =============================================================================

class TestALARAOutputParsing:
    """Tests for parsing ALARA output."""
    
    def test_parse_simple_output(self):
        """Test parsing simple output text."""
        output_text = """
ALARA 2.9.2
Set verbose level to 3.

Cooling Time: 0s
Total Specific Activity: 1.234e+10
Total Decay Heat: 5.678e-03

Cooling Time: 1d
Total Specific Activity: 9.876e+09
Total Decay Heat: 4.321e-03
"""
        result = parse_alara_output(output_text)
        
        assert len(result['cooling_times']) == 2
        assert len(result['totals']['activity']) == 2
        assert len(result['totals']['heat']) == 2
        assert result['totals']['activity'][0] == pytest.approx(1.234e+10)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
