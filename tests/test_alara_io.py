"""Tests for ALARA I/O module."""

import pytest
import os
from fluxforge.io.alara import (
    ALARASettings,
    ALARAInputGenerator,
    parse_alara_output,
)

@pytest.fixture
def sample_alara_settings():
    return ALARASettings(
        material_name="Eurofer97",
        density=7.8,
        flux_file="flux.in",
        material_lib="matlib",
        element_lib="elelib",
        data_library="fendl3",
        cooling_times=["1h", "1d"],
        irradiation_schedule=[("2h", "flux_1")],
    )

def test_alara_input_generation(sample_alara_settings, tmp_path):
    """Test generation of ALARA input file."""
    generator = ALARAInputGenerator(sample_alara_settings)
    output_path = tmp_path / "test.inp"
    
    generator.write(str(output_path))
    
    assert output_path.exists()
    content = output_path.read_text()
    
    assert "Material: Eurofer97" in content
    assert "geometry rectangular" in content
    assert "mixture mix_eurofer97" in content
    assert "material EUROFER97 1.0 7.8" in content
    assert "cooling" in content
    assert "1h" in content
    assert "1d" in content

def test_parse_alara_output():
    """Test parsing of ALARA output text."""
    sample_output = """
    ALARA Output
    ...
    Cooling Time: 1h
    ...
    Total Specific Activity: 1.5e10
    Total Decay Heat: 2.5e-3
    ...
    Cooling Time: 1d
    ...
    Total Specific Activity: 1.2e10
    Total Decay Heat: 2.0e-3
    ...
    """
    
    results = parse_alara_output(sample_output)
    
    assert results['cooling_times'] == ['1h', '1d']
    assert results['totals']['activity'] == [1.5e10, 1.2e10]
    assert results['totals']['heat'] == [2.5e-3, 2.0e-3]
