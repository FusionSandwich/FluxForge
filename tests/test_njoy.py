"""
Tests for the NJOY processing pipeline module.

Tests input generation, result structures, and pipeline specifications.
Note: Actual NJOY execution tests are skipped unless NJOY is available.
"""

import numpy as np
import pytest
from pathlib import Path

from fluxforge.data.njoy import (
    NJOYModule,
    GroupStructure,
    GROUP_STRUCTURE_DATA,
    NJOYInput,
    NJOYResult,
    NJOYPipelineSpec,
    generate_reconr_input,
    generate_broadr_input,
    generate_groupr_input,
    generate_njoy_input,
    check_njoy_available,
    create_dosimetry_pipeline,
)


class TestNJOYModule:
    """Tests for NJOYModule enum."""
    
    def test_module_values(self):
        """Test module value assignments."""
        assert NJOYModule.RECONR.value == "reconr"
        assert NJOYModule.BROADR.value == "broadr"
        assert NJOYModule.GROUPR.value == "groupr"
        assert NJOYModule.ERRORR.value == "errorr"
    
    def test_all_modules_defined(self):
        """Test all standard modules are defined."""
        expected = ['RECONR', 'BROADR', 'UNRESR', 'HEATR', 
                    'THERMR', 'GROUPR', 'ERRORR', 'ACER']
        for name in expected:
            assert hasattr(NJOYModule, name)


class TestGroupStructure:
    """Tests for GroupStructure enum."""
    
    def test_structure_values(self):
        """Test structure IGN values."""
        assert GroupStructure.VITAMIN_J.value == 1
        assert GroupStructure.SAND_II.value == 6
        assert GroupStructure.CUSTOM.value == 0
    
    def test_structure_data_entries(self):
        """Test GROUP_STRUCTURE_DATA has expected entries."""
        assert GroupStructure.SAND_II in GROUP_STRUCTURE_DATA
        data = GROUP_STRUCTURE_DATA[GroupStructure.SAND_II]
        assert data['n_groups'] == 640
        assert 'SAND-II' in data['name']


class TestNJOYInput:
    """Tests for NJOYInput dataclass."""
    
    def test_default_values(self):
        """Test default input values."""
        config = NJOYInput(
            endf_file=Path("/tmp/test.endf"),
            mat_number=9228,
        )
        
        assert config.temperatures == [300.0]
        assert config.group_structure == GroupStructure.SAND_II
        assert NJOYModule.RECONR in config.modules
        assert config.tolerance == 0.001
    
    def test_custom_temperatures(self):
        """Test custom temperature specification."""
        config = NJOYInput(
            endf_file=Path("/tmp/test.endf"),
            mat_number=9228,
            temperatures=[300.0, 600.0, 900.0],
        )
        
        assert len(config.temperatures) == 3
        assert 600.0 in config.temperatures
    
    def test_custom_group_structure(self):
        """Test custom group structure."""
        boundaries = np.logspace(-5, 7.3, 51)
        config = NJOYInput(
            endf_file=Path("/tmp/test.endf"),
            mat_number=9228,
            group_structure=GroupStructure.CUSTOM,
            custom_boundaries=boundaries,
        )
        
        assert config.group_structure == GroupStructure.CUSTOM
        assert len(config.custom_boundaries) == 51


class TestNJOYResult:
    """Tests for NJOYResult dataclass."""
    
    def test_default_values(self):
        """Test default result values."""
        result = NJOYResult(success=False)
        
        assert result.success is False
        assert result.n_groups == 0
        assert len(result.cross_sections) == 0
        assert len(result.errors) == 0
    
    def test_successful_result(self):
        """Test populating successful result."""
        result = NJOYResult(
            success=True,
            group_boundaries=np.linspace(1e-5, 20e6, 51),
            cross_sections={1: np.ones(50), 18: np.ones(50) * 0.5},
            mat_number=9228,
            temperatures=[300.0],
        )
        
        assert result.success
        assert result.n_groups == 50
        assert result.get_xs(1) is not None
        assert result.get_xs(18) is not None
        assert result.get_xs(999) is None
    
    def test_summary_generation(self):
        """Test summary string generation."""
        result = NJOYResult(
            success=True,
            group_boundaries=np.linspace(1, 1000, 11),
            cross_sections={102: np.ones(10)},
            mat_number=7925,
            temperatures=[300.0],
        )
        
        summary = result.summary()
        assert "Success: True" in summary
        assert "MAT: 7925" in summary
        assert "Groups: 10" in summary


class TestInputGeneration:
    """Tests for NJOY input generation functions."""
    
    def test_reconr_input(self):
        """Test RECONR module input generation."""
        inp = generate_reconr_input(mat=9228, tape_in=20, tape_out=21)
        
        assert "reconr" in inp
        assert "9228" in inp
        assert "20 21" in inp
    
    def test_broadr_input(self):
        """Test BROADR module input generation."""
        inp = generate_broadr_input(
            mat=9228,
            temperatures=[300.0, 600.0],
            tape_in_endf=20,
            tape_in_pendf=21,
            tape_out=22,
        )
        
        assert "broadr" in inp
        assert "300.0 600.0" in inp
        assert "2" in inp  # Number of temperatures
    
    def test_groupr_input(self):
        """Test GROUPR module input generation."""
        inp = generate_groupr_input(
            mat=9228,
            group_structure=6,  # SAND-II
            temperatures=[300.0],
            tape_in_endf=20,
            tape_in_pendf=22,
            tape_out=23,
        )
        
        assert "groupr" in inp
        assert "9228 6" in inp  # MAT and IGN
        assert "FluxForge" in inp
    
    def test_full_input_generation(self):
        """Test complete input deck generation."""
        config = NJOYInput(
            endf_file=Path("/tmp/test.endf"),
            mat_number=9228,
            temperatures=[300.0],
            group_structure=GroupStructure.SAND_II,
            modules=[NJOYModule.RECONR, NJOYModule.BROADR, NJOYModule.GROUPR],
        )
        
        deck = generate_njoy_input(config)
        
        assert "FluxForge" in deck
        assert "reconr" in deck
        assert "broadr" in deck
        assert "groupr" in deck
        assert "stop" in deck
        assert "9228" in deck


class TestNJOYPipelineSpec:
    """Tests for NJOYPipelineSpec dataclass."""
    
    def test_basic_creation(self):
        """Test creating a pipeline specification."""
        spec = NJOYPipelineSpec(
            name="Test Pipeline",
            description="Test processing",
            endf_library="ENDF/B-VIII.0",
            materials=[{"mat": 9228, "name": "U-235"}],
            group_structure=GroupStructure.VITAMIN_J,
        )
        
        assert spec.name == "Test Pipeline"
        assert spec.endf_library == "ENDF/B-VIII.0"
        assert len(spec.materials) == 1
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        spec = NJOYPipelineSpec(
            name="Test",
            description="Desc",
            endf_library="IRDFF-II",
            materials=[],
            group_structure=GroupStructure.SAND_II,
            temperatures=[300.0, 600.0],
        )
        
        d = spec.to_dict()
        
        assert d['name'] == "Test"
        assert d['group_structure'] == "SAND_II"
        assert d['temperatures'] == [300.0, 600.0]
        assert 'created_at' in d
    
    def test_from_dict(self):
        """Test dictionary deserialization."""
        data = {
            'name': 'Restored Pipeline',
            'description': 'From dict',
            'endf_library': 'ENDF/B-VIII.0',
            'materials': [{'mat': 7925}],
            'group_structure': 'VITAMIN_J',
            'temperatures': [300.0],
            'modules': ['reconr', 'broadr'],
            'tolerance': 0.001,
        }
        
        spec = NJOYPipelineSpec.from_dict(data)
        
        assert spec.name == "Restored Pipeline"
        assert spec.group_structure == GroupStructure.VITAMIN_J
        assert NJOYModule.RECONR in spec.modules
    
    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = NJOYPipelineSpec(
            name="Roundtrip Test",
            description="Testing serialization",
            endf_library="IRDFF-II",
            materials=[{"mat": 9228}, {"mat": 2631}],
            group_structure=GroupStructure.SAND_II,
            temperatures=[300.0, 500.0],
        )
        
        data = original.to_dict()
        restored = NJOYPipelineSpec.from_dict(data)
        
        assert restored.name == original.name
        assert restored.group_structure == original.group_structure
        assert restored.temperatures == original.temperatures


class TestDosimetryPipeline:
    """Tests for pre-built dosimetry pipeline."""
    
    def test_create_dosimetry_pipeline(self):
        """Test creating standard dosimetry pipeline."""
        pipeline = create_dosimetry_pipeline()
        
        assert "IRDFF" in pipeline.name
        assert pipeline.endf_library == "IRDFF-II"
        assert pipeline.group_structure == GroupStructure.SAND_II
        assert NJOYModule.GROUPR in pipeline.modules


class TestNJOYAvailability:
    """Tests for NJOY availability checking."""
    
    def test_check_njoy_available(self):
        """Test checking NJOY availability."""
        available, message = check_njoy_available()
        
        # Just check it returns valid types
        assert isinstance(available, bool)
        assert isinstance(message, str)
        assert len(message) > 0


class TestGroupStructureData:
    """Tests for group structure metadata."""
    
    def test_sand_ii_structure(self):
        """Test SAND-II structure data."""
        data = GROUP_STRUCTURE_DATA[GroupStructure.SAND_II]
        
        assert data['n_groups'] == 640
        assert 'activation' in data['description'].lower()
    
    def test_vitamin_j_structure(self):
        """Test VITAMIN-J structure data."""
        data = GROUP_STRUCTURE_DATA[GroupStructure.VITAMIN_J]
        
        assert data['n_groups'] == 175


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
