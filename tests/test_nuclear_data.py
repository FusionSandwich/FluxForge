"""
Tests for the unified nuclear data interface.

Tests NuclearData container, ENDF-IRDFF bridge, and temperature support.
"""

import numpy as np
import pytest

from fluxforge.data.nuclear_data import (
    DataType,
    DataLibrary,
    ReactionIdentifier,
    TemperatureData,
    NuclearData,
    ReactionMapping,
    ENDF_IRDFF_MAPPINGS,
    ENDFIRDFFBridge,
    create_temperature_set,
    interpolate_temperature,
    create_nuclear_data,
    create_multigroup_data,
)


class TestReactionIdentifier:
    """Tests for ReactionIdentifier dataclass."""
    
    def test_basic_creation(self):
        """Test creating a reaction identifier."""
        rxn = ReactionIdentifier(
            target="U-235",
            mt=18,
            product="fission products",
        )
        
        assert rxn.target == "U-235"
        assert rxn.mt == 18
    
    def test_reaction_string(self):
        """Test reaction string generation."""
        rxn = ReactionIdentifier(target="Au-197", mt=102)
        assert rxn.reaction_string == "Au-197(n,g)"
        
        rxn2 = ReactionIdentifier(target="Ni-58", mt=103)
        assert rxn2.reaction_string == "Ni-58(n,p)"
        
        rxn3 = ReactionIdentifier(target="Nb-93", mt=16)
        assert rxn3.reaction_string == "Nb-93(n,2n)"
    
    def test_hash_and_equality(self):
        """Test hashing and equality."""
        rxn1 = ReactionIdentifier(target="Fe-56", mt=103)
        rxn2 = ReactionIdentifier(target="Fe-56", mt=103)
        rxn3 = ReactionIdentifier(target="Fe-56", mt=102)
        
        assert rxn1 == rxn2
        assert rxn1 != rxn3
        assert hash(rxn1) == hash(rxn2)
        
        # Can be used in sets/dicts
        rxn_set = {rxn1, rxn2, rxn3}
        assert len(rxn_set) == 2


class TestTemperatureData:
    """Tests for TemperatureData dataclass."""
    
    def test_basic_creation(self):
        """Test creating temperature data."""
        energies = np.logspace(-5, 7, 100)
        values = np.ones(100)
        
        data = TemperatureData(
            temperature_K=300.0,
            energies=energies,
            values=values,
        )
        
        assert data.temperature_K == 300.0
        assert len(data.energies) == 100
    
    def test_interpolate(self):
        """Test energy interpolation."""
        energies = np.array([1.0, 10.0, 100.0, 1000.0])
        values = np.array([10.0, 5.0, 2.0, 1.0])
        
        data = TemperatureData(
            temperature_K=300.0,
            energies=energies,
            values=values,
        )
        
        # Exact point
        assert data.interpolate(10.0) == 5.0
        
        # Interpolated
        interp = data.interpolate(50.0)
        assert 2.0 < interp < 5.0


class TestNuclearData:
    """Tests for NuclearData container."""
    
    def test_basic_creation(self):
        """Test creating nuclear data object."""
        rxn_id = ReactionIdentifier(target="Au-197", mt=102)
        
        temp_data = TemperatureData(
            temperature_K=300.0,
            energies=np.logspace(-5, 7, 100),
            values=np.ones(100),
        )
        
        data = NuclearData(
            reaction_id=rxn_id,
            data_type=DataType.CONTINUOUS_ENERGY,
            library=DataLibrary.ENDF_B_VIII0,
            temperatures={300.0: temp_data},
        )
        
        assert data.reaction_id.target == "Au-197"
        assert data.data_type == DataType.CONTINUOUS_ENERGY
        assert data.available_temperatures == [300.0]
    
    def test_multiple_temperatures(self):
        """Test multi-temperature data."""
        rxn_id = ReactionIdentifier(target="Fe-56", mt=102)
        
        temps = {}
        for T in [300.0, 600.0, 900.0]:
            temps[T] = TemperatureData(
                temperature_K=T,
                energies=np.logspace(-5, 7, 50),
                values=np.ones(50) * (1 + T/1000),
            )
        
        data = NuclearData(
            reaction_id=rxn_id,
            data_type=DataType.CONTINUOUS_ENERGY,
            library=DataLibrary.ENDF_B_VIII0,
            temperatures=temps,
        )
        
        assert data.available_temperatures == [300.0, 600.0, 900.0]
    
    def test_get_at_temperature_exact(self):
        """Test getting data at exact temperature."""
        rxn_id = ReactionIdentifier(target="Fe-56", mt=102)
        
        temps = {
            300.0: TemperatureData(300.0, np.array([1.0]), np.array([10.0])),
            600.0: TemperatureData(600.0, np.array([1.0]), np.array([20.0])),
        }
        
        data = NuclearData(
            reaction_id=rxn_id,
            data_type=DataType.CONTINUOUS_ENERGY,
            library=DataLibrary.ENDF_B_VIII0,
            temperatures=temps,
        )
        
        result = data.get_at_temperature(300.0)
        assert result is not None
        assert result.values[0] == 10.0
    
    def test_get_at_temperature_interpolated(self):
        """Test temperature interpolation."""
        rxn_id = ReactionIdentifier(target="Fe-56", mt=102)
        
        temps = {
            300.0: TemperatureData(300.0, np.array([1.0]), np.array([10.0])),
            900.0: TemperatureData(900.0, np.array([1.0]), np.array([20.0])),
        }
        
        data = NuclearData(
            reaction_id=rxn_id,
            data_type=DataType.CONTINUOUS_ENERGY,
            library=DataLibrary.ENDF_B_VIII0,
            temperatures=temps,
        )
        
        result = data.get_at_temperature(600.0, interpolate=True)
        assert result is not None
        assert 10.0 < result.values[0] < 20.0
    
    def test_multigroup_data(self):
        """Test multi-group data properties."""
        rxn_id = ReactionIdentifier(target="Au-197", mt=102)
        
        boundaries = np.logspace(-5, 7, 51)
        
        data = NuclearData(
            reaction_id=rxn_id,
            data_type=DataType.MULTI_GROUP,
            library=DataLibrary.IRDFF_II,
            group_structure=boundaries,
        )
        
        assert data.n_groups == 50
        assert data.data_type == DataType.MULTI_GROUP
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        rxn_id = ReactionIdentifier(target="Au-197", mt=102, product="Au-198")
        
        temps = {
            300.0: TemperatureData(300.0, np.array([1.0, 10.0]), np.array([5.0, 2.0])),
        }
        
        data = NuclearData(
            reaction_id=rxn_id,
            data_type=DataType.CONTINUOUS_ENERGY,
            library=DataLibrary.IRDFF_II,
            temperatures=temps,
        )
        
        d = data.to_dict()
        
        assert d['reaction']['target'] == "Au-197"
        assert d['data_type'] == "ce"
        assert d['library'] == "IRDFF-II"
        assert '300.0' in d['temperatures']


class TestENDFIRDFFBridge:
    """Tests for ENDF-IRDFF bridge utilities."""
    
    def test_default_mappings(self):
        """Test default mapping table."""
        assert len(ENDF_IRDFF_MAPPINGS) > 10
        
        # Check some key mappings exist
        targets = [m.endf_target for m in ENDF_IRDFF_MAPPINGS]
        assert "Au-197" in targets
        assert "Ni-58" in targets
        assert "Al-27" in targets
    
    def test_bridge_creation(self):
        """Test creating bridge object."""
        bridge = ENDFIRDFFBridge()
        
        assert len(bridge.mappings) > 0
    
    def test_endf_to_irdff_mapping(self):
        """Test ENDF to IRDFF mapping."""
        bridge = ENDFIRDFFBridge()
        
        # Gold capture
        irdff = bridge.endf_to_irdff("Au-197", 102)
        assert irdff == "Au-197(n,g)Au-198"
        
        # Nickel (n,p)
        irdff = bridge.endf_to_irdff("Ni-58", 103)
        assert irdff == "Ni-58(n,p)Co-58"
        
        # Non-existent mapping
        irdff = bridge.endf_to_irdff("Xe-131", 102)
        assert irdff is None
    
    def test_irdff_to_endf_mapping(self):
        """Test IRDFF to ENDF mapping."""
        bridge = ENDFIRDFFBridge()
        
        endf = bridge.irdff_to_endf("Au-197(n,g)Au-198")
        assert endf == ("Au-197", 102)
        
        endf = bridge.irdff_to_endf("Nb-93(n,2n)Nb-92m")
        assert endf == ("Nb-93", 16)
    
    def test_list_mappings(self):
        """Test listing all mappings."""
        bridge = ENDFIRDFFBridge()
        
        mappings = bridge.list_mappings()
        assert len(mappings) > 0
        assert 'endf_target' in mappings[0]
        assert 'irdff_reaction' in mappings[0]
    
    def test_validate_consistency(self):
        """Test cross section consistency validation."""
        bridge = ENDFIRDFFBridge()
        
        # Consistent data
        endf_xs = np.array([1.0, 2.0, 3.0, 4.0])
        irdff_xs = np.array([1.05, 1.98, 3.02, 4.1])
        energies = np.array([1.0, 10.0, 100.0, 1000.0])
        
        consistent, max_dev, msg = bridge.validate_consistency(
            endf_xs, irdff_xs, energies, tolerance=0.1
        )
        
        assert consistent
        assert max_dev < 0.1
        assert "Consistent" in msg
    
    def test_validate_inconsistent(self):
        """Test detection of inconsistent data."""
        bridge = ENDFIRDFFBridge()
        
        endf_xs = np.array([1.0, 2.0, 3.0, 4.0])
        irdff_xs = np.array([2.0, 4.0, 6.0, 8.0])  # 100% different
        energies = np.array([1.0, 10.0, 100.0, 1000.0])
        
        consistent, max_dev, msg = bridge.validate_consistency(
            endf_xs, irdff_xs, energies, tolerance=0.1
        )
        
        assert not consistent
        assert max_dev > 0.1
        assert "exceeds" in msg


class TestTemperatureUtilities:
    """Tests for temperature-related utilities."""
    
    def test_create_temperature_set(self):
        """Test creating temperature set from base data."""
        base = TemperatureData(
            temperature_K=300.0,
            energies=np.logspace(-5, 7, 50),
            values=np.ones(50) * 10.0,
        )
        
        temps = create_temperature_set(base, [300.0, 600.0, 900.0])
        
        assert len(temps) == 3
        assert 300.0 in temps
        assert 600.0 in temps
        assert 900.0 in temps
    
    def test_interpolate_temperature(self):
        """Test temperature interpolation utility."""
        data_dict = {
            300.0: TemperatureData(300.0, np.array([1.0]), np.array([10.0])),
            900.0: TemperatureData(900.0, np.array([1.0]), np.array([20.0])),
        }
        
        result = interpolate_temperature(data_dict, 600.0)
        
        assert result is not None
        assert result.temperature_K == 600.0
        assert 10.0 < result.values[0] < 20.0
    
    def test_interpolate_boundary(self):
        """Test interpolation at boundaries."""
        data_dict = {
            300.0: TemperatureData(300.0, np.array([1.0]), np.array([10.0])),
            600.0: TemperatureData(600.0, np.array([1.0]), np.array([15.0])),
        }
        
        # Below range
        result = interpolate_temperature(data_dict, 200.0)
        assert result.values[0] == 10.0
        
        # Above range
        result = interpolate_temperature(data_dict, 800.0)
        assert result.values[0] == 15.0


class TestFactoryFunctions:
    """Tests for convenience factory functions."""
    
    def test_create_nuclear_data(self):
        """Test create_nuclear_data function."""
        energies = np.logspace(-5, 7, 100)
        values = np.ones(100) * 98.0  # Gold capture ~98 barns
        
        data = create_nuclear_data(
            target="Au-197",
            mt=102,
            energies=energies,
            values=values,
            temperature_K=300.0,
            library=DataLibrary.ENDF_B_VIII0,
            product="Au-198",
        )
        
        assert data.reaction_id.target == "Au-197"
        assert data.reaction_id.mt == 102
        assert data.data_type == DataType.CONTINUOUS_ENERGY
        assert 300.0 in data.temperatures
    
    def test_create_multigroup_data(self):
        """Test create_multigroup_data function."""
        boundaries = np.logspace(-5, 7, 51)
        values = np.ones(50) * 5.0
        
        data = create_multigroup_data(
            target="Fe-56",
            mt=102,
            group_boundaries=boundaries,
            group_values=values,
            library=DataLibrary.IRDFF_II,
        )
        
        assert data.data_type == DataType.MULTI_GROUP
        assert data.n_groups == 50
        assert data.group_structure is not None


class TestDataEnums:
    """Tests for data type enumerations."""
    
    def test_data_type_values(self):
        """Test DataType enum values."""
        assert DataType.CONTINUOUS_ENERGY.value == "ce"
        assert DataType.MULTI_GROUP.value == "mg"
    
    def test_data_library_values(self):
        """Test DataLibrary enum values."""
        assert DataLibrary.ENDF_B_VIII0.value == "ENDF/B-VIII.0"
        assert DataLibrary.IRDFF_II.value == "IRDFF-II"
        assert DataLibrary.JEFF_33.value == "JEFF-3.3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
