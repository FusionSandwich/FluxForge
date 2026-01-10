"""Tests for interoperability module (STAYSL data exchange)."""

import numpy as np
import pytest
from pathlib import Path

from fluxforge.io.interop import (
    SaturationRateData,
    read_saturation_rates_csv,
    write_saturation_rates_csv,
    read_lower_triangular_matrix,
    write_lower_triangular_matrix,
    STAYSLBundle,
    export_staysl_bundle,
    import_staysl_bundle,
)


class TestSaturationRatesCSV:
    """Tests for saturation rate CSV I/O."""

    def test_read_simple_csv(self, tmp_path):
        csv_file = tmp_path / "rates.csv"
        csv_file.write_text("""reaction,rate,uncertainty
Au197_ng,1.5e-14,1.5e-15
Fe58_ng,2.0e-16,2.0e-17
""")
        
        data = read_saturation_rates_csv(csv_file)
        
        assert len(data) == 2
        assert data[0].reaction_id == "Au197_ng"
        assert data[0].rate == pytest.approx(1.5e-14)
        assert data[0].uncertainty == pytest.approx(1.5e-15)

    def test_read_with_optional_columns(self, tmp_path):
        csv_file = tmp_path / "rates.csv"
        csv_file.write_text("""reaction,rate,uncertainty,half_life_s,target_atoms,notes
Au197,1e-14,1e-15,233280,1e18,gold wire
""")
        
        data = read_saturation_rates_csv(csv_file)
        
        assert len(data) == 1
        assert data[0].half_life_s == pytest.approx(233280)
        assert data[0].target_atoms == pytest.approx(1e18)
        assert data[0].notes == "gold wire"

    def test_read_with_custom_delimiter(self, tmp_path):
        csv_file = tmp_path / "rates.tsv"
        csv_file.write_text("reaction\trate\tuncertainty\nAu197\t1e-14\t1e-15\n")
        
        data = read_saturation_rates_csv(csv_file, delimiter="\t")
        
        assert len(data) == 1
        assert data[0].reaction_id == "Au197"

    def test_write_and_read_roundtrip(self, tmp_path):
        original = [
            SaturationRateData("Au197_ng", 1.5e-14, 1.5e-15, 233280, 1e18, "test"),
            SaturationRateData("Fe58_ng", 2.0e-16, 2.0e-17),
        ]
        
        csv_file = tmp_path / "roundtrip.csv"
        write_saturation_rates_csv(original, csv_file)
        
        loaded = read_saturation_rates_csv(csv_file)
        
        assert len(loaded) == 2
        assert loaded[0].reaction_id == "Au197_ng"
        assert loaded[0].rate == pytest.approx(1.5e-14, rel=1e-5)


class TestLowerTriangularMatrix:
    """Tests for lower-triangular matrix I/O."""

    def test_read_simple(self, tmp_path):
        mat_file = tmp_path / "matrix.txt"
        mat_file.write_text("""1.0
0.5 2.0
0.3 0.4 3.0
""")
        
        matrix = read_lower_triangular_matrix(mat_file, n=3)
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[1, 0] == pytest.approx(0.5)
        assert matrix[0, 1] == pytest.approx(0.5)  # Symmetric
        assert matrix[2, 2] == pytest.approx(3.0)

    def test_read_infer_size(self, tmp_path):
        """Test automatic size inference from element count."""
        mat_file = tmp_path / "matrix.txt"
        # 6 elements = 3x3 lower triangular
        mat_file.write_text("1 2 3 4 5 6")
        
        matrix = read_lower_triangular_matrix(mat_file)
        
        assert matrix.shape == (3, 3)

    def test_read_with_comments(self, tmp_path):
        mat_file = tmp_path / "matrix.txt"
        mat_file.write_text("""# Comment line
1.0
# Another comment
0.5 2.0
""")
        
        matrix = read_lower_triangular_matrix(mat_file, n=2)
        
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == pytest.approx(1.0)

    def test_write_and_read_roundtrip(self, tmp_path):
        original = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 2.0, 0.5],
            [0.2, 0.5, 3.0],
        ])
        
        mat_file = tmp_path / "roundtrip.txt"
        write_lower_triangular_matrix(original, mat_file)
        
        loaded = read_lower_triangular_matrix(mat_file, n=3)
        
        np.testing.assert_array_almost_equal(loaded, original)

    def test_invalid_element_count_raises(self, tmp_path):
        mat_file = tmp_path / "bad.txt"
        mat_file.write_text("1 2 3 4 5")  # 5 elements, not triangular
        
        with pytest.raises(ValueError, match="Cannot form square matrix"):
            read_lower_triangular_matrix(mat_file)


class TestSTAYSLBundle:
    """Tests for STAYSLBundle."""

    def create_test_bundle(self):
        """Create a simple test bundle."""
        n_groups = 3
        n_reactions = 2
        
        return STAYSLBundle(
            prior_flux=np.array([1e10, 1e9, 1e8]),
            prior_covariance=np.diag([1e18, 1e16, 1e14]),
            energy_bounds_eV=np.array([1e-3, 1.0, 1e3, 1e6]),
            reaction_ids=["Au197_ng", "Fe58_ng"],
            measured_rates=np.array([1.5e-14, 2.0e-16]),
            measurement_covariance=np.diag([1e-28, 1e-32]),
            response_matrix=np.array([
                [100.0, 1.0, 0.01],
                [1.0, 0.5, 0.1],
            ]),
            title="Test Bundle",
            notes="For testing",
        )

    def test_bundle_creation(self):
        bundle = self.create_test_bundle()
        
        assert len(bundle.prior_flux) == 3
        assert len(bundle.reaction_ids) == 2
        assert bundle.response_matrix.shape == (2, 3)

    def test_bundle_validation_pass(self):
        bundle = self.create_test_bundle()
        warnings = bundle.validate()
        
        assert len(warnings) == 0

    def test_bundle_validation_fail(self):
        bundle = self.create_test_bundle()
        # Corrupt dimensions
        bundle.prior_covariance = np.eye(5)  # Wrong size
        
        warnings = bundle.validate()
        
        assert len(warnings) > 0
        assert "Prior covariance" in warnings[0]


class TestSTAYSLBundleIO:
    """Tests for STAYSL bundle import/export."""

    def create_test_bundle(self):
        """Create a simple test bundle."""
        return STAYSLBundle(
            prior_flux=np.array([1e10, 1e9, 1e8]),
            prior_covariance=np.diag([1e18, 1e16, 1e14]),
            energy_bounds_eV=np.array([1e-3, 1.0, 1e3, 1e6]),
            reaction_ids=["Au197_ng", "Fe58_ng"],
            measured_rates=np.array([1.5e-14, 2.0e-16]),
            measurement_covariance=np.diag([1e-28, 1e-32]),
            response_matrix=np.array([
                [100.0, 1.0, 0.01],
                [1.0, 0.5, 0.1],
            ]),
            title="Test Bundle",
            notes="For testing",
        )

    def test_export_creates_files(self, tmp_path):
        bundle = self.create_test_bundle()
        
        files = export_staysl_bundle(bundle, tmp_path, prefix="test")
        
        assert "prior_flux" in files
        assert "prior_covariance" in files
        assert "measurements" in files
        assert "response_matrix" in files
        assert "metadata" in files
        
        for path in files.values():
            assert path.exists()

    def test_export_import_roundtrip(self, tmp_path):
        original = self.create_test_bundle()
        
        export_staysl_bundle(original, tmp_path, prefix="roundtrip")
        loaded = import_staysl_bundle(tmp_path, prefix="roundtrip")
        
        # Check key attributes
        np.testing.assert_array_almost_equal(loaded.prior_flux, original.prior_flux)
        np.testing.assert_array_almost_equal(loaded.energy_bounds_eV, original.energy_bounds_eV)
        assert loaded.reaction_ids == original.reaction_ids
        np.testing.assert_array_almost_equal(loaded.measured_rates, original.measured_rates, decimal=5)
        np.testing.assert_array_almost_equal(loaded.response_matrix, original.response_matrix, decimal=5)
        assert loaded.title == original.title

    def test_export_subdirectory(self, tmp_path):
        bundle = self.create_test_bundle()
        subdir = tmp_path / "nested" / "path"
        
        files = export_staysl_bundle(bundle, subdir)
        
        assert subdir.exists()
        assert len(files) > 0
