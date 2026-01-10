"""
Unit tests for ENDF covariance ingestion.

Tests MF33 parsing, validation, and SVD conditioning.
"""

import math
import pytest
import numpy as np

import sys
sys.path.insert(0, '/filespace/s/smandych/CAE/projects/ALARA/FluxForge/src')

from fluxforge.data.endf_covariance import (
    MF33Format,
    ReactionType,
    EnergyGrid,
    CovarianceMatrix,
    CovarianceValidationResult,
    CovarianceLibrary,
    DOSIMETRY_REACTIONS,
    read_endf_mf33_section,
    read_endf_mf31_section,
    read_endf_mf34_section,
    validate_covariance_matrix,
    condition_covariance_svd,
    ensure_positive_definite,
    create_default_dosimetry_library,
)


class TestENDFSectionParsing:
    """Targeted tests for ENDF section extraction/parsing helpers."""

    def test_mf33_list_parsing_uses_values(self, tmp_path):
        """MF33 parser should consume LIST payload rather than defaulting."""

        def endf_line(c1, c2, l1, l2, n1, n2, mat, mf, mt, nc=1):
            # 6 fields of width 11, then MAT(4) MF(2) MT(3) NC(5)
            return (
                f"{c1:11.5E}{c2:11.5E}{l1:11d}{l2:11d}{n1:11d}{n2:11d}"
                f"{mat:4d}{mf:2d}{mt:3d}{nc:5d}\n"
            )

        def endf_data(vals, mat, mf, mt, nc=2):
            fields = "".join(f"{v:11.5E}" for v in vals)
            fields = fields.ljust(66)
            return f"{fields}{mat:4d}{mf:2d}{mt:3d}{nc:5d}\n"

        mat = 125
        mt = 102
        # LIST values: boundaries (1,10,100,1000) then diagonal cov (0.1,0.2,0.3)
        npl = 7
        text = "".join(
            [
                endf_line(0.0, 0.0, 0, 0, npl, 0, mat, 33, mt, nc=1),
                endf_data([1.0, 10.0, 100.0, 1000.0, 0.1, 0.2], mat, 33, mt, nc=2),
                endf_data([0.3], mat, 33, mt, nc=3),
            ]
        )

        p = tmp_path / "synthetic.endf"
        p.write_text(text)

        rec = read_endf_mf33_section(p, mat=mat, mt=mt)
        assert rec is not None
        assert rec.mt == mt
        assert len(rec.energy_boundaries) == 4
        assert rec.energy_boundaries[0] == pytest.approx(1.0)
        assert rec.energy_boundaries[-1] == pytest.approx(1000.0)

        # Ensure we got the LIST covariance values (not default 0.01)
        assert rec.covariance_values[0] == pytest.approx(0.1)
        assert rec.covariance_values[1] == pytest.approx(0.2)
        assert rec.covariance_values[2] == pytest.approx(0.3)

        cov = rec.to_covariance_matrix()
        assert cov.n_groups == 3
        assert cov.matrix[0, 0] == pytest.approx(0.1)

        # Should not be marked degraded when full diagonal is present
        assert rec.degraded is False
        assert rec.degraded_warnings == []

    def test_mf33_incomplete_payload_is_flagged_degraded(self, tmp_path):
        """If MF33 does not contain enough values for inferred groups, it must be flagged."""

        def endf_line(c1, c2, l1, l2, n1, n2, mat, mf, mt, nc=1):
            return (
                f"{c1:11.5E}{c2:11.5E}{l1:11d}{l2:11d}{n1:11d}{n2:11d}"
                f"{mat:4d}{mf:2d}{mt:3d}{nc:5d}\n"
            )

        def endf_data(vals, mat, mf, mt, nc=2):
            fields = "".join(f"{v:11.5E}" for v in vals)
            fields = fields.ljust(66)
            return f"{fields}{mat:4d}{mf:2d}{mt:3d}{nc:5d}\n"

        mat = 125
        mt = 102
        # boundaries imply 3 groups (4 boundaries), but only provide 2 diagonal values
        npl = 6
        text = "".join(
            [
                endf_line(0.0, 0.0, 0, 0, npl, 0, mat, 33, mt, nc=1),
                endf_data([1.0, 10.0, 100.0, 1000.0, 0.1, 0.2], mat, 33, mt, nc=2),
            ]
        )

        p = tmp_path / "synthetic_incomplete.endf"
        p.write_text(text)

        rec = read_endf_mf33_section(p, mat=mat, mt=mt)
        assert rec is not None
        assert rec.degraded is True
        assert rec.degraded_warnings

    def test_mf31_mf34_section_accessors(self, tmp_path):
        """MF31/MF34 accessors should return a section when present."""

        def endf_line(c1, c2, l1, l2, n1, n2, mat, mf, mt, nc=1):
            return (
                f"{c1:11.5E}{c2:11.5E}{l1:11d}{l2:11d}{n1:11d}{n2:11d}"
                f"{mat:4d}{mf:2d}{mt:3d}{nc:5d}\n"
            )

        mat = 125
        text = "".join(
            [
                endf_line(0.0, 0.0, 0, 0, 0, 0, mat, 31, 452, nc=1),
                endf_line(0.0, 0.0, 0, 0, 0, 0, mat, 34, 2, nc=1),
            ]
        )
        p = tmp_path / "synthetic_sections.endf"
        p.write_text(text)

        s31 = read_endf_mf31_section(p, mat=mat, mt=452)
        s34 = read_endf_mf34_section(p, mat=mat, mt=2)

        assert s31 is not None
        assert s31.mf == 31
        assert s31.mt == 452

        assert s34 is not None
        assert s34.mf == 34
        assert s34.mt == 2


class TestEnergyGrid:
    """Test energy grid class."""
    
    def test_grid_creation(self):
        """Test energy grid creation."""
        energies = np.logspace(-5, 7, 26)  # 25 groups
        grid = EnergyGrid(energies_eV=energies)
        
        assert grid.n_groups == 25
        assert len(grid.group_widths_eV) == 25
        assert len(grid.group_centers_eV) == 25
    
    def test_lethargy_widths(self):
        """Test lethargy width calculation."""
        energies = np.array([1, 10, 100, 1000])
        grid = EnergyGrid(energies_eV=energies)
        
        # Equal lethargy spacing
        u = grid.lethargy_widths
        assert np.allclose(u, np.log(10))


class TestCovarianceMatrix:
    """Test covariance matrix class."""
    
    def test_matrix_creation(self):
        """Test matrix creation."""
        n = 10
        matrix = np.eye(n) * 0.01  # 10% diagonal
        energies = np.logspace(-5, 7, n + 1)
        
        cov = CovarianceMatrix(
            matrix=matrix,
            energy_grid=EnergyGrid(energies),
            is_relative=True,
            reaction_mt=102,
            material_za=79197,
        )
        
        assert cov.n_groups == n
        assert cov.reaction_mt == 102
    
    def test_diagonal_extraction(self):
        """Test diagonal variance extraction."""
        matrix = np.diag([0.01, 0.02, 0.03, 0.04])
        energies = np.array([1, 10, 100, 1000, 10000])
        
        cov = CovarianceMatrix(
            matrix=matrix,
            energy_grid=EnergyGrid(energies),
        )
        
        variances = cov.diagonal_variances
        assert len(variances) == 4
        assert variances[1] == pytest.approx(0.02)
        
        std_devs = cov.diagonal_std_devs
        assert std_devs[1] == pytest.approx(np.sqrt(0.02))
    
    def test_to_correlation(self):
        """Test conversion to correlation matrix."""
        # Covariance with off-diagonal elements
        matrix = np.array([
            [0.04, 0.01],
            [0.01, 0.09],
        ])
        energies = np.array([1, 10, 100])
        
        cov = CovarianceMatrix(
            matrix=matrix,
            energy_grid=EnergyGrid(energies),
        )
        
        corr = cov.to_correlation_matrix()
        
        # Diagonal should be 1
        assert corr[0, 0] == pytest.approx(1.0)
        assert corr[1, 1] == pytest.approx(1.0)
        
        # Off-diagonal should be r = cov / sqrt(var1 * var2)
        expected_corr = 0.01 / np.sqrt(0.04 * 0.09)
        assert corr[0, 1] == pytest.approx(expected_corr)


class TestCovarianceValidation:
    """Test covariance matrix validation."""
    
    def test_valid_matrix(self):
        """Valid covariance should pass validation."""
        n = 10
        cov = np.eye(n) * 0.01
        
        result = validate_covariance_matrix(cov)
        
        assert result.is_valid
        assert result.is_symmetric
        assert result.is_positive_definite
        assert not result.has_negative_diagonal
    
    def test_asymmetric_fails(self):
        """Asymmetric matrix should fail."""
        cov = np.array([
            [0.01, 0.002],
            [0.001, 0.01],  # Asymmetric
        ])
        
        result = validate_covariance_matrix(cov)
        
        assert not result.is_symmetric
        assert not result.is_valid
    
    def test_not_positive_definite(self):
        """Non-positive definite should fail."""
        cov = np.array([
            [0.01, 0.02],
            [0.02, 0.01],  # |cov(i,j)| > sqrt(var_i * var_j)
        ])
        
        result = validate_covariance_matrix(cov)
        
        assert not result.is_positive_definite
    
    def test_negative_diagonal(self):
        """Negative diagonal should fail."""
        cov = np.array([
            [-0.01, 0],
            [0, 0.01],
        ])
        
        result = validate_covariance_matrix(cov)
        
        assert result.has_negative_diagonal
        assert not result.is_valid
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        cov = np.eye(5) * 0.01
        result = validate_covariance_matrix(cov)
        
        summary = result.summary()
        
        assert "VALID" in summary
        assert "Symmetric" in summary


class TestSVDConditioning:
    """Test SVD-based covariance conditioning."""
    
    def test_well_conditioned_unchanged(self):
        """Well-conditioned matrix should be mostly unchanged."""
        n = 10
        cov = np.eye(n) * 0.01
        
        cov_cond, diag = condition_covariance_svd(cov)
        
        # Should be nearly unchanged
        assert np.allclose(cov, cov_cond, rtol=0.01)
    
    def test_poorly_conditioned_improved(self):
        """Poorly conditioned matrix should be improved."""
        n = 10
        # Create poorly conditioned matrix
        s = np.logspace(0, -12, n)  # Wide range of singular values
        U = np.eye(n)  # Simple case
        cov = U @ np.diag(s) @ U.T
        
        original_cond = s[0] / s[-1]
        
        cov_cond, diag = condition_covariance_svd(cov, target_condition=1e6)
        
        assert diag["new_condition"] < original_cond
        assert diag["n_truncated"] > 0
    
    def test_conditioning_preserves_symmetry(self):
        """Conditioned matrix should remain symmetric."""
        n = 10
        # Random symmetric positive definite
        A = np.random.randn(n, n)
        cov = A @ A.T + np.eye(n) * 0.01
        
        cov_cond, _ = condition_covariance_svd(cov)
        
        assert np.allclose(cov_cond, cov_cond.T)


class TestEnsurePositiveDefinite:
    """Test positive definiteness enforcement."""
    
    def test_already_pd(self):
        """Already PD matrix should be unchanged."""
        cov = np.eye(5) * 0.01
        
        cov_pd = ensure_positive_definite(cov)
        
        assert np.allclose(cov, cov_pd)
    
    def test_not_pd_fixed(self):
        """Non-PD matrix should be made PD."""
        # Create matrix with negative eigenvalue
        cov = np.array([
            [1.0, 1.5],
            [1.5, 1.0],
        ])
        
        cov_pd = ensure_positive_definite(cov, min_eigenvalue=1e-6)
        
        # Check it's now PD
        eigenvalues = np.linalg.eigvalsh(cov_pd)
        assert np.all(eigenvalues >= 1e-6)


class TestCovarianceLibrary:
    """Test covariance library class."""
    
    def test_library_creation(self):
        """Test library creation."""
        library = CovarianceLibrary("test_library")
        
        assert library.name == "test_library"
        assert len(library.reactions) == 0
    
    def test_add_covariance(self):
        """Test adding covariance data."""
        library = CovarianceLibrary("test")
        
        matrix = np.eye(10) * 0.01
        energies = np.logspace(-5, 7, 11)
        
        cov = CovarianceMatrix(
            matrix=matrix,
            energy_grid=EnergyGrid(energies),
            reaction_mt=102,
            material_za=79197,
        )
        
        library.add_covariance(79197, 102, cov, "test source")
        
        assert library.has_covariance(79197, 102)
        assert not library.has_covariance(79197, 103)
        
        retrieved = library.get_covariance(79197, 102)
        assert retrieved is not None
        assert retrieved.reaction_mt == 102
    
    def test_library_validation(self):
        """Test validating all library covariances."""
        library = CovarianceLibrary("test")
        
        # Add valid covariance
        matrix = np.eye(5) * 0.01
        energies = np.logspace(-5, 7, 6)
        
        cov = CovarianceMatrix(
            matrix=matrix,
            energy_grid=EnergyGrid(energies),
        )
        
        library.add_covariance(79197, 102, cov)
        
        results = library.validate_all()
        
        assert (79197, 102) in results
        assert results[(79197, 102)].is_valid
    
    def test_library_serialization(self):
        """Test library export to dict."""
        library = CovarianceLibrary("test_export")
        
        matrix = np.eye(5) * 0.01
        energies = np.logspace(-5, 7, 6)
        
        cov = CovarianceMatrix(
            matrix=matrix,
            energy_grid=EnergyGrid(energies),
            reaction_mt=102,
            material_za=79197,
        )
        
        library.add_covariance(79197, 102, cov)
        
        data = library.to_dict()
        
        assert "schema" in data
        assert data["n_reactions"] == 1


class TestDosimetryReactions:
    """Test dosimetry reaction catalog."""
    
    def test_reactions_defined(self):
        """Standard dosimetry reactions should be defined."""
        assert "Au-197(n,g)Au-198" in DOSIMETRY_REACTIONS
        assert "Ni-58(n,p)Co-58" in DOSIMETRY_REACTIONS
        assert "Al-27(n,a)Na-24" in DOSIMETRY_REACTIONS
    
    def test_reaction_za_mt(self):
        """Reactions should have correct ZA and MT."""
        za, mt = DOSIMETRY_REACTIONS["Au-197(n,g)Au-198"]
        assert za == 79197
        assert mt == 102


class TestDefaultDosimetryLibrary:
    """Test default dosimetry library creation."""
    
    def test_library_creation(self):
        """Test creating default library."""
        library = create_default_dosimetry_library()
        
        assert len(library.reactions) > 0
        assert library.name == "default_dosimetry"
    
    def test_library_has_gold(self):
        """Library should have Au-197(n,g) covariance."""
        library = create_default_dosimetry_library()
        
        assert library.has_covariance(79197, 102)
    
    def test_library_covariances_valid(self):
        """All covariances should be valid."""
        library = create_default_dosimetry_library()
        
        results = library.validate_all()
        
        for key, result in results.items():
            assert result.is_valid, f"Invalid covariance for {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
