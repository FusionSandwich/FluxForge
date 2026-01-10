"""
ENDF MF33 covariance data ingestion.

This module provides capabilities to ingest and validate nuclear data
covariance information from ENDF-formatted files, particularly MF33
(reaction cross section covariances).

Provides:
- ENDF MF33 parsing and validation
- Covariance matrix extraction and conditioning
- SVD conditioning for poorly-conditioned matrices
- Integration with IRDFF-II and ENDF/B-VIII.0

Reference: ENDF-102 manual, MF33 format specification
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg


class MF33Format(Enum):
    """MF33 sub-library formats."""
    
    LB0 = 0  # Absolute covariance
    LB1 = 1  # Relative covariance (most common)
    LB2 = 2  # Relative with cross terms
    LB5 = 5  # Compact format
    LB6 = 6  # Correlation + diagonal variances


class ReactionType(Enum):
    """Common neutron reaction types for dosimetry."""
    
    N_GAMMA = 102  # (n,γ)
    N_P = 103      # (n,p)
    N_ALPHA = 107  # (n,α)
    N_2N = 16      # (n,2n)
    N_NP = 28      # (n,n'p)
    N_F = 18       # (n,f)
    N_TOTAL = 1    # total
    N_ELASTIC = 2  # elastic


@dataclass
class EnergyGrid:
    """
    Energy grid for covariance data.
    
    Attributes:
        energies_eV: Energy boundaries in eV
        n_groups: Number of energy groups
        group_widths_eV: Group widths
    """
    
    energies_eV: np.ndarray
    
    @property
    def n_groups(self) -> int:
        return len(self.energies_eV) - 1
    
    @property
    def group_widths_eV(self) -> np.ndarray:
        return np.diff(self.energies_eV)
    
    @property
    def group_centers_eV(self) -> np.ndarray:
        """Geometric mean of group boundaries."""
        return np.sqrt(self.energies_eV[:-1] * self.energies_eV[1:])
    
    @property
    def lethargy_widths(self) -> np.ndarray:
        """Lethargy width of each group."""
        return np.log(self.energies_eV[1:] / self.energies_eV[:-1])


@dataclass
class CovarianceMatrix:
    """
    Covariance matrix with metadata.
    
    Attributes:
        matrix: Covariance matrix (relative or absolute)
        energy_grid: Associated energy grid
        is_relative: True if relative covariance (σ²/μ²)
        is_correlation: True if correlation matrix
        reaction_mt: ENDF MT number
        material_za: ZA identifier
    """
    
    matrix: np.ndarray
    energy_grid: EnergyGrid
    is_relative: bool = True
    is_correlation: bool = False
    reaction_mt: int = 0
    material_za: int = 0
    source: str = ""
    
    @property
    def n_groups(self) -> int:
        return self.matrix.shape[0]
    
    @property
    def diagonal_variances(self) -> np.ndarray:
        """Extract diagonal variances."""
        return np.diag(self.matrix)
    
    @property
    def diagonal_std_devs(self) -> np.ndarray:
        """Extract diagonal standard deviations."""
        return np.sqrt(np.maximum(self.diagonal_variances, 0))
    
    def to_correlation_matrix(self) -> np.ndarray:
        """Convert to correlation matrix."""
        if self.is_correlation:
            return self.matrix.copy()
        
        d = np.sqrt(np.diag(self.matrix))
        d[d == 0] = 1e-10  # Avoid division by zero
        D_inv = np.diag(1.0 / d)
        return D_inv @ self.matrix @ D_inv


@dataclass
class CovarianceValidationResult:
    """Result of covariance matrix validation."""
    
    is_valid: bool
    is_symmetric: bool
    is_positive_definite: bool
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    has_negative_diagonal: bool
    max_correlation: float
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary string."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Covariance Validation: {status}",
            f"  Symmetric: {self.is_symmetric}",
            f"  Positive Definite: {self.is_positive_definite}",
            f"  Eigenvalue range: [{self.min_eigenvalue:.2e}, {self.max_eigenvalue:.2e}]",
            f"  Condition number: {self.condition_number:.2e}",
            f"  Max off-diagonal correlation: {self.max_correlation:.3f}",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def validate_covariance_matrix(
    cov: np.ndarray,
    tol_symmetry: float = 1e-10,
    tol_eigenvalue: float = -1e-10,
    max_condition: float = 1e12,
) -> CovarianceValidationResult:
    """
    Validate covariance matrix properties.
    
    Checks:
    - Symmetry
    - Positive semi-definiteness
    - Conditioning
    - Correlation bounds
    
    Args:
        cov: Covariance matrix
        tol_symmetry: Tolerance for symmetry check
        tol_eigenvalue: Minimum acceptable eigenvalue
        max_condition: Maximum acceptable condition number
        
    Returns:
        CovarianceValidationResult
    """
    warnings = []
    n = cov.shape[0]
    
    # Check symmetry
    asymmetry = np.max(np.abs(cov - cov.T))
    is_symmetric = asymmetry < tol_symmetry
    if not is_symmetric:
        warnings.append(f"Matrix asymmetry: {asymmetry:.2e}")
    
    # Symmetrize for eigenvalue computation
    cov_sym = (cov + cov.T) / 2
    
    # Check eigenvalues
    try:
        eigenvalues = linalg.eigvalsh(cov_sym)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        is_pd = min_eig > tol_eigenvalue
        
        if max_eig > 0:
            condition = max_eig / max(min_eig, 1e-100)
        else:
            condition = np.inf
    except Exception:
        eigenvalues = np.array([])
        min_eig = np.nan
        max_eig = np.nan
        is_pd = False
        condition = np.inf
        warnings.append("Eigenvalue computation failed")
    
    if not is_pd:
        warnings.append(f"Matrix not positive definite, min eigenvalue: {min_eig:.2e}")
    
    if condition > max_condition:
        warnings.append(f"Poorly conditioned: {condition:.2e}")
    
    # Check diagonal
    diag = np.diag(cov)
    has_negative_diagonal = np.any(diag < 0)
    if has_negative_diagonal:
        warnings.append("Negative diagonal elements (negative variance)")
    
    # Check correlations
    d = np.sqrt(np.maximum(diag, 0))
    d[d == 0] = 1
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 0)
    max_corr = np.max(np.abs(corr))
    
    if max_corr > 1.0:
        warnings.append(f"Correlation exceeds 1: {max_corr:.3f}")
    
    is_valid = is_symmetric and is_pd and not has_negative_diagonal and max_corr <= 1.0
    
    return CovarianceValidationResult(
        is_valid=is_valid,
        is_symmetric=is_symmetric,
        is_positive_definite=is_pd,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        condition_number=condition,
        has_negative_diagonal=has_negative_diagonal,
        max_correlation=max_corr,
        warnings=warnings,
    )


def condition_covariance_svd(
    cov: np.ndarray,
    target_condition: float = 1e6,
    min_singular_fraction: float = 1e-8,
) -> Tuple[np.ndarray, Dict]:
    """
    Condition a covariance matrix using SVD truncation.
    
    Truncates small singular values to improve conditioning
    while preserving matrix structure.
    
    Args:
        cov: Original covariance matrix
        target_condition: Target condition number
        min_singular_fraction: Minimum fraction of max singular value
        
    Returns:
        Tuple of (conditioned matrix, diagnostics dict)
    """
    # Symmetrize
    cov_sym = (cov + cov.T) / 2
    
    # SVD decomposition
    U, s, Vh = linalg.svd(cov_sym, full_matrices=False)
    
    s_original = s.copy()
    s_max = s[0]
    
    # Apply minimum threshold
    threshold = max(
        s_max / target_condition,
        s_max * min_singular_fraction,
    )
    
    n_truncated = np.sum(s < threshold)
    s[s < threshold] = threshold
    
    # Reconstruct
    cov_conditioned = U @ np.diag(s) @ Vh
    
    # Ensure symmetry
    cov_conditioned = (cov_conditioned + cov_conditioned.T) / 2
    
    diagnostics = {
        "original_condition": s_original[0] / s_original[-1] if s_original[-1] > 0 else np.inf,
        "new_condition": s[0] / s[-1],
        "n_truncated": int(n_truncated),
        "threshold": threshold,
        "singular_values_original": s_original,
        "singular_values_conditioned": s,
    }
    
    return cov_conditioned, diagnostics


def ensure_positive_definite(
    cov: np.ndarray,
    min_eigenvalue: float = 1e-10,
) -> np.ndarray:
    """
    Ensure matrix is positive definite by adjusting eigenvalues.
    
    Projects to nearest positive definite matrix.
    
    Args:
        cov: Input matrix
        min_eigenvalue: Minimum eigenvalue to enforce
        
    Returns:
        Positive definite matrix
    """
    cov_sym = (cov + cov.T) / 2
    
    eigenvalues, eigenvectors = linalg.eigh(cov_sym)
    
    # Clip eigenvalues
    eigenvalues_clipped = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct
    cov_pd = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
    
    return cov_pd


@dataclass
class ENDFCovarianceRecord:
    """
    Parsed ENDF covariance record.
    
    Represents a single MF33 section for one reaction.
    """
    
    mat: int  # Material number
    mt: int   # Reaction type
    za: int   # ZA identifier
    energy_boundaries: np.ndarray
    covariance_values: np.ndarray
    lb_format: MF33Format
    raw_text: Optional[str] = None
    degraded: bool = False
    degraded_warnings: List[str] = field(default_factory=list)
    
    def to_covariance_matrix(self) -> CovarianceMatrix:
        """Convert to CovarianceMatrix object."""
        n_groups = len(self.energy_boundaries) - 1
        
        if len(self.covariance_values) == n_groups:
            # Diagonal only
            matrix = np.diag(self.covariance_values)
        elif len(self.covariance_values) == n_groups * n_groups:
            # Full matrix
            matrix = self.covariance_values.reshape((n_groups, n_groups))
        else:
            # Try to interpret as lower triangular
            n = n_groups
            matrix = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    if idx < len(self.covariance_values):
                        matrix[i, j] = self.covariance_values[idx]
                        matrix[j, i] = matrix[i, j]
                        idx += 1
        
        # Determine if relative
        is_relative = self.lb_format in [MF33Format.LB1, MF33Format.LB2]
        
        return CovarianceMatrix(
            matrix=matrix,
            energy_grid=EnergyGrid(self.energy_boundaries),
            is_relative=is_relative,
            reaction_mt=self.mt,
            material_za=self.za,
            source=f"ENDF MAT{self.mat} MT{self.mt}",
        )


def parse_endf_line(line: str) -> Tuple[float, float, int, int, int, int]:
    """
    Parse a standard ENDF formatted line.
    
    ENDF uses 11-character fields for floats, 5-character for integers.
    """
    try:
        c1 = float(line[0:11].replace('D', 'E').replace('d', 'e').strip() or '0')
        c2 = float(line[11:22].replace('D', 'E').replace('d', 'e').strip() or '0')
        l1 = int(line[22:33].strip() or '0')
        l2 = int(line[33:44].strip() or '0')
        n1 = int(line[44:55].strip() or '0')
        n2 = int(line[55:66].strip() or '0')
    except (ValueError, IndexError):
        return 0.0, 0.0, 0, 0, 0, 0
    
    return c1, c2, l1, l2, n1, n2


def parse_endf_float_field(field: str) -> float:
    """Parse an 11-character ENDF float field."""
    try:
        return float(field.replace('D', 'E').replace('d', 'e').strip() or '0')
    except ValueError:
        return 0.0


def parse_endf_six_floats(line: str) -> List[float]:
    """Parse up to 6 float fields from an ENDF data line (columns 1-66)."""
    floats = []
    for i in range(0, 66, 11):
        floats.append(parse_endf_float_field(line[i:i + 11]))
    return floats


@dataclass
class ENDFSection:
    """Generic ENDF section capture (MAT/MF/MT) with optional LIST payload."""

    mat: int
    mf: int
    mt: int
    raw_text: str
    list_records: List[np.ndarray] = field(default_factory=list)


def _extract_section_lines_from_text(
    text: str,
    mat: int,
    mf: int,
    mt: int,
) -> List[str]:
    lines = text.splitlines(True)
    out: List[str] = []
    in_section = False
    for line in lines:
        if len(line) < 75:
            continue
        try:
            line_mat = int(line[66:70])
            line_mf = int(line[70:72])
            line_mt = int(line[72:75])
        except ValueError:
            continue

        if line_mat == mat and line_mf == mf and line_mt == mt:
            in_section = True
            out.append(line)
            continue
        if in_section:
            # Section ends when MF/MT changes
            if line_mf != mf or line_mt != mt or line_mat != mat:
                break
            out.append(line)
    return out


def read_endf_section_text(
    text: str,
    mat: int,
    mf: int,
    mt: int,
) -> Optional[ENDFSection]:
    """Read an ENDF section from raw text and parse LIST records.

    This is intentionally partial: it focuses on extracting LIST payloads (NPL items)
    which is sufficient for many covariance sections.
    """
    section_lines = _extract_section_lines_from_text(text, mat, mf, mt)
    if not section_lines:
        return None

    list_records: List[np.ndarray] = []

    i = 0
    while i < len(section_lines):
        c1, c2, l1, l2, n1, n2 = parse_endf_line(section_lines[i])
        npl = int(n1)
        if npl > 0:
            values: List[float] = []
            j = i + 1
            while j < len(section_lines) and len(values) < npl:
                values.extend(parse_endf_six_floats(section_lines[j]))
                j += 1
            values = values[:npl]
            list_records.append(np.array(values, dtype=float))
            i = j
        else:
            i += 1

    return ENDFSection(
        mat=mat,
        mf=mf,
        mt=mt,
        raw_text="".join(section_lines),
        list_records=list_records,
    )


def read_endf_section(
    file_path: Union[str, Path],
    mat: int,
    mf: int,
    mt: int,
) -> Optional[ENDFSection]:
    """Read an ENDF section from a file."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    return read_endf_section_text(file_path.read_text(errors="ignore"), mat, mf, mt)


def read_endf_mf33_section(
    file_path: Union[str, Path],
    mat: int,
    mt: int,
) -> Optional[ENDFCovarianceRecord]:
    """
    Read MF33 section from ENDF file.
    
    This is a simplified parser that handles common formats.
    For production use, consider using more robust libraries.
    
    Args:
        file_path: Path to ENDF file
        mat: Material number
        mt: Reaction MT number
        
    Returns:
        ENDFCovarianceRecord if found, None otherwise
    """
    section = read_endf_section(file_path, mat=mat, mf=33, mt=mt)
    if section is None:
        return None

    lines = section.raw_text.splitlines(True)
    if len(lines) < 1:
        return None

    za, awr, l1, l2, n1, n2 = parse_endf_line(lines[0])

    # Heuristic: if LIST records exist, try to interpret the first increasing
    # sequence as energy boundaries and the remaining values as covariance entries.
    boundaries: Optional[np.ndarray] = None
    cov_values: Optional[np.ndarray] = None

    if section.list_records:
        flat = np.concatenate(section.list_records) if len(section.list_records) > 1 else section.list_records[0]
        # Remove zeros and negatives
        flat_pos = flat[np.isfinite(flat) & (flat > 0)]
        if len(flat_pos) >= 3:
            # Try to take a strictly increasing prefix as boundaries
            b = [float(flat_pos[0])]
            for v in flat_pos[1:]:
                if v > b[-1] * (1.0 + 1e-12):
                    b.append(float(v))
                else:
                    # Stop once we have a plausible boundary list
                    if len(b) >= 4:
                        break
            if len(b) >= 3:
                boundaries = np.array(b, dtype=float)
                remainder = flat_pos[len(b):]
                if len(remainder) > 0:
                    cov_values = remainder.copy()

    if boundaries is None:
        # Fallback: use C1/C2 fields across the section as candidate energies
        energies: List[float] = []
        for line in lines:
            c1, c2, _, _, _, _ = parse_endf_line(line)
            if c1 > 0:
                energies.append(c1)
            if c2 > 0:
                energies.append(c2)
        if len(energies) < 2:
            return None
        boundaries = np.array(sorted(set(energies)), dtype=float)

    n_groups = max(len(boundaries) - 1, 0)
    if n_groups <= 0:
        return None

    degraded = False
    degraded_warnings: List[str] = []

    if cov_values is None or len(cov_values) == 0:
        # Default to 10% relative uncertainty diagonal if we couldn't parse entries.
        # This is explicitly flagged as degraded (no silent drop).
        covariance_values = np.ones(n_groups, dtype=float) * 0.01
        degraded = True
        degraded_warnings.append(
            "MF33 section had no parseable covariance values; using a conservative default diagonal (0.01)."
        )
    else:
        # If values look like diagonal only, accept n_groups values.
        if len(cov_values) >= n_groups:
            covariance_values = np.array(cov_values[:n_groups], dtype=float)
        else:
            # Not enough values; pad with a conservative default.
            covariance_values = np.concatenate([
                np.array(cov_values, dtype=float),
                np.ones(n_groups - len(cov_values), dtype=float) * 0.01,
            ])
            degraded = True
            degraded_warnings.append(
                "MF33 covariance LIST payload was shorter than the inferred number of groups; padding missing values with 0.01."
            )

    return ENDFCovarianceRecord(
        mat=mat,
        mt=mt,
        za=int(za),
        energy_boundaries=boundaries,
        covariance_values=covariance_values,
        lb_format=MF33Format.LB1,
        raw_text=section.raw_text,
        degraded=degraded,
        degraded_warnings=degraded_warnings,
    )


def read_endf_mf31_section(
    file_path: Union[str, Path],
    mat: int,
    mt: int,
) -> Optional[ENDFSection]:
    """Read MF31 (nubar covariance) section as a generic ENDFSection."""
    return read_endf_section(file_path, mat=mat, mf=31, mt=mt)


def read_endf_mf34_section(
    file_path: Union[str, Path],
    mat: int,
    mt: int,
) -> Optional[ENDFSection]:
    """Read MF34 (angular distribution covariance) section as a generic ENDFSection."""
    return read_endf_section(file_path, mat=mat, mf=34, mt=mt)


class CovarianceLibrary:
    """
    Library of covariance data for multiple reactions.
    
    Provides unified access to covariance information from
    ENDF, IRDFF-II, or other sources.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self._covariances: Dict[Tuple[int, int], CovarianceMatrix] = {}
        self._sources: Dict[Tuple[int, int], str] = {}
        self._degraded: Dict[Tuple[int, int], bool] = {}
        self._degraded_warnings: Dict[Tuple[int, int], List[str]] = {}
    
    def add_covariance(
        self,
        za: int,
        mt: int,
        covariance: CovarianceMatrix,
        source: str = "",
        degraded: bool = False,
        degraded_warnings: Optional[List[str]] = None,
    ):
        """Add covariance data for a reaction."""
        key = (za, mt)
        self._covariances[key] = covariance
        self._sources[key] = source or covariance.source
        self._degraded[key] = bool(degraded)
        self._degraded_warnings[key] = list(degraded_warnings or [])
    
    def get_covariance(
        self,
        za: int,
        mt: int,
    ) -> Optional[CovarianceMatrix]:
        """Get covariance for a reaction."""
        return self._covariances.get((za, mt))
    
    def has_covariance(self, za: int, mt: int) -> bool:
        """Check if covariance data exists for reaction."""
        return (za, mt) in self._covariances
    
    @property
    def reactions(self) -> List[Tuple[int, int]]:
        """List of (ZA, MT) pairs with covariance data."""
        return list(self._covariances.keys())
    
    def validate_all(self) -> Dict[Tuple[int, int], CovarianceValidationResult]:
        """Validate all covariance matrices."""
        results = {}
        for key, cov in self._covariances.items():
            results[key] = validate_covariance_matrix(cov.matrix)
        return results
    
    def to_dict(self) -> dict:
        """Export library summary."""
        return {
            "schema": "fluxforge.covariance_library.v1",
            "name": self.name,
            "n_reactions": len(self._covariances),
            "reactions": [
                {
                    "za": za,
                    "mt": mt,
                    "n_groups": self._covariances[(za, mt)].n_groups,
                    "source": self._sources.get((za, mt), ""),
                    "covariance_degraded": bool(self._degraded.get((za, mt), False)),
                    "covariance_degraded_warnings": list(self._degraded_warnings.get((za, mt), [])),
                }
                for za, mt in self.reactions
            ],
        }


# Standard dosimetry reactions
DOSIMETRY_REACTIONS = {
    "Au-197(n,g)Au-198": (79197, 102),
    "Co-59(n,g)Co-60": (27059, 102),
    "Ni-58(n,p)Co-58": (28058, 103),
    "Fe-54(n,p)Mn-54": (26054, 103),
    "Ti-46(n,p)Sc-46": (22046, 103),
    "Fe-56(n,p)Mn-56": (26056, 103),
    "Al-27(n,a)Na-24": (13027, 107),
    "Ni-58(n,2n)Ni-57": (28058, 16),
    "Cu-63(n,a)Co-60": (29063, 107),
    "Nb-93(n,2n)Nb-92m": (41093, 16),
    "Zr-90(n,2n)Zr-89": (40090, 16),
    "In-115(n,n')In-115m": (49115, 4),
}


def create_default_dosimetry_library() -> CovarianceLibrary:
    """
    Create default covariance library for common dosimetry reactions.
    
    Uses simplified covariance models based on typical ENDF uncertainties.
    For production use, load actual ENDF/IRDFF-II covariances.
    
    Returns:
        CovarianceLibrary with default covariances
    """
    library = CovarianceLibrary("default_dosimetry")
    
    # Default 25-group energy structure (simplified)
    energy_boundaries = np.logspace(
        np.log10(1e-5),  # 10 meV
        np.log10(20e6),  # 20 MeV
        26
    )
    
    for name, (za, mt) in DOSIMETRY_REACTIONS.items():
        n_groups = 25
        
        # Default uncertainty model based on reaction type
        if mt == 102:  # (n,γ) - typically well-known
            base_uncertainty = 0.05  # 5%
        elif mt in [103, 107]:  # (n,p), (n,α) - threshold reactions
            base_uncertainty = 0.10  # 10%
        elif mt == 16:  # (n,2n) - high threshold
            base_uncertainty = 0.15  # 15%
        else:
            base_uncertainty = 0.10
        
        # Create diagonal covariance with energy-dependent uncertainty
        variances = np.ones(n_groups) * base_uncertainty**2
        
        # Increase uncertainty at threshold and high energies
        variances[-5:] *= 4  # Higher uncertainty at high energies
        variances[:3] *= 2   # Higher uncertainty at thermal
        
        # Create simple correlation structure
        corr = np.eye(n_groups)
        for i in range(n_groups):
            for j in range(n_groups):
                if i != j:
                    dist = abs(i - j)
                    corr[i, j] = 0.5 * np.exp(-dist / 5)  # Correlation length ~5 groups
        
        # Convert to covariance
        d = np.sqrt(variances)
        cov = np.outer(d, d) * corr
        
        cov_matrix = CovarianceMatrix(
            matrix=cov,
            energy_grid=EnergyGrid(energy_boundaries),
            is_relative=True,
            reaction_mt=mt,
            material_za=za,
            source=f"FluxForge default ({name})",
        )
        
        library.add_covariance(za, mt, cov_matrix)
    
    return library
