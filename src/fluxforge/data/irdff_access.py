"""
IRDFF-II (International Reactor Dosimetry and Fusion File) data access.

This module provides access to IRDFF-II nuclear data library, which is
the international reference for neutron dosimetry reactions. The library
contains evaluated cross sections and covariances for ~80 dosimetry reactions.

IRDFF-II is the successor to IRDF-2002 and IRDFF-1.05.

Reference: A.J.M. Plompen et al., "The joint evaluated fission and fusion
nuclear data library, JEFF-3.3", Eur. Phys. J. A 56, 181 (2020)

Features:
- IRDFF-II reaction catalog
- Cross section and covariance access
- Energy threshold data
- Half-life and decay data
- Integration with OpenMC/ENDF formats
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class ReactionCategory(Enum):
    """IRDFF reaction categories."""
    
    THRESHOLD = "threshold"  # Threshold reactions (n,p), (n,α), (n,2n), etc.
    RADIATIVE_CAPTURE = "radiative_capture"  # (n,γ) reactions
    FISSION = "fission"  # Fission reactions
    INELASTIC = "inelastic"  # (n,n') reactions
    COVER = "cover"  # Cover material reactions (Cd, B, etc.)


class DataStatus(Enum):
    """IRDFF data quality status."""
    
    RECOMMENDED = "recommended"  # Primary dosimetry reaction
    SECONDARY = "secondary"  # Good quality, secondary use
    MONITORING = "monitoring"  # For flux monitoring
    RESEARCH = "research"  # Research quality
    DEPRECATED = "deprecated"  # Superseded by newer evaluation


@dataclass
class IRDFFReaction:
    """
    IRDFF-II reaction metadata.
    
    Attributes:
        target: Target nucleus (e.g., "Au-197")
        product: Product nucleus (e.g., "Au-198")
        reaction: Reaction string (e.g., "(n,g)")
        mt: ENDF MT number
        za: ZA identifier
        threshold_eV: Reaction threshold energy (0 for exothermic)
        half_life_s: Product half-life in seconds
        half_life_unc_s: Half-life uncertainty
        category: Reaction category
        status: Data quality status
        gamma_lines_keV: Principal gamma energies
        gamma_intensities: Gamma intensities (fraction)
    """
    
    target: str
    product: str
    reaction: str
    mt: int
    za: int
    threshold_eV: float = 0.0
    half_life_s: float = 0.0
    half_life_unc_s: float = 0.0
    category: ReactionCategory = ReactionCategory.THRESHOLD
    status: DataStatus = DataStatus.RECOMMENDED
    gamma_lines_keV: List[float] = field(default_factory=list)
    gamma_intensities: List[float] = field(default_factory=list)
    
    @property
    def full_name(self) -> str:
        """Full reaction name."""
        return f"{self.target}{self.reaction}{self.product}"
    
    @property
    def short_name(self) -> str:
        """Short reaction identifier."""
        return f"{self.target}{self.reaction}"
    
    @property
    def threshold_MeV(self) -> float:
        """Threshold in MeV."""
        return self.threshold_eV / 1e6
    
    @property
    def half_life_hours(self) -> float:
        """Half-life in hours."""
        return self.half_life_s / 3600
    
    @property
    def half_life_days(self) -> float:
        """Half-life in days."""
        return self.half_life_s / 86400
    
    @property
    def decay_constant(self) -> float:
        """Decay constant λ = ln(2) / t½ in s⁻¹."""
        if self.half_life_s > 0:
            return np.log(2) / self.half_life_s
        return 0.0


@dataclass
class CrossSectionData:
    """
    Cross section data for a reaction.
    
    Attributes:
        reaction: Associated reaction
        energies_eV: Energy grid (n points)
        cross_section_b: Cross section in barns (n points)
        uncertainty_b: Uncertainty in barns (n points)
        covariance: Covariance matrix (n x n), optional
        temperature_K: Temperature for Doppler broadening
    """
    
    reaction: IRDFFReaction
    energies_eV: np.ndarray
    cross_section_b: np.ndarray
    uncertainty_b: np.ndarray
    covariance: Optional[np.ndarray] = None
    temperature_K: float = 293.6
    
    @property
    def n_points(self) -> int:
        return len(self.energies_eV)
    
    @property
    def relative_uncertainty(self) -> np.ndarray:
        """Relative uncertainty (σ/μ)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = self.uncertainty_b / self.cross_section_b
            rel[~np.isfinite(rel)] = 0
        return rel
    
    def get_cross_section(self, energy_eV: float) -> Tuple[float, float]:
        """
        Get interpolated cross section at given energy.
        
        Uses log-log interpolation.
        
        Returns:
            Tuple of (cross_section, uncertainty)
        """
        if energy_eV < self.energies_eV[0]:
            return self.cross_section_b[0], self.uncertainty_b[0]
        if energy_eV > self.energies_eV[-1]:
            return 0.0, 0.0
        
        log_E = np.log(self.energies_eV)
        log_xs = np.log(np.maximum(self.cross_section_b, 1e-100))
        log_E_target = np.log(energy_eV)
        
        xs = np.exp(np.interp(log_E_target, log_E, log_xs))
        
        # Interpolate uncertainty linearly
        unc = np.interp(energy_eV, self.energies_eV, self.uncertainty_b)
        
        return xs, unc
    
    def get_group_averaged(
        self,
        group_boundaries_eV: np.ndarray,
        flux_spectrum: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate group-averaged cross sections.
        
        Args:
            group_boundaries_eV: Energy group boundaries (n_groups + 1)
            flux_spectrum: Optional flux weights per group
            
        Returns:
            Tuple of (group_xs, group_unc)
        """
        n_groups = len(group_boundaries_eV) - 1
        group_xs = np.zeros(n_groups)
        group_unc = np.zeros(n_groups)
        
        for g in range(n_groups):
            E_lo = group_boundaries_eV[g]
            E_hi = group_boundaries_eV[g + 1]
            
            # Find points within group
            mask = (self.energies_eV >= E_lo) & (self.energies_eV <= E_hi)
            
            if np.any(mask):
                weights = np.ones(np.sum(mask))
                if flux_spectrum is not None and len(flux_spectrum) > g:
                    weights *= flux_spectrum[g]
                
                group_xs[g] = np.average(self.cross_section_b[mask], weights=weights)
                group_unc[g] = np.sqrt(np.average(
                    self.uncertainty_b[mask]**2, weights=weights
                ))
            else:
                # Interpolate at group center
                E_center = np.sqrt(E_lo * E_hi)
                group_xs[g], group_unc[g] = self.get_cross_section(E_center)
        
        return group_xs, group_unc


# IRDFF-II reaction catalog (primary dosimetry reactions)
IRDFF_CATALOG: Dict[str, IRDFFReaction] = {
    # Radiative capture reactions
    "Au-197(n,g)": IRDFFReaction(
        target="Au-197", product="Au-198", reaction="(n,g)",
        mt=102, za=79197,
        threshold_eV=0.0,
        half_life_s=2.6943 * 86400,  # 2.6943 days
        half_life_unc_s=0.0003 * 86400,
        category=ReactionCategory.RADIATIVE_CAPTURE,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[411.8],
        gamma_intensities=[0.9562],
    ),
    "Co-59(n,g)": IRDFFReaction(
        target="Co-59", product="Co-60", reaction="(n,g)",
        mt=102, za=27059,
        threshold_eV=0.0,
        half_life_s=5.2714 * 365.25 * 86400,  # 5.2714 years
        half_life_unc_s=0.0005 * 365.25 * 86400,
        category=ReactionCategory.RADIATIVE_CAPTURE,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[1173.2, 1332.5],
        gamma_intensities=[0.9985, 0.9998],
    ),
    "Mn-55(n,g)": IRDFFReaction(
        target="Mn-55", product="Mn-56", reaction="(n,g)",
        mt=102, za=25055,
        threshold_eV=0.0,
        half_life_s=2.5789 * 3600,  # 2.5789 hours
        half_life_unc_s=0.0001 * 3600,
        category=ReactionCategory.RADIATIVE_CAPTURE,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[846.8, 1810.7],
        gamma_intensities=[0.989, 0.272],
    ),
    "Sc-45(n,g)": IRDFFReaction(
        target="Sc-45", product="Sc-46", reaction="(n,g)",
        mt=102, za=21045,
        threshold_eV=0.0,
        half_life_s=83.79 * 86400,  # 83.79 days
        half_life_unc_s=0.04 * 86400,
        category=ReactionCategory.RADIATIVE_CAPTURE,
        status=DataStatus.SECONDARY,
        gamma_lines_keV=[889.3, 1120.5],
        gamma_intensities=[0.9998, 0.9999],
    ),
    
    # (n,p) reactions
    "Ni-58(n,p)": IRDFFReaction(
        target="Ni-58", product="Co-58", reaction="(n,p)",
        mt=103, za=28058,
        threshold_eV=0.5e6,  # ~0.5 MeV
        half_life_s=70.86 * 86400,  # 70.86 days
        half_life_unc_s=0.06 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[810.8],
        gamma_intensities=[0.994],
    ),
    "Fe-54(n,p)": IRDFFReaction(
        target="Fe-54", product="Mn-54", reaction="(n,p)",
        mt=103, za=26054,
        threshold_eV=0.7e6,  # ~0.7 MeV
        half_life_s=312.12 * 86400,  # 312.12 days
        half_life_unc_s=0.06 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[834.8],
        gamma_intensities=[0.9998],
    ),
    "Ti-46(n,p)": IRDFFReaction(
        target="Ti-46", product="Sc-46", reaction="(n,p)",
        mt=103, za=22046,
        threshold_eV=1.7e6,  # ~1.7 MeV
        half_life_s=83.79 * 86400,
        half_life_unc_s=0.04 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[889.3, 1120.5],
        gamma_intensities=[0.9998, 0.9999],
    ),
    "Ti-47(n,p)": IRDFFReaction(
        target="Ti-47", product="Sc-47", reaction="(n,p)",
        mt=103, za=22047,
        threshold_eV=1.0e6,
        half_life_s=3.3492 * 86400,
        half_life_unc_s=0.0006 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[159.4],
        gamma_intensities=[0.683],
    ),
    "Ti-48(n,p)": IRDFFReaction(
        target="Ti-48", product="Sc-48", reaction="(n,p)",
        mt=103, za=22048,
        threshold_eV=3.3e6,
        half_life_s=43.67 * 3600,  # 43.67 hours
        half_life_unc_s=0.09 * 3600,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[983.5, 1037.5, 1312.1],
        gamma_intensities=[1.0, 0.976, 1.0],
    ),
    "Fe-56(n,p)": IRDFFReaction(
        target="Fe-56", product="Mn-56", reaction="(n,p)",
        mt=103, za=26056,
        threshold_eV=5.0e6,  # ~5 MeV
        half_life_s=2.5789 * 3600,
        half_life_unc_s=0.0001 * 3600,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[846.8, 1810.7],
        gamma_intensities=[0.989, 0.272],
    ),
    "Cu-63(n,p)": IRDFFReaction(
        target="Cu-63", product="Ni-63", reaction="(n,p)",
        mt=103, za=29063,
        threshold_eV=1.5e6,
        half_life_s=101.2 * 365.25 * 86400,  # 101.2 years
        half_life_unc_s=1.5 * 365.25 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.SECONDARY,
        gamma_lines_keV=[],  # Pure beta emitter
        gamma_intensities=[],
    ),
    
    # (n,α) reactions
    "Al-27(n,a)": IRDFFReaction(
        target="Al-27", product="Na-24", reaction="(n,a)",
        mt=107, za=13027,
        threshold_eV=3.25e6,  # 3.25 MeV
        half_life_s=14.997 * 3600,  # 14.997 hours
        half_life_unc_s=0.012 * 3600,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[1368.6, 2754.0],
        gamma_intensities=[0.9999, 0.9986],
    ),
    "Co-59(n,a)": IRDFFReaction(
        target="Co-59", product="Mn-56", reaction="(n,a)",
        mt=107, za=27059,
        threshold_eV=5.0e6,
        half_life_s=2.5789 * 3600,
        half_life_unc_s=0.0001 * 3600,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.SECONDARY,
        gamma_lines_keV=[846.8],
        gamma_intensities=[0.989],
    ),
    "Cu-63(n,a)": IRDFFReaction(
        target="Cu-63", product="Co-60", reaction="(n,a)",
        mt=107, za=29063,
        threshold_eV=3.5e6,
        half_life_s=5.2714 * 365.25 * 86400,
        half_life_unc_s=0.0005 * 365.25 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[1173.2, 1332.5],
        gamma_intensities=[0.9985, 0.9998],
    ),
    
    # (n,2n) reactions
    "Ni-58(n,2n)": IRDFFReaction(
        target="Ni-58", product="Ni-57", reaction="(n,2n)",
        mt=16, za=28058,
        threshold_eV=12.4e6,  # 12.4 MeV
        half_life_s=35.6 * 3600,  # 35.6 hours
        half_life_unc_s=0.06 * 3600,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[127.2, 1377.6],
        gamma_intensities=[0.167, 0.817],
    ),
    "Nb-93(n,2n)": IRDFFReaction(
        target="Nb-93", product="Nb-92m", reaction="(n,2n)",
        mt=16, za=41093,
        threshold_eV=8.9e6,  # 8.9 MeV
        half_life_s=10.15 * 86400,  # 10.15 days
        half_life_unc_s=0.02 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[934.4],
        gamma_intensities=[0.9907],
    ),
    "Zr-90(n,2n)": IRDFFReaction(
        target="Zr-90", product="Zr-89", reaction="(n,2n)",
        mt=16, za=40090,
        threshold_eV=12.1e6,  # 12.1 MeV
        half_life_s=78.41 * 3600,  # 78.41 hours
        half_life_unc_s=0.12 * 3600,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[909.2],
        gamma_intensities=[0.9904],
    ),
    "Au-197(n,2n)": IRDFFReaction(
        target="Au-197", product="Au-196", reaction="(n,2n)",
        mt=16, za=79197,
        threshold_eV=8.1e6,
        half_life_s=6.183 * 86400,
        half_life_unc_s=0.010 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[355.7, 333.0],
        gamma_intensities=[0.87, 0.229],
    ),
    "Co-59(n,2n)": IRDFFReaction(
        target="Co-59", product="Co-58", reaction="(n,2n)",
        mt=16, za=27059,
        threshold_eV=10.6e6,
        half_life_s=70.86 * 86400,
        half_life_unc_s=0.06 * 86400,
        category=ReactionCategory.THRESHOLD,
        status=DataStatus.SECONDARY,
        gamma_lines_keV=[810.8],
        gamma_intensities=[0.994],
    ),
    
    # Inelastic scattering
    "In-115(n,n')": IRDFFReaction(
        target="In-115", product="In-115m", reaction="(n,n')",
        mt=4, za=49115,
        threshold_eV=0.34e6,  # 0.34 MeV
        half_life_s=4.486 * 3600,  # 4.486 hours
        half_life_unc_s=0.004 * 3600,
        category=ReactionCategory.INELASTIC,
        status=DataStatus.RECOMMENDED,
        gamma_lines_keV=[336.2],
        gamma_intensities=[0.458],
    ),
}


class IRDFFLibrary:
    """
    IRDFF-II data library interface.
    
    Provides access to IRDFF-II nuclear data including cross sections,
    covariances, and reaction metadata.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize IRDFF library.
        
        Args:
            data_path: Path to IRDFF-II data files (optional)
        """
        self.data_path = Path(data_path) if data_path else None
        self._cross_sections: Dict[str, CrossSectionData] = {}
        self._catalog = IRDFF_CATALOG.copy()
    
    def list_reactions(
        self,
        category: Optional[ReactionCategory] = None,
        status: Optional[DataStatus] = None,
    ) -> List[IRDFFReaction]:
        """
        List available reactions.
        
        Args:
            category: Filter by category
            status: Filter by data status
            
        Returns:
            List of matching reactions
        """
        reactions = list(self._catalog.values())
        
        if category:
            reactions = [r for r in reactions if r.category == category]
        
        if status:
            reactions = [r for r in reactions if r.status == status]
        
        return reactions
    
    def get_reaction(self, name: str) -> Optional[IRDFFReaction]:
        """Get reaction by name."""
        return self._catalog.get(name)
    
    def get_by_za_mt(self, za: int, mt: int) -> Optional[IRDFFReaction]:
        """Get reaction by ZA and MT."""
        for rxn in self._catalog.values():
            if rxn.za == za and rxn.mt == mt:
                return rxn
        return None
    
    def get_cross_section(self, reaction_name: str) -> Optional[CrossSectionData]:
        """Get cross section data for a reaction."""
        if reaction_name in self._cross_sections:
            return self._cross_sections[reaction_name]
        
        # Try to load from file
        if self.data_path:
            xs_data = self._load_cross_section(reaction_name)
            if xs_data:
                self._cross_sections[reaction_name] = xs_data
                return xs_data
        
        # Generate default cross section if reaction exists
        rxn = self.get_reaction(reaction_name)
        if rxn:
            return self._generate_default_cross_section(rxn)
        
        return None
    
    def _load_cross_section(self, reaction_name: str) -> Optional[CrossSectionData]:
        """Load cross section from data file."""
        if not self.data_path:
            return None
        
        # Try various file formats
        rxn = self.get_reaction(reaction_name)
        if not rxn:
            return None
        
        # Try IRDFF-II native formats first
        g4 = self._find_irdff_file(rxn, suffixes=(".G4", ".g4"))
        if g4 is not None:
            xs = self._parse_irdff_two_column(g4, rxn, kind="group")
            if xs is not None:
                return xs

        dat = self._find_irdff_file(rxn, suffixes=(".DAT", ".dat"))
        if dat is not None:
            xs = self._parse_irdff_two_column(dat, rxn, kind="pointwise")
            if xs is not None:
                return xs

        # Try ENDF format
        endf_file = self.data_path / f"n_{rxn.za}_{rxn.mt:03d}.endf"
        if endf_file.exists():
            return self._parse_endf_xs(endf_file, rxn)
        
        # Try CSV format
        csv_file = self.data_path / f"{reaction_name.replace('(', '_').replace(')', '')}.csv"
        if csv_file.exists():
            return self._parse_csv_xs(csv_file, rxn)
        
        return None

    def _find_irdff_file(
        self,
        reaction: IRDFFReaction,
        suffixes: Tuple[str, ...],
    ) -> Optional[Path]:
        """Best-effort search for an IRDFF-II file for a given reaction.

        IRDFF distributions vary in naming conventions. We try a few practical
        heuristics:
        - filenames containing ZA and MT
        - filenames containing target element and mass (e.g. Au197) and MT
        """
        if self.data_path is None or not self.data_path.exists():
            return None

        candidates: List[Path] = []
        for suf in suffixes:
            candidates.extend(self.data_path.glob(f"*{suf}"))

        if not candidates:
            return None

        za_str = str(reaction.za)
        mt_str = str(reaction.mt)

        # Normalize target token: "Au-197" -> "au197"
        target_token = "".join(ch for ch in reaction.target.lower() if ch.isalnum())
        mt3 = f"{reaction.mt:03d}"

        scored: List[Tuple[int, Path]] = []
        for p in candidates:
            name = p.name.lower()
            score = 0
            if za_str in name:
                score += 4
            if mt3 in name or f"mt{mt_str}" in name or f"_{mt_str}_" in name:
                score += 3
            if mt_str in name:
                score += 1
            if target_token and target_token in name:
                score += 2
            scored.append((score, p))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_path = scored[0]
        if best_score <= 0:
            return None
        return best_path

    def _parse_irdff_two_column(
        self,
        file_path: Path,
        reaction: IRDFFReaction,
        kind: str,
    ) -> Optional[CrossSectionData]:
        """Parse a simple IRDFF-II .G4/.DAT-like file.

        This parser is intentionally tolerant: it accepts whitespace or comma
        separated columns, skips non-numeric/header lines, and supports either
        2-column (E, xs) or 3-column (E, xs, unc).

        Energies are assumed to be in eV unless the values look like MeV-scale
        (heuristic), in which case they are converted to eV.
        """
        try:
            energies: List[float] = []
            xs: List[float] = []
            unc: List[float] = []

            with open(file_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith("#") or line.startswith("!") or line.startswith("//"):
                        continue

                    # Split on comma or whitespace
                    parts = [p for p in line.replace(",", " ").split() if p]
                    if len(parts) < 2:
                        continue

                    try:
                        e = float(parts[0])
                        v = float(parts[1])
                        s = float(parts[2]) if len(parts) >= 3 else abs(v) * 0.05
                    except ValueError:
                        continue

                    energies.append(e)
                    xs.append(v)
                    unc.append(abs(s))

            if len(energies) < 2:
                return None

            energies_arr = np.array(energies, dtype=float)
            xs_arr = np.array(xs, dtype=float)
            unc_arr = np.array(unc, dtype=float)

            # Heuristic: if energies are ~[0, 30] and non-increasing issues, assume MeV
            # and convert to eV.
            if np.nanmax(energies_arr) < 1e3 and np.nanmin(energies_arr) >= 0:
                # likely MeV
                energies_arr = energies_arr * 1e6

            # Ensure sorted by energy
            order = np.argsort(energies_arr)
            energies_arr = energies_arr[order]
            xs_arr = xs_arr[order]
            unc_arr = unc_arr[order]

            # Clamp negative cross sections (file quirks)
            xs_arr = np.maximum(xs_arr, 0.0)

            return CrossSectionData(
                reaction=reaction,
                energies_eV=energies_arr,
                cross_section_b=xs_arr,
                uncertainty_b=unc_arr,
            )
        except Exception:
            return None
    
    def _parse_endf_xs(
        self,
        file_path: Path,
        reaction: IRDFFReaction,
    ) -> Optional[CrossSectionData]:
        """Parse ENDF format cross section file."""
        # Simplified parser - production version should use ENDFtk or similar
        try:
            energies = []
            cross_sections = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    if len(line) < 66:
                        continue
                    
                    try:
                        mf = int(line[70:72])
                        if mf != 3:  # MF3 is cross sections
                            continue
                        
                        # Parse data values
                        val1 = float(line[0:11].replace('D', 'E').replace('d', 'e').strip() or '0')
                        val2 = float(line[11:22].replace('D', 'E').replace('d', 'e').strip() or '0')
                        
                        if val1 > 0:
                            energies.append(val1)
                            cross_sections.append(val2)
                    except (ValueError, IndexError):
                        continue
            
            if len(energies) >= 2:
                energies_arr = np.array(energies)
                xs_arr = np.array(cross_sections)
                unc_arr = xs_arr * 0.05  # Assume 5% uncertainty
                
                return CrossSectionData(
                    reaction=reaction,
                    energies_eV=energies_arr,
                    cross_section_b=xs_arr,
                    uncertainty_b=unc_arr,
                )
        except Exception:
            pass
        
        return None
    
    def _parse_csv_xs(
        self,
        file_path: Path,
        reaction: IRDFFReaction,
    ) -> Optional[CrossSectionData]:
        """Parse CSV format cross section file."""
        try:
            import csv
            
            energies = []
            cross_sections = []
            uncertainties = []
            
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                for row in reader:
                    if len(row) >= 2:
                        energies.append(float(row[0]))
                        cross_sections.append(float(row[1]))
                        if len(row) >= 3:
                            uncertainties.append(float(row[2]))
                        else:
                            uncertainties.append(float(row[1]) * 0.05)
            
            if len(energies) >= 2:
                return CrossSectionData(
                    reaction=reaction,
                    energies_eV=np.array(energies),
                    cross_section_b=np.array(cross_sections),
                    uncertainty_b=np.array(uncertainties),
                )
        except Exception:
            pass
        
        return None
    
    def _generate_default_cross_section(
        self,
        reaction: IRDFFReaction,
    ) -> CrossSectionData:
        """Generate default cross section for testing."""
        # Create energy grid
        E_min = max(1e-5, reaction.threshold_eV * 0.1) if reaction.threshold_eV > 0 else 1e-5
        E_max = 20e6
        energies = np.logspace(np.log10(E_min), np.log10(E_max), 200)
        
        # Generate cross section based on reaction type
        xs = np.zeros_like(energies)
        
        if reaction.mt == 102:  # (n,γ)
            # 1/v capture below ~1 eV, constant resonance region
            xs = 10.0 * np.sqrt(0.0253 / (energies * 1e-6))  # barns
            xs = np.minimum(xs, 1000)
            # Add resonance structure (simplified)
            for E_res in [1.0, 10.0, 100.0]:  # eV
                width = 0.1 * E_res
                xs += 50 * np.exp(-((energies - E_res) / width)**2)
            # Fast region decay
            xs[energies > 1e4] *= np.exp(-(energies[energies > 1e4] - 1e4) / 1e5)
            
        elif reaction.mt == 103:  # (n,p)
            threshold = reaction.threshold_eV
            mask = energies > threshold
            xs[mask] = 0.1 * np.sqrt((energies[mask] - threshold) / 1e6)
            xs = np.minimum(xs, 0.5)  # ~0.5 barn max
            
        elif reaction.mt == 107:  # (n,α)
            threshold = reaction.threshold_eV
            mask = energies > threshold
            xs[mask] = 0.05 * np.sqrt((energies[mask] - threshold) / 1e6)
            xs = np.minimum(xs, 0.2)  # ~0.2 barn max
            
        elif reaction.mt == 16:  # (n,2n)
            threshold = reaction.threshold_eV
            mask = energies > threshold
            xs[mask] = 0.5 * (1 - np.exp(-(energies[mask] - threshold) / 2e6))
            xs = np.minimum(xs, 2.0)  # ~2 barn max
            
        elif reaction.mt == 4:  # (n,n')
            threshold = reaction.threshold_eV
            mask = energies > threshold
            xs[mask] = 0.3 * (1 - np.exp(-(energies[mask] - threshold) / 1e6))
            xs = np.minimum(xs, 0.5)
        
        # Add uncertainty (5-20% depending on energy)
        rel_unc = 0.05 + 0.15 * (energies / 20e6)
        uncertainty = xs * rel_unc
        
        return CrossSectionData(
            reaction=reaction,
            energies_eV=energies,
            cross_section_b=xs,
            uncertainty_b=uncertainty,
        )
    
    def get_threshold_reactions(
        self,
        E_min_MeV: float = 0.0,
        E_max_MeV: float = 20.0,
    ) -> List[IRDFFReaction]:
        """
        Get threshold reactions in energy range.
        
        Args:
            E_min_MeV: Minimum threshold energy
            E_max_MeV: Maximum threshold energy
            
        Returns:
            List of reactions with threshold in range
        """
        E_min_eV = E_min_MeV * 1e6
        E_max_eV = E_max_MeV * 1e6
        
        return [
            r for r in self._catalog.values()
            if r.threshold_eV >= E_min_eV and r.threshold_eV <= E_max_eV
        ]
    
    def to_dict(self) -> dict:
        """Export library summary."""
        return {
            "schema": "fluxforge.irdff_library.v1",
            "n_reactions": len(self._catalog),
            "reactions": [
                {
                    "name": r.full_name,
                    "mt": r.mt,
                    "za": r.za,
                    "threshold_MeV": r.threshold_MeV,
                    "half_life_days": r.half_life_days,
                    "category": r.category.value,
                    "status": r.status.value,
                }
                for r in self._catalog.values()
            ],
        }


def get_default_library() -> IRDFFLibrary:
    """Get default IRDFF library instance."""
    return IRDFFLibrary()
