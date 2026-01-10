"""
STAYSL-class output and reporting module.

Provides formatted output tables matching STAYSL PNNL report formats:
- Dosimetry input correlation matrix
- Input vs output flux correlations
- Differential flux tables (unadjusted and adjusted)
- Spectral-averaged reaction rates
- Plot-ready stepwise spectrum data

Reference: PNNL-22253, STAYSL PNNL User Guide
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class FluxTableEntry:
    """
    Single entry in a differential flux table.
    
    Attributes:
        group: Energy group index (1-based)
        energy_low_eV: Lower energy bound
        energy_high_eV: Upper energy bound
        energy_mid_eV: Geometric mean energy
        lethargy_width: Lethargy width of group
        flux: Group flux value
        flux_uncertainty: Absolute uncertainty in flux
        flux_rel_unc: Relative uncertainty (fraction)
    """
    
    group: int
    energy_low_eV: float
    energy_high_eV: float
    energy_mid_eV: float
    lethargy_width: float
    flux: float
    flux_uncertainty: float
    flux_rel_unc: float


@dataclass 
class DifferentialFluxTable:
    """
    Complete differential flux table for reporting.
    
    Attributes:
        entries: List of flux table entries per group
        total_fluence: Integrated fluence (n/cm²/s or n/cm²)
        energy_weighted_avg: Energy-weighted average energy
        label: Description label (e.g., "Prior", "Adjusted")
    """
    
    entries: List[FluxTableEntry]
    total_fluence: float = 0.0
    energy_weighted_avg: float = 0.0
    label: str = ""
    
    @classmethod
    def from_spectrum(
        cls,
        flux: np.ndarray,
        energy_bounds_eV: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        label: str = "",
    ) -> "DifferentialFluxTable":
        """
        Create flux table from spectrum arrays.
        
        Args:
            flux: Group flux values (n-1 groups for n boundaries)
            energy_bounds_eV: Energy boundaries in eV (ascending)
            uncertainty: Optional uncertainty array
            label: Table label
            
        Returns:
            DifferentialFluxTable instance
        """
        n_groups = len(flux)
        if len(energy_bounds_eV) != n_groups + 1:
            raise ValueError(
                f"Energy bounds ({len(energy_bounds_eV)}) must have "
                f"one more element than flux ({n_groups})"
            )
        
        if uncertainty is None:
            uncertainty = np.zeros_like(flux)
        
        entries = []
        total_fluence = 0.0
        energy_sum = 0.0
        
        for g in range(n_groups):
            E_lo = energy_bounds_eV[g]
            E_hi = energy_bounds_eV[g + 1]
            E_mid = np.sqrt(E_lo * E_hi)  # Geometric mean
            lethargy = np.log(E_hi / E_lo) if E_lo > 0 else 0.0
            
            phi = flux[g]
            unc = uncertainty[g]
            rel_unc = unc / phi if phi > 0 else 0.0
            
            entries.append(FluxTableEntry(
                group=g + 1,
                energy_low_eV=E_lo,
                energy_high_eV=E_hi,
                energy_mid_eV=E_mid,
                lethargy_width=lethargy,
                flux=phi,
                flux_uncertainty=unc,
                flux_rel_unc=rel_unc,
            ))
            
            total_fluence += phi * lethargy
            energy_sum += phi * lethargy * E_mid
        
        if total_fluence > 0:
            energy_weighted_avg = energy_sum / total_fluence
        else:
            energy_weighted_avg = 0.0
        
        return cls(
            entries=entries,
            total_fluence=total_fluence,
            energy_weighted_avg=energy_weighted_avg,
            label=label,
        )
    
    def to_text(self, fmt: str = "staysl") -> str:
        """
        Format table as text string.
        
        Args:
            fmt: Output format ("staysl", "csv", "markdown")
            
        Returns:
            Formatted table string
        """
        if fmt == "csv":
            return self._to_csv()
        elif fmt == "markdown":
            return self._to_markdown()
        else:
            return self._to_staysl()
    
    def _to_staysl(self) -> str:
        """STAYSL-style fixed-width format."""
        lines = []
        lines.append(f"  DIFFERENTIAL FLUX TABLE: {self.label}")
        lines.append("=" * 80)
        lines.append(
            f"{'Grp':>4}  {'E_low (eV)':>12}  {'E_high (eV)':>12}  "
            f"{'Flux':>12}  {'Unc':>10}  {'Rel%':>6}"
        )
        lines.append("-" * 80)
        
        for e in self.entries:
            lines.append(
                f"{e.group:>4}  {e.energy_low_eV:>12.4E}  {e.energy_high_eV:>12.4E}  "
                f"{e.flux:>12.4E}  {e.flux_uncertainty:>10.3E}  {e.flux_rel_unc*100:>6.1f}"
            )
        
        lines.append("-" * 80)
        lines.append(f"Total fluence: {self.total_fluence:.4E}")
        lines.append(f"Energy-weighted avg: {self.energy_weighted_avg:.4E} eV")
        
        return "\n".join(lines)
    
    def _to_csv(self) -> str:
        """CSV format."""
        lines = ["group,E_low_eV,E_high_eV,E_mid_eV,lethargy,flux,uncertainty,rel_unc"]
        for e in self.entries:
            lines.append(
                f"{e.group},{e.energy_low_eV:.6E},{e.energy_high_eV:.6E},"
                f"{e.energy_mid_eV:.6E},{e.lethargy_width:.6f},{e.flux:.6E},"
                f"{e.flux_uncertainty:.6E},{e.flux_rel_unc:.6f}"
            )
        return "\n".join(lines)
    
    def _to_markdown(self) -> str:
        """Markdown table format."""
        lines = [f"## {self.label}", ""]
        lines.append("| Grp | E_low (eV) | E_high (eV) | Flux | Unc | Rel% |")
        lines.append("|-----|------------|-------------|------|-----|------|")
        
        for e in self.entries:
            lines.append(
                f"| {e.group} | {e.energy_low_eV:.3E} | {e.energy_high_eV:.3E} | "
                f"{e.flux:.3E} | {e.flux_uncertainty:.2E} | {e.flux_rel_unc*100:.1f} |"
            )
        
        lines.append("")
        lines.append(f"**Total fluence:** {self.total_fluence:.4E}")
        
        return "\n".join(lines)


@dataclass
class SpectralReactionRate:
    """
    Spectral-averaged reaction rate for a single reaction.
    
    Attributes:
        reaction_id: Reaction identifier
        rate: Spectral-averaged reaction rate
        rate_uncertainty: Uncertainty in rate
        cross_section_avg: Spectrum-averaged cross section (barns)
        sensitivity: Sensitivity to flux (optional)
    """
    
    reaction_id: str
    rate: float
    rate_uncertainty: float
    cross_section_avg: float = 0.0
    sensitivity: Optional[np.ndarray] = None


@dataclass
class ReactionRateTable:
    """
    Table of spectral-averaged reaction rates.
    
    Attributes:
        rates: List of reaction rate entries
        flux_label: Label of flux used (e.g., "Adjusted")
    """
    
    rates: List[SpectralReactionRate]
    flux_label: str = ""
    
    @classmethod
    def from_cross_sections(
        cls,
        flux: np.ndarray,
        flux_uncertainty: np.ndarray,
        cross_sections: Dict[str, np.ndarray],
        energy_bounds_eV: np.ndarray,
        flux_label: str = "",
    ) -> "ReactionRateTable":
        """
        Calculate spectral-averaged reaction rates.
        
        Args:
            flux: Group flux values
            flux_uncertainty: Flux uncertainties
            cross_sections: Dict mapping reaction_id to cross section arrays
            energy_bounds_eV: Energy boundaries
            flux_label: Label for the flux used
            
        Returns:
            ReactionRateTable with computed rates
        """
        rates = []
        
        # Compute lethargy widths
        n_groups = len(flux)
        lethargy = np.zeros(n_groups)
        for g in range(n_groups):
            if energy_bounds_eV[g] > 0:
                lethargy[g] = np.log(energy_bounds_eV[g+1] / energy_bounds_eV[g])
        
        total_fluence = np.sum(flux * lethargy)
        
        for rxn_id, sigma in cross_sections.items():
            if len(sigma) != n_groups:
                continue
            
            # Reaction rate = Σ_g (σ_g × φ_g × Δu_g)
            rate = np.sum(sigma * flux * lethargy)
            
            # Spectrum-averaged cross section
            if total_fluence > 0:
                sigma_avg = rate / total_fluence
            else:
                sigma_avg = 0.0
            
            # Propagate uncertainty (simplified diagonal)
            rate_var = np.sum((sigma * lethargy)**2 * flux_uncertainty**2)
            rate_unc = np.sqrt(rate_var)
            
            rates.append(SpectralReactionRate(
                reaction_id=rxn_id,
                rate=rate,
                rate_uncertainty=rate_unc,
                cross_section_avg=sigma_avg,
            ))
        
        return cls(rates=rates, flux_label=flux_label)
    
    def to_text(self, fmt: str = "staysl") -> str:
        """Format table as text string."""
        if fmt == "csv":
            return self._to_csv()
        elif fmt == "markdown":
            return self._to_markdown()
        else:
            return self._to_staysl()
    
    def _to_staysl(self) -> str:
        """STAYSL-style format."""
        lines = []
        lines.append(f"  SPECTRAL-AVERAGED REACTION RATES: {self.flux_label}")
        lines.append("=" * 70)
        lines.append(
            f"{'Reaction':>24}  {'Rate':>12}  {'Unc':>10}  {'σ_avg (b)':>10}"
        )
        lines.append("-" * 70)
        
        for r in self.rates:
            lines.append(
                f"{r.reaction_id:>24}  {r.rate:>12.4E}  "
                f"{r.rate_uncertainty:>10.3E}  {r.cross_section_avg:>10.4E}"
            )
        
        return "\n".join(lines)
    
    def _to_csv(self) -> str:
        """CSV format."""
        lines = ["reaction_id,rate,uncertainty,sigma_avg_barns"]
        for r in self.rates:
            lines.append(
                f"{r.reaction_id},{r.rate:.6E},{r.rate_uncertainty:.6E},"
                f"{r.cross_section_avg:.6E}"
            )
        return "\n".join(lines)
    
    def _to_markdown(self) -> str:
        """Markdown format."""
        lines = [f"## Spectral-Averaged Rates: {self.flux_label}", ""]
        lines.append("| Reaction | Rate | Unc | σ_avg (b) |")
        lines.append("|----------|------|-----|-----------|")
        
        for r in self.rates:
            lines.append(
                f"| {r.reaction_id} | {r.rate:.3E} | {r.rate_uncertainty:.2E} | "
                f"{r.cross_section_avg:.3E} |"
            )
        
        return "\n".join(lines)


@dataclass
class CorrelationMatrix:
    """
    Correlation matrix with labels.
    
    Attributes:
        matrix: 2D correlation matrix
        labels: Row/column labels
        matrix_type: Type description (e.g., "dosimetry input", "flux output")
    """
    
    matrix: np.ndarray
    labels: List[str]
    matrix_type: str = ""
    
    @classmethod
    def from_covariance(
        cls,
        covariance: np.ndarray,
        labels: List[str],
        matrix_type: str = "",
    ) -> "CorrelationMatrix":
        """
        Create correlation matrix from covariance matrix.
        
        Args:
            covariance: Covariance matrix
            labels: Row/column labels
            matrix_type: Description string
            
        Returns:
            CorrelationMatrix instance
        """
        n = covariance.shape[0]
        if len(labels) != n:
            raise ValueError("Labels must match matrix dimension")
        
        # Convert covariance to correlation
        diag = np.sqrt(np.diag(covariance))
        diag = np.where(diag > 0, diag, 1.0)  # Avoid division by zero
        
        correlation = covariance / np.outer(diag, diag)
        
        return cls(matrix=correlation, labels=labels, matrix_type=matrix_type)
    
    def to_text(self, fmt: str = "staysl", max_cols: int = 10) -> str:
        """Format matrix as text string."""
        if fmt == "csv":
            return self._to_csv()
        else:
            return self._to_staysl(max_cols)
    
    def _to_staysl(self, max_cols: int = 10) -> str:
        """STAYSL-style lower-triangular format."""
        lines = []
        lines.append(f"  CORRELATION MATRIX: {self.matrix_type}")
        lines.append("=" * 60)
        
        n = self.matrix.shape[0]
        
        # Print in blocks if matrix is large
        for block_start in range(0, n, max_cols):
            block_end = min(block_start + max_cols, n)
            
            # Header row
            header = "        " + "  ".join(
                f"{self.labels[j][:6]:>6}" for j in range(block_start, block_end)
            )
            lines.append(header)
            lines.append("-" * len(header))
            
            # Lower triangular portion
            for i in range(block_start, n):
                row_end = min(i + 1, block_end)
                if row_end <= block_start:
                    continue
                    
                row = f"{self.labels[i][:6]:>6}  "
                row += "  ".join(
                    f"{self.matrix[i, j]:>6.3f}" 
                    for j in range(block_start, row_end)
                )
                lines.append(row)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _to_csv(self) -> str:
        """Full matrix CSV format."""
        lines = ["," + ",".join(self.labels)]
        for i, label in enumerate(self.labels):
            row = label + "," + ",".join(f"{v:.6f}" for v in self.matrix[i, :])
            lines.append(row)
        return "\n".join(lines)


@dataclass
class StepwiseSpectrum:
    """
    Plot-ready stepwise spectrum data.
    
    Attributes:
        energy_eV: Energy points (duplicated for step plot)
        flux: Flux values (duplicated for step plot)
        label: Spectrum label
    """
    
    energy_eV: np.ndarray
    flux: np.ndarray
    label: str = ""
    
    @classmethod
    def from_histogram(
        cls,
        flux: np.ndarray,
        energy_bounds_eV: np.ndarray,
        label: str = "",
    ) -> "StepwiseSpectrum":
        """
        Create stepwise spectrum from histogram data.
        
        The output arrays are formatted for step plotting:
        energy_eV = [E0, E1, E1, E2, E2, ...]
        flux = [φ0, φ0, φ1, φ1, φ2, ...]
        
        Args:
            flux: Group flux values
            energy_bounds_eV: Energy boundaries
            label: Spectrum label
            
        Returns:
            StepwiseSpectrum instance
        """
        n = len(flux)
        
        # Create step-plot arrays
        step_energy = np.zeros(2 * n)
        step_flux = np.zeros(2 * n)
        
        for i in range(n):
            step_energy[2*i] = energy_bounds_eV[i]
            step_energy[2*i + 1] = energy_bounds_eV[i + 1]
            step_flux[2*i] = flux[i]
            step_flux[2*i + 1] = flux[i]
        
        return cls(energy_eV=step_energy, flux=step_flux, label=label)
    
    def to_csv(self) -> str:
        """Export as CSV for external plotting."""
        lines = ["energy_eV,flux"]
        for e, f in zip(self.energy_eV, self.flux):
            lines.append(f"{e:.6E},{f:.6E}")
        return "\n".join(lines)


@dataclass
class UnfoldingReport:
    """
    Complete unfolding report with STAYSL-class outputs.
    
    Attributes:
        prior_flux_table: Prior (input) flux table
        adjusted_flux_table: Adjusted (output) flux table
        input_correlation: Dosimetry input correlation matrix
        output_correlation: Output flux correlation matrix
        reaction_rates: Spectral-averaged reaction rates
        chi_squared: Chi-squared statistic
        degrees_of_freedom: Degrees of freedom
        timestamp: Report generation timestamp
    """
    
    prior_flux_table: DifferentialFluxTable
    adjusted_flux_table: DifferentialFluxTable
    input_correlation: Optional[CorrelationMatrix] = None
    output_correlation: Optional[CorrelationMatrix] = None
    reaction_rates: Optional[ReactionRateTable] = None
    chi_squared: float = 0.0
    degrees_of_freedom: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def generate_full_report(self, fmt: str = "staysl") -> str:
        """
        Generate complete report text.
        
        Args:
            fmt: Output format ("staysl", "csv", "markdown")
            
        Returns:
            Complete report string
        """
        sections = []
        
        # Header
        if fmt == "markdown":
            sections.append("# FluxForge Spectral Adjustment Report")
            sections.append(f"\nGenerated: {self.timestamp}\n")
        else:
            sections.append("=" * 80)
            sections.append("  FLUXFORGE SPECTRAL ADJUSTMENT REPORT")
            sections.append(f"  Generated: {self.timestamp}")
            sections.append("=" * 80)
        
        # Chi-squared
        if self.degrees_of_freedom > 0:
            reduced_chi2 = self.chi_squared / self.degrees_of_freedom
            sections.append(f"\nChi-squared: {self.chi_squared:.4f}")
            sections.append(f"Degrees of freedom: {self.degrees_of_freedom}")
            sections.append(f"Reduced chi-squared: {reduced_chi2:.4f}\n")
        
        # Prior flux table
        sections.append("\n" + self.prior_flux_table.to_text(fmt))
        
        # Adjusted flux table
        sections.append("\n" + self.adjusted_flux_table.to_text(fmt))
        
        # Input correlation
        if self.input_correlation is not None:
            sections.append("\n" + self.input_correlation.to_text(fmt))
        
        # Output correlation
        if self.output_correlation is not None:
            sections.append("\n" + self.output_correlation.to_text(fmt))
        
        # Reaction rates
        if self.reaction_rates is not None:
            sections.append("\n" + self.reaction_rates.to_text(fmt))
        
        return "\n".join(sections)
    
    def save(self, path: Union[str, Path], fmt: str = "staysl") -> None:
        """Save report to file."""
        Path(path).write_text(self.generate_full_report(fmt))
    
    def get_stepwise_spectra(self) -> Tuple[StepwiseSpectrum, StepwiseSpectrum]:
        """
        Get plot-ready stepwise spectra.
        
        Returns:
            (prior_spectrum, adjusted_spectrum) tuple
        """
        # Extract energy bounds and flux from tables
        prior_entries = self.prior_flux_table.entries
        adj_entries = self.adjusted_flux_table.entries
        
        n = len(prior_entries)
        
        # Build energy bounds array
        energy_bounds = np.zeros(n + 1)
        prior_flux = np.zeros(n)
        adj_flux = np.zeros(n)
        
        for i, (pe, ae) in enumerate(zip(prior_entries, adj_entries)):
            energy_bounds[i] = pe.energy_low_eV
            prior_flux[i] = pe.flux
            adj_flux[i] = ae.flux
        energy_bounds[n] = prior_entries[-1].energy_high_eV
        
        return (
            StepwiseSpectrum.from_histogram(prior_flux, energy_bounds, "Prior"),
            StepwiseSpectrum.from_histogram(adj_flux, energy_bounds, "Adjusted"),
        )


def create_unfolding_report(
    prior_flux: np.ndarray,
    adjusted_flux: np.ndarray,
    energy_bounds_eV: np.ndarray,
    prior_uncertainty: Optional[np.ndarray] = None,
    adjusted_uncertainty: Optional[np.ndarray] = None,
    adjusted_covariance: Optional[np.ndarray] = None,
    dosimetry_covariance: Optional[np.ndarray] = None,
    reaction_labels: Optional[List[str]] = None,
    cross_sections: Optional[Dict[str, np.ndarray]] = None,
    chi_squared: float = 0.0,
    degrees_of_freedom: int = 0,
) -> UnfoldingReport:
    """
    Create a complete unfolding report from adjustment results.
    
    Args:
        prior_flux: Prior flux spectrum
        adjusted_flux: Adjusted flux spectrum
        energy_bounds_eV: Energy group boundaries
        prior_uncertainty: Prior flux uncertainties
        adjusted_uncertainty: Adjusted flux uncertainties
        adjusted_covariance: Full posterior covariance matrix
        dosimetry_covariance: Dosimetry input covariance
        reaction_labels: Labels for dosimetry reactions
        cross_sections: Cross section dict for rate calculations
        chi_squared: Chi-squared statistic
        degrees_of_freedom: Degrees of freedom
        
    Returns:
        UnfoldingReport instance
    """
    # Create flux tables
    prior_table = DifferentialFluxTable.from_spectrum(
        prior_flux, 
        energy_bounds_eV, 
        prior_uncertainty,
        label="Prior Flux",
    )
    
    adjusted_table = DifferentialFluxTable.from_spectrum(
        adjusted_flux,
        energy_bounds_eV,
        adjusted_uncertainty,
        label="Adjusted Flux",
    )
    
    # Create correlation matrices if covariances provided
    input_corr = None
    if dosimetry_covariance is not None and reaction_labels is not None:
        input_corr = CorrelationMatrix.from_covariance(
            dosimetry_covariance,
            reaction_labels,
            matrix_type="Dosimetry Input Correlations",
        )
    
    output_corr = None
    if adjusted_covariance is not None:
        # For large matrices, use group indices as labels
        n_groups = adjusted_covariance.shape[0]
        group_labels = [f"G{i+1}" for i in range(n_groups)]
        output_corr = CorrelationMatrix.from_covariance(
            adjusted_covariance,
            group_labels,
            matrix_type="Output Flux Correlations",
        )
    
    # Calculate reaction rates if cross sections provided
    rates = None
    if cross_sections is not None and adjusted_uncertainty is not None:
        rates = ReactionRateTable.from_cross_sections(
            adjusted_flux,
            adjusted_uncertainty,
            cross_sections,
            energy_bounds_eV,
            flux_label="Adjusted Flux",
        )
    
    return UnfoldingReport(
        prior_flux_table=prior_table,
        adjusted_flux_table=adjusted_table,
        input_correlation=input_corr,
        output_correlation=output_corr,
        reaction_rates=rates,
        chi_squared=chi_squared,
        degrees_of_freedom=degrees_of_freedom,
    )
