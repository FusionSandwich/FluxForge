"""
Spectrum Unfolding Workflow Module

Provides complete workflow for neutron spectrum unfolding using:
- IRDFF-II cross sections from the IAEA database
- GRAVEL/MLEM iterative solvers
- MCNP spectrum as initial guess
- Response matrix construction from reaction cross sections

This module implements the methodology described in:
- Matzke, H., "Unfolding of Particle Spectra", PTB Report PTB-N-19 (1994)
- Reginatto, M., "The 'Few-Channel' Unfolding Programs in the UMG Package",
  PTB Report PTB-N-6 (2003)

References for IRDFF-II:
- A. Trkov et al., Nuclear Data Sheets 163, 1-108 (2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# FluxForge imports
from fluxforge.data.irdff import (
    IRDFFDatabase,
    IRDFFCrossSection,
    get_flux_wire_energy_groups,
    get_activation_energy_groups,
    build_response_matrix,
    IRDFF_REACTIONS,
)
from fluxforge.solvers.iterative import gravel, mlem, IterativeSolution


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FluxWireMeasurement:
    """
    Container for a flux wire measurement result.
    
    Attributes
    ----------
    reaction : str
        Reaction identifier (e.g., 'Ti-46(n,p)Sc-46')
    activity_Bq : float
        Measured activity at EOI in Bq
    uncertainty_Bq : float
        Uncertainty in Bq
    saturation_factor : float
        Saturation correction factor
    decay_factor : float
        Decay correction factor
    irradiation_time : float
        Irradiation time in seconds
    cooling_time : float
        Cooling time in seconds
    sample_mass_g : float
        Sample mass in grams
    isotope_abundance : float
        Isotope abundance (0-1)
    """
    reaction: str
    activity_Bq: float
    uncertainty_Bq: float = 0.0
    saturation_factor: float = 1.0
    decay_factor: float = 1.0
    irradiation_time: float = 0.0
    cooling_time: float = 0.0
    sample_mass_g: float = 1.0
    isotope_abundance: float = 1.0
    
    @property
    def reaction_rate_per_atom(self) -> float:
        """Calculate reaction rate per target atom per second."""
        # R = A / (N * S * D)
        # where A = activity, N = number of atoms, S = saturation factor, D = decay factor
        return self.activity_Bq / (self.saturation_factor * self.decay_factor)
    
    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as fraction."""
        if self.activity_Bq > 0:
            return self.uncertainty_Bq / self.activity_Bq
        return 0.0


@dataclass
class UnfoldingResult:
    """
    Container for spectrum unfolding results.
    
    Attributes
    ----------
    energy_edges : np.ndarray
        Energy group boundaries in eV
    flux : np.ndarray
        Unfolded flux spectrum (per unit energy)
    flux_uncertainty : np.ndarray
        Flux uncertainties
    energy_midpoints : np.ndarray
        Group midpoint energies in eV
    energy_widths : np.ndarray
        Group widths in eV
    reactions_used : List[str]
        Reactions used in unfolding
    response_matrix : np.ndarray
        Response matrix used
    measured_rates : np.ndarray
        Input measured reaction rates
    predicted_rates : np.ndarray
        Predicted rates from unfolded spectrum
    chi_squared : float
        Chi-squared per degree of freedom
    iterations : int
        Number of iterations performed
    converged : bool
        Whether algorithm converged
    method : str
        Unfolding method used ('GRAVEL' or 'MLEM')
    initial_guess_source : str
        Source of initial guess ('uniform', 'MCNP', 'user')
    metadata : Dict[str, Any]
        Additional metadata
    """
    energy_edges: np.ndarray
    flux: np.ndarray
    flux_uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_midpoints: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_widths: np.ndarray = field(default_factory=lambda: np.array([]))
    reactions_used: List[str] = field(default_factory=list)
    response_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    measured_rates: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_rates: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_squared: float = 0.0
    iterations: int = 0
    converged: bool = False
    method: str = "GRAVEL"
    initial_guess_source: str = "uniform"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived quantities."""
        if len(self.energy_midpoints) == 0 and len(self.energy_edges) > 1:
            self.energy_midpoints = np.sqrt(
                self.energy_edges[:-1] * self.energy_edges[1:]
            )
        if len(self.energy_widths) == 0 and len(self.energy_edges) > 1:
            self.energy_widths = self.energy_edges[1:] - self.energy_edges[:-1]
    
    @property
    def n_groups(self) -> int:
        """Number of energy groups."""
        return len(self.flux)
    
    @property
    def integral_flux(self) -> float:
        """Total integral flux."""
        return float(np.sum(self.flux * self.energy_widths))
    
    @property
    def thermal_flux(self, e_max: float = 0.55) -> float:
        """Thermal flux (E < Cd cutoff)."""
        mask = self.energy_midpoints < e_max
        return float(np.sum(self.flux[mask] * self.energy_widths[mask]))
    
    @property
    def fast_flux(self, e_min: float = 1e5) -> float:
        """Fast flux (E > 100 keV)."""
        mask = self.energy_midpoints > e_min
        return float(np.sum(self.flux[mask] * self.energy_widths[mask]))
    
    def get_flux_at_energy(self, energy_eV: float) -> float:
        """Get flux at a specific energy by interpolation."""
        return float(np.interp(energy_eV, self.energy_midpoints, self.flux))
    
    def to_lethargy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to lethargy representation (E * φ(E)).
        
        Returns
        -------
        lethargy : np.ndarray
            Lethargy values (u = ln(E_ref/E))
        flux_per_lethargy : np.ndarray
            Flux per unit lethargy (E * φ(E))
        """
        e_ref = 20e6  # 20 MeV reference
        lethargy = np.log(e_ref / self.energy_midpoints)
        flux_per_lethargy = self.energy_midpoints * self.flux
        return lethargy, flux_per_lethargy


# =============================================================================
# Spectrum Unfolding Workflow
# =============================================================================

class SpectrumUnfolder:
    """
    Main class for neutron spectrum unfolding.
    
    Combines IRDFF-II cross sections with iterative solvers to unfold
    measured reaction rates into continuous neutron spectra.
    
    Examples
    --------
    >>> unfolder = SpectrumUnfolder()
    >>> unfolder.add_reaction("Ti-46(n,p)Sc-46", activity=1.23e5, uncertainty=1.23e3)
    >>> unfolder.add_reaction("Ni-58(n,p)Co-58", activity=4.56e5, uncertainty=4.56e3)
    >>> unfolder.set_mcnp_initial_guess("spectrum.csv")
    >>> result = unfolder.unfold(method="GRAVEL")
    """
    
    def __init__(
        self,
        energy_structure: str = "flux_wire",
        custom_energy_edges: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Initialize spectrum unfolder.
        
        Parameters
        ----------
        energy_structure : str
            Energy group structure: 'flux_wire', 'activation', 'sand725', 'mcnp640'
        custom_energy_edges : np.ndarray, optional
            Custom energy edges in eV (overrides energy_structure)
        verbose : bool
            Print status messages
        """
        self.verbose = verbose
        
        # Set up energy structure
        if custom_energy_edges is not None:
            self.energy_edges = custom_energy_edges
        elif energy_structure == "flux_wire":
            self.energy_edges = get_flux_wire_energy_groups()
        elif energy_structure == "activation":
            self.energy_edges = get_activation_energy_groups()
        else:
            db = IRDFFDatabase()
            self.energy_edges = db.get_energy_grid(energy_structure)
        
        self.n_groups = len(self.energy_edges) - 1
        
        # Initialize IRDFF database
        self.irdff_db = IRDFFDatabase(verbose=verbose)
        
        # Measurement storage
        self.measurements: List[FluxWireMeasurement] = []
        
        # Initial guess
        self.initial_flux: Optional[np.ndarray] = None
        self.initial_guess_source = "uniform"
        
        # Response matrix (built when needed)
        self._response_matrix: Optional[np.ndarray] = None
        self._reaction_list: List[str] = []
        
        if self.verbose:
            print(f"SpectrumUnfolder initialized:")
            print(f"  Energy groups: {self.n_groups}")
            print(f"  Energy range: {self.energy_edges[0]:.2e} - {self.energy_edges[-1]:.2e} eV")
    
    def add_reaction(
        self,
        reaction: str,
        activity_Bq: float,
        uncertainty_Bq: float = 0.0,
        saturation_factor: float = 1.0,
        decay_factor: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Add a measured reaction for unfolding.
        
        Parameters
        ----------
        reaction : str
            Reaction identifier (e.g., 'Ti-46(n,p)Sc-46')
        activity_Bq : float
            Measured activity at EOI in Bq
        uncertainty_Bq : float
            Uncertainty in Bq
        saturation_factor : float
            Saturation correction
        decay_factor : float
            Decay correction
        **kwargs
            Additional parameters passed to FluxWireMeasurement
        """
        meas = FluxWireMeasurement(
            reaction=reaction,
            activity_Bq=activity_Bq,
            uncertainty_Bq=uncertainty_Bq,
            saturation_factor=saturation_factor,
            decay_factor=decay_factor,
            **kwargs,
        )
        self.measurements.append(meas)
        
        # Invalidate cached response matrix
        self._response_matrix = None
        
        if self.verbose:
            print(f"  Added: {reaction} - {activity_Bq:.3e} ± {uncertainty_Bq:.3e} Bq")
    
    def add_measurements_from_dataframe(
        self,
        df,
        reaction_col: str = "reaction",
        activity_col: str = "activity_Bq",
        uncertainty_col: str = "uncertainty_Bq",
    ) -> None:
        """
        Add measurements from a pandas DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame with reaction data
        reaction_col : str
            Column name for reaction identifier
        activity_col : str
            Column name for activity in Bq
        uncertainty_col : str
            Column name for uncertainty in Bq
        """
        for _, row in df.iterrows():
            self.add_reaction(
                reaction=row[reaction_col],
                activity_Bq=row[activity_col],
                uncertainty_Bq=row.get(uncertainty_col, 0.0),
            )
    
    def set_mcnp_initial_guess(
        self,
        spectrum_file: Union[str, Path],
        energy_col: int = 0,
        flux_col: int = 1,
        skiprows: int = 0,
        energy_units: str = "MeV",
        flux_units: str = "per_cm2_per_s",
    ) -> None:
        """
        Load MCNP spectrum as initial guess.
        
        Parameters
        ----------
        spectrum_file : str or Path
            Path to spectrum file (CSV or text)
        energy_col : int
            Column index for energy
        flux_col : int
            Column index for flux
        skiprows : int
            Number of header rows to skip
        energy_units : str
            Units of energy in file ('MeV' or 'eV')
        flux_units : str
            Units of flux in file
        """
        spectrum_file = Path(spectrum_file)
        
        if spectrum_file.suffix == '.csv':
            import csv
            with open(spectrum_file, 'r') as f:
                reader = csv.reader(f)
                for _ in range(skiprows):
                    next(reader)
                data = list(reader)
            
            energies = []
            fluxes = []
            for row in data:
                try:
                    e = float(row[energy_col])
                    f = float(row[flux_col])
                    energies.append(e)
                    fluxes.append(f)
                except (ValueError, IndexError):
                    continue
            
            energies = np.array(energies)
            fluxes = np.array(fluxes)
        else:
            data = np.loadtxt(spectrum_file, skiprows=skiprows)
            energies = data[:, energy_col]
            fluxes = data[:, flux_col]
        
        # Convert energy units to eV
        if energy_units.lower() == "mev":
            energies = energies * 1e6
        
        # Interpolate to our energy structure
        group_centers = np.sqrt(self.energy_edges[:-1] * self.energy_edges[1:])
        self.initial_flux = np.interp(group_centers, energies, fluxes, left=0, right=0)
        
        # Ensure positive
        self.initial_flux = np.maximum(self.initial_flux, 1e-30)
        
        self.initial_guess_source = "MCNP"
        
        if self.verbose:
            print(f"  Loaded MCNP spectrum from {spectrum_file}")
            print(f"  Interpolated to {self.n_groups} groups")
    
    def set_initial_guess(
        self,
        flux: np.ndarray,
        source: str = "user",
    ) -> None:
        """
        Set custom initial guess.
        
        Parameters
        ----------
        flux : np.ndarray
            Initial flux guess (length = n_groups)
        source : str
            Description of source
        """
        if len(flux) != self.n_groups:
            raise ValueError(f"Flux length {len(flux)} != n_groups {self.n_groups}")
        
        self.initial_flux = np.array(flux)
        self.initial_guess_source = source
    
    def _build_response_matrix(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Build response matrix from measurements."""
        if self._response_matrix is not None and len(self._reaction_list) == len(self.measurements):
            return self._response_matrix, self._reaction_list, self._response_unc
        
        reactions = [m.reaction for m in self.measurements]
        
        response, valid_reactions, uncertainties = build_response_matrix(
            reactions=reactions,
            energy_edges=self.energy_edges,
            db=self.irdff_db,
            verbose=self.verbose,
        )
        
        self._response_matrix = response
        self._reaction_list = valid_reactions
        self._response_unc = uncertainties
        
        return response, valid_reactions, uncertainties
    
    def unfold(
        self,
        method: str = "GRAVEL",
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        chi2_tolerance: float = 0.01,
        relaxation: float = 0.7,
    ) -> UnfoldingResult:
        """
        Perform spectrum unfolding.
        
        Parameters
        ----------
        method : str
            Unfolding method: 'GRAVEL' or 'MLEM'
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance (relative change)
        chi2_tolerance : float
            Chi-squared per DOF threshold
        relaxation : float
            Under-relaxation factor (0-1)
        
        Returns
        -------
        UnfoldingResult
            Unfolded spectrum and diagnostics
        """
        if len(self.measurements) == 0:
            raise ValueError("No measurements added. Use add_reaction() first.")
        
        # Build response matrix
        response_matrix, valid_reactions, response_unc = self._build_response_matrix()
        
        if len(valid_reactions) == 0:
            raise ValueError("No valid reactions found with cross section data.")
        
        # Get measured rates and uncertainties
        measured_rates = []
        rate_uncertainties = []
        for m in self.measurements:
            if m.reaction in valid_reactions:
                measured_rates.append(m.reaction_rate_per_atom)
                rate_uncertainties.append(
                    m.reaction_rate_per_atom * m.relative_uncertainty
                    if m.relative_uncertainty > 0 else m.reaction_rate_per_atom * 0.1
                )
        
        measured_rates = np.array(measured_rates)
        rate_uncertainties = np.array(rate_uncertainties)
        
        # Prepare initial guess
        if self.initial_flux is not None:
            initial = self.initial_flux.tolist()
        else:
            # Default: flat spectrum scaled to match measurements
            avg_rate = np.mean(measured_rates)
            avg_xs = np.mean(response_matrix)
            initial = [avg_rate / (avg_xs * self.n_groups) if avg_xs > 0 else 1.0] * self.n_groups
        
        if self.verbose:
            print(f"\nStarting {method} unfolding:")
            print(f"  Reactions: {len(valid_reactions)}")
            print(f"  Energy groups: {self.n_groups}")
            print(f"  Initial guess: {self.initial_guess_source}")
        
        # Run unfolding
        if method.upper() == "GRAVEL":
            result = gravel(
                response=response_matrix.tolist(),
                measurements=measured_rates.tolist(),
                initial_flux=initial,
                measurement_uncertainty=rate_uncertainties.tolist(),
                max_iters=max_iterations,
                tolerance=tolerance,
                chi2_tolerance=chi2_tolerance,
                relaxation=relaxation,
                verbose=self.verbose,
            )
        elif method.upper() == "MLEM":
            result = mlem(
                response=response_matrix.tolist(),
                measurements=measured_rates.tolist(),
                initial_flux=initial,
                measurement_uncertainty=rate_uncertainties.tolist(),
                max_iters=max_iterations,
                tolerance=tolerance,
                chi2_tolerance=chi2_tolerance,
                relaxation=relaxation,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'GRAVEL' or 'MLEM'.")
        
        # Calculate predicted rates
        flux_array = np.array(result.flux)
        predicted_rates = response_matrix @ flux_array
        
        # Estimate flux uncertainties (simplified - from response matrix propagation)
        flux_uncertainty = self._estimate_flux_uncertainty(
            flux_array, response_matrix, rate_uncertainties
        )
        
        if self.verbose:
            print(f"\nUnfolding complete:")
            print(f"  Iterations: {result.iterations}")
            print(f"  Converged: {result.converged}")
            print(f"  Chi²/dof: {result.chi_squared:.4f}")
        
        return UnfoldingResult(
            energy_edges=self.energy_edges,
            flux=flux_array,
            flux_uncertainty=flux_uncertainty,
            reactions_used=valid_reactions,
            response_matrix=response_matrix,
            measured_rates=measured_rates,
            predicted_rates=predicted_rates,
            chi_squared=result.chi_squared,
            iterations=result.iterations,
            converged=result.converged,
            method=method.upper(),
            initial_guess_source=self.initial_guess_source,
            metadata={
                "chi2_history": result.chi_squared_history,
                "final_residuals": result.final_residuals,
                "tolerance": tolerance,
                "relaxation": relaxation,
            },
        )
    
    def _estimate_flux_uncertainty(
        self,
        flux: np.ndarray,
        response: np.ndarray,
        rate_unc: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate flux uncertainties via pseudo-inverse propagation.
        
        This is a simplified uncertainty estimate. For rigorous uncertainty
        quantification, use Monte Carlo propagation.
        """
        # Sensitivity matrix: dφ/dy ~ (R^T R)^{-1} R^T
        try:
            RtR = response.T @ response
            # Add regularization for stability
            reg = 1e-10 * np.trace(RtR) / RtR.shape[0] * np.eye(RtR.shape[0])
            RtR_inv = np.linalg.inv(RtR + reg)
            sensitivity = RtR_inv @ response.T
            
            # Propagate uncertainties
            flux_var = np.sum((sensitivity * rate_unc) ** 2, axis=1)
            flux_unc = np.sqrt(flux_var)
        except np.linalg.LinAlgError:
            # Fall back to simple relative uncertainty
            avg_rel_unc = np.mean(rate_unc / np.maximum(np.abs(flux), 1e-30))
            flux_unc = flux * avg_rel_unc
        
        return flux_unc
    
    def compare_with_mcnp(
        self,
        mcnp_spectrum: Union[str, Path, np.ndarray],
        unfolded_result: UnfoldingResult,
    ) -> Dict[str, Any]:
        """
        Compare unfolded spectrum with MCNP reference.
        
        Parameters
        ----------
        mcnp_spectrum : str, Path, or np.ndarray
            MCNP spectrum (file path or array)
        unfolded_result : UnfoldingResult
            Unfolded spectrum
        
        Returns
        -------
        Dict
            Comparison metrics
        """
        if isinstance(mcnp_spectrum, (str, Path)):
            # Load from file
            self.set_mcnp_initial_guess(mcnp_spectrum)
            mcnp_flux = self.initial_flux
        else:
            mcnp_flux = np.array(mcnp_spectrum)
        
        unfolded_flux = unfolded_result.flux
        
        # Normalize for comparison
        mcnp_norm = mcnp_flux / np.sum(mcnp_flux)
        unfolded_norm = unfolded_flux / np.sum(unfolded_flux)
        
        # Calculate metrics
        ratio = np.where(mcnp_norm > 1e-30, unfolded_norm / mcnp_norm, 1.0)
        residual = unfolded_norm - mcnp_norm
        
        return {
            "mcnp_flux": mcnp_flux,
            "mcnp_normalized": mcnp_norm,
            "unfolded_normalized": unfolded_norm,
            "ratio": ratio,
            "residual": residual,
            "mean_ratio": np.mean(ratio),
            "std_ratio": np.std(ratio),
            "max_deviation": np.max(np.abs(residual)),
            "rms_deviation": np.sqrt(np.mean(residual**2)),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_unfold(
    reactions: Dict[str, float],
    uncertainties: Optional[Dict[str, float]] = None,
    initial_spectrum: Optional[np.ndarray] = None,
    method: str = "GRAVEL",
    energy_structure: str = "flux_wire",
    verbose: bool = True,
) -> UnfoldingResult:
    """
    Quick spectrum unfolding from dictionary of reactions.
    
    Parameters
    ----------
    reactions : Dict[str, float]
        Dictionary mapping reaction -> activity (Bq)
    uncertainties : Dict[str, float], optional
        Dictionary mapping reaction -> uncertainty (Bq)
    initial_spectrum : np.ndarray, optional
        Initial flux guess
    method : str
        Unfolding method
    energy_structure : str
        Energy group structure
    verbose : bool
        Print status
    
    Returns
    -------
    UnfoldingResult
        Unfolded spectrum
    
    Examples
    --------
    >>> result = quick_unfold({
    ...     "Ti-46(n,p)Sc-46": 1.23e5,
    ...     "Ni-58(n,p)Co-58": 4.56e5,
    ...     "Co-59(n,g)Co-60": 7.89e3,
    ... })
    """
    if uncertainties is None:
        uncertainties = {rxn: 0.1 * act for rxn, act in reactions.items()}
    
    unfolder = SpectrumUnfolder(
        energy_structure=energy_structure,
        verbose=verbose,
    )
    
    for rxn, activity in reactions.items():
        unfolder.add_reaction(
            reaction=rxn,
            activity_Bq=activity,
            uncertainty_Bq=uncertainties.get(rxn, activity * 0.1),
        )
    
    if initial_spectrum is not None:
        unfolder.set_initial_guess(initial_spectrum, source="user")
    
    return unfolder.unfold(method=method)


def build_flux_wire_response_matrix(
    reactions: Optional[List[str]] = None,
    energy_structure: str = "flux_wire",
    custom_edges: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build response matrix for standard flux wire reactions.
    
    Parameters
    ----------
    reactions : List[str], optional
        Reactions to include. If None, uses standard set.
    energy_structure : str
        Energy group structure
    custom_edges : np.ndarray, optional
        Custom energy edges
    
    Returns
    -------
    response_matrix : np.ndarray
        Response matrix (n_reactions x n_groups)
    energy_edges : np.ndarray
        Energy group edges (eV)
    reactions : List[str]
        Reaction names
    """
    if reactions is None:
        reactions = [
            "Ti-46(n,p)Sc-46",
            "Ti-47(n,p)Sc-47",
            "Ti-48(n,p)Sc-48",
            "Ni-58(n,p)Co-58",
            "Fe-56(n,p)Mn-56",
            "In-115(n,n')In-115m",
            "Al-27(n,a)Na-24",
            "Co-59(n,g)Co-60",
            "Sc-45(n,g)Sc-46",
            "Fe-58(n,g)Fe-59",
            "Cu-63(n,g)Cu-64",
        ]
    
    if custom_edges is not None:
        energy_edges = custom_edges
    elif energy_structure == "flux_wire":
        energy_edges = get_flux_wire_energy_groups()
    elif energy_structure == "activation":
        energy_edges = get_activation_energy_groups()
    else:
        db = IRDFFDatabase()
        energy_edges = db.get_energy_grid(energy_structure)
    
    response, valid_reactions, _ = build_response_matrix(
        reactions=reactions,
        energy_edges=energy_edges,
        verbose=False,
    )
    
    return response, energy_edges, valid_reactions
