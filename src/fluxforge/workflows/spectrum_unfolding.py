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
from math import exp, log
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# FluxForge imports
from fluxforge.core.unfolding_diagnostics import merge_flux_diagnostics
from fluxforge.core.unfolding_inputs import require_nonnegative
from fluxforge.data.irdff import (
    IRDFFDatabase,
    IRDFFCrossSection,
    get_flux_wire_energy_groups,
    get_activation_energy_groups,
    build_response_matrix,
    IRDFF_REACTIONS,
)
from fluxforge.data.flux_wire_unfolding import (
    get_flux_wire_isotope_fraction,
    load_flux_wire_sample_defaults,
)
from fluxforge.data.nndc import Isotope
from fluxforge.solvers.iterative import gravel, mlem, IterativeSolution
from fluxforge.unfolding import MLSeedUnfolder, MaxedUnfolder, RMLEUnfolder


# =============================================================================
# Data Classes
# =============================================================================


_AVOGADRO = 6.02214076e23
_BARN_TO_CM2 = 1.0e-24
_FLUX_WIRE_SAMPLE_DEFAULTS = load_flux_wire_sample_defaults()


def _get_reaction_metadata(reaction: str) -> Dict[str, Any]:
    """Return IRDFF metadata for one reaction when available."""
    for category in IRDFF_REACTIONS.values():
        if reaction in category:
            return dict(category[reaction])
    return {}


def _canonical_isotope_or_none(name: str) -> Optional[str]:
    """Return a canonical isotope string when parsing succeeds."""
    try:
        return Isotope.from_string(name).name
    except Exception:
        return None


def _reaction_target_and_product(reaction: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract target and product isotopes from one reaction identifier."""
    metadata = _get_reaction_metadata(reaction)
    target = metadata.get("target")
    product = metadata.get("product")
    if target and product:
        return str(target), str(product)

    match = re.match(r"^\s*([A-Za-z]+-\d+m?)\([^)]*\)([A-Za-z]+-\d+m?)\s*$", reaction)
    if match:
        return match.group(1), match.group(2)
    return None, None


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
    rate_per_atom : float | None
        Optional pre-normalized reaction rate in reactions/atom/s. When set,
        this bypasses the activity-to-rate conversion path.
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
    rate_per_atom: Optional[float] = None

    @property
    def target_isotope(self) -> Optional[str]:
        """Target isotope parsed from the reaction identifier."""
        target, _ = _reaction_target_and_product(self.reaction)
        return target

    @property
    def product_isotope(self) -> Optional[str]:
        """Activation product isotope parsed from the reaction identifier."""
        _, product = _reaction_target_and_product(self.reaction)
        return product

    @property
    def effective_isotope_abundance(self) -> float:
        """Resolved target-isotope abundance fraction."""
        if self.isotope_abundance > 0.0 and not np.isclose(self.isotope_abundance, 1.0):
            return float(self.isotope_abundance)

        target = self.target_isotope
        if target:
            try:
                element = target.split("-", 1)[0]
                default_fraction = get_flux_wire_isotope_fraction(self.reaction, element)
                if default_fraction > 0.0 and not np.isclose(default_fraction, 1.0):
                    return float(default_fraction)
            except Exception:
                pass

            canonical_target = _canonical_isotope_or_none(target)
            if canonical_target is not None:
                isotope = Isotope.from_string(canonical_target)
                if isotope.abundance is not None and isotope.abundance > 0.0:
                    return float(isotope.abundance)

        return float(self.isotope_abundance)

    @property
    def target_atom_count(self) -> float:
        """Number of target atoms in the measured wire."""
        if self.sample_mass_g <= 0.0:
            return 0.0

        target = self.target_isotope
        if target is None:
            return 0.0

        element = target.split("-", 1)[0]
        atomic_mass = None
        defaults = _FLUX_WIRE_SAMPLE_DEFAULTS.get(element, {})
        if defaults:
            atomic_mass = defaults.get("atomic_mass")

        if atomic_mass is None:
            canonical_target = _canonical_isotope_or_none(target)
            if canonical_target is not None:
                atomic_mass = Isotope.from_string(canonical_target).atomic_mass

        if atomic_mass is None or atomic_mass <= 0.0:
            return 0.0

        abundance = self.effective_isotope_abundance
        if abundance <= 0.0:
            return 0.0

        return float((self.sample_mass_g / atomic_mass) * _AVOGADRO * abundance)

    @property
    def effective_saturation_factor(self) -> float:
        """Saturation factor, derived from timing metadata when available."""
        if self.saturation_factor > 0.0 and not np.isclose(self.saturation_factor, 1.0):
            return float(self.saturation_factor)

        product = self.product_isotope
        if product is None or self.irradiation_time <= 0.0:
            return float(self.saturation_factor)

        canonical_product = _canonical_isotope_or_none(product)
        if canonical_product is None:
            return float(self.saturation_factor)

        half_life_s = Isotope.from_string(canonical_product).half_life_s
        if half_life_s is None or half_life_s <= 0.0 or half_life_s == float("inf"):
            return float(self.saturation_factor)

        decay_constant = log(2.0) / half_life_s
        return float(1.0 - exp(-decay_constant * self.irradiation_time))

    @property
    def effective_decay_factor(self) -> float:
        """Decay factor, derived from timing metadata when available."""
        if self.decay_factor > 0.0 and not np.isclose(self.decay_factor, 1.0):
            return float(self.decay_factor)

        product = self.product_isotope
        if product is None or self.cooling_time <= 0.0:
            return float(self.decay_factor)

        canonical_product = _canonical_isotope_or_none(product)
        if canonical_product is None:
            return float(self.decay_factor)

        half_life_s = Isotope.from_string(canonical_product).half_life_s
        if half_life_s is None or half_life_s <= 0.0 or half_life_s == float("inf"):
            return float(self.decay_factor)

        decay_constant = log(2.0) / half_life_s
        return float(exp(-decay_constant * self.cooling_time))

    @property
    def reaction_rate_per_atom(self) -> float:
        """Calculate reaction rate per target atom per second."""
        if self.rate_per_atom is not None:
            return float(self.rate_per_atom)
        n_target_atoms = self.target_atom_count
        saturation = self.effective_saturation_factor
        decay = self.effective_decay_factor
        denominator = n_target_atoms * saturation * decay
        if denominator <= 0.0:
            return 0.0
        return self.activity_Bq / denominator

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
            print(
                f"  Energy range: {self.energy_edges[0]:.2e} - {self.energy_edges[-1]:.2e} eV"
            )

    def add_reaction(
        self,
        reaction: str,
        activity_Bq: float,
        uncertainty_Bq: float = 0.0,
        saturation_factor: float = 1.0,
        decay_factor: float = 1.0,
        rate_per_atom: Optional[float] = None,
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
        rate_per_atom : float, optional
            Pre-normalized reaction rate in reactions/atom/s. Use this when the
            upstream workflow already converted activity to reaction rate.
        **kwargs
            Additional parameters passed to FluxWireMeasurement
        """
        meas = FluxWireMeasurement(
            reaction=reaction,
            activity_Bq=activity_Bq,
            uncertainty_Bq=uncertainty_Bq,
            saturation_factor=saturation_factor,
            decay_factor=decay_factor,
            rate_per_atom=rate_per_atom,
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

        if spectrum_file.suffix == ".csv":
            import csv

            with open(spectrum_file, "r") as f:
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

        self.initial_flux = require_nonnegative("initial_flux", flux).reshape(-1)
        self.initial_guess_source = source

    def _build_response_matrix(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Build response matrix from measurements."""
        if self._response_matrix is not None and len(self._reaction_list) == len(
            self.measurements
        ):
            return self._response_matrix, self._reaction_list, self._response_unc

        reactions = [m.reaction for m in self.measurements]

        response, valid_reactions, uncertainties = build_response_matrix(
            reactions=reactions,
            energy_edges=self.energy_edges,
            db=self.irdff_db,
            verbose=self.verbose,
        )

        group_widths = np.diff(self.energy_edges).reshape(1, -1)
        response = response * group_widths * _BARN_TO_CM2
        uncertainties = uncertainties * group_widths * _BARN_TO_CM2

        self._response_matrix = response
        self._reaction_list = valid_reactions
        self._response_unc = uncertainties

        return response, valid_reactions, uncertainties

    def _prepare_support_filtered_problem(
        self,
        response_matrix: np.ndarray,
        measured_rates: np.ndarray,
        rate_uncertainties: np.ndarray,
        initial_flux: np.ndarray,
        *,
        support_threshold: Optional[float] = None,
        support_metric: str = "prior_contribution",
        floor: float = 1e-30,
    ) -> Dict[str, Any]:
        """Reduce the unfolding problem to energy bins materially constrained by the prior."""
        n_groups = response_matrix.shape[1]
        active_mask = np.ones(n_groups, dtype=bool)
        support_scores = np.ones(n_groups, dtype=float)
        fixed_prediction = np.zeros_like(measured_rates, dtype=float)

        if (
            support_threshold is None
            or support_threshold <= 0.0
            or initial_flux.size != n_groups
        ):
            return {
                "response": response_matrix,
                "measurements": measured_rates,
                "uncertainties": rate_uncertainties,
                "initial_flux": initial_flux,
                "active_mask": active_mask,
                "support_scores": support_scores,
                "fixed_prediction": fixed_prediction,
            }

        if support_metric == "prior_contribution":
            group_contributions = response_matrix * initial_flux.reshape(1, -1)
            total_prediction = np.sum(group_contributions, axis=1, keepdims=True)
            support_scores = np.max(
                group_contributions / np.maximum(total_prediction, floor),
                axis=0,
            )
        elif support_metric == "response_sum":
            group_sensitivity = np.sum(response_matrix, axis=0)
            max_sensitivity = float(np.max(group_sensitivity)) if group_sensitivity.size else 0.0
            if max_sensitivity > 0.0:
                support_scores = group_sensitivity / max_sensitivity
            else:
                support_scores = np.zeros(n_groups, dtype=float)
        else:
            raise ValueError(
                f"Unknown support metric: {support_metric}. Use 'prior_contribution' or 'response_sum'."
            )

        active_mask = support_scores >= float(support_threshold)
        if not np.any(active_mask) or np.all(active_mask):
            return {
                "response": response_matrix,
                "measurements": measured_rates,
                "uncertainties": rate_uncertainties,
                "initial_flux": initial_flux,
                "active_mask": np.ones(n_groups, dtype=bool),
                "support_scores": support_scores,
                "fixed_prediction": fixed_prediction,
            }

        fixed_prediction = response_matrix[:, ~active_mask] @ initial_flux[~active_mask]
        adjusted_measurements = np.maximum(measured_rates - fixed_prediction, floor)

        return {
            "response": response_matrix[:, active_mask],
            "measurements": adjusted_measurements,
            "uncertainties": rate_uncertainties,
            "initial_flux": initial_flux[active_mask],
            "active_mask": active_mask,
            "support_scores": support_scores,
            "fixed_prediction": fixed_prediction,
        }

    def _build_prior_shape_basis(
        self,
        prior_flux: np.ndarray,
        basis_edges: np.ndarray,
        *,
        floor: float = 1e-30,
    ) -> Dict[str, Any]:
        """
        Build a coarse prior-shaped basis for few-channel unfolding.

        Each coarse coefficient scales the prior spectrum within one contiguous
        energy interval. This keeps the solve dimension aligned with the number
        of informative monitors while preserving the prior shape inside each
        coarse interval.
        """
        native_midpoints = np.sqrt(
            np.maximum(self.energy_edges[:-1], floor)
            * np.maximum(self.energy_edges[1:], floor)
        )
        resolved_edges = np.asarray(basis_edges, dtype=float).reshape(-1)
        if resolved_edges.size < 2:
            raise ValueError("basis_edges must contain at least two boundaries")
        if not np.all(np.diff(resolved_edges) > 0.0):
            raise ValueError("basis_edges must be strictly increasing")
        if resolved_edges[0] > float(self.energy_edges[0]) or resolved_edges[-1] < float(
            self.energy_edges[-1]
        ):
            raise ValueError(
                "basis_edges must span the full unfolding energy range "
                f"({self.energy_edges[0]:.3e} to {self.energy_edges[-1]:.3e} eV)"
            )

        basis_columns: List[np.ndarray] = []
        active_edges: List[float] = [float(resolved_edges[0])]
        for index, (e_lo, e_hi) in enumerate(zip(resolved_edges[:-1], resolved_edges[1:])):
            if index == len(resolved_edges) - 2:
                mask = (native_midpoints >= e_lo) & (native_midpoints <= e_hi)
            else:
                mask = (native_midpoints >= e_lo) & (native_midpoints < e_hi)
            if not np.any(mask):
                continue

            column = np.zeros_like(prior_flux, dtype=float)
            column[mask] = np.maximum(prior_flux[mask], floor)
            if not np.any(column > floor):
                column[mask] = 1.0
            basis_columns.append(column)
            active_edges.append(float(e_hi))

        if not basis_columns:
            raise ValueError("basis_edges did not capture any native energy groups")

        basis_matrix = np.column_stack(basis_columns)
        reconstructed_prior = basis_matrix @ np.ones(basis_matrix.shape[1], dtype=float)
        if not np.allclose(reconstructed_prior, np.maximum(prior_flux, floor)):
            raise ValueError("basis_edges do not partition the prior spectrum cleanly")

        return {
            "basis_matrix": basis_matrix,
            "basis_edges": np.asarray(active_edges, dtype=float),
            "n_basis_groups": int(basis_matrix.shape[1]),
        }

    def _aggregate_duplicate_reaction_rows(
        self,
        response_matrix: np.ndarray,
        valid_reactions: List[str],
        measured_rates: np.ndarray,
        rate_uncertainties: np.ndarray,
        response_uncertainties: Optional[np.ndarray] = None,
        *,
        floor: float = 1e-30,
    ) -> Dict[str, Any]:
        """
        Aggregate repeated reaction rows using inverse-variance weighting.

        This is useful when multiple wire replicates map to the same reaction
        response row. Treating those replicates as fully independent rows
        artificially over-weights one response shape in the inversion.
        """
        if len(valid_reactions) <= 1:
            return {
                "response": response_matrix,
                "reactions": list(valid_reactions),
                "measurements": measured_rates,
                "uncertainties": rate_uncertainties,
                "response_uncertainties": response_uncertainties,
                "metadata": {
                    "applied": False,
                    "original_rows": int(len(valid_reactions)),
                    "aggregated_rows": int(len(valid_reactions)),
                    "reaction_counts": {
                        str(reaction): 1 for reaction in valid_reactions
                    },
                },
            }

        order: List[str] = []
        grouped_indices: Dict[str, List[int]] = {}
        for idx, reaction in enumerate(valid_reactions):
            if reaction not in grouped_indices:
                grouped_indices[reaction] = []
                order.append(reaction)
            grouped_indices[reaction].append(idx)

        has_duplicates = any(len(indices) > 1 for indices in grouped_indices.values())
        if not has_duplicates:
            return {
                "response": response_matrix,
                "reactions": list(valid_reactions),
                "measurements": measured_rates,
                "uncertainties": rate_uncertainties,
                "response_uncertainties": response_uncertainties,
                "metadata": {
                    "applied": False,
                    "original_rows": int(len(valid_reactions)),
                    "aggregated_rows": int(len(valid_reactions)),
                    "reaction_counts": {
                        str(reaction): 1 for reaction in valid_reactions
                    },
                },
            }

        aggregated_response_rows: List[np.ndarray] = []
        aggregated_reactions: List[str] = []
        aggregated_measurements: List[float] = []
        aggregated_uncertainties: List[float] = []
        aggregated_response_unc_rows: List[np.ndarray] = []

        for reaction in order:
            indices = grouped_indices[reaction]
            representative_idx = indices[0]
            representative_row = np.asarray(response_matrix[representative_idx], dtype=float)
            weights = 1.0 / np.maximum(rate_uncertainties[indices], floor) ** 2
            weight_sum = float(np.sum(weights))
            aggregated_measurement = float(
                np.sum(weights * measured_rates[indices]) / max(weight_sum, floor)
            )
            aggregated_uncertainty = float(np.sqrt(1.0 / max(weight_sum, floor)))

            aggregated_response_rows.append(representative_row)
            aggregated_reactions.append(reaction)
            aggregated_measurements.append(aggregated_measurement)
            aggregated_uncertainties.append(aggregated_uncertainty)

            if response_uncertainties is not None:
                aggregated_response_unc_rows.append(
                    np.asarray(response_uncertainties[representative_idx], dtype=float)
                )

        return {
            "response": np.asarray(aggregated_response_rows, dtype=float),
            "reactions": aggregated_reactions,
            "measurements": np.asarray(aggregated_measurements, dtype=float),
            "uncertainties": np.asarray(aggregated_uncertainties, dtype=float),
            "response_uncertainties": (
                np.asarray(aggregated_response_unc_rows, dtype=float)
                if response_uncertainties is not None
                else None
            ),
            "metadata": {
                "applied": True,
                "original_rows": int(len(valid_reactions)),
                "aggregated_rows": int(len(aggregated_reactions)),
                "reaction_counts": {
                    str(reaction): int(len(grouped_indices[reaction]))
                    for reaction in order
                },
            },
        }

    def unfold(
        self,
        method: str = "GRAVEL",
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        chi2_tolerance: float = 0.01,
        relaxation: float = 0.7,
        prior_strength: float = 0.0,
        smoothing_strength: float = 0.0,
        use_ml_seed: bool = False,
        ml_seed_threshold: float = 0.6,
        support_threshold: Optional[float] = None,
        support_metric: str = "prior_contribution",
        basis_edges: Optional[np.ndarray] = None,
        aggregate_duplicate_reactions: bool = False,
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
        prior_strength : float
            Geometric pull toward the initial spectrum after each iteration.
            Zero disables this regularization.
        smoothing_strength : float
            Log-space nearest-neighbour smoothing strength applied after each
            iterative update. Zero disables smoothing.
        use_ml_seed : bool
            Use the ML Seed approximation to initialize GRAVEL or RMLE
        ml_seed_threshold : float
            Confidence threshold for accepting the ML seed initializer
        support_threshold : float, optional
            When set, bins with support scores below this threshold are frozen to the
            initial spectrum and only the active subset is iteratively unfolded.
        support_metric : str
            Support score definition: 'prior_contribution' (default) or
            'response_sum'.
        basis_edges : np.ndarray, optional
            Coarse energy boundaries for a few-channel prior-shaped basis solve.
            When provided, the solver updates one coefficient per basis interval
            while preserving the initial spectrum shape inside that interval.
        aggregate_duplicate_reactions : bool
            Combine repeated rows with the same reaction identifier using
            inverse-variance weighting before unfolding. This is useful when
            replicate wires map to identical response functions.

        Returns
        -------
        UnfoldingResult
            Unfolded spectrum and diagnostics
        """
        if len(self.measurements) == 0:
            raise ValueError("No measurements added. Use add_reaction() first.")

        # Build response matrix
        response_matrix, valid_reactions, response_unc = self._build_response_matrix()
        response_matrix = require_nonnegative("response_matrix", response_matrix)

        if len(valid_reactions) == 0:
            raise ValueError("No valid reactions found with cross section data.")

        # Get measured rates and uncertainties
        measured_rates = []
        rate_uncertainties = []
        for m in self.measurements:
            if m.reaction in valid_reactions:
                rate_value = float(m.reaction_rate_per_atom)
                if rate_value <= 0.0 and float(m.activity_Bq) > 0.0:
                    # Preserve historical behavior for synthetic/legacy fixtures:
                    # when atom-normalization metadata is unavailable, use
                    # activity as the proxy reaction-rate observable.
                    rate_value = float(m.activity_Bq)
                measured_rates.append(rate_value)
                rel_uncertainty = float(m.relative_uncertainty)
                rate_uncertainties.append(
                    rate_value * rel_uncertainty
                    if rel_uncertainty > 0.0
                    else rate_value * 0.1
                )

        measured_rates = require_nonnegative("measured_rates", measured_rates).reshape(-1)
        rate_uncertainties = require_nonnegative(
            "rate_uncertainties",
            rate_uncertainties,
        ).reshape(-1)

        duplicate_metadata: Dict[str, Any] = {
            "applied": False,
            "original_rows": int(len(valid_reactions)),
            "aggregated_rows": int(len(valid_reactions)),
            "reaction_counts": {
                str(reaction): 1 for reaction in valid_reactions
            },
        }
        if aggregate_duplicate_reactions:
            aggregation_payload = self._aggregate_duplicate_reaction_rows(
                response_matrix,
                valid_reactions,
                measured_rates,
                rate_uncertainties,
                response_unc,
            )
            response_matrix = require_nonnegative(
                "response_matrix",
                aggregation_payload["response"],
            )
            valid_reactions = list(aggregation_payload["reactions"])
            measured_rates = require_nonnegative(
                "measured_rates",
                aggregation_payload["measurements"],
            ).reshape(-1)
            rate_uncertainties = require_nonnegative(
                "rate_uncertainties",
                aggregation_payload["uncertainties"],
            ).reshape(-1)
            response_unc = aggregation_payload["response_uncertainties"]
            duplicate_metadata = dict(aggregation_payload["metadata"])

        # Prepare initial guess
        if self.initial_flux is not None:
            initial = self.initial_flux.tolist()
        else:
            # Default: flat spectrum scaled to match measurements
            avg_rate = np.mean(measured_rates)
            avg_xs = np.mean(response_matrix)
            initial = [
                avg_rate / (avg_xs * self.n_groups) if avg_xs > 0 else 1.0
            ] * self.n_groups
        initial_array = require_nonnegative("initial_flux", initial).reshape(-1)

        basis_matrix: Optional[np.ndarray] = None
        basis_metadata: Dict[str, Any] = {
            "basis_mode": "native",
            "basis_edges": [],
            "basis_groups": int(self.n_groups),
        }
        full_solver_initial = initial_array
        solver_base_response = response_matrix
        if basis_edges is not None:
            basis_payload = self._build_prior_shape_basis(initial_array, basis_edges)
            basis_matrix = np.asarray(basis_payload["basis_matrix"], dtype=float)
            solver_base_response = response_matrix @ basis_matrix
            full_solver_initial = np.ones(basis_matrix.shape[1], dtype=float)
            basis_metadata = {
                "basis_mode": "prior_shape",
                "basis_edges": basis_payload["basis_edges"].tolist(),
                "basis_groups": int(basis_payload["n_basis_groups"]),
            }

        support_problem = self._prepare_support_filtered_problem(
            solver_base_response,
            measured_rates,
            rate_uncertainties,
            full_solver_initial,
            support_threshold=support_threshold,
            support_metric=support_metric,
        )
        solver_response_matrix = require_nonnegative(
            "solver_response_matrix",
            support_problem["response"],
        )
        solver_measurements = require_nonnegative(
            "solver_measurements",
            support_problem["measurements"],
        ).reshape(-1)
        solver_rate_uncertainties = require_nonnegative(
            "solver_rate_uncertainties",
            support_problem["uncertainties"],
        ).reshape(-1)
        solver_initial = require_nonnegative(
            "solver_initial",
            support_problem["initial_flux"],
        ).reshape(-1)
        active_mask = np.asarray(support_problem["active_mask"], dtype=bool)
        support_scores = np.asarray(support_problem["support_scores"], dtype=float)
        fixed_prediction = np.asarray(support_problem["fixed_prediction"], dtype=float)

        if self.verbose:
            print(f"\nStarting {method} unfolding:")
            print(f"  Reactions: {len(valid_reactions)}")
            print(f"  Energy groups: {self.n_groups}")
            print(f"  Initial guess: {self.initial_guess_source}")
            if duplicate_metadata["applied"]:
                print(
                    "  Duplicate aggregation: "
                    f"{duplicate_metadata['original_rows']} -> {duplicate_metadata['aggregated_rows']} rows"
                )
            if not np.all(active_mask):
                print(
                    f"  Active groups: {int(np.count_nonzero(active_mask))}/{len(active_mask)} "
                    f"(metric={support_metric}, threshold={support_threshold:.3e})"
                )

        # Run unfolding
        seed_metadata: Dict[str, Any] = {}
        if method.upper() == "GRAVEL":
            if use_ml_seed:
                seed_result = MLSeedUnfolder().unfold(
                    solver_measurements,
                    solver_response_matrix,
                    initial_flux=solver_initial,
                    measurement_uncertainty=solver_rate_uncertainties,
                    confidence_threshold=ml_seed_threshold,
                )
                seed_metadata = {
                    "seed_with_ml": True,
                    "seed_accepted": bool(
                        seed_result.parameters_used.get("accepted", False)
                    ),
                    "seed_confidence_score": float(
                        seed_result.parameters_used.get("confidence_score", 0.0)
                    ),
                    "seed_backend": str(
                        seed_result.parameters_used.get("backend", "")
                    ),
                }
                if bool(seed_result.parameters_used.get("accepted", False)):
                    solver_initial = np.asarray(seed_result.flux, dtype=float)
            result = gravel(
                response=solver_response_matrix.tolist(),
                measurements=solver_measurements.tolist(),
                initial_flux=solver_initial.tolist(),
                measurement_uncertainty=solver_rate_uncertainties.tolist(),
                max_iters=max_iterations,
                tolerance=tolerance,
                chi2_tolerance=chi2_tolerance,
                relaxation=relaxation,
                prior_strength=prior_strength,
                smoothing_strength=smoothing_strength,
                verbose=self.verbose,
            )
        elif method.upper() == "MLEM":
            result = mlem(
                response=solver_response_matrix.tolist(),
                measurements=solver_measurements.tolist(),
                initial_flux=solver_initial.tolist(),
                measurement_uncertainty=solver_rate_uncertainties.tolist(),
                max_iters=max_iterations,
                tolerance=tolerance,
                chi2_tolerance=chi2_tolerance,
                relaxation=relaxation,
                prior_strength=prior_strength,
                smoothing_strength=smoothing_strength,
                verbose=self.verbose,
            )
        elif method.upper() == "MAXED":
            result = MaxedUnfolder(
                max_iterations=max_iterations,
            ).unfold(
                solver_measurements,
                solver_response_matrix,
                initial_flux=solver_initial,
                measurement_uncertainty=solver_rate_uncertainties,
            )
        elif method.upper() == "ML_SEED":
            result = MLSeedUnfolder().unfold(
                solver_measurements,
                solver_response_matrix,
                initial_flux=solver_initial,
                measurement_uncertainty=solver_rate_uncertainties,
                confidence_threshold=ml_seed_threshold,
            )
        elif method.upper() == "RMLE":
            result = RMLEUnfolder(
                max_iterations=max_iterations,
                tolerance=tolerance,
            ).unfold(
                solver_measurements,
                solver_response_matrix,
                initial_flux=solver_initial,
                measurement_uncertainty=solver_rate_uncertainties,
                seed_with_ml=use_ml_seed,
                confidence_threshold=ml_seed_threshold,
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'GRAVEL', 'MLEM', 'MAXED', 'RMLE', or 'ML_SEED'."
            )

        # Calculate predicted rates
        solver_flux = np.array(result.flux, dtype=float)
        if np.all(active_mask):
            solved_state = solver_flux
        else:
            solved_state = full_solver_initial.copy()
            solved_state[active_mask] = solver_flux
        if basis_matrix is None:
            flux_array = solved_state
        else:
            flux_array = basis_matrix @ solved_state
        predicted_rates = response_matrix @ flux_array

        # Estimate flux uncertainties (simplified - from response matrix propagation)
        result_uncertainty = getattr(result, "uncertainties", None)
        if result_uncertainty is not None:
            solved_uncertainty = np.asarray(result_uncertainty, dtype=float)
            full_state_uncertainty = np.zeros_like(solved_state)
            if np.all(active_mask):
                full_state_uncertainty = solved_uncertainty
            else:
                full_state_uncertainty[active_mask] = solved_uncertainty
            if basis_matrix is None:
                flux_uncertainty = full_state_uncertainty
            else:
                flux_uncertainty = basis_matrix @ full_state_uncertainty
        else:
            if basis_matrix is None and np.all(active_mask):
                flux_uncertainty = self._estimate_flux_uncertainty(
                    flux_array, response_matrix, rate_uncertainties
                )
            else:
                active_unc = self._estimate_flux_uncertainty(
                    solver_flux,
                    solver_response_matrix,
                    solver_rate_uncertainties,
                )
                full_state_uncertainty = np.zeros_like(solved_state)
                full_state_uncertainty[active_mask] = active_unc
                if basis_matrix is None:
                    flux_uncertainty = full_state_uncertainty
                else:
                    flux_uncertainty = basis_matrix @ full_state_uncertainty

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
            metadata=merge_flux_diagnostics(
                {
                    **dict(getattr(result, "parameters_used", {})),
                    **dict(getattr(result, "diagnostics", {})),
                    **seed_metadata,
                    "chi2_history": list(getattr(result, "chi_squared_history", []))
                    or list(getattr(result, "convergence_history", [])),
                    "final_residuals": getattr(result, "final_residuals", [])
                    or list(getattr(result, "residuals", [])),
                    "support_mask_applied": bool(not np.all(active_mask)),
                    "support_metric": support_metric,
                    "support_threshold": support_threshold,
                    "support_mask_active_bins": int(np.count_nonzero(active_mask)),
                    "support_mask_total_bins": int(len(active_mask)),
                    "support_mask_fixed_prediction": fixed_prediction.tolist(),
                    "support_scores_max": float(np.max(support_scores))
                    if support_scores.size
                    else 0.0,
                    "aggregate_duplicate_reactions": bool(aggregate_duplicate_reactions),
                    "duplicate_reaction_aggregation_applied": bool(
                        duplicate_metadata["applied"]
                    ),
                    "duplicate_reaction_original_rows": int(
                        duplicate_metadata["original_rows"]
                    ),
                    "duplicate_reaction_aggregated_rows": int(
                        duplicate_metadata["aggregated_rows"]
                    ),
                    "duplicate_reaction_counts": dict(
                        duplicate_metadata["reaction_counts"]
                    ),
                    **basis_metadata,
                    "tolerance": tolerance,
                    "relaxation": relaxation,
                    "prior_strength": prior_strength,
                    "smoothing_strength": smoothing_strength,
                },
                flux_array,
            ),
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
    use_ml_seed: bool = False,
    ml_seed_threshold: float = 0.6,
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
    use_ml_seed : bool
        Use the ML Seed approximation to initialize GRAVEL or RMLE
    ml_seed_threshold : float
        Confidence threshold for accepting the ML seed initializer

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

    return unfolder.unfold(
        method=method,
        use_ml_seed=use_ml_seed,
        ml_seed_threshold=ml_seed_threshold,
    )


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
