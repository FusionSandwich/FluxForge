"""
Radioactive Decay Chain Solver (Bateman Equations)

Epic T - Curie Parity

Implements analytical and numerical solutions to the Bateman equations
for radioactive decay chains with optional production rates.

The Bateman equations describe the time evolution of nuclide concentrations
in a decay chain:
    dN_1/dt = P_1 - λ_1 N_1
    dN_i/dt = P_i + λ_{i-1} BR_{i-1→i} N_{i-1} - λ_i N_i

where:
    N_i = number of atoms of nuclide i
    λ_i = decay constant of nuclide i
    BR = branching ratio
    P_i = production rate (from external source)

References:
    - Bateman (1910) Proc. Cambridge Phil. Soc. 15, 423
    - curie package: github.com/jtmorrell/curie
    - ALARA manual for decay chain treatment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg


# Time unit conversions to seconds
TIME_UNITS = {
    'ns': 1e-9,
    'us': 1e-6,
    'ms': 1e-3,
    's': 1.0,
    'min': 60.0,
    'm': 60.0,
    'h': 3600.0,
    'hr': 3600.0,
    'd': 86400.0,
    'day': 86400.0,
    'y': 365.25 * 86400.0,
    'yr': 365.25 * 86400.0,
    'ky': 365.25 * 86400.0 * 1e3,
    'My': 365.25 * 86400.0 * 1e6,
    'Gy': 365.25 * 86400.0 * 1e9,
}


@dataclass
class Nuclide:
    """
    Representation of a nuclide in a decay chain.
    
    Attributes
    ----------
    name : str
        Nuclide name (e.g., 'Co60', 'Mn56')
    half_life_s : float
        Half-life in seconds
    decay_products : dict
        Decay products with branching ratios {product_name: BR}
    """
    name: str
    half_life_s: float
    decay_products: Dict[str, float] = field(default_factory=dict)
    
    @property
    def decay_constant(self) -> float:
        """Decay constant λ = ln(2) / t_half."""
        if self.half_life_s <= 0 or self.half_life_s == float('inf'):
            return 0.0
        return np.log(2) / self.half_life_s
    
    @property
    def is_stable(self) -> bool:
        """Check if nuclide is stable (no decay products or very long half-life)."""
        return len(self.decay_products) == 0 or self.half_life_s > 1e18


@dataclass
class DecayChainResult:
    """
    Result of decay chain calculation.
    
    Attributes
    ----------
    times : np.ndarray
        Time points
    activities : dict
        Activities in Bq for each nuclide {name: activity_array}
    atoms : dict
        Number of atoms for each nuclide {name: atom_array}
    decays : dict, optional
        Cumulative decays for each nuclide
    """
    times: np.ndarray
    activities: Dict[str, np.ndarray]
    atoms: Dict[str, np.ndarray]
    decays: Optional[Dict[str, np.ndarray]] = None
    
    def get_activity(self, nuclide: str, time: Optional[float] = None) -> np.ndarray:
        """Get activity of a nuclide, optionally interpolated to specific time."""
        if nuclide not in self.activities:
            raise ValueError(f"Nuclide '{nuclide}' not in decay chain")
        
        if time is None:
            return self.activities[nuclide]
        
        return np.interp(time, self.times, self.activities[nuclide])


class DecayChain:
    """
    Radioactive decay chain solver using Bateman equations.
    
    Supports:
    - Linear decay chains (A → B → C → ...)
    - Branching decay (parent decays to multiple products)
    - Production during irradiation
    - Analytical solution (for linear chains)
    - Matrix exponential solution (general case)
    
    Parameters
    ----------
    parent : str or Nuclide
        Parent nuclide
    nuclide_data : dict, optional
        Nuclide data {name: {'half_life_s': float, 'decay_products': dict}}
        If not provided, uses simple linear chain with provided half-lives
    
    Examples
    --------
    >>> # Simple Mn-56 decay
    >>> chain = DecayChain('Mn56', nuclide_data={
    ...     'Mn56': {'half_life_s': 9285.6, 'decay_products': {'Fe56': 1.0}}
    ... })
    >>> result = chain.decay(initial_activity={'Mn56': 1000}, times=[0, 3600, 7200])
    >>> result.activities['Mn56']
    """
    
    def __init__(
        self,
        parent: Union[str, Nuclide],
        nuclide_data: Optional[Dict] = None,
        max_chain_length: int = 20
    ):
        self.parent_name = parent if isinstance(parent, str) else parent.name
        self.nuclides: Dict[str, Nuclide] = {}
        self.chain_order: List[str] = []
        self._transition_matrix: Optional[np.ndarray] = None
        
        # Build chain from nuclide data
        if nuclide_data:
            self._build_chain_from_data(nuclide_data, max_chain_length)
        else:
            # Single nuclide with no progeny
            self.nuclides[self.parent_name] = Nuclide(
                name=self.parent_name,
                half_life_s=float('inf'),
                decay_products={}
            )
            self.chain_order = [self.parent_name]
    
    def _build_chain_from_data(
        self,
        nuclide_data: Dict,
        max_length: int
    ) -> None:
        """Build decay chain from nuclide data dictionary."""
        # Start with parent
        if self.parent_name not in nuclide_data:
            raise ValueError(f"Parent nuclide '{self.parent_name}' not in data")
        
        to_process = [self.parent_name]
        processed = set()
        
        while to_process and len(self.chain_order) < max_length:
            current = to_process.pop(0)
            if current in processed:
                continue
            
            if current in nuclide_data:
                data = nuclide_data[current]
                products = data.get('decay_products', {})
                
                self.nuclides[current] = Nuclide(
                    name=current,
                    half_life_s=data.get('half_life_s', float('inf')),
                    decay_products=products
                )
                self.chain_order.append(current)
                
                # Add decay products to processing queue
                for product in products:
                    if product not in processed:
                        to_process.append(product)
            else:
                # Unknown nuclide - assume stable
                self.nuclides[current] = Nuclide(
                    name=current,
                    half_life_s=float('inf'),
                    decay_products={}
                )
                self.chain_order.append(current)
            
            processed.add(current)
    
    def add_nuclide(
        self,
        name: str,
        half_life_s: float,
        decay_products: Optional[Dict[str, float]] = None
    ) -> None:
        """Add a nuclide to the chain."""
        self.nuclides[name] = Nuclide(
            name=name,
            half_life_s=half_life_s,
            decay_products=decay_products or {}
        )
        if name not in self.chain_order:
            self.chain_order.append(name)
        self._transition_matrix = None  # Reset
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Build the transition matrix for the decay chain.
        
        The matrix M satisfies dN/dt = M @ N, where N is the vector of
        atom counts for each nuclide.
        
        M[i,i] = -λ_i (decay out)
        M[j,i] = λ_i × BR_{i→j} (decay in from parent)
        """
        n = len(self.chain_order)
        M = np.zeros((n, n))
        
        for i, name in enumerate(self.chain_order):
            nuclide = self.nuclides[name]
            lam = nuclide.decay_constant
            
            # Decay out
            M[i, i] = -lam
            
            # Decay in from parents
            for parent_name, parent in self.nuclides.items():
                if name in parent.decay_products:
                    j = self.chain_order.index(parent_name)
                    br = parent.decay_products[name]
                    M[i, j] += parent.decay_constant * br
        
        return M
    
    def decay(
        self,
        initial_activity: Optional[Dict[str, float]] = None,
        initial_atoms: Optional[Dict[str, float]] = None,
        times: Union[np.ndarray, List[float], None] = None,
        t_end: Optional[float] = None,
        n_points: int = 100,
        production_rates: Optional[Dict[str, float]] = None,
        units: str = 's'
    ) -> DecayChainResult:
        """
        Calculate decay chain evolution over time.
        
        Parameters
        ----------
        initial_activity : dict, optional
            Initial activities in Bq for each nuclide
        initial_atoms : dict, optional
            Initial number of atoms for each nuclide
            (only one of initial_activity or initial_atoms should be provided)
        times : array, optional
            Time points at which to calculate activities
        t_end : float, optional
            End time (if times not provided)
        n_points : int
            Number of time points (if t_end provided)
        production_rates : dict, optional
            Constant production rates in atoms/s for each nuclide
        units : str
            Time units for input times
        
        Returns
        -------
        DecayChainResult
            Activities and atom counts over time
        """
        # Convert units
        unit_factor = TIME_UNITS.get(units, 1.0)
        
        # Set up time array
        if times is not None:
            times = np.asarray(times) * unit_factor
        elif t_end is not None:
            times = np.linspace(0, t_end * unit_factor, n_points)
        else:
            raise ValueError("Either 'times' or 't_end' must be provided")
        
        # Convert initial conditions to atoms
        n = len(self.chain_order)
        N0 = np.zeros(n)
        
        if initial_atoms is not None:
            for name, count in initial_atoms.items():
                if name in self.chain_order:
                    i = self.chain_order.index(name)
                    N0[i] = count
        elif initial_activity is not None:
            for name, activity in initial_activity.items():
                if name in self.chain_order:
                    i = self.chain_order.index(name)
                    lam = self.nuclides[name].decay_constant
                    if lam > 0:
                        N0[i] = activity / lam  # A = λN → N = A/λ
        
        # Production rates
        P = np.zeros(n)
        if production_rates:
            for name, rate in production_rates.items():
                if name in self.chain_order:
                    i = self.chain_order.index(name)
                    P[i] = rate
        
        # Build transition matrix
        M = self._build_transition_matrix()
        
        # Solve using matrix exponential
        atoms = {}
        activities = {}
        
        for name in self.chain_order:
            atoms[name] = np.zeros(len(times))
            activities[name] = np.zeros(len(times))
        
        for ti, t in enumerate(times):
            if t == 0:
                Nt = N0.copy()
            else:
                # For constant production: N(t) = exp(Mt) N0 + M^(-1)(exp(Mt) - I) P
                expMt = linalg.expm(M * t)
                Nt = expMt @ N0
                
                if np.any(P > 0):
                    try:
                        M_inv = np.linalg.inv(M)
                        Nt += M_inv @ (expMt - np.eye(n)) @ P
                    except np.linalg.LinAlgError:
                        # Fall back to pseudo-inverse
                        M_inv = np.linalg.pinv(M)
                        Nt += M_inv @ (expMt - np.eye(n)) @ P
            
            # Store results
            for j, name in enumerate(self.chain_order):
                atoms[name][ti] = max(0, Nt[j])
                lam = self.nuclides[name].decay_constant
                activities[name][ti] = max(0, Nt[j] * lam)
        
        return DecayChainResult(
            times=times / unit_factor,  # Convert back to original units
            activities=activities,
            atoms=atoms
        )
    
    def activity_at_time(
        self,
        nuclide: str,
        time: float,
        initial_activity: Optional[Dict[str, float]] = None,
        initial_atoms: Optional[Dict[str, float]] = None,
        production_rates: Optional[Dict[str, float]] = None,
        units: str = 's'
    ) -> float:
        """
        Calculate activity of a nuclide at a specific time.
        
        This is more efficient than calling decay() for a single time point.
        """
        result = self.decay(
            initial_activity=initial_activity,
            initial_atoms=initial_atoms,
            times=[time],
            production_rates=production_rates,
            units=units
        )
        return result.activities[nuclide][0]
    
    def saturation_activity(
        self,
        nuclide: str,
        production_rate: float
    ) -> float:
        """
        Calculate saturation activity for constant production.
        
        For a single nuclide with production rate P:
            A_sat = P (at equilibrium)
        
        For a chain, the saturation activity depends on the chain position.
        """
        lam = self.nuclides[nuclide].decay_constant
        if lam <= 0:
            return 0.0
        
        # Simple case: direct production of this nuclide
        return production_rate
    
    def decays_in_interval(
        self,
        nuclide: str,
        t_start: float,
        t_stop: float,
        initial_activity: Optional[Dict[str, float]] = None,
        units: str = 's'
    ) -> float:
        """
        Calculate number of decays in a time interval.
        
        Integrates A(t) from t_start to t_stop.
        """
        # Use trapezoidal integration
        n_points = 100
        times = np.linspace(t_start, t_stop, n_points)
        result = self.decay(
            initial_activity=initial_activity,
            times=times,
            units=units
        )
        
        return np.trapz(result.activities[nuclide], times)


# =============================================================================
# Convenience Functions
# =============================================================================


def simple_decay(
    initial_activity: float,
    half_life_s: float,
    times: np.ndarray
) -> np.ndarray:
    """
    Simple exponential decay (single nuclide).
    
    A(t) = A_0 × exp(-λt)
    
    Parameters
    ----------
    initial_activity : float
        Initial activity (Bq)
    half_life_s : float
        Half-life in seconds
    times : np.ndarray
        Time points
    
    Returns
    -------
    np.ndarray
        Activity at each time point
    """
    lam = np.log(2) / half_life_s
    return initial_activity * np.exp(-lam * times)


def irradiation_saturation_factor(
    decay_constant: float,
    irradiation_time: float
) -> float:
    """
    Saturation factor S = 1 - exp(-λ × t_irr).
    
    Used in activation analysis for the build-up during irradiation.
    """
    if decay_constant <= 0:
        return 0.0
    return 1.0 - np.exp(-decay_constant * irradiation_time)


def cooling_factor(
    decay_constant: float,
    cooling_time: float
) -> float:
    """
    Cooling factor D = exp(-λ × t_cool).
    
    Used in activation analysis for decay during cooling.
    """
    return np.exp(-decay_constant * cooling_time)


def counting_factor(
    decay_constant: float,
    counting_time: float
) -> float:
    """
    Counting factor C = (1 - exp(-λ × t_count)) / (λ × t_count).
    
    Accounts for decay during counting period.
    """
    if decay_constant * counting_time < 1e-6:
        # Small argument approximation
        return 1.0 - 0.5 * decay_constant * counting_time
    
    lam_t = decay_constant * counting_time
    return (1.0 - np.exp(-lam_t)) / lam_t


def activity_from_irradiation(
    reaction_rate: float,
    half_life_s: float,
    t_irradiation: float,
    t_cooling: float = 0.0,
    t_counting: float = 0.0
) -> float:
    """
    Calculate activity after irradiation, cooling, and counting.
    
    A = R × S × D × C
    
    where:
        R = reaction rate (atoms/s)
        S = saturation factor
        D = cooling factor
        C = counting correction
    
    Parameters
    ----------
    reaction_rate : float
        Reaction rate during irradiation (atoms/s)
    half_life_s : float
        Product half-life in seconds
    t_irradiation : float
        Irradiation time in seconds
    t_cooling : float
        Cooling time in seconds
    t_counting : float
        Counting time in seconds (if 0, returns instantaneous activity)
    
    Returns
    -------
    float
        Activity in Bq (or average activity during counting if t_counting > 0)
    """
    lam = np.log(2) / half_life_s
    
    S = irradiation_saturation_factor(lam, t_irradiation)
    D = cooling_factor(lam, t_cooling)
    
    if t_counting > 0:
        C = counting_factor(lam, t_counting)
    else:
        C = 1.0
    
    return reaction_rate * S * D * C
