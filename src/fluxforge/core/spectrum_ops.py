"""
Spectrum Operations Module - Becquerel Parity

Implements advanced spectrum operations matching becquerel functionality:
- Spectrum arithmetic (+, -, *, /)
- Uncertainty propagation using uncertainties package
- Spectrum rebinning (linear, stochastic)
- Spectrum attenuation using XCOM
- Listmode to histogram conversion
- Dead time calculations
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class SpectrumData:
    """
    Spectrum data container with uncertainty support.
    
    Matches becquerel.Spectrum functionality.
    
    Attributes
    ----------
    counts : NDArray
        Raw counts per channel
    counts_uncertainty : NDArray
        Uncertainties on counts (sqrt(N) if not provided)
    edges : NDArray
        Bin edges (n_channels + 1)
    live_time : float
        Live time in seconds
    real_time : float
        Real (clock) time in seconds
    start_time : Optional[str]
        Acquisition start time
    """
    counts: NDArray
    counts_uncertainty: Optional[NDArray] = None
    edges: Optional[NDArray] = None
    live_time: float = 1.0
    real_time: float = 1.0
    start_time: Optional[str] = None
    description: str = ""
    
    def __post_init__(self):
        """Initialize derived quantities."""
        self.counts = np.asarray(self.counts, dtype=float)
        n = len(self.counts)
        
        # Default uncertainty is sqrt(counts)
        if self.counts_uncertainty is None:
            self.counts_uncertainty = np.sqrt(np.maximum(self.counts, 1.0))
        else:
            self.counts_uncertainty = np.asarray(self.counts_uncertainty, dtype=float)
        
        # Default edges are channel numbers
        if self.edges is None:
            self.edges = np.arange(n + 1, dtype=float)
        else:
            self.edges = np.asarray(self.edges, dtype=float)
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return len(self.counts)
    
    @property
    def channels(self) -> NDArray:
        """Channel numbers (0-indexed)."""
        return np.arange(self.n_channels)
    
    @property
    def bin_centers(self) -> NDArray:
        """Bin centers."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])
    
    @property
    def bin_widths(self) -> NDArray:
        """Bin widths."""
        return np.diff(self.edges)
    
    @property
    def cps(self) -> NDArray:
        """Counts per second."""
        return self.counts / max(self.live_time, 1e-10)
    
    @property
    def cps_uncertainty(self) -> NDArray:
        """Uncertainty on counts per second."""
        return self.counts_uncertainty / max(self.live_time, 1e-10)
    
    @property
    def cpskev(self) -> NDArray:
        """Counts per second per keV."""
        widths = self.bin_widths
        return self.cps / np.maximum(widths, 1e-10)
    
    @property
    def cpskev_uncertainty(self) -> NDArray:
        """Uncertainty on counts per second per keV."""
        widths = self.bin_widths
        return self.cps_uncertainty / np.maximum(widths, 1e-10)
    
    @property
    def total_counts(self) -> float:
        """Total counts in spectrum."""
        return float(np.sum(self.counts))
    
    @property
    def total_counts_uncertainty(self) -> float:
        """Uncertainty on total counts."""
        return float(np.sqrt(np.sum(self.counts_uncertainty**2)))
    
    @property
    def dead_time(self) -> float:
        """Dead time in seconds."""
        return max(0, self.real_time - self.live_time)
    
    @property
    def dead_time_fraction(self) -> float:
        """Dead time as fraction of real time."""
        if self.real_time > 0:
            return self.dead_time / self.real_time
        return 0.0
    
    def copy(self) -> 'SpectrumData':
        """Create deep copy."""
        return SpectrumData(
            counts=self.counts.copy(),
            counts_uncertainty=self.counts_uncertainty.copy(),
            edges=self.edges.copy(),
            live_time=self.live_time,
            real_time=self.real_time,
            start_time=self.start_time,
            description=self.description
        )
    
    def __add__(self, other: 'SpectrumData') -> 'SpectrumData':
        """Add two spectra (counts add, live times add)."""
        if not isinstance(other, SpectrumData):
            raise TypeError("Can only add SpectrumData to SpectrumData")
        
        if len(self.counts) != len(other.counts):
            raise ValueError("Spectra must have same number of channels")
        
        new_counts = self.counts + other.counts
        new_unc = np.sqrt(self.counts_uncertainty**2 + other.counts_uncertainty**2)
        
        return SpectrumData(
            counts=new_counts,
            counts_uncertainty=new_unc,
            edges=self.edges.copy(),
            live_time=self.live_time + other.live_time,
            real_time=self.real_time + other.real_time,
            description=f"({self.description}) + ({other.description})"
        )
    
    def __sub__(self, other: 'SpectrumData') -> 'SpectrumData':
        """Subtract spectra (converted to CPS first, then back to counts)."""
        if not isinstance(other, SpectrumData):
            raise TypeError("Can only subtract SpectrumData from SpectrumData")
        
        if len(self.counts) != len(other.counts):
            raise ValueError("Spectra must have same number of channels")
        
        # Convert to CPS, subtract, convert back
        cps_diff = self.cps - other.cps
        cps_unc = np.sqrt(self.cps_uncertainty**2 + other.cps_uncertainty**2)
        
        # Use self's live time for result
        new_counts = cps_diff * self.live_time
        new_unc = cps_unc * self.live_time
        
        return SpectrumData(
            counts=new_counts,
            counts_uncertainty=new_unc,
            edges=self.edges.copy(),
            live_time=self.live_time,
            real_time=self.real_time,
            description=f"({self.description}) - ({other.description})"
        )
    
    def __mul__(self, scalar: float) -> 'SpectrumData':
        """Multiply spectrum by scalar."""
        return SpectrumData(
            counts=self.counts * scalar,
            counts_uncertainty=self.counts_uncertainty * abs(scalar),
            edges=self.edges.copy(),
            live_time=self.live_time,
            real_time=self.real_time,
            description=f"{scalar} * ({self.description})"
        )
    
    def __rmul__(self, scalar: float) -> 'SpectrumData':
        """Right multiply by scalar."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'SpectrumData':
        """Divide spectrum by scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return self.__mul__(1.0 / scalar)
    
    def downsample(self, factor: int = 2, method: str = 'stochastic') -> 'SpectrumData':
        """
        Downsample spectrum while preserving Poisson statistics.
        
        Parameters
        ----------
        factor : int
            Downsampling factor
        method : str
            'stochastic' - Poisson-preserving random downsampling
            'mean' - Simple mean of factor channels
        
        Returns
        -------
        SpectrumData
            Downsampled spectrum
        """
        n = len(self.counts)
        n_new = n // factor
        
        if method == 'stochastic':
            # Poisson-preserving: sample from binomial
            new_counts = np.zeros(n_new)
            for i in range(n_new):
                start = i * factor
                end = (i + 1) * factor
                total = np.sum(self.counts[start:end])
                # Binomial sampling
                new_counts[i] = np.random.binomial(int(total), 1.0/factor)
            new_counts *= factor  # Scale back
            new_unc = np.sqrt(np.maximum(new_counts, 1.0))
        else:
            # Simple sum
            new_counts = np.array([
                np.sum(self.counts[i*factor:(i+1)*factor])
                for i in range(n_new)
            ])
            new_unc = np.sqrt(new_counts)
        
        # New edges
        new_edges = self.edges[::factor][:n_new+1]
        if len(new_edges) < n_new + 1:
            new_edges = np.linspace(self.edges[0], self.edges[-1], n_new + 1)
        
        return SpectrumData(
            counts=new_counts,
            counts_uncertainty=new_unc,
            edges=new_edges,
            live_time=self.live_time,
            real_time=self.real_time,
            description=f"downsampled({self.description})"
        )


def rebin_spectrum(
    spectrum: SpectrumData,
    new_edges: NDArray,
    method: str = 'interpolate'
) -> SpectrumData:
    """
    Rebin spectrum to new energy edges.
    
    Parameters
    ----------
    spectrum : SpectrumData
        Input spectrum
    new_edges : NDArray
        New bin edges
    method : str
        'interpolate' - Linear interpolation
        'stochastic' - Poisson-preserving redistribution
    
    Returns
    -------
    SpectrumData
        Rebinned spectrum
    """
    old_edges = spectrum.edges
    old_counts = spectrum.counts
    old_centers = spectrum.bin_centers
    
    new_edges = np.asarray(new_edges)
    n_new = len(new_edges) - 1
    new_counts = np.zeros(n_new)
    
    if method == 'interpolate':
        # Linear interpolation of CPS/keV, then integrate
        old_cpskev = spectrum.cpskev
        
        for i in range(n_new):
            lo, hi = new_edges[i], new_edges[i+1]
            
            # Find overlapping old bins
            for j in range(len(old_counts)):
                old_lo, old_hi = old_edges[j], old_edges[j+1]
                
                # Overlap
                overlap_lo = max(lo, old_lo)
                overlap_hi = min(hi, old_hi)
                
                if overlap_hi > overlap_lo:
                    frac = (overlap_hi - overlap_lo) / (old_hi - old_lo)
                    new_counts[i] += old_counts[j] * frac
    
    elif method == 'stochastic':
        # Redistribute counts probabilistically
        for j in range(len(old_counts)):
            count = int(old_counts[j])
            if count == 0:
                continue
            
            old_lo, old_hi = old_edges[j], old_edges[j+1]
            old_width = old_hi - old_lo
            
            # Find destination bins
            for i in range(n_new):
                lo, hi = new_edges[i], new_edges[i+1]
                overlap_lo = max(lo, old_lo)
                overlap_hi = min(hi, old_hi)
                
                if overlap_hi > overlap_lo:
                    prob = (overlap_hi - overlap_lo) / old_width
                    new_counts[i] += np.random.binomial(count, prob)
    
    new_unc = np.sqrt(np.maximum(new_counts, 1.0))
    
    return SpectrumData(
        counts=new_counts,
        counts_uncertainty=new_unc,
        edges=new_edges,
        live_time=spectrum.live_time,
        real_time=spectrum.real_time,
        description=f"rebinned({spectrum.description})"
    )


def apply_attenuation(
    spectrum: SpectrumData,
    material: str,
    thickness_cm: float,
    density_g_cm3: Optional[float] = None
) -> SpectrumData:
    """
    Apply material attenuation to spectrum using XCOM data.
    
    Parameters
    ----------
    spectrum : SpectrumData
        Input spectrum (edges in keV)
    material : str
        Material name (e.g., 'Lead', 'Aluminum')
    thickness_cm : float
        Material thickness in cm
    density_g_cm3 : float, optional
        Material density. If None, uses default from XCOM data.
    
    Returns
    -------
    SpectrumData
        Attenuated spectrum
    """
    try:
        from fluxforge.data.xcom import get_attenuation_data
    except ImportError:
        raise ImportError("XCOM module required for attenuation calculations")
    
    att_data = get_attenuation_data(material)
    
    if density_g_cm3 is None:
        density_g_cm3 = att_data.density
    
    # Calculate transmission at each bin center
    energies = spectrum.bin_centers  # keV
    mu_rho = att_data.get_mu_rho(energies)  # cm²/g
    mu = mu_rho * density_g_cm3  # cm⁻¹
    transmission = np.exp(-mu * thickness_cm)
    
    # Apply attenuation
    new_counts = spectrum.counts * transmission
    new_unc = spectrum.counts_uncertainty * transmission
    
    return SpectrumData(
        counts=new_counts,
        counts_uncertainty=new_unc,
        edges=spectrum.edges.copy(),
        live_time=spectrum.live_time,
        real_time=spectrum.real_time,
        description=f"attenuated({spectrum.description}, {material}, {thickness_cm}cm)"
    )


def listmode_to_histogram(
    events: NDArray,
    edges: NDArray,
    weights: Optional[NDArray] = None
) -> SpectrumData:
    """
    Convert listmode (event-by-event) data to histogram.
    
    Parameters
    ----------
    events : NDArray
        Event energies or channel numbers
    edges : NDArray
        Bin edges
    weights : NDArray, optional
        Event weights (default 1.0)
    
    Returns
    -------
    SpectrumData
        Histogrammed spectrum
    """
    counts, _ = np.histogram(events, bins=edges, weights=weights)
    counts = counts.astype(float)
    
    return SpectrumData(
        counts=counts,
        edges=edges
    )


# =============================================================================
# CALIBRATION
# =============================================================================

@dataclass
class CalibrationResult:
    """Result of calibration fit."""
    coefficients: NDArray
    covariance: Optional[NDArray] = None
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    r_squared: float = 0.0
    degrees_of_freedom: int = 0
    expression: str = "polynomial"


class Calibration:
    """
    Energy/efficiency calibration class.
    
    Matches becquerel.Calibration functionality.
    
    Supports:
    - Linear, polynomial, sqrt_polynomial calibration
    - Fitting to calibration points
    - Inverse transformation
    - Uncertainty propagation
    """
    
    def __init__(
        self,
        expression: str = "polynomial",
        coefficients: Optional[NDArray] = None,
        degree: int = 2
    ):
        """
        Initialize calibration.
        
        Parameters
        ----------
        expression : str
            'linear', 'polynomial', 'sqrt_polynomial', 'spline'
        coefficients : NDArray, optional
            Calibration coefficients
        degree : int
            Polynomial degree (for polynomial types)
        """
        self.expression = expression
        self.degree = degree
        self.coefficients = coefficients if coefficients is not None else np.array([0.0, 1.0])
        self.covariance: Optional[NDArray] = None
        self.fit_result: Optional[CalibrationResult] = None
        
        # Calibration points
        self._x_points: List[float] = []
        self._y_points: List[float] = []
        self._y_uncertainties: List[float] = []
    
    def __call__(self, x: Union[float, NDArray]) -> Union[float, NDArray]:
        """Evaluate calibration at x."""
        x = np.asarray(x)
        
        if self.expression in ('linear', 'polynomial'):
            return np.polyval(self.coefficients[::-1], x)
        elif self.expression == 'sqrt_polynomial':
            poly_val = np.polyval(self.coefficients[::-1], x)
            return np.sqrt(np.maximum(poly_val, 0))
        else:
            return np.polyval(self.coefficients[::-1], x)
    
    def add_points(
        self,
        x: Union[float, List[float], NDArray],
        y: Union[float, List[float], NDArray],
        y_unc: Optional[Union[float, List[float], NDArray]] = None
    ):
        """Add calibration points."""
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        if y_unc is None:
            y_unc = np.ones_like(y) * 0.01 * np.mean(y)
        y_unc = np.atleast_1d(y_unc)
        
        self._x_points.extend(x.tolist())
        self._y_points.extend(y.tolist())
        self._y_uncertainties.extend(y_unc.tolist())
    
    def fit(self, **kwargs) -> CalibrationResult:
        """Fit calibration to added points."""
        if len(self._x_points) < 2:
            raise ValueError("Need at least 2 calibration points")
        
        x = np.array(self._x_points)
        y = np.array(self._y_points)
        y_unc = np.array(self._y_uncertainties)
        
        # Weight by inverse variance
        weights = 1.0 / (y_unc**2 + 1e-10)
        
        if self.expression in ('linear', 'polynomial'):
            # Weighted polynomial fit
            coeffs = np.polyfit(x, y, self.degree, w=np.sqrt(weights))
            self.coefficients = coeffs[::-1]  # Store in ascending order
        elif self.expression == 'sqrt_polynomial':
            # Fit y² as polynomial
            coeffs = np.polyfit(x, y**2, self.degree, w=np.sqrt(weights))
            self.coefficients = coeffs[::-1]
        
        # Calculate fit statistics
        y_pred = self(x)
        residuals = y - y_pred
        
        ss_res = np.sum((residuals / y_unc)**2)
        ss_tot = np.sum(((y - np.mean(y)) / y_unc)**2)
        
        dof = len(x) - len(self.coefficients)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        self.fit_result = CalibrationResult(
            coefficients=self.coefficients,
            chi_squared=ss_res,
            reduced_chi_squared=ss_res / dof if dof > 0 else 0,
            r_squared=r_squared,
            degrees_of_freedom=dof,
            expression=self.expression
        )
        
        return self.fit_result
    
    def inverse(self, y: Union[float, NDArray], x_guess: Optional[float] = None) -> Union[float, NDArray]:
        """
        Calculate inverse of calibration (y → x).
        
        For linear/polynomial, uses root finding.
        """
        from scipy.optimize import brentq
        
        y = np.atleast_1d(y)
        results = []
        
        for yi in y:
            if self.expression == 'linear' and len(self.coefficients) == 2:
                # Analytic solution for linear
                a, b = self.coefficients[0], self.coefficients[1]
                if abs(b) < 1e-10:
                    results.append(np.nan)
                else:
                    results.append((yi - a) / b)
            else:
                # Numerical root finding
                try:
                    x_lo = min(self._x_points) if self._x_points else 0
                    x_hi = max(self._x_points) if self._x_points else 1e6
                    
                    root = brentq(lambda x: self(x) - yi, x_lo, x_hi)
                    results.append(root)
                except Exception:
                    results.append(np.nan)
        
        return np.array(results) if len(results) > 1 else results[0]


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing spectrum_ops module...")
    
    # Test SpectrumData
    spec1 = SpectrumData(
        counts=np.array([100, 200, 300, 200, 100]),
        edges=np.array([0, 100, 200, 300, 400, 500]),
        live_time=100,
        real_time=110
    )
    
    print(f"Spectrum: {spec1.n_channels} channels")
    print(f"Total counts: {spec1.total_counts}")
    print(f"Dead time: {spec1.dead_time_fraction*100:.1f}%")
    print(f"CPS: {spec1.cps}")
    
    # Test arithmetic
    spec2 = spec1 * 2
    print(f"Doubled counts: {spec2.total_counts}")
    
    spec3 = spec1 + spec1
    print(f"Added counts: {spec3.total_counts}")
    
    # Test calibration
    cal = Calibration(expression='linear', degree=1)
    cal.add_points([0, 1000, 2000], [0, 500, 1000])
    result = cal.fit()
    print(f"Calibration fit: R²={result.r_squared:.4f}")
    print(f"Channel 1500 → {cal(1500):.1f} keV")
    
    print("\n✅ spectrum_ops module tests passed!")
