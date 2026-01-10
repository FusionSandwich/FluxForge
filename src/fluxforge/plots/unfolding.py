"""
Spectrum Unfolding Visualization Module

Provides publication-quality plots for neutron spectrum unfolding results,
including:
- Fig 10 style: Spectrum comparison with uncertainty bands
- Fig 12 style: Ratio plots (unfolded/reference)
- Cross-section response visualization
- Convergence diagnostics

References:
- Similar visualization approach to Reginatto et al., UMG Package documentation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import from workflow module
try:
    from fluxforge.workflows.spectrum_unfolding import UnfoldingResult
except ImportError:
    UnfoldingResult = None


# =============================================================================
# Plot Style Configuration
# =============================================================================

PLOT_STYLE = {
    "figure.figsize": (10, 7),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def apply_plot_style():
    """Apply publication-quality plot style."""
    if HAS_MATPLOTLIB:
        plt.rcParams.update(PLOT_STYLE)


# =============================================================================
# Color Schemes
# =============================================================================

COLORS = {
    "mcnp": "#1f77b4",       # Blue
    "unfolded": "#d62728",   # Red
    "gravel": "#ff7f0e",     # Orange
    "mlem": "#2ca02c",       # Green
    "uncertainty": "#9467bd", # Purple
    "ratio_line": "#17becf",  # Cyan
}

FILL_ALPHA = 0.25


# =============================================================================
# Spectrum Comparison Plots (Fig 10 Style)
# =============================================================================

def plot_spectrum_comparison(
    result: "UnfoldingResult",
    reference_flux: Optional[np.ndarray] = None,
    reference_label: str = "MCNP Reference",
    title: str = "Neutron Spectrum Comparison",
    xlabel: str = "Energy (MeV)",
    ylabel: str = "Flux per unit lethargy (n/cm²/s)",
    show_uncertainty: bool = True,
    log_x: bool = True,
    log_y: bool = True,
    energy_units: str = "MeV",
    lethargy_plot: bool = True,
    figsize: Tuple[float, float] = (10, 7),
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Create Fig 10 style spectrum comparison plot.
    
    Shows unfolded spectrum with uncertainty bands compared to reference.
    
    Parameters
    ----------
    result : UnfoldingResult
        Unfolding result object
    reference_flux : np.ndarray, optional
        Reference flux (e.g., MCNP) to compare against
    reference_label : str
        Label for reference spectrum
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    show_uncertainty : bool
        Show uncertainty bands
    log_x, log_y : bool
        Use logarithmic axes
    energy_units : str
        'MeV' or 'eV'
    lethargy_plot : bool
        Plot as E*φ(E) (flux per unit lethargy)
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    apply_plot_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get energy midpoints
    energies = result.energy_midpoints.copy()
    if energy_units.lower() == "mev":
        energies = energies / 1e6
        energy_factor = 1e6
    else:
        energy_factor = 1.0
    
    # Prepare flux for plotting
    flux = result.flux.copy()
    flux_unc = result.flux_uncertainty.copy() if len(result.flux_uncertainty) > 0 else np.zeros_like(flux)
    
    if lethargy_plot:
        # E * φ(E) representation
        flux = result.energy_midpoints * flux
        flux_unc = result.energy_midpoints * flux_unc
        if reference_flux is not None:
            reference_flux = result.energy_midpoints * reference_flux
    
    # Plot unfolded spectrum with uncertainty band
    if show_uncertainty and np.any(flux_unc > 0):
        ax.fill_between(
            energies,
            np.maximum(flux - flux_unc, 1e-30),
            flux + flux_unc,
            color=COLORS["unfolded"],
            alpha=FILL_ALPHA,
            label=f"±1σ uncertainty"
        )
    
    ax.plot(
        energies, flux,
        color=COLORS["unfolded"],
        linewidth=2,
        label=f"Unfolded ({result.method})"
    )
    
    # Plot reference spectrum if provided
    if reference_flux is not None:
        ax.plot(
            energies, reference_flux,
            color=COLORS["mcnp"],
            linewidth=2,
            linestyle="--",
            label=reference_label
        )
    
    # Formatting
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    
    # Add statistics annotation
    stats_text = (
        f"χ²/dof = {result.chi_squared:.3f}\n"
        f"Iterations: {result.iterations}\n"
        f"Converged: {result.converged}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def plot_spectrum_uncertainty_bands(
    result: "UnfoldingResult",
    confidence_levels: List[float] = [0.68, 0.95],
    title: str = "Unfolded Neutron Spectrum with Uncertainty Bands",
    xlabel: str = "Energy (MeV)",
    ylabel: str = "Flux per unit lethargy (n/cm²/s)",
    log_x: bool = True,
    log_y: bool = True,
    energy_units: str = "MeV",
    lethargy_plot: bool = True,
    colors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 7),
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot spectrum with multiple confidence level uncertainty bands.
    
    Creates publication-quality plot with:
    - Central estimate (solid line)
    - 1-sigma (68%) confidence band 
    - 2-sigma (95%) confidence band
    
    This implements capability G1.1 for spectrum uncertainty visualization.
    
    Parameters
    ----------
    result : UnfoldingResult
        Unfolding result object with flux and uncertainty
    confidence_levels : List[float]
        Confidence levels for bands (default: 68%, 95%)
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    log_x, log_y : bool
        Use logarithmic axes
    energy_units : str
        'MeV' or 'eV'
    lethargy_plot : bool
        Plot as E*φ(E) (flux per unit lethargy)
    colors : List[str], optional
        Colors for confidence bands (from outer to inner)
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    
    Examples
    --------
    >>> result = spectrum_unfolder.unfold(method='gravel')
    >>> fig, ax = plot_spectrum_uncertainty_bands(
    ...     result,
    ...     confidence_levels=[0.68, 0.95, 0.99],
    ...     save_path="spectrum_uncertainty.png"
    ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    from scipy import stats
    
    apply_plot_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Default colors: outer bands lighter
    if colors is None:
        colors = ["#ffcccc", "#ff9999", "#ff6666"]  # Light to dark red
    
    # Get energy midpoints
    energies = result.energy_midpoints.copy()
    if energy_units.lower() == "mev":
        energies = energies / 1e6
    
    # Prepare flux for plotting
    flux = result.flux.copy()
    flux_unc = result.flux_uncertainty.copy() if len(result.flux_uncertainty) > 0 else np.zeros_like(flux)
    
    if lethargy_plot:
        # E * φ(E) representation
        flux = result.energy_midpoints * flux
        flux_unc = result.energy_midpoints * flux_unc
    
    # Sort confidence levels from largest to smallest (outer bands first)
    sorted_levels = sorted(confidence_levels, reverse=True)
    
    # Plot uncertainty bands from outer to inner
    for i, cl in enumerate(sorted_levels):
        # Convert confidence level to z-score (number of sigmas)
        z = stats.norm.ppf((1 + cl) / 2)
        
        color = colors[i % len(colors)]
        label = f"{cl*100:.0f}% confidence"
        
        lower = np.maximum(flux - z * flux_unc, 1e-30)
        upper = flux + z * flux_unc
        
        ax.fill_between(
            energies, lower, upper,
            color=color,
            alpha=0.6,
            label=label,
            edgecolor="none"
        )
    
    # Plot central estimate
    ax.plot(
        energies, flux,
        color="#d62728",  # Red
        linewidth=2,
        label=f"Best estimate ({result.method})"
    )
    
    # Formatting
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    
    # Add statistics annotation
    stats_text = (
        f"Method: {result.method}\n"
        f"χ²/dof = {result.chi_squared:.3f}\n"
        f"Iterations: {result.iterations}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def plot_spectrum_ratio(
    result: "UnfoldingResult",
    reference_flux: np.ndarray,
    title: str = "Unfolded / Reference Ratio",
    xlabel: str = "Energy (MeV)",
    ylabel: str = "Ratio (Unfolded / Reference)",
    show_uncertainty: bool = True,
    log_x: bool = True,
    energy_units: str = "MeV",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Create Fig 12 style ratio plot.
    
    Shows ratio of unfolded to reference spectrum with uncertainty bands.
    
    Parameters
    ----------
    result : UnfoldingResult
        Unfolding result object
    reference_flux : np.ndarray
        Reference flux to compare against
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    show_uncertainty : bool
        Show uncertainty bands
    log_x : bool
        Use logarithmic x-axis
    energy_units : str
        'MeV' or 'eV'
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    apply_plot_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get energy midpoints
    energies = result.energy_midpoints.copy()
    if energy_units.lower() == "mev":
        energies = energies / 1e6
    
    # Calculate ratio
    flux = result.flux.copy()
    flux_unc = result.flux_uncertainty.copy() if len(result.flux_uncertainty) > 0 else np.zeros_like(flux)
    
    # Avoid division by zero
    ref_safe = np.where(reference_flux > 1e-30, reference_flux, 1e-30)
    ratio = flux / ref_safe
    
    # Propagate uncertainty to ratio
    ratio_unc = flux_unc / ref_safe
    
    # Plot unity line
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    
    # Add ±10% and ±20% bands
    ax.axhline(y=1.1, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(y=1.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    
    # Plot ratio with uncertainty band
    if show_uncertainty and np.any(ratio_unc > 0):
        ax.fill_between(
            energies,
            ratio - ratio_unc,
            ratio + ratio_unc,
            color=COLORS["ratio_line"],
            alpha=FILL_ALPHA,
            label="±1σ uncertainty"
        )
    
    ax.plot(
        energies, ratio,
        color=COLORS["ratio_line"],
        linewidth=2,
        label=f"Ratio ({result.method})"
    )
    
    # Formatting
    if log_x:
        ax.set_xscale("log")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.5, 1.5)  # Focus on ±50% range
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")
    
    # Add mean ratio annotation
    valid_mask = (reference_flux > 1e-20) & (flux > 1e-20)
    if np.any(valid_mask):
        mean_ratio = np.mean(ratio[valid_mask])
        std_ratio = np.std(ratio[valid_mask])
        stats_text = f"Mean ratio: {mean_ratio:.3f} ± {std_ratio:.3f}"
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def plot_spectrum_with_ratio(
    result: "UnfoldingResult",
    reference_flux: np.ndarray,
    reference_label: str = "MCNP Reference",
    title: str = "Neutron Spectrum Unfolding",
    energy_units: str = "MeV",
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create combined plot with spectrum and ratio (stacked).
    
    Like Fig 10 + Fig 12 combined in one figure.
    
    Parameters
    ----------
    result : UnfoldingResult
        Unfolding result object
    reference_flux : np.ndarray
        Reference flux to compare against
    reference_label : str
        Label for reference spectrum
    title : str
        Overall figure title
    energy_units : str
        'MeV' or 'eV'
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    
    Returns
    -------
    fig, (ax1, ax2)
        Matplotlib figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    apply_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        height_ratios=[2, 1],
        sharex=True
    )
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Top: Spectrum comparison
    plot_spectrum_comparison(
        result, reference_flux, reference_label,
        title="",
        xlabel="",
        energy_units=energy_units,
        ax=ax1,
    )
    
    # Bottom: Ratio plot
    plot_spectrum_ratio(
        result, reference_flux,
        title="",
        energy_units=energy_units,
        ax=ax2,
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, (ax1, ax2)


# =============================================================================
# Response Matrix Visualization
# =============================================================================

def plot_response_matrix(
    response_matrix: np.ndarray,
    energy_edges: np.ndarray,
    reactions: List[str],
    title: str = "Response Matrix (IRDFF-II Cross Sections)",
    energy_units: str = "MeV",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Visualize the response matrix.
    
    Parameters
    ----------
    response_matrix : np.ndarray
        Response matrix (n_reactions x n_groups)
    energy_edges : np.ndarray
        Energy group edges
    reactions : List[str]
        Reaction names
    title : str
        Plot title
    energy_units : str
        'MeV' or 'eV'
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    apply_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Energy midpoints
    energies = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    if energy_units.lower() == "mev":
        energies = energies / 1e6
    
    # Plot each reaction's cross section
    for i, rxn in enumerate(reactions):
        xs = response_matrix[i, :]
        ax.semilogy(energies, xs, label=rxn, linewidth=1.5)
    
    ax.set_xscale("log")
    ax.set_xlabel(f"Energy ({energy_units})")
    ax.set_ylabel("Cross Section (barn)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def plot_cross_section_comparison(
    reactions: List[str],
    energy_range: Tuple[float, float] = (1e-5, 20e6),
    n_points: int = 500,
    title: str = "IRDFF-II Cross Sections",
    energy_units: str = "MeV",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Plot cross sections for multiple reactions from IRDFF-II.
    
    Parameters
    ----------
    reactions : List[str]
        Reaction identifiers
    energy_range : tuple
        Energy range (eV)
    n_points : int
        Number of energy points
    title : str
        Plot title
    energy_units : str
        'MeV' or 'eV'
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    from fluxforge.data.irdff import IRDFFDatabase
    
    apply_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    db = IRDFFDatabase()
    energies_eV = np.geomspace(energy_range[0], energy_range[1], n_points)
    
    if energy_units.lower() == "mev":
        energies_plot = energies_eV / 1e6
    else:
        energies_plot = energies_eV
    
    for rxn in reactions:
        xs = db.get_cross_section(rxn)
        if xs is not None:
            sigma = xs.evaluate(energies_eV)
            ax.loglog(energies_plot, sigma, label=rxn, linewidth=1.5)
    
    ax.set_xlabel(f"Energy ({energy_units})")
    ax.set_ylabel("Cross Section (barn)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


# =============================================================================
# Convergence Diagnostics
# =============================================================================

def plot_convergence(
    result: "UnfoldingResult",
    title: str = "Unfolding Convergence",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Plot convergence diagnostics.
    
    Parameters
    ----------
    result : UnfoldingResult
        Unfolding result with chi2_history in metadata
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    apply_plot_style()
    
    chi2_history = result.metadata.get("chi2_history", [])
    if len(chi2_history) == 0:
        print("No convergence history available")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = np.arange(1, len(chi2_history) + 1)
    
    ax.semilogy(iterations, chi2_history, "b-", linewidth=2, label="χ²/dof")
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=1.5, label="χ²/dof = 1")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("χ²/dof")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # Annotate final value
    final_chi2 = chi2_history[-1]
    ax.annotate(
        f"Final: {final_chi2:.4f}",
        xy=(len(chi2_history), final_chi2),
        xytext=(len(chi2_history) * 0.8, final_chi2 * 2),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def plot_measured_vs_predicted(
    result: "UnfoldingResult",
    title: str = "Measured vs Predicted Reaction Rates",
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Plot measured vs predicted reaction rates.
    
    Parameters
    ----------
    result : UnfoldingResult
        Unfolding result
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Save figure to path
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    apply_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    measured = result.measured_rates
    predicted = result.predicted_rates
    
    # 1:1 line
    min_val = min(measured.min(), predicted.min()) * 0.5
    max_val = max(measured.max(), predicted.max()) * 2
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="1:1 line")
    
    # Data points
    ax.scatter(measured, predicted, s=80, c=COLORS["unfolded"], edgecolors="black", alpha=0.8)
    
    # Label points with reaction names
    for i, rxn in enumerate(result.reactions_used):
        # Shorten reaction name for label
        short_name = rxn.split("(")[0]
        ax.annotate(
            short_name,
            xy=(measured[i], predicted[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured Rate")
    ax.set_ylabel("Predicted Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    
    # Add χ² annotation
    ax.text(
        0.02, 0.98, f"χ²/dof = {result.chi_squared:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax
