"""
Activation Analysis Plotting Module

Provides publication-quality plots for comparing experimental measurements
with ALARA activation simulations, specifically designed for thesis figures.

Features:
- Experimental vs simulation bar charts with error bars
- Decay curve comparisons with half-life annotations  
- Ratio plots (C/E) with uncertainty propagation
- Multi-material comparison panels
- Time-series activity evolution
- Cd-ratio analysis visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Plot Style Configuration (Thesis Quality)
# =============================================================================

THESIS_STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


# Color schemes for materials
MATERIAL_COLORS = {
    'CNA': '#1f77b4',           # Blue
    'EUROFER97': '#2ca02c',     # Green
    'EUROFER97_A': '#d62728',   # Red
    'EUROFER97_B': '#ff7f0e',   # Orange
    'EUROFER97_C': '#9467bd',   # Purple
    'SS316': '#8c564b',         # Brown
    'Fe': '#17becf',            # Cyan
    'W': '#7f7f7f',             # Gray
}

# Cooling time colors
COOLING_COLORS = {
    '300s': '#1f77b4',
    '2h': '#ff7f0e', 
    '24h': '#2ca02c',
    '4d': '#d62728',
    '15d': '#9467bd',
    'shutdown': '#7f7f7f',
}

# Source colors (experimental vs simulation)
SOURCE_COLORS = {
    'experimental': '#2ca02c',  # Green
    'alara': '#1f77b4',         # Blue
    'mcnp': '#d62728',          # Red
}


def apply_thesis_style():
    """Apply publication-quality style for thesis figures."""
    if HAS_MATPLOTLIB:
        plt.rcParams.update(THESIS_STYLE)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ActivityMeasurement:
    """Container for activity measurement data."""
    isotope: str
    activity_Bq: float
    uncertainty_Bq: float
    cooling_time: str = ""
    material: str = ""
    source: str = "experimental"  # 'experimental' or 'alara'
    
    @property
    def relative_uncertainty(self) -> float:
        """Return relative uncertainty as fraction."""
        if self.activity_Bq > 0:
            return self.uncertainty_Bq / self.activity_Bq
        return 0.0


@dataclass 
class ComparisonResult:
    """Container for experimental vs simulation comparison."""
    isotope: str
    experimental_Bq: float
    experimental_unc_Bq: float
    simulated_Bq: float
    simulated_unc_Bq: float
    cooling_time: str = ""
    material: str = ""
    
    @property
    def ratio_CE(self) -> float:
        """Calculated/Experimental ratio."""
        if self.experimental_Bq > 0:
            return self.simulated_Bq / self.experimental_Bq
        return np.nan
    
    @property
    def ratio_CE_uncertainty(self) -> float:
        """Uncertainty in C/E ratio (propagated)."""
        if self.experimental_Bq > 0 and self.simulated_Bq > 0:
            rel_exp = self.experimental_unc_Bq / self.experimental_Bq
            rel_sim = self.simulated_unc_Bq / self.simulated_Bq if self.simulated_Bq > 0 else 0
            rel_total = np.sqrt(rel_exp**2 + rel_sim**2)
            return self.ratio_CE * rel_total
        return np.nan
    
    @property
    def difference_percent(self) -> float:
        """Percent difference (C-E)/E * 100."""
        if self.experimental_Bq > 0:
            return (self.simulated_Bq - self.experimental_Bq) / self.experimental_Bq * 100
        return np.nan


# =============================================================================
# Comparison Bar Charts
# =============================================================================

def plot_activity_comparison_bar(
    comparisons: List[ComparisonResult],
    title: str = "Experimental vs ALARA Activity Comparison",
    ylabel: str = "Specific Activity (Bq/g)",
    log_scale: bool = True,
    show_ratio: bool = True,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create grouped bar chart comparing experimental and simulated activities.
    
    Parameters
    ----------
    comparisons : List[ComparisonResult]
        List of comparison results
    title : str
        Plot title
    ylabel : str
        Y-axis label
    log_scale : bool
        Use logarithmic y-axis
    show_ratio : bool
        Show C/E ratio as secondary axis
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
    
    apply_thesis_style()
    
    n = len(comparisons)
    if n == 0:
        raise ValueError("No comparison data provided")
    
    x = np.arange(n)
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Extract data
    isotopes = [c.isotope for c in comparisons]
    exp_vals = np.array([c.experimental_Bq for c in comparisons])
    exp_errs = np.array([c.experimental_unc_Bq for c in comparisons])
    sim_vals = np.array([c.simulated_Bq for c in comparisons])
    sim_errs = np.array([c.simulated_unc_Bq for c in comparisons])
    
    # Create bars
    bars1 = ax1.bar(x - width/2, exp_vals, width, 
                    yerr=exp_errs, capsize=3,
                    label='Experimental', color=SOURCE_COLORS['experimental'],
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, sim_vals, width,
                    yerr=sim_errs, capsize=3,
                    label='ALARA', color=SOURCE_COLORS['alara'],
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Isotope')
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(isotopes, rotation=45, ha='right')
    
    if log_scale:
        ax1.set_yscale('log')
    
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add C/E ratio on secondary axis
    if show_ratio:
        ax2 = ax1.twinx()
        ratios = np.array([c.ratio_CE for c in comparisons])
        ratio_errs = np.array([c.ratio_CE_uncertainty for c in comparisons])
        
        valid = ~np.isnan(ratios)
        ax2.errorbar(x[valid], ratios[valid], yerr=ratio_errs[valid],
                     fmt='D', color='red', markersize=6,
                     label='C/E Ratio', capsize=3, zorder=10)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('C/E Ratio', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax1


def plot_ce_ratio_summary(
    comparisons: List[ComparisonResult],
    title: str = "C/E Ratio Summary",
    group_by: str = "isotope",  # 'isotope', 'cooling_time', or 'material'
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Plot C/E ratios with error bars for validation summary.
    
    Parameters
    ----------
    comparisons : List[ComparisonResult]
        Comparison results
    title : str
        Plot title
    group_by : str
        How to group data ('isotope', 'cooling_time', 'material')
    figsize : tuple
        Figure size
    save_path : Path, optional
        Save location
        
    Returns
    -------
    fig, ax
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    apply_thesis_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group and sort
    if group_by == 'isotope':
        labels = [c.isotope for c in comparisons]
    elif group_by == 'cooling_time':
        labels = [f"{c.isotope}\n{c.cooling_time}" for c in comparisons]
    elif group_by == 'material':
        labels = [f"{c.isotope}\n{c.material}" for c in comparisons]
    else:
        labels = [c.isotope for c in comparisons]
    
    ratios = np.array([c.ratio_CE for c in comparisons])
    ratio_errs = np.array([c.ratio_CE_uncertainty for c in comparisons])
    
    x = np.arange(len(comparisons))
    
    # Color by agreement quality
    colors = []
    for r in ratios:
        if np.isnan(r):
            colors.append('gray')
        elif 0.8 <= r <= 1.2:
            colors.append('#2ca02c')  # Green - good
        elif 0.5 <= r <= 2.0:
            colors.append('#ff7f0e')  # Orange - fair
        else:
            colors.append('#d62728')  # Red - poor
    
    ax.bar(x, ratios, yerr=ratio_errs, capsize=3, color=colors,
           edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Reference lines
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5, label='Perfect agreement')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1.2, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between([-0.5, len(comparisons)-0.5], 0.8, 1.2, 
                    color='green', alpha=0.1, label='±20% band')
    
    ax.set_xlabel(group_by.replace('_', ' ').title())
    ax.set_ylabel('C/E Ratio')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, max(2.0, np.nanmax(ratios + ratio_errs) * 1.1))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# =============================================================================
# Decay Curve Plots
# =============================================================================

def plot_decay_curves(
    activity_data: Dict[str, List[Tuple[float, float, float]]],
    title: str = "Activity Decay Curves",
    xlabel: str = "Cooling Time (s)",
    ylabel: str = "Specific Activity (Bq/g)",
    log_y: bool = True,
    show_half_life: bool = True,
    half_lives: Optional[Dict[str, float]] = None,
    figsize: Tuple[float, float] = (10, 7),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Plot activity decay curves for multiple isotopes.
    
    Parameters
    ----------
    activity_data : Dict[str, List[Tuple[float, float, float]]]
        Dict of isotope -> list of (time_s, activity_Bq, uncertainty_Bq)
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    log_y : bool
        Use log y-axis
    show_half_life : bool
        Annotate half-lives
    half_lives : Dict[str, float], optional
        Half-lives in seconds for each isotope
    figsize : tuple
        Figure size
    save_path : Path, optional
        Save location
        
    Returns
    -------
    fig, ax
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    apply_thesis_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(activity_data)))
    
    for i, (isotope, data) in enumerate(activity_data.items()):
        times = np.array([d[0] for d in data])
        activities = np.array([d[1] for d in data])
        uncertainties = np.array([d[2] for d in data])
        
        color = colors[i]
        
        # Plot with error bars
        ax.errorbar(times, activities, yerr=uncertainties,
                    fmt='o-', color=color, label=isotope,
                    capsize=3, markersize=5, linewidth=1.5)
        
        # Annotate half-life if available
        if show_half_life and half_lives and isotope in half_lives:
            t_half = half_lives[isotope]
            if times.min() <= t_half <= times.max():
                # Find activity at half-life
                idx = np.argmin(np.abs(times - t_half))
                ax.axvline(x=t_half, color=color, linestyle=':', alpha=0.5)
                ax.annotate(f't½={t_half/3600:.1f}h', 
                           xy=(t_half, activities[idx]),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=8, color=color)
    
    if log_y:
        ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# =============================================================================
# Multi-Panel Material Comparison
# =============================================================================

def plot_material_comparison_grid(
    data_by_material: Dict[str, List[ComparisonResult]],
    title: str = "Activation Comparison by Material",
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create multi-panel grid comparing experimental vs ALARA by material.
    
    Parameters
    ----------
    data_by_material : Dict[str, List[ComparisonResult]]
        Comparison data organized by material
    title : str
        Overall figure title
    figsize : tuple
        Figure size
    save_path : Path, optional
        Save location
        
    Returns
    -------
    fig, axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    apply_thesis_style()
    
    n_materials = len(data_by_material)
    ncols = min(2, n_materials)
    nrows = (n_materials + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, (material, comparisons) in enumerate(data_by_material.items()):
        ax = axes[i]
        
        n = len(comparisons)
        x = np.arange(n)
        width = 0.35
        
        isotopes = [c.isotope for c in comparisons]
        exp_vals = np.array([c.experimental_Bq for c in comparisons])
        exp_errs = np.array([c.experimental_unc_Bq for c in comparisons])
        sim_vals = np.array([c.simulated_Bq for c in comparisons])
        sim_errs = np.array([c.simulated_unc_Bq for c in comparisons])
        
        color = MATERIAL_COLORS.get(material, '#1f77b4')
        
        ax.bar(x - width/2, exp_vals, width, yerr=exp_errs, capsize=2,
               label='Exp', color=SOURCE_COLORS['experimental'], alpha=0.8)
        ax.bar(x + width/2, sim_vals, width, yerr=sim_errs, capsize=2,
               label='ALARA', color=color, alpha=0.8)
        
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(isotopes, rotation=45, ha='right', fontsize=8)
        ax.set_title(material, fontsize=11)
        ax.set_ylabel('Activity (Bq)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


# =============================================================================
# Flux Wire Validation Plots
# =============================================================================

def plot_flux_wire_validation(
    measured_rates: Dict[str, float],
    measured_uncertainties: Dict[str, float],
    calculated_rates: Dict[str, float],
    calculated_uncertainties: Dict[str, float],
    title: str = "Flux Wire Reaction Rate Validation",
    ylabel: str = "Reaction Rate (reactions/atom/s)",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Compare measured vs calculated reaction rates for flux wire validation.
    
    Parameters
    ----------
    measured_rates : Dict[str, float]
        Measured reaction rates by reaction name
    measured_uncertainties : Dict[str, float]
        Measurement uncertainties
    calculated_rates : Dict[str, float]
        Calculated rates from unfolded spectrum
    calculated_uncertainties : Dict[str, float]
        Calculation uncertainties
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    save_path : Path, optional
        Save location
        
    Returns
    -------
    fig, (ax1, ax2)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    apply_thesis_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [2, 1]},
                                    sharex=True)
    
    # Common reactions
    reactions = sorted(set(measured_rates.keys()) & set(calculated_rates.keys()))
    n = len(reactions)
    x = np.arange(n)
    width = 0.35
    
    meas = np.array([measured_rates[r] for r in reactions])
    meas_unc = np.array([measured_uncertainties.get(r, 0) for r in reactions])
    calc = np.array([calculated_rates[r] for r in reactions])
    calc_unc = np.array([calculated_uncertainties.get(r, 0) for r in reactions])
    
    # Top panel: rates comparison
    ax1.bar(x - width/2, meas, width, yerr=meas_unc, capsize=3,
            label='Measured', color='#2ca02c', alpha=0.8)
    ax1.bar(x + width/2, calc, width, yerr=calc_unc, capsize=3,
            label='Calculated', color='#1f77b4', alpha=0.8)
    
    ax1.set_yscale('log')
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: C/E ratio
    ratio = calc / meas
    ratio_unc = ratio * np.sqrt((meas_unc/meas)**2 + (calc_unc/calc)**2)
    
    colors = ['#2ca02c' if 0.8 <= r <= 1.2 else '#d62728' for r in ratio]
    ax2.bar(x, ratio, yerr=ratio_unc, capsize=3, color=colors, alpha=0.8)
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5)
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1.2, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between([-0.5, n-0.5], 0.8, 1.2, color='green', alpha=0.1)
    
    ax2.set_xlabel('Reaction')
    ax2.set_ylabel('C/E')
    ax2.set_ylim(0, 2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(reactions, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


# =============================================================================
# Cd-Ratio Visualization
# =============================================================================

def plot_cd_ratio_analysis(
    wire_data: Dict[str, Dict[str, float]],
    title: str = "Cd-Ratio Analysis for Flux Characterization",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Visualize Cd-ratio analysis results for multiple wire types.
    
    Parameters
    ----------
    wire_data : Dict[str, Dict[str, float]]
        Wire data with keys like:
        {'Co': {'bare_activity': 1e5, 'cd_activity': 8e4, 'cd_ratio': 1.25}}
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : Path, optional
        Save location
        
    Returns
    -------
    fig, ax
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    apply_thesis_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    wires = list(wire_data.keys())
    x = np.arange(len(wires))
    
    # Left: Activity comparison
    bare = [wire_data[w].get('bare_activity', 0) for w in wires]
    cd = [wire_data[w].get('cd_activity', 0) for w in wires]
    
    width = 0.35
    ax1.bar(x - width/2, bare, width, label='Bare', color='#1f77b4', alpha=0.8)
    ax1.bar(x + width/2, cd, width, label='Cd-covered', color='#ff7f0e', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(wires)
    ax1.set_xlabel('Wire Material')
    ax1.set_ylabel('Activity (Bq)')
    ax1.set_title('Bare vs Cd-Covered Activities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Cd-ratios
    ratios = [wire_data[w].get('cd_ratio', 1) for w in wires]
    colors = ['#2ca02c' if r > 2 else '#ff7f0e' if r > 1.5 else '#d62728' for r in ratios]
    
    ax2.barh(x, ratios, color=colors, alpha=0.8)
    ax2.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=2.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(wires)
    ax2.set_xlabel('Cd Ratio (Bare/Cd)')
    ax2.set_title('Cd Ratios by Wire')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Annotate interpretation
    for i, (w, r) in enumerate(zip(wires, ratios)):
        if r > 2:
            interp = "thermal"
        elif r > 1.5:
            interp = "mixed"
        else:
            interp = "epithermal"
        ax2.annotate(interp, xy=(r, i), xytext=(5, 0),
                    textcoords='offset points', fontsize=8, va='center')
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


# =============================================================================
# Summary Statistics Table Plot
# =============================================================================

def plot_validation_summary_table(
    comparisons: List[ComparisonResult],
    title: str = "Validation Summary Statistics",
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create a table figure summarizing validation statistics.
    
    Parameters
    ----------
    comparisons : List[ComparisonResult]
        Comparison results
    title : str
        Table title
    figsize : tuple
        Figure size
    save_path : Path, optional
        Save location
        
    Returns
    -------
    fig, ax
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")
    
    apply_thesis_style()
    
    # Calculate statistics
    ratios = np.array([c.ratio_CE for c in comparisons])
    valid_ratios = ratios[~np.isnan(ratios)]
    
    stats = {
        'Metric': [
            'Number of comparisons',
            'Mean C/E ratio',
            'Std Dev C/E',
            'Within ±20%',
            'Within ±50%',
            'Max overestimate',
            'Max underestimate',
        ],
        'Value': [
            f'{len(valid_ratios)}',
            f'{np.mean(valid_ratios):.3f}',
            f'{np.std(valid_ratios):.3f}',
            f'{np.sum((valid_ratios >= 0.8) & (valid_ratios <= 1.2))}/{len(valid_ratios)} ({100*np.sum((valid_ratios >= 0.8) & (valid_ratios <= 1.2))/len(valid_ratios):.0f}%)',
            f'{np.sum((valid_ratios >= 0.5) & (valid_ratios <= 2.0))}/{len(valid_ratios)} ({100*np.sum((valid_ratios >= 0.5) & (valid_ratios <= 2.0))/len(valid_ratios):.0f}%)',
            f'{np.max(valid_ratios):.2f}x',
            f'{np.min(valid_ratios):.2f}x',
        ]
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    df = pd.DataFrame(stats)
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#4472C4', '#4472C4'],
                     colWidths=[0.5, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(color='white', fontweight='bold')
    
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


__all__ = [
    # Data classes
    'ActivityMeasurement',
    'ComparisonResult',
    # Comparison plots
    'plot_activity_comparison_bar',
    'plot_ce_ratio_summary',
    'plot_material_comparison_grid',
    # Decay curves
    'plot_decay_curves',
    # Flux wire
    'plot_flux_wire_validation',
    'plot_cd_ratio_analysis',
    # Summary
    'plot_validation_summary_table',
    # Style
    'apply_thesis_style',
    'MATERIAL_COLORS',
    'COOLING_COLORS',
    'SOURCE_COLORS',
]
