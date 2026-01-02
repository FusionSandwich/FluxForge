"""
FluxForge - Spectrum Unfolding and Gamma Analysis Toolkit

Modules:
- io: Spectrum file I/O (SPE, Genie-2000)
- data: Efficiency models, cross sections
- analysis: Peak fitting, detection, isotope matching
- physics: Activation calculations
- solvers: Spectrum unfolding (GLS, GRAVEL, MLEM)
- workflows: Complete analysis pipelines
"""

from importlib.metadata import version

__all__ = ["__version__"]

try:
    __version__ = version("fluxforge")
except Exception:  # fallback for editable installs before metadata exists
    __version__ = "0.1.0"

# Convenient imports for common operations
from fluxforge.io import GammaSpectrum, read_spe_file, read_genie_spectrum
from fluxforge.data import EfficiencyModel, EfficiencyCurve
from fluxforge.analysis import (
    fit_single_peak,
    auto_find_peaks,
    detect_peaks_segmented,
    match_peaks_three_tier,
    build_gamma_database,
)

__all__ += [
    # I/O
    'GammaSpectrum',
    'read_spe_file',
    'read_genie_spectrum',
    # Data
    'EfficiencyModel',
    'EfficiencyCurve',
    # Analysis
    'fit_single_peak',
    'auto_find_peaks',
    'detect_peaks_segmented',
    'match_peaks_three_tier',
    'build_gamma_database',
]
