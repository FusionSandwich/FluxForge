"""
FluxForge TRIGA-Specific Modules
================================

Specialized modules for TRIGA reactor flux characterization,
Cd-ratio analysis, and k0-standardization.

Modules
-------
cd_ratio : Cd-ratio analysis for thermal/epithermal separation
k0 : k0-standardization workflow for TRIGA reactors

Example Usage
-------------
>>> from fluxforge.triga import CdRatioAnalyzer, TRIGAk0Workflow
>>> analyzer = CdRatioAnalyzer()
>>> flux_params = analyzer.characterize_flux(measurements)
>>> print(f"f = {flux_params.f}, α = {flux_params.alpha}")
"""

from .cd_ratio import (
    CdRatioMeasurement,
    FluxParameters,
    CdRatioAnalyzer,
    calculate_cd_ratio,
    estimate_f,
    estimate_alpha_multi,
    calculate_Q0_alpha,
    STANDARD_MONITORS,
)

from .k0 import (
    TRIGAk0Workflow,
    TRIGAIrradiationParams,
    validate_triga_flux_params,
    triple_monitor_method,
    TripleMonitorResult,
    calculate_sdc_factors,
    TRIPLE_MONITOR_DATA,
    EXTENDED_MONITOR_DATA,
)

from .reconcile import (
    UnfoldedFluxParameters,
    ReconciliationResult,
    ReconciliationStatus,
    compute_f_from_spectrum,
    compute_alpha_from_spectrum,
    compute_flux_parameters_from_spectrum,
    reconcile_flux_parameters,
    reconcile_with_unfold_result,
    quick_f_check,
)

__all__ = [
    # Cd-ratio analysis
    'CdRatioMeasurement',
    'FluxParameters',
    'CdRatioAnalyzer',
    'calculate_cd_ratio',
    'estimate_f',
    'estimate_alpha_multi',
    'calculate_Q0_alpha',
    'STANDARD_MONITORS',
    # k0 workflow
    'TRIGAk0Workflow',
    'TRIGAIrradiationParams',
    'validate_triga_flux_params',
    'triple_monitor_method',
    'TripleMonitorResult',
    'calculate_sdc_factors',
    'TRIPLE_MONITOR_DATA',
    'EXTENDED_MONITOR_DATA',
    # f/α reconciliation
    'UnfoldedFluxParameters',
    'ReconciliationResult',
    'ReconciliationStatus',
    'compute_f_from_spectrum',
    'compute_alpha_from_spectrum',
    'compute_flux_parameters_from_spectrum',
    'reconcile_flux_parameters',
    'reconcile_with_unfold_result',
    'quick_f_check',
]
