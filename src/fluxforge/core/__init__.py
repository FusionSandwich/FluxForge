"""Core data structures and utilities."""

from importlib import import_module

from fluxforge.core.calibration import (
    ASTM_E181_ENERGY_LIMIT_KEV,
    ASTM_E181_LOCKED_ORDER,
    CalibrationOrderResolution,
    EnergyDeviationPair,
    EnergyCalibrationFit,
    EnergyCalibrationPoint,
    FWHMCalibrationFit,
    FWHMCalibrationPoint,
    QuickCalibrationResult,
    apply_energy_deviation_pairs,
    energy_calibration_slope,
    estimate_local_fwhm_channels,
    evaluate_energy_calibration,
    evaluate_energy_deviation_pairs,
    fit_energy_calibration,
    fit_fwhm_calibration,
    fit_quick_slider_calibration,
    resolve_energy_calibration_order,
    standard_requires_astm_e181,
)
from fluxforge.core.prior_covariance import (
    PriorCovarianceConfig,
    PriorCovarianceModel,
    ResponseUncertaintyConfig,
    ResponseUncertaintyPolicy,
)
from fluxforge.core.planning_models import LineMaskingResult, OptimizationScenario
from fluxforge.core.response import (
    EnergyGroupStructure,
    ReactionCrossSection,
    ResponseMatrix,
    build_response_matrix,
)
from fluxforge.core.sample import Container, Cover, MaterialComponent, Sample
from fluxforge.core.validation import (
    CEEntry,
    CETable,
    ClosureMetrics,
    ValidationBundle,
    ValidationStatus,
    calculate_ce_table,
    calculate_closure_metrics,
    create_validation_bundle,
)

__all__ = [
    "ASTM_E181_ENERGY_LIMIT_KEV",
    "ASTM_E181_LOCKED_ORDER",
    "CalibrationOrderResolution",
    "EnergyDeviationPair",
    "EnergyGroupStructure",
    "EnergyCalibrationFit",
    "EnergyCalibrationPoint",
    "FWHMCalibrationFit",
    "FWHMCalibrationPoint",
    "QuickCalibrationResult",
    "ReactionCrossSection",
    "ResponseMatrix",
    "apply_energy_deviation_pairs",
    "build_response_matrix",
    "LineMaskingResult",
    "OptimizationScenario",
    "PriorCovarianceConfig",
    "PriorCovarianceModel",
    "ResponseUncertaintyConfig",
    "ResponseUncertaintyPolicy",
    "Sample",
    "MaterialComponent",
    "Cover",
    "Container",
    "energy_calibration_slope",
    "estimate_local_fwhm_channels",
    "evaluate_energy_calibration",
    "evaluate_energy_deviation_pairs",
    "fit_energy_calibration",
    "fit_fwhm_calibration",
    "fit_quick_slider_calibration",
    "resolve_energy_calibration_order",
    "standard_requires_astm_e181",
    # Validation
    "CEEntry",
    "CETable",
    "ClosureMetrics",
    "ValidationBundle",
    "ValidationStatus",
    "calculate_ce_table",
    "calculate_closure_metrics",
    "create_validation_bundle",
    # Loaded lazily so fluxforge.io.spe can import calibration without a
    # core -> workspace_document -> io package cycle.
    "WORKSPACE_DOCUMENT_SCHEMA",
    "WORKSPACE_DOCUMENT_VERSION",
    "AnalysisROI",
    "CalibrationModel",
    "CanvasViewport",
    "CorrectionSettings",
    "DetectorGeometry",
    "DetectorProfile",
    "EfficiencyModelState",
    "FitDiagnostics",
    "NuclideAssignment",
    "PeakComponent",
    "PeakModel",
    "SpectrumRoleAssignment",
    "WorkspaceDocument",
    "WorkspaceSpectrum",
    "WorkspaceValidationError",
]


_WORKSPACE_DOCUMENT_EXPORTS = frozenset(__all__[-17:])


def __getattr__(name: str):
    if name not in _WORKSPACE_DOCUMENT_EXPORTS:
        raise AttributeError(name)
    module = import_module("fluxforge.core.workspace_document")
    value = getattr(module, name)
    globals()[name] = value
    return value
