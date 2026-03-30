"""Core data structures and utilities."""

from fluxforge.core.calibration import (
    ASTM_E181_ENERGY_LIMIT_KEV,
    ASTM_E181_LOCKED_ORDER,
    CalibrationOrderResolution,
    EnergyCalibrationFit,
    EnergyCalibrationPoint,
    FWHMCalibrationFit,
    FWHMCalibrationPoint,
    energy_calibration_slope,
    estimate_local_fwhm_channels,
    evaluate_energy_calibration,
    fit_energy_calibration,
    fit_fwhm_calibration,
    resolve_energy_calibration_order,
    standard_requires_astm_e181,
)
from fluxforge.core.prior_covariance import (
    PriorCovarianceConfig,
    PriorCovarianceModel,
    ResponseUncertaintyConfig,
    ResponseUncertaintyPolicy,
)
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
    "EnergyGroupStructure",
    "EnergyCalibrationFit",
    "EnergyCalibrationPoint",
    "FWHMCalibrationFit",
    "FWHMCalibrationPoint",
    "ReactionCrossSection",
    "ResponseMatrix",
    "build_response_matrix",
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
    "fit_energy_calibration",
    "fit_fwhm_calibration",
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
]
