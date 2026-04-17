"""FluxForge workflows module for complete analysis pipelines."""

from fluxforge.workflows.batch_processing import (
    BatchProcessingConfig,
    SpectrumResult,
    BatchResult,
    load_efficiency_model,
    process_single_spectrum,
    build_report_map,
    process_batch,
    results_to_dataframe,
    save_results_csv,
)

from fluxforge.workflows.spectrum_unfolding import (
    FluxWireMeasurement,
    UnfoldingResult,
    SpectrumUnfolder,
    quick_unfold,
    build_flux_wire_response_matrix,
)
from fluxforge.workflows.irradiation_optimization import (
    SecondIrradiationCandidate,
    SecondIrradiationPlan,
    SecondIrradiationScore,
    build_phase6_support_artifacts,
    build_second_irradiation_candidates,
    parse_second_irradiation_candidates,
    plan_second_irradiation,
    serialize_second_irradiation_plan,
)

__all__ = [
    # Batch processing
    "BatchProcessingConfig",
    "SpectrumResult",
    "BatchResult",
    "load_efficiency_model",
    "process_single_spectrum",
    "build_report_map",
    "process_batch",
    "results_to_dataframe",
    "save_results_csv",
    # Spectrum unfolding
    "FluxWireMeasurement",
    "UnfoldingResult",
    "SpectrumUnfolder",
    "quick_unfold",
    "build_flux_wire_response_matrix",
    # Phase 6 irradiation optimization
    "SecondIrradiationCandidate",
    "SecondIrradiationPlan",
    "SecondIrradiationScore",
    "build_phase6_support_artifacts",
    "build_second_irradiation_candidates",
    "parse_second_irradiation_candidates",
    "plan_second_irradiation",
    "serialize_second_irradiation_plan",
]
