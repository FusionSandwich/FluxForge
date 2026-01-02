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

__all__ = [
    # Batch processing
    'BatchProcessingConfig',
    'SpectrumResult',
    'BatchResult',
    'load_efficiency_model',
    'process_single_spectrum',
    'build_report_map',
    'process_batch',
    'results_to_dataframe',
    'save_results_csv',
    # Spectrum unfolding
    'FluxWireMeasurement',
    'UnfoldingResult',
    'SpectrumUnfolder',
    'quick_unfold',
    'build_flux_wire_response_matrix',
]
