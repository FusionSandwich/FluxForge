"""FluxForge analysis module for gamma spectroscopy."""

from fluxforge.analysis.peakfit import (
    GaussianPeak,
    PeakFitResult,
    fit_single_peak,
    fit_multiple_peaks,
    auto_find_peaks,
    subtract_background,
    estimate_background,
    calculate_activity,
    peak_report,
)

from fluxforge.analysis.hpge_processor import (
    HPGeProcessor,
    HPGeAnalysisResult,
    GammaLine,
    ACTIVATION_GAMMA_LINES,
    get_gamma_lines_for_isotope,
    identify_isotope_from_energy,
    process_spe_file,
    batch_process_spe,
)

from fluxforge.analysis.segmented_detection import (
    RegionParams,
    SegmentedDetectionConfig,
    DetectedPeak,
    detect_peaks_segmented,
    merge_duplicate_peaks,
    filter_peaks_by_energy,
    create_report_peaks,
    combine_with_report_peaks,
)

from fluxforge.analysis.ensdf_matching import (
    normalize_isotope_label,
    element_from_isotope,
    GammaLine as ENSDFGammaLine,
    GammaDatabase,
    build_gamma_database,
    build_gamma_database_paceensdf,
    build_fallback_database,
    FALLBACK_GAMMA_LINES,
    IsotopeMatch,
    match_peaks_three_tier,
    build_tier1_isotopes,
    get_data_source,
    create_matching_databases,
    HAS_PACEENSDF,
)

__all__ = [
    # Peak fitting
    'GaussianPeak',
    'PeakFitResult',
    'fit_single_peak',
    'fit_multiple_peaks',
    'auto_find_peaks',
    'subtract_background',
    'estimate_background',
    'calculate_activity',
    'peak_report',
    # HPGe processing
    'HPGeProcessor',
    'HPGeAnalysisResult',
    'GammaLine',
    'ACTIVATION_GAMMA_LINES',
    'get_gamma_lines_for_isotope',
    'identify_isotope_from_energy',
    'process_spe_file',
    'batch_process_spe',
    # Segmented detection
    'RegionParams',
    'SegmentedDetectionConfig',
    'DetectedPeak',
    'detect_peaks_segmented',
    'merge_duplicate_peaks',
    'filter_peaks_by_energy',
    'create_report_peaks',
    'combine_with_report_peaks',
    # ENSDF matching
    'normalize_isotope_label',
    'element_from_isotope',
    'ENSDFGammaLine',
    'GammaDatabase',
    'build_gamma_database',
    'build_gamma_database_paceensdf',
    'build_fallback_database',
    'FALLBACK_GAMMA_LINES',
    'IsotopeMatch',
    'match_peaks_three_tier',
    'build_tier1_isotopes',
    'get_data_source',
    'create_matching_databases',
    'HAS_PACEENSDF',
]

