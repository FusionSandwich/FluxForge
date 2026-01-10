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

from fluxforge.analysis.k0_naa import (
    K0Parameters,
    K0NuclideData,
    K0_DATABASE,
    K0Measurement,
    K0Result,
    K0Calculator,
    calculate_k0_parameters,
    calculate_Q0_alpha,
    saturation_factor,
    decay_factor,
    counting_factor,
    sdc_factor,
    get_k0_data,
    identify_isotope_from_gamma,
    create_k0_measurement_from_peak,
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

from fluxforge.analysis.peak_finders import (
    PeakInfo,
    snip_background,
    estimate_background_linear,
    savitzky_golay_smooth,
    SimplePeakFinder,
    WindowPeakFinder,
    ChunkedPeakFinder,
    ScipyPeakFinder,
    refine_peak_centroids,
    merge_nearby_peaks,
)

from fluxforge.analysis.flux_wire_analysis import (
    FLUX_WIRE_NUCLIDES,
    ELEMENT_TO_ISOTOPES,
    FluxWireAnalysisResult,
    analyze_flux_wire,
    compare_raw_vs_processed,
    get_sample_element,
    get_expected_isotopes,
    build_gamma_library,
    IdentifiedPeak,
)

from fluxforge.analysis.flux_unfold import (
    FluxWireReaction,
    extract_reactions_from_processed,
    unfold_discrete_bins,
    unfold_gls,
    unfold_flux_wires,
    FluxWireUnfoldResult,
    DiscreteUnfoldResult,
    GLSUnfoldResult,
    THERMAL_CROSS_SECTIONS,
    REACTION_ENERGIES,
)

# Flux wire selection advisor (INL reactor dosimetry workflow)
from fluxforge.analysis.flux_wire_selection import (
    WireCategory,
    FluxWireReaction as FluxWireReactionData,
    WireCombinationScore,
    FLUX_WIRE_DATABASE,
    INL_ROBUST_COMBOS,
    INL_WELL_CHARACTERIZED_COMBOS,
    get_wire_reactions,
    analyze_wire_combination,
    suggest_wire_combinations,
    recommend_wire_additions,
    print_wire_summary,
    calculate_1mev_equivalent_fluence,
    calculate_dpa,
)

# Wire set robustness diagnostics
from fluxforge.analysis.robustness import (
    RobustnessLevel,
    ConditioningMetrics,
    EnergyCoverage,
    LeaveOneOutResult,
    WireSetDiagnostics,
    calculate_condition_metrics,
    analyze_energy_coverage,
    leave_one_out_analysis,
    diagnose_wire_set,
    quick_condition_check,
    estimate_optimal_wire_count,
)

# NAA-ANN imports (optional - requires TensorFlow)
try:
    from fluxforge.analysis.naa_ann import (
        NAAANNConfig,
        NAAANNResult,
        AugmentationConfig,
        SpectralAugmentor,
        NAAANNModel,
        NAAANNAnalyzer,
        create_training_dataset,
        train_naa_ann_model,
        HAS_TENSORFLOW,
    )
    _HAS_NAA_ANN = True
except ImportError:
    _HAS_NAA_ANN = False

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
    # k0-NAA
    'K0Parameters',
    'K0NuclideData',
    'K0_DATABASE',
    'K0Measurement',
    'K0Result',
    'K0Calculator',
    'calculate_k0_parameters',
    'calculate_Q0_alpha',
    'saturation_factor',
    'decay_factor',
    'counting_factor',
    'sdc_factor',
    'get_k0_data',
    'identify_isotope_from_gamma',
    'create_k0_measurement_from_peak',
    # Advanced peak finders
    'PeakInfo',
    'snip_background',
    'estimate_background_linear',
    'savitzky_golay_smooth',
    'SimplePeakFinder',
    'WindowPeakFinder',
    'ChunkedPeakFinder',
    'ScipyPeakFinder',
    'refine_peak_centroids',
    'merge_nearby_peaks',
    # Flux wire analysis
    'FLUX_WIRE_NUCLIDES',
    'ELEMENT_TO_ISOTOPES',
    'FluxWireAnalysisResult',
    'analyze_flux_wire',
    'compare_raw_vs_processed',
    'get_sample_element',
    'get_expected_isotopes',
    'build_gamma_library',
    'IdentifiedPeak',
    # Flux unfolding
    'FluxWireReaction',
    'extract_reactions_from_processed',
    'unfold_discrete_bins',
    'unfold_gls',
    'unfold_flux_wires',
    'FluxWireUnfoldResult',
    'DiscreteUnfoldResult',
    'GLSUnfoldResult',
    'THERMAL_CROSS_SECTIONS',
    'REACTION_ENERGIES',
    # Flux wire selection (INL reactor dosimetry)
    'WireCategory',
    'FluxWireReactionData',
    'WireCombinationScore',
    'FLUX_WIRE_DATABASE',
    'INL_ROBUST_COMBOS',
    'INL_WELL_CHARACTERIZED_COMBOS',
    'get_wire_reactions',
    'analyze_wire_combination',
    'suggest_wire_combinations',
    'recommend_wire_additions',
    'print_wire_summary',
    'calculate_1mev_equivalent_fluence',
    'calculate_dpa',
    # Wire set robustness diagnostics
    'RobustnessLevel',
    'ConditioningMetrics',
    'EnergyCoverage',
    'LeaveOneOutResult',
    'WireSetDiagnostics',
    'calculate_condition_metrics',
    'analyze_energy_coverage',
    'leave_one_out_analysis',
    'diagnose_wire_set',
    'quick_condition_check',
    'estimate_optimal_wire_count',
]

# Add NAA-ANN exports if available
if _HAS_NAA_ANN:
    __all__.extend([
        'NAAANNConfig',
        'NAAANNResult',
        'AugmentationConfig',
        'SpectralAugmentor',
        'NAAANNModel',
        'NAAANNAnalyzer',
        'create_training_dataset',
        'train_naa_ann_model',
        'HAS_TENSORFLOW',
    ])

