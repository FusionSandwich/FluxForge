"""FluxForge analysis module for gamma spectroscopy."""

from importlib import import_module

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

from fluxforge.analysis.k0_workflow import (
    CAPABILITY_FLAGS as K0_CAPABILITY_FLAGS,
    PeakObservation,
    aggregate_k0_analysis_bundles,
    analyze_k0_observations,
    build_k0_report_payload,
    build_detector_characterization,
    build_facility_characterization,
    classify_peak_observation,
    evaluate_k0_qaqc,
    evaluate_detector_characterization,
    peak_report_to_observations,
    resolve_governed_libraries,
)

from fluxforge.analysis.astm_e261 import (
    analyze_astm_e261_plan,
    equivalent_irradiation_duration_s,
    target_atom_count,
)

from fluxforge.analysis.astm_e262 import (
    analyze_astm_e262_plan,
)

from fluxforge.analysis.optimization_difom import (
    DIFOMLineTerm,
    DIFOMLineScore,
    DIFOMEvaluation,
    DIFOMScheduleCandidate,
    DIFOMScheduleScore,
    build_difom_terms_from_activity_results,
    compute_difom_score,
    difom_line_score,
    evaluate_difom,
    parse_difom_sweep_payload,
    rank_difom_schedules,
    serialize_difom_ranking,
)

from fluxforge.analysis.optimization_fim import (
    FIMDiagnostics,
    FIMEvaluation,
    FIMScheduleScore,
    build_fisher_information,
    evaluate_fim,
    rank_fim_schedules,
    serialize_fim_ranking,
)

from fluxforge.analysis.optimization_mwdcs import (
    MWDCSLineTerm,
    MWDCSWindow,
    MWDCSWindowScore,
    MWDCSEvaluation,
    MWDCSScheduleCandidate,
    MWDCSScheduleScore,
    build_mwdcs_candidate_from_activity_results,
    candidate_from_difom_candidate,
    evaluate_mwdcs,
    parse_mwdcs_sweep_payload,
    rank_mwdcs_schedules,
    serialize_mwdcs_ranking,
)

from fluxforge.analysis.optimization_bassd import (
    BASSDAction,
    BASSDActionScore,
    BASSDLineState,
    BASSDEvaluation,
    BASSDScheduleCandidate,
    BASSDScheduleScore,
    build_bassd_candidate_from_activity_results,
    candidate_from_difom_candidate as bassd_candidate_from_difom_candidate,
    evaluate_bassd,
    parse_bassd_sweep_payload,
    posterior_variance_update,
    rank_bassd_schedules,
    serialize_bassd_ranking,
)

from fluxforge.analysis.optimization_stbdmr import (
    STBDMRLineTerm,
    STBDMRWindow,
    STBDMRWindowScore,
    STBDMRDiagnostics,
    STBDMREvaluation,
    STBDMRScheduleCandidate,
    STBDMRScheduleScore,
    build_interference_graph,
    build_stbdmr_candidate_from_activity_results,
    candidate_from_difom_candidate as stbdmr_candidate_from_difom_candidate,
    evaluate_stbdmr,
    parse_stbdmr_sweep_payload,
    rank_stbdmr_schedules,
    serialize_stbdmr_ranking,
)

from fluxforge.analysis.isotope_priority import (
    IsotopePriorityWeights,
    IsotopePriorityScore,
    rank_isotopes_from_activity_review_payload,
    serialize_isotope_priority_ranking,
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
    DirectScipyPeakFinder,
    WaveletPeakFinder,
    RelativeExtremaPeakFinder,
    refine_peak_centroids,
    merge_nearby_peaks,
)

from fluxforge.analysis.spectrum_math import (
    add_spectra,
    nonnegative_counts_for_algorithm,
    subtract_measured_background,
    subtract_spectra,
    moving_average,
)

from fluxforge.analysis.line_search import (
    LineMatch,
    search_decay_lines,
    list_nuclide_lines,
)
from fluxforge.analysis.spectroscopy_tools import (
    PeakCandidate,
    PeakFitSummary,
    prominence_peaks,
    fit_gaussian_baseline,
)
from fluxforge.analysis.detector_calibration import (
    EfficiencyPoint,
    EfficiencyFit,
    ResolutionCurve,
    ResolutionFit,
    fit_efficiency_curve,
    fit_resolution_curve,
)

from fluxforge.analysis.flux_wire_analysis import (
    FLUX_WIRE_NUCLIDES,
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

_NAA_ANN_EXPORTS = (
    "NAAANNConfig",
    "NAAANNResult",
    "AugmentationConfig",
    "SpectralAugmentor",
    "NAAANNModel",
    "NAAANNAnalyzer",
    "create_training_dataset",
    "train_naa_ann_model",
    "HAS_TENSORFLOW",
)

__all__ = [
    # Peak fitting
    "GaussianPeak",
    "PeakFitResult",
    "fit_single_peak",
    "fit_multiple_peaks",
    "auto_find_peaks",
    "subtract_background",
    "estimate_background",
    "calculate_activity",
    "peak_report",
    # HPGe processing
    "HPGeProcessor",
    "HPGeAnalysisResult",
    "GammaLine",
    "ACTIVATION_GAMMA_LINES",
    "get_gamma_lines_for_isotope",
    "identify_isotope_from_energy",
    "process_spe_file",
    "batch_process_spe",
    # Segmented detection
    "RegionParams",
    "SegmentedDetectionConfig",
    "DetectedPeak",
    "detect_peaks_segmented",
    "merge_duplicate_peaks",
    "filter_peaks_by_energy",
    "create_report_peaks",
    "combine_with_report_peaks",
    # ENSDF matching
    "normalize_isotope_label",
    "element_from_isotope",
    "ENSDFGammaLine",
    "GammaDatabase",
    "build_gamma_database",
    "build_gamma_database_paceensdf",
    "build_fallback_database",
    "FALLBACK_GAMMA_LINES",
    "IsotopeMatch",
    "match_peaks_three_tier",
    "build_tier1_isotopes",
    "get_data_source",
    "create_matching_databases",
    "HAS_PACEENSDF",
    # Spectrum math
    "add_spectra",
    "subtract_spectra",
    "subtract_measured_background",
    "nonnegative_counts_for_algorithm",
    "moving_average",
    # k0-NAA
    "K0Parameters",
    "K0NuclideData",
    "K0_DATABASE",
    "K0Measurement",
    "K0Result",
    "K0Calculator",
    "calculate_k0_parameters",
    "calculate_Q0_alpha",
    "saturation_factor",
    "decay_factor",
    "counting_factor",
    "sdc_factor",
    "get_k0_data",
    "identify_isotope_from_gamma",
    "create_k0_measurement_from_peak",
    "K0_CAPABILITY_FLAGS",
    "PeakObservation",
    "classify_peak_observation",
    "evaluate_detector_characterization",
    "peak_report_to_observations",
    "resolve_governed_libraries",
    "analyze_astm_e261_plan",
    "analyze_astm_e262_plan",
    "equivalent_irradiation_duration_s",
    "target_atom_count",
    "build_detector_characterization",
    "build_facility_characterization",
    "analyze_k0_observations",
    "aggregate_k0_analysis_bundles",
    "evaluate_k0_qaqc",
    "build_k0_report_payload",
    # Irradiation optimization (Method 1 DI-FOM)
    "DIFOMLineTerm",
    "DIFOMLineScore",
    "DIFOMEvaluation",
    "DIFOMScheduleCandidate",
    "DIFOMScheduleScore",
    "build_difom_terms_from_activity_results",
    "compute_difom_score",
    "difom_line_score",
    "evaluate_difom",
    "parse_difom_sweep_payload",
    "rank_difom_schedules",
    "serialize_difom_ranking",
    # Irradiation optimization (Method 2 FIM)
    "FIMDiagnostics",
    "FIMEvaluation",
    "FIMScheduleScore",
    "build_fisher_information",
    "evaluate_fim",
    "rank_fim_schedules",
    "serialize_fim_ranking",
    # Irradiation optimization (Method 3 MWDCS)
    "MWDCSLineTerm",
    "MWDCSWindow",
    "MWDCSWindowScore",
    "MWDCSEvaluation",
    "MWDCSScheduleCandidate",
    "MWDCSScheduleScore",
    "build_mwdcs_candidate_from_activity_results",
    "candidate_from_difom_candidate",
    "evaluate_mwdcs",
    "parse_mwdcs_sweep_payload",
    "rank_mwdcs_schedules",
    "serialize_mwdcs_ranking",
    # Irradiation optimization (Method N1 BASS-D)
    "BASSDAction",
    "BASSDActionScore",
    "BASSDLineState",
    "BASSDEvaluation",
    "BASSDScheduleCandidate",
    "BASSDScheduleScore",
    "build_bassd_candidate_from_activity_results",
    "bassd_candidate_from_difom_candidate",
    "evaluate_bassd",
    "parse_bassd_sweep_payload",
    "posterior_variance_update",
    "rank_bassd_schedules",
    "serialize_bassd_ranking",
    # Irradiation optimization (Method N2 STBD-MR)
    "STBDMRLineTerm",
    "STBDMRWindow",
    "STBDMRWindowScore",
    "STBDMRDiagnostics",
    "STBDMREvaluation",
    "STBDMRScheduleCandidate",
    "STBDMRScheduleScore",
    "build_interference_graph",
    "build_stbdmr_candidate_from_activity_results",
    "stbdmr_candidate_from_difom_candidate",
    "evaluate_stbdmr",
    "parse_stbdmr_sweep_payload",
    "rank_stbdmr_schedules",
    "serialize_stbdmr_ranking",
    # Isotope-priority ranking workflow
    "IsotopePriorityWeights",
    "IsotopePriorityScore",
    "rank_isotopes_from_activity_review_payload",
    "serialize_isotope_priority_ranking",
    # Advanced peak finders
    "PeakInfo",
    "snip_background",
    "estimate_background_linear",
    "savitzky_golay_smooth",
    "SimplePeakFinder",
    "WindowPeakFinder",
    "ChunkedPeakFinder",
    "ScipyPeakFinder",
    "DirectScipyPeakFinder",
    "WaveletPeakFinder",
    "RelativeExtremaPeakFinder",
    "refine_peak_centroids",
    "merge_nearby_peaks",
    # Line search
    "LineMatch",
    "search_decay_lines",
    "list_nuclide_lines",
    # Spectroscopy tools
    "PeakCandidate",
    "PeakFitSummary",
    "prominence_peaks",
    "fit_gaussian_baseline",
    # Flux wire analysis
    "FLUX_WIRE_NUCLIDES",
    "FluxWireAnalysisResult",
    "analyze_flux_wire",
    "compare_raw_vs_processed",
    "get_sample_element",
    "get_expected_isotopes",
    "build_gamma_library",
    "IdentifiedPeak",
    # Flux unfolding
    "FluxWireReaction",
    "extract_reactions_from_processed",
    "unfold_discrete_bins",
    "unfold_gls",
    "unfold_flux_wires",
    "FluxWireUnfoldResult",
    "DiscreteUnfoldResult",
    "GLSUnfoldResult",
    "THERMAL_CROSS_SECTIONS",
    "REACTION_ENERGIES",
    # Flux wire selection (INL reactor dosimetry)
    "WireCategory",
    "FluxWireReactionData",
    "WireCombinationScore",
    "FLUX_WIRE_DATABASE",
    "INL_ROBUST_COMBOS",
    "INL_WELL_CHARACTERIZED_COMBOS",
    "get_wire_reactions",
    "analyze_wire_combination",
    "suggest_wire_combinations",
    "recommend_wire_additions",
    "print_wire_summary",
    "calculate_1mev_equivalent_fluence",
    "calculate_dpa",
    # Wire set robustness diagnostics
    "RobustnessLevel",
    "ConditioningMetrics",
    "EnergyCoverage",
    "LeaveOneOutResult",
    "WireSetDiagnostics",
    "calculate_condition_metrics",
    "analyze_energy_coverage",
    "leave_one_out_analysis",
    "diagnose_wire_set",
    "quick_condition_check",
    "estimate_optimal_wire_count",
]

__all__.extend(_NAA_ANN_EXPORTS)


def __getattr__(name):
    if name in _NAA_ANN_EXPORTS:
        _naa_ann = import_module("fluxforge.analysis.naa_ann")

        for export_name in _NAA_ANN_EXPORTS:
            if hasattr(_naa_ann, export_name):
                globals()[export_name] = getattr(_naa_ann, export_name)
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module 'fluxforge.analysis' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
