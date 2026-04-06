"""FluxForge plotting module for spectrum visualization."""

from fluxforge.plots.unfolding import (
    plot_spectrum_comparison,
    plot_spectrum_ratio,
    plot_spectrum_with_ratio,
    plot_spectrum_uncertainty_bands,
    plot_residuals_pulls,
    plot_covariance_correlation_heatmaps,
    plot_response_matrix,
    plot_cross_section_comparison,
    plot_convergence,
    plot_measured_vs_predicted,
)

from fluxforge.plots.activation import (
    ActivityMeasurement,
    ComparisonResult,
    plot_activity_comparison_bar,
    plot_ce_ratio_summary,
    plot_material_comparison_grid,
    plot_decay_curves,
    plot_flux_wire_validation,
    plot_cd_ratio_analysis,
    plot_validation_summary_table,
    apply_thesis_style,
    MATERIAL_COLORS,
    COOLING_COLORS,
    SOURCE_COLORS,
)

from fluxforge.plots.master_suite import (
    MasterPlotInputs,
    generate_master_plan_plots,
    load_example_plot_inputs,
    load_plot_inputs_from_artifacts,
    normalize_plot_formats,
)
from fluxforge.plots.spectrum_inspection import plot_gamma_spectrum

__all__ = [
    # Unfolding plots
    "plot_spectrum_comparison",
    "plot_spectrum_ratio",
    "plot_spectrum_with_ratio",
    "plot_spectrum_uncertainty_bands",
    "plot_residuals_pulls",
    "plot_covariance_correlation_heatmaps",
    "plot_response_matrix",
    "plot_cross_section_comparison",
    "plot_convergence",
    "plot_measured_vs_predicted",
    # Activation plots
    "ActivityMeasurement",
    "ComparisonResult",
    "plot_activity_comparison_bar",
    "plot_ce_ratio_summary",
    "plot_material_comparison_grid",
    "plot_decay_curves",
    "plot_flux_wire_validation",
    "plot_cd_ratio_analysis",
    "plot_validation_summary_table",
    "apply_thesis_style",
    "MATERIAL_COLORS",
    "COOLING_COLORS",
    "SOURCE_COLORS",
    # Master plot suite helpers
    "MasterPlotInputs",
    "generate_master_plan_plots",
    "load_example_plot_inputs",
    "load_plot_inputs_from_artifacts",
    "normalize_plot_formats",
    "plot_gamma_spectrum",
]
