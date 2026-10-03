"""Shared constants for FluxForge GUI."""

ALLOWED_REACTION_CATEGORIES = ("all", "thermal", "epithermal", "fast", "fission")
GUI_PEAK_IDENTIFICATION_METHODS = ("line_match", "nuclide_consensus", "hybrid_ranked")
GUI_PEAK_COUNTING_METHODS = (
    "gaussian_fit",
    "hypermet",
    "covell_local",
    "gilmore_minimum",
    "iec_tiered",
)
GUI_RAFM_COUNTING_METHODS = (
    "qg",
    "quantum_gold",
    "covell_local",
    "gilmore_minimum",
    "iec_tiered",
)
GUI_BUFFER_OPERATIONS = ("sum", "subtract", "average", "ratio")
GUI_UNFOLD_METHODS = ("gls", "gravel", "mlem")
GUI_UNFOLD_MLEM_CONVERGENCE_MODES = ("relative", "ddJ")
