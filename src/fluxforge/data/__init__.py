"""FluxForge data module for efficiency curves and cross sections."""

from fluxforge.data.efficiency import (
    EfficiencyCurve,
    CALIBRATION_SOURCES,
    calculate_efficiency_from_source,
    distance_correction,
)

from fluxforge.data.crosssections import (
    CrossSection,
    CrossSectionLibrary,
    IRDFF_II_REACTIONS,
    create_irdff_placeholder_library,
    load_csv_cross_section,
)

from fluxforge.data.efficiency_models import (
    EfficiencyModelParams,
    EfficiencyModel,
    apply_efficiency_correction,
)

from fluxforge.data.irdff import (
    IRDFFDatabase,
    IRDFFCrossSection,
    IRDFF_REACTIONS,
    get_flux_wire_energy_groups,
    get_activation_energy_groups,
    build_response_matrix,
    get_irdff_database,
    list_dosimetry_reactions,
    get_cross_section,
)

__all__ = [
    # Efficiency curves (calibration-based)
    'EfficiencyCurve',
    'CALIBRATION_SOURCES',
    'calculate_efficiency_from_source',
    'distance_correction',
    # Efficiency models (equation-based)
    'EfficiencyModelParams',
    'EfficiencyModel',
    'apply_efficiency_correction',
    # Cross sections (legacy)
    'CrossSection',
    'CrossSectionLibrary',
    'IRDFF_II_REACTIONS',
    'create_irdff_placeholder_library',
    'load_csv_cross_section',
    # IRDFF-II database (new)
    'IRDFFDatabase',
    'IRDFFCrossSection',
    'IRDFF_REACTIONS',
    'get_flux_wire_energy_groups',
    'get_activation_energy_groups',
    'build_response_matrix',
    'get_irdff_database',
    'list_dosimetry_reactions',
    'get_cross_section',
]
