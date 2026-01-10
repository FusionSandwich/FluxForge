"""FluxForge data module for efficiency curves, cross sections, and nuclear databases."""

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

from fluxforge.data.gamma_database import (
    GammaLine,
    DecayData,
    GammaDatabase,
    get_database as get_gamma_database,
    find_gamma_matches,
    identify_nuclides,
    parse_nuclide_name,
)

from fluxforge.data.elements import (
    ELEMENT_SYMBOLS,
    ATOMIC_NUMBERS,
    ATOMIC_MASSES,
    element_from_z,
    z_from_element,
    atomic_mass,
    parse_isotope,
    make_zai,
    zai_components,
)

from fluxforge.data.njoy import (
    NJOYModule,
    GroupStructure,
    GROUP_STRUCTURE_DATA,
    NJOYInput,
    NJOYResult,
    NJOYPipelineSpec,
    generate_njoy_input,
    run_njoy,
    check_njoy_available,
    create_dosimetry_pipeline,
)

from fluxforge.data.nuclear_data import (
    DataType,
    DataLibrary,
    ReactionIdentifier,
    TemperatureData,
    NuclearData,
    NuclearDataProvider,
    ReactionMapping,
    ENDF_IRDFF_MAPPINGS,
    ENDFIRDFFBridge,
    create_temperature_set,
    interpolate_temperature,
    create_nuclear_data,
    create_multigroup_data,
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
    # Gamma database
    'GammaLine',
    'DecayData',
    'GammaDatabase',
    'get_gamma_database',
    'find_gamma_matches',
    'identify_nuclides',
    'parse_nuclide_name',
    # Element data
    'ELEMENT_SYMBOLS',
    'ATOMIC_NUMBERS',
    'ATOMIC_MASSES',
    'element_from_z',
    'z_from_element',
    'atomic_mass',
    'parse_isotope',
    'make_zai',
    'zai_components',
    # NJOY processing
    'NJOYModule',
    'GroupStructure',
    'GROUP_STRUCTURE_DATA',
    'NJOYInput',
    'NJOYResult',
    'NJOYPipelineSpec',
    'generate_njoy_input',
    'run_njoy',
    'check_njoy_available',
    'create_dosimetry_pipeline',
    # Unified nuclear data interface
    'DataType',
    'DataLibrary',
    'ReactionIdentifier',
    'TemperatureData',
    'NuclearData',
    'NuclearDataProvider',
    'ReactionMapping',
    'ENDF_IRDFF_MAPPINGS',
    'ENDFIRDFFBridge',
    'create_temperature_set',
    'interpolate_temperature',
    'create_nuclear_data',
    'create_multigroup_data',
]
