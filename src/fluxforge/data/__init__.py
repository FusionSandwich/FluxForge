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
from fluxforge.data.kayzero_k0 import (
    KayzeroImportResult,
    import_kayzero_k0_library,
    write_governed_library_json,
    write_import_report_json,
)
from fluxforge.data.rafm_decay import (
    load_rafm_decay_library,
    get_rafm_decay_entry,
    get_rafm_gamma_lines,
    get_rafm_half_life,
)
from fluxforge.data.flux_wire_catalog import (
    FluxWireCatalogEntry,
    load_flux_wire_catalog,
    list_flux_wire_isotopes,
    get_flux_wire_catalog_entry,
    get_flux_wire_isotopes_for_element,
    list_flux_wire_elements,
)
from fluxforge.data.flux_wire_unfolding import (
    load_flux_wire_unfolding_defaults,
    load_flux_wire_sample_defaults,
    load_flux_wire_reaction_defaults,
    load_flux_wire_product_reactions,
    get_flux_wire_reaction_id,
    get_flux_wire_isotope_fraction,
    get_flux_wire_reaction_cross_section_defaults,
    get_flux_wire_reaction_characteristic_energies,
    get_flux_wire_response_parameters,
)
from fluxforge.data.nuclear_data_sources import (
    NuclearDataSourceRecord,
    get_nuclear_data_source,
    list_nuclear_data_sources,
    load_gamma_identification_source,
    summarize_nuclear_data_source,
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
    # Kayzero k0 importer
    'KayzeroImportResult',
    'import_kayzero_k0_library',
    'write_governed_library_json',
    'write_import_report_json',
    # RAFM decay subset
    'load_rafm_decay_library',
    'get_rafm_decay_entry',
    'get_rafm_gamma_lines',
    'get_rafm_half_life',
    # Flux-wire reaction metadata
    'FluxWireCatalogEntry',
    'load_flux_wire_catalog',
    'list_flux_wire_isotopes',
    'get_flux_wire_catalog_entry',
    'get_flux_wire_isotopes_for_element',
    'list_flux_wire_elements',
    # Flux-wire unfolding defaults
    'load_flux_wire_unfolding_defaults',
    'load_flux_wire_sample_defaults',
    'load_flux_wire_reaction_defaults',
    'load_flux_wire_product_reactions',
    'get_flux_wire_reaction_id',
    'get_flux_wire_isotope_fraction',
    'get_flux_wire_reaction_cross_section_defaults',
    'get_flux_wire_reaction_characteristic_energies',
    'get_flux_wire_response_parameters',
    # Nuclear data sources / provenance-aware registry
    'NuclearDataSourceRecord',
    'get_nuclear_data_source',
    'list_nuclear_data_sources',
    'load_gamma_identification_source',
    'summarize_nuclear_data_source',
]
