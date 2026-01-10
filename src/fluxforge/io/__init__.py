"""FluxForge I/O module for spectrum file parsing."""

from fluxforge.io.spe import (
    GammaSpectrum,
    read_spe_file,
    write_spe_file,
    read_multiple_spe,
)

from fluxforge.io.genie import (
    parse_genie_header,
    read_genie_spectrum,
    parse_asc_filename,
    normalize_timepoint,
    ReportPeak,
    parse_genie_report,
    extract_report_id,
    SpectrumPair,
    discover_spectrum_pairs,
    load_spectrum_pair,
)
from fluxforge.io.hpge import HPGeReport, read_hpge_report
from fluxforge.io.csv_readers import (
    EfficiencyExport,
    FluxWireTiming,
    read_efficiency_export,
    read_flux_wire_timing_csv,
)
from fluxforge.io.alara import (
    ALARASettings,
    ALARAInputGenerator,
    ALARAMixture,
    ALARASchedule,
    ALARAPulseHistory,
    ALARAFlux,
    ALARAZoneResult,
    ALARAOutput,
    parse_alara_output,
    read_alara_input,
    read_alara_output,
    write_alara_flux,
    read_alara_flux,
    fluxforge_spectrum_to_alara,
    create_alara_activation_input,
    # Voxel-averaged statistics
    DEFAULT_TIME_GROUPS,
    format_time_label,
    build_zone_stats,
    build_isotope_table,
)
from fluxforge.io.mcnp import (
    parse_mcnp_input,
    read_meshtal_hdf5,
    MCNPSpectrum,
    read_mcnp_spectrum_csv,
    MCTALTally,
    MCTALFile,
    read_mctal,
    read_mcnp_flux_tally,
)
from fluxforge.io.openmc import (
    read_openmc_tally,
    OpenMCSpectrum,
    StatepointInfo,
    read_statepoint_info,
    read_openmc_flux_spectrum,
)
from fluxforge.io.cnf import (
    read_cnf_file,
    CNFData,
    CNFHeader,
    CNFCalibration,
    parse_cnf_binary,
    can_read_cnf,
)
from fluxforge.io.n42 import (
    N42Measurement,
    N42Document,
    read_n42_file,
    read_n42_spectrum,
    write_n42_file,
)
from fluxforge.io.interop import (
    SaturationRateData,
    read_saturation_rates_csv,
    write_saturation_rates_csv,
    read_lower_triangular_matrix,
    write_lower_triangular_matrix,
    STAYSLBundle,
    export_staysl_bundle,
    import_staysl_bundle,
)

from fluxforge.io.flux_wire import (
    FluxWireData,
    NuclideResult,
    EfficiencyCalibration,
    read_raw_asc,
    read_processed_txt,
    read_flux_wire,
    load_flux_wire_directory,
)

from fluxforge.io.artifacts import (
    read_artifact,
    write_artifact,
    read_line_activities,
    read_peak_report,
    read_reaction_rates,
    read_report_bundle,
    read_response_bundle,
    read_spectrum_file,
    read_unfold_result,
    read_validation_bundle,
    write_line_activities,
    write_peak_report,
    write_reaction_rates,
    write_report_bundle,
    write_response_bundle,
    write_spectrum_file,
    write_unfold_result,
    write_validation_bundle,
)

__all__ = [
    # SPE format
    'GammaSpectrum',
    'read_spe_file',
    'write_spe_file',
    'read_multiple_spe',
    # Genie-2000 format
    'parse_genie_header',
    'read_genie_spectrum',
    'parse_asc_filename',
    'normalize_timepoint',
    'ReportPeak',
    'parse_genie_report',
    'extract_report_id',
    'SpectrumPair',
    'discover_spectrum_pairs',
    'load_spectrum_pair',
    # CSV/HPGe exports
    'EfficiencyExport',
    'FluxWireTiming',
    'read_efficiency_export',
    'read_flux_wire_timing_csv',
    'HPGeReport',
    'read_hpge_report',
    # Artifact I/O
    'read_artifact',
    'write_artifact',
    'read_line_activities',
    'read_peak_report',
    'read_reaction_rates',
    'read_report_bundle',
    'read_response_bundle',
    'read_spectrum_file',
    'read_unfold_result',
    'read_validation_bundle',
    'write_line_activities',
    'write_peak_report',
    'write_reaction_rates',
    'write_report_bundle',
    'write_response_bundle',
    'write_spectrum_file',
    'write_unfold_result',
    'write_validation_bundle',
    # ALARA I/O
    "ALARASettings",
    "ALARAInputGenerator",
    "parse_alara_output",
    # MCNP I/O
    "parse_mcnp_input",
    "read_meshtal_hdf5",
    # OpenMC I/O
    "read_openmc_tally",
    # CNF I/O
    "read_cnf_file",
    # N42 I/O
    "N42Measurement",
    "N42Document",
    "read_n42_file",
    "read_n42_spectrum",
    "write_n42_file",
    # Flux Wire I/O
    "FluxWireData",
    "NuclideResult",
    "EfficiencyCalibration",
    "read_raw_asc",
    "read_processed_txt",
    "read_flux_wire",
    "load_flux_wire_directory",
    # Interoperability
    "SaturationRateData",
    "read_saturation_rates_csv",
    "write_saturation_rates_csv",
    "read_lower_triangular_matrix",
    "write_lower_triangular_matrix",
    "STAYSLBundle",
    "export_staysl_bundle",
    "import_staysl_bundle",
    "MCNPSpectrum",
    "read_mcnp_spectrum_csv",
]
