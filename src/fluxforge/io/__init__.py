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
]
