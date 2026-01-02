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
]
