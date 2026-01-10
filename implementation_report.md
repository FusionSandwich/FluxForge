# Implementation Report: MCNP-ALARA Integration

## Overview
This session focused on addressing the "Hybrid Modeling" gaps identified in the capability analysis. Specifically, we implemented the integration between MCNP (transport code) and ALARA (activation code).

## Implemented Features

### 1. MCNP I/O (`src/fluxforge/io/mcnp.py`)
*   **Input Parsing**: Added `parse_mcnp_input` to extract material definitions (composition, density) from MCNP input files (`.i`).
    *   *Status*: Verified with `tests/test_mcnp_io.py`.
*   **Tally Ingestion**: Added `read_meshtal_hdf5` to read neutron flux and relative errors from MCNP HDF5 output files (`meshtal.h5` or `runtpe.h5`).
    *   *Dependency*: Requires `h5py` (optional).
    *   *Status*: Verified with `tests/test_mcnp_io.py`.

### 2. ALARA I/O (`src/fluxforge/io/alara.py`)
*   **Input Generation**: Added `ALARAInputGenerator` class to programmatically create ALARA input files (`.inp`).
    *   Supports configuration of geometry, materials, flux files, irradiation schedules, and cooling times via `ALARASettings`.
    *   *Status*: Verified with `tests/test_alara_io.py`.
*   **Output Parsing**: Added `parse_alara_output` to extract results from ALARA output files (`.out`).
    *   Parses total specific activity and decay heat for different cooling times.
    *   *Status*: Verified with `tests/test_alara_io.py`.

## Verification
*   Created unit tests in `tests/test_mcnp_io.py` and `tests/test_alara_io.py`.
*   All tests passed successfully.

## Remaining Gaps
*   **CNF File Support**: Requires a library to read Canberra binary format. Existing `hdtv` solution relies on C libraries.
*   **Coincidence Summing**: Requires implementation of cascade summing corrections. Existing `actigamma` library only handles binning, not true coincidence corrections.

## Recommendations
*   For CNF support, consider using an external tool to convert CNF to SPE (text format) which is easier to parse, or investigate `specutils` or similar Python packages.
*   For Coincidence Summing, a dedicated physics module implementing algorithms like Vidmar's method is needed.
