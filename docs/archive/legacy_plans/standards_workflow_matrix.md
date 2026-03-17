# Standards Workflow Matrix

This note records the current FluxForge coverage for the standards-oriented workflows discussed during the RAFM / parity work and the external capability inventory in `testing/writeup.md`.

## Implemented workflow families

### ASTM / INL reactor dosimetry

Status: implemented, tested, surfaced in GUI presets, and now includes a dedicated ASTM E261 CLI/GUI reduction workflow

Core components:
- `fluxforge.data.rafm_profile` bundled profiles
- `fluxforge.data.irdff` IRDFF-II reaction data and response-matrix helpers
- `fluxforge.workflows.spectrum_unfolding.SpectrumUnfolder`
- `fluxforge.analysis.flux_wire_selection`
- `fluxforge.analysis.astm_e261`

Tests:
- `tests/test_flux_wire_parity.py`
- `tests/test_rafm_background_integration.py`
- `tests/test_rafm_workflow.py`
- `tests/test_irdff.py`
- `tests/test_irdff_access.py`
- `tests/test_astm_e261.py`

GUI support:
- ASTM/INL preset
- US ASTM reactor dosimetry preset
- IRDFF reaction browser
- Standards-tab ASTM E261 runner and preview

Dedicated ASTM E261 documentation:
- [ASTM E261 workflow](astm_e261_workflow.md)

### IAEA IRDFF / GMA style dosimetry

Status: implemented in code, tested, now explicitly surfaced in GUI presets

Core components:
- `fluxforge.data.irdff`
- `fluxforge.workflows.spectrum_unfolding`
- `fluxforge.solvers.advanced`
- `fluxforge.core.validation`

Tests:
- `tests/test_irdff.py`
- `tests/test_irdff_access.py`
- `tests/test_unfold.py`
- `tests/test_validation.py`
- `tests/test_epic_implementations.py`

GUI support:
- `IAEA IRDFF / GMA dosimetry` preset
- existing Unfold / Compare tabs
- IRDFF reaction browsing on the Standards tab

Notes:
- This is an IAEA-aligned dosimetry workflow, not a dedicated IAEA-clearance-release workflow.
- Clearance data files exist in the broader ALARA workspace, but FluxForge does not yet expose a first-class clearance-assessment GUI or CLI workflow.

### IAEA / k0-NAA

Status: implemented, unit-tested, surfaced in GUI presets

Core components:
- `fluxforge.analysis.k0_naa`
- `fluxforge.triga.k0`
- `fluxforge.triga.reconcile`
- `fluxforge.uncertainty.budget.create_k0_naa_budget`

Tests:
- `tests/test_k0_naa.py`
- `tests/test_uncertainty_budget.py`

GUI support:
- `k0-NAA` preset
- manual ROI, background-subtracted peaks, and activity overrides

Notes:
- The GUI currently presets the existing CLI-mapped tabs; it does not yet provide a dedicated comparator mass / k0 factor form.

### Comparator NAA

Status: partially implemented, tested indirectly, surfaced in GUI presets

Core components:
- manual ROI peak path in CLI / GUI
- activity and reaction-ID overrides in activity processing
- report / rates bundling

Tests:
- covered indirectly by CLI and GUI tests

GUI support:
- `Comparator NAA` preset

Gaps:
- no dedicated comparator-standard command or structured comparator metadata form yet

### Curie-style activation / spectroscopy parity

Status: implemented in backend modules, tested, now explicitly surfaced in GUI presets as an API-backed workflow

Core components:
- `fluxforge.physics.decay_chain`
- `fluxforge.physics.stacked_target`
- `fluxforge.physics.stopping_power`
- `fluxforge.physics.attenuation`
- `fluxforge.physics.dose`
- `fluxforge.io.spectrum_export`

Tests:
- `tests/test_epic_implementations.py`
- `tests/test_attenuation.py`
- `tests/test_gamma_attenuation.py`
- `tests/test_stopping_tools.py`
- `tests/test_decay_schedule.py`

GUI support:
- `Curie-style activation / spectroscopy` preset for discoverability

Gaps:
- stacked-target and decay-chain parity modules are still Python-API-first rather than dedicated GUI tasks

## Capability summary against the external writeup

Broadly covered in FluxForge:
- IRDFF-based dosimetry and unfolding
- GMA-like advanced GLS and LM adjustment pieces
- k0-NAA parameter calculations
- NAA-ANN support
- SNIP background estimation and spectroscopy helpers
- XCOM attenuation
- Bateman decay chains
- stacked-target energy degradation
- dose-rate calculations
- SPE export

Still not first-class GUI workflows:
- IAEA clearance / release-limit assessment
- dedicated comparator-NAA form workflow
- dedicated stacked-target task flow
- direct notebook-style workflow shells for all external repo patterns
