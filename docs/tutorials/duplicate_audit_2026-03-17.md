# FluxForge Duplicate Audit (2026-03-17)

## Scope
Systematic folder-by-folder duplicate-code audit across `src/fluxforge` and key `examples/RAFM_irradiation` scripts.

## Decision Rules
- `safe-unify`: behaviorally equivalent utility logic can be centralized.
- `flex-helper`: similar logic with slight contextual differences can use one helper with explicit mode/flags.
- `keep-separate`: standards/workflow behavior differs and must stay explicit unless user opts in.

## Analysis (`src/fluxforge/analysis`)
- `safe-unify`: k0 timing and Q0(alpha) helpers were unified into `fluxforge.k0_physics`.
- `keep-separate`: ASTM workflow modules (`astm_e2005.py`, `astm_e261.py`, `astm_e262.py`, `astm_e3376.py`) should remain separate due to standard-specific equations and report semantics.
- `flex-helper`: repeated path/bootstrap glue in example compare scripts moved to `examples/RAFM_irradiation/_compare_common.py`.

## Data + IO (`src/fluxforge/data`, `src/fluxforge/io`)
- `safe-unify`: isotope/nuclide name parsing and formatting helpers were centralized in `fluxforge.data.isotope_names`, with compatibility wrappers retained in `elements.py`, `nndc.py`, and `gamma_database.py`.
- `safe-unify`: repeated whitespace/comma numeric-text parsing in `io/n42.py` was centralized into one internal helper (`_parse_numeric_values`) reused by channel and calibration parsing.
- `safe-unify`: missing-field QC detection was centralized with reusable helpers in `io/metadata.py` (`is_missing_value`, `append_missing_field_flags`) and adopted in `io/csv_readers.py`.
- `safe-unify`: required-float CSV parsing with QC flagging in `io/csv_readers.py` was consolidated into `_parse_required_float`, replacing repeated inline parsing logic.
- `keep-separate`: spectrum formats (`spe`, `iec`, `n42`, `cnf`) contain format-specific parsing and metadata differences.
- `flex-helper` candidate: shared metadata/QC helpers for missing fields should be centralized only after fixture-backed parity tests.

## Physics (`src/fluxforge/physics`)
- `flex-helper` candidate: decay/saturation/counting factors overlap with k0 helpers but some implementations include numerical shortcuts and domain assumptions; consolidate only with explicit tests.
- `keep-separate`: dose and attenuation pipelines differ in physics model scope.

## TRIGA (`src/fluxforge/triga`)
- `safe-unify`: shared k0 timing and Q0(alpha) now delegated to canonical helper module.
- `keep-separate`: monitor-method orchestration and TRIGA-specific conventions remain in TRIGA modules.

## Workflows (`src/fluxforge/workflows`)
- `keep-separate`: activation, unfolding, and standards workflows differ in objective and artifacts.
- `flex-helper` candidate: common C/E summary stat utilities can be extracted without changing workflow equations.

## Solvers (`src/fluxforge/solvers`)
- `keep-separate`: solver math and constraints differ (GLS, iterative, RMLE).
- `flex-helper` candidate: shared reporting/normalization utility should be extracted if and only if output parity tests are added.

## Examples (`examples/RAFM_irradiation`)
- `safe-unify`: shared path/bootstrap and sample-pairing logic extracted to `_compare_common.py`.
- `keep-separate`: standards comparison scripts keep independent plan payload construction and ASTM-specific semantics.

## Guardrail
No standards equations are made user-flexible by default. If flexibility is introduced, it must require explicit method selection and preserve current default behavior.