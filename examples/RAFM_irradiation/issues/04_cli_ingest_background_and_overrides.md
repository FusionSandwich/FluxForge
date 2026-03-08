# [CLI] Add ingest options for background and overrides

## Goal
Make background subtraction and metadata overrides accessible through CLI ingest.

## Scope
- Add CLI args:
  - `--background-file`
  - `--background-scale-mode`
  - `--background-scale-factor`
  - `--energy-calibration`
  - `--efficiency-coefficients`
- Enforce user-override precedence over file headers.

## Acceptance Criteria
- Ingest works for `.asc/.txt/.spe` with optional background file.
- Manual scale mode applies expected factor.
- CLI tests cover parser and subtraction behavior.
