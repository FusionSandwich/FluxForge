# FluxForge GUI Capability Program

**Status:** Active  
**Scope:** Native Windows/Linux desktop GUI parity, plot parity, offline delivery, and interactive-analysis coverage.

## Purpose

This document defines how FluxForge closes the remaining gap between:

- the current native desktop GUI,
- the CLI and reusable core APIs,
- the plotting and notebook-style behaviors studied in developer-side reference audits.

Detailed provenance and repo-by-repo comparison notes remain in the developer-side `testing/` materials. FluxForge’s product repo tracks only the adoption contract, acceptance evidence, and implementation targets.

## Product Rules

- FluxForge remains a native desktop application built with `Tkinter + ttk + Matplotlib`.
- Supported Windows/Linux workflows must run without a browser and without internet access.
- Any adopted interactive capability must exist in four places:
  - reusable core logic in `src/fluxforge/`,
  - CLI entry point in `src/fluxforge/cli/app.py`,
  - desktop GUI surface in `src/fluxforge_gui/`,
  - saved artifact, report, or exported figure that can be replayed later.
- GUI actions must continue to expose or copy the equivalent CLI command.
- Plotting parity is measured from shared artifact-driven renderers, not from separate GUI-only figure code.

## Capability Buckets

### Desktop analysis ergonomics

- Spectrum loading, overlay management, ROI editing, calibration fitting, peak counting, and export workflows must be fully usable from the desktop GUI.
- Dense analysis panels must remain reachable on common lab displays. The current Spectrum workspace now uses a scrollable controls rail so ROI, calibration, efficiency, and fit tools remain accessible during native desktop runs.
- Native acceptance evidence must include screenshots, saved artifacts, and copied CLI commands from an actual GUI session.

### Plot parity

- GUI preview figures, exported GUI figures, CLI `spectrum-plot`, and CLI `plots` outputs must come from shared rendering helpers.
- Adopted plot families include:
  - spectrum view and ROI overlays,
  - calibration residuals,
  - activity and rate uncertainty summaries,
  - unfold diagnostics, parity plots, and covariance/correlation views,
  - report figure bundles and master plot suites.
- New plot work is not complete until both GUI and CLI paths produce the same artifact class and a parity test exists.

### Standards-driven coverage review

- FluxForge now exposes first-pass GUI and CLI surfaces for the standards backbone already called out in the product review:
  - ASTM E261 / activation-foil reduction,
  - ASTM E262,
  - ASTM E3376 detector-calibration workflow support,
  - HPGe spectrum review and activity/rate derivation,
  - response build, unfold, compare, report, and master plot export,
  - k0 workflows including Kayzero import and governed-library preview/reporting.
- The desktop GUI now includes live plot panels for:
  - spectrum inspection with default log-count view, grouped isotope-colored peak markers, ROI overlays, and CLI-equivalent export,
  - activity uncertainty review,
  - reaction-rate uncertainty review,
  - unfold diagnostics.
- The standards/manual review also identifies remaining gaps that are still tracked as product work rather than being silently implied as complete:
  - true-coincidence summing and pile-up/random-summing correction workflows,
  - full detector-profile and geometry-transfer wizards,
  - explicit f/alpha multi-monitor facility-characterization assistants,
  - formal MDA / peak-free-region detection-limit workflows,
  - richer QA/QC trending across multiple irradiations and spectra,
  - broader accepted-format import beyond the current SPE / Genie / FluxForge artifact path.
- Any standards-aligned claim in docs must keep this distinction explicit: implemented surfaces are available in both CLI and GUI; advanced correction and QA modules remain roadmap items until their artifact-backed tests exist.

### Notebook-to-workflow translation

- Notebook-only analysis behavior must be translated into explicit FluxForge workflows, tables, plots, and saved artifacts.
- FluxForge does not embed notebook execution as a product feature.
- Adopted notebook-style capabilities must land as guided GUI panels, report previews, exported figures, or reusable CLI outputs.

### Cross-platform and offline delivery

- Native Linux and native Windows remain required release targets.
- `FLUXFORGE_OFFLINE=1` is the offline contract for supported workflows.
- Packaging acceptance requires:
  - offline Python installs from a wheelhouse,
  - native desktop launch on Linux,
  - native desktop launch on Windows.

## Acceptance Map

| Capability Area | Core Surface | CLI Surface | GUI Surface | Acceptance Evidence |
|---|---|---|---|---|
| Spectrum inspection | spectrum IO + plot helpers | `ingest`, `spectrum-plot` | Spectrum tab | native desktop screenshots, saved preview PNG, copied CLI |
| ROI and calibration editing | spectrum ops + calibration fit | `spectrum-plot`, downstream peak/activity flows | Spectrum tab ROI + calibration panels | native desktop ROI create/edit, calibration fit, saved ROI JSON |
| Peak analysis and counting | peak finders + counting backends | `peaks`, `activity`, `rates` | Spectrum + Peaks tabs | helper regressions plus native desktop artifact runs |
| Response / unfold / compare diagnostics | response builders + unfold + validation | `response`, `unfold`, `compare` | Unfold + Compare tabs | plot parity tests and saved report figures |
| Report and plot bundles | reporting + plot suite helpers | `report`, `plots` | Report tab | native desktop plot-suite run, figure bundle artifacts, copied CLI |
| Standards and k0 workflows | standards/k0 APIs | ASTM + `k0-*` commands | Standards tab | artifact-based workflow tests and future native desktop acceptance tasks |
| Offline source policy | runtime + nuclear-data source loaders | all supported commands | offline-aware GUI state and logs | offline tests with blocked remote sources |

## Native Desktop Test Contract

- Helper/unit GUI tests remain useful, but they are no longer sufficient as primary acceptance.
- Primary desktop acceptance now requires:
  - a real Tk window,
  - visible screenshots,
  - event-driven interaction through the desktop UI,
  - generated artifacts on disk,
  - copied CLI command evidence.
- The current Linux acceptance entry point is `tests/test_gui_desktop_native.py`, backed by `tests/gui_desktop_driver.py`.
- Windows uses the same acceptance test file with a Windows-specific input backend.

## Next Implementation Targets

1. Extend native desktop acceptance from the current Spectrum + Report workflow to Peaks, Standards/k0, and Unfold/Compare flows.
2. Expand the scrollable-control treatment to other dense tabs that still exceed common screen heights.
3. Add artifact-backed plot parity tests for every adopted plot family.
4. Keep the detailed capability provenance in the developer-side `testing/` matrix, but mirror each adopted item here as a FluxForge acceptance requirement.
