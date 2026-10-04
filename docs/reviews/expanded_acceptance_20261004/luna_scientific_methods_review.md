# Luna scientific methods review

Reviewed read-only at `c8a996efd029d691280ffd0c1988a79e1819d36e` in the integration worktree. Scope: `qg_protocol.py`, `activity_combination.py`, `joint_poisson.py`, `multiplet_validation.py`, `repeated_count_validation.py`, `physics/efficiency_fidelity.py`, and their declared method limits/examples. No source changes or tests were run.

## Findings

No concrete mathematical or data-integrity blocker was identified in these six additive methods. The main equations and numerical boundaries are made explicit: ambient and sample spectra retain separate native grids and nonnegative Poisson observations; decay normalization integrates the live-accepted finite count window once; activity aggregation checks covariance diagonals/PSD and reports singular modes rather than regularizing; and the multiplet diagnostic compares the same ROI/noise/response family with single and broad-single controls. The efficiency module keeps source export, report-effective reconstruction, and unvalidated model alternatives separate, with geometry/range checks and no activity mutation.

The qualification limits are material and should travel with any integration. QG saved-state reconstruction is scoped to byte/hash-pinned rev4 study files and explicitly sets `exact_vendor_parity=false`; saved settings do not establish final-report processing. The PGT/McMaster implementation details, modern evaluated yields, certificate/covariance evidence, and independent absolute-efficiency validation remain unavailable. The source-export percent/fraction choice is conditional. The alternative XCOM-shaped curve is explicitly unvalidated and must not become a default.

`joint_poisson` profile intervals use an asymptotic chi-square likelihood-ratio cutoff; they are not calibrated for sparse counts or nuisance boundaries and exclude response/efficiency systematics. Fit convergence alone is not model adequacy or physical applicability. `multiplet_validation` is high-count weighted least squares with fixed external separation/resolution, not an isotope identifier; its component-evidence strings are caller attestations, not independently verified facts. `repeated_count_validation` assumes simple decay with no feeding/re-irradiation and a caller-declared shared clock/acceptance; it is a consistency diagnostic only and does not infer reaction rate or source identity. `activity_combination` propagates fixed weights, does not inflate for scatter, and complete GLS is unavailable when required covariance is missing.

## Usage and validation status

The APIs and bounded examples are present, but these new methods remain opt-in analysis/example functionality; the production workflow/UI does not thereby gain qualified QG parity or scientific admission. No current engine result from these primitives should be described as physically admitted. I did not rerun the focused tests per the read-only bounded-review request; this receipt records source inspection only. The prior example documentation and receipts explicitly preserve unavailable vendor methods/uncertainties rather than filling them with assumptions.

