# Repeated-count time-reference diagnostic (#241)

This component uses engine base `a7bcc680d1f5e06b1d9dae405241fc380087ca2b`,
a locally verified descendant of `4615e61bbb262d974e0326a44107bc952e4cb903`.
It reuses `GammaLineMeasurement` and `count_decay_factor` from the current engine;
the shared activation module is byte-identical to the frozen validation engine.
No historical core, shared workflow, schedule alias, calibration default or GUI
is changed. The original Ti reports already present in this engine checkout are
byte-identical to the three reports at historical evidence commit `46096eb`.
`fixtures.json` records those joins and hashes; no duplicate originals are needed.

From the repository root, using existing Python and runtime dependencies:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/test_repeated_count_validation.py tests/test_activation_count_clock.py tests/test_count_decay_time.py
python examples/RAFM_irradiation/repeated_count_validation/run_example.py --output new_receipt.json
```

The example refuses an existing output file. It reads exactly three reports,
checks hashes before/after analysis, and performs no spectrum fits or full campaign.
The final receipt is `EXAMPLE_RECEIPT_final.json`. Earlier receipts are retained
locally as superseded tranche evidence and are not PR inputs. `TEST_RECEIPT.json`
and `SOL_REVIEW.json` bind validation to the final component hashes.

## Method and assumptions

`CountObservation` declares net accepted counts, a live-normalized count-average,
or a processed point activity. Net counts use the canonical relation
`C = efficiency * gamma_probability * A_start * L * count_decay_factor(half_life, R)`.
Decay is integrated over real time `R`; live acceptance is handled once through
`L` under the explicit `uniform_live_fraction` assumption. Long half-life limits
use the existing stable `expm1` primitive. Processed point activities require the
actual boolean `includes_count_decay=True` and a point-reference choice; they
receive zero finite-window corrections. A processed count-average requires False
and receives one. Conflicting declarations are unavailable.

The point-reference choices are count start, count end, EOI, or a supplied
timestamp. Count end may be derived from a known start and measured real duration;
EOI and missing start timestamps are never inferred. Unknown time zones are
unavailable unless a named shared-naive-clock scenario is deliberately selected.
Aware timestamps use elapsed UTC, reject nonexistent local DST times, and honor
valid DST folds. Overflow and underflow do not emit infinite or false-zero results.

Pairwise comparison requires unique measurement IDs and source hashes, nonoverlapping
acquisitions, matching specimen/nuclide/line/calibration/background/count-basis/unit/
half-life identities, and a declared simple-decay/no-feeding/no-reirradiation model.
Different known EOIs are incompatible. Unknown irradiation history remains unknown:
the interval decay assumption does not invent buildup, saturation or a reaction rate.
Input IDs and hashes are caller declarations; the reusable module checks identities,
while an upstream source adapter must verify actual input bytes as this example does.
Files containing several acquisitions need a future explicit source-segment binding;
this file-bound API conservatively rejects a repeated hash.

The residual is `log(A_j/A_i) + lambda*(t_j-t_i)`. The caller-selected tolerance is
symmetric: `abs(residual) <= log1p(relative_tolerance)`. This deterministic diagnostic
is not a statistical significance test; it assumes no uncertainties or covariance.
Every pair must be available and consistent for overall CONSISTENT. Mixed evaluated
and unsupported pairs produce PARTIAL. Unknowns have null values and reasons;
incompatible identities and excluded sources remain distinct from decay discrepancies.
All outputs retain `scientific_admission=False`; none emits a reaction rate or flux.

## Bounded Ti outcome

Strict source validation cannot qualify these counts: the catalog preserves three
distinct labels, clocks have no verified zone, background settings are unqualified,
and “Measurement Date” does not establish the precise report correction convention.
Consequently processed reference validation never supplies a corrected Bq activity.
The [source evidence](https://github.com/FusionSandwich/FluxForge/blob/46096eb/artifacts/validation/quantumgold_documentation_20261003/FINDINGS.txt)
is a saved-settings investigation, bounded to 32 study files; it cannot prove final
report settings. The [PGT manual](https://ludlums.com/images/product_manuals/QTMmanual.pdf)
is linked as context, not bundled or used to infer unknown per-report choices.

The separate conditional scenario explicitly assumes one Ti wire, a common naive
clock, time-independent printed response at the same gamma line, comparable vendor
background treatment, simple decay, and uniform live acceptance. It compares report
net counts divided by live time, with one canonical finite-window correction.
Its units are `response_scaled_counts_per_s`, never Bq. Comparing identical lines
cancels the assumed constant response; cross-energy comparisons are not admitted.
A common fixed 5% tolerance is chosen for reporting, with no per-sample target tuning.

| Same-line comparison across three counts | Conditional result |
| --- | --- |
| Sc-47, 159.4 keV | All three pairs consistent |
| Sc-46, 1120.5 keV | All three pairs consistent |
| Sc-46, 889.3 keV | Discrepancy in the 1a pairs |
| Sc-48, all four report lines | Excluded; ambiguous yields and inconsistent references |

A discrepancy alone does not identify a timestamp fault, a calibration drift,
background bias, or a peak extraction problem. Sc-48 source values are retained
without yield substitution. These component checks establish neither absolute
activity accuracy nor scientific/full-workflow parity.

## Proposed integration hook (owner #232/#220)

Construct observations only after the existing source-identity/timing adapter has
bound spectrum/report hashes, count basis, method, response/geometry and background
scenario. Preserve unknown fields, explicitly declare the chosen decay and live
acceptance assumptions, then call `compare_repeated_counts`. Export the conditional
status and each pair's reasons/residuals alongside originals. Do not use a diagnostic
CONSISTENT status to admit reaction rates or change defaults. Existing #198 correction
and #234/#235 schedule/rate repair ownership stays with the shared integration task.
