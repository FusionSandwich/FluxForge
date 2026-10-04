# Same-line activity combination control (#238)

Run `python examples/activity_combination/compare_same_lines.py --output NEW.json`
from a checkout descended from `4615e61bbb262d974e0326a44107bc952e4cb903`.
Existing output files are refused. Run the bounded tests with
`python -m pytest -q tests/test_activity_combination.py`.

`SOURCE_MANIFEST.json` binds two byte-identical receipts copied additively from
`46096eb:artifacts/validation/quantumgold_documentation_20261003`. The example
checks those hashes before use. No original spectra, manual PDF or private chat
links are copied. Source reductions are **not** rerun or attributed to the current
engine: their engine identity is not established by these two receipts. The
current descendant executes the new aggregation module only; its checkout commit,
module hash, Python executable/version and NumPy version are exported. Local
testing used Python 3.12.10 / NumPy 2.5.1 / pytest 8.4.2; NumPy is outside the
project's declared `<2.0` range. No environment was changed; full supported-runtime
and workflow verification remain integration work.

The first group freezes the two selected Co60 ASC activities and declared errors.
The second uses rounded report activities and count-error-only reconstructed
errors. The historical A/sigma equation and inverse variance equation reproduce
the receipt values; diagonal GLS is the same independent-error control. These
three methods see identical inputs **within each group**. Source counts, yields,
efficiencies and qualification do not change. The Co-Cd method shift remains
approximately -0.166974%; no target activity is used for fitting. Other rows in
the archived receipt, including unqualified Ni57 and inconsistent Sc48, are not
admitted. Tests separately preserve their upstream exclusions for every method.

Missing efficiency/yield covariance is declared unavailable, never inferred from
zero saved settings. Complete uncertainty GLS is unavailable; explicitly opted-in
partial controls are conditional. The separate synthetic physical GLS group has
four equal 100 Bq activities, 2 Bq independent count errors and 5 Bq fully shared
calibration error. Its variance is `25 + 4/4 = 26 Bq²`, demonstrating the shared
error survives averaging. This synthetic calibration is not assigned to study
data, and no absolute physical accuracy or whole-workflow parity is claimed.

The standalone API requires a method, analysis role, engine identity, uncertainty
definition, per-line provenance, count basis, reference time and frozen boolean
qualification with a reason. It never selects or rejects lines by fit residual.
Unqualified rows stay in exclusions. Complete additive covariance components must
each be PSD and have provenance; their summed diagonal must match declared
one-sigma errors. For partial budgets those errors must describe the same partial
budget. Do not pass both a total covariance and its counted subcomponents.

Normalized A/sigma and inverse-variance weights use log ratios without uncertainty
floors. GLS minimizes variance with weights summing to one and exports negative
weights without clipping. Propagated error is `sqrt(w.T C w)`, conditional on fixed
weights; it does not include uncertainty from estimating historical data-dependent
weights or silently inflate error for scatter. A positive-definite GLS model with
condition number above `1e12` is unavailable. Singular/numerically unresolved GLS
is also unavailable by default. `singular_policy="exact_constraints"` explicitly
declares unresolved modes exact; the source must justify that assumption. Rank
tolerance/eigenvalues and null-constraint residuals are exported. Contradictory
exact constraints return `inconsistent`, retaining the mathematical estimate only
as a diagnostic. Fixed-weight methods can propagate a supplied singular matrix
without inversion; exact variance cancellation is explicit and unresolved
rank-dependent diagnostics stay unavailable without the exact-model opt-in.

Residual uncertainty includes covariance between each input and the combined
estimate, using `(I - 1 w.T) C (I - 1 w.T).T`. Centered arithmetic preserves small
contrasts at large activity baselines. Three-sigma line flags are diagnostics;
they do not change admission. Chi-square is a GLS goodness-of-fit diagnostic;
for other weights it is descriptive. JSON serialization rejects nonfinite derived
diagnostics rather than publishing Infinity or a fabricated zero.

Integration owner: chat `01a103df-5eda-7cd3-b276-2ad70d21decc`, issues #232/#220.
The existing `combine_peak_activities`, workflow, report, GUI and physical defaults
are unchanged. An adapter must freeze the existing qualified set once, distinguish
physical and comparison count bases, persist the explicit method choice and
surface unavailable/conditional/inconsistent results before integrating this API.
