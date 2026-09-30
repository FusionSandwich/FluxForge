# Publication integrity continuation

This feature integrates the frozen PR205 physical GLS head `3c28ed9`, PR207
provenance head `9d04eff`, and PR210 response head `cff9dc7`. It repairs software
contracts for #211, #195, #199, #203, #204 and #187. Scientific admission remains
with the publication audit owner.

## Response cache (#211)

Every request validates current covered-monitor specifications, physical
constructors, evaluated arrays, and positive increasing group boundaries before
checking the cache. Its fingerprint binds observation and physics identities,
database identity, exact boundaries, evaluated values/uncertainties/provenance,
and the archive pin. Current reaction and absorption source bytes are checked
against loaded evaluations. Changed or missing source bytes require a fresh
database; changed valid physical inputs rebuild the response. Rate-only changes
reuse the operator. Public results own copies of response, boundaries, identities
and response metadata so editing a result cannot poison subsequent cache hits.

Hashing the local evaluated archives on each request costs local I/O. It avoids
using modification time as proof of source identity.

## Monte Carlo qualification (#195)

`SpectrumUnfolder.unfold(uncertainty_method="monte_carlo")` defaults to
`uncertainty_estimator="converged"`. All requested draws must be finite and
converged, and the main fit must converge. The derived covariance and standard
deviation must also be finite; overflow yields unavailable with a numerical reason. Otherwise uncertainties are NaN and
qualification is `unavailable`. Failed draws are recorded, never selected away
to compute a spread over the survivors. Each draw records its convergence,
finite status, stopping reason, iteration count and number of clipped rates.

Explicit `uncertainty_estimator="capped"` estimates the spread of the capped
algorithm if every requested draw is usable. It records `capped_diagnostic` and
includes the full sample flux covariance. This spread is conditional on the
recorded rate-error model and fixed response/prior; it is not a measured total
spectrum uncertainty. The clipping of Gaussian negative rates is recorded.

## Rate budgets and covariance (#199)

`UncertaintyComponent.from_input` propagates a supplied standard uncertainty
through a signed derivative of log rate. Relative components can declare named
`covers` terms already included in a source aggregate. Overlapping coverage is
rejected to prevent counting a detector term twice when an itemized activity
error already includes it. A source specification replaces the same named
opaque term. Unnamed missing terms remain missing.

Source specifications may be in `rate_uncertainty_components` globally or per
wire, or in `rate_uncertainty_budgets[observation_id]` for a specific observation.
Each specification supplies `source`, optional `correlation_group`/`covers`, and
either `relative` or `standard_uncertainty` plus `log_sensitivity`. Group keys
identify the same named shared component; no group means independent. Opposite
sensitivities produce negative cross-covariance. Model floors are recomputed
after source replacements and remain explicit diagnostic terms.

Measurements accept `rate_uncertainty_budget`; `unfold` accepts a full
`rate_covariance`. Budget identities/rates must match current observations and
its covariance diagonal must match reported errors. A separately supplied
matrix must agree with supplied budgets. `require_complete_rate_budget=True`
rejects missing component/source coverage, including a bare matrix without
budgets. Complete declared coverage is still separate from independent source
qualification: receipts always retain `scientific_admission=False`.

Compatible duplicate rows use covariance-aware linear weights and retain
`T C T^T`, including covariance between groups and fully shared errors that
cannot shrink with replication. A transformed spectral factor forms the aggregate
Gram covariance to preserve positive semidefiniteness when shared errors cancel.
Replicates inconsistent with noiseless covariance directions are rejected rather
than averaged away. Monte Carlo samples the full covariance.
GRAVEL/MLEM objectives remain diagonal; their metadata states this limitation.
Physical GLS remains the primary interface for a fit with full covariance.

## Standalone holdouts, abundance and GLS receipt

Standalone holdout prediction validates symmetric PSD prior and observation
covariance at their physical scales (#203). Singular and noiseless models use
spectral conditioning; incompatible deterministic data are rejected. Compatible
zero-variance directions contribute zero to the residual statistic and are not
interpreted as uncertain measurements.

Missing `isotope_abundance=None` requests natural lookup (#204). Every explicit
finite fraction in `(0, 1]`, including `1.0` and `0.999999`, is preserved exactly.
The unfolding metadata records the supplied/effective fraction and basis.

The physical GLS receipt names `postfit_total_error_chi2` and defines its
covariance as observation plus response error (#187). The old
`postfit_observation_chi2` receipt field is a labeled compatibility alias. This
postfit diagnostic has no automatic goodness-test or publication interpretation.

## Actual-data replay and remaining gates

`tools/validate_monitor_response_rafm.py` reconstructs 13 Co/Sc/Ti observations
from the existing hash-bound QG/source join. The continuation tool
`tools/validate_publication_integrity_rafm.py` compares them with the frozen
PR210 receipt, checks warm/fresh cache equivalence, rejects missing cover data,
records bounded MC qualification, and tests the physical GLS receipt on the
same operator. Its prior and 1% response-error term are explicitly synthetic
interface diagnostics. Existing audit receipts and raw input bytes are retained.

All 13 actual rate budgets lack detector efficiency, gamma yield, half-life,
target mass, isotopic abundance, and irradiation history components. Their
reported activity errors have unitemized composition. The strict source budget
gate rejects every row; no missing component is fabricated or treated as zero.

The frozen common operator has rank 7 over 20 groups. The audit owner's NNLS
weighted residual is 241.600385 with a KKT gap of 8.53e-13, whereas signed LS
has residual 1.455869 with 11 negative groups. These are diagnostics under an
unqualified input/response/covariance contract, not a newly established solver
bug or physical impossibility. See issue #191. No measured values or model
floors were adjusted to force agreement.

SpecKit's original run settings and quasi-single benchmark remain unresolved;
its reported objective omits the implemented smoothness gradient. Comparator
agreement is not experimental admission. Measured history, calibration,
geometry/certificates, covariance and comparator/source qualification remain
with the publication audit owner. No external solver repair is included here.
