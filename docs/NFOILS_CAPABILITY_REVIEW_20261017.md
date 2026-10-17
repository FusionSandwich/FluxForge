# nFoils capability review and FluxForge implementation map

Reviewed 17 October 2026. nFoils's public default branch is `develop` and its
latest reviewed revision remains
[`fb6e34d`](https://github.com/louisbutt338/nFoils/commit/fb6e34dbf23ab57f818687bb5ca07ac6f914a803)
(11 October). That change revises published-study examples and prior mathematics;
it is not evidence that all listed capabilities were newly introduced that day.
The 8 October installation revision and earlier particle-agnostic changes were
also considered. Public source inspection, not marketing wording, defines this map.

FluxForge baseline is the science integration branch
`quantumgold-methods-integration` at `701288a30f418ed7dc331e6deda0044eb220e72e`
(draft PR #248). Default `main` is substantially older. These additions form a
reviewable stack; existing draft branches are not merged or declared accepted.

| nFoils capability | FluxForge component after this stack | Verification and remaining gate |
|---|---|---|
| Bayesian user-defined parametric spectrum unfolding (`unfold.py`) | `solvers.parametric`, explicit bounded parameters, group integration, multi-chain draws | Analytical posterior and numerical quadrature tests; measured validation, modern ESS/rank-normalized diagnostics and GUI controls remain |
| Nuclear-response uncertainty (`reaction.py`, `unfold.py`) | `GaussianResponseModel` full Gaussian response marginalization, including energy/cross-reaction covariance and flux-dependent determinant | Independent latent-response integration; Gaussian/independence assumptions explicit, non-Gaussian joint nuisance inference remains |
| SANDY/NJOY response extraction (`reaction.py`) | `data.sandy_response`, labelled ERRORR mean/covariance conversion and explicit local processing; `ResponseBundle` interchange | Permuted-label/unit/contract tests; actual evaluated-ENDF external-engine execution remains unverified |
| Detector calibration and uncertainty (`fitting.py`, `coincidence.py`) | Existing measured polynomial/Gray/HPGe calibration, coincidence corrections; new cross-energy efficiency covariance | Existing detector tests and independent covariance Jacobian; certificate/geometry applicability and full experimental budget remain |
| Foil activity and reaction-rate calculation (`activity.py`) | Existing `physics.activation`, `analysis.activity_combination`, piecewise histories, count-clock correction and correlated line combination | Existing clock/activation/combination regressions; unknown systematics are not physically qualified |
| Predicted/experimental activity comparison (`ce.py`) | `analysis.prediction_comparison` full joint C/E and residual covariance; existing method controls | Jacobian and shared-normalization tests; no vendor-as-truth or superiority claim |
| Lithium-target Be-7/current normalization (`target.py`) | `physics.target_monitor`, explicit isotope areal density/charge/effective cross section, production decay and correlated uncertainty | Independent decay integral/Jacobian; thin-target assumptions, thick-target yield physics and actual source calibration remain |
| Irradiation histories (`history.py`) | Existing source-bound `physics.operating_history`; new FISPACT fragment export with explicit flux units and a solve card after each interval | Format/clock/overwrite tests; external FISPACT execution remains unverified |
| FISPACT JSON activity retrieval (`fispact.py`) | `io.fispact`, exact ZAI including isomers, explicit interval, missing distinct from zero | JSON identity/failure tests; only supported inventory_data layout is claimed |
| Gamma-emission spectrum generation (`fispact.py`) | Existing `physics.gamma_spectrum` inventory/line aggregation with explicit energy bins | Existing component; source decay data and detector response applicability still require qualification |
| ASCII/SPE spectrum reading and summation (`ascii.py`) | Existing SPE/Genie/HPGe IO; repaired native-grid summation with source hashes, covariance and acquisition intervals | IO/math regressions and new summation counterexamples; an aggregate is not automatically one contiguous decay clock |
| Published-study scripts and plots (`lab/`, `materials/`) | Existing RAFM/source forensics plus new reproducible parametric and held-out-count examples | FluxForge examples in this stack are synthetic; published lithium-source replications and a matched nFoils execution have not been performed |

## What the additions improve, and how to test that claim

The nFoils default unfolding path perturbs individual response entries and pools
response-conditional chains with equal response weights. This does not generally
equal a joint response-marginal posterior. The new normalized Gaussian likelihood
retains covariance and integrates that explicitly declared response model exactly.
This is a specific numerical/statistical capability improvement, not universal
evidence of greater accuracy: a misspecified Gaussian response or spectrum family
can still fail. nFoils permits custom models/priors and can be extended as well.

The nFoils revised lithium example does not enforce all its stated parameter
bounds in the inspected prior predicate. FluxForge's new interface checks every
bound before calling the model. FluxForge's own legacy log-flux proposal also
needed a Hastings correction for its documented linear-flux target; #254 supplies
that correction and an independent target-distribution test. The comparator must
be prepared fairly, preserving stock and corrected/extended runs separately.

Cross-energy efficiency covariance, reaction-family splitting, source integrity,
native-bin predictive scoring and an explicit irradiation-to-count calculation
make uncertainty assumptions and failure modes more inspectable. They are useful
engineering/statistical improvements; these methods themselves are prior art.
SANDY/NJOY and detector calibration are reused established tools and methods.

## Independently testable differentiating result

Use the procedure in [COUNT_PREDICTION_BENCHMARK.md](COUNT_PREDICTION_BENCHMARK.md).
Run a common-rate inference track and a common-original-count workflow track.
Freeze matching source/calibration/nuclear inputs, priors, family split, nuisance
handling and scores. Publish absolute held-out count predictions before evaluation
and retain all failures. Include the unchanged transport/prior and GLS baselines.
Use joint predictive scores plus count adequacy and predictive coverage/width;
several independent experimental blocks are needed for a superiority claim.

Minimum data are identified specimen/exposure and sufficiently informative
training responses plus one withheld reaction family; integer spectra with native
grids/live/real/start times; foil mass/isotopic composition/covers/position;
irradiation/power history; applicable independent efficiency calibration and
covariance; background records; and versioned response/decay data with units and
uncertainty. Holding out a later count or one line is a narrower detector/decay
test. Previously inspected RAFM measurements are retrospective challenges.

Current RAFM evidence does not satisfy all those physical gates. Unknown South
calibration covariance/applicability, unresolved clock/source identities and
inadequate weak Co-Cd fits are not repaired by adding an unfolding method. Vendor
agreement and numerical convergence cannot establish physical qualification.

## Scope of completion

This stack supplies code paths across the documented capability categories and
adds stricter covariance and predictive-validation tools. It is not a completed
cross-package benchmark, a full GUI integration, an externally executed NJOY or
FISPACT validation, a published-study reproduction, or demonstrated experimental
superiority. The response-marginalized solver also does not automatically generate
joint conditional detector/nuclear nuisance draws for the count model. Those
must be supplied under an appropriate joint model before claiming a complete
experimental predictive uncertainty budget.

Potentially publishable contributions are a traceable validation dataset,
demonstrated out-of-sample measured-count prediction, or a reproducible physical
failure diagnosis. GLS, GRAVEL/MLEM, Bayesian parametric inference, Poisson fitting
and Gaussian conditional prediction are established prior art. The new synthetic
checks establish bounded software behavior only.
