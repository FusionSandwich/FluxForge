# Joint Poisson first tranche — issue #239

This opt-in component fits **original nonnegative integer native counts**, not a
signed ambient-subtracted residual. Engine base is
`a7bcc680d1f5e06b1d9dae405241fc380087ca2b`, verified locally as a descendant of
`4615e61bbb262d974e0326a44107bc952e4cb903`. No shared workflow, physical defaults,
original inputs, dependency environments or historical core were changed.

## Model and contract

For sample and background live times `ts,tb`, the integrated means are
`mu_s = A P_s + C_s c + k B_s b` and `mu_b = (tb/ts) B_b b`.
`A` is full Gaussian response area in **sample live-time counts**; `k` is a
dimensionless multiplier, so the background observation-to-sample scale is
`k*ts/tb`. Peak positions/widths and continuum/background choices are fixed
declared inputs. Each likelihood uses original counts and response integrals on
its own native calibrated edges; no fractional rebin observations are created.
The current count-conserving rebin and covariance implementation remains intact.

`CountObservation` requires native integer counts, positive finite live time,
explicit acquisition identity, and energy edges. Nonfinite/negative/fractional
counts, signed or rebinned count bases, invalid grids/exposures and reused
acquisition identities fail explicitly. Unknown dates stay `None`; supplied ISO
dates cannot contradict a later/earlier conditional label. This declaration is
not a detector/background qualification or proof that supplied counts are native.

`BackgroundChoice` declares applicability, rationale, constant/linear/step/no
ambient continuum, ambient peak components, and fixed/free/independently measured
Gaussian auxiliary normalization. The vendor no-separate-ambient case requires
no separate observation or unused ambient nuisances. Ambient peaks can coincide
with the sample peak. Free normalization and a matching sample continuum/peak
can be unidentifiable; rank and projected-gradient checks prevent false success.
Optimizer exceptions, malformed results and failed profiles remain failures.

All amplitudes are nonnegative. The signal reaches exactly zero. Nuisance
parameters have recorded scaled `1e-12` numerical floors to preserve the log
domain during line searches; active boundaries are listed. Gaussian responses
and step continua are analytically integrated. Existing response and Poisson
helpers are reused, with exact zero-mean and tiny-mean treatment; the optimizer
uses the mathematically equivalent saturated-relative Poisson objective for
numerical stability. Deviance/NLL diagnostics contain count terms only; declared
Gaussian auxiliary terms enter inference and profiles.

Intervals reoptimize nuisances and use a nominal chi-square(1) LR cutoff, clipped
at zero. They are asymptotic LR sets, **not calibrated sparse-count or nuisance-
boundary coverage**, and require an adequate model. No fixed percentage error is
used. The signal-only empty observation has a finite one-sided upper LR limit;
the planted 100-count signal-only control matches analytic LR endpoints
`[81.6593793, 120.9005008]`. Response, efficiency, yields and undeclared systematic
uncertainties are excluded. Numerical convergence and model adequacy are separate.

## Bounded Co-Cd evidence

`co_cd_pilot.json` contains two isolated Co60 ROIs, eight fixed-shape fits and two
free-normalization tradeoff controls. Comparators are the current endpoint-linear
continuum component and signed Gaussian GLS component with full rebin covariance.
The sample ROI, calibration, resolution, efficiency and yield inputs are held
identical across methods. GLS fits its centroid/width; the joint primitive fixes
them from nominal line/profile inputs. These are **component comparisons**, not
the current whole-workflow heuristics or QuantumGold report reproduction.

| Energy keV | Scenario | Endpoint continuum net | Joint linear area | Joint step area |
| --- | --- | ---: | ---: | ---: |
| 1173.228 | Later North conditional | 7229.94 | 11057.63 | 11039.84 |
| 1173.228 | No separate ambient vendor | 10522.00 | 11057.63 | 11045.64 |
| 1332.492 | Later North conditional | 7086.27 | 10633.03 | 10890.58 |
| 1332.492 | No separate ambient vendor | 11147.50 | 11083.74 | 11065.35 |

Sample acquired 2025-08-28, 129600 s live; North ambient acquired 2026-03-02,
14400 s live. The later cross-detector ambient has **unqualified physical
applicability**. The vendor scenario is supported by saved study settings, which
do not establish final report settings. No vendor count/activity targets enter
the fit, choice of scenario, calibration or comparison. The two free-scale
controls report `unidentifiable`.

**Every fitted pilot model flags strong observed lack of fit.** The profile
shape is too restrictive to qualify these study estimates physically. Large
sample and ambient deviances, active nuisance boundaries and screening caveats
are recorded. Approximate goodness-of-fit tail probabilities are screening
diagnostics, not calibrated probabilities. Deviances from ambient and vendor
scenarios include different observations and must not select a background.
Point estimates and nominal intervals are exploratory conditional diagnostics.
The conversion shown is count-average rate divided by fixed efficiency/yield;
it includes no decay, summing, attenuation or efficiency uncertainty correction.
Joint physical-model counts, residual comparison counts and vendor report counts
remain separate; vendor report counts are explicitly unavailable in this pilot.

The source-bound Co-Cd bytes match read-only commit `46096eb` evidence, SHA-256
`7482016df6c9d370b68e5d5646cd64c5fa74def9fe586e58d19d89dab0c87cba`.
The receipt records sample/background/config/profile and unchanged-after-run
hashes, canonical Git bytes for the verified engine modules, implementation and
pilot identities, native edges and exact dependency versions. No originals,
copyrighted manuals or private chat links were copied into this change.

## Verification and reproduction

Existing local Python 3.12.10, NumPy 2.5.1, SciPy 1.18.0, pytest 8.4.2:

```powershell
python -m pytest -q tests/test_joint_poisson.py tests/test_signed_peak_fit_covariance.py tests/test_joint_peak_covariance.py tests/test_histogram_background.py tests/test_background_covariance.py
python examples/RAFM_irradiation/joint_poisson_pilot.py --output NEW.json
```

The final targeted receipt records **96 passed**; Black check and `git diff
--check` also pass. Tests include known-truth strong/weak/zero peaks, nonzero
coincident ambient peaks, fixed/free exposure scaling on unequal native grids,
seeded Poisson counts, invalid observations/exposures, optimizer/profile failure
injection, independent coarse-bin step integration, signal-only analytic LR
limits, and pilot ROI/hash/adequacy checks. Existing signed-fit and histogram-
rebin/covariance regression tests pass.

**Runtime qualification limit:** installed NumPy 2.5.1 is outside the declared
repository range `>=1.26,<2.0`; no new dependency or environment change was made.
Recheck the component on the supported validation runtime before integration or
release. Flake8 is not installed; no installation was attempted. Black 25.12.0
was available and used; the repository pins 24.10 for development.

## Integration handoff

Owner of shared workflow/report integration is task
`01a103df-5eda-7cd3-b276-2ad70d21decc` under #232/#220. This tranche supplies a
tested standalone primitive, component pilot and draft PR only. Integration
must keep explicit method/background choices, count bases, acquisition and
calibration identity, unknown states, profile/optimizer/identifiability failures,
and separate numerical/model-adequacy statuses. Register/expose the method only
through a reviewed integration proposal; do not change the current default.
Background selection and whole-campaign replay remain with their owners.
The next scientific work is a justified response/calibration model and adequacy
study, supported-runtime check, and sparse/nonregular interval calibration;
neither whole-workflow parity nor physical completion is claimed.
