# Source-bound South Co-Cd joint Poisson pilot (#249)

The opt-in pilot completes bounded software diagnostics. All 24 declared
fixed-response fits retain `strong_lack_of_fit`; the two free-normalization
controls remain `unidentifiable`. No model is selected or physically admitted.
Vendor agreement and neutron-flux inversion qualification remain unestablished.
The integration owner retains full replay and #248 acceptance.

## Reproduce

Use the existing repository-compatible runtime; no installation is needed:

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& C:\Users\Josh\projects\FluxForge\.venv\Scripts\python.exe examples/RAFM_irradiation/south_joint_poisson_pilot.py --output NEW_FOLDER
& C:\Users\Josh\projects\FluxForge\.venv\Scripts\python.exe -m pytest tests/test_south_joint_poisson_pilot.py tests/test_joint_poisson.py tests/test_histogram_background.py tests/test_background_covariance.py -q
```

The script refuses an existing output directory. It reuses the read-only source
verifier and South decoder from `run_portable_qg_example.py`; neither that driver,
`rafm_workflow.py`, `run_integrated_qg_methods.py`, manifests nor spectra are edited.
The primary checkout and integration worktrees are preserved. The feature branch
starts at integration commit `701288a30f418ed7dc331e6deda0044eb220e72e`.
Canonical-LF code hashes identify the actual implementation even without Git.

## Inputs and controls

- Original independent Co-Cd ASC hash:
  `7482016df6c9d370b68e5d5646cd64c5fa74def9fe586e58d19d89dab0c87cba`.
  The source verifier also matches its entire 8192-channel array to original ANS.
  ASC exposure is 129600 s live / 129691 s real. The ANS/report real-time precision
  and approximately three-hour clock disagreement remain in the full timeline.
- Native South ANS hash:
  `96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344`.
  Its original 8192 integer counts total 543427; exposure is 14400 s live /
  14409.5 s real. Its decoded unzoned start is 2025-10-03T15:49:14, later than the
  sample. Its original float32 energy polynomial is retained, with no fabricated
  ASC. Same detector does not resolve temporal applicability.
- Sample energies use the current integration's nominal `rafm_25cm` profile.
  Original ASC and ANS energy polynomials are also saved. Their disagreement is
  not silently repaired. Sample and ambient grids remain distinct.
- The original 1173.228/1332.492 keV sample ROIs are fixed before all scenarios.
  Ambient includes complete native bins touching those same energy supports.
  Integrated responses use each observation's native edges; integer Poisson
  observations are never rebinned or subtracted.
- IEC comparison uses the current fixed Covell singlet component of the
  IEC-inspired policy on exactly those ROIs, with declared adjacent sidebands.
  Signed South subtraction uses the current bin-overlap engine and full
  `W C W.T` covariance, including ROI/sideband cross terms. Overlap count
  conservation is checked explicitly; outside-grid counts are separately saved.
  This fixed component is **not** the full tiered workflow's moving-window result.
- Ambient-off means no separate ambient likelihood, retaining local sample
  continuum. It is a declared sensitivity, not proof of final vendor settings.
  Do not compare total deviance between South and ambient-off to select a
  background: the observation sets differ.
- Sample continuum alternatives are existing linear and integrated step bases.
  Residual evidence justified a finite additional sample-response challenge:
  centroid +/- half a nominal sample bin; width x0.8 or x1.2, separately.
  The ambient response remains nominal. These are declared diagnostic values,
  not fitted calibration, calibrated systematic bounds or a response search.

## Results and model inadequacy

The nominal linear fits give the following **conditional count-average** activity
conversions, using the same nominal efficiency/yield. These are exploratory
conversions of inadequate models, not qualified activities:

| Line (keV) | South Poisson (Bq) | South fixed IEC (Bq) | Ambient-off Poisson (Bq) | Ambient-off fixed IEC (Bq) |
| --- | ---: | ---: | ---: | ---: |
| 1173.228 | 160.150 | 161.015 | 240.371 | 244.422 |
| 1332.492 | 156.814 | 171.425 | 252.896 | 249.149 |

The full integration IEC Co-Cd result quoted in the handoff (134.325 Bq) uses
tiered selection/combination and is not this fixed singlet comparison. No QG
activity target enters fitting, response choices or acceptance. Values resembling
vendor activities do not establish agreement, especially with different activity
references and an inadequate fit. No line combination is promoted here.

Nominal South linear deviance is 136.935 / 162.668, split sample 113.850 /
149.197 and ambient 23.085 / 13.472. The same sample mismatch persists with
ambient-off. Signed deviance residuals have positive excursions below the
nominal centroid and deficits above it, reaching about -5.68 / -7.67. This
points to fixed-response/calibration inadequacy rather than only an exposure
normalization error; it does not identify a unique physical cause.

The step continuum reduces nominal South total deviance to 106.330 / 147.759,
but remains inadequate. A half-bin lower sample centroid reduces South totals
to 75.513 / 79.388 and still retains strong lack of fit. It does not improve
ambient-off totals relative to nominal linear. Opposite shifts and width
challenges also fail adequacy. Therefore a continuum or single fixed response
adjustment does not establish a valid background-specific activity estimate.
Every candidate, interval, nuisance boundary, residual, rank ratio and failure
message remains in JSON. No preferred response or background is selected.

Independent deterministic synthetic challenges use CDF-integrated truth on
unequal grids with coincident sample/background lines. Rounded Asimov zero,
weak and strong signals recover planted areas within 6 counts and nominal
intervals include truth. Deliberately shifted/broad responses retain strong lack
of fit despite convergence. Free normalization is unidentifiable and a one-step
optimizer budget fails explicitly. These are diagnostic witnesses, not sparse
Poisson coverage calibration.

## Evidence and unresolved qualification

`results/` contains implementation/source/runtime hashes, all candidate JSON,
the comparison CSV, native sample and ambient residual plots and output hashes.
`focused_tests.xml` records focused tests. `independent_sol_review.json` contains
the independent review and its evidence. Development baseline receipts may be
retained locally; the published result set contains all required failed models.

Unknown calibration/efficiency covariance is explicit and excluded from
conditional asymptotic intervals, never treated as known zero. Geometry,
temporal background applicability, fixed shape, sparse-bin/nuisance-boundary
coverage and report activity-reference semantics remain unresolved. Rates and
physical inversion are not produced. North remains a cross-detector sensitivity
in the existing #239 pilot. Software completion does not establish vendor
agreement or physical qualification.
