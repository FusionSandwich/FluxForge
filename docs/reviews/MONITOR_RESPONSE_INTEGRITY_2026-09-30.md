# Monitor response integrity (#208/#209)

Implementation: `2756b00f655bc90001f8e27220268a83da972f07`, based on PR207
`9d04eff6ebc867227ae439ecdcc37ddb40732b7f`. This separate feature branch
is stacked on PR207. PR205's physical GLS draft remains separate.

## Repair

Cover identity now includes resolved density/atomic mass, exact thickness and
uncertainty, material and both models. Body identity includes exact dimension
and uncertainty, number density, complete energy/value tables and source.
Versioned SHA-256 keys retain round-trip float precision and full digests.
Body tables own immutable copies of input values.

Aggregation uses keys to nominate candidates, then compares nominal operators,
response uncertainties and supplied component models exactly. Incompatible
rows stay separate. Compatible replicates retain inverse-variance weighting,
shared response uncertainty, first-observation order and input-index/observation-ID
memberships. Scaled weights avoid overflow at small uncertainty. Aggregation
remains off by default.

Constructors, physical response construction and low-level kernels reject
NaN/Inf/negative physical inputs, invalid uncertainties and energy/value arrays,
and unknown models. Dimensions/densities/masses are positive; uncertainties and
cross sections are nonnegative. Kernels preserve zero-depth/zero-cross-section
limits. Caller body energies strictly increase. Evaluated archive coordinates
may be nondecreasing: the local Cd table has 357 repeated coordinates at printed
precision; its existing interpolation is preserved without rewriting data.

## Tests and actual data

The first 60 regression cases ran before source edits: **50 failed, 10 passed**.
The final affected suite passed **216 tests**: monitor response/integrity;
IRDFF/archive/access; unfolding diagnostics and 10-bin example; normalization
defaults and irradiation history; RAFM workflow/background integration.
The audit's synthetic 44.599 retained-row versus 45.089 averaged-measurement
counterexample is in coefficient scale, not a physical reaction rate.

Existing runtime: `D:\FluxForgeQA\envs\fluxforge-py312\Scripts\python.exe`,
Python 3.12.10, NumPy 1.26.4, SciPy 1.17.1, pytest 9.1.1. Final tests and replays
use `FLUXFORGE_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUTF8=1`,
`MPLBACKEND=Agg`, and pytest `-p no:cacheprovider`. No acquisition or remote
execution occurred. The 10-bin test rewrites two historical CSVs; its generated
changes were restored. Primary-checkout work and existing drafts were preserved.

`tools/validate_monitor_response_rafm.py` checks the audit's source joins and raw
ASC/QG/saved-analysis hashes for **13 actual observations**: Co/Sc bare and Cd
pairs, and three Ti specimens with three reactions each. It checks QG activities,
rebuilds target-normalized rates with reviewed mass metadata and saved timing,
and uses the local evaluated IRDFF reaction and absorption archives without
downloads. It verifies source bytes again afterward and writes a fresh additive
output directory. Co masses are adjusted element masses: no second 0.0046 factor.

- Valid physical rows, response uncertainties and fixed-prior forward folds
  are bitwise unchanged.
- Four distinct Co/Sc operators stay separate; nine compatible Ti observations
  group to three operators: **13 to 7 rows**, before and after.
- GRAVEL/MLEM each ran for 30 iterations with aggregation off/on. Off-mode
  outputs are unchanged; aggregated prediction changes are at most
  `4.440892098500626e-16` relative. Comparisons use `rtol=1e-12`, `atol=0`.
- Four modified copies of an actual Co-Cd input (density, atomic mass,
  next-representable thickness, thickness uncertainty) incorrectly merged before;
  all stay separate after.
- Twelve NaN/Inf/negative cover edits in copies of that input reached ordinary
  MLEM construction before; all are rejected after. Frozen construction is
  deliberately bypassed to test the build boundary.

Full receipts and numerical outputs are committed in
`artifacts/validation/monitor_response_integrity_20260930`; original additive
evidence is at `D:\FluxForgeQA\receipts\response_integrity_20260930`.
Before receipt SHA-256: `a35f32ec11a3c32da8450db3159471a6df18ed06abb26bc3cc8f5117e564f1ce`.
After receipt SHA-256: `411b29b0ce47221c34b36ed5ed620ee5490a102777748fc735aa27e5f28fdf2d`.
The checkouts have different prefixes. All data digests match; the mass-review
reference Markdown differs only in LF/CRLF bytes, with identical normalized text.
`tools/compare_monitor_response_receipts.py` verifies that distinction explicitly.
Receipts include input/output hashes, code identity, units, prior and assumptions.

## Scientific limitations

Nominal Cd is a 0.0508 cm isotropic slab. As-built dimensions/closure/gaps and
body geometry are unqualified; body self-shielding is not admitted by this replay.
Ti replicate compatibility is conditional on the shared assumed operator.
Composition certificates/uncertainty, count-specific efficiency, full covariance,
reactor prior and an independent published comparator remain unresolved.
Ni-57 is excluded; Cu-Cd lacks admitted raw spectrum/timing here. Historical GLS
supplies only energy boundaries, not flux. QG activity agreement is not independent
activity validation. No measured unfolded spectrum, confidence interval or physical
validation is admitted. Issue repair sign-off does not clear publication gates.
