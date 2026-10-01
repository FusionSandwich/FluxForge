# UWNR source data use and covariance integration — September 30, 2026

The authorized example `examples/RAFM_irradiation/calibration/South Small Vial 25cm.csv`
retains exact source bytes, SHA256 `03d01e0acbb4da14c3775b00ed239b86187e6b9c16b8d7ed4c2a6be32d3a0000`.
Its April 30, 2025 transmission date is correspondence provenance, not a calibration
acquisition date, certificate, covariance or proof of August applicability.
No private mail, contacts, chat links or operating-workbook contents are published.

The table energies are keV. A percent efficiency interpretation follows historical
Quantum 4.04 documentation and remains an explicit diagnostic assumption because
active-version applicability is unverified. The provenance JSON records these
limits. Vendor `Error=0.00458934` remains an unspecified coefficient; it is not a
standard uncertainty. Negative efficiencies remain in the CSV and in the curve
QC rows. The reader excludes nonpositive rows/adjacent interpolation brackets
without clipping or skipping across invalid data. Out-of-range queries are excluded.
The near-contact FeCd profile is separate; matching South/25cm coefficients
establishes only nominal comparison. The report/table DI numeric mismatch and
geometry/active-profile identity remain unresolved.

Run the all-data diagnostic with existing dependencies, from the repository root:

```text
PYTHONPATH=src python tools/validate_uwnr_data_use.py \
  --inventory <all_reports_inventory.json> \
  --reconciliation <measurement_reconciliation_2026-09-16> \
  --count-matrix <count_metadata_matrix_32_2026-09-27.json> \
  --operator <fresh validate_monitor_response_rafm receipt.json> \
  --out <fresh directory>
```

The source inventory covers 31 original reports, 288 ROI rows and 88 summaries.
Every ROI's source hash and original text line are checked. No source row is
silently dropped. Conditional curve/yield hypotheses and source exclusions are
exported per ROI; activities and rates remain unchanged. Raw ASC/ANS availability
is a bounded named-folder observation, not a disk-wide absence claim. Shared
background, schedules, decay data, external nuclear/method references and
unavailable inputs have explicit roles. All physical rows remain excluded while
full calibration/library/history/source uncertainty is unqualified.

Recovered original roots contain 32 hash-matched native ANS counts and 30 ASC
exports. The 32nd count, RAFM-A-2hr, has no corroborated QG report and remains an
explicit missing-report input. Six older ASC exports outside these observations
have a separate historical role. Native-prefix ASCII labels can corroborate the
detector/library label; undocumented binary fields do not identify calibration
coefficients or uncertainty. Historical Quantum manual sections 9.9.4–9.9.5
provide a recovery lead, with deployed-version applicability still unknown.

The privately retained operating workbook spans September 17, 2009–April 28,
2025 and does not cover the August 5 campaign. Cumulative operating totals and
rod-height snapshots (inches) are not an intraday power history. Recovered E8
10:28–12:28 correspondence provides chronology, not a complete power/rod log.
The maintained legacy schedule uses August 4; that conflict remains explicit and
is not silently changed in replay. No timing or operating uncertainty is invented.

## Repaired integration

`UncertaintyComponent.from_covariance` accepts named inputs, their units, a full
symmetric positive-semidefinite source covariance and signed log-rate sensitivities.
It retains the covariance, parameter ordering, sensitivities, source/group identity
and a covariance-binding digest. Shared rows use
`C_rate[i,j] = R_i R_j J_i C_source J_j^T`.
Mismatched shared covariance, ordering or units is rejected. Independent source
blocks remain independent. Joint coverage must represent a source once per row;
coverage overlap is rejected to avoid double counting an opaque activity error.
Budget serialization retains all metadata and diagnostic assumptions.
Uncertainty scope explicitly distinguishes itemized, conditional and marginalized
terms from an opaque `reported_total_unknown`. A conditional coefficient covariance
must not silently include certificate or yield uncertainty again. Shared assay,
certificate or nuclear-data primitives need one joint block with their cross
covariance; independent blocks are justified only by independent source inputs.
Replacing an opaque total requires reconstructed coverage rather than adding
unresolved terms to it. The three Ti counts share one irradiated monitor; common
calibration does not shrink with repeated counting. Singular covariance retains
its rank and receives no diagonal jitter.

This is first-order propagation following [JCGM 102:2011 §6.2.1.3, equation 3](https://doi.org/10.59161/JCGM102-2011).
Nonlinear models need appropriate propagation validation; no recovered covariance
or nonlinear mean correction is asserted for UWNR. A synthetic log-efficiency
acceptance case has rates 100/200 and absolute covariance `[[6.2,6],[6,15.2]]`;
its log-ratio variance is 0.0004 because the common intercept cancels. These are
synthetic software checks, not calibration data.

For declared `kind=irradiation_history` covariance, the workflow calculates signed
sensitivities of log rate to each segment's duration (seconds) and relative power,
with EOI activity held fixed. Parameter names/units/order must match the segments.
Half-life and EOI conversion effects require their own correctly itemized/joint
source inputs. Segment histories and operating-log bindings are carried through
RAFM3, RAFM4 and wire schedules. A missing relative power is rejected rather than
assumed one. Nonfinite histories and non-boolean count-decay declarations are
rejected. Existing real-time/count-start distinctions remain intact.

`build_flux_wire_reactions(..., mode="physical")` requires complete source/component
budgets, rejects explicitly assumed terms, and requires a byte-bound complete
operating-log index with timezone-aware start/end, exact segment coverage, monitor
identity, schedule EOI/duration agreement, and bound reactor-power/control-rod evidence.
It also rejects unknown uncertainty scope. A scalar power history requires a
declared `separable_local_spectrum` model and byte-bound spectrum-shape evidence:
local flux must be representable as a common shape times the power history, with
an explicit reference power. Reactor power alone does not qualify the E8 spectrum.
A log cannot supply missing calibration or uncertainty. Diagnostic construction
preserves legacy values and labels unresolved history/model assumptions.

A prospective log index uses schema `fluxforge-operating-history-v1`,
`coverage_complete=true`, `source_id`, `monitor_ids`, timezone-aware `start`/`end`,
`power_basis`, ordered `segments` (`duration_s`, `relative_power`), and `evidence`
entries for `reactor_power` and `control_rods` (each with path, SHA256 and units).
Physical use additionally binds `spectrum_shape_evidence` and declares
`history_model="separable_local_spectrum"`, with `reference_power` containing a
finite positive `value` and nonempty `units`.
The schedule binds the index by path/SHA256 in `irradiation_operating_log`.
These input checks are not experimental validation or scientific admission.

No unsupported blanket activity/100 correction was found in maintained source,
tools or RAFM Python replays. Percent-yield and percent-uncertainty conversions
remain. Historical whole-rate sensitivities remain explicitly diagnostic evidence.
The 13 actual replay budgets still lack six measured components; their full
shared calibration/history covariance remains unknown, and strict physical
construction rejects them. Scientific admission remains false.
