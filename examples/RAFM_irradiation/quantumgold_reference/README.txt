Portable QuantumGold RAFM reference example

Tracked by https://github.com/FusionSandwich/FluxForge/issues/220.

This is the canonical recovered 2025 RAFM3/RAFM4 and INL monitor campaign.
It contains 32 native ANS spectra, 30 available original ASC exports, 31 original
QuantumGold reports, all 288 extracted ROI rows and 88 isotope summaries. Original
bytes are retained. manifest.json names every required data resource relative to
the FluxForge repository root, with SHA256 and size. identity_labels.json binds
measurement IDs, specimen/cohort labels, original filenames and workflow roles.

Extracted information
The saved count matrix includes QG spectrum/file labels, detector material/shape/
diameter/distance, printed efficiency equation/coefficients, detector model fields,
energy calibration, live/real durations, dead-time/pile-up settings, gamma-library
name and activity-reference text. Original reports retain all other fields.
ROI rows retain source line, nuclide assignment and energy, radiation intensity,
gross/net counts and printed errors, and line activity. Isotope summaries retain
reported half-life, activity, error and units. Source lines are verified verbatim
using the historical Windows cp1252 report encoding. No peak or summary is silently
discarded from the source extraction, including nonpositive/ambiguous results.
Native arrays use the empirically corroborated layout for this exact pinned set:
8192 little-endian uint32 channels at byte 1548; energy polynomial float32 at 424.
All 30 available ASC arrays are checked for channel-by-channel equality to ANS.
This is a dataset-specific adapter, not a general undocumented ANS file reader.

Run on another computer
Clone/check out this branch, including examples and src. Python 3.11 or 3.12 is
supported by the project. The input audit needs only Python's standard library:

  python examples/RAFM_irradiation/run_portable_qg_example.py --verify-only

For the full software example, use FluxForge's runtime dependencies declared in
pyproject.toml (normal project installation: python -m pip install -e .).
No QuantumGold installation, email account, OneDrive, Downloads folder, local
drive-letter data, native ANH/GammaLib.mdb or reactor transport executable is used.

  python examples/RAFM_irradiation/run_portable_qg_example.py --output replay_output

The output directory must be new. The source tree and existing results are not
overwritten. The script determines source locations from its own repository path;
the command can also be called by absolute path from an unrelated working folder.
On Windows, Linux and macOS the path handling uses pathlib and POSIX manifest
paths. Original-byte attributes prevent line-ending conversion across platforms.

Workflow stages and outputs
1. Verify all source hashes, native/ASC arrays, identity labels, timings and 376
   extracted report lines. INPUT_AUDIT.json records observed completeness.
2. Export native_channel_arrays for all 32 physical counts.
3. Rebuild working_baseline from bundled inputs: recovered South 25 cm curve,
   QG-conditioned response, accepted same-count activity-per-net-count factors,
   explicit Sc48 corrected-yield scenarios and irradiation timing alternatives.
4. Stage only this campaign's authoritative ASC/report files and frozen maintained
   workflow metadata/background/prior. Older RAFM1 files do not enter this replay.
5. Run the maintained FluxForge raw analysis/comparison/activity/reaction-rate/
   diagnostic-unfolding workflow for the 30 available ASC spectra. Run the QG
   flux-wire reference workflow for all 12 monitor reports, including native-only
   Cu-Cd and Fe-Cd. REPLAY_RECEIPT.json records both workflow summaries, source
   manifest identity and missing-file policy.
6. Reconcile exported sample_group labels against the source roster. The legacy
   benchmark infers RAFM1 for some RAFM-1 monitor names; these become flux_wires,
   with legacy_sample_group retained and OUTPUT_LABEL_RECONCILIATION.json saved.
   No measured values, rates, timing assumptions or numerical results change.

Exact limits and labels
RAFM-A-2hr has native+ASC data but no recovered QG report. It is analyzed as raw
data and retained without fabricated QG peak/activity targets. Cu-Cd-RAFM-1 and
Fe-Cd-RAFM-1 have native+QG data but no original ASC export: arrays and QG results
are preserved, but the ASC-based raw workflow does not invent their exports or
perform two additional raw activity reductions. Fe-Cd is near-contact; never use
the South 25 cm curve to assign it an absolute activity. Same-count QG anchoring
and the separately retained report benchmark provide its working comparison.
Repeated A/B/C/N letters bind to their RAFM3 or RAFM4 cohort; they do not establish
the same physical specimen across campaigns. Ti's three counts retain their
distinct count IDs and the correspondence-based shared-wire geometry information.

The working-reference method is documented in:
  artifacts/validation/south_working_baseline_20261003/METHOD_AND_USE.txt
All dataset paths in that provenance are historical labels only. Runtime opens
package_path/repository-relative resources, never original_local_path.

Timing/uncertainty/physics scope
QG/ANS acquisition clocks and live/real durations are retained. ASC/QG clock
discrepancies receive no automatic timezone adjustment. The baseline records the
maintained Aug 4 EOI and advisor-correspondence Aug 5 RAFM4 EOI as separate working
scenarios. The maintained workflow stage replays its original Aug 4 schedule;
the package does not silently change its historical analysis behavior. Count-
reference line comparison can be done before choosing an EOI scenario.
The original workflow's diagnostic uncertainty controls, detector preset and
unfolding assumptions are retained and recorded; they are not a newly measured
calibration, uncertainty budget or physical response qualification. In particular,
the current GLS placeholder response and method differences remain diagnostics.
Software completion does not make failed count/activity comparison thresholds
pass. Inspect raw_replay/validation_summary.json and unfolding admission reviews.
Sc48 ambiguous 1.00 intensities and near-contact geometry are not pooled into the
working 25 cm response. No arbitrary uncertainty floor is added by the working-
reference builder. QG-derived response is not an independent absolute calibration.

Older example files
legacy_file_roles.json labels the pre-existing examples separately. In particular,
QG_processed_gamma_data/RAFM1/RAFM1_Long_144h_EOI.txt is actually an ASC channel
export with a misleading historical extension, not a QG activity report. Its
bytes and historical paths/results are preserved, and it is excluded here.
RAFM1_Long_70d_EOI.txt is an older QG report outside the 32-count campaign. The
older maintained example files remain available; do not mix them into RAFM3/4.
The older July 1 QG report is also byte-pinned in supplemental_inputs, with all
12 ROI rows, 6 nuclide summaries and every original text line extracted and
verified separately. Its blank ID/truncated File label and unmatched native/ASC
identity remain explicit. Its 3600 s live/3848.07 s real time is retained as
printed; no 70-day EOI or previous inferred-table duration is substituted.
The recovered South 4hr native background ANS is preserved there too. It has no
corroborating ASC and is not silently substituted for the maintained North 4hr
background used by the legacy diagnostic replay. Both detector/source roles are
explicit; this package does not assert that the North background is a South
measurement. The input audit verifies these supplementary bytes and extraction.

Opt-in South background sensitivity
The independent raw 25 cm monitor comparison can use the hash-bound native South
background instead of the historical North ASC:

  python examples/RAFM_irradiation/run_portable_qg_example.py --raw-sample Co-Cd-RAFM-1 --background-mode south_native --output south_probe

Use a fresh output directory. The receipt records the background source SHA256,
native live/real time, polynomial, and detector. The South file starts on Oct 3,
2025, later than some campaign counts; its applicability and the background used
by QG remain unknown. It is an explicit sensitivity scenario, not the default.
This portable branch aligns unequal energy grids by linear interpolation of channel
counts, which is not generally count conserving (issue #228); the separate
`4615e61` validation engine uses integrated overlap and covariance. Background source
qualification is tracked in issue #229. This mode excludes near-contact, missing
ASC and Ti/Sc48 ambiguity cases. QG reported activities remain separate reference
values and do not drive raw activity estimation in this mode.

Tests
  python -m unittest discover -s tests -p test_portable_qg_example.py -v

Tests relocate all required reference data into a path with spaces, invoke an
isolated Python interpreter from an unrelated folder, and challenge wrong paths,
changed/missing source bytes, duplicate IDs, wrong source reports and altered
cohort/routing/timing labels. The full replay is also exercised from a Git archive
so the test covers committed bytes rather than only this Windows working tree.

Recorded acceptance
PORTABILITY_ACCEPTANCE.json records the source-bound Git-archive replay, eight
focused tests, separately tested label reconciliation and supplemental input
audit. It also records observed dependency versions and the limits of the
software/example acceptance. Numerical replay requires the project runtime
dependencies; the standard-library input audit can run independently.
