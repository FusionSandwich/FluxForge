# RAFM archive intake and example review — 2026-10-03

## Result

Ten downloaded ZIPs and the nested converted-spectra ZIP were extracted locally.
Original archives were retained. Extraction rejected escaping paths, symbolic
links and duplicate paths; existing staged files could only be reused with
identical hashes. Embedded Git directories were excluded and archive source
programs were not executed. No software was downloaded or installed.

The maintained example now contains 32 raw acquisitions, 87 converted spectra,
four element-composition tables, one historical QG-derived activation table and
eleven historical simulated activation tables. The [feature guide](../../examples/RAFM_irradiation/FEATURE_GUIDE.md)
provides executable commands and a feature-by-feature evidence/limitations map.

PR #206 remains a draft against the unchanged `unfolding-physics-fixes` base
`b25ac3f7a7201d120a516ddfbda46eee67453e7e`. Its previous remote head was
`50e9635426cc7336bb923ab69500255aa7d36278`. Both were checked again before this
upload. The repository's default `main` remains the older locally available
`52759a9b74fe6c5c386c74ce1105ff951cbc05cc`. The user's separate HPGe checkout was
preserved.

## Reconciliation

- Sixteen downloaded RAFM3/4 ASC spectra have alternative calibration headers.
  Their channel counts, live/real times and acquisition timestamps exactly match
  the existing spectra. Both coefficient sets and source hashes are recorded;
  committed headers were preserved.
- Four Experimental RAFM3 15d reports are byte-identical to the existing RAFM4
  reports. They were not added as new acquisitions.
- CHN/SPE conversions preserve all channel counts but lack acquisition timing
  and usable calibration. All 58 are classified `counts_only`. All 29 binary SPC
  files are explicitly unsupported. None qualifies an absolute activity.
- New RAFM2 dates are April/July 2025. Their irradiation histories, material/
  mass identity and independent QG reports are absent. EOI correction remains
  unavailable; shared-profile count-start activity remains conditional.
- The composition workbook provides element wt%, with missing entries retained
  as null. Its suspect isotope labels were excluded. Density 7.7 g/cm³ is a
  workbook assumption, distinct from the existing geometry model's 7.87.
- INL Co masses remain 4.0661/3.6703 mg of Co element despite the 0.46 wt% alloy.
  No additional dilution was applied. Geometry inferred by applying pure-Co
  density to element-adjusted mass is not certified wire geometry.
- OneDrive_2 contains only download errors; the tables ZIP has seven empty CSVs.
  Neither adds usable measurements. The original reactor background and formal
  reference material remain local pending detector/specimen reconciliation.

## Code fixes exposed by running the examples

1. CSV import lowercased headers but not its `energy_keV` lookup, discarding the
   exported energy column. It also ignored supplied count uncertainty. Both now
   survive import, including signed counts. Partial measurement rows, missing
   declared channels/energies/uncertainties, and nonfinite values are rejected.
2. The Phase 6 runner joined PYTHONPATH with `:` on Windows, preventing child CLI
   imports. It now uses `os.pathsep`.
3. The raw-recovery driver no longer hardcodes 29 for `--all-raw`; it verifies
   coverage against the discovered raw inventory and records that count.
4. The new feature runner verifies complete raw/converted inventories and binds
   background, workflow metadata and detector-profile inputs. It records
   counts-only/unsupported states and explicitly leaves accuracy unqualified.
5. The planning example writes source/output hashes and labels its historical
   bundled activity/unfolding inputs as a planning demonstration. It still
   infers half-lives from activity ratios; those are not independent decay data.

## Execution evidence

| Run | Observed result | Evidence |
| --- | --- | --- |
| Feature runner | 32 raw acquisitions; signed background/covariance JSON round trips; 58 count-identical CHN/SPE; 29 unsupported SPC; distinct-acquisition sum; CSV counts/energies/uncertainties checked | `RAFM_ARCHIVE_FEATURE_REPLAY_2026-10-03.json` |
| RAFM2 fresh raw extraction | 144h:31 peaks/9 isotopes; 70d:28/8; 72h:60/6; all three comparison results unknown and all EOI activities withheld | `RAFM2_RAW_REPLAY_2026-10-03.json` and committed sample artifacts/reports |
| Phase 6 RAFM4-C | 50 outputs; twelve inventory rows, six masking candidates, five optimization objectives, four second-irradiation candidates and `.ffexp` | `RAFM_PLANNING_REPLAY_2026-10-03.json` |

The conditional Co1173 window has 37,044.43 signed net counts with counting
uncertainty 215.54, and profile-dependent Currie sensitivity 14.96 Bq. These are
example calculations, not an independently qualified Co activity or detection
limit. The window sum does not subtract the local continuum or replace peak fits.

The simplified [raw/background inspection figure](RAFM_ARCHIVE_SPECTRA_2026-10-03.png)
was visually checked. The inherited full-range peak-annotation plots are
diagnostic and have crowded labels; use their tables and ROI views for detailed
assignment review.

The raw-recovery source hashes, six loaded workflow metadata files, background,
profile and copied artifacts were checked again. The informational archive
manifest changed after the three-spectrum replay to use LF-normalized hashes for
runtime JSON/profile portability. Its exact captured snapshot is retained in
`RAFM_ARCHIVE_MANIFEST_AT_RAW_REPLAY_2026-10-03.json`; its SHA-256 matches the
original receipt. The portable raw receipt records this difference. No scientific
input changed and no resumed checkpoint was reused. The feature replay uses the
current manifest. Measured/archive file hashes remain exact bytes.

## Acceptance review

A separate adversarial Sol agent inspected code, failure cases and decisive
evidence. It verified all sixteen calibration-header variants and the actual
32/87 inventory. Its findings led to runtime-data hashing, exact converted-file
inventory/partner checks, mandatory 29-partner guard, and strict CSV row handling.
Regression tests include redistributed bins with the same total, shifted
channels, changed/missing hashes, escaping/duplicate paths, unlisted files,
deleted file plus manifest row, deleted count guard, altered efficiency, and
missing/nonfinite CSV measurements.

The final targeted suite covers archive integrity, CSV import, Windows child
environment, raw dispatcher/checkpoints, existing RAFM workflows, independent
validation, background covariance and spectrum I/O parity. Test completion and
the final adversarial result are recorded in the acceptance receipt.

Local Python 3.12.10 and existing scientific libraries were used. The installed
NumPy is 2.5.1, outside the declared `<2.0` dependency constraint; no supported-
dependency-environment claim is made. Black was checked with the existing local
installation. Flake8 was unavailable and was not installed for this task.

## Remaining limits

This intake expands example coverage. It does not close issues #24–26 or certify
the earlier raw activities, absolute calibration/efficiency uncertainty, common
systematics, Cd geometry, response/cross-section normalization or independent
EOI reference. The six previously unmatched QG records still lack raw partners.
GUI interaction, standards-specific qualification, certified-source calibration,
live acquisition, scintillator response, ANN accuracy and plutonium isotopics
were not demonstrated by these RAFM archives.

Local extraction and execution root:
`C:/Users/joshu/Documents/UWNR_work/archive_intake_2026-10-03`.
