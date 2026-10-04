# Current FluxForge integration and original RAFM acceptance

The current Qt application, six additional analysis methods, example execution,
and selected regression checks pass their bounded software checks. Original
RAFM accuracy acceptance remains **incomplete**: the fresh independent replay
ran all 30 original ASC acquisitions, with **3 passing comparisons, 26 failing
comparisons, and 1 unavailable comparison**. These results do not qualify an
absolute activity, reaction rate, or neutron spectrum.

This review uses analysis revision `d017a7162d51d97ef3e50fac98a0c748337b9ccb`.
The subsequently added South pilot and evidence do not change package analysis
sources. Public receipts and screenshots are in
[`expanded_acceptance_20261004`](expanded_acceptance_20261004/EVIDENCE_MANIFEST.json).
Original inputs, full local outputs, failed attempts, and earlier reviews remain
preserved. The receipt manifest binds the compact published evidence by bytes
and SHA-256; it is an evidence-integrity check, not scientific qualification.

## Updates incorporated

Remote branches were refreshed, including the Qt report/instrument/consolidation
updates and `quantumgold-methods-integration` through `701288a`. The scientific
component commits were selectively integrated to preserve the existing physical
count diagnostics, background covariance, provenance validation, ambient-off
controls, and explicit comparator tolerances. The old Tk application is archived;
the supported launcher uses `FluxForgeMainWindow` in `fluxforge.gui`.

The additional methods cover saved QG protocol reconstruction, efficiency-source
fidelity, declared line-activity combinations including covariance GLS, native
joint Poisson fits, constrained overlap diagnostics, and repeated-count decay
consistency. They remain opt-in APIs/examples with their conditional,
unavailable, rejected, and unidentifiable outcomes intact. They do not silently
replace the production workflow or create scientific admission.

Integration fixes resolve monitor timing aliases with ambiguity rejection,
carry explicit activity-reference dates through aggregation, and export missing
EOI activities/rates as null rather than measured zeros. A valid measured zero
remains distinct. Empty comparison sets now report unavailable metrics.

The example runner explicitly sets its subprocess source path to this checkout.
An initial run revealed that the multiplet example otherwise imported a
different locally installed FluxForge. The regression checks the actual imported
module path even with a foreign inherited source path. All six examples then
completed, with source-input and engine-identity checks before and after.
Historical source references are retained as comparison metadata; actual current
source hashes identify the execution after selective integration.

The newer `codex/south-co-cd-joint-poisson` pilot through `54cc517` is also
included. Its original strict integration-ancestor prerequisite was replaced
with current content identity plus an explicitly historical base field. No
counts, response choices, or numerical results were tuned to vendor targets.

## Verification scope

| Check | Recorded result | Limit |
| --- | --- | --- |
| Six methods and integration contracts | 299 passed | Selected modules, not the entire repository suite |
| Existing workflow regressions | 121 passed; 12 subtests passed | Existing raw-count/background/parity/independence paths |
| Subprocess source-path regression group | 16 passed | Overlaps the integration tests; not additional unique coverage |
| South Poisson and covariance contracts | 91 passed | Overlaps existing method/background modules |
| GitHub integration CI | All 5 jobs passed | Core, external reference parity, optional ML, Qt Windows/Linux |
| Current native Qt probe | 30 spectra loaded; report exported and hash-verified | Import/render/export, with analysis accuracy assessed separately |
| Original independent replay | 30/30 executed; 30/30 report-withholding identities | Accuracy: 3 pass / 26 fail / 1 unavailable |
| Native-only acquisitions | Cu conditional analysis; Fe excluded by geometry | No invented ASC export or qualified rate |
| INL comparison example | Co, Ti, Sc, Ni completed | Conditional QG-derived calibration/count comparison |
| Phase 6 planning example | Completed with qualification receipt | Historical bundled planning demonstration |

The five-job CI run for the integration revision is
[37183685135](https://github.com/FusionSandwich/FluxForge/actions/runs/37183685135).
The newer South pilot is separately covered by its local 91-test receipt and is
included in the CI method-contract step for subsequent pushes. Test groups
overlap and must not be summed as a unique whole-suite count.

The native GUI probe uses the Windows Qt platform, loads the source-bound
30-spectrum session, and verifies the guided analysis toolbar and empty measured
unfolding startup. Running measured unfolding stays disabled until inputs exist.
The report ZIP contains one plot, nine tables, a snapshot, and a verified manifest.
Workspace and viewport remain unchanged. All 120 unavailable instrument fields
remain in the snapshot; the compact status display does not invent settings or
allow an incomplete instructional export. A separate interactive new-GUI session
remains open; its in-memory revision is `3ea8077`, which has the same GUI source.

## Source inventory and sample outcomes

The source manifest SHA-256 is
`68105c616c3e154861251816e195cb959bd4a8e88625605c7718956400823fda`.
It binds 32 original ANS acquisitions, 30 original ASC exports, 31 original QG
reports, 288 extracted ROI rows, and 88 isotope summaries. Acquisition counts
are not counts of independent irradiated specimens. Earlier checks on the
curated 32-ASC example set remain valid for that set but must not be relabeled
as this original inventory.

The complete independent replay takes 610.218 seconds with four isolated workers.
Each sample has a terminal receipt, unchanged source/input hashes, and an
identical prediction with the QG comparison report withheld. The passing
comparison samples are bare Co, Cu, and Cd-covered Sc. `RAFM3-A_2hrEOI` lacks an
original QG report and is unavailable, rather than a fabricated agreement.

Cu-Cd has original ANS counts and a 25 cm report geometry but no original ASC;
the bounded native decoder analyzes those original integer counts conditionally.
Near-contact Fe-Cd is excluded from the 25 cm efficiency model. Neither native
case becomes a fabricated ASC acquisition. The per-case receipts explicitly
identify their unmatched reports; the published reconciliation corrects the
coordinator aggregate's omitted list without changing any fit or original receipt.

## Why scientific acceptance remains open

The latest South pilot retains all 24 converged fixed-response fits with
`strong_lack_of_fit`; two free-normalization controls remain `unidentifiable`.
Convergence and plausible activity values therefore do not establish adequacy.
South and ambient-off sensitivities retain the same original integer sample
counts and separate native background grids. The covariance-based subtraction
control is distinct from the joint Poisson likelihood. Changing background or
one fixed response assumption does not qualify a preferred model.

Independent absolute-efficiency uncertainty/covariance, calibration applicability,
foil weighing uncertainties, and measured irradiation power/time uncertainties
remain unqualified. The recovered South efficiency export exists, but its `Error`
field has no established uncertainty definition; it cannot be silently treated
as a calibration standard uncertainty. Source/report geometry and clock conflicts
also remain observable. The recovered South background postdates the samples;
same detector identity does not establish temporal applicability.

Line-specific QG intensity conventions, vendor-library values, attenuation/model
details, and report activity references still require reconciliation. Current
physical peak counts remain separate from comparison counts and report-reproduced
activities. A QG report is a processed comparison, not independent truth. Missing
shared systematics remain missing in activity combinations and likelihood
intervals; they are not replaced by zero covariance. No rate or neutron-flux
inversion is promoted using these conditional results.

Luna independently inspected the six components, their integration, and the
South pilot and found no concrete software blocker. The three review receipts
identify their scope and state that Luna did not rerun the tests. They preserve
method assumptions and two non-blocking maintenance cautions concerning exact
prose-based unavailable-rate classification and older historical metadata names.
