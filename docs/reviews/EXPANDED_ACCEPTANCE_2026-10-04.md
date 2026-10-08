# Current FluxForge integration and original RAFM acceptance

The current Qt application, six additional analysis methods, example execution,
and selected regression checks pass their bounded software checks. Original
RAFM accuracy acceptance remains **incomplete**: the fresh independent replay
ran all 30 original ASC acquisitions, with **3 passing comparisons, 26 failing
comparisons, and 1 unavailable comparison**. These results do not qualify an
absolute activity, reaction rate, or neutron spectrum.

This review uses analysis revision `d017a7162d51d97ef3e50fac98a0c748337b9ccb`.
The subsequent South pilot does not change package analysis sources. Later Qt
fixes through `2f915f3f1e462273d17350d49113fbd2a985ce80` change seven GUI files; numerical
analysis sources used by the replay remain unchanged. Public receipts and screenshots are in
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

A final remote check identified `codex/qt-gui-review-20261004`. Its four GUI fixes
through `c5eee22` are integrated: calibration stays bound to its original
document/acquisition when selection changes; stale document/acquisition replacement
rejects apply; calibration plots scroll while action buttons remain accessible;
and mode/theme/canvas controls fit narrower laptop windows. The new GUI regression
group passes five tests, and 37 existing calibration/canvas/consolidation tests
pass. Luna reviewed all four changes and found no concrete blocker. Existing
conditioning warnings on synthetic calibration and unfolding tests remain visible.

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
allow an incomplete instructional export. The later native probe repeats this
30-spectrum/report check at `5a1ffca`; its receipt and screenshots use the
`latest_` prefix. That GUI was opened during the October 4 review. The October 8 verification below uses the newer code; earlier session artifacts are preserved.

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

Luna independently inspected the six components, their integration, the
South pilot, and the later GUI fixes and found no concrete software blocker. The review receipts
identify their scope and state that Luna did not rerun the tests. They preserve
method assumptions and two non-blocking maintenance cautions concerning exact
prose-based unavailable-rate classification and older historical metadata names.

## Final GUI integration, October 8

The latest local GUI update `1070ff3` was incorporated as `e3801d1`, adding
Save/Discard/Cancel protection for dirty sessions at close, session replacement,
example loading and reset. Report rendering/export uses a worker with a detached
snapshot captured on the GUI thread, and the dialog and main window wait for
active export workers before closing. Reused acquisition IDs cannot carry manual
instrument overrides into a different acquisition. The inspector context scrolls.

The earlier Linux CI width failure at `966dcba` remains recorded rather than
relabeled as a pass. The width fix `89efc2f` separates six canvas buttons into two
rows and keeps full status text in tooltips. Final fix `91ea0d2` also scrolls the
Forecasts tab, keeps metadata on one shrinking line, and preserves a 120-pixel
plot floor. This resolves the hidden forecast panel forcing an 863-pixel window.
The imported session test now creates a valid ROI with both background sidebands.
Luna identified text encoding damage during edits; it was corrected before the
final source commit, and the original Unicode strings are preserved.

All six native Windows compact/calibration tests pass at the final GUI code
revision, including 1280 by 720 operation, reachable canvas and forecast controls,
selection-safe calibration, stale-acquisition rejection and undo. The final native
30-spectrum GUI/report probe again verifies one plot, nine tables, three hashed
bundle files, unchanged workspace and viewport, and all 120 missing instrument
fields with scientific admission disabled. The final captures and receipts use
the `final_` prefix. All 37 checks across the seven affected GUI test files pass in separate processes
at `2f915f3`, avoiding accumulated Qt stylesheet work across closed windows.
The prior 37-test width scope overlaps and must not be summed as a unique suite.
A combined development probe found an empty manual-override record after replacement;
`2f915f3` removes completely blank overrides and tests calibration preservation,
same-ID acquisition replacement and reload. A generic test-window cleanup experiment
caused a Qt abort and was fully reverted; failed and stopped probes remain preserved.

Luna source-reviewed the width fix, worker/session integration and compact fix.
The encoding finding is resolved; no remaining software blocker was found in
those scopes. Luna did not independently rerun the tests. Numerical package
sources remain byte-identical to the 30-acquisition replay at `d017a71`; seven
subsequent package changes are confined to GUI files. The source bridge binds
those GUI bytes and retains the original replay revision and 3/26/1 outcomes.
