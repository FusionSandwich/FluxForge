Issue #236: source-bound historical protocol component

From the repository root with an existing Python 3.11/3.12 environment:
  python examples/qg_protocol/run_example.py --output saved-comparison-new.json
  python examples/qg_protocol/run_example.py --ambient on --output ambient-on-new.json
  python examples/qg_protocol/run_example.py --continuum off --output continuum-off-new.json
  python examples/qg_protocol/run_example.py --aggregation activity_over_sigma --output aggregation-choice-new.json
  python -m pytest -q tests/test_qg_protocol.py

Outputs must be new files. No network, dependencies, environments or workflows
are modified. --originals may point to a separate source-originals root with
ANS/ and QG_report/ directories; the exact 32-file ANS set must match the manifest.
The single Co-Cd ASC fixture and copied current-engine background are independently
hash-bound. Fixture .gitattributes prevents newline conversion of originals.

This is configuration and line-count evidence, not a vendor activity replay.
The example exports the declared historical protocol, all 32 saved headers,
two Co-60 lines, physical_count_basis and comparison_count_basis. It uses the
current engine's local sideband estimator and measured-background subtraction,
including its channel covariance. The continuum estimator is explicitly an
assumed current-engine approximation; continuum-on in a saved header does not
identify the final report's algorithm. Ambient precedes continuum when selected.
Ambient-off keeps local continuum-on; the controls are independent. Toggling
either comparison control leaves the conditional physical counts unchanged.

The physical output is the existing North measured-background scenario. Its
applicability to the experiment is not established. No closeness to a QuantumGold
activity can qualify it. Efficiency, gamma intensities, corrections and isotope
aggregation are not evaluated. --aggregation persists metadata only. No activity
is generated and no exact vendor parity or whole-workflow completion is claimed.

Protocol API
  parse_saved_state(bytes, expected_sha256=..., layout_id=LAYOUT_ID,
                    report=..., expected_report_sha256=...)
  HistoricalProtocol.from_saved_state(state, scenario_name=...)
  protocol.with_assumptions(scenario_name=..., rationale=..., **choices)
  protocol.to_json(); HistoricalProtocol.from_dict(json.loads(...))
  restored.verify_sources(ans_bytes, report=report_bytes)

Each field exports value/status/source. Statuses are saved_state,
report_confirmed, assumed, unknown, or unsupported. Explicit choices include
ambient and continuum controls, ROI FWHM width, sideband channel width, FWHM gap,
continuum method, ambient identity/hash/normalization, gamma library name/hash/
revision, gamma intensities/corrections, efficiency identity/hash/units, library
efficiency use, aggregation and activity-reference basis/timestamp/timezone.
Installed vendor software version is unknown. Unknown fields serialize as null;
they must not be converted to zero, unity or a current library default. An
unsupported method persists as such and cannot execute through roi_parameters().
Consumers must check field status and method support before calculation.

Serialized JSON alone is a configuration claim, not a fresh source audit.
verify_sources() must rebind it to the original bytes before reproduction; it
checks both hashes and the serialized header snapshot. Header/report assertions
cannot be relabeled or changed as if confirmed; alternatives use assumed status
and preserve the original saved_header. Configuration has no physical defaults.

Evidence and bounds
The fixtures are additive, byte-identical copies of only the 32 ANS files,
31 paired reports and one focused ASC from:
  commit 46096eb, examples/RAFM_irradiation/quantumgold_reference/originals
The source audit and its SHA-256 are recorded in fixtures/manifest.json:
  artifacts/validation/quantumgold_documentation_20261003/saved_settings_audit.json
North-background.ASC is a byte-preserving copy of the current engine's
src/fluxforge/data/backgrounds/background.ASC, with source attribution and hash
in the manifest. Copying this one input into -text fixtures avoids platform
newline conversion changing its source identity; its physical use is conditional.
Primary documentation/findings:
  https://github.com/FusionSandwich/FluxForge/blob/46096eb/artifacts/validation/quantumgold_documentation_20261003/FINDINGS.txt
  https://ludlums.com/images/product_manuals/QTMmanual.pdf
The manufacturer document is referenced, never copied here.

The observed layout has revision 4, a 1548-byte header, channels 0..8191 and
50-byte ROI records. Structural/timing/library/report anchors and exact source
hashes are mandatory; the manual's literal offsets are not a universal decoder.
All study headers have AnalysisCtrl=1: saved ambient OFF, continuum ON, ROI
width 4 FWHM, background width 1 channel and gap 0 FWHM. The 31 paired reports
independently establish library efficiencies ignored and measurement-date
activity reference; one unpaired header has no report confirmation. Some saved
files have zero ROIs despite report peaks. Saved settings cannot establish the
final report correction controls or the installed vendor version. Missing
original GammaLib contents, intensities and summation corrections stay unknown.

Current engine base is a7bcc680d1f5e06b1d9dae405241fc380087ca2b, a verified
descendant of 4615e61bbb262d974e0326a44107bc952e4cb903. The example verifies
canonical-LF SHA-256 identities of its reused engine files, independently of
working-directory paths or a claimed revision. Component/example raw source
digests and actual Python/NumPy/SciPy versions are exported with each run.
Any engine changes require separately reviewed digest updates; the example
fails instead of silently importing the historical portable engine.

Integration handoff: #232/#220 owns shared workflow/report integration.
This PR adds no workflow, core, GUI, engine-integration repair or physical
defaults changes. The integration owner should consume the serializable
HistoricalProtocol and explicit physical/comparison count bases, reverify
sources, and keep all unknown/method choices visible. Activity combination,
efficiency fidelity and other scientific-method work remain independently owned.
