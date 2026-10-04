Issue #237: source-bound efficiency audit and alternative qualification

Run with existing FluxForge dependencies from any working directory:
  python examples/efficiency_audit/run_audit.py --output NEW_DIRECTORY --selected-method south_source_export_percent

The output directory must be new. AUDIT.json records input/module hashes, engine
commit, runtime versions, coefficient and thickness checks, curve identities,
literal source reference controls, historical yields, and explicit limitations.
energy_deltas.csv exports pointwise differences and comparison admissibility over
the original exported range 40..1999 keV. No spectrum fitting or activity changes
are performed. Invalid original efficiencies at 40..56 keV remain unchanged.
Positive adjacent brackets only; no bridging or extrapolation is permitted.

The module reuses QGEfficiencyTable, EfficiencyCurve, existing report parsing and
attenuation helpers. It does not replace existing valid methods or set defaults.
Method selection is mandatory and is saved in both outputs. Supported example
choices are source export percent/fraction, current-engine header model,
density-aware PGT-shaped XCOM alternative, unsupported vendor McMaster model,
and historical report-effective response. The module also adapts existing
EfficiencyCurve alternatives, without fitting this study's target activities.

Source identity and units
The calibration CSV/provenance already present in examples/RAFM_irradiation/
calibration is reused with its pinned SHA256. Percent is a conditional export
unit assumption supported by manual v4.04.00 A.5; the deployed version and
actual source export unit declaration remain unknown. A plausible fraction
interpretation can still be below 1, so range checks alone cannot establish units.
Independent arithmetic controls catch the factor 100 instead of fitting it away.

Only five required original study fixtures were copied byte-for-byte from
commit 46096eb. Their original paths/hashes and the source manual link are in
fixtures/manifest.json. No historical production core, manual PDF, private mail
or private chat links are copied. Text and binary originals have -text attributes.

PGT formula and supported alternatives
The report prints a detector model times C1+C2*Log(E)+C3*Log(E)^2+C4*Log(E)^3.
It is a multiplicative residual, not exp(polynomial). The natural-log convention
is an explicit alternative here; vendor implementation details are unverified.
The proprietary McMaster attenuation routines and extended line/edge tables are
unavailable. The raw CSV MuWin/MuGe columns do not establish a complete vendor
attenuation provider. Vendor-model evaluation therefore returns unsupported.

The density-aware PGT-shaped alternative uses the existing local XCOM-labeled
Al/Ge tables, mass coefficients cm^2/g, embedded densities g/cm^3, and converted
thicknesses cm. It uses the existing log-log linear interpolation within table
support, and blocks extrapolation. This implementation's algebra/unit controls
do not independently verify the embedded attenuation data against NIST. The
current engine's historical cm times mu/rho convention is preserved as a named
comparison with its existing behavior; no scientific default is changed.

The saved header/export/report C1-C4 and A agree at their decimal precision.
Window 1000 um -> 0.1 cm; dead layer 700 um -> 0.07 cm. The observed study dead
layer at byte824 matches report um, while Appendix C's printed byte816/cm does
not. Export DI=1.39 matches a calibration slot, not report physical thickness
6.450 cm. The exact vendor meaning of that slot remains unsupported. Header
Error about 4.58934e-5 versus export 0.00458934 is preserved and never interpreted
as a standard uncertainty. The byte reader requires a separately pinned study
hash, revision/channel/ROI-length closure, and makes no universal ANS claim.
Saved settings do not establish final report processing.

The bounded read-only 32-file pass found four actual saved-header/report
conflicts: RAFM-A/B/C/N-300s save 20 cm while their reports print 25 cm, with
coefficient differences too. Twenty-seven geometry anchors corroborate at
printed precision; one count lacks a report. Only 26 coefficient sets match the
25 cm source export in all three views. The A-300s pair is included as a pinned
counterexample: it must return contradiction, not be overwritten or retuned.

Geometry and provenance
The original curve is South HPGe, nominal small vial 0.5mL at 25 cm. Treat the
Co-Cd match to that vial geometry as an explicit comparison assumption. Unknown
or mismatched geometry is excluded; near-contact Fe-Cd cannot use a 25 cm curve.
There is no inverse-square geometry transfer, global multiplier or per-sample fit.

A source export, a report-effective response and a model have distinct kind,
validation-status, attenuation/formula/interpolation identity, and count basis.
Report-effective = QG net counts/(printed line Bq * live seconds * historical
yield fraction). It can embed processing, decay and attenuation conventions;
it is not an intrinsic efficiency or an independent absolute calibration.
Historical RAD INT and explicit unit assumptions are retained separately from
modern evaluated inputs. Modern inputs are unavailable in this example; the
existing bundled decay_2012 diagnostic is separately labeled and never promoted
to a modern reference or silently used to replace a historical yield.

Validation and integration limits
Known-value controls test percent/fraction, um/mm/cm, density factors, log base,
coefficient order/precision, original hashes, invalid rows and geometry. Literal
source knots are withheld reference controls for reproduction, not independent
absolute calibration validation. QG target/effective origins are barred from
independent validation even if held out of a fit. Source SHA aliases are compared
to the curve's own identities as well as fitted-source IDs. Independent origins
require a separately identified, SHA-bound certificate; that declaration does
not verify certificate authenticity or complete physical qualification. The
checks remain pointwise controls, not curve-wide qualification.

Targeted command:
  python -m pytest -q tests/test_efficiency_fidelity.py

Certificate, covariance, acquisition date, active August applicability, vendor
attenuation identity and complete gamma-library qualification remain unknown.
No result is physically/scientifically admitted. #232/#220 owns shared workflow
and reporting integration. Integration should consume the explicit method,
provenance and admissibility exports without changing defaults or mixing
historical comparison counts with physical reduction inputs. This component
does not establish whole-workflow parity, physical efficiency or neutron flux.
