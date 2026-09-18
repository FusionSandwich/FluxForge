# SLOWPOKE activation-analysis capability audit

This is a capability audit and qualification plan. It does not claim that FluxForge is qualified for a particular SLOWPOKE facility. Facility flux, spectrum, irradiation position, pneumatic-transfer timing, detector calibration, sample geometry, and nuclear-data provenance must be supplied and accepted per facility.

## Facility-facing capability requirements

SLOWPOKE work must support these distinct cases:

| Capability | Why it is required | Current FluxForge status |
|---|---|---|
| Thermal, epithermal, and fast spectral components | SLOWPOKE activation is not guaranteed to be purely thermal; fast-component `(n,p)` and `(n,alpha)` products can be important interferences. | Partial: neutron corrections, response matrices, unfolding, and k0/monitor models exist. Facility spectrum and interference qualification are missing. |
| k0-NAA and comparator NAA | SLOWPOKE NAA commonly uses thermal/epithermal standardization or comparator measurements. | Partial: k0 data/workflow and comparator-oriented surfaces exist; facility-specific constants, monitor selection, and acceptance receipts are missing. |
| Multi-element qualitative and quantitative INAA | The reactor is used for chemical analysis of varied materials, with short and long irradiations and repeated counts. | Partial: HPGe I/O, peak/ROI, activity, inventory, and decay workflows exist. Absolute efficiency, blank, geometry, and line acceptance remain open. |
| HPGe gamma spectroscopy | Facility descriptions identify high-resolution Ge spectrometers and automated sample measurement. | Software coverage exists; detector identity, energy/efficiency calibration, dead-time, summing, and facility geometry are not qualified. |
| Pneumatic transfer and short-lived products | Transfer delay and count start affect short-lived nuclides and early masking. | Partial: finite counting and timing fields exist. Facility transfer-time distribution and timestamp contract are not bound. |
| Sample changer / batch processing | Automated measurement requires stable specimen identity, order, geometry, and count metadata. | Partial: batch ingest, artifact/session persistence, and inventory workflows exist. Native facility sequence and chain-of-custody receipt are missing. |
| Decay-chain and feeding corrections | Parent/daughter production and decay affect EOI and count-time activity. | Software fixtures and decay inventory exist. Qualified nuclear-data version, parent feeding, and facility-specific EOI history remain open. |
| Self-shielding and geometry corrections | Dense, large, or absorbing samples need thermal/epithermal self-shielding and attenuation treatment. | Partial: self-shielding and attenuation modules/tests exist; sample composition, dimensions, density, and validated correction model are not bound. |
| Interference and impurity treatment | Gamma overlap, spectral background, and nuclear interferences determine detection and reporting limits. | Partial: masking, multi-line review, background/ROI uncertainty, and interference tables exist. Unresolved overlaps and covariance prevent acceptance. |
| Detection-limit / minimum-detectable-activity planning | SLOWPOKE planning includes a priori detection-limit estimates over elements, matrices, and interfering products. | Partial: ROI/statistical tooling exists. A validated SLOWPOKE-specific detection-limit model and reference cases are missing. |
| Flux characterization and monitor reduction | Pool/site flux may vary with position and operating state; monitor activities are used to infer thermal/epithermal/fast components. | Partial: monitor inventory and reaction-rate structures exist. Bound monitor histories, target amounts, cross sections, and covariance are missing. |
| Radioactive-tracer production | SLOWPOKE facilities may produce research/medical tracers in addition to NAA. | Not in the current first-four acceptance scope; no production recipe or radiochemical qualification should be inferred from the analysis modules. |
| Prompt-gamma or radiography modes | These are facility capabilities in some SLOWPOKE installations, but are not equivalent to post-irradiation HPGe activation analysis. | Out of scope unless separately specified; do not claim support from INAA workflows. |

## Required validation layers

1. **Facility profile:** reactor identity, irradiation site, operating power/flux settings, thermal/epithermal/fast spectrum representation, monitor set, sample limits, transfer path, and detector/sample-changer geometry.
2. **Measurement contract:** specimen and monitor IDs, mass/composition, irradiation start/end, transfer delay, count start and live/real time, detector calibration, efficiency, dead-time/summing/attenuation corrections, and source hashes.
3. **Physics contract:** reaction channels, cross-section and decay-data versions, parent feeding, self-shielding, interference list, blank/LOD model, and covariance terms.
4. **Reference cases:** at least one thermal/epithermal monitor case, one fast-interference case, one short-lived transfer case, one long-lived multi-count case, and one matrix/self-shielding case with independently accepted expected results.
5. **Acceptance receipts:** per-case measured-versus-calculated comparison, uncertainty budget, provenance, and explicit unsupported/blocked statuses.

## Current conclusion

FluxForge contains many reusable activation-analysis building blocks, but it cannot currently claim “all SLOWPOKE activation analysis” is supported. The missing facility profile, calibration/timing bindings, covariance-complete reductions, and independent SLOWPOKE reference cases are qualification blockers. SLOWPOKE remains a separate gated workstream; completing the software inventory does not promote it to a completed goal.

The audit is consistent with the facility and literature evidence: Polytechnique describes SLOWPOKE NAA with multiple high-resolution Ge spectrometers and automated sample handling; published SLOWPOKE work treats thermal, epithermal, and fast components, uses activation monitors, and includes fast-neutron `(n,p)`/`(n,alpha)` interferences in detection-limit analysis.
