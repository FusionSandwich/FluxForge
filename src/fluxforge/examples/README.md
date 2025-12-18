# UWNR example scaffold

This folder now includes a concrete activation-wire example based on the
Fe-Cd-RAFM-1 wire (Fe58(n,γ)→Fe59) measured at 0 cm with a Cd cover. The JSON
files under `fe_cd_rafm_1/` can be used directly with the CLI to build a simple
response matrix and perform GLS spectrum adjustment.

* `boundaries.json`: coarse energy group edges in eV (converted from the reported
  3.05e-8–6.36e-1 MeV span).
* `cross_sections.json`: placeholder group-averaged cross sections in barns for
  Fe58(n,γ)→Fe59. Replace with IRDFF-II or another dosimetry-grade library when
  available.
* `number_densities.json`: atom densities derived from the 3.0209 mg Fe wire
  mass; adjust to your exact geometry units (atoms/barn-cm) as needed.
* `measurements.json`: gamma-line counts, efficiencies, and irradiation segment
  for the reported HPGe assay (live time 43.2 ks, dead time 0.1%, four Fe-59
  lines). Activities are assumed to be reported at the measurement date; adjust
  cooling times if your reference differs.
* `prior_flux.json`: illustrative prior spectrum for GLS adjustment.

Example usage:

```bash
# Build response matrix and infer spectrum for the Fe-Cd-RAFM-1 wire
python -m fluxforge.cli.app build-response \
  --cross-section-file src/fluxforge/examples/fe_cd_rafm_1/cross_sections.json \
  --number-densities-file src/fluxforge/examples/fe_cd_rafm_1/number_densities.json \
  --boundaries-file src/fluxforge/examples/fe_cd_rafm_1/boundaries.json \
  --output fe_cd_rafm_response.json

python -m fluxforge.cli.app infer-spectrum \
  --measurements-file src/fluxforge/examples/fe_cd_rafm_1/measurements.json \
  --response-file fe_cd_rafm_response.json \
  --prior-flux-file src/fluxforge/examples/fe_cd_rafm_1/prior_flux.json \
  --output fe_cd_rafm_spectrum.json
```

Note: the cross sections and number density are placeholders to demonstrate file
formats and CLI flow. Replace them with library values (e.g., IRDFF-II Fe58(n,γ)
reaction data) and geometry-specific number densities before drawing physics
conclusions. The SpecKit and Neutron-Unfolding repositories show similar
response-matrix and measurement JSON/YAML patterns.
