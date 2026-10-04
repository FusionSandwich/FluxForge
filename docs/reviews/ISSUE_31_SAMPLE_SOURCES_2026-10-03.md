# Issue 31: separate reference properties from nominal sample defaults

The unfolding defaults file previously mixed elemental properties, target
fractions, nominal wire masses/geometry/purity, product reactions and simplified
cross-section response parameters. The public helpers now compose distinct
packaged resources:

| Resource | Content |
| --- | --- |
| `flux_wire_catalog.json` | Context-specific product/reaction IDs and gamma search targets |
| `flux_wire_sample_reference.json` | Historical elemental properties and nominal natural target fractions |
| `flux_wire_nominal_samples.json` | Example-only wire mass, geometry and purity |
| `flux_wire_unfolding_defaults.json` | Simplified diagnostic cross-section and response parameters |

The compatibility APIs and module constants retain their numeric values and
public shape. New APIs expose reference and nominal data separately. The
nominal sample metadata explicitly declines scientific admission; its masses
still require `allow_default_mass=True`. The actual UWNR/INL measured sample
metadata remains the source for those specimens. Nominal defaults were not
copied into measured RAFM records or represented as laboratory measurements.

Validation:

```console
python -m pytest -q tests/test_flux_wire_sample_sources.py tests/test_flux_wire_catalog_context.py tests/test_data_reference_modules.py tests/test_no_silent_defaults.py
```

**50 passed.** Tests independently compute target atoms for all seven legacy
wire elements, retain the explicit mass gate, check fresh nested payloads,
and verify that reference resources do not contain nominal sample geometry.
Issue #31 remains open pending integration.
