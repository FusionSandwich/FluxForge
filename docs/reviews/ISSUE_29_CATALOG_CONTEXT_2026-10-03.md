# Issue 29: wire context in the shared reaction catalog

Sc-46 occurs in both Sc and Ti wires. The spectroscopy catalog previously
listed both expected elements but carried only the Sc-45 capture label; a
separate unfolding table carried the Ti-46 production reaction. The two tables
could drift, and an incompatible supplied element could fall back to another
wire's reaction.

The catalog now carries per-element reaction labels and full reaction IDs.
Spectroscopy metadata preserves every context, and unfolding builds its
product map from that same catalog. A supplied incompatible context returns
`Unknown(...)`; an omitted context for a product with different production
reactions raises an actionable error. The context-free catalog accessor keeps
its historical primary-parent fields for compatibility and exposes both maps.
The legacy catalog browser displays every context.

All 13 element/product combinations have reaction and decay-line coverage,
including the previously omitted Ti-50 capture route to Ti-51. Its nominal
natural Ti-50 fraction is 0.0518, sourced from the
[CIAAW titanium table](https://www.ciaaw.org/titanium.htm).
This reference composition is not a measured composition of an enriched or
depleted specimen. Missing target fractions now raise instead of implying a
pure target isotope. No cross section or physical response is synthesized for
the newly exposed Ti-51 route.

Validation in the existing Windows Python 3.12 environment:

```console
python -m pytest -q tests/test_flux_wire_catalog_context.py tests/test_data_reference_modules.py tests/test_no_silent_defaults.py tests/test_flux_wire_parity.py tests/test_rafm_workflow.py
```

Result: **70 passed**. The real-data RAFM/wire regressions retained their
existing behavior. This verifies software metadata consistency; it does not
qualify calibration, peak recovery, measured covariance or scientific accuracy.
Issue #29 remains open for integration of this branch.
