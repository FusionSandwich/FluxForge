# Externalize Flux-Wire Sample and Reaction Defaults from `flux_unfold.py`

## Problem
Even after removing the hardcoded product catalog from `flux_wire_analysis.py`, `flux_unfold.py` still carries hardcoded flux-wire sample-property defaults, isotope fractions, and reaction lookup tables.

## Why this matters
- It is the same maintainability problem in a different layer of the workflow.
- It makes the unfolding path harder to audit against committed RAFM example metadata.
- It increases the chance that the activity-analysis layer and unfolding layer drift apart.

## Current evidence
- `FLUX_WIRE_SAMPLES`
- `THERMAL_CROSS_SECTIONS`
- `REACTION_ENERGIES`
- parts of isotope/reaction mapping logic in `get_reaction_id()` and `get_isotope_fraction()`

## Current mitigation
- The simplified unfolding defaults are now loaded from `src/fluxforge/data/flux_wire_unfolding_defaults.json`.
- `flux_unfold.py` keeps compatibility-level module constants, but they are data-backed views instead of hardcoded dictionaries.
- Sample-element/product reaction lookup is now resolved from bundled metadata instead of inline special-case maps.

## Remaining work
- Separate immutable physics/reference data from example-specific sample metadata.
- Move example sample masses and isotopic fractions into committed RAFM metadata where appropriate.
- Keep only generic algorithmic logic in `flux_unfold.py`.
- Add tests that confirm unfolding uses bundled data sources rather than hardcoded dictionaries in code.

## Acceptance
- Flux-wire unfolding no longer depends on large hardcoded data dictionaries inside `flux_unfold.py`.
- Sample-property defaults and reaction metadata are loaded from committed data/metadata files with tests.
