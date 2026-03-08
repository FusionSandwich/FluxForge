# Move Flux-Wire Product/Reaction Metadata Out of `flux_wire_analysis.py`

## Problem
`flux_wire_analysis.py` carried hardcoded flux-wire product expectations and reaction metadata. That made the analysis path harder to audit and easy to drift away from the committed validation inputs.

## Current mitigation
- Flux-wire expected products and search-line targets are now loaded from `src/fluxforge/data/flux_wire_catalog.json`.
- The helper API lives in `src/fluxforge/data/flux_wire_catalog.py`.
- `flux_wire_analysis.py` no longer defines `FLUX_WIRE_REACTION_META` or `ELEMENT_TO_ISOTOPES`.

## Remaining work
- Extend the catalog so reaction metadata is represented cleanly for isotopes that appear in multiple wire contexts, especially `Sc46` in both `Sc` and `Ti` wires.
- Mirror any remaining flux-wire sample-property defaults out of analysis code if they are still specific to the RAFM example workflow.
- Add catalog coverage tests for every bundled RAFM flux-wire element/product combination.

## Acceptance
- Flux-wire isotope expectations and line targets are fully data-backed.
- No flux-wire reaction/product mapping remains hardcoded in `flux_wire_analysis.py`.
- Adding or correcting a flux-wire product requires only a data-file change plus tests.
