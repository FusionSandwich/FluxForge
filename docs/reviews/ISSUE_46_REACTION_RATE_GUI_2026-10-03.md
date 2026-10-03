# Issue 46: activity-to-reaction-rate UI

Added a native Qt workspace with editable isotope, target-element, measured
mass/composition, half-life and EOI activity fields, plus resolved reaction,
target-atom and saturated SigPhi results. It shares the corrected activation
API published by the analysis owner and integrated into acceptance commit
`536fd07`; no duplicate activation math or fake count-based activity sigma was
introduced. The target-atom API receives both explicit composition fractions
and never opts into a nominal mass.

Found isotope lines are added without copying count-reference activity into an
EOI field. Missing sigma remains unavailable; supplied sigma is visibly scoped
to activity-only propagation. User edits and new histories clear stale results,
and JSON export recalculates before writing. The editor is limited to catalog
products and does not qualify a complete uncertainty budget or physical solve.

Validation: **50 passed** across translator, irradiation-history and catalog
checks. Independent arithmetic verifies target atoms, single-segment relative
power, rate normalization and absolute activity sigma. Boundary checks include
zero activity with positive sigma, missing inputs, incompatible products,
nonfinite fractions and zero buildup. Qt checks cover repeated found-row import,
blank EOI/mass fields, conversion, current-history export, stale-output clearing,
row removal and window reuse. The production catalog and existing irradiation /
reference-parity regression checks are run separately.
