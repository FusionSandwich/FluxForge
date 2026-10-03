# Activity to reaction-rate workspace

Open **Tools → Activity to Reaction Rate** after applying a history in
**Tools → Irradiation History**. The table links product identity, explicit
specimen constraints, product half-life and EOI activity to a saturated rate.
It currently supports the flux-wire product/reaction catalog. Target element
resolves products that can arise from more than one reaction (such as Sc-46).

Enter monitor mass in mg, the target element's mass fraction in that monitor,
and the target isotope's atom fraction within that element. Both fractions
must be explicitly supplied in (0, 1]. The table does not infer a nominal
wire mass, purity or natural enrichment. The shared target-atom calculation
uses the bundled elemental molar mass; use the qualified physical workflow
when sample-specific reference inputs are required.

Activity must already be referenced to the end of irradiation. **Add found
isotopes** adds product names, half-lives and the original count-reference
activity in a read-only comparison column. It leaves EOI activity and mass
constraints blank. Repeated imports do not duplicate the same product/line.
Individual lines are not automatically averaged or treated as independent.

**Convert** uses the shared activation engine to calculate the saturation
production rate and divides by target atoms to display SigPhi per target atom
per second. The calculation honors relative power and decay during gaps.
Missing history, zero buildup, incompatible element/product pairs and
incomplete or nonfinite measurements stop conversion with an error.

An optional activity sigma is propagated conditional on the fixed specimen,
half-life and history inputs. The result is labeled `activity_only_conditional`;
blank sigma remains `unavailable`. Zero activity retains a supplied positive
sigma. This is not a complete uncertainty budget or a qualified unfolding input.

Changing fields or applying a different history clears stale output. Export
JSON recomputes against the current history and includes inputs, duration
segments, resolved reactions, target atoms, rates, sigma scope, and explicit
`scientific_admission: false` / `complete_uncertainty_budget: false` markers.
Invalid export does not replace an existing file. Closing/reopening the
workspace retains rows for the current window; they are not yet persisted in
workspace session files.
