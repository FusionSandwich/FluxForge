# Independent acceptance checklist

The bundled QG export has exactly one textual Tb154m peak assignment: `examples/RAFM_irradiation/QG_processed_gamma_data/RAFM4/RAFM4-N_15dEOI.txt`, source line 92, center 264.28 keV, assignment `Tb154m@ 265.`, net 65,931 +/- 1,206. This is a candidate for the user's previously corrected exception. The bundled sample gamma library includes Ta182 at 264.076 keV (intensity 0.035084398616). No tracked source-keyed correction manifest or document proving the correction was found. Do not generalize an exception to all Tb154m rows or confuse this with the opposite Ta182-versus-Tb154m misidentification near 1189 keV.

Independent tests required:

1. Inventory every source row by report relative path, report SHA256, source line number, source line text, observed center, source assignment and isotope. Include parser loss checks against raw export row counts.
2. Apply exactly one source-keyed reviewed correction, preserving original Tb identity and correction provenance. Reject wrong file, changed SHA, different line/energy, duplicate correction and missing correction target. Preserve all other rows unchanged.
3. Compare genuinely detected/assigned peaks before any QG report parity manipulation. Empty raw peaks must leave every reference row missing; wrong raw isotope must remain a mismatch. Reference imports and forced parity must never count as recovered peaks.
4. Match one-to-one. A single detected peak near two distinct reference rows can satisfy at most one. Verify deterministic matching independent of source and peak ordering, with explicit energy bounds and same-isotope comparison.
5. Every eligible corrected reference row must be accounted for as matching, missing, mismatched, or unresolved. Never silently discard low-energy, low-count, missing-library or unpaired-report rows when claiming all peaks.
6. Replay raw generic spectra without QG parity (`apply_generic_qg_report_parity` injects missing peaks and overwrites isotope identity). Require an independent baseline/reference match table, retaining raw peak channel, fitted center, fitted sigma/significance and actual gamma-library ID.
7. Separate detection/ID agreement from counts/activity agreement and source nuclear-data correctness. QG per-line counts cannot be copied to make an uncertainty/count replay pass.

Existing hazards:

- `rafm_workflow.match_peak` prefers matching isotope over nearest center within generous 3/3/2 keV tolerance, and is called independently for each row. One raw peak can be reused for multiple references.
- `build_peak_comparison_records` omits report source line identifiers even though `qg_reference_peaks` and line diagnostics preserve them.
- `qg_reference_peaks` drops nonpositive net rows. `build_peak_comparison_records` further filters energy range and minimum net counts.
- Default generic method is qg. `analyze_generic_sample` calls `apply_generic_qg_report_parity`, which fabricates missing peaks with channel zero/fwhm zero and relabels existing peaks; this would hide the Tb correction if applied after automatic identification.
- Existing 1189-keV correction selected the stronger library line and fixed FluxForge identity to QG Ta182. It does not prove the user's QG Tb exception.

## Final implementation review

Owned full RAFM/matching run completed: 34 passed in 346.14 seconds. Matching followup completed: 17 passed in 29.83 seconds. Runs overlap. Aggregate execution receipts and terminal summaries are saved separately; no per-case JUnit output was collected at launch.

The source-bound Tb manifest and corrected_reference_ids implementation preserve original IDs, require exact SHA/line/ID/energy binding, reject duplicate corrections and invalid corrected IDs, and avoid altering native peaks. The single explicit correction matches the sole bundled Tb source row identified independently. The runtime scientific admission remains false.

Native replay construction computes raw detections before parsing the QG report and uses iec_tiered rather than copied QG counts. Flux-wire targeted extraction uses threshold zero, so peak ID agreement alone must not be presented as comprehensive detection-significance qualification.

Outstanding validator review findings sent to parent:

- Validate every expected paired raw and QG input is in input_sha256; a merely nonempty mapping is insufficient.
- Match each cached raw_file to the actual paired raw source, not just qg_file coverage.
- Reject correction manifest report names absent from discovered inputs, rather than silently leaving an exception unused.
- Normalize native-cache peak fields: replay exports net/sigma; match_peak_set currently accesses net_counts in its deterministic candidate ordering. Actual CLI replay must exercise this integration.
- Distinguish native extraction runtime source hashes from the current audit runtime source hashes if extraction started before implementation edits.
