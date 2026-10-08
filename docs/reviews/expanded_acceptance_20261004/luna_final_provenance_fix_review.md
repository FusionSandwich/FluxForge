# Final instrument provenance fix review (read-only)

Reviewed commit `2f915f3f1e462273d17350d49113fbd2a985ce80` and its focused source/test diff. No blocking issue found.

`_save_instrument_inputs` centralizes editor persistence for both capture and spectrum switching. It removes the spectrum's override record when every field is blank, preventing an empty stale dictionary from surviving document replacement. A nonblank string such as `"0"` remains present (`strip()` is nonempty), preserving explicitly recorded zero high voltage. For partial entries, blanks remain in the mapping but `instrument_provenance` treats them as absent and falls back to imported metadata, which matches its existing contract.

The added acquisition-identity regression retains manual gain through a calibration-only update, replaces the spectrum's counts object under the same document and spectrum IDs, and asserts both the visible gain and stored override are cleared. It then checks replacement by a new example workspace also clears the override. This exercises the replacement signal path rather than merely invoking the helper.

No tests were run because the parent is running the affected test files in separate processes. Review was read-only; the unrelated untracked `CRASH` file was not inspected.
