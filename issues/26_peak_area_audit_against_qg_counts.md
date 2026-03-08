# Issue 26: Peak-Area Audit Against QG Counts

## Problem
We need a defensible statement on whether remaining parity failures are caused by missed area under the peaks or by later activity-conversion steps. Right now that answer is known from ad-hoc inspection, but it is not preserved in the workflow artifacts.

## Required work
1. Use the line-diagnostics CSV to audit peak-area parity directly.
2. Highlight lines where:
   - count parity is good but activity parity is bad
   - both count and activity parity are bad
3. Prefer using this audit before changing the efficiency model.
4. Keep the `80 keV` lower cutoff in place for the audit.

## Acceptance
- The workflow outputs make it obvious whether a failure is a peak-area problem or an activity-conversion problem.
- The audit can be rerun without access to the QG analysis software, using only the committed processed outputs.
