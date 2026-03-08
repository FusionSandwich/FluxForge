# Audit Measurement-Time Activity Parity Against QG

## Problem
Peak matching in the RAFM and flux-wire workflows is substantially better than the remaining activity validation, but several matched spectra still fail measurement-time activity parity against the QG processed files.

## Current evidence
- `testing_validation/rafm_results/FLUX_WIRE_REPORT.md` shows the expected flux-wire lines are being found reliably.
- The current RAFM reports show that many failures now come from activity parity rather than missing peaks.
- The current Ti, In-Cd, Ni, and Sc-Cd flux-wire discrepancies are dominated by measurement-time activity mismatch, not EOI back-correction.

## Remaining work
- Compare FluxForge and QG activity calculations line-by-line for the remaining failing RAFM3, RAFM4, and Ti-wire spectra.
- Separate low-energy efficiency issues from true peak-area discrepancies.
- Document which failures are measurement-time only and which propagate into EOI/reaction-rate outputs.

## Acceptance
- Every remaining failing spectrum is categorized as peak-area limited, efficiency limited, or export-data limited.
- The RAFM summary and reports clearly distinguish measurement-time parity failures from EOI-only differences.
