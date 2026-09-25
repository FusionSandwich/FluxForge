# HPGe efficiency calibration in the analysis workspace

Open **Activity Results → Fit efficiency** after selecting a spectrum. Add measured calibration lines, import a CSV, or explicitly seed demo points. Select a model and fit it before applying. The dialog shows measured and fitted absolute efficiency, measured-minus-fitted percentage residuals, a ±3% review band, point status, covariance, and model comparisons. Applying the fit stores its points and detector metadata in the selected spectrum's detector profile; undo restores the prior profile.

## Calibration input

CSV import accepts UTF-8 CSV with these exact headers (column order may vary):

```csv
energy_keV,net_counts,count_uncertainty,live_time_s,activity_bq,activity_rel_unc,emission_probability,probability_uncertainty,geometry_factor
661.657,19000,138,100,100000,0.02,0.851,0.002,1
```

Every row must contain nine finite numbers. Energy, counts, live time, activity, emission probability, and geometry factor must be positive; emission probability cannot exceed one. Count uncertainty is absolute counts, activity uncertainty is a relative fraction, and probability uncertainty is an absolute probability. A zero count uncertainty uses the supplied value as zero; leave it unspecified only through the Python API to request the Poisson default. CSV import replaces the table only after every row passes validation. Certificate and PDF import are unsupported.

The table has an additional **Activity Source ID** column. Give lines that share one source's activity certificate the same non-empty ID. Source IDs are preserved with the points and make their activity uncertainties fully correlated in fitting and chi square review. The exact nine-column CSV leaves this column blank; enter IDs after import. A blank ID means the correlation is unknown, so any nonzero activity uncertainty requires review.

The calculated absolute efficiency is `net_counts / (live_time_s × activity_bq × emission_probability × geometry_factor)`. Keep activity and geometry factors traceable to the calibration measurement. A result outside `(0, 1]` is rejected.

## Fit and review

Log polynomial and Gray fits use generalized least squares in log efficiency with the activity-source covariance. The semi empirical HPGe fit uses the same covariance in nonlinear least squares. With fewer than eight distinct lines, its two Ge length terms are fixed reference assumptions and its covariance is conditional on those values. A fitted detector dimension, window attenuation, or dead layer must not be inferred from that conditional fit.

The fit review requires both a maximum absolute point residual of at most 3% and a chi square upper-tail p value of at least 0.05. If there are no residual degrees of freedom, review remains required. Count and line-probability errors are treated as independent; activity errors are fully correlated for lines with the same Activity Source ID. Correlations beyond that grouping, including between different certificates or gamma emission probabilities, are not modeled. The p value and covariance remain conditional on these assumptions. Independent standard or replicate measurements are needed to establish a validated uncertainty claim.

Detector geometry, legacy C1–C4 coefficients, and certificate fields in the dialog are persisted metadata. They do not change the fitted efficiency curve. Each calibration point's `geometry_factor` does change the measured efficiency used in fitting; it is not an automatic transfer correction to a different sample geometry. Applying the curve to another geometry requires a separately validated correction.

The active spectrum's profile supplies efficiency and fit uncertainty to activity review. Applying a new fit invalidates prior activity results; switching spectra clears prior activity results. The measured spectrum counts are unchanged. Session persistence stores the profile, fit data, points, covariance, and provenance together. Older workspace fits migrate to the active spectrum's profile on load.

## Verification scope

The branch 03 tests cover model fixtures, malformed input and covariance counterexamples, CSV and Qt interaction, profile migration, session round trips, and undo. Native Windows and Linux GUI evidence and any unavailable platform or installer checks are recorded in the branch evidence when performed. The broader parity ledger remains conditional until unmodeled cross-certificate and gamma-probability correlations, geometry transfer, certificate ingestion, standards reference fixtures, and release platform gates are complete.
