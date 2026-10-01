# UWNR data-use replay receipt

The final replay is bound to source/metadata commit `27839e128dabbc7b901f7a1acf80f19ba68d949d`.
Runtime acceptance and all 240 affected tests are bound to `7b24c0b783b6504e8193138965aa45780720c9bf`;
the later change only completes the documented CLI command. All runtime/test source hashes are unchanged.

`receipt.json` retains 31 reports, 288 ROI rows and 88 summaries and verifies 144 input hashes.
All 32 native ANS and 30 campaign ASC files are hash joined; 31/29 belong to reported measurements.
The 32nd RAFM-A-2hr count lacks a corroborated QG report. Six older ASC files have explicit historical exclusions.
Native ASCII labels corroborate identity only; undocumented coefficient fields and GammaLib contents remain unknown.

`operator_receipt.json` retains all 13 rates and uncertainties exactly, seven aggregate rows,
four reaction-variant comparisons and twelve invalid-input rejections. Actual covariance shape is 13×13.
Named actual terms are activity, Ti/Cd model terms and model floor; all rows still lack six measured components.
No measured shared-source IDs, calibration covariance or history covariance are supplied.
The three Ti counts belong to one irradiated monitor and do not become independent activation experiments.

The authorized curve CSV retains its original SHA256. Seventeen nonpositive values remain raw and excluded.
The vendor Error coefficient is not a standard uncertainty. Conditional profile comparison does not qualify calibration.
Physical inclusion is zero and scientific admission remains false. Unknown source/library/calibration/history/uncertainty gates remain.
No activity or summary correction, raw spectrum refit, private mail or operating-workbook content is included.

`source_binding.json` distinguishes local text-byte hashes from committed Git blob hashes.
The scoped `.gitattributes` preserves every evidence file byte; `artifact_sha256.json` excludes itself.
