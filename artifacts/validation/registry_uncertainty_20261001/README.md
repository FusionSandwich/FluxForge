# Registry uncertainty qualification evidence

Frozen runtime/test/script commit: `b465604d8a364dce1fae4f5ba0d648cc540c0b94`.
Base PR213 head: `f672374e9a0dcf82e47d9d8a0bd11d4b3808d142` (preserved).
Related issues: #195 (ridge/proxy) and #214 (RMLE default percentage).

`receipt.json` SHA256: `52f384c163a7ea60a440ae8ce96b1396ae41219ad74bde28f9f2067784bb5918`.
It binds 34 input files and 20 runtime/test/script files, with both working-byte
and frozen Git-blob SHA256 values. The actual 13-row, 20-group weighted response
has numerical rank 7. All six recorded GRAVEL/MAXED flux estimates, predictions,
standardized residuals and convergence flags match the historical replay exactly.
All three GRAVEL cases remain nonconverged; all three MAXED cases converge
numerically. None has qualified bin uncertainty. MAXED cases vary entropy prior
and initial iterate together. The two actual-array CLI artifacts retain covariance,
flux uncertainty and predicted-rate uncertainty as null, with explicit reasons.
Their unknown is u=group-integral flux/1e11; source energy edges are retained.

`affected_tests.txt`: 213 passed, 2 unrelated CLI cases deselected, 18 warnings,
12.55 seconds. Command (existing Python only, offline, Qt offscreen):

```
python -m pytest -p no:cacheprovider tests/test_registry_uncertainty_qualification.py tests/test_unfolding_registry.py tests/test_unfolding_workflows.py tests/test_unfolding_workspace_qt.py tests/test_artifacts_io.py tests/test_cli_app.py tests/test_rmle.py -k 'not test_cmd_spectrum and not test_cmd_detect'
```

The qualification tests cover linear identity/per-row unit changes, extreme finite
sigmas, rank deficiency, nonconvergence, null JSON validation and CSV/text reasons,
RMLE count identity with unchanged [25,400] flux, default/one/four MC samples,
forced optimizer/Gaussian covariance failure, and finite-flux covariance overflow.
Qt tests exercise N/A reasons and disabled bands for unavailable built-in methods.
MC results are unqualified; failed/fallback replicate counts remain diagnostic.
No new activation-rate Poisson fit, transport, dependency acquisition or source
correction was performed. The warning from the existing ill-conditioned RMLE
Gaussian probe remains a diagnostic; its registry uncertainty is unavailable.

Measured rates, masses, activities and histories remain unchanged. Calibration,
history/event, geometry and full measured covariance gates remain unresolved.
Scientific admission is false. Frozen PR212/PR213 evidence remains intact.

`acceptance.json` records independent Sol review of the exact frozen source and
decisive evidence. `artifact_manifest.json` binds the publication payload bytes.
