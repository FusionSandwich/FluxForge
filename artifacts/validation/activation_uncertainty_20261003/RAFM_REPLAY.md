The bounded replay passes all 12 bundled QG wire reports, covering 19 reported nuclide activities and 40 supplied line count uncertainties. Generic reaction-rate, shared count-to-activity, ASTM E261, and ASTM E262 uncertainty propagation agrees with an independent chronological activation integral and real-clock acquisition integral. The maximum relative line-sigma error is 1.78e-15.

Run from the isolated checkout with its `src` on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
& 'C:\Users\Josh\projects\FluxForge\.venv\Scripts\python.exe' tools/validate_activation_uncertainty_rafm.py --root . --out artifacts/validation/activation_uncertainty_20261003/rafm_replay_fresh.json
```

The tool refuses to overwrite an existing receipt. The JSON records all report and metadata input hashes, runtime source hashes, tool hash, individual comparisons, and three independent full-source covariance unit-rescaling checks. Rerun after source edits to bind the evidence to final source bytes.

Supplied line sigmas are 1.07–10.46 times the simple square root of the net counts. For Co-Cd at 1173 keV, the report supplies 11202 ± 520 counts; replacing 520 with sqrt(11202) understates that supplied uncertainty by 4.91 times.

These are propagation diagnostics. Header uncertainties retain an unknown reported-total composition, and line sigma is never added to header sigma. Line checks fix efficiency and gamma yield to one and use explicit unit inventory/cross section; resulting line activities and fluences are conditional mathematical outputs, not physical sample estimates. Cu-Cd, bare Cu, and Fe-Cd lack exact filename schedule joins and explicitly use a conditional common phase-two irradiation plus naive report-date subtraction. Timing, calibration, nuclear-data, and material covariance remain unqualified. The frozen 13×20 response evidence supplies a diagonal rate-sigma vector, not a qualified full physical rate covariance. Scientific admission remains false.
