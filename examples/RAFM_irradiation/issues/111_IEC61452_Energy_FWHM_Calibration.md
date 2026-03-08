# Issue: Implementation: IEC 61452 / ASTM E3376 Energy & FWHM Calibration Base

**Status:** Planned
**Context:** Deep Research Report Standardization

## Description & Implementation Mechanics
**Target:** `src/fluxforge/analysis/detector_calibration.py`
**Goal:** Fit energy (keV) and resolution (FWHM) as a function of channel.
**Logic / Details:**
```python
def fit_fwhm_curve(energies, fwhms, weights):
    # IEC 61452 defines FWHM(E) typically as: FWHM_E = a + b*E^0.5
    # Let x = sqrt(E). We do a weighted linear fit.
    x = np.sqrt(energies)
    slope, intercept = np.polyfit(x, fwhms, 1, w=weights)
    return {'a': intercept, 'b': slope}
    
def test_calibration_quality(fwhm_residuals):
    # ASTM E3376 asserts acceptable drift/shift boundaries. Flag if off > x%
    pass
```
