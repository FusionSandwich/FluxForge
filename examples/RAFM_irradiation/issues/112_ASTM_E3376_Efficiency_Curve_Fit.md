# Issue: Implementation: ASTM E3376 Polynomial Efficiency Curve Models

**Status:** Planned
**Context:** Deep Research Report Standardization

## Description & Implementation Mechanics
**Target:** `src/fluxforge/analysis/detector_calibration.py`
**Goal:** Implement full-energy peak efficiency modeling handling low/high energy crossovers (dual log-log polynomials) typical of ASTM rules.
**Logic / Details:**
```python
def fit_efficiency_curve(energies, efficiencies, uncs, degree=5):
    # Typically fit in log-log space: ln(eff) = sum_i( a_i * (ln(E/E0))^i )
    log_E = np.log(energies / 1000.0) # Normalized to 1 MeV arbitrarily
    log_eff = np.log(efficiencies)
    
    # weights inversely proportional to variance of ln(eff) (~ unc / eff)
    w = efficiencies / uncs 
    
    coeffs, cov = np.polyfit(log_E, log_eff, deg=degree, w=w, cov=True)
    return coeffs, cov
```
