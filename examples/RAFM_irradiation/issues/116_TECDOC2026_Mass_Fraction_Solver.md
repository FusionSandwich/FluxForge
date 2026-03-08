# Issue: Implementation: k0-NAA Mass Fraction Determination

**Status:** Planned
**Context:** Deep Research Report Standardization

## Description & Implementation Mechanics
**Target:** `src/fluxforge/physics/k0_naa.py`
**Goal:** Execute the primary k0-method equations solving for ppm.
**Logic / Details:**
- Relates unknown element to standard k0_Au factors.
```python
def k0_mass_fraction(A_sp, A_sp_asterisk, k0_factor, f, alpha, eff_ratio, Q0_alpha, Q0_alpha_asterisk):
    # A_sp: specific activity of unknown
    # A_sp_asterisk: specific activity of reference monitor (e.g., Au-198)
    # Mass fraction rho = (A_sp / A_sp_asterisk) * (1 / k0_factor) * (f + Q0_alpha_asterisk) / (f + Q0_alpha) * eff_ratio
    numerator = f + Q0_alpha_asterisk
    denominator = f + Q0_alpha
    mass_fraction = (A_sp / A_sp_asterisk) * (1.0 / k0_factor) * (numerator / denominator) * eff_ratio
    return mass_fraction
```
