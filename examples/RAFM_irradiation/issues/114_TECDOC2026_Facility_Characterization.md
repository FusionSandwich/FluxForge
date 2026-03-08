# Issue: Implementation: TECDOC-2026 k0-NAA Monitor Factors (f, alpha)

**Status:** Planned
**Context:** Deep Research Report Standardization

## Description & Implementation Mechanics
**Target:** `src/fluxforge/physics/k0_naa.py`
**Goal:** Determine the thermal-to-epithermal flux ratio (f) and epithermal shape factor (alpha).
**Logic / Details:**
- Derive `f` from Bare and Cd-covered Au monitors.
```python
def compute_f_cadmium_ratio(A_bare, A_cd, Q0, F_cd, G_th, G_epi):
    # Formula derived from k0-NAA standard practices
    # Rc = Cadmium ratio = A_bare / A_cd
    Rc = A_bare / A_cd
    # f = (G_epi / G_th) * Q0 / ( (Rc / F_cd) - 1 )
    f_val = (G_epi / G_th) * Q0 / ((Rc / F_cd) - 1)
    return f_val
```
