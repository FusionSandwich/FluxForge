# Issue: Implementation: ASTM E261 Radioactivation Rate Math & Uncertainty Prop

**Status:** Planned
**Context:** Deep Research Report Standardization

## Description & Implementation Mechanics
**Target:** `src/fluxforge/physics/activation.py`
**Goal:** Implement the literal radioactivation equation for computing reaction rates *from* activity with rigorous error propagation arrays as dictated by E261.
**Logic / Details:**
- Implement saturated reaction rate `R_sat` derivation.
- Add variance-covariance matrix propagation for error.
```python
def calculate_e261_reaction_rate(activity_bq, lambda_decay, t_irr, t_cool, t_count, mass, N_A, atomic_weight):
    # Base equation: A = N * R_sat * (1 - exp(-lambda * t_irr)) * exp(-lambda * t_cool)
    # Rearranging for R_sat (saturated reaction rate per target nucleus)
    N_target = (mass * N_A) / atomic_weight
    saturation_term = (1 - math.exp(-lambda_decay * t_irr))
    decay_term = math.exp(-lambda_decay * t_cool)
    
    R_sat = activity_bq / (N_target * saturation_term * decay_term)
    
    # Needs implementation: Partial derivatives for uncertainty propagation 
    # dR/dA, dR/dMass, dR/dT_irr, etc. using `uncertainties` package or numpy arrays.
    return R_sat
```
