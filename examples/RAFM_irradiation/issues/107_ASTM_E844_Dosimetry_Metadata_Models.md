# Issue: Implementation: ASTM E844 Dosimetry Schema and Traceability

**Status:** Planned
**Context:** Deep Research Report Standardization

## Description & Implementation Mechanics
**Target:** `src/fluxforge/physics/dosimetry_models.py`
**Goal:** Define standard-compliant data structures representing sensor sets, foil impurity tracking, and QC boundaries required for reactor surveillance (ASTM E844).
**Logic / Details:**
- Use PyDantic or `dataclasses` to require standard fields.
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class E844FoilSensor:
    id: str
    target_isotope: str # e.g. "Fe54"
    mass_g: float
    isotopic_abundance: float
    impurities_ppm: dict = field(default_factory=dict)
    qc_trace_number: Optional[str] = None
    encapsulation_material: Optional[str] = "Bare"

    def validate_mass_uncertainty(self):
        # E844 mandates mass uncertainties < 1% for standard geometries
        pass
```
