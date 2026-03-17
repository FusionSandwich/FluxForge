import os
import glob

issues_dir = "/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples/RAFM_irradiation/issues"

# Find max ID
existing_files = glob.glob(os.path.join(issues_dir, "*.md"))
max_num = 0
for f in existing_files:
    basename = os.path.basename(f)
    if basename[0].isdigit():
        try:
            num = int(basename.split("_")[0])
            max_num = max(max_num, num)
        except ValueError:
            pass

start_num = max_num + 1

tasks = [
    # ASTM E261 / E844
    (
        "ASTM_E844_Dosimetry_Metadata_Models.md",
        "Implementation: ASTM E844 Dosimetry Schema and Traceability",
        """**Target:** `src/fluxforge/physics/dosimetry_models.py`
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
```""",
    ),
    (
        "ASTM_E261_Reaction_Rate_Math.md",
        "Implementation: ASTM E261 Radioactivation Rate Math & Uncertainty Prop",
        """**Target:** `src/fluxforge/physics/activation.py`
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
```""",
    ),
    (
        "ASTM_E261_E844_Unit_Tests.md",
        "Testing: E261 Fraction Uncertainty & E844 Schema Rules",
        """**Target:** `tests/test_astm_dosimetry.py`
**Goal:** Unit test constraints mapping to ASTM allowed variance budgets and ensuring missing QC metadata flags halt analysis.""",
    ),
    (
        "ASTM_E261_E844_Workflow_Example.md",
        "Example: End-to-End E844 Sensor Set to E261 Reaction Rate Matrix",
        """**Target:** `examples/ASTM_E844_reactor_surveillance.py`
**Goal:** Example script scaffolding a multi-foil set (Fe, Ni, Ti), calculating specific reaction rates mapped for a solver like STAYSL.""",
    ),
    # IEC 61452 / ASTM E3376
    (
        "IEC61452_Energy_FWHM_Calibration.md",
        "Implementation: IEC 61452 / ASTM E3376 Energy & FWHM Calibration Base",
        """**Target:** `src/fluxforge/analysis/detector_calibration.py`
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
```""",
    ),
    (
        "ASTM_E3376_Efficiency_Curve_Fit.md",
        "Implementation: ASTM E3376 Polynomial Efficiency Curve Models",
        """**Target:** `src/fluxforge/analysis/detector_calibration.py`
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
```""",
    ),
    (
        "IEC61452_Corrections_TCS_PileUp.md",
        "Implementation: IEC 61452 Coincidence & Pile-Up Math",
        """**Target:** `src/fluxforge/corrections/corrections.py`
**Goal:** Apply standard bounds for pile-up/random-sum losses and skeleton out true coincidence summing (TCS).
**Logic / Details:**
```python
def calculate_pileup_correction(gross_count_rate_cps, pulse_shaping_tau):
    # Basic IEC pile-up correction for random summing based on total system rate
    # True count rate R_t ~ Measured R_m * exp(R_t * 2*tau)
    # Using approx valid for dead times < 10%
    correction_factor = np.exp(gross_count_rate_cps * 2 * pulse_shaping_tau)
    return correction_factor
```""",
    ),
    # IAEA TECDOC-2026 k0-NAA Models
    (
        "TECDOC2026_Facility_Characterization.md",
        "Implementation: TECDOC-2026 k0-NAA Monitor Factors (f, alpha)",
        """**Target:** `src/fluxforge/physics/k0_naa.py`
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
```""",
    ),
    (
        "TECDOC2026_Activation_Corrections.md",
        "Implementation: k0-NAA Intermittent Irradiation & non-1/v",
        """**Target:** `src/fluxforge/physics/k0_naa.py`
**Goal:** Support TECDOC required Westcott g(T) adjustments and intermittent beam power array modeling.""",
    ),
    (
        "TECDOC2026_Mass_Fraction_Solver.md",
        "Implementation: k0-NAA Mass Fraction Determination",
        """**Target:** `src/fluxforge/physics/k0_naa.py`
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
```""",
    ),
    # IAEA TRS 487 QA/QC
    (
        "TRS487_Traceability_Provenance_Engine.md",
        "Implementation: TRS-487 Traceability & Library Stamping",
        """**Target:** `src/fluxforge/qaqc/traceability.py`
**Goal:** Provide ISO-17025 style constraints where every computed value stores the UUID and git/version hash of the nuclear data utilized.
**Logic / Details:**
- Attach provenance logs explicitly to every `Spectrum` and `Measurement` object output.""",
    ),
    (
        "TRS487_Blank_Interference_Correction.md",
        "Implementation: TRS-487 Spectrum Interferences & Blank Handling",
        """**Target:** `src/fluxforge/qaqc/interference.py`
**Goal:** Explicit logic tagging known threshold matrix reactions (e.g. Al(n,a)Na-24 masking independent Na results) and generating Blank subtraction margins as defined uniformly by TRS 487 guidance.""",
    ),
]

for filename, title, desc in tasks:
    num = start_num
    start_num += 1
    full_name = f"{num:02d}_{filename}"
    path = os.path.join(issues_dir, full_name)

    with open(path, "w") as f:
        f.write(f"# Issue: {title}\n\n")
        f.write("**Status:** Planned\n")
        f.write(f"**Context:** Deep Research Report Standardization\n\n")
        f.write("## Description & Implementation Mechanics\n")
        f.write(f"{desc}\n")
    print(f"Created {full_name}")
