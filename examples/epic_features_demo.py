#!/usr/bin/env python3
"""
FluxForge New Epic Features Demo

Demonstrates all the newly implemented features from:
- Epic R: Becquerel parity (auto-calibration)
- Epic S: GMApy parity (GLS workflow)
- Epic T: Curie parity (stacked targets, spectrum export)
- Epic U: NPAT parity (stopping power)
- Epic W: PyUnfold parity (covariance options)
- Epic X: actigamma parity (metastable states)
- Epic Y: SpecKit parity (log-smoothness, ddJ convergence)
"""

import math
import tempfile
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("FluxForge New Epic Features Demo")
print("=" * 70)


# =============================================================================
# EPIC R: AUTO-CALIBRATION
# =============================================================================
print("\n" + "=" * 70)
print("EPIC R: Auto-Calibration (R1.5)")
print("=" * 70)

from fluxforge.analysis.auto_calibration import (
    CALIBRATION_SOURCES, find_peak_candidates, auto_calibrate
)
import numpy as np

# Show available calibration sources
print("\nAvailable calibration sources:")
for source, lines in list(CALIBRATION_SOURCES.items())[:5]:
    energies = [f"{line[0]:.1f}" for line in lines[:3]]
    print(f"  {source}: {', '.join(energies)} keV ...")

# Create synthetic spectrum
print("\nGenerating synthetic Co-60 spectrum...")
n_ch = 2048
counts = np.random.poisson(50, n_ch).astype(float)

# True calibration: E = 0.0 + 1.5 * ch (keV)
for E_peak in [1173.23, 1332.49]:  # Co-60 lines
    ch = int(E_peak / 1.5)
    for dch in range(-30, 31):
        if 0 <= ch + dch < n_ch:
            counts[ch + dch] += 3000 * np.exp(-dch**2 / 100)

# Find peaks
peaks = find_peak_candidates(counts.tolist(), min_prominence=500)
print(f"Found {len(peaks)} peaks at channels: {[p[0] for p in peaks]}")

# Auto-calibrate
result = auto_calibrate(counts.tolist(), sources=['Co-60'])
if result:
    a, b = result.coefficients[:2]
    print(f"Auto-calibration: E = {a:.2f} + {b:.4f} * ch")
    print(f"True calibration: E = 0.0 + 1.5 * ch")
    print(f"Matched {len(result.matches)} lines")


# =============================================================================
# EPIC S: GLS WORKFLOW
# =============================================================================
print("\n" + "=" * 70)
print("EPIC S: GLS Workflow (S1.6, S1.8)")
print("=" * 70)

from fluxforge.evaluation.gma_workflow import (
    Experiment, ExperimentalDatabase, GMAWorkflow
)

# Create experimental database
db = ExperimentalDatabase(name="Cross-Section Evaluation Demo")

np.random.seed(42)
true_xs = lambda E: 100 * np.exp(-0.1 * E) * (1 + 0.5 * np.sin(E))

for i, E in enumerate(np.linspace(1, 20, 12)):
    true_val = true_xs(E)
    measured = true_val + np.random.normal(0, true_val * 0.05)
    db.add_experiment(Experiment(
        name=f"Exp-{i+1}",
        value=measured,
        uncertainty=true_val * 0.05,
        energy_MeV=E,
        reaction="Example(n,g)",
        laboratory="Demo Lab"
    ))

print(f"Created database with {len(db.experiments)} experiments")

# Save to JSON
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    json_path = Path(f.name)
db.to_json(json_path)
print(f"Saved to JSON: {json_path.stat().st_size} bytes")

# Run GLS workflow
workflow = GMAWorkflow("Demo Evaluation")
workflow.load_experiments(db)

# Prior: polynomial fit
prior_vals = np.array([100, -5, 0.1])
prior_cov = np.diag([20, 2, 0.05])**2
workflow.set_prior(prior_vals, prior_cov, ['c0', 'c1', 'c2'])

# Response: polynomial model
def response(params):
    return np.array([
        params[0] + params[1] * e.energy_MeV + params[2] * e.energy_MeV**2
        for e in db.experiments
    ])

workflow.calculate_sensitivities(response)
result = workflow.run_gls()

print(f"\nGLS Results:")
print(f"  Chi-squared/dof: {result.chi2_per_dof:.2f}")
print(f"  Posterior: c0={result.posterior_values[0]:.2f}±{result.posterior_uncertainties[0]:.2f}")

json_path.unlink()


# =============================================================================
# EPIC T: STACKED TARGET & SPECTRUM EXPORT
# =============================================================================
print("\n" + "=" * 70)
print("EPIC T: Stacked Target & Spectrum Export (T1.4-T1.6, T1.9)")
print("=" * 70)

from fluxforge.physics.stacked_target import (
    StackedTarget, Projectile, CrossSectionLibrary
)
from fluxforge.io.spectrum_export import SpectrumExporter, SpectrumMetadata

# Create stacked target
stack = StackedTarget(
    projectile=Projectile.PROTON,
    beam_energy_MeV=25.0
)

stack.add_foil('aluminum', 50)   # Degrader
stack.add_foil('copper', 20, "(p,n)", "Cu-63", "Zn-63")
stack.add_foil('aluminum', 100)  # Degrader
stack.add_foil('titanium', 25)   # Monitor
stack.add_foil('aluminum', 150)  # Degrader
stack.add_foil('copper', 20, "(p,n)", "Cu-63", "Zn-63")

print(f"Stacked target: {len(stack.foils)} foils, {stack.beam_energy_MeV} MeV beam")

# Calculate energies
energies = stack.calculate_energies()
print("\nEnergy at each foil:")
for e in energies:
    foil = stack.foils[e.foil_index]
    print(f"  {foil.material.name:10} ({foil.thickness_um:4.0f}μm): "
          f"E={e.energy_mean_MeV:.2f}±{e.straggling.sigma_MeV:.3f} MeV")

# Cross-section lookup
print("\nCross-section library:")
for rxn in CrossSectionLibrary.MONITOR_REACTIONS:
    lib, name = rxn.split(":", 1)
    xs_data = CrossSectionLibrary.MONITOR_REACTIONS[rxn]
    print(f"  {name}: {lib}, {len(xs_data.energies_MeV)} points")

# Spectrum export demo
print("\nSpectrum Export Formats:")
meta = SpectrumMetadata(
    title="Demo Spectrum",
    detector_id="HPGe-001",
    live_time=3600,
    real_time=3610,
    energy_coefficients=[0.0, 0.5, 0.0]
)

counts = [int(1000 * math.exp(-0.01 * ch) + 50) for ch in range(1024)]
exporter = SpectrumExporter(counts, meta)

with tempfile.TemporaryDirectory() as tmpdir:
    for fmt, method in [('SPE', 'to_spe'), ('CSV', 'to_csv'), 
                        ('IEC', 'to_iec'), ('MCNP', 'to_mcnp_sdef')]:
        path = Path(tmpdir) / f"test.{fmt.lower()}"
        getattr(exporter, method)(path)
        print(f"  {fmt}: {path.stat().st_size} bytes")


# =============================================================================
# EPIC U: STOPPING POWER
# =============================================================================
print("\n" + "=" * 70)
print("EPIC U: Stopping Power (U1.1-U1.6)")
print("=" * 70)

from fluxforge.physics.stopping_power import (
    Projectile, Material, STANDARD_MATERIALS,
    total_stopping_power, calculate_range, 
    calculate_energy_loss, calculate_straggling
)

# Stopping power table
print("\nStopping power in Aluminum (MeV/(mg/cm²)):")
print("  Energy     Proton    Alpha")
for E in [1, 5, 10, 20, 50]:
    S_p = total_stopping_power(E, Projectile.PROTON, STANDARD_MATERIALS['aluminum'])
    S_a = total_stopping_power(E, Projectile.ALPHA, STANDARD_MATERIALS['aluminum'])
    print(f"  {E:4} MeV   {S_p:.4f}    {S_a:.4f}")

# Range calculation
print("\nRange of 5 MeV alpha in various materials:")
for mat_name in ['aluminum', 'copper', 'gold', 'mylar']:
    mat = STANDARD_MATERIALS[mat_name]
    R = calculate_range(5.0, Projectile.ALPHA, mat)
    R_um = R / mat.density_g_cm3 / 0.1  # Convert to μm
    print(f"  {mat_name:10}: {R:.2f} mg/cm² = {R_um:.1f} μm")

# Energy loss example
print("\nProton energy through Al foils:")
E = 20.0
print(f"  Initial: {E:.2f} MeV")
for thickness in [10, 20, 50]:
    strag = calculate_straggling(E, Projectile.PROTON, STANDARD_MATERIALS['aluminum'], thickness)
    print(f"  After {thickness:2} mg/cm²: {strag.mean_energy_MeV:.2f}±{strag.sigma_MeV:.3f} MeV")


# =============================================================================
# EPIC W: ADVANCED UNFOLDING
# =============================================================================
print("\n" + "=" * 70)
print("EPIC W: Advanced Unfolding (W1.5, W1.6)")
print("=" * 70)

from fluxforge.solvers.advanced_unfolding import (
    CovarianceModel, build_covariance_matrix,
    mlem_with_covariance, adye_error_propagation
)
from fluxforge.core.linalg import matmul

# Demonstrate covariance models
measurements = [100, 200, 300, 400, 500]
print("\nCovariance models (diagonal variance for 5 measurements):")
for model in CovarianceModel:
    cov = build_covariance_matrix(measurements, model)
    diag = [cov[i][i] for i in range(5)]
    print(f"  {model.value:12}: {[f'{d:.1f}' for d in diag]}")

# Run MLEM with covariance
print("\nMLEM with Poisson covariance:")
np.random.seed(123)
n_flux, n_meas = 15, 8

true_flux = [1000 * math.exp(-0.2 * g) for g in range(n_flux)]
response = [[0.3 * math.exp(-0.5 * (g - 1.8*i)**2) for g in range(n_flux)]
            for i in range(n_meas)]
measurements = matmul(response, true_flux)
measurements = [m + np.random.normal(0, math.sqrt(m)) for m in measurements]
measurements = [max(m, 1) for m in measurements]

result = mlem_with_covariance(
    response, measurements,
    cov_model=CovarianceModel.POISSON,
    max_iters=300,
    convergence_mode="ddJ",
    compute_errors=True
)

print(f"  Converged: {result.converged} at iteration {result.iterations}")
print(f"  Chi²/dof: {result.chi_squared:.2f}")
if result.flux_uncertainty:
    avg_rel_unc = np.mean([u/f for f, u in zip(result.flux, result.flux_uncertainty) if f > 0])
    print(f"  Average relative uncertainty: {avg_rel_unc:.1%}")


# =============================================================================
# EPIC X: METASTABLE STATES
# =============================================================================
print("\n" + "=" * 70)
print("EPIC X: Metastable States (X1.8)")
print("=" * 70)

from fluxforge.physics.gamma_spectrum import (
    normalize_nuclide_name, is_metastable, get_ground_state,
    parse_nuclide, get_decay_lines
)

# Demonstrate nuclide name normalization
print("\nNuclide name normalization:")
test_names = ['tc99m', 'Tc-99m', 'TC_99M', 'In116m1', 'Co60', 'co-60']
for name in test_names:
    normalized = normalize_nuclide_name(name)
    meta = is_metastable(normalized)
    print(f"  {name:12} → {normalized:12} (metastable: {meta})")

# Parse nuclide
print("\nNuclide parsing:")
for nuclide in ['Tc-99m', 'In-116m1', 'Co-60']:
    elem, mass, meta = parse_nuclide(nuclide)
    print(f"  {nuclide}: element={elem}, mass={mass}, metastable={meta}")

# Decay lines for metastable
print("\nDecay lines for Tc-99m:")
lines = get_decay_lines('Tc-99m')
for line in lines[:3]:
    print(f"  {line.energy_keV:.2f} keV ({line.intensity:.1%} {line.emission_type})")


# =============================================================================
# EPIC Y: LOG-SMOOTHNESS & DDJ CONVERGENCE
# =============================================================================
print("\n" + "=" * 70)
print("EPIC Y: Log-Smoothness & ddJ Convergence (Y1.1, Y1.7)")
print("=" * 70)

from fluxforge.solvers.advanced_unfolding import (
    log_smoothness_penalty, compute_ddJ_convergence, mlem_with_covariance
)

# Compare smoothness of different spectra
print("\nLog-smoothness penalty comparison:")
smooth_spec = [1000 * math.exp(-0.1 * g) for g in range(20)]
noisy_spec = [1000 * (1 + 0.5 * ((-1)**g)) * math.exp(-0.1 * g) for g in range(20)]
flat_spec = [500] * 20

for name, spec in [('Smooth exponential', smooth_spec), 
                   ('Noisy oscillating', noisy_spec),
                   ('Flat', flat_spec)]:
    penalty = log_smoothness_penalty(spec)
    print(f"  {name:20}: {penalty:.2f}")

# ddJ convergence demonstration
print("\nddJ convergence criterion:")
J_converging = [1000, 500, 300, 200, 150, 120, 105, 98, 94, 91, 89, 88, 87.5, 87.2]
J_oscillating = [100, 95, 98, 92, 97, 93, 96, 94, 95, 93, 95, 94, 94.5, 94.3]

for name, J_hist in [('Converging', J_converging), ('Oscillating', J_oscillating)]:
    dJ, ddJ = compute_ddJ_convergence(J_hist)
    print(f"  {name:12}: dJ={dJ:.2f}, ddJ={ddJ:.4f}")

# MLEM comparison with/without smoothness
print("\nMLEM with smoothness regularization:")
np.random.seed(42)
n_flux, n_meas = 12, 6
true_flux = [500 * math.exp(-0.15 * g) for g in range(n_flux)]
response = [[0.4 * math.exp(-0.3 * (g - 2*i)**2) for g in range(n_flux)]
            for i in range(n_meas)]
measurements = matmul(response, true_flux)
measurements = [max(m + np.random.normal(0, math.sqrt(max(m, 1))), 1) for m in measurements]

for weight, label in [(0.0, 'None'), (0.01, 'Low'), (0.1, 'High')]:
    result = mlem_with_covariance(
        response, measurements,
        smoothness_weight=weight,
        max_iters=200
    )
    rmse = math.sqrt(sum((f-t)**2 for f, t in zip(result.flux, true_flux)) / n_flux)
    smooth = log_smoothness_penalty(result.flux)
    print(f"  Smoothness {label:5}: RMSE={rmse:.1f}, penalty={smooth:.1f}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: All Epic Features Demonstrated!")
print("=" * 70)
print("""
✅ Epic R: Auto-calibration using known isotope lines
✅ Epic S: Complete GLS workflow with JSON database
✅ Epic T: Stacked target energy degradation & multi-format export
✅ Epic U: Stopping power, range, energy loss, straggling
✅ Epic W: Multinomial/Poisson covariance, Adye error propagation
✅ Epic X: Metastable state handling (Tc-99m, In-116m, etc.)
✅ Epic Y: Log-smoothness regularization, ddJ convergence

All features are fully implemented and ready for use!
""")
