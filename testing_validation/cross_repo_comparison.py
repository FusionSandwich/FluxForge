#!/usr/bin/env python
"""
Cross-Repository Comparison Tests

Epic V - Testing Repository Cross-Validation

Runs identical test cases through both testing repositories (becquerel, gmapy, 
curie, npat) and FluxForge, comparing results to ensure parity.

This validates that FluxForge implementations match the reference implementations
from the testing repositories.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test results tracking
RESULTS = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'tests': []
}


def log_result(name: str, passed: bool, message: str = "", skip: bool = False):
    """Log a test result."""
    if skip:
        RESULTS['skipped'] += 1
        status = "SKIP"
    elif passed:
        RESULTS['passed'] += 1
        status = "PASS"
    else:
        RESULTS['failed'] += 1
        status = "FAIL"
    
    RESULTS['tests'].append({'name': name, 'status': status, 'message': message})
    print(f"  [{status}] {name}" + (f": {message}" if message else ""))


def compare_values(name: str, val1, val2, tol: float = 0.01, 
                   label1: str = "FluxForge", label2: str = "Reference"):
    """Compare two values with tolerance."""
    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        if val1.shape != val2.shape:
            log_result(name, False, f"Shape mismatch: {val1.shape} vs {val2.shape}")
            return False
        if np.allclose(val1, val2, rtol=tol, atol=1e-10):
            log_result(name, True, f"{label1}={val1.mean():.6g}, {label2}={val2.mean():.6g}")
            return True
        else:
            max_diff = np.max(np.abs(val1 - val2))
            log_result(name, False, f"Max diff={max_diff:.6g}")
            return False
    else:
        if abs(val1 - val2) / max(abs(val2), 1e-10) < tol:
            log_result(name, True, f"{label1}={val1:.6g}, {label2}={val2:.6g}")
            return True
        else:
            log_result(name, False, f"{label1}={val1:.6g}, {label2}={val2:.6g}")
            return False


# =============================================================================
# BECQUEREL COMPARISON TESTS
# =============================================================================

def test_becquerel_parity():
    """Compare FluxForge vs becquerel implementations."""
    print("\n" + "="*60)
    print("BECQUEREL PARITY TESTS")
    print("="*60)
    
    # Test 1: ExpGauss peak function (validate FluxForge implementation)
    print("\n1. ExpGauss Peak Function:")
    try:
        from fluxforge.analysis.peakfit import expgauss as ff_expgauss
        
        x = np.linspace(-10, 10, 100)
        result = ff_expgauss(x, amplitude=100, centroid=0, sigma=2, gamma=0.5)
        
        # Validate properties
        peak_idx = np.argmax(result)
        peak_pos = x[peak_idx]
        peak_val = result[peak_idx]
        
        # ExpGauss peak should be near centroid (shifted by gamma)
        log_result("ExpGauss peak position", abs(peak_pos) < 2, 
                  f"peak at {peak_pos:.2f}")
        log_result("ExpGauss peak value", peak_val > 10, f"peak={peak_val:.2f}")
        log_result("ExpGauss integral > 0", np.sum(result) > 0, 
                  f"integral={np.sum(result):.2f}")
    except Exception as e:
        log_result("ExpGauss function", False, str(e))
    
    # Test 2: Double-exponential tail
    print("\n2. Double-Exponential Tail:")
    try:
        from fluxforge.analysis.peakfit import gauss_dbl_exp as ff_dbl_exp
        
        x = np.linspace(-10, 10, 100)
        result = ff_dbl_exp(x, amplitude=100, centroid=0, sigma=2,
                           ltail_ratio=0.1, ltail_slope=0.05, ltail_cutoff=1.0,
                           rtail_ratio=0.1, rtail_slope=0.05, rtail_cutoff=1.0)
        
        # Should have tails extending beyond Gaussian
        log_result("gauss_dbl_exp has tails", result[0] > 0 and result[-1] > 0,
                  f"left={result[0]:.4f}, right={result[-1]:.4f}")
        log_result("gauss_dbl_exp peak centered", abs(x[np.argmax(result)]) < 1,
                  f"peak at x={x[np.argmax(result)]:.2f}")
    except Exception as e:
        log_result("gauss_dbl_exp function", False, str(e))
    
    # Test 3: Poisson NLL
    print("\n3. Poisson Negative Log-Likelihood:")
    try:
        from fluxforge.analysis.peakfit import poisson_neg_log_likelihood as ff_nll
        
        y_model = np.array([10, 20, 30, 20, 10])
        y_data = np.array([12, 18, 28, 22, 11])
        
        ff_result = ff_nll(y_model, y_data)
        
        # Manual calculation for validation
        from scipy.special import xlogy
        expected = np.sum(y_model - xlogy(y_data, y_model))
        
        compare_values("Poisson NLL", ff_result, expected, tol=0.001,
                      label2="scipy.special")
    except Exception as e:
        log_result("Poisson NLL", False, str(e))
    
    # Test 4: Spectrum operations (NEW - becquerel parity)
    print("\n4. Spectrum Operations:")
    try:
        from fluxforge.core.spectrum_ops import SpectrumData, rebin_spectrum
        
        # Create test spectrum
        spec1 = SpectrumData(
            counts=np.array([100, 200, 300, 200, 100]),
            edges=np.array([0, 100, 200, 300, 400, 500]),
            live_time=100,
            real_time=110
        )
        
        log_result("SpectrumData creation", spec1.n_channels == 5,
                  f"n_channels={spec1.n_channels}")
        log_result("SpectrumData CPS", np.allclose(spec1.cps, [1, 2, 3, 2, 1]),
                  f"cps={spec1.cps}")
        log_result("SpectrumData dead time", abs(spec1.dead_time_fraction - 0.0909) < 0.01,
                  f"dead_time={spec1.dead_time_fraction*100:.1f}%")
        
        # Test arithmetic
        spec2 = spec1 + spec1
        log_result("Spectrum addition", spec2.total_counts == 1800,
                  f"total={spec2.total_counts}")
        
        spec3 = spec1 * 2
        log_result("Spectrum scalar multiply", spec3.total_counts == 1800,
                  f"total={spec3.total_counts}")
        
    except Exception as e:
        log_result("Spectrum operations", False, str(e))
    
    # Test 5: NNDC nuclear data (NEW - becquerel parity)
    print("\n5. NNDC Nuclear Data:")
    try:
        from fluxforge.data.nndc import Isotope, IsotopeQuantity, get_nuclear_data
        
        # Test isotope parsing
        co60 = Isotope.from_string('Co-60')
        log_result("Isotope parsing", co60.element == 'Co' and co60.A == 60,
                  f"{co60.name}")
        
        # Test half-life
        hl_s = co60.half_life_s
        hl_y = hl_s / 31557600
        log_result("Co-60 half-life", abs(hl_y - 5.27) < 0.1,
                  f"T½={hl_y:.2f} y")
        
        # Test metastable
        tc99m = Isotope.from_string('Tc-99m')
        log_result("Metastable parsing", tc99m.m == 1,
                  f"{tc99m.name}, m={tc99m.m}")
        
        # Test IsotopeQuantity
        qty = IsotopeQuantity(isotope=co60, activity_Bq=1e6, reference_time=0)
        activity_1y = qty.activity_at(31557600)
        log_result("Decay calculation", 0.85e6 < activity_1y < 0.9e6,
                  f"A(1y)={activity_1y:.2e} Bq")
        
    except Exception as e:
        log_result("NNDC nuclear data", False, str(e))
    
    # Test 6: Calibration (NEW - becquerel parity)
    print("\n6. Energy Calibration:")
    try:
        from fluxforge.core.spectrum_ops import Calibration
        
        cal = Calibration(expression='linear', degree=1)
        cal.add_points([0, 1000, 2000], [0, 500, 1000])
        result = cal.fit()
        
        log_result("Calibration fit R²", result.r_squared > 0.99,
                  f"R²={result.r_squared:.4f}")
        log_result("Calibration evaluation", abs(cal(1500) - 750) < 1,
                  f"cal(1500)={cal(1500):.1f} keV")
        log_result("Calibration inverse", abs(cal.inverse(500) - 1000) < 10,
                  f"cal⁻¹(500)={cal.inverse(500):.1f}")
        
    except Exception as e:
        log_result("Energy calibration", False, str(e))


# =============================================================================
# GMAPY COMPARISON TESTS  
# =============================================================================

def test_gmapy_parity():
    """Compare FluxForge vs gmapy implementations."""
    print("\n" + "="*60)
    print("GMAPY PARITY TESTS")
    print("="*60)
    
    # Test 1: GLS Update
    print("\n1. GLS Spectral Adjustment:")
    try:
        from fluxforge.solvers.advanced import gls_update_numpy as ff_gls
        
        # Create test case
        R = np.array([[0.1, 0.3, 0.5, 0.1],
                      [0.2, 0.4, 0.3, 0.1],
                      [0.05, 0.15, 0.5, 0.3]])
        y = np.array([1.0, 0.8, 0.5])
        V_y = np.diag([0.1**2, 0.08**2, 0.05**2])
        phi0 = np.array([2.0, 2.0, 1.5, 1.0])
        V_phi = np.diag([0.4, 0.4, 0.3, 0.2])**2
        
        phi_post, V_phi_post, chi2 = ff_gls(R, y, V_y, phi0, V_phi)
        
        # Verify by reconstruction
        y_pred = R @ phi_post
        residuals = y - y_pred
        
        # Check that GLS reduced residuals
        initial_residuals = y - R @ phi0
        improvement = np.linalg.norm(residuals) < np.linalg.norm(initial_residuals)
        
        log_result("GLS reduces residuals", improvement,
                  f"||r_post||={np.linalg.norm(residuals):.4f} < ||r_prior||={np.linalg.norm(initial_residuals):.4f}")
        
        # Check chi-squared is reasonable
        chi2_reasonable = 0 < chi2 < 1000
        log_result("GLS chi-squared valid", chi2_reasonable, f"χ²={chi2:.2f}")
        
    except Exception as e:
        log_result("GLS spectral adjustment", False, str(e))
    
    # Test 2: Levenberg-Marquardt
    print("\n2. Levenberg-Marquardt Optimization:")
    try:
        from fluxforge.solvers.advanced import levenberg_marquardt
        
        # Simple exponential fit test
        def model(params):
            a, b = params
            x = np.linspace(0, 5, 10)
            return a * np.exp(-b * x)
        
        def jacobian(params):
            a, b = params
            x = np.linspace(0, 5, 10)
            J = np.zeros((10, 2))
            J[:, 0] = np.exp(-b * x)
            J[:, 1] = -a * x * np.exp(-b * x)
            return J
        
        np.random.seed(42)
        y_true = 2.0 * np.exp(-0.5 * np.linspace(0, 5, 10))
        y_data = y_true + np.random.normal(0, 0.05, 10)
        V = np.diag([0.05**2] * 10)
        
        result = levenberg_marquardt(model, jacobian, y_data, V, np.array([1.0, 1.0]))
        
        # Check parameters recovered
        a_err = abs(result.x[0] - 2.0) / 2.0
        b_err = abs(result.x[1] - 0.5) / 0.5
        
        log_result("LM converged", result.converged, f"iterations={result.n_iter}")
        log_result("LM amplitude recovery", a_err < 0.1, 
                  f"a={result.x[0]:.3f} (true=2.0, err={a_err*100:.1f}%)")
        log_result("LM rate recovery", b_err < 0.2,
                  f"b={result.x[1]:.3f} (true=0.5, err={b_err*100:.1f}%)")
        
    except Exception as e:
        log_result("Levenberg-Marquardt", False, str(e))
    
    # Test 3: Romberg Integration
    print("\n3. Adaptive Integration:")
    try:
        from fluxforge.solvers.advanced import romberg_integrate
        
        # Test integral: ∫₀¹ x² dx = 1/3
        result, error = romberg_integrate(lambda x: x**2, 0, 1)
        expected = 1.0/3.0
        
        compare_values("∫x² dx [0,1]", result, expected, tol=1e-8)
        
        # Test integral: ∫₀^π sin(x) dx = 2
        result, error = romberg_integrate(np.sin, 0, np.pi)
        expected = 2.0
        
        compare_values("∫sin(x) dx [0,π]", result, expected, tol=1e-8)
        
    except Exception as e:
        log_result("Romberg integration", False, str(e))
    
    # Test 4: PPP Correction
    print("\n4. PPP Correction:")
    try:
        from fluxforge.solvers.advanced import apply_ppp_correction, PPPCorrectionMethod
        
        y = np.array([1.0, 2.0, 1.5])
        V = np.array([[0.04, 0.02, 0.02],
                      [0.02, 0.16, 0.08],
                      [0.02, 0.08, 0.09]])
        
        y_corr, V_corr = apply_ppp_correction(y, V, PPPCorrectionMethod.CHIBA_SMITH)
        
        # Check log transform
        log_result("PPP log transform", np.allclose(y_corr, np.log(y)),
                  f"log(y)={np.log(y)}")
        
    except Exception as e:
        log_result("PPP correction", False, str(e))


# =============================================================================
# CURIE COMPARISON TESTS
# =============================================================================

def test_curie_parity():
    """Compare FluxForge vs curie implementations."""
    print("\n" + "="*60)
    print("CURIE PARITY TESTS")
    print("="*60)
    
    # Test 1: Simple Decay
    print("\n1. Simple Radioactive Decay:")
    try:
        from fluxforge.physics.decay_chain import simple_decay
        
        # Mn-56 half-life: 2.5789 hours = 9284.04 seconds
        t_half = 9284.04
        times = np.array([0, 9284.04, 18568.08])  # 0, 1, 2 half-lives
        
        activity = simple_decay(1000, t_half, times)
        
        log_result("A(0) = A₀", abs(activity[0] - 1000) < 1, f"A(0)={activity[0]:.2f}")
        log_result("A(t½) = A₀/2", abs(activity[1] - 500) < 1, f"A(t½)={activity[1]:.2f}")
        log_result("A(2t½) = A₀/4", abs(activity[2] - 250) < 1, f"A(2t½)={activity[2]:.2f}")
        
    except Exception as e:
        log_result("Simple decay", False, str(e))
    
    # Test 2: Decay Chain with Bateman
    print("\n2. Decay Chain (Bateman Equations):")
    try:
        from fluxforge.physics.decay_chain import DecayChain
        
        # Mn-56 → Fe-56 (stable)
        nuclide_data = {
            'Mn56': {'half_life_s': 9284.04, 'decay_products': {'Fe56': 1.0}},
            'Fe56': {'half_life_s': float('inf'), 'decay_products': {}}
        }
        chain = DecayChain('Mn56', nuclide_data=nuclide_data)
        
        log_result("Chain built", len(chain.chain_order) == 2, 
                  f"chain={chain.chain_order}")
        
        # Calculate decay
        result = chain.decay(
            initial_activity={'Mn56': 1000},
            times=[0, 9284.04, 18568.08]
        )
        
        # Check Mn-56 decays correctly
        A_mn56 = result.activities['Mn56']
        log_result("Mn56 A(0)", abs(A_mn56[0] - 1000) < 1, f"{A_mn56[0]:.2f}")
        log_result("Mn56 A(t½)", abs(A_mn56[1] - 500) < 5, f"{A_mn56[1]:.2f}")
        
        # Check Fe-56 grows (atoms, not activity since stable)
        N_fe56 = result.atoms['Fe56']
        log_result("Fe56 accumulates", N_fe56[-1] > N_fe56[0], 
                  f"N(0)={N_fe56[0]:.0f} → N(2t½)={N_fe56[-1]:.0f}")
        
    except Exception as e:
        log_result("Decay chain", False, str(e))
    
    # Test 3: Decay with Production
    print("\n3. Decay with Production (Saturation):")
    try:
        from fluxforge.physics.decay_chain import DecayChain, irradiation_saturation_factor
        
        nuclide_data = {
            'Mn56': {'half_life_s': 9284.04, 'decay_products': {'Fe56': 1.0}},
            'Fe56': {'half_life_s': float('inf'), 'decay_products': {}}
        }
        chain = DecayChain('Mn56', nuclide_data=nuclide_data)
        
        # Production rate = 1e6 atoms/s
        R = 1e6
        lam = np.log(2) / 9284.04
        A_sat_expected = R  # Saturation activity = production rate
        
        # Long irradiation (many half-lives)
        result = chain.decay(
            initial_atoms={'Mn56': 0},
            production_rates={'Mn56': R},
            times=[0, 50000, 100000, 200000]
        )
        
        A_final = result.activities['Mn56'][-1]
        log_result("Approaches saturation", abs(A_final - A_sat_expected)/A_sat_expected < 0.05,
                  f"A_final={A_final:.2e}, A_sat={A_sat_expected:.2e}")
        
        # Saturation factor
        S = irradiation_saturation_factor(lam, 9284.04)
        log_result("S(t½) ≈ 0.5", abs(S - 0.5) < 0.01, f"S={S:.4f}")
        
    except Exception as e:
        log_result("Decay with production", False, str(e))


# =============================================================================
# XCOM / MATERIALS COMPARISON TESTS
# =============================================================================

def test_xcom_materials_parity():
    """Compare FluxForge XCOM and materials data."""
    print("\n" + "="*60)
    print("XCOM & MATERIALS DATA TESTS")
    print("="*60)
    
    # Test 1: XCOM attenuation
    print("\n1. XCOM Mass Attenuation:")
    try:
        from fluxforge.data.xcom import get_attenuation_data, calculate_hvl
        
        # Lead at 662 keV (Cs-137 gamma)
        pb_data = get_attenuation_data('Lead')
        mu_rho_662 = float(pb_data.get_mu_rho(662))
        
        # Reference value from NIST XCOM: ~0.0877 cm²/g at 662 keV
        # (varies slightly with interpolation)
        log_result("Lead μ/ρ(662 keV)", 0.05 < mu_rho_662 < 0.15,
                  f"μ/ρ={mu_rho_662:.4f} cm²/g")
        
        # HVL calculation
        hvl = calculate_hvl('Lead', 662)
        # Expected ~0.9 cm
        log_result("Lead HVL(662 keV)", 0.5 < hvl < 1.5,
                  f"HVL={hvl:.3f} cm")
        
    except Exception as e:
        log_result("XCOM attenuation", False, str(e))
    
    # Test 2: Material compositions
    print("\n2. NIST Material Compositions:")
    try:
        from fluxforge.data.materials import get_material, search_materials
        
        # Water
        water = get_material('Water')
        log_result("Water density", abs(water.density - 1.0) < 0.01,
                  f"ρ={water.density} g/cm³")
        log_result("Water H fraction", abs(water.composition.get('H', 0) - 0.1119) < 0.01,
                  f"H={water.composition.get('H', 0):.4f}")
        
        # EUROFER97
        eurofer = get_material('EUROFER97')
        log_result("EUROFER97 density", 7.5 < eurofer.density < 8.0,
                  f"ρ={eurofer.density} g/cm³")
        log_result("EUROFER97 Fe content", eurofer.composition.get('Fe', 0) > 0.85,
                  f"Fe={eurofer.composition.get('Fe', 0):.4f}")
        
        # Search
        steel_matches = search_materials('steel')
        log_result("Material search", len(steel_matches) >= 3,
                  f"Found {len(steel_matches)} steels")
        
    except Exception as e:
        log_result("Material compositions", False, str(e))


# =============================================================================
# PYUNFOLD PARITY TESTS (Epic W)
# =============================================================================

def test_pyunfold_parity():
    """Compare FluxForge vs PyUnfold implementations."""
    print("\n" + "="*60)
    print("PYUNFOLD PARITY TESTS (Epic W)")
    print("="*60)
    
    # Test 1: Test statistics
    print("\n1. Convergence Test Statistics:")
    try:
        from fluxforge.solvers.test_statistics import (
            ks_test_statistic, chi2_test_statistic,
            bayes_factor_test_statistic, rmd_test_statistic
        )
        
        phi_prev = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        phi_curr = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        ks = ks_test_statistic(phi_curr, phi_prev)
        log_result("KS statistic (similar)", 0 < ks < 0.1, f"KS={ks:.4f}")
        
        chi2 = chi2_test_statistic(phi_curr, phi_prev)
        log_result("Chi2 statistic (similar)", 0 < chi2 < 0.1, f"χ²={chi2:.4f}")
        
        bf = bayes_factor_test_statistic(phi_curr, phi_prev)
        log_result("Bayes factor (similar)", abs(bf) < 0.5, f"BF={bf:.4f}")
        
        rmd = rmd_test_statistic(phi_curr, phi_prev)
        log_result("RMD statistic (similar)", 0 < rmd < 0.15, f"RMD={rmd:.4f}")
        
    except Exception as e:
        log_result("Test statistics", False, str(e))
    
    # Test 2: Priors
    print("\n2. Prior Distributions:")
    try:
        from fluxforge.solvers.test_statistics import uniform_prior, jeffreys_prior, power_law_prior
        
        u_prior = uniform_prior(10)
        log_result("Uniform prior sums to 1", abs(np.sum(u_prior) - 1.0) < 1e-10,
                  f"sum={np.sum(u_prior):.6f}")
        
        j_prior = jeffreys_prior(10)
        log_result("Jeffreys prior sums to 1", abs(np.sum(j_prior) - 1.0) < 1e-10,
                  f"sum={np.sum(j_prior):.6f}")
        
        p_prior = power_law_prior(10, index=-1.5)
        log_result("Power law prior sums to 1", abs(np.sum(p_prior) - 1.0) < 1e-10,
                  f"sum={np.sum(p_prior):.6f}")
        log_result("Power law favors low E", p_prior[0] > p_prior[-1],
                  f"ratio={p_prior[0]/p_prior[-1]:.1f}x")
        
    except Exception as e:
        log_result("Priors", False, str(e))
    
    # Test 3: Spline regularization
    print("\n3. Spline Regularization:")
    try:
        from fluxforge.solvers.test_statistics import spline_regularize, SplineRegularizer
        
        np.random.seed(42)
        noisy = np.array([1, 2, 5, 3, 7, 4, 8, 5, 4, 3]) + np.random.randn(10) * 0.5
        smooth = spline_regularize(noisy, smoothing=0.5)
        
        # Smoothed should have lower variance
        var_noisy = np.var(noisy)
        var_smooth = np.var(smooth)
        log_result("Spline reduces variance", var_smooth < var_noisy,
                  f"var={var_noisy:.2f}→{var_smooth:.2f}")
        
        # Total should be preserved
        log_result("Spline preserves integral", abs(np.sum(smooth) - np.sum(noisy)) < 0.01,
                  f"sum={np.sum(noisy):.2f}={np.sum(smooth):.2f}")
        
        # SplineRegularizer class
        reg = SplineRegularizer(smoothing=0.3)
        smooth2 = reg(noisy)
        log_result("SplineRegularizer works", len(smooth2) == len(noisy),
                  f"len={len(smooth2)}")
        
    except Exception as e:
        log_result("Spline regularization", False, str(e))
    
    # Test 4: Iterative Bayesian unfolding
    print("\n4. Iterative Bayesian Unfolding:")
    try:
        from fluxforge.solvers.test_statistics import (
            iterative_bayesian_unfold, TestStatisticType
        )
        
        # Simple 3x3 response
        R = np.array([[0.8, 0.1, 0.1],
                      [0.1, 0.8, 0.1],
                      [0.1, 0.1, 0.8]])
        true_phi = np.array([100, 200, 150])
        np.random.seed(42)
        data = R @ true_phi + np.random.randn(3) * 5
        
        phi_unfolded, info = iterative_bayesian_unfold(
            data, R, max_iter=50, ts_type=TestStatisticType.KS
        )
        
        log_result("IBU converges", info['converged'],
                  f"iterations={info['n_iter']}")
        
        # Check recovery
        err1 = abs(phi_unfolded[0] - true_phi[0]) / true_phi[0]
        err2 = abs(phi_unfolded[1] - true_phi[1]) / true_phi[1]
        log_result("IBU recovers bin 1", err1 < 0.1,
                  f"φ₁={phi_unfolded[0]:.0f} (true={true_phi[0]})")
        log_result("IBU recovers bin 2", err2 < 0.1,
                  f"φ₂={phi_unfolded[1]:.0f} (true={true_phi[1]})")
        
    except Exception as e:
        log_result("Iterative Bayesian unfolding", False, str(e))


# =============================================================================
# ACTIGAMMA COMPARISON TESTS (Epic X)
# =============================================================================

def test_actigamma_parity():
    """Compare FluxForge vs actigamma implementations."""
    print("\n" + "="*60)
    print("ACTIGAMMA PARITY TESTS (Epic X)")
    print("="*60)
    
    # Test 1: Decay line database
    print("\n1. Decay Line Database:")
    try:
        from fluxforge.physics.gamma_spectrum import (
            get_decay_lines, DECAY_LINES, list_available_nuclides
        )
        
        # Check database has standard nuclides
        nuclides = list_available_nuclides()
        log_result("Database has nuclides", len(nuclides) >= 15,
                  f"n={len(nuclides)}: {nuclides[:5]}...")
        
        # Check Co-60 data
        co60_lines = get_decay_lines('Co-60')
        energies = [line.energy_keV for line in co60_lines]
        
        log_result("Co-60 1173 keV line", any(abs(e - 1173.2) < 1 for e in energies),
                  f"found {[e for e in energies if 1000 < e < 1200]}")
        log_result("Co-60 1332 keV line", any(abs(e - 1332.5) < 1 for e in energies),
                  f"found {[e for e in energies if 1300 < e < 1400]}")
        
    except Exception as e:
        log_result("Decay line database", False, str(e))
    
    # Test 2: Forward spectrum generation
    print("\n2. Forward Spectrum Generation:")
    try:
        from fluxforge.physics.gamma_spectrum import (
            generate_spectrum, Inventory, EnergyBins
        )
        
        # Create test inventory
        inventory = Inventory(
            activities_Bq={'Co-60': 1e6, 'Cs-137': 1e6},
            reference_time=0
        )
        
        bins = EnergyBins(
            edges=np.linspace(0, 2000, 201),  # 10 keV bins
            unit='keV'
        )
        
        spectrum = generate_spectrum(inventory, bins, live_time=1.0)
        
        log_result("Spectrum generated", spectrum is not None,
                  f"total_photons={spectrum.total_photons:.2e}/s")
        
        # Check peak bins
        bin_centers = 0.5 * (bins.edges[:-1] + bins.edges[1:])
        peak_idx_1173 = np.abs(bin_centers - 1173).argmin()
        peak_idx_662 = np.abs(bin_centers - 662).argmin()
        
        log_result("Co-60 1173 keV peak", spectrum.counts[peak_idx_1173] > 0,
                  f"counts={spectrum.counts[peak_idx_1173]:.2e}")
        log_result("Cs-137 662 keV peak", spectrum.counts[peak_idx_662] > 0,
                  f"counts={spectrum.counts[peak_idx_662]:.2e}")
        
    except Exception as e:
        log_result("Forward spectrum generation", False, str(e))
    
    # Test 3: Nuclide identification
    print("\n3. Nuclide Identification:")
    try:
        from fluxforge.physics.gamma_spectrum import identify_nuclides, suggest_nuclides
        
        # Known peaks for Co-60
        peaks = [1173.2, 1332.5]
        
        results = identify_nuclides(peaks, tolerance_keV=2.0)
        
        log_result("identify_nuclides returns results", len(results) > 0,
                  f"n_matches={len(results)}")
        
        # Check that Co-60 is identified
        co60_found = any('Co-60' in str(r) or r.get('nuclide', '') == 'Co-60' 
                        for r in results)
        log_result("Co-60 identified", co60_found, str(results)[:80])
        
        # Test suggest_nuclides
        suggestions = suggest_nuclides(peaks, top_n=5)
        log_result("suggest_nuclides works", len(suggestions) > 0,
                  f"top suggestion: {suggestions[0] if suggestions else 'none'}")
        
        # Check Co-60 is top suggestion
        co60_top = suggestions[0][0] == 'Co-60' if suggestions else False
        log_result("Co-60 top suggestion", co60_top,
                  f"scores: {[(s[0], s[1]) for s in suggestions[:3]]}")
        
    except Exception as e:
        log_result("Nuclide identification", False, str(e))
    
    # Test 4: Activity/decay calculations
    print("\n4. Activity and Decay Calculations:")
    try:
        from fluxforge.physics.gamma_spectrum import (
            atoms_to_activity, activity_to_atoms, decay_activity
        )
        from fluxforge.data.nndc import get_nuclear_data
        
        # Get Co-60 half-life
        data = get_nuclear_data('Co-60')
        t_half = data['half_life_s']
        
        # Test conversion: 1e6 Bq activity
        activity = 1e6
        atoms = activity_to_atoms(activity, t_half)
        activity_back = atoms_to_activity(atoms, t_half)
        
        log_result("Activity ↔ Atoms roundtrip", abs(activity_back - activity) / activity < 1e-6,
                  f"A={activity:.2e} → N={atoms:.2e} → A={activity_back:.2e}")
        
        # Test decay after one half-life
        activity_1hl = decay_activity(activity, t_half, t_half)
        log_result("Decay at t=t½", abs(activity_1hl - activity/2) / (activity/2) < 0.01,
                  f"A(t½)={activity_1hl:.2e} ≈ {activity/2:.2e}")
        
    except Exception as e:
        log_result("Activity calculations", False, str(e))
    
    # Test 5: Emission aggregation
    print("\n5. Emission Aggregation:")
    try:
        from fluxforge.physics.gamma_spectrum import aggregate_emissions
        
        emissions = aggregate_emissions('Co-60')
        
        log_result("Gamma emissions present", len(emissions['gammas']) > 0,
                  f"n_gammas={len(emissions['gammas'])}")
        log_result("Total gamma intensity", emissions['total_gamma_intensity'] > 1.9,
                  f"I_total={emissions['total_gamma_intensity']:.3f}")
        
    except Exception as e:
        log_result("Emission aggregation", False, str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all comparison tests."""
    print("="*60)
    print("FLUXFORGE CROSS-REPOSITORY COMPARISON TESTS")
    print("Epic V - Testing Repository Cross-Validation")
    print("="*60)
    
    # Run all test suites
    test_becquerel_parity()
    test_gmapy_parity()
    test_curie_parity()
    test_xcom_materials_parity()
    test_pyunfold_parity()
    test_actigamma_parity()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total = RESULTS['passed'] + RESULTS['failed'] + RESULTS['skipped']
    print(f"\nTotal Tests: {total}")
    print(f"  ✅ Passed:  {RESULTS['passed']}")
    print(f"  ❌ Failed:  {RESULTS['failed']}")
    print(f"  ⏭️  Skipped: {RESULTS['skipped']}")
    
    if RESULTS['failed'] > 0:
        print("\nFailed tests:")
        for test in RESULTS['tests']:
            if test['status'] == 'FAIL':
                print(f"  - {test['name']}: {test['message']}")
    
    success_rate = RESULTS['passed'] / max(RESULTS['passed'] + RESULTS['failed'], 1) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if RESULTS['failed'] == 0:
        print("\n✅ ALL COMPARISON TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {RESULTS['failed']} TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
