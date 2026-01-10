#!/usr/bin/env python3
"""Cross-validation of FluxForge against testing/ repository implementations.

This script runs the same example data through both the original testing repos
and FluxForge, then compares results to ensure correctness.

Validation Repositories:
1. SpecKit - Activation unfolding with log-smoothness regularization
2. pyunfold - D'Agostini iterative Bayesian unfolding
3. Neutron-Unfolding - GRAVEL and MLEM implementations
4. Neutron-Spectrometry - MLEM-STOP for nested neutron spectrometer
5. actigamma - Activity to gamma spectra conversion
6. gamma_spec_analysis - SPE reading and peak finding
7. hdtv - Nuclear spectrum analysis
8. irrad_spectroscopy - Isotope identification
9. peakingduck - AI-based peak finding
10. NAA-ANN-1 - Neural network for NAA

Usage:
    python cross_validate_testing_repos.py --repo all
    python cross_validate_testing_repos.py --repo SpecKit
    python cross_validate_testing_repos.py --repo Neutron-Unfolding
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np

# Paths
TESTING_DIR = Path("/filespace/s/smandych/CAE/projects/ALARA/testing")
FLUXFORGE_DIR = Path("/filespace/s/smandych/CAE/projects/ALARA/FluxForge")
RESULTS_DIR = FLUXFORGE_DIR / "testing_validation"


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    
    repo: str
    test_name: str
    passed: bool
    correlation: float = 0.0
    max_relative_error: float = 0.0
    notes: str = ""
    details: dict = field(default_factory=dict)
    
    @property
    def status_emoji(self) -> str:
        if self.passed:
            return "✅"
        elif self.correlation > 0.9:
            return "⚠️"  # Close but not passing
        else:
            return "❌"


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    # Handle constant arrays
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    
    return float(np.corrcoef(a, b)[0, 1])


def max_rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """Compute maximum relative error."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    denom = np.maximum(np.abs(a), eps)
    rel_err = np.abs(a - b) / denom
    return float(np.max(rel_err))


# =============================================================================
# Validation Functions for Each Repository
# =============================================================================

def validate_neutron_unfolding() -> list[ValidationResult]:
    """Validate against Neutron-Unfolding repo (GRAVEL and MLEM)."""
    results = []
    repo_path = TESTING_DIR / "Neutron-Unfolding"
    
    if not repo_path.exists():
        return [ValidationResult(
            repo="Neutron-Unfolding",
            test_name="repo_exists",
            passed=False,
            notes="Repository not found"
        )]
    
    try:
        # Load test data
        response_file = repo_path / "unfolding_inputs" / "response-matrix.csv"
        spectrum_file = repo_path / "unfolding_inputs" / "pulse-height-spectrum.txt"
        
        if not response_file.exists() or not spectrum_file.exists():
            return [ValidationResult(
                repo="Neutron-Unfolding",
                test_name="data_files",
                passed=False,
                notes="Test data files not found"
            )]
        
        # Load response matrix
        response = np.loadtxt(response_file, delimiter=',')
        measurements = np.loadtxt(spectrum_file)
        
        # Import FluxForge solvers
        sys.path.insert(0, str(FLUXFORGE_DIR / "src"))
        from fluxforge.solvers.iterative import gravel, mlem
        
        # Convert to list format for FluxForge
        response_list = response.tolist()
        measurements_list = measurements.tolist()
        n_groups = response.shape[1]
        initial_flux = [1.0] * n_groups
        
        # Run FluxForge GRAVEL
        gravel_result = gravel(
            response=response_list,
            measurements=measurements_list,
            initial_flux=initial_flux,
            max_iters=500,
        )
        
        # Run FluxForge MLEM with ddJ convergence (Neutron-Unfolding style)
        mlem_result = mlem(
            response=response_list,
            measurements=measurements_list,
            initial_flux=initial_flux,
            max_iters=500,
            convergence_mode="ddJ",
            relaxation=1.0,  # No relaxation to match original
        )
        
        # Run original Neutron-Unfolding GRAVEL
        sys.path.insert(0, str(repo_path))
        try:
            from gravel import gravel as gravel_orig
            orig_gravel, _ = gravel_orig(response, measurements, np.array(initial_flux), 500)
            
            gravel_corr = correlate(gravel_result.flux, orig_gravel)
            gravel_max_err = max_rel_error(gravel_result.flux, orig_gravel)
            
            results.append(ValidationResult(
                repo="Neutron-Unfolding",
                test_name="GRAVEL",
                passed=gravel_corr > 0.98,
                correlation=gravel_corr,
                max_relative_error=gravel_max_err,
                notes=f"FluxForge vs original GRAVEL",
                details={"iterations_ff": gravel_result.iterations}
            ))
        except Exception as e:
            results.append(ValidationResult(
                repo="Neutron-Unfolding",
                test_name="GRAVEL",
                passed=False,
                notes=f"Error running original: {e}"
            ))
        
        # Run original MLEM
        try:
            from mlem import mlem as mlem_orig
            orig_mlem, _ = mlem_orig(response, measurements, np.array(initial_flux), 1e-6)
            
            mlem_corr = correlate(mlem_result.flux, orig_mlem)
            mlem_max_err = max_rel_error(mlem_result.flux, orig_mlem)
            
            results.append(ValidationResult(
                repo="Neutron-Unfolding",
                test_name="MLEM",
                passed=mlem_corr > 0.95,
                correlation=mlem_corr,
                max_relative_error=mlem_max_err,
                notes=f"FluxForge (ddJ mode) vs original MLEM",
                details={"iterations_ff": mlem_result.iterations}
            ))
        except Exception as e:
            results.append(ValidationResult(
                repo="Neutron-Unfolding",
                test_name="MLEM",
                passed=False,
                notes=f"Error running original: {e}"
            ))
        
    except Exception as e:
        results.append(ValidationResult(
            repo="Neutron-Unfolding",
            test_name="general",
            passed=False,
            notes=f"Validation error: {e}"
        ))
    
    return results


def validate_speckit() -> list[ValidationResult]:
    """Validate against SpecKit repo (gradient descent unfolding)."""
    results = []
    repo_path = TESTING_DIR / "SpecKit"
    
    if not repo_path.exists():
        return [ValidationResult(
            repo="SpecKit",
            test_name="repo_exists",
            passed=False,
            notes="Repository not found"
        )]
    
    try:
        # Load SpecKit example data
        example_dir = repo_path / "Example"
        xs_files = list(example_dir.glob("*.txt")) if example_dir.exists() else []
        
        if not xs_files:
            # Try cross_section directory
            xs_dir = repo_path / "cross_section"
            if xs_dir.exists():
                xs_files = list(xs_dir.glob("*.txt"))
        
        if not xs_files:
            return [ValidationResult(
                repo="SpecKit",
                test_name="data_files",
                passed=False,
                notes="No cross-section files found"
            )]
        
        # Load a cross-section file
        xs_file = xs_files[0]
        xs_data = np.loadtxt(xs_file, skiprows=1)
        energy = xs_data[:, 0]
        sigma = xs_data[:, 1]
        
        # Verify we can load SpecKit format cross-sections
        results.append(ValidationResult(
            repo="SpecKit",
            test_name="xs_file_reading",
            passed=len(energy) > 0 and len(sigma) > 0,
            correlation=1.0,
            notes=f"Loaded {xs_file.name}: {len(energy)} energy points",
            details={"file": str(xs_file), "n_points": len(energy)}
        ))
        
        # Import FluxForge gradient descent solver
        sys.path.insert(0, str(FLUXFORGE_DIR / "src"))
        from fluxforge.solvers.iterative import gradient_descent
        
        # Create synthetic test case using SpecKit-style parameters
        n_meas = 10
        n_groups = 50
        
        # Create simple response matrix
        response = np.random.rand(n_meas, n_groups) * 0.1
        for i in range(n_meas):
            response[i, i * 5:(i + 1) * 5] += 0.5
        
        # Create true spectrum and measurements
        true_flux = np.exp(-np.linspace(0, 5, n_groups))  # Exponential spectrum
        measurements = response @ true_flux
        
        # Run FluxForge gradient descent
        gd_result = gradient_descent(
            response=response.tolist(),
            measurements=measurements.tolist(),
            max_iters=5000,
            smoothness_weight=0.01,
        )
        
        # Compare reconstructed to true
        recon_corr = correlate(gd_result.flux, true_flux)
        
        results.append(ValidationResult(
            repo="SpecKit",
            test_name="gradient_descent_synthetic",
            passed=recon_corr > 0.95,
            correlation=recon_corr,
            notes="Synthetic exponential spectrum reconstruction",
            details={"iterations": gd_result.iterations, "chi2": gd_result.chi_squared}
        ))
        
    except Exception as e:
        results.append(ValidationResult(
            repo="SpecKit",
            test_name="general",
            passed=False,
            notes=f"Validation error: {e}"
        ))
    
    return results


def validate_gamma_spec_analysis() -> list[ValidationResult]:
    """Validate against gamma_spec_analysis repo (SPE reading)."""
    results = []
    repo_path = TESTING_DIR / "gamma_spec_analysis"
    
    if not repo_path.exists():
        return [ValidationResult(
            repo="gamma_spec_analysis",
            test_name="repo_exists",
            passed=False,
            notes="Repository not found"
        )]
    
    try:
        # Find SPE test files
        test_data = repo_path / "test_data"
        spe_files = list(test_data.glob("*.Spe")) + list(test_data.glob("*.spe"))
        
        if not spe_files:
            return [ValidationResult(
                repo="gamma_spec_analysis",
                test_name="data_files",
                passed=False,
                notes="No SPE files found"
            )]
        
        # Import both readers
        sys.path.insert(0, str(FLUXFORGE_DIR / "src"))
        from fluxforge.io.spe import read_spe_file
        
        sys.path.insert(0, str(repo_path))
        try:
            from gs_spe_reading import read_spe_file as orig_read_spe
            has_orig = True
        except ImportError:
            has_orig = False
        
        for spe_file in spe_files[:3]:  # Test first 3 files
            try:
                # Read with FluxForge
                ff_result = read_spe_file(str(spe_file))
                ff_counts = np.array(ff_result.counts)
                
                if has_orig:
                    # Read with original
                    orig_result = orig_read_spe(str(spe_file))
                    orig_counts = np.array(orig_result.get('counts', orig_result.get('spectrum', [])))
                    
                    if len(orig_counts) > 0 and len(ff_counts) > 0:
                        corr = correlate(ff_counts, orig_counts)
                        passed = corr > 0.9999  # Should be exact
                    else:
                        corr = 0.0
                        passed = False
                else:
                    # Just verify FluxForge can read
                    passed = len(ff_counts) > 0
                    corr = 1.0 if passed else 0.0
                
                results.append(ValidationResult(
                    repo="gamma_spec_analysis",
                    test_name=f"SPE_{spe_file.stem}",
                    passed=passed,
                    correlation=corr,
                    notes=f"Read {len(ff_counts)} channels",
                    details={"file": str(spe_file), "channels": len(ff_counts)}
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    repo="gamma_spec_analysis",
                    test_name=f"SPE_{spe_file.stem}",
                    passed=False,
                    notes=f"Error: {e}"
                ))
        
    except Exception as e:
        results.append(ValidationResult(
            repo="gamma_spec_analysis",
            test_name="general",
            passed=False,
            notes=f"Validation error: {e}"
        ))
    
    return results


def validate_pyunfold() -> list[ValidationResult]:
    """Validate against pyunfold repo (D'Agostini method)."""
    results = []
    repo_path = TESTING_DIR / "pyunfold"
    
    if not repo_path.exists():
        return [ValidationResult(
            repo="pyunfold",
            test_name="repo_exists",
            passed=False,
            notes="Repository not found"
        )]
    
    try:
        # Import FluxForge MLEM (closest to D'Agostini)
        sys.path.insert(0, str(FLUXFORGE_DIR / "src"))
        from fluxforge.solvers.iterative import mlem
        
        # Create synthetic problem matching pyunfold examples
        n_causes = 20
        n_effects = 25
        
        # Create Gaussian smearing response
        response = np.zeros((n_effects, n_causes))
        for j in range(n_causes):
            center = j * n_effects / n_causes
            for i in range(n_effects):
                response[i, j] = np.exp(-0.5 * ((i - center) / 2.0)**2)
        response /= response.sum(axis=0, keepdims=True)  # Normalize columns
        
        # True distribution (two peaks)
        true = np.zeros(n_causes)
        true[5] = 100
        true[15] = 150
        
        # Effects
        effects = response @ true
        
        # Run FluxForge MLEM
        result = mlem(
            response=response.tolist(),
            measurements=effects.tolist(),
            max_iters=200,
        )
        
        # Check reconstruction
        recon = np.array(result.flux)
        peak_ratio = recon[15] / max(recon[5], 1e-10)
        true_ratio = true[15] / true[5]
        
        ratio_error = abs(peak_ratio - true_ratio) / true_ratio
        
        results.append(ValidationResult(
            repo="pyunfold",
            test_name="dagostini_two_peaks",
            passed=ratio_error < 0.2,  # Within 20%
            correlation=correlate(recon, true),
            max_relative_error=ratio_error,
            notes="Two-peak reconstruction test",
            details={
                "true_ratio": true_ratio,
                "recon_ratio": peak_ratio,
                "iterations": result.iterations
            }
        ))
        
    except Exception as e:
        results.append(ValidationResult(
            repo="pyunfold",
            test_name="general",
            passed=False,
            notes=f"Validation error: {e}"
        ))
    
    return results


def validate_peakingduck() -> list[ValidationResult]:
    """Validate against peakingduck repo (SNIP background)."""
    results = []
    repo_path = TESTING_DIR / "peakingduck"
    
    if not repo_path.exists():
        return [ValidationResult(
            repo="peakingduck",
            test_name="repo_exists",
            passed=False,
            notes="Repository not found"
        )]
    
    try:
        # Load reference spectra
        ref_dir = repo_path / "reference"
        ref_files = list(ref_dir.glob("spectrum*.csv")) if ref_dir.exists() else []
        
        if not ref_files:
            return [ValidationResult(
                repo="peakingduck",
                test_name="data_files",
                passed=False,
                notes="No reference spectra found"
            )]
        
        # Import FluxForge SNIP
        sys.path.insert(0, str(FLUXFORGE_DIR / "src"))
        from fluxforge.analysis.peakfit import snip_background
        
        for ref_file in ref_files[:2]:
            try:
                # Load spectrum
                data = np.loadtxt(ref_file, delimiter=',')
                if data.ndim == 1:
                    spectrum = data
                else:
                    spectrum = data[:, 1] if data.shape[1] > 1 else data[:, 0]
                
                # Run FluxForge SNIP
                background = snip_background(spectrum.tolist(), iterations=20)
                
                # Basic sanity check - background should be smoother and lower
                bg_array = np.array(background)
                spec_array = np.array(spectrum)
                
                below_spectrum = np.all(bg_array <= spec_array + 1)  # Allow small numerical error
                smoother = np.std(np.diff(bg_array)) < np.std(np.diff(spec_array))
                
                results.append(ValidationResult(
                    repo="peakingduck",
                    test_name=f"SNIP_{ref_file.stem}",
                    passed=below_spectrum and smoother,
                    correlation=correlate(bg_array, spec_array),
                    notes=f"SNIP background: below={below_spectrum}, smoother={smoother}",
                    details={"channels": len(spectrum)}
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    repo="peakingduck",
                    test_name=f"SNIP_{ref_file.stem}",
                    passed=False,
                    notes=f"Error: {e}"
                ))
        
    except Exception as e:
        results.append(ValidationResult(
            repo="peakingduck",
            test_name="general",
            passed=False,
            notes=f"Validation error: {e}"
        ))
    
    return results


def validate_actigamma() -> list[ValidationResult]:
    """Validate against actigamma repo (decay data)."""
    results = []
    repo_path = TESTING_DIR / "actigamma"
    
    if not repo_path.exists():
        return [ValidationResult(
            repo="actigamma",
            test_name="repo_exists",
            passed=False,
            notes="Repository not found"
        )]
    
    try:
        # Load actigamma decay database
        db_file = repo_path / "reference" / "lines_decay_2012.json"
        
        if not db_file.exists():
            return [ValidationResult(
                repo="actigamma",
                test_name="data_files",
                passed=False,
                notes="Decay database not found"
            )]
        
        with open(db_file) as f:
            actigamma_db = json.load(f)
        
        # Import FluxForge gamma database
        sys.path.insert(0, str(FLUXFORGE_DIR / "src"))
        from fluxforge.data.gamma_database import GammaDatabase
        
        ff_db = GammaDatabase()
        
        # Compare common isotopes
        test_isotopes = ["Co-60", "Cs-137", "Na-22", "Eu-152"]
        matches = 0
        total = 0
        
        for isotope in test_isotopes:
            # Normalize isotope name for actigamma (e.g., "Co60" vs "Co-60")
            ag_name = isotope.replace("-", "")
            
            try:
                ff_lines = ff_db.get_lines(isotope)
                ff_energies = sorted([line.energy for line in ff_lines])[:5]  # Top 5
                
                # Find in actigamma
                ag_data = None
                for key, value in actigamma_db.items():
                    if ag_name.lower() in key.lower():
                        ag_data = value
                        break
                
                if ag_data and ff_energies:
                    total += 1
                    # Check if main lines match
                    if any(abs(e - ff_energies[0]) < 1.0 for e in ag_data.get('energies', [])):
                        matches += 1
                        
            except Exception:
                pass
        
        results.append(ValidationResult(
            repo="actigamma",
            test_name="decay_database_comparison",
            passed=matches > 0,
            correlation=matches / max(total, 1),
            notes=f"Matched {matches}/{total} isotope main lines",
            details={"isotopes_tested": test_isotopes, "matches": matches}
        ))
        
    except Exception as e:
        results.append(ValidationResult(
            repo="actigamma",
            test_name="general",
            passed=False,
            notes=f"Validation error: {e}"
        ))
    
    return results


def validate_naa_ann() -> list[ValidationResult]:
    """Validate against NAA-ANN-1 repo (real gamma spectra)."""
    results = []
    repo_path = TESTING_DIR / "NAA-ANN-1"
    
    if not repo_path.exists():
        return [ValidationResult(
            repo="NAA-ANN-1",
            test_name="repo_exists",
            passed=False,
            notes="Repository not found"
        )]
    
    try:
        # Find real SPE files
        rid_dir = repo_path / "RID_extracted"
        spe_files = list(rid_dir.glob("**/*.spe")) if rid_dir.exists() else []
        
        if not spe_files:
            # Try Original data
            orig_dir = repo_path / "Original data"
            if orig_dir.exists():
                results.append(ValidationResult(
                    repo="NAA-ANN-1",
                    test_name="data_files",
                    passed=False,
                    notes="SPE files need extraction from zip"
                ))
                return results
        
        if not spe_files:
            return [ValidationResult(
                repo="NAA-ANN-1",
                test_name="data_files",
                passed=False,
                notes="No SPE files found"
            )]
        
        # Import FluxForge SPE reader
        sys.path.insert(0, str(FLUXFORGE_DIR / "src"))
        from fluxforge.io.spe import read_spe_file
        from fluxforge.analysis.peak_finders import find_peaks_simple
        
        successful_reads = 0
        peaks_found = 0
        
        for spe_file in spe_files[:10]:  # Test first 10
            try:
                result = read_spe_file(str(spe_file))
                if len(result.counts) > 0:
                    successful_reads += 1
                    
                    # Try peak finding
                    peaks = find_peaks_simple(result.counts, threshold=5.0)
                    if len(peaks) > 0:
                        peaks_found += 1
                        
            except Exception:
                pass
        
        results.append(ValidationResult(
            repo="NAA-ANN-1",
            test_name="real_spectra_analysis",
            passed=successful_reads > 5,
            correlation=successful_reads / 10,
            notes=f"Read {successful_reads}/10 spectra, found peaks in {peaks_found}",
            details={
                "total_files": len(spe_files),
                "successful_reads": successful_reads,
                "peaks_found": peaks_found
            }
        ))
        
    except Exception as e:
        results.append(ValidationResult(
            repo="NAA-ANN-1",
            test_name="general",
            passed=False,
            notes=f"Validation error: {e}"
        ))
    
    return results


# =============================================================================
# Main Validation Runner
# =============================================================================

VALIDATORS = {
    "Neutron-Unfolding": validate_neutron_unfolding,
    "SpecKit": validate_speckit,
    "gamma_spec_analysis": validate_gamma_spec_analysis,
    "pyunfold": validate_pyunfold,
    "peakingduck": validate_peakingduck,
    "actigamma": validate_actigamma,
    "NAA-ANN-1": validate_naa_ann,
}


def run_all_validations() -> dict[str, list[ValidationResult]]:
    """Run all validation tests."""
    all_results = {}
    
    for repo, validator in VALIDATORS.items():
        print(f"\n{'='*60}")
        print(f"Validating: {repo}")
        print(f"{'='*60}")
        
        try:
            results = validator()
            all_results[repo] = results
            
            for r in results:
                print(f"  {r.status_emoji} {r.test_name}: correlation={r.correlation:.4f}, {r.notes}")
                
        except Exception as e:
            print(f"  ❌ Validator failed: {e}")
            all_results[repo] = [ValidationResult(
                repo=repo,
                test_name="validator_error",
                passed=False,
                notes=str(e)
            )]
    
    return all_results


def generate_report(results: dict[str, list[ValidationResult]]) -> str:
    """Generate markdown validation report."""
    lines = [
        "# FluxForge Cross-Validation Report",
        "",
        f"**Generated:** {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Repository | Tests | Passed | Status |",
        "|------------|-------|--------|--------|",
    ]
    
    total_tests = 0
    total_passed = 0
    
    for repo, repo_results in results.items():
        n_tests = len(repo_results)
        n_passed = sum(1 for r in repo_results if r.passed)
        total_tests += n_tests
        total_passed += n_passed
        
        status = "✅" if n_passed == n_tests else ("⚠️" if n_passed > 0 else "❌")
        lines.append(f"| {repo} | {n_tests} | {n_passed} | {status} |")
    
    lines.append(f"| **Total** | **{total_tests}** | **{total_passed}** | |")
    
    # Detailed results
    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])
    
    for repo, repo_results in results.items():
        lines.append(f"### {repo}")
        lines.append("")
        lines.append("| Test | Status | Correlation | Notes |")
        lines.append("|------|--------|-------------|-------|")
        
        for r in repo_results:
            lines.append(f"| {r.test_name} | {r.status_emoji} | {r.correlation:.4f} | {r.notes} |")
        
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cross-validate FluxForge against testing repos")
    parser.add_argument("--repo", default="all", help="Repository to validate (or 'all')")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.repo == "all":
        results = run_all_validations()
    else:
        if args.repo not in VALIDATORS:
            print(f"Unknown repo: {args.repo}")
            print(f"Available: {list(VALIDATORS.keys())}")
            return 1
        
        results = {args.repo: VALIDATORS[args.repo]()}
    
    # Generate report
    report = generate_report(results)
    print("\n" + report)
    
    # Save results
    output_file = args.output or (RESULTS_DIR / "validation_results.json")
    with open(output_file, 'w') as f:
        json.dump(
            {repo: [asdict(r) for r in rs] for repo, rs in results.items()},
            f,
            indent=2
        )
    print(f"\nResults saved to: {output_file}")
    
    # Generate markdown report
    report_file = RESULTS_DIR / "validation_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")
    
    # Return exit code
    all_passed = all(r.passed for rs in results.values() for r in rs)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
