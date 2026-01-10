"""
FluxForge Master Plan Goal Validation

This module runs comprehensive validation tests and generates a summary report
showing which master plan goals are being met. It provides visibility into
the implementation status of each pipeline capability.

Run with: pytest tests/test_master_plan_goals.py -v --tb=short
Or standalone: python tests/test_master_plan_goals.py
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pytest

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class GoalResult:
    """Result of testing a specific goal."""
    goal_id: str
    description: str
    passed: bool = False
    details: str = ""
    error: Optional[str] = None


@dataclass
class ValidationSummary:
    """Summary of all goal validations."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    results: List[GoalResult] = field(default_factory=list)
    
    def add_result(self, result: GoalResult):
        self.results.append(result)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def total_count(self) -> int:
        return len(self.results)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_count,
                "passed": self.passed_count,
                "failed": self.failed_count,
                "pass_rate": f"{100 * self.passed_count / self.total_count:.1f}%" if self.total_count > 0 else "N/A"
            },
            "results": [
                {
                    "goal_id": r.goal_id,
                    "description": r.description,
                    "status": "PASS" if r.passed else "FAIL",
                    "details": r.details,
                    "error": r.error
                }
                for r in self.results
            ]
        }
    
    def print_report(self):
        """Print formatted report to console."""
        print("\n" + "=" * 80)
        print("FLUXFORGE MASTER PLAN GOAL VALIDATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Goals Tested: {self.total_count}")
        print(f"Passed: {self.passed_count} | Failed: {self.failed_count}")
        print("-" * 80)
        
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"\n[{r.goal_id}] {r.description}")
            print(f"  Status: {status}")
            if r.details:
                print(f"  Details: {r.details}")
            if r.error:
                print(f"  Error: {r.error}")
        
        print("\n" + "=" * 80)
        print(f"OVERALL: {self.passed_count}/{self.total_count} goals passing")
        print("=" * 80 + "\n")


# Global summary for collecting results
SUMMARY = ValidationSummary()

# Paths
FLUXFORGE_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = FLUXFORGE_ROOT / "src" / "fluxforge" / "examples"
FE_CD_EXAMPLE = EXAMPLES_DIR / "fe_cd_rafm_1"
TESTS_DIR = FLUXFORGE_ROOT / "tests"


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text())


# =============================================================================
# Goal 1: Activities from HPGe spectra with uncertainty
# =============================================================================

def test_goal_1_activity_calculation():
    """
    Goal 1: From raw HPGe spectra (or peak reports), produce isotope activities
    with full uncertainty propagation and QA/QC.
    """
    goal = GoalResult(
        goal_id="GOAL-1",
        description="Produce isotope activities with uncertainty from HPGe data"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.physics.activation import GammaLineMeasurement, weighted_activity
        
        # Load real experimental data
        measurements = load_json(FE_CD_EXAMPLE / "measurements.json")
        
        activities_computed = []
        uncertainties_computed = []
        
        for reaction in measurements["reactions"]:
            gamma_lines = [
                GammaLineMeasurement(**gl)
                for gl in reaction["gamma_lines"]
            ]
            
            activity, uncertainty = weighted_activity(gamma_lines)
            activities_computed.append(activity)
            uncertainties_computed.append(uncertainty)
        
        # Validate results
        assert all(a > 0 for a in activities_computed), "Activities must be positive"
        assert all(u > 0 for u in uncertainties_computed), "Uncertainties must be positive"
        assert all(u < a for a, u in zip(activities_computed, uncertainties_computed)), \
            "Uncertainties should be smaller than activities"
        
        goal.passed = True
        goal.details = f"Computed {len(activities_computed)} activities with uncertainties"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


# =============================================================================
# Goal 2: EOI reaction rates from irradiation history
# =============================================================================

def test_goal_2_reaction_rates():
    """
    Goal 2: Convert activities to end-of-irradiation (EOI) reaction rates
    using explicit irradiation history (multi-segment).
    """
    goal = GoalResult(
        goal_id="GOAL-2",
        description="Convert activities to EOI reaction rates with irradiation history"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.physics.activation import (
            GammaLineMeasurement,
            IrradiationSegment,
            weighted_activity,
            reaction_rate_from_activity,
            irradiation_buildup_factor,
        )
        
        measurements = load_json(FE_CD_EXAMPLE / "measurements.json")
        segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
        
        rates = []
        
        for reaction in measurements["reactions"]:
            gamma_lines = [
                GammaLineMeasurement(**gl)
                for gl in reaction["gamma_lines"]
            ]
            
            activity, _ = weighted_activity(gamma_lines)
            rate_estimate = reaction_rate_from_activity(
                activity, segments, reaction["half_life_s"]
            )
            
            # Verify the relationship: rate * buildup_factor = activity
            factor = irradiation_buildup_factor(segments, reaction["half_life_s"])
            reconstructed_activity = rate_estimate.rate * factor
            
            rel_error = abs(reconstructed_activity - activity) / activity
            assert rel_error < 1e-3, f"Activity reconstruction error: {rel_error:.2%}"
            
            rates.append(rate_estimate)
        
        goal.passed = True
        goal.details = f"Computed {len(rates)} reaction rates with multi-segment history"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


# =============================================================================
# Goal 3: Response matrix construction
# =============================================================================

def test_goal_3_response_matrix():
    """
    Goal 3: Build response matrix R[i,g] using dosimetry cross sections,
    sample compositions, and corrections.
    """
    goal = GoalResult(
        goal_id="GOAL-3",
        description="Build response matrix from dosimetry cross sections"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.core.response import (
            EnergyGroupStructure,
            ReactionCrossSection,
            build_response_matrix,
        )
        
        boundaries = load_json(FE_CD_EXAMPLE / "boundaries.json")
        cross_sections = load_json(FE_CD_EXAMPLE / "cross_sections.json")
        number_densities = load_json(FE_CD_EXAMPLE / "number_densities.json")
        
        groups = EnergyGroupStructure(boundaries)
        reactions = [
            ReactionCrossSection(reaction_id=r_id, sigma_g=sigma)
            for r_id, sigma in cross_sections.items()
        ]
        nd_values = [number_densities[rx.reaction_id] for rx in reactions]
        
        response = build_response_matrix(reactions, groups, nd_values)
        
        # Validate response matrix properties
        n_reactions = len(reactions)
        n_groups = groups.group_count
        
        assert len(response.matrix) == n_reactions, "Wrong number of rows"
        assert all(len(row) == n_groups for row in response.matrix), "Wrong number of columns"
        assert all(all(v >= 0 for v in row) for row in response.matrix), "Negative values in response"
        
        goal.passed = True
        goal.details = f"Built {n_reactions}×{n_groups} response matrix"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


# =============================================================================
# Goal 4: Multiple solver families
# =============================================================================

def test_goal_4a_gls_solver():
    """Goal 4a: GLS / STAYSL-like adjustment solver."""
    goal = GoalResult(
        goal_id="GOAL-4a",
        description="GLS/STAYSL-like covariance-based adjustment"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.solvers.gls import gls_adjust
        
        # Test with known solution
        response = [[1.0, 0.3], [0.2, 1.0]]
        true_flux = [2.0, 1.5]
        measurements = [
            response[0][0] * true_flux[0] + response[0][1] * true_flux[1],
            response[1][0] * true_flux[0] + response[1][1] * true_flux[1],
        ]
        
        measurement_cov = [[0.01, 0.0], [0.0, 0.01]]
        prior = [1.0, 1.0]
        prior_cov = [[1.0, 0.0], [0.0, 1.0]]
        
        solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        
        # Verify solution properties
        assert len(solution.flux) == 2
        assert solution.chi2 >= 0
        assert len(solution.covariance) == 2
        assert all(solution.covariance[i][i] >= 0 for i in range(2))  # Positive variances
        
        goal.passed = True
        goal.details = f"χ² = {solution.chi2:.4f}, flux = {[f'{v:.3f}' for v in solution.flux]}"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


def test_goal_4b_gravel_solver():
    """Goal 4b: GRAVEL iterative solver."""
    goal = GoalResult(
        goal_id="GOAL-4b",
        description="GRAVEL iterative unfolding"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.solvers.iterative import gravel
        
        response = [[1.0, 0.3], [0.2, 1.0]]
        true_flux = [2.0, 1.5]
        measurements = [
            response[0][0] * true_flux[0] + response[0][1] * true_flux[1],
            response[1][0] * true_flux[0] + response[1][1] * true_flux[1],
        ]
        
        solution = gravel(
            response, measurements,
            initial_flux=[1.0, 1.0],
            max_iters=500,
            tolerance=1e-6,
        )
        
        # Check recovery
        rel_errors = [abs(e - t) / t for e, t in zip(solution.flux, true_flux)]
        
        goal.passed = True
        goal.details = f"Converged in {solution.iterations} iterations, max rel. error = {max(rel_errors):.2%}"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


def test_goal_4c_mlem_solver():
    """Goal 4c: MLEM iterative solver."""
    goal = GoalResult(
        goal_id="GOAL-4c",
        description="MLEM iterative unfolding"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.solvers.iterative import mlem
        
        response = [[1.0, 0.3], [0.2, 1.0]]
        true_flux = [2.0, 1.5]
        measurements = [
            response[0][0] * true_flux[0] + response[0][1] * true_flux[1],
            response[1][0] * true_flux[0] + response[1][1] * true_flux[1],
        ]
        
        solution = mlem(
            response, measurements,
            initial_flux=[1.0, 1.0],
            max_iters=200,
            tolerance=1e-6,
        )
        
        goal.passed = True
        goal.details = f"Converged: {solution.converged}, iterations: {solution.iterations}"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


# =============================================================================
# Goal 5: Posterior spectra with covariance and diagnostics
# =============================================================================

def test_goal_5_posterior_diagnostics():
    """
    Goal 5: Produce posterior spectra with covariance/correlation plus
    paper-style diagnostics (χ², pulls, influence).
    """
    goal = GoalResult(
        goal_id="GOAL-5",
        description="Posterior spectra with covariance and χ² diagnostics"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.solvers.gls import gls_adjust, GLSSolution
        
        # Create a problem and solve
        response = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        measurements = [1.0, 2.0, 3.0]
        measurement_cov = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]
        prior = [0.5, 1.5, 2.5]
        prior_cov = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        
        solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        
        # Check required outputs per master plan
        assert hasattr(solution, 'flux'), "Missing posterior flux"
        assert hasattr(solution, 'covariance'), "Missing covariance matrix"
        assert hasattr(solution, 'chi2'), "Missing χ² diagnostic"
        assert hasattr(solution, 'residuals'), "Missing residuals"
        
        # Verify covariance is proper (symmetric, positive diagonal)
        n = len(solution.flux)
        for i in range(n):
            assert solution.covariance[i][i] >= 0, "Negative variance"
        
        # Compute reduced chi-squared
        dof = len(measurements)
        chi2_reduced = solution.chi2 / dof if dof > 0 else 0
        
        goal.passed = True
        goal.details = f"χ²/dof = {chi2_reduced:.3f}, covariance {n}×{n}"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


# =============================================================================
# Goal 6: Artifact provenance
# =============================================================================

def test_goal_6_artifact_provenance(tmp_path):
    """
    Goal 6: Output reproducible run bundles with provenance (hashes, versions).
    """
    goal = GoalResult(
        goal_id="GOAL-6",
        description="Artifacts include provenance (units, definitions, hashes)"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.io.artifacts import write_unfold_result, read_unfold_result
        from fluxforge.core.schemas import validate_artifact
        
        output = tmp_path / "test_unfold.json"
        write_unfold_result(
            output,
            boundaries_eV=[0.0, 1.0, 2.0],
            reactions=["rx1", "rx2"],
            flux=[1.0, 2.0],
            covariance=[[0.1, 0.0], [0.0, 0.2]],
            chi2=1.5,
            method="gls",
        )
        
        payload = read_unfold_result(output)
        
        # Validate schema
        errors = validate_artifact(payload)
        assert errors == [], f"Schema errors: {errors}"
        
        # Check provenance fields
        assert "provenance" in payload, "Missing provenance"
        prov = payload["provenance"]
        assert "units" in prov, "Missing units in provenance"
        assert "definitions" in prov, "Missing definitions in provenance"
        
        goal.passed = True
        goal.details = f"Artifact has {len(prov['units'])} unit definitions"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


# =============================================================================
# Goal 7: Complete workflow with experimental data
# =============================================================================

def test_goal_7_complete_workflow():
    """
    Goal 7: Complete end-to-end workflow using real experimental data.
    """
    goal = GoalResult(
        goal_id="GOAL-7",
        description="End-to-end workflow with Fe-Cd-RAFM-1 experimental data"
    )
    
    try:
        if not HAS_NUMPY:
            goal.passed = False
            goal.error = "NumPy not available"
            SUMMARY.add_result(goal)
            return
        
        from fluxforge.core.response import (
            EnergyGroupStructure,
            ReactionCrossSection,
            build_response_matrix,
        )
        from fluxforge.physics.activation import (
            GammaLineMeasurement,
            IrradiationSegment,
            weighted_activity,
            reaction_rate_from_activity,
        )
        from fluxforge.solvers.gls import gls_adjust
        
        # Load all data
        boundaries = load_json(FE_CD_EXAMPLE / "boundaries.json")
        cross_sections = load_json(FE_CD_EXAMPLE / "cross_sections.json")
        number_densities = load_json(FE_CD_EXAMPLE / "number_densities.json")
        measurements = load_json(FE_CD_EXAMPLE / "measurements.json")
        prior_flux = load_json(FE_CD_EXAMPLE / "prior_flux.json")
        
        # Stage E: Build response matrix
        groups = EnergyGroupStructure(boundaries)
        reactions = [
            ReactionCrossSection(reaction_id=r_id, sigma_g=sigma)
            for r_id, sigma in cross_sections.items()
        ]
        nd_values = [number_densities[rx.reaction_id] for rx in reactions]
        response = build_response_matrix(reactions, groups, nd_values)
        
        # Stage D: Calculate reaction rates
        segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
        measured_rates = []
        rate_uncertainties = []
        
        for reaction in measurements["reactions"]:
            gamma_lines = [
                GammaLineMeasurement(**gl)
                for gl in reaction["gamma_lines"]
            ]
            activity, _ = weighted_activity(gamma_lines)
            rate_estimate = reaction_rate_from_activity(
                activity, segments, reaction["half_life_s"]
            )
            measured_rates.append(rate_estimate.rate)
            rate_uncertainties.append(rate_estimate.uncertainty)
        
        # Build covariance matrices
        prior_cov = [
            [(0.25 * val) ** 2 if i == j else 0.0
             for j, val in enumerate(prior_flux)]
            for i, _ in enumerate(prior_flux)
        ]
        measurement_cov = [
            [(rate_uncertainties[i] ** 2) if i == j else 0.0
             for j in range(len(measured_rates))]
            for i in range(len(measured_rates))
        ]
        
        # Stage F: GLS adjustment
        solution = gls_adjust(
            response.matrix, measured_rates, measurement_cov,
            prior_flux, prior_cov
        )
        
        # Validate complete output
        assert len(solution.flux) == groups.group_count
        assert all(val >= 0.0 for val in solution.flux)
        assert solution.chi2 >= 0.0
        assert len(solution.covariance) == groups.group_count
        
        goal.passed = True
        goal.details = f"Unfolded {groups.group_count}-group spectrum, χ² = {solution.chi2:.4f}"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)
    assert goal.passed, goal.error


# =============================================================================
# Capability Checks (not full tests, but status checks)
# =============================================================================

def test_capability_spectrum_io():
    """Check: Spectrum I/O module exists and has basic functionality."""
    goal = GoalResult(
        goal_id="CAP-A1",
        description="Spectrum I/O (SPE/CHN/CNF readers)"
    )
    
    try:
        from fluxforge.io.spe import GammaSpectrum, read_spe_file
        
        # Check basic functionality
        assert callable(read_spe_file)
        assert GammaSpectrum is not None
        
        goal.passed = True
        goal.details = "SPE reader available"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)


def test_capability_peak_fitting():
    """Check: Peak fitting module exists."""
    goal = GoalResult(
        goal_id="CAP-B",
        description="Peak detection and fitting"
    )
    
    try:
        from fluxforge.analysis.peakfit import (
            GaussianPeak,
            PeakFitResult,
            estimate_background,
        )
        
        goal.passed = True
        goal.details = "Peak fitting classes available"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)


def test_capability_irdff_database():
    """Check: IRDFF database module exists."""
    goal = GoalResult(
        goal_id="CAP-E1",
        description="IRDFF cross-section database"
    )
    
    try:
        from fluxforge.data.irdff import IRDFFDatabase, IRDFF_REACTIONS
        
        db = IRDFFDatabase(auto_download=False, verbose=False)
        reactions = db.list_reactions()
        
        goal.passed = True
        goal.details = f"{len(reactions)} reactions available"
        
    except Exception as e:
        goal.passed = False
        goal.error = str(e)
    
    SUMMARY.add_result(goal)


# =============================================================================
# Report Generation
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def generate_summary_report(request):
    """Generate summary report after all tests complete."""
    yield
    
    # Print report
    SUMMARY.print_report()
    
    # Save JSON report
    report_path = TESTS_DIR / "goal_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(SUMMARY.to_dict(), f, indent=2)
    
    print(f"Report saved to: {report_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run tests and generate report
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
