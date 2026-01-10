#!/usr/bin/env python3
"""
FluxForge Capability Audit

This script audits the FluxForge codebase against the capability specification
and generates a detailed report showing:
- What capabilities are implemented
- What capabilities are missing
- Reference implementations available in testing/

Run: python tests/audit_capabilities.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


class Status(Enum):
    IMPLEMENTED = "✅"
    PARTIAL = "🔶"
    MISSING = "❌"
    PLANNED = "📋"


@dataclass
class Capability:
    id: str
    description: str
    status: Status
    module_path: str = ""
    notes: str = ""
    reference: str = ""


def check_import(module_path: str) -> bool:
    """Check if a module can be imported."""
    try:
        parts = module_path.split(".")
        mod = __import__(parts[0])
        for part in parts[1:]:
            mod = getattr(mod, part)
        return True
    except (ImportError, AttributeError):
        return False


def check_class_exists(module_path: str, class_name: str) -> bool:
    """Check if a class exists in a module."""
    try:
        parts = module_path.split(".")
        mod = __import__(parts[0])
        for part in parts[1:]:
            mod = getattr(mod, part)
        return hasattr(mod, class_name)
    except (ImportError, AttributeError):
        return False


def check_function_exists(module_path: str, func_name: str) -> bool:
    """Check if a function exists in a module."""
    try:
        parts = module_path.split(".")
        mod = __import__(parts[0])
        for part in parts[1:]:
            mod = getattr(mod, part)
        return hasattr(mod, func_name) and callable(getattr(mod, func_name))
    except (ImportError, AttributeError):
        return False


# =============================================================================
# Capability Definitions
# =============================================================================

def audit_capabilities() -> List[Capability]:
    """Audit all capabilities from the specification."""
    capabilities = []
    
    # -------------------------------------------------------------------------
    # Stage A: HPGe Spectrum Ingestion
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="A1.1",
        description="Read SPE gamma-spectroscopy format",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.io.spe", "read_spe_file") else Status.MISSING,
        module_path="fluxforge.io.spe",
        notes="SPE reader fully implemented"
    ))
    
    capabilities.append(Capability(
        id="A1.2",
        description="Read CHN format",
        status=Status.PARTIAL if check_import("fluxforge.io.hpge") else Status.MISSING,
        module_path="fluxforge.io.hpge",
        notes="Generic HPGe reader exists, CHN format partial",
        reference="testing/irrad_spectroscopy"
    ))
    
    capabilities.append(Capability(
        id="A1.3",
        description="Read CNF format",
        status=Status.MISSING,
        notes="CNF (Canberra) format not implemented",
        reference="testing/hdtv for CNF support"
    ))
    
    capabilities.append(Capability(
        id="A1.4",
        description="Read N42/IEC format",
        status=Status.MISSING,
        notes="N42 XML format not implemented"
    ))
    
    capabilities.append(Capability(
        id="A1.5",
        description="Background estimation (SNIP)",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.analysis.peakfit", "estimate_background") else Status.MISSING,
        module_path="fluxforge.analysis.peakfit"
    ))
    
    capabilities.append(Capability(
        id="A1.6",
        description="Dead-time validation",
        status=Status.IMPLEMENTED if check_class_exists("fluxforge.io.spe", "GammaSpectrum") else Status.MISSING,
        module_path="fluxforge.io.spe",
        notes="GammaSpectrum.dead_time_fraction property"
    ))
    
    # -------------------------------------------------------------------------
    # Stage B: Peak Detection and Fitting
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="B1.1",
        description="Automated peak finding with tunable sensitivity",
        status=Status.PARTIAL if check_import("fluxforge.analysis.peakfit") else Status.MISSING,
        module_path="fluxforge.analysis.peakfit",
        notes="Basic peak finding exists, needs sensitivity tuning",
        reference="testing/peakingduck for advanced peak finding"
    ))
    
    capabilities.append(Capability(
        id="B1.2",
        description="Gaussian peak fitting",
        status=Status.IMPLEMENTED if check_class_exists("fluxforge.analysis.peakfit", "GaussianPeak") else Status.MISSING,
        module_path="fluxforge.analysis.peakfit"
    ))
    
    capabilities.append(Capability(
        id="B1.3",
        description="Peak tailing terms (Hypermet)",
        status=Status.MISSING,
        notes="Only Gaussian implemented, need Hypermet/EMG",
        reference="testing/hdtv has advanced peak shapes"
    ))
    
    capabilities.append(Capability(
        id="B1.4",
        description="Multiplet handling",
        status=Status.PARTIAL,
        notes="Basic multiplet awareness, needs constrained fitting",
        reference="testing/hdtv"
    ))
    
    capabilities.append(Capability(
        id="B1.5",
        description="Peak fit covariance output",
        status=Status.IMPLEMENTED if check_class_exists("fluxforge.analysis.peakfit", "PeakFitResult") else Status.MISSING,
        module_path="fluxforge.analysis.peakfit"
    ))
    
    # -------------------------------------------------------------------------
    # Stage C: Efficiency and Activity
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="C1.1",
        description="Efficiency curve fitting",
        status=Status.IMPLEMENTED if check_import("fluxforge.data.efficiency") else Status.MISSING,
        module_path="fluxforge.data.efficiency"
    ))
    
    capabilities.append(Capability(
        id="C1.2",
        description="Activity calculation from peak area",
        status=Status.IMPLEMENTED if check_class_exists("fluxforge.physics.activation", "GammaLineMeasurement") else Status.MISSING,
        module_path="fluxforge.physics.activation"
    ))
    
    capabilities.append(Capability(
        id="C1.3",
        description="Weighted activity from multiple gamma lines",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.physics.activation", "weighted_activity") else Status.MISSING,
        module_path="fluxforge.physics.activation"
    ))
    
    capabilities.append(Capability(
        id="C1.4",
        description="Coincidence summing corrections",
        status=Status.MISSING,
        notes="Not yet implemented",
        reference="testing/actigamma"
    ))
    
    # -------------------------------------------------------------------------
    # Stage D: Reaction Rates
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="D1.1",
        description="Irradiation history model (multi-segment)",
        status=Status.IMPLEMENTED if check_class_exists("fluxforge.physics.activation", "IrradiationSegment") else Status.MISSING,
        module_path="fluxforge.physics.activation"
    ))
    
    capabilities.append(Capability(
        id="D1.2",
        description="Decay during irradiation/cooling/counting",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.physics.activation", "irradiation_buildup_factor") else Status.MISSING,
        module_path="fluxforge.physics.activation"
    ))
    
    capabilities.append(Capability(
        id="D1.3",
        description="EOI reaction rate calculation",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.physics.activation", "reaction_rate_from_activity") else Status.MISSING,
        module_path="fluxforge.physics.activation"
    ))
    
    # -------------------------------------------------------------------------
    # Stage E: Response Matrix Construction
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="E1.1",
        description="Energy group structure definition",
        status=Status.IMPLEMENTED if check_class_exists("fluxforge.core.response", "EnergyGroupStructure") else Status.MISSING,
        module_path="fluxforge.core.response"
    ))
    
    capabilities.append(Capability(
        id="E1.2",
        description="Response matrix construction",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.core.response", "build_response_matrix") else Status.MISSING,
        module_path="fluxforge.core.response"
    ))
    
    capabilities.append(Capability(
        id="E1.3",
        description="IRDFF cross sections",
        status=Status.IMPLEMENTED if check_class_exists("fluxforge.data.irdff", "IRDFFDatabase") else Status.MISSING,
        module_path="fluxforge.data.irdff"
    ))
    
    capabilities.append(Capability(
        id="E1.4",
        description="Self-shielding corrections",
        status=Status.MISSING,
        notes="Self-shielding not implemented",
        reference="STAYSL documentation"
    ))
    
    capabilities.append(Capability(
        id="E1.5",
        description="Cadmium cover corrections",
        status=Status.MISSING,
        notes="Cd cover corrections not implemented"
    ))
    
    # -------------------------------------------------------------------------
    # Stage F: Unfolding Solvers
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="F1.1",
        description="GLS spectrum adjustment (STAYSL-like)",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.solvers.gls", "gls_adjust") else Status.MISSING,
        module_path="fluxforge.solvers.gls"
    ))
    
    capabilities.append(Capability(
        id="F1.2",
        description="GRAVEL iterative unfolding",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.solvers.iterative", "gravel") else Status.MISSING,
        module_path="fluxforge.solvers.iterative",
        reference="testing/Neutron-Unfolding/gravel.py"
    ))
    
    capabilities.append(Capability(
        id="F1.3",
        description="MLEM iterative unfolding",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.solvers.iterative", "mlem") else Status.MISSING,
        module_path="fluxforge.solvers.iterative",
        reference="testing/Neutron-Unfolding/mlem.py"
    ))
    
    capabilities.append(Capability(
        id="F1.4",
        description="Bayesian MCMC unfolding",
        status=Status.MISSING,
        notes="MCMC solver not implemented",
        reference="testing/Neutron-Spectrometry"
    ))
    
    capabilities.append(Capability(
        id="F1.5",
        description="Gradient-descent solver (SAND-II style)",
        status=Status.MISSING,
        notes="Gradient solver not implemented",
        reference="testing/SpecKit/src/neutron_spectrum_solver.py"
    ))
    
    capabilities.append(Capability(
        id="F1.6",
        description="Positivity constraints",
        status=Status.IMPLEMENTED,
        module_path="fluxforge.solvers.gls",
        notes="GLS enforces non-negativity"
    ))
    
    capabilities.append(Capability(
        id="F1.7",
        description="Chi-square diagnostics",
        status=Status.IMPLEMENTED,
        module_path="fluxforge.solvers.gls",
        notes="GLSSolution.chi2 available"
    ))
    
    # -------------------------------------------------------------------------
    # Stage G: Plotting and Reporting
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="G1.1",
        description="Spectrum with uncertainty bands",
        status=Status.IMPLEMENTED if check_import("fluxforge.plots.unfolding") else Status.MISSING,
        module_path="fluxforge.plots.unfolding"
    ))
    
    capabilities.append(Capability(
        id="G1.2",
        description="Prior vs posterior overlay",
        status=Status.IMPLEMENTED,
        module_path="fluxforge.plots.unfolding"
    ))
    
    capabilities.append(Capability(
        id="G1.3",
        description="Residual/pull plots",
        status=Status.IMPLEMENTED,
        notes="In examples/generate_plots.py"
    ))
    
    capabilities.append(Capability(
        id="G1.4",
        description="Covariance/correlation heatmaps",
        status=Status.IMPLEMENTED,
        notes="In examples/generate_plots.py"
    ))
    
    capabilities.append(Capability(
        id="G1.5",
        description="Parity plot (predicted vs measured)",
        status=Status.IMPLEMENTED,
        notes="In examples/generate_plots.py"
    ))
    
    # -------------------------------------------------------------------------
    # Stage H: Model Comparison (OpenMC / MCNP)
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="H1.1",
        description="OpenMC statepoint ingestion",
        status=Status.MISSING,
        notes="OpenMC integration not implemented"
    ))
    
    capabilities.append(Capability(
        id="H1.2",
        description="MCNP tally ingestion",
        status=Status.MISSING,
        notes="MCNP integration not implemented"
    ))
    
    capabilities.append(Capability(
        id="H1.3",
        description="ALARA input generation",
        status=Status.MISSING,
        notes="ALARA workflow not implemented"
    ))
    
    capabilities.append(Capability(
        id="H1.4",
        description="ALARA output parsing",
        status=Status.MISSING,
        notes="ALARA workflow not implemented"
    ))
    
    # -------------------------------------------------------------------------
    # Stage I: TRIGA / k0-NAA
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="I1.1",
        description="Cd-ratio calculations",
        status=Status.MISSING,
        notes="Cd-ratio not implemented"
    ))
    
    capabilities.append(Capability(
        id="I1.2",
        description="f and alpha parameter fitting",
        status=Status.MISSING,
        notes="Spectral parameter fitting not implemented"
    ))
    
    capabilities.append(Capability(
        id="I1.3",
        description="k0-standardization module",
        status=Status.MISSING,
        notes="k0-NAA not implemented"
    ))
    
    # -------------------------------------------------------------------------
    # Artifacts and Provenance
    # -------------------------------------------------------------------------
    
    capabilities.append(Capability(
        id="J1.1",
        description="JSON artifact output",
        status=Status.IMPLEMENTED if check_function_exists("fluxforge.io.artifacts", "write_unfold_result") else Status.MISSING,
        module_path="fluxforge.io.artifacts"
    ))
    
    capabilities.append(Capability(
        id="J1.2",
        description="Provenance metadata",
        status=Status.IMPLEMENTED if check_import("fluxforge.core.provenance") else Status.MISSING,
        module_path="fluxforge.core.provenance"
    ))
    
    capabilities.append(Capability(
        id="J1.3",
        description="Unit metadata validation",
        status=Status.IMPLEMENTED if check_import("fluxforge.core.schemas") else Status.MISSING,
        module_path="fluxforge.core.schemas"
    ))
    
    return capabilities


def generate_report(capabilities: List[Capability]) -> str:
    """Generate audit report."""
    report = []
    report.append("=" * 80)
    report.append("FLUXFORGE CAPABILITY AUDIT REPORT")
    report.append("=" * 80)
    
    # Summary counts
    counts = {s: 0 for s in Status}
    for cap in capabilities:
        counts[cap.status] += 1
    
    total = len(capabilities)
    report.append(f"\nSUMMARY: {total} capabilities audited")
    for status, count in counts.items():
        pct = 100 * count / total if total > 0 else 0
        report.append(f"  {status.value} {status.name}: {count} ({pct:.1f}%)")
    
    # By category
    categories = {}
    for cap in capabilities:
        cat = cap.id.split(".")[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(cap)
    
    cat_names = {
        "A1": "Stage A: HPGe Spectrum Ingestion",
        "B1": "Stage B: Peak Detection & Fitting",
        "C1": "Stage C: Efficiency & Activity",
        "D1": "Stage D: Reaction Rates",
        "E1": "Stage E: Response Matrix",
        "F1": "Stage F: Unfolding Solvers",
        "G1": "Stage G: Plotting & Reporting",
        "H1": "Stage H: Model Comparison",
        "I1": "Stage I: TRIGA / k0-NAA",
        "J1": "Stage J: Artifacts & Provenance",
    }
    
    for cat_id, caps in categories.items():
        cat_name = cat_names.get(cat_id, cat_id)
        report.append(f"\n{'─' * 80}")
        report.append(f"{cat_name}")
        report.append(f"{'─' * 80}")
        
        for cap in caps:
            status_icon = cap.status.value
            report.append(f"\n  [{cap.id}] {status_icon} {cap.description}")
            if cap.module_path:
                report.append(f"      Module: {cap.module_path}")
            if cap.notes:
                report.append(f"      Notes: {cap.notes}")
            if cap.reference:
                report.append(f"      Reference: {cap.reference}")
    
    report.append("\n" + "=" * 80)
    report.append("END OF AUDIT REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    print("Running FluxForge capability audit...")
    capabilities = audit_capabilities()
    report = generate_report(capabilities)
    print(report)
    
    # Save report
    report_path = repo_root / "docs" / "capability_audit.txt"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")
    
    # Return exit code based on critical missing capabilities
    missing = sum(1 for c in capabilities if c.status == Status.MISSING)
    return 0 if missing < 10 else 1


if __name__ == "__main__":
    sys.exit(main())
