#!/usr/bin/env python3
"""
Generate Flux Spectrum from Experimental Data

This script demonstrates the complete FluxForge workflow using the
Fe-Cd-RAFM-1 experimental data to produce an unfolded neutron flux spectrum.

It performs the following steps:
1. Load experimental measurements (gamma line activities)
2. Calculate reaction rates from activities using irradiation history
3. Build response matrix from dosimetry cross sections
4. Unfold neutron flux spectrum using GLS, GRAVEL, and MLEM solvers
5. Generate output artifacts with provenance
6. Create comparison plots

Usage:
    python examples/generate_flux_spectrum.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add FluxForge to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

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
from fluxforge.solvers.iterative import gravel, mlem
from fluxforge.io.artifacts import write_unfold_result


def load_example_data() -> Dict:
    """Load all Fe-Cd-RAFM-1 experimental data."""
    example_dir = repo_root / "src" / "fluxforge" / "examples" / "fe_cd_rafm_1"
    
    return {
        "boundaries": json.loads((example_dir / "boundaries.json").read_text()),
        "cross_sections": json.loads((example_dir / "cross_sections.json").read_text()),
        "number_densities": json.loads((example_dir / "number_densities.json").read_text()),
        "measurements": json.loads((example_dir / "measurements.json").read_text()),
        "prior_flux": json.loads((example_dir / "prior_flux.json").read_text()),
    }


def calculate_reaction_rates(measurements: Dict) -> tuple:
    """Calculate EOI reaction rates from measured activities."""
    print("\n" + "=" * 80)
    print("STAGE 1: Calculate Reaction Rates from Activities")
    print("=" * 80)
    
    segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
    
    rates = []
    rate_uncertainties = []
    reaction_ids = []
    
    for reaction in measurements["reactions"]:
        # Combine multiple gamma lines
        gamma_lines = [
            GammaLineMeasurement(**gl)
            for gl in reaction["gamma_lines"]
        ]
        
        activity, activity_unc = weighted_activity(gamma_lines)
        
        # Convert to reaction rate
        rate_estimate = reaction_rate_from_activity(
            activity, segments, reaction["half_life_s"]
        )
        
        rates.append(rate_estimate.rate)
        rate_uncertainties.append(rate_estimate.uncertainty)
        reaction_ids.append(reaction["reaction_id"])
        
        print(f"\n{reaction['reaction_id']}:")
        print(f"  Activity:      {activity:.3e} ± {activity_unc:.3e} Bq")
        print(f"  Half-life:     {reaction['half_life_s'] / 86400:.2f} days")
        print(f"  Reaction Rate: {rate_estimate.rate:.3e} ± {rate_estimate.uncertainty:.3e} reactions/s")
        print(f"  Relative Unc:  {100 * rate_estimate.uncertainty / rate_estimate.rate:.2f}%")
    
    return rates, rate_uncertainties, reaction_ids


def build_response(boundaries: List[float], cross_sections: Dict, number_densities: Dict) -> tuple:
    """Build response matrix."""
    print("\n" + "=" * 80)
    print("STAGE 2: Build Response Matrix")
    print("=" * 80)
    
    groups = EnergyGroupStructure(boundaries)
    print(f"\nEnergy group structure: {groups.group_count} groups")
    print(f"  Lower bound: {groups.boundaries_eV[0]:.4e} eV")
    print(f"  Upper bound: {groups.boundaries_eV[-1]:.4e} eV")
    
    reactions = [
        ReactionCrossSection(reaction_id=r_id, sigma_g=sigma)
        for r_id, sigma in cross_sections.items()
    ]
    
    nd_values = [number_densities[rx.reaction_id] for rx in reactions]
    
    response = build_response_matrix(reactions, groups, nd_values)
    
    print(f"\nResponse matrix shape: {len(response.matrix)} reactions × {len(response.matrix[0])} groups")
    print(f"Reactions: {[rx.reaction_id for rx in reactions]}")
    
    # Print response matrix
    print("\nResponse Matrix (R[i,g]):")
    print(f"{'Reaction':<20} " + " ".join(f"Group{g:>2}" for g in range(groups.group_count)))
    for i, rx in enumerate(reactions):
        values = " ".join(f"{response.matrix[i][g]:>7.2e}" for g in range(groups.group_count))
        print(f"{rx.reaction_id:<20} {values}")
    
    return response, groups, reactions


def unfold_spectrum_gls(
    response_matrix: List[List[float]],
    measured_rates: List[float],
    rate_uncertainties: List[float],
    prior_flux: List[float],
    prior_uncertainty_fraction: float = 0.25,
) -> Dict:
    """Unfold spectrum using GLS adjustment."""
    print("\n" + "=" * 80)
    print("STAGE 3a: GLS (STAYSL-like) Adjustment")
    print("=" * 80)
    
    # Build covariance matrices
    measurement_cov = [
        [(rate_uncertainties[i] ** 2) if i == j else 0.0
         for j in range(len(measured_rates))]
        for i in range(len(measured_rates))
    ]
    
    prior_cov = [
        [(prior_uncertainty_fraction * val) ** 2 if i == j else 0.0
         for j, val in enumerate(prior_flux)]
        for i, _ in enumerate(prior_flux)
    ]
    
    print(f"\nPrior uncertainty: {100*prior_uncertainty_fraction:.0f}%")
    print(f"Measurement uncertainties: {[f'{100*u/r:.1f}%' for u, r in zip(rate_uncertainties, measured_rates)]}")
    
    solution = gls_adjust(
        response_matrix, measured_rates, measurement_cov,
        prior_flux, prior_cov
    )
    
    print(f"\nGLS Solution:")
    print(f"  χ² = {solution.chi2:.4f}")
    print(f"  χ²/dof = {solution.chi2 / len(measured_rates):.4f}")
    print(f"  Method: Direct (non-iterative)")
    
    print(f"\n{'Group':<8} {'Prior':<12} {'Posterior':<12} {'Uncertainty':<12} {'Rel. Unc.':<12}")
    for g in range(len(solution.flux)):
        prior = prior_flux[g]
        posterior = solution.flux[g]
        unc = np.sqrt(solution.covariance[g][g])
        rel_unc = 100 * unc / posterior if posterior > 0 else 0
        print(f"{g:<8} {prior:<12.3e} {posterior:<12.3e} {unc:<12.3e} {rel_unc:<12.2f}%")
    
    return {
        "flux": solution.flux,
        "covariance": solution.covariance,
        "chi2": solution.chi2,
        "residuals": solution.residuals,
    }


def unfold_spectrum_gravel(
    response_matrix: List[List[float]],
    measured_rates: List[float],
    prior_flux: List[float],
    max_iters: int = 500,
) -> Dict:
    """Unfold spectrum using GRAVEL iterative method."""
    print("\n" + "=" * 80)
    print("STAGE 3b: GRAVEL Iterative Unfolding")
    print("=" * 80)
    
    solution = gravel(
        response_matrix,
        measured_rates,
        initial_flux=prior_flux,
        max_iters=max_iters,
        tolerance=1e-6,
    )
    
    print(f"\nGRAVEL Solution:")
    print(f"  Iterations: {solution.iterations}")
    print(f"  Converged: {solution.converged}")
    print(f"  χ² = {solution.chi_squared:.4f}")
    
    print(f"\n{'Group':<8} {'Prior':<12} {'Posterior':<12}")
    for g in range(len(solution.flux)):
        print(f"{g:<8} {prior_flux[g]:<12.3e} {solution.flux[g]:<12.3e}")
    
    return {
        "flux": solution.flux,
        "iterations": solution.iterations,
        "converged": solution.converged,
        "chi_squared": solution.chi_squared,
    }


def unfold_spectrum_mlem(
    response_matrix: List[List[float]],
    measured_rates: List[float],
    prior_flux: List[float],
    max_iters: int = 200,
) -> Dict:
    """Unfold spectrum using MLEM."""
    print("\n" + "=" * 80)
    print("STAGE 3c: MLEM Iterative Unfolding")
    print("=" * 80)
    
    solution = mlem(
        response_matrix,
        measured_rates,
        initial_flux=prior_flux,
        max_iters=max_iters,
        tolerance=1e-6,
    )
    
    print(f"\nMLEM Solution:")
    print(f"  Iterations: {solution.iterations}")
    print(f"  Converged: {solution.converged}")
    
    print(f"\n{'Group':<8} {'Prior':<12} {'Posterior':<12}")
    for g in range(len(solution.flux)):
        print(f"{g:<8} {prior_flux[g]:<12.3e} {solution.flux[g]:<12.3e}")
    
    return {
        "flux": solution.flux,
        "iterations": solution.iterations,
        "converged": solution.converged,
    }


def compare_solutions(boundaries: List[float], prior: List[float], gls: Dict, gravel: Dict, mlem: Dict):
    """Print comparison of all three methods."""
    print("\n" + "=" * 80)
    print("COMPARISON: All Three Methods")
    print("=" * 80)
    
    print(f"\n{'Group':<8} {'E_lower (eV)':<15} {'E_upper (eV)':<15} {'Prior':<12} {'GLS':<12} {'GRAVEL':<12} {'MLEM':<12}")
    print("-" * 110)
    
    for g in range(len(prior)):
        e_low = boundaries[g]
        e_high = boundaries[g + 1]
        print(f"{g:<8} {e_low:<15.3e} {e_high:<15.3e} {prior[g]:<12.3e} "
              f"{gls['flux'][g]:<12.3e} {gravel['flux'][g]:<12.3e} {mlem['flux'][g]:<12.3e}")
    
    # Print integral comparison
    print("\n" + "-" * 110)
    total_prior = sum(prior)
    total_gls = sum(gls['flux'])
    total_gravel = sum(gravel['flux'])
    total_mlem = sum(mlem['flux'])
    
    print(f"{'TOTAL':<8} {'---':<15} {'---':<15} {total_prior:<12.3e} "
          f"{total_gls:<12.3e} {total_gravel:<12.3e} {total_mlem:<12.3e}")
    
    # Relative differences
    print(f"\nRelative difference vs prior:")
    print(f"  GLS:    {100 * (total_gls - total_prior) / total_prior:+.2f}%")
    print(f"  GRAVEL: {100 * (total_gravel - total_prior) / total_prior:+.2f}%")
    print(f"  MLEM:   {100 * (total_mlem - total_prior) / total_prior:+.2f}%")


def save_outputs(boundaries: List[float], reactions: List, gls: Dict, gravel: Dict, mlem: Dict):
    """Save output artifacts."""
    print("\n" + "=" * 80)
    print("STAGE 4: Save Output Artifacts")
    print("=" * 80)
    
    output_dir = repo_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save GLS result
    gls_file = output_dir / "fe_cd_rafm_1_gls.json"
    write_unfold_result(
        gls_file,
        boundaries_eV=boundaries,
        reactions=[rx.reaction_id for rx in reactions],
        flux=gls["flux"],
        covariance=gls["covariance"],
        chi2=gls["chi2"],
        method="gls",
    )
    print(f"\nSaved GLS result: {gls_file}")
    
    # Save GRAVEL result
    gravel_file = output_dir / "fe_cd_rafm_1_gravel.json"
    write_unfold_result(
        gravel_file,
        boundaries_eV=boundaries,
        reactions=[rx.reaction_id for rx in reactions],
        flux=gravel["flux"],
        covariance=None,  # GRAVEL doesn't provide covariance
        chi2=None,
        method="gravel",
    )
    print(f"Saved GRAVEL result: {gravel_file}")
    
    # Save MLEM result
    mlem_file = output_dir / "fe_cd_rafm_1_mlem.json"
    write_unfold_result(
        mlem_file,
        boundaries_eV=boundaries,
        reactions=[rx.reaction_id for rx in reactions],
        flux=mlem["flux"],
        covariance=None,  # MLEM doesn't provide covariance
        chi2=None,
        method="mlem",
    )
    print(f"Saved MLEM result: {mlem_file}")


def main():
    """Run complete workflow."""
    print("\n" + "=" * 80)
    print("FLUXFORGE: Fe-Cd-RAFM-1 Neutron Flux Spectrum Unfolding")
    print("=" * 80)
    print("\nExperimental Data: Fe-58(n,γ)Fe-59 activation measurements")
    print("Methods: GLS (STAYSL-like), GRAVEL, MLEM")
    
    # Load data
    data = load_example_data()
    
    # Stage 1: Calculate reaction rates
    measured_rates, rate_uncertainties, reaction_ids = calculate_reaction_rates(data["measurements"])
    
    # Stage 2: Build response matrix
    response, groups, reactions = build_response(
        data["boundaries"],
        data["cross_sections"],
        data["number_densities"],
    )
    
    # Stage 3: Unfold spectrum with all three methods
    gls_result = unfold_spectrum_gls(
        response.matrix,
        measured_rates,
        rate_uncertainties,
        data["prior_flux"],
    )
    
    gravel_result = unfold_spectrum_gravel(
        response.matrix,
        measured_rates,
        data["prior_flux"],
    )
    
    mlem_result = unfold_spectrum_mlem(
        response.matrix,
        measured_rates,
        data["prior_flux"],
    )
    
    # Compare results
    compare_solutions(
        data["boundaries"],
        data["prior_flux"],
        gls_result,
        gravel_result,
        mlem_result,
    )
    
    # Stage 4: Save outputs
    save_outputs(
        data["boundaries"],
        reactions,
        gls_result,
        gravel_result,
        mlem_result,
    )
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {repo_root / 'output'}")
    print("\nNext steps:")
    print("  1. Review output JSON files with unfolded spectra")
    print("  2. Compare results with transport calculations (OpenMC/MCNP)")
    print("  3. Generate publication-quality plots")
    print("  4. Perform sensitivity/uncertainty analysis")


if __name__ == "__main__":
    main()
