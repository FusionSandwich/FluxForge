import json
from pathlib import Path

from fluxforge.core.artifacts import extract_payload
from fluxforge.core.response import EnergyGroupStructure, ReactionCrossSection, build_response_matrix
from fluxforge.physics.activation import GammaLineMeasurement, IrradiationSegment, reaction_rate_from_activity, weighted_activity
from fluxforge.solvers.gls import gls_adjust


EXAMPLE_DIR = Path("src/fluxforge/examples/fe_cd_rafm_1")


def test_fe_cd_example_cli_flow():
    boundaries = json.loads((EXAMPLE_DIR / "boundaries.json").read_text())
    cross_sections = json.loads((EXAMPLE_DIR / "cross_sections.json").read_text())
    number_densities = json.loads((EXAMPLE_DIR / "number_densities.json").read_text())
    measurements = extract_payload(json.loads((EXAMPLE_DIR / "measurements.json").read_text()))
    prior_flux = json.loads((EXAMPLE_DIR / "prior_flux.json").read_text())

    groups = EnergyGroupStructure(boundaries)
    reactions = [ReactionCrossSection(reaction_id=r_id, sigma_g=sigma) for r_id, sigma in cross_sections.items()]
    number_density_values = [number_densities[rx.reaction_id] for rx in reactions]
    response = build_response_matrix(reactions, groups, number_density_values)

    segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
    measured_rates = []
    rate_uncertainties = []
    for reaction in measurements["reactions"]:
        gamma_lines = [GammaLineMeasurement(**gl) for gl in reaction["gamma_lines"]]
        activity, _ = weighted_activity(gamma_lines)
        rate_estimate = reaction_rate_from_activity(activity, segments, reaction["half_life_s"])
        measured_rates.append(rate_estimate.rate)
        rate_uncertainties.append(rate_estimate.uncertainty)

    prior_cov = [
        [(0.25 * val) ** 2 if i == j else 0.0 for j, val in enumerate(prior_flux)]
        for i, _ in enumerate(prior_flux)
    ]
    measurement_cov = [
        [(rate_uncertainties[i] ** 2) if i == j else 0.0 for j in range(len(measured_rates))]
        for i in range(len(measured_rates))
    ]

    solution = gls_adjust(response.matrix, measured_rates, measurement_cov, prior_flux, prior_cov)

    assert len(solution.flux) == groups.group_count
    assert all(val >= 0.0 for val in solution.flux)
    assert solution.chi2 >= 0.0
