"""Command-line interface for FluxForge using argparse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from fluxforge.core.response import EnergyGroupStructure, ReactionCrossSection, build_response_matrix
from fluxforge.physics.activation import GammaLineMeasurement, IrradiationSegment, reaction_rate_from_activity, weighted_activity
from fluxforge.solvers.gls import gls_adjust


def _load_json(path: Path):
    return json.loads(path.read_text())


def cmd_build_response(args: argparse.Namespace) -> None:
    boundaries = [float(x) for x in _load_json(args.boundaries_file)]
    groups = EnergyGroupStructure(boundaries)
    cross_sections_raw = _load_json(args.cross_section_file)
    number_densities = _load_json(args.number_densities_file)

    reactions = []
    number_density_values: List[float] = []
    for reaction_id, sigma in cross_sections_raw.items():
        reactions.append(ReactionCrossSection(reaction_id=reaction_id, sigma_g=[float(s) for s in sigma]))
        number_density_values.append(float(number_densities[reaction_id]))

    response = build_response_matrix(reactions, groups, number_density_values)
    output_data = {
        "matrix": response.matrix,
        "reactions": response.reactions,
        "boundaries": response.energy_groups.boundaries_eV,
    }
    Path(args.output).write_text(json.dumps(output_data, indent=2))
    print(f"Wrote response matrix to {args.output}")


def cmd_infer_spectrum(args: argparse.Namespace) -> None:
    response_data = _load_json(args.response_file)
    response_matrix = response_data["matrix"]
    boundaries = response_data["boundaries"]
    reactions = response_data["reactions"]
    groups = EnergyGroupStructure([float(b) for b in boundaries])

    measurements_data = _load_json(args.measurements_file)
    half_life_map = {rx["reaction_id"]: float(rx["half_life_s"]) for rx in measurements_data["reactions"]}
    segments = [IrradiationSegment(**seg) for seg in measurements_data["segments"]]

    measured_rates: List[float] = []
    rate_uncertainties: List[float] = []
    for reaction in measurements_data["reactions"]:
        gamma_lines = [GammaLineMeasurement(**line) for line in reaction["gamma_lines"]]
        activity, _ = weighted_activity(gamma_lines)
        rate_estimate = reaction_rate_from_activity(activity, segments, half_life_map[reaction["reaction_id"]])
        measured_rates.append(rate_estimate.rate)
        rate_uncertainties.append(rate_estimate.uncertainty)

    if args.prior_flux_file:
        prior_flux = [float(v) for v in _load_json(args.prior_flux_file)]
    else:
        avg_response = sum(sum(row) for row in response_matrix) / max(len(response_matrix) * len(response_matrix[0]), 1)
        prior_flux = [sum(measured_rates) / max(avg_response, 1e-12) for _ in range(groups.group_count)]
    prior_cov = [[(args.prior_uncertainty * val) ** 2 if i == j else 0.0 for j in range(groups.group_count)] for i, val in enumerate(prior_flux)]
    measurement_cov = [[(rate_uncertainties[i] ** 2) if i == j else 0.0 for j in range(len(measured_rates))] for i in range(len(measured_rates))]

    solution = gls_adjust(response_matrix, measured_rates, measurement_cov, prior_flux, prior_cov)
    result = {
        "boundaries_eV": boundaries,
        "reactions": reactions,
        "flux": solution.flux,
        "covariance": solution.covariance,
        "chi2": solution.chi2,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"Saved inferred spectrum to {args.output}")


def cmd_validate(args: argparse.Namespace) -> None:
    response_data = _load_json(args.response_file)
    r = response_data["matrix"]
    phi_true = [float(v) for v in _load_json(args.truth_flux_file)]
    y = [sum(row[i] * phi_true[i] for i in range(len(phi_true))) for row in r]
    noise = [max(abs(val) * args.noise_fraction, 1e-12) for val in y]
    cy = [[(noise[i] ** 2) if i == j else 0.0 for j in range(len(y))] for i in range(len(y))]
    prior = [v * 0.8 for v in phi_true]
    c0 = [[(args.noise_fraction * prior[i]) ** 2 if i == j else 0.0 for j in range(len(prior))] for i in range(len(prior))]
    solution = gls_adjust(r, y, cy, prior, c0)
    mae = sum(abs(a - b) / max(b, 1e-12) for a, b in zip(solution.flux, phi_true)) / len(phi_true)
    print(f"Validation chi2: {solution.chi2:.2f}")
    print(f"Mean absolute percent error: {mae * 100:.2f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UWNR Flux-Wire–Driven Neutron Spectrum Reconstruction Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-response", help="Build response matrix from cross sections")
    build.add_argument("--cross-section-file", type=Path, required=True)
    build.add_argument("--number-densities-file", type=Path, required=True)
    build.add_argument("--boundaries-file", type=Path, required=True)
    build.add_argument("--output", type=Path, default=Path("response.json"))
    build.set_defaults(func=cmd_build_response)

    infer = subparsers.add_parser("infer-spectrum", help="Infer spectrum using GLS")
    infer.add_argument("--measurements-file", type=Path, required=True)
    infer.add_argument("--response-file", type=Path, required=True)
    infer.add_argument("--prior-flux-file", type=Path)
    infer.add_argument("--prior-uncertainty", type=float, default=0.25)
    infer.add_argument("--output", type=Path, default=Path("spectrum.json"))
    infer.set_defaults(func=cmd_infer_spectrum)

    validate = subparsers.add_parser("validate", help="Run synthetic validation")
    validate.add_argument("--response-file", type=Path, required=True)
    validate.add_argument("--truth-flux-file", type=Path, required=True)
    validate.add_argument("--noise-fraction", type=float, default=0.05)
    validate.set_defaults(func=cmd_validate)

    subparsers.add_parser("report", help="Report generation placeholder")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        print("Report generation not yet implemented. Track inputs and outputs manually for now.")


if __name__ == "__main__":
    main()
