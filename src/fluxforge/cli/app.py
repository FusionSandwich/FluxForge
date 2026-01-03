"""Command-line interface for FluxForge using argparse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from fluxforge.analysis.segmented_detection import SegmentedDetectionConfig, detect_peaks_segmented
from fluxforge.core.response import EnergyGroupStructure, ReactionCrossSection, build_response_matrix
from fluxforge.io.artifacts import (
    read_line_activities,
    read_peak_report,
    read_reaction_rates,
    read_response_bundle,
    read_spectrum_file,
    read_unfold_result,
    read_validation_bundle,
    write_line_activities,
    write_peak_report,
    write_reaction_rates,
    write_report_bundle,
    write_response_bundle,
    write_spectrum_file,
    write_unfold_result,
    write_validation_bundle,
)
from fluxforge.io.genie import read_genie_spectrum
from fluxforge.io.spe import GammaSpectrum, read_spe_file
from fluxforge.physics.activation import IrradiationSegment, reaction_rate_from_activity
from fluxforge.solvers.gls import gls_adjust


def _load_json(path: Path):
    return json.loads(path.read_text())


def cmd_ingest(args: argparse.Namespace) -> None:
    input_path = args.input
    suffix = input_path.suffix.lower()

    if suffix in {".spe"}:
        spectrum = read_spe_file(input_path)
    elif suffix in {".asc", ".txt"}:
        spectrum = read_genie_spectrum(input_path)
    else:
        payload = read_spectrum_file(input_path)
        spectrum = GammaSpectrum.from_dict(payload["spectrum"])

    write_spectrum_file(args.output, spectrum, source_path=input_path)
    print(f"Wrote spectrum file to {args.output}")


def cmd_peaks(args: argparse.Namespace) -> None:
    spectrum_payload = read_spectrum_file(args.spectrum_file)
    spectrum = GammaSpectrum.from_dict(spectrum_payload["spectrum"])
    energies = spectrum.energies if spectrum.energies is not None else spectrum.channels.astype(float)
    if args.sensitivity == "sensitive":
        config = SegmentedDetectionConfig.sensitive()
    elif args.sensitivity == "conservative":
        config = SegmentedDetectionConfig.conservative()
    else:
        config = SegmentedDetectionConfig()

    peaks = detect_peaks_segmented(
        spectrum.channels,
        energies,
        spectrum.counts,
        config=config,
        refine_centroids=True,
        fit_window=args.fit_window,
    )
    peak_payload = [
        {
            "channel": peak.channel,
            "energy_keV": peak.energy_keV,
            "amplitude": peak.amplitude,
            "raw_counts": peak.raw_counts,
            "sigma_keV": peak.sigma_keV,
            "area": peak.area,
            "region": peak.region,
            "is_report": peak.is_report,
            "report_isotope": peak.report_isotope,
            "report_file": peak.report_file,
        }
        for peak in peaks
    ]

    write_peak_report(
        args.output,
        spectrum_id=spectrum.spectrum_id,
        live_time_s=spectrum.live_time,
        peaks=peak_payload,
        source_path=args.spectrum_file,
    )
    print(f"Wrote peak report to {args.output}")


def cmd_activity(args: argparse.Namespace) -> None:
    peak_report = read_peak_report(args.peaks_file)
    live_time_s = peak_report.get("live_time_s") or args.live_time_s
    if live_time_s is None:
        raise ValueError("Live time is required to compute activities.")

    lines = []
    for idx, peak in enumerate(peak_report["peaks"]):
        net_counts = peak.get("area") or peak.get("raw_counts") or peak.get("amplitude")
        efficiency = args.efficiency
        emission_probability = args.emission_probability
        activity = net_counts / max(efficiency * emission_probability * live_time_s, 1e-12)
        activity_unc = activity / np.sqrt(max(net_counts, 1e-12))
        isotope = peak.get("report_isotope") or args.isotope or "unknown"
        reaction_id = args.reaction_id or isotope or f"reaction_{idx + 1}"
        lines.append(
            {
                "energy_keV": peak["energy_keV"],
                "isotope": isotope,
                "reaction_id": reaction_id,
                "net_counts": net_counts,
                "activity_Bq": activity,
                "activity_unc_Bq": activity_unc,
                "efficiency": efficiency,
                "emission_probability": emission_probability,
                "half_life_s": args.half_life_s,
            }
        )

    write_line_activities(
        args.output,
        spectrum_id=peak_report.get("spectrum_id", ""),
        lines=lines,
        source_path=args.peaks_file,
    )
    print(f"Wrote line activities to {args.output}")


def cmd_rates(args: argparse.Namespace) -> None:
    line_payload = read_line_activities(args.lines_file)
    segments = None
    if args.segments_file:
        segments = _load_json(args.segments_file)
    if segments is None:
        segments = [{"duration_s": args.duration_s, "relative_power": 1.0}]

    segment_objs = [IrradiationSegment(**seg) for seg in segments]
    rates = []
    for idx, line in enumerate(line_payload["lines"]):
        half_life_s = line.get("half_life_s", args.half_life_s)
        rate_estimate = reaction_rate_from_activity(line["activity_Bq"], segment_objs, half_life_s)
        reaction_id = line.get("reaction_id") or line.get("isotope") or f"reaction_{idx + 1}"
        rates.append(
            {
                "reaction_id": reaction_id,
                "rate": rate_estimate.rate,
                "uncertainty": rate_estimate.uncertainty,
                "half_life_s": half_life_s,
            }
        )

    write_reaction_rates(
        args.output,
        rates=rates,
        segments=segments,
        source_path=args.lines_file,
    )
    print(f"Wrote reaction rates to {args.output}")


def cmd_response(args: argparse.Namespace) -> None:
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
    write_response_bundle(
        args.output,
        matrix=response.matrix,
        reactions=response.reactions,
        boundaries_eV=response.energy_groups.boundaries_eV,
        source_path=args.cross_section_file,
    )
    print(f"Wrote response bundle to {args.output}")


def cmd_unfold(args: argparse.Namespace) -> None:
    response_data = read_response_bundle(args.response_file)
    response_matrix = response_data["matrix"]
    boundaries = response_data["boundaries_eV"]
    reactions = response_data["reactions"]
    groups = EnergyGroupStructure([float(b) for b in boundaries])

    rates_payload = read_reaction_rates(args.rates_file)
    measured_rates = [float(rx["rate"]) for rx in rates_payload["rates"]]
    rate_uncertainties = [float(rx["uncertainty"]) for rx in rates_payload["rates"]]

    if args.prior_flux_file:
        prior_flux = [float(v) for v in _load_json(args.prior_flux_file)]
    else:
        avg_response = sum(sum(row) for row in response_matrix) / max(len(response_matrix) * len(response_matrix[0]), 1)
        prior_flux = [sum(measured_rates) / max(avg_response, 1e-12) for _ in range(groups.group_count)]
    prior_cov = [
        [(args.prior_uncertainty * val) ** 2 if i == j else 0.0 for j in range(groups.group_count)]
        for i, val in enumerate(prior_flux)
    ]
    measurement_cov = [
        [(rate_uncertainties[i] ** 2) if i == j else 0.0 for j in range(len(measured_rates))]
        for i in range(len(measured_rates))
    ]

    solution = gls_adjust(response_matrix, measured_rates, measurement_cov, prior_flux, prior_cov)
    write_unfold_result(
        args.output,
        boundaries_eV=boundaries,
        reactions=reactions,
        flux=solution.flux,
        covariance=solution.covariance,
        chi2=solution.chi2,
        method="gls",
        source_path=args.rates_file,
    )
    print(f"Saved unfolded spectrum to {args.output}")


def cmd_compare(args: argparse.Namespace) -> None:
    unfold_data = read_unfold_result(args.unfold_file)
    predicted_flux = unfold_data["flux"]
    truth_flux = [float(v) for v in _load_json(args.truth_flux_file)]
    residuals = [p - t for p, t in zip(predicted_flux, truth_flux)]
    mae = sum(abs(r) for r in residuals) / max(len(residuals), 1)
    rmse = np.sqrt(sum(r * r for r in residuals) / max(len(residuals), 1))
    metrics = {
        "mae": mae,
        "rmse": float(rmse),
        "chi2": unfold_data.get("chi2"),
    }
    write_validation_bundle(
        args.output,
        metrics=metrics,
        truth_flux=truth_flux,
        predicted_flux=predicted_flux,
        residuals=residuals,
        source_path=args.unfold_file,
    )
    print(f"Wrote validation bundle to {args.output}")


def cmd_report(args: argparse.Namespace) -> None:
    inputs = {}
    summary = {}
    if args.spectrum_file:
        inputs["spectrum_file"] = str(args.spectrum_file)
    if args.peaks_file:
        inputs["peaks_file"] = str(args.peaks_file)
        peak_report = read_peak_report(args.peaks_file)
        summary["peak_count"] = len(peak_report.get("peaks", []))
    if args.lines_file:
        inputs["lines_file"] = str(args.lines_file)
        line_payload = read_line_activities(args.lines_file)
        summary["line_count"] = len(line_payload.get("lines", []))
    if args.rates_file:
        inputs["rates_file"] = str(args.rates_file)
        rates_payload = read_reaction_rates(args.rates_file)
        summary["rate_count"] = len(rates_payload.get("rates", []))
    if args.unfold_file:
        inputs["unfold_file"] = str(args.unfold_file)
        unfold_payload = read_unfold_result(args.unfold_file)
        summary["chi2"] = unfold_payload.get("chi2")
    if args.validation_file:
        inputs["validation_file"] = str(args.validation_file)
        validation_payload = read_validation_bundle(args.validation_file)
        summary["mae"] = validation_payload.get("metrics", {}).get("mae")

    write_report_bundle(args.output, summary=summary, inputs=inputs or None)
    print(f"Wrote report bundle to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UWNR Flux-Wire–Driven Neutron Spectrum Reconstruction Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest spectrum files into schema artifacts")
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--output", type=Path, default=Path("spectrum.json"))
    ingest.set_defaults(func=cmd_ingest)

    peaks = subparsers.add_parser("peaks", help="Detect peaks from a spectrum artifact")
    peaks.add_argument("--spectrum-file", type=Path, required=True)
    peaks.add_argument("--output", type=Path, default=Path("peaks.json"))
    peaks.add_argument("--sensitivity", choices=["default", "sensitive", "conservative"], default="default")
    peaks.add_argument("--fit-window", type=int, default=6)
    peaks.set_defaults(func=cmd_peaks)

    activity = subparsers.add_parser("activity", help="Compute line activities from peak report")
    activity.add_argument("--peaks-file", type=Path, required=True)
    activity.add_argument("--output", type=Path, default=Path("activities.json"))
    activity.add_argument("--live-time-s", type=float)
    activity.add_argument("--efficiency", type=float, default=1.0)
    activity.add_argument("--emission-probability", type=float, default=1.0)
    activity.add_argument("--half-life-s", type=float, default=1.0)
    activity.add_argument("--isotope", type=str)
    activity.add_argument("--reaction-id", type=str)
    activity.set_defaults(func=cmd_activity)

    rates = subparsers.add_parser("rates", help="Compute reaction rates from line activities")
    rates.add_argument("--lines-file", type=Path, required=True)
    rates.add_argument("--segments-file", type=Path)
    rates.add_argument("--duration-s", type=float, default=1.0)
    rates.add_argument("--half-life-s", type=float, default=1.0)
    rates.add_argument("--output", type=Path, default=Path("rates.json"))
    rates.set_defaults(func=cmd_rates)

    response = subparsers.add_parser("response", help="Build response matrix from cross sections")
    response.add_argument("--cross-section-file", type=Path, required=True)
    response.add_argument("--number-densities-file", type=Path, required=True)
    response.add_argument("--boundaries-file", type=Path, required=True)
    response.add_argument("--output", type=Path, default=Path("response.json"))
    response.set_defaults(func=cmd_response)

    unfold = subparsers.add_parser("unfold", help="Infer spectrum using GLS")
    unfold.add_argument("--rates-file", type=Path, required=True)
    unfold.add_argument("--response-file", type=Path, required=True)
    unfold.add_argument("--prior-flux-file", type=Path)
    unfold.add_argument("--prior-uncertainty", type=float, default=0.25)
    unfold.add_argument("--output", type=Path, default=Path("spectrum.json"))
    unfold.set_defaults(func=cmd_unfold)

    compare = subparsers.add_parser("compare", help="Compare unfolded spectrum with reference")
    compare.add_argument("--unfold-file", type=Path, required=True)
    compare.add_argument("--truth-flux-file", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("validation.json"))
    compare.set_defaults(func=cmd_compare)

    report = subparsers.add_parser("report", help="Compile a report bundle from artifacts")
    report.add_argument("--spectrum-file", type=Path)
    report.add_argument("--peaks-file", type=Path)
    report.add_argument("--lines-file", type=Path)
    report.add_argument("--rates-file", type=Path)
    report.add_argument("--unfold-file", type=Path)
    report.add_argument("--validation-file", type=Path)
    report.add_argument("--output", type=Path, default=Path("report.json"))
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        print("No command provided.")


if __name__ == "__main__":
    main()
