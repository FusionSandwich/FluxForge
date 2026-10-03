"""Replay supplied RAFM uncertainty through generic and ASTM activation paths.

Inputs are read-only. QG headers retain unknown total uncertainty scope. Line
checks condition on fixed unit efficiency/yield: they test propagation using
real supplied count sigmas, not physical activities or independent QG truth.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import fluxforge
from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan
from fluxforge.analysis.astm_e262 import analyze_astm_e262_plan
from fluxforge.examples.rafm_workflow import (
    load_rafm_example_metadata,
    resolve_measurement_timing,
    parse_iso_datetime,
)
from fluxforge.io.flux_wire import read_processed_txt
from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    reaction_rate_from_activity,
)
from fluxforge.uncertainty.reaction_rate_budget import (
    RateUncertaintyBudget,
    UncertaintyComponent,
    rate_covariance,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def buildup(segments, decay):
    """Independent chronological activation integral, no production helper."""
    result = 0.0
    for duration, power in segments:
        result = result * math.exp(-decay * duration) + power * (
            -math.expm1(-decay * duration)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if not Path(fluxforge.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError("Import must come from the replay worktree src")
    if out.exists():
        raise FileExistsError(out)
    example = root / "examples/RAFM_irradiation"
    metadata = load_rafm_example_metadata(example)
    inputs = {
        str(p.relative_to(root)): sha(p)
        for p in sorted((example / "metadata").glob("*.json"))
    }
    headers, lines, report_names = [], [], []
    errors = []
    for path in sorted((example / "QG_processed_gamma_data/flux_wires").glob("*.txt")):
        inputs[str(path.relative_to(root))] = sha(path)
        report_names.append(path.stem)
        report = read_processed_txt(path)
        timing = resolve_measurement_timing(path.stem, report.start_time, metadata)
        cooldown = timing.decay_time_s
        duration = timing.irradiation_time_s
        timing_source = timing.schedule_source
        if cooldown is None or not duration:
            # Some wire filenames lack a direct schedule join. Use only an
            # explicitly labeled conditional common-phase/date subtraction.
            phase = metadata.sample_schedule["irradiation_info"]["phase2"]
            end = parse_iso_datetime(phase["end"])
            if report.start_time is None or end is None:
                raise ValueError(f"Missing conditional timing: {path.stem}")
            duration = phase["duration_seconds"]
            cooldown = (report.start_time - end).total_seconds()
            timing_source = (
                "conditional common phase2 + naive report datetime; unqualified join"
            )
        history = timing.irradiation_history or [(duration, 1.0)]
        segments = [IrradiationSegment(duration, power) for duration, power in history]
        for nuclide in report.nuclides:
            decay = math.log(2) / nuclide.half_life_seconds
            factor = buildup(history, decay)
            count_factor = -math.expm1(-decay * report.real_time) / (
                decay * report.real_time
            )
            correction = math.exp(decay * cooldown)
            if not metadata.config["qg_report_activity_includes_count_decay"]:
                correction /= count_factor
            activity = nuclide.activity_bq * correction
            sigma = nuclide.activity_unc_bq * correction
            result = reaction_rate_from_activity(
                activity,
                segments,
                nuclide.half_life_seconds,
                activity_uncertainty_bq=sigma,
            )
            expected_rate, expected_sigma = activity / factor, sigma / factor
            np.testing.assert_allclose(
                [result.rate, result.uncertainty],
                [expected_rate, expected_sigma],
                rtol=2e-12,
            )
            headers.append(
                dict(
                    sample=path.stem,
                    isotope=nuclide.isotope,
                    report_activity_Bq=nuclide.activity_bq,
                    report_activity_sigma_Bq=nuclide.activity_unc_bq,
                    activity_reference="bundled count-decay setting and cooling schedule",
                    reported_total_scope="unknown",
                    reaction_rate_s=result.rate,
                    timing_source=timing_source,
                    cooling_time_s=cooldown,
                    reaction_rate_sigma_s=result.uncertainty,
                    expected_sigma_s=expected_sigma,
                    legacy_invented_sigma_s=result.rate
                    / math.sqrt(max(activity, 1e-12)),
                )
            )
            for peak in nuclide.peaks:
                count, count_sigma = peak["net_counts"], peak["net_unc"]
                # Unit efficiency/yield deliberately avoids inferring unqualified
                # detector calibration or trusting unspecified RAD intensity units.
                scale = math.exp(decay * cooldown) / (report.live_time * count_factor)
                expected_activity, expected_activity_sigma = (
                    count * scale,
                    count_sigma * scale,
                )
                kwargs = dict(
                    net_counts=count,
                    net_counts_unc=count_sigma,
                    live_time_s=report.live_time,
                    real_time_s=report.real_time,
                    efficiency=1.0,
                    gamma_intensity=1.0,
                    half_life_s=nuclide.half_life_seconds,
                    cooling_time_s=cooldown,
                )
                measurement = GammaLineMeasurement(**kwargs)
                np.testing.assert_allclose(
                    [
                        measurement.activity_at_reference(),
                        measurement.activity_uncertainty_at_reference(),
                    ],
                    [expected_activity, expected_activity_sigma],
                    rtol=2e-12,
                )
                row = dict(
                    kwargs,
                    sample_mass_g=1.0,
                    atomic_mass_g_mol=1.0,
                    effective_cross_section_barn=1.0,
                    sigma_0_barn=1.0,
                )
                plan = dict(
                    irradiation={
                        "segments": [
                            dict(duration_s=d, relative_power=p) for d, p in history
                        ]
                    },
                    measurements=[row],
                )
                astm = {}
                for name, analyze in [
                    ("E261", analyze_astm_e261_plan),
                    ("E262", analyze_astm_e262_plan),
                ]:
                    item = analyze(plan)["measurements"][0]
                    np.testing.assert_allclose(
                        [item["activity_eoi_unc_Bq"], item["reaction_rate_unc_s"]],
                        [expected_activity_sigma, expected_activity_sigma / factor],
                        rtol=2e-12,
                    )
                    astm[name] = item["reaction_rate_unc_s"]
                observed = measurement.activity_uncertainty_at_reference()
                errors.append(abs(observed / expected_activity_sigma - 1))
                lines.append(
                    dict(
                        sample=path.stem,
                        isotope=nuclide.isotope,
                        energy_keV=peak["center_keV"],
                        supplied_net_counts=count,
                        supplied_net_sigma=count_sigma,
                        sigma_over_poisson=(
                            count_sigma / math.sqrt(count) if count else None
                        ),
                        activity_sigma_conditioned_Bq=observed,
                        expected_activity_sigma_conditioned_Bq=expected_activity_sigma,
                        astm_rate_sigmas_s=astm,
                        source_file_sha256=inputs[str(path.relative_to(root))],
                        source_line_number=peak.get("source_line_number"),
                    )
                )
    # Existing source covariance algebra, with independent expected signed block.
    C = np.array([[4.0, 0.5], [0.5, 9.0]])
    J = np.array([[0.2, -0.3], [-0.4, 0.1]])
    rates = np.array([10.0, 20.0])
    expected = rates[:, None] * (J @ C @ J.T) * rates[None, :]
    covariance_cases = []
    for units in ([1.0, 1.0], [1e-8, 1e8], [3600.0, 0.01]):
        d = np.asarray(units)
        budgets = [
            RateUncertaintyBudget(
                str(i),
                float(rate),
                [
                    UncertaintyComponent.from_covariance(
                        "calibration",
                        C * d[:, None] * d[None, :],
                        jac / d,
                        input_names=["a", "b"],
                        input_units=["scaled_a", "scaled_b"],
                        source="analytic conditional probe",
                        correlation_group="shared",
                        uncertainty_scope="conditional",
                    )
                ],
            )
            for i, (rate, jac) in enumerate(zip(rates, J))
        ]
        actual = rate_covariance(budgets)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)
        covariance_cases.append(dict(units=units, covariance=actual.tolist()))
    frozen = root / "artifacts/validation/registry_uncertainty_20261001/receipt.json"
    inputs[str(frozen.relative_to(root))] = sha(frozen)
    receipt = json.loads(frozen.read_text())
    payload = dict(
        scientific_admission=False,
        scope="Propagation diagnostics conditioned on bundled schedule and declared fixed inputs; QG is not physical truth",
        fixed_line_inputs={
            "efficiency": 1.0,
            "gamma_intensity": 1.0,
            "target_mass_g": 1.0,
            "molar_mass_g_mol": 1.0,
            "cross_section_barn": 1.0,
        },
        caveats=[
            "Header sigma is reported total of unknown composition; line sigma is never added to header sigma.",
            "Schedule, count-decay setting, calibration, nuclear-data and material covariance remain unqualified.",
            "Unit detector/yield inputs make line outputs conditioned diagnostics, not sample activities.",
        ],
        metrics=dict(
            qg_reports=len(report_names),
            header_observations=len(headers),
            line_observations=len(lines),
            maximum_line_sigma_relative_error=max(errors),
        ),
        input_sha256=inputs,
        runtime_source_sha256={
            str(p.relative_to(root)): sha(p)
            for p in sorted((root / "src/fluxforge").rglob("*.py"))
        },
        tool_sha256=sha(Path(__file__).resolve()),
        header_comparisons=headers,
        line_comparisons=lines,
        source_covariance_unit_cases=covariance_cases,
        frozen_response_evidence=dict(
            shape=list(np.asarray(receipt["row_cross_sections_barn"]).shape),
            rate_sigmas=receipt["replay_rate_uncertainties"],
            uncertainty_scope="Frozen diagonal input; no qualified full physical rate covariance",
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["metrics"]))


if __name__ == "__main__":
    main()
