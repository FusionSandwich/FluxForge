"""Run a full Phase 6 worked example on RAFM LDRD-backed irradiation data."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "RAFM_irradiation"
ANALYSIS_ROOT = EXAMPLE_ROOT / "results" / "analysis_json"
SCHEDULES_PATH = EXAMPLE_ROOT / "metadata" / "sample_schedules.json"
UNFOLD_PATH = EXAMPLE_ROOT / "results" / "unfolding" / "mlem.json"
DEFAULT_SAMPLE_ID = "RAFM4-C_15dEOI"


def default_output_root(sample_id: str = DEFAULT_SAMPLE_ID) -> Path:
    return EXAMPLE_ROOT / "results" / "phase6_ldrd_worked_example" / sample_id


def _sample_letter(sample_id: str) -> str:
    stem = Path(sample_id).stem
    for token in stem.replace("_", "-").split("-"):
        if len(token) == 1 and token.isalpha():
            return token.upper()
    raise ValueError(f"Unable to resolve sample letter from {sample_id!r}.")


def _normalize_nuclide(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    letters = ""
    digits = ""
    for char in text:
        if char.isalpha() and not digits:
            letters += char
        elif char.isdigit():
            digits += char
        else:
            return text
    if not letters or not digits:
        return text
    symbol = letters[0].upper() + letters[1:].lower()
    return f"{symbol}-{digits}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _infer_half_life_s(
    count_time_activity_bq: float,
    irradiation_time_activity_bq: float,
    cooling_time_s: float,
) -> float:
    count_time = max(float(count_time_activity_bq), 0.0)
    irradiation_time = max(float(irradiation_time_activity_bq), 0.0)
    cooling = max(float(cooling_time_s), 0.0)
    if cooling <= 0.0 or count_time <= 0.0 or irradiation_time <= count_time:
        return max(cooling, 1.0) * 1.0e6
    ratio = irradiation_time / count_time
    if ratio <= 1.0:
        return max(cooling, 1.0) * 1.0e6
    return math.log(2.0) * cooling / math.log(ratio)


def _load_analysis(sample_id: str) -> dict[str, Any]:
    path = ANALYSIS_ROOT / f"{sample_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_schedule_entry(sample_id: str) -> dict[str, Any]:
    payload = json.loads(SCHEDULES_PATH.read_text(encoding="utf-8"))
    schedules = payload.get("schedules") or {}
    letter = _sample_letter(sample_id)
    entry = schedules.get(letter)
    if not isinstance(entry, dict):
        raise KeyError(f"Missing schedule metadata for sample {sample_id}.")
    return entry


def _build_activity_review_payload(sample_id: str) -> dict[str, Any]:
    analysis = _load_analysis(sample_id)
    timing = analysis.get("timing") or {}
    cooling_time_s = max(_safe_float(timing.get("decay_time_s")), 0.0)
    measurement_qc = (analysis.get("measurement_qc") or [{}])[0]
    live_time_s = max(_safe_float(measurement_qc.get("live_time_s")), 1.0)

    isotopes = analysis.get("isotopes") or {}
    isotope_summaries: list[dict[str, Any]] = []
    half_lives_s: dict[str, float] = {}
    for raw_nuclide, payload in isotopes.items():
        if not isinstance(payload, dict):
            continue
        nuclide = _normalize_nuclide(str(raw_nuclide))
        if not nuclide:
            continue
        count_time_activity = _safe_float(payload.get("activity_bq"))
        count_time_unc = _safe_float(payload.get("activity_unc_bq"))
        irradiation_activity = max(
            _safe_float(payload.get("activity_eoi_bq")),
            count_time_activity,
        )
        irradiation_unc = max(
            _safe_float(payload.get("activity_eoi_unc_bq")),
            count_time_unc,
        )
        half_life_s = _infer_half_life_s(
            count_time_activity,
            irradiation_activity,
            cooling_time_s,
        )
        half_lives_s[nuclide] = half_life_s
        peak_energies = tuple(
            sorted(
                float(value)
                for value in (payload.get("peak_energies") or [])
                if value is not None
            )
        )
        isotope_summaries.append(
            {
                "nuclide": nuclide,
                "line_count": int(payload.get("n_peaks") or len(peak_energies)),
                "peak_energies_keV": ", ".join(f"{value:.3f}" for value in peak_energies),
                "matched_line_energies_keV": ", ".join(
                    f"{value:.3f}" for value in peak_energies
                ),
                "total_net_counts": float(payload.get("total_net_counts") or 0.0),
                "half_life_s": float(half_life_s),
                "cooling_time_s": float(cooling_time_s),
                "count_time_activity_Bq": float(count_time_activity),
                "count_time_activity_unc_Bq": float(count_time_unc),
                "irradiation_time_activity_Bq": float(irradiation_activity),
                "irradiation_time_activity_unc_Bq": float(irradiation_unc),
                "relative_uncertainty": float(
                    irradiation_unc / max(irradiation_activity, 1.0e-30)
                ),
                "chain_summary": f"{nuclide} activity back-corrected to EOI",
            }
        )

    peaks = [item for item in (analysis.get("peaks") or []) if isinstance(item, dict)]
    line_results: list[dict[str, Any]] = []
    for index, peak in enumerate(peaks, start=1):
        nuclide = _normalize_nuclide(str(peak.get("isotope") or ""))
        if not nuclide:
            continue
        peak_energy = _safe_float(peak.get("energy_keV"))
        net_counts = max(_safe_float(peak.get("net_counts")), 0.0)
        net_counts_unc = max(
            _safe_float(peak.get("net_counts_unc")),
            math.sqrt(max(net_counts, 1.0)),
        )
        count_time_activity = max(_safe_float(peak.get("activity_bq")), 0.0)
        count_time_unc = max(_safe_float(peak.get("activity_unc_bq")), 0.0)
        irradiation_activity = max(_safe_float(peak.get("eoi_activity_bq")), count_time_activity)
        irradiation_unc = max(_safe_float(peak.get("eoi_activity_unc_bq")), count_time_unc)
        line_results.append(
            {
                "peak_id": f"{sample_id}-peak-{index}",
                "nuclide": nuclide,
                "peak_energy_keV": float(peak_energy),
                "matched_line_energy_keV": float(peak_energy),
                "line_delta_keV": 0.0,
                "net_counts": float(net_counts),
                "net_counts_uncertainty": float(net_counts_unc),
                "efficiency": max(_safe_float(peak.get("efficiency"), 1.0e-6), 1.0e-12),
                "efficiency_rel_uncertainty": 0.05,
                "emission_probability": 1.0,
                "emission_probability_uncertainty": 0.0,
                "half_life_s": float(half_lives_s.get(nuclide, 1.0e6)),
                "cooling_time_s": float(cooling_time_s),
                "count_time_activity_Bq": float(count_time_activity),
                "count_time_activity_unc_Bq": float(count_time_unc),
                "irradiation_time_activity_Bq": float(irradiation_activity),
                "irradiation_time_activity_unc_Bq": float(irradiation_unc),
            }
        )

    isotope_summaries.sort(
        key=lambda item: float(item.get("irradiation_time_activity_Bq") or 0.0),
        reverse=True,
    )

    return {
        "schema": "fluxforge.activity_review.v1",
        "source_id": "fluxforge_bundled_gamma",
        "custom_gamma_path": None,
        "spectrum_id": sample_id,
        "sample_group": analysis.get("sample_group"),
        "analysis_json_source": str(ANALYSIS_ROOT / f"{sample_id}.json"),
        "schedule_source": str(SCHEDULES_PATH),
        "live_time_s": float(live_time_s),
        "cooling_time_s": float(cooling_time_s),
        "irradiation_reference": "end_of_irradiation",
        "plot_horizon_s": float(max(cooling_time_s, 7.0 * 24.0 * 3600.0)),
        "line_results": line_results,
        "isotope_summaries": isotope_summaries,
    }


def _build_second_irradiation_inputs(sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule_entry = _load_schedule_entry(sample_id)
    phase1 = schedule_entry.get("phase1") or {}
    phase2 = schedule_entry.get("phase2") or {}

    candidates = {
        "candidates": [
            {
                "label": f"{sample_id}_window_{row.get('label')}",
                "flux_scale": 1.0,
                "duration_factor": 1.0,
                "second_cooling_time_s": float(row.get("seconds") or 0.0),
            }
            for row in phase1.get("cooling_times") or []
            if _safe_float(row.get("seconds")) > 0.0
        ]
    }

    target_weights = {}
    if len(candidates["candidates"]) > 0:
        target_weights = {
            "Cr-51": 1.0,
            "Ta-182": 0.8,
            "Fe-59": 0.6,
        }

    schedule = {
        "sample_id": sample_id,
        "schedule_source": str(SCHEDULES_PATH),
        "first_cooling_time_s": float(
            (phase2.get("cooling_times") or [{"seconds": 0.0}])[0].get("seconds") or 0.0
        ),
        "second_irradiation_time_s": float(phase2.get("irradiation_seconds") or 0.0),
        "target_weights": target_weights,
    }
    return schedule, candidates


def _run_cli(args: list[str]) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    repo_pythonpath = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        repo_pythonpath
        if not existing_pythonpath
        else f"{repo_pythonpath}:{existing_pythonpath}"
    )
    command = [sys.executable, "-m", "fluxforge.cli.app", *args]
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "CLI command failed:\n"
            + " ".join(command)
            + "\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )


def _write_summary(
    sample_id: str,
    output_dir: Path,
    objective_outputs: dict[str, Path],
    masking_output: Path,
    inventory_output: Path,
    second_output: Path,
    ffexp_output: Path,
) -> Path:
    top_rows = []
    for objective, path in objective_outputs.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranked = payload.get("ranked_candidates") or []
        if not ranked:
            continue
        top = ranked[0]
        score = top.get(
            "difom_score",
            top.get("objective_score", top.get("total_score", top.get("total_utility"))),
        )
        top_rows.append((objective, str(top.get("label")), score))

    masking_payload = json.loads(masking_output.read_text(encoding="utf-8"))
    second_payload = json.loads(second_output.read_text(encoding="utf-8"))
    inventory_payload = json.loads(inventory_output.read_text(encoding="utf-8"))
    ffexp_payload = json.loads(ffexp_output.read_text(encoding="utf-8"))

    lines = [
        "# Phase 6 LDRD Worked Example",
        "",
        f"Sample: {sample_id}",
        f"Output root: {output_dir}",
        "",
        "## Objective Winners",
    ]
    for objective, label, score in top_rows:
        lines.append(f"- {objective}: {label} (score={score})")

    lines.extend(
        [
            "",
            "## Additional Outputs",
            f"- Masking candidates: {len(masking_payload.get('line_masking_results') or [])}",
            f"- Inventory time-series rows: {len(inventory_payload.get('time_series_rows') or [])}",
            f"- Second-irradiation candidates: {len(second_payload.get('ranked_candidates') or [])}",
            f"- ffexp format: {ffexp_payload.get('format')}",
            "",
            "## Key Files",
            "- activity_review.json",
            "- inventory_review.json",
            "- masking_review.json",
            "- optimization_di_fom.json",
            "- optimization_fim_d.json",
            "- optimization_mwdcs.json",
            "- optimization_bass_d.json",
            "- optimization_stbd_mr.json",
            "- second_irradiation_plan.json",
            "- benchmark_experimental_bundle.ffexp",
        ]
    )

    summary_path = output_dir / "WORKED_EXAMPLE_SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_phase6_ldrd_worked_example(
    *,
    sample_id: str = DEFAULT_SAMPLE_ID,
    output_root: Path | None = None,
) -> Path:
    output_dir = Path(output_root) if output_root is not None else default_output_root(sample_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    activity_review_payload = _build_activity_review_payload(sample_id)
    activity_review_path = output_dir / "activity_review.json"
    activity_review_path.write_text(
        json.dumps(activity_review_payload, indent=2),
        encoding="utf-8",
    )

    grids = {
        "irradiation_grid_s": "1800,3600,7200",
        "cooldown_grid_s": "360,9645,97565,346788,1385628",
        "count_grid_s": "300,600,900",
    }

    inventory_output = output_dir / "inventory_review.json"
    _run_cli(
        [
            "inventory-review",
            "--activity-review-file",
            str(activity_review_path),
            "--output",
            str(inventory_output),
            "--time-origin",
            "eoi",
            "--time-points-s",
            "0,360,9645,97565,346788,1385628",
            "--observable",
            "activity",
            "--distance-cm",
            "25.0",
            "--top-n",
            "10",
        ]
    )

    masking_output = output_dir / "masking_review.json"
    _run_cli(
        [
            "masking-review",
            "--activity-review-file",
            str(activity_review_path),
            "--output",
            str(masking_output),
            "--energy-window-keV",
            "10.0",
            "--top-n",
            "100",
        ]
    )

    objective_outputs = {
        "di-fom": output_dir / "optimization_di_fom.json",
        "fim-d": output_dir / "optimization_fim_d.json",
        "mwdcs": output_dir / "optimization_mwdcs.json",
        "bass-d": output_dir / "optimization_bass_d.json",
        "stbd-mr": output_dir / "optimization_stbd_mr.json",
    }

    for objective, path in objective_outputs.items():
        cli_args = [
            "optimization-sweep",
            "--activity-review-file",
            str(activity_review_path),
            "--output",
            str(path),
            "--objective",
            objective,
            "--irradiation-grid-s",
            grids["irradiation_grid_s"],
            "--cooldown-grid-s",
            grids["cooldown_grid_s"],
            "--count-grid-s",
            grids["count_grid_s"],
            "--reference-irradiation-time-s",
            "3600",
            "--unfold-file",
            str(UNFOLD_PATH),
        ]
        if objective in {"bass-d", "stbd-mr"}:
            cli_args.append("--enable-advanced-objectives")
        if objective == "mwdcs":
            cli_args.extend(["--mwdcs-full-spectrum-mode", "--mwdcs-overlap-penalty", "0.1"])
        if objective == "stbd-mr":
            cli_args.extend(
                [
                    "--stbdmr-window-offsets-s",
                    "0,3600,21600",
                    "--stbdmr-window-count-time-s",
                    "900",
                    "--stbdmr-masking-regularization",
                    "0.2",
                    "--stbdmr-differentiable-graph",
                ]
            )
        if objective == "bass-d":
            cli_args.extend(
                [
                    "--bassd-dose-weight",
                    "0.03",
                    "--bassd-exploration-temperature",
                    "0.0",
                    "--bassd-seed",
                    "17",
                ]
            )
        _run_cli(cli_args)

    second_schedule, second_candidates = _build_second_irradiation_inputs(sample_id)
    second_schedule_path = output_dir / "second_irradiation_schedule.json"
    second_candidates_path = output_dir / "second_irradiation_candidates.json"
    second_schedule_path.write_text(json.dumps(second_schedule, indent=2), encoding="utf-8")
    second_candidates_path.write_text(
        json.dumps(second_candidates, indent=2),
        encoding="utf-8",
    )

    second_output = output_dir / "second_irradiation_plan.json"
    _run_cli(
        [
            "second-irradiation-plan",
            "--inventory-file",
            str(activity_review_path),
            "--schedule-file",
            str(second_schedule_path),
            "--candidates-file",
            str(second_candidates_path),
            "--output",
            str(second_output),
        ]
    )

    ffexp_output = output_dir / "benchmark_experimental_bundle.ffexp"
    _run_cli(
        [
            "ffexp-export",
            "--activity-review-file",
            str(activity_review_path),
            "--inventory-review-file",
            str(inventory_output),
            "--masking-file",
            str(masking_output),
            "--optimization-file",
            str(objective_outputs["di-fom"]),
            "--second-irradiation-file",
            str(second_output),
            "--plot-paths",
            "inventory_review_activity.png,optimization_di_fom_optimization_grid.csv",
            "--output",
            str(ffexp_output),
        ]
    )

    return _write_summary(
        sample_id,
        output_dir,
        objective_outputs,
        masking_output,
        inventory_output,
        second_output,
        ffexp_output,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6 LDRD worked example using RAFM analysis data.",
    )
    parser.add_argument(
        "--sample-id",
        default=DEFAULT_SAMPLE_ID,
        help="Sample ID from examples/RAFM_irradiation/results/analysis_json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(DEFAULT_SAMPLE_ID),
        help="Output directory for worked-example artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = run_phase6_ldrd_worked_example(
        sample_id=args.sample_id,
        output_root=args.output_root,
    )
    print(f"Wrote Phase 6 LDRD worked example artifacts to {args.output_root}")
    print(f"Summary: {summary_path}")
    return 0


__all__ = [
    "DEFAULT_SAMPLE_ID",
    "default_output_root",
    "run_phase6_ldrd_worked_example",
    "main",
]
