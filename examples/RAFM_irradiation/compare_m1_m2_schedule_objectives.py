#!/usr/bin/env python3
"""Compare Method 1 (DI-FOM) and Method 2 (FIM-D) schedule rankings on RAFM artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from fluxforge.analysis.optimization_difom import (
    DIFOMLineTerm,
    DIFOMScheduleCandidate,
    rank_difom_schedules,
)
from fluxforge.analysis.optimization_fim import rank_fim_schedules


@dataclass(frozen=True)
class CandidateContext:
    sample_id: str
    sample_group: str
    irradiation_time_s: float
    cooldown_time_s: float
    count_time_s: float


def _load_candidate(
    payload: dict,
    *,
    max_lines: int = 8,
) -> tuple[DIFOMScheduleCandidate, CandidateContext] | None:
    sample_id = str(payload.get("sample_id") or "").strip()
    if not sample_id:
        return None

    timing = payload.get("timing") or {}
    if not isinstance(timing, dict):
        timing = {}

    irradiation_time_s = float(timing.get("irradiation_time_s") or 0.0)
    cooldown_time_s = float(timing.get("decay_time_s") or 0.0)

    count_time_s = float(payload.get("live_time_s") or 1.0)
    if count_time_s <= 0.0:
        count_time_s = 1.0

    peaks = payload.get("peaks") or []
    if not isinstance(peaks, list):
        return None

    sorted_peaks = sorted(
        [peak for peak in peaks if isinstance(peak, dict)],
        key=lambda row: float(row.get("net_counts") or 0.0),
        reverse=True,
    )

    terms: list[DIFOMLineTerm] = []
    for peak in sorted_peaks:
        nuclide = str(peak.get("isotope") or "").strip()
        signal = float(peak.get("net_counts") or 0.0)
        if not nuclide or signal <= 0.0:
            continue
        terms.append(
            DIFOMLineTerm(
                nuclide=nuclide,
                line_energy_keV=float(peak.get("energy_keV") or 0.0),
                signal_counts=signal,
                background_counts=max(float(peak.get("background") or 0.0), 0.0),
                interference_counts=0.0,
            )
        )
        if len(terms) >= max_lines:
            break

    if len(terms) == 0:
        return None

    candidate = DIFOMScheduleCandidate(
        label=sample_id,
        irradiation_time_s=irradiation_time_s,
        cooldown_time_s=cooldown_time_s,
        count_time_s=count_time_s,
        lines=tuple(terms),
    )
    context = CandidateContext(
        sample_id=sample_id,
        sample_group=str(timing.get("sample_group") or "unknown"),
        irradiation_time_s=irradiation_time_s,
        cooldown_time_s=cooldown_time_s,
        count_time_s=count_time_s,
    )
    return candidate, context


def _spearman_rank_correlation(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    labels = sorted(set(rank_a).intersection(rank_b))
    n = len(labels)
    if n < 2:
        return 1.0
    d_squared = 0.0
    for label in labels:
        delta = float(rank_a[label] - rank_b[label])
        d_squared += delta * delta
    return 1.0 - (6.0 * d_squared) / (n * (n * n - 1.0))


def main() -> int:
    root = Path(__file__).resolve().parent
    analysis_root = root / "results" / "analysis_json"
    benchmark_root = root / "results" / "method_benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    candidates: list[DIFOMScheduleCandidate] = []
    context_by_label: dict[str, CandidateContext] = {}

    for path in sorted(analysis_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        loaded = _load_candidate(payload)
        if loaded is None:
            continue
        candidate, context = loaded
        candidates.append(candidate)
        context_by_label[candidate.label] = context

    if len(candidates) == 0:
        raise RuntimeError(f"No candidate schedules were found in {analysis_root}")

    ranked_difom = rank_difom_schedules(candidates)
    ranked_fim = rank_fim_schedules(
        candidates,
        objective="fim-d",
        nuisance_variance_fraction=0.05,
        regularization=1.0e-6,
    )

    difom_rank = {item.label: idx for idx, item in enumerate(ranked_difom, start=1)}
    fim_rank = {item.label: idx for idx, item in enumerate(ranked_fim, start=1)}
    difom_score = {item.label: float(item.total_score) for item in ranked_difom}
    fim_score = {item.label: float(item.objective_score) for item in ranked_fim}

    csv_path = benchmark_root / "m1_m2_schedule_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "sample_group",
                "irradiation_time_s",
                "cooldown_time_s",
                "count_time_s",
                "difom_rank",
                "difom_score",
                "fim_d_rank",
                "fim_d_score",
                "rank_delta_fim_minus_difom",
            ],
        )
        writer.writeheader()
        for label in sorted(context_by_label):
            ctx = context_by_label[label]
            writer.writerow(
                {
                    "sample_id": label,
                    "sample_group": ctx.sample_group,
                    "irradiation_time_s": ctx.irradiation_time_s,
                    "cooldown_time_s": ctx.cooldown_time_s,
                    "count_time_s": ctx.count_time_s,
                    "difom_rank": difom_rank.get(label),
                    "difom_score": difom_score.get(label),
                    "fim_d_rank": fim_rank.get(label),
                    "fim_d_score": fim_score.get(label),
                    "rank_delta_fim_minus_difom": (
                        (fim_rank.get(label) or 0) - (difom_rank.get(label) or 0)
                    ),
                }
            )

    top_k = min(5, len(candidates))
    top_difom = [item.label for item in ranked_difom[:top_k]]
    top_fim = [item.label for item in ranked_fim[:top_k]]
    overlap = len(set(top_difom).intersection(top_fim))
    rho = _spearman_rank_correlation(difom_rank, fim_rank)

    md_path = benchmark_root / "m1_m2_schedule_comparison.md"
    lines = [
        "# M1 vs M2 Schedule Comparison",
        "",
        "Comparison scope: RAFM analysis artifacts in `results/analysis_json/*.json`.",
        "",
        f"- Candidate schedules compared: {len(candidates)}",
        f"- Spearman rank correlation (M1 DI-FOM vs M2 FIM-D): {rho:.4f}",
        f"- Top-{top_k} overlap count: {overlap}",
        "",
        "## Top DI-FOM schedules",
    ]
    lines.extend(f"- {label}" for label in top_difom)
    lines.append("")
    lines.append("## Top FIM-D schedules")
    lines.extend(f"- {label}" for label in top_fim)
    lines.append("")
    lines.append(f"CSV output: `{csv_path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote M1/M2 comparison CSV: {csv_path}")
    print(f"Wrote M1/M2 comparison summary: {md_path}")
    print(f"Spearman rank correlation: {rho:.4f}")
    print(f"Top-{top_k} overlap: {overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
