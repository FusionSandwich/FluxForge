#!/usr/bin/env python3
"""Compare M1/M2/M3 objective rankings against the legacy scheduler objective."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from fluxforge.analysis.optimization_difom import (
    DIFOMLineTerm,
    DIFOMScheduleCandidate,
    rank_difom_schedules,
)
from fluxforge.analysis.optimization_fim import rank_fim_schedules
from fluxforge.analysis.optimization_mwdcs import (
    candidate_from_difom_candidate,
    rank_mwdcs_schedules,
)


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


def _load_legacy_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("legacy_schedule_optimizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy module from {path}")
    module = importlib.util.module_from_spec(spec)
    legacy_parent = str(path.parent)
    inserted = False
    if legacy_parent not in sys.path:
        sys.path.insert(0, legacy_parent)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted and legacy_parent in sys.path:
            sys.path.remove(legacy_parent)
    return module


def _legacy_score(candidate: DIFOMScheduleCandidate, legacy_module: ModuleType) -> float:
    z_score = getattr(legacy_module, "z_score", None)
    if not callable(z_score):
        raise RuntimeError("Legacy schedule optimizer module does not expose z_score().")
    total = 0.0
    for line in candidate.lines:
        signal = max(float(line.signal_counts), 0.0)
        noise = max(float(line.background_counts + line.interference_counts), 0.0)
        total += float(z_score(signal, noise))
    return float(total)


def _rank_from_scores(scores: dict[str, float]) -> dict[str, int]:
    sorted_rows = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {label: index for index, (label, _score) in enumerate(sorted_rows, start=1)}


def main() -> int:
    root = Path(__file__).resolve().parent
    analysis_root = root / "results" / "analysis_json"
    benchmark_root = root / "results" / "method_benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    default_legacy_path = (
        Path(__file__).resolve().parents[2].parent
        / "rafm_irradiation_ldrd_copy"
        / "scripts"
        / "schedule_optimizer.py"
    )
    if not default_legacy_path.exists():
        raise FileNotFoundError(
            "Legacy optimizer script not found. "
            f"Expected: {default_legacy_path}"
        )
    legacy_module = _load_legacy_module(default_legacy_path)

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
    mwdcs_candidates = tuple(
        candidate_from_difom_candidate(
            candidate,
            window_offsets_s=(0.0, 3600.0, 21600.0),
            window_count_time_s=max(candidate.count_time_s, 1.0),
        )
        for candidate in candidates
    )
    ranked_mwdcs = rank_mwdcs_schedules(
        mwdcs_candidates,
        full_spectrum_mode=True,
        overlap_penalty=0.1,
    )

    difom_rank = {item.label: idx for idx, item in enumerate(ranked_difom, start=1)}
    fim_rank = {item.label: idx for idx, item in enumerate(ranked_fim, start=1)}
    mwdcs_rank = {item.label: idx for idx, item in enumerate(ranked_mwdcs, start=1)}

    difom_score = {item.label: float(item.total_score) for item in ranked_difom}
    fim_score = {item.label: float(item.objective_score) for item in ranked_fim}
    mwdcs_score = {item.label: float(item.total_score) for item in ranked_mwdcs}
    legacy_score = {
        candidate.label: _legacy_score(candidate, legacy_module)
        for candidate in candidates
    }
    legacy_rank = _rank_from_scores(legacy_score)

    csv_path = benchmark_root / "m1_m2_m3_legacy_schedule_comparison.csv"
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
                "mwdcs_rank",
                "mwdcs_score",
                "legacy_rank",
                "legacy_score",
                "rank_delta_m1_minus_legacy",
                "rank_delta_m2_minus_legacy",
                "rank_delta_m3_minus_legacy",
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
                    "mwdcs_rank": mwdcs_rank.get(label),
                    "mwdcs_score": mwdcs_score.get(label),
                    "legacy_rank": legacy_rank.get(label),
                    "legacy_score": legacy_score.get(label),
                    "rank_delta_m1_minus_legacy": (
                        (difom_rank.get(label) or 0) - (legacy_rank.get(label) or 0)
                    ),
                    "rank_delta_m2_minus_legacy": (
                        (fim_rank.get(label) or 0) - (legacy_rank.get(label) or 0)
                    ),
                    "rank_delta_m3_minus_legacy": (
                        (mwdcs_rank.get(label) or 0) - (legacy_rank.get(label) or 0)
                    ),
                }
            )

    top_k = min(5, len(candidates))
    top_difom = [item.label for item in ranked_difom[:top_k]]
    top_fim = [item.label for item in ranked_fim[:top_k]]
    top_mwdcs = [item.label for item in ranked_mwdcs[:top_k]]
    top_legacy = [
        label
        for label, _score in sorted(
            legacy_score.items(), key=lambda row: row[1], reverse=True
        )[:top_k]
    ]

    corr_m1_legacy = _spearman_rank_correlation(difom_rank, legacy_rank)
    corr_m2_legacy = _spearman_rank_correlation(fim_rank, legacy_rank)
    corr_m3_legacy = _spearman_rank_correlation(mwdcs_rank, legacy_rank)

    overlap_m1_legacy = len(set(top_difom).intersection(top_legacy))
    overlap_m2_legacy = len(set(top_fim).intersection(top_legacy))
    overlap_m3_legacy = len(set(top_mwdcs).intersection(top_legacy))

    md_path = benchmark_root / "m1_m2_m3_legacy_schedule_comparison.md"
    lines = [
        "# M1 vs M2 vs M3 vs Legacy Schedule Comparison",
        "",
        "Comparison scope: RAFM analysis artifacts in `results/analysis_json/*.json`.",
        "",
        f"- Candidate schedules compared: {len(candidates)}",
        f"- Legacy objective source: `{default_legacy_path}` (z_score aggregation)",
        f"- Spearman rank correlation (M1 DI-FOM vs Legacy): {corr_m1_legacy:.4f}",
        f"- Spearman rank correlation (M2 FIM-D vs Legacy): {corr_m2_legacy:.4f}",
        f"- Spearman rank correlation (M3 MWDCS vs Legacy): {corr_m3_legacy:.4f}",
        f"- Top-{top_k} overlap (M1 vs Legacy): {overlap_m1_legacy}",
        f"- Top-{top_k} overlap (M2 vs Legacy): {overlap_m2_legacy}",
        f"- Top-{top_k} overlap (M3 vs Legacy): {overlap_m3_legacy}",
        "",
        "## Top Legacy schedules",
    ]
    lines.extend(f"- {label}" for label in top_legacy)
    lines.append("")
    lines.append("## Top DI-FOM schedules")
    lines.extend(f"- {label}" for label in top_difom)
    lines.append("")
    lines.append("## Top FIM-D schedules")
    lines.extend(f"- {label}" for label in top_fim)
    lines.append("")
    lines.append("## Top MWDCS schedules")
    lines.extend(f"- {label}" for label in top_mwdcs)
    lines.append("")
    lines.append(f"CSV output: `{csv_path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote M1/M2/M3/legacy comparison CSV: {csv_path}")
    print(f"Wrote M1/M2/M3/legacy comparison summary: {md_path}")
    print(f"Spearman rank correlation (M1 vs legacy): {corr_m1_legacy:.4f}")
    print(f"Spearman rank correlation (M2 vs legacy): {corr_m2_legacy:.4f}")
    print(f"Spearman rank correlation (M3 vs legacy): {corr_m3_legacy:.4f}")
    print(f"Top-{top_k} overlap with legacy (M1): {overlap_m1_legacy}")
    print(f"Top-{top_k} overlap with legacy (M2): {overlap_m2_legacy}")
    print(f"Top-{top_k} overlap with legacy (M3): {overlap_m3_legacy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
