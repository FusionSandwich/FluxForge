#!/usr/bin/env python3
"""Compute pairwise method-vs-method ranking similarities from benchmark CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from fluxforge.analysis.method_comparison import (
    compute_pairwise_method_metrics,
    serialize_pairwise_method_metrics,
)


def _load_rank_map(csv_path: Path, column: str) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id") or "").strip()
            rank_raw = row.get(column)
            if not sample_id or rank_raw is None or str(rank_raw).strip() == "":
                continue
            rank_map[sample_id] = int(float(rank_raw))
    return rank_map


def main() -> int:
    root = Path(__file__).resolve().parent
    benchmark_root = root / "results" / "method_benchmark"
    input_csv = benchmark_root / "m1_m2_m3_n1_n2_legacy_schedule_comparison.csv"
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing benchmark CSV: {input_csv}. Run compare_m1_m2_m3_n1_n2_legacy_schedule_objectives.py first."
        )

    rank_by_method = {
        "M1": _load_rank_map(input_csv, "difom_rank"),
        "M2": _load_rank_map(input_csv, "fim_d_rank"),
        "M3": _load_rank_map(input_csv, "mwdcs_rank"),
        "N1": _load_rank_map(input_csv, "bassd_rank"),
        "N2": _load_rank_map(input_csv, "stbdmr_rank"),
        "Legacy": _load_rank_map(input_csv, "legacy_rank"),
    }

    metrics = compute_pairwise_method_metrics(rank_by_method, top_k=5)
    metric_rows = serialize_pairwise_method_metrics(metrics)

    csv_path = benchmark_root / "method_pairwise_similarity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_a",
                "method_b",
                "sample_count",
                "spearman_rank_correlation",
                "top_k_overlap",
                "top_k",
            ],
        )
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    md_path = benchmark_root / "method_pairwise_similarity.md"
    lines = [
        "# Pairwise Method Similarity",
        "",
        f"Input ranking source: `{input_csv}`",
        "",
        "| Method A | Method B | Samples | Spearman | Top-5 Overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| "
            f"{row['method_a']} | {row['method_b']} | {row['sample_count']} | "
            f"{float(row['spearman_rank_correlation']):.4f} | {row['top_k_overlap']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote pairwise method similarity CSV: {csv_path}")
    print(f"Wrote pairwise method similarity summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
