"""Compare frozen RAFM receipts without treating QG parity as physical truth."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_peaks(receipt):
    return {
        (sample["sample"], peak["isotope"], round(peak["expected_energy"], 4)): peak
        for sample in receipt["samples"]
        for peak in sample["found"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads((args.directory / "rafm_before.json").read_text())
    after = json.loads((args.directory / "rafm_final.json").read_text())
    if before["input_sha256"] != after["input_sha256"]:
        raise ValueError("Before/after input hashes differ")
    for relative, expected in after["runtime_source_sha256"].items():
        actual = hashlib.sha256((args.root / "src" / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Final receipt has stale source: {relative}")

    old, new = index_peaks(before), index_peaks(after)
    rows = []
    for key in sorted(old.keys() | new.keys()):
        previous, current = old.get(key, {}), new.get(key, {})
        qg = current.get("qg_net_counts") or previous.get("qg_net_counts")
        qg_sigma = current.get("qg_net_sigma") or previous.get("qg_net_sigma")
        row = {
            "sample": key[0],
            "isotope": key[1],
            "expected_energy_keV": key[2],
            "before_present": key in old,
            "after_present": key in new,
            "qg_net_counts": qg,
            "qg_net_sigma": qg_sigma,
        }
        for prefix, record in (("before", previous), ("after", current)):
            for field in (
                "fitted_energy",
                "physical_net_counts",
                "physical_net_sigma",
                "raw_comparison_net_counts",
                "raw_comparison_net_sigma",
            ):
                row[f"{prefix}_{field}"] = record.get(field)
        row["after_physical_sigma_over_qg"] = (
            current["physical_net_sigma"] / qg_sigma
            if current and qg_sigma and qg_sigma > 0
            else None
        )
        rows.append(row)
    write_csv(args.directory / "peak_count_comparison.csv", rows)

    clocks = []
    for previous, current in zip(
        before["clock_activity_rows"], after["clock_activity_rows"], strict=True
    ):
        key = ("sample", "isotope", "energy")
        if any(previous[k] != current[k] for k in key):
            raise ValueError("Clock row pairing mismatch")
        old_activity = previous["shared_helper_count_start_activity_bq"]
        new_activity = current["shared_helper_count_start_activity_bq"]
        clocks.append(
            {
                **{k: current[k] for k in key},
                "live_time_s": current["live_time_s"],
                "real_time_s": current["real_time_s"],
                "before_count_start_activity_bq": old_activity,
                "after_count_start_activity_bq": new_activity,
                "before_over_after": old_activity / new_activity,
                "qg_line_activity_bq": current["qg_line_activity_bq"],
            }
        )
    write_csv(args.directory / "clock_activity_comparison.csv", clocks)

    uncertainties = sorted(
        (r for r in rows if r["after_physical_sigma_over_qg"]),
        key=lambda r: abs(np.log(r["after_physical_sigma_over_qg"])),
        reverse=True,
    )
    summary = {
        "scientific_admission": False,
        "input_hashes_identical": True,
        "final_runtime_hashes_verified": True,
        "raw_sample_spectra": len(after["samples"]),
        "wire_spectra": sum(s["wire"] for s in after["samples"]),
        "selected_lines_before": len(old),
        "selected_lines_after": len(new),
        "off_energy_found_before": sum(
            len(c["found"]) for c in before["off_energy_cases"]
        ),
        "off_energy_found_after": sum(
            len(c["found"]) for c in after["off_energy_cases"]
        ),
        "clock_activity_rows": len(clocks),
        "unfolding_unit_cases": [
            {
                "prior": a["prior"],
                "warm_iterations": a["warm"],
                "before_max_absolute_flux_difference": b[
                    "max_absolute_unit_difference"
                ],
                "after_max_absolute_flux_difference": a["max_absolute_unit_difference"],
                "before_weighted_residual_sum_squares": b["raw"][
                    "weighted_residual_sum_squares"
                ],
                "after_weighted_residual_sum_squares": a["raw"][
                    "weighted_residual_sum_squares"
                ],
                "after_confidence_is_heuristic": True,
                "after_seed_accepted": a["raw"]["accepted"],
            }
            for b, a in zip(
                before["unfolding_unit_cases"],
                after["unfolding_unit_cases"],
                strict=True,
            )
        ],
        "largest_count_uncertainty_ratio_discrepancies": uncertainties[:10],
        "qg_comparison_note": "Physical residual counts and raw reference counts have different backgrounds. QG agreement does not establish physical truth or qualified uncertainty.",
    }
    (args.directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for prefix, marker, label in (
        ("before", "o", "Baseline"),
        ("after", "x", "Repaired"),
    ):
        selected = [
            r
            for r in rows
            if r["qg_net_counts"] and r[f"{prefix}_raw_comparison_net_counts"]
        ]
        axes[0].scatter(
            [r["qg_net_counts"] for r in selected],
            [r[f"{prefix}_raw_comparison_net_counts"] for r in selected],
            s=15,
            marker=marker,
            alpha=0.6,
            label=label,
        )
    axes[0].plot([1, 1e8], [1, 1e8], color="gray", linewidth=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="QG net counts",
        ylabel="Raw comparison net counts",
    )
    axes[0].legend()
    axes[1].scatter(
        [r["qg_net_counts"] for r in uncertainties],
        [r["after_physical_sigma_over_qg"] for r in uncertainties],
        s=15,
        alpha=0.7,
    )
    axes[1].axhline(1, color="gray", linewidth=1)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="QG net counts",
        ylabel="Repaired physical count sigma / QG sigma",
    )
    fig.suptitle(
        "Selected RAFM lines: raw-count parity and background-corrected uncertainty"
    )
    fig.savefig(args.directory / "rafm_count_comparison.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
