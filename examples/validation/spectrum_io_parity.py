#!/usr/bin/env python3
"""FluxForge spectrum-IO parity check against stored reference baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fluxforge.io.cnf import read_cnf_file
from fluxforge.io.hpge import read_chn_file
from fluxforge.io.iec import read_iec_file
from fluxforge.io.spe import read_spe_file

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
BASELINE_PATH = TEST_DATA_DIR / "parity_baselines" / "external_reference.json"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "validation" / "spectrum_io" / "spectrum_io_parity.json"
SPECTRUM_DATA_DIR = TEST_DATA_DIR / "spectrum_io"


def _digest(counts: np.ndarray) -> str:
    arr = np.rint(np.asarray(counts, dtype=float)).astype(np.int64, copy=False)
    return hashlib.sha256(arr.astype("<i8", copy=False).tobytes()).hexdigest()


def _summarize(counts: np.ndarray) -> dict[str, float | int | str]:
    arr = np.rint(np.asarray(counts, dtype=float)).astype(np.int64, copy=False)
    return {
        "n_channels": int(arr.size),
        "sum_counts": float(arr.sum(dtype=np.float64)),
        "max_counts": float(arr.max() if arr.size else 0.0),
        "mean_counts": float(arr.mean(dtype=np.float64) if arr.size else 0.0),
        "sha256_counts_int64_le": _digest(arr),
    }


def main() -> None:
    baselines = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["spectrum_io"]

    observed = {
        "spe_eu_calib_7cm": _summarize(
            read_spe_file(SPECTRUM_DATA_DIR / "examples" / "eu_calib_7cm.Spe").counts
        ),
        "cnf_canberra_sample": _summarize(
            read_cnf_file(
                SPECTRUM_DATA_DIR
                / "samples"
                / "01122014152731-GT01122014182338-GA37.4963000N-GO122.4633000W.cnf"
            ).counts
        ),
        "iec_hpge_dummy_01": _summarize(
            read_iec_file(SPECTRUM_DATA_DIR / "samples" / "hpge_dummy_test_01.iec").counts
        ),
        "chn_eu_calib_7cm": _summarize(
            read_chn_file(SPECTRUM_DATA_DIR / "samples" / "eu_calib_7cm.Chn").counts
        ),
    }

    report = {}
    all_match = True
    for key, obs in observed.items():
        ref = baselines[key]
        match = obs["sha256_counts_int64_le"] == ref["sha256_counts_int64_le"]
        all_match = all_match and match
        report[key] = {
            "match": match,
            "observed": obs,
            "reference": {
                "n_channels": ref["n_channels"],
                "sum_counts": ref["sum_counts"],
                "max_counts": ref["max_counts"],
                "mean_counts": ref["mean_counts"],
                "sha256_counts_int64_le": ref["sha256_counts_int64_le"],
            },
        }
        print(f"{key}: {'PASS' if match else 'FAIL'}")

    report["all_match"] = all_match
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
