from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fluxforge.io.cnf import read_cnf_file
from fluxforge.io.hpge import read_chn_file
from fluxforge.io.iec import read_iec_file
from fluxforge.io.spe import read_spe_file

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
BASELINE_PATH = TEST_DATA_DIR / "parity_baselines" / "external_reference.json"
SPECTRUM_DATA_DIR = TEST_DATA_DIR / "spectrum_io"


def _load_baselines() -> dict[str, dict[str, object]]:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return payload["spectrum_io"]


def _summarize_counts(counts: np.ndarray) -> dict[str, float | int | str]:
    arr = np.rint(np.asarray(counts, dtype=float)).astype(np.int64, copy=False)
    return {
        "n_channels": int(arr.size),
        "sum_counts": float(arr.sum(dtype=np.float64)),
        "max_counts": float(arr.max() if arr.size else 0.0),
        "mean_counts": float(arr.mean(dtype=np.float64) if arr.size else 0.0),
        "sha256_counts_int64_le": hashlib.sha256(
            arr.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def _assert_matches_baseline(dataset_key: str, counts: np.ndarray) -> None:
    baseline = _load_baselines()[dataset_key]
    observed = _summarize_counts(counts)

    assert observed["n_channels"] == baseline["n_channels"]
    assert observed["sum_counts"] == pytest.approx(baseline["sum_counts"], rel=0, abs=0)
    assert observed["max_counts"] == pytest.approx(baseline["max_counts"], rel=0, abs=0)
    assert observed["mean_counts"] == pytest.approx(
        baseline["mean_counts"], rel=0, abs=0
    )
    assert observed["sha256_counts_int64_le"] == baseline["sha256_counts_int64_le"]


def test_spe_parity_against_external_baseline() -> None:
    path = SPECTRUM_DATA_DIR / "examples" / "eu_calib_7cm.Spe"
    ff = read_spe_file(path)
    _assert_matches_baseline("spe_eu_calib_7cm", ff.counts)


def test_cnf_parity_against_external_baseline() -> None:
    path = (
        SPECTRUM_DATA_DIR
        / "samples"
        / "01122014152731-GT01122014182338-GA37.4963000N-GO122.4633000W.cnf"
    )
    ff = read_cnf_file(path)
    _assert_matches_baseline("cnf_canberra_sample", ff.counts)


def test_iec_parity_against_external_baseline() -> None:
    path = SPECTRUM_DATA_DIR / "samples" / "hpge_dummy_test_01.iec"
    ff = read_iec_file(path)
    _assert_matches_baseline("iec_hpge_dummy_01", ff.counts)


def test_chn_parity_against_local_sample_baseline() -> None:
    path = SPECTRUM_DATA_DIR / "samples" / "eu_calib_7cm.Chn"
    ff = read_chn_file(path)
    _assert_matches_baseline("chn_eu_calib_7cm", ff.counts)
