from __future__ import annotations

import csv
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fluxforge.solvers.iterative import gravel, mlem

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
BASELINE_PATH = TEST_DATA_DIR / "parity_baselines" / "external_reference.json"
CASE_DIR = TEST_DATA_DIR / "external_cases" / "unfolding_case_001"


def _load_measurements(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        values = [float(row["NEUTRON 1"]) for row in reader]
    return np.asarray(values, dtype=float)


def _load_response(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    delimiter = "," if "," in first_line else None
    response = np.genfromtxt(path, delimiter=delimiter)
    if response.ndim == 1:
        response = np.atleast_2d(response)
    if np.isnan(response).any():
        valid_cols = ~np.all(np.isnan(response), axis=0)
        response = response[:, valid_cols]
    return response.T


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _digest_flux(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=float).astype("<f8", copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def _run_fluxforge_unfolding() -> dict[str, np.ndarray]:
    response = _load_response(CASE_DIR / "response_matrix.txt")
    measurements = _load_measurements(CASE_DIR / "reduced_data.csv")
    initial_flux = np.ones(response.shape[1], dtype=float).tolist()

    gravel_solution = gravel(
        response=response.tolist(),
        measurements=measurements.tolist(),
        initial_flux=initial_flux,
        max_iters=500,
    )
    mlem_solution = mlem(
        response=response.tolist(),
        measurements=measurements.tolist(),
        initial_flux=initial_flux,
        max_iters=500,
        convergence_mode="ddJ",
        relaxation=1.0,
    )
    return {
        "gravel_flux": np.asarray(gravel_solution.flux, dtype=float),
        "mlem_flux": np.asarray(mlem_solution.flux, dtype=float),
    }


def _load_baseline_case() -> dict[str, dict[str, object]]:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return payload["unfolding_case_001"]


@pytest.mark.parametrize("check_id", ["gravel_flux", "mlem_flux"])
def test_unfolding_case_matches_external_baseline(check_id: str) -> None:
    baseline = _load_baseline_case()[check_id]
    observed = _run_fluxforge_unfolding()[check_id]
    reference = np.asarray(baseline["reference_flux"], dtype=float)
    min_corr = float(baseline["min_corr"])

    assert observed.shape == reference.shape
    assert _correlate(observed, reference) >= min_corr
    assert _digest_flux(reference) == baseline["sha256_flux_float64_le"]
