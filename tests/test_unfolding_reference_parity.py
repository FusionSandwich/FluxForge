from __future__ import annotations

import csv
import importlib.util
import io
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from fluxforge.solvers.iterative import gravel, mlem
from fluxforge.unfold import neutron_ibu as neutron_ibu_module
from fluxforge.unfold._types import ReactionRates, ResponseBundle
from fluxforge.unfolding import GravelUnfolder


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = WORKSPACE_ROOT / "testing"
NEUTRON_REFERENCE_ROOT = REFERENCE_ROOT / "Neutron-Unfolding"
PYUNFOLD_REFERENCE_ROOT = REFERENCE_ROOT / "pyunfold"


def _normalize_flux(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array / np.linalg.norm(array)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _silence_external_output() -> Iterator[None]:
    with (
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message="np.find_common_type is deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in sqrt",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="numpy\\.ufunc size changed, may indicate binary incompatibility.*",
            category=RuntimeWarning,
        )
        yield


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if path_str in sys.path:
            sys.path.remove(path_str)


def _load_neutron_reference_inputs() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    inputs_root = NEUTRON_REFERENCE_ROOT / "unfolding_inputs"
    response = np.loadtxt(inputs_root / "response-matrix.txt", delimiter=",").T
    with (inputs_root / "reduced_data.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    measurements = np.asarray([float(row["NEUTRON 1"]) for row in rows], dtype=float)
    energy_spectrum = np.loadtxt(inputs_root / "energy-spectrum.txt")
    return response, measurements, {
        "constant": np.ones(response.shape[1], dtype=float),
        "true": np.asarray(energy_spectrum[0], dtype=float),
    }


def _load_local_pyunfold_symbols():
    with _prepend_sys_path(PYUNFOLD_REFERENCE_ROOT), _silence_external_output():
        from pyunfold.priors import jeffreys_prior
        from pyunfold.tests.testing_utils import diagonal_response, triangular_response
        from pyunfold.unfold import iterative_unfold

    return iterative_unfold, jeffreys_prior, diagonal_response, triangular_response


def _pyunfold_reference_cases():
    iterative_unfold, jeffreys_prior, diagonal_response, triangular_response = (
        _load_local_pyunfold_symbols()
    )
    cases: list[dict[str, object]] = []
    causes = np.arange(2, dtype=float) + 0.5
    cases.append(
        {
            "name": "example_1",
            "data": np.array([100.0, 150.0], dtype=float),
            "data_err": np.array([10.0, 12.2], dtype=float),
            "response": np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float),
            "response_err": np.full((2, 2), 0.01, dtype=float),
            "efficiencies": np.array([0.4, 0.67], dtype=float),
            "efficiencies_err": np.array([0.01, 0.01], dtype=float),
            "prior": jeffreys_prior(causes=causes),
        }
    )
    cases.append(
        {
            "name": "example_2",
            "data": np.array([100.0, 150.0], dtype=float),
            "data_err": np.array([10.0, 12.2], dtype=float),
            "response": np.array([[0.8, 0.1], [0.2, 0.9]], dtype=float),
            "response_err": np.full((2, 2), 0.01, dtype=float),
            "efficiencies": np.array([0.4, 0.67], dtype=float),
            "efficiencies_err": np.array([0.01, 0.01], dtype=float),
            "prior": np.array([0.34, 0.66], dtype=float),
        }
    )
    cases.append(
        {
            "name": "example_3",
            "data": np.array([100.0, 150.0], dtype=float),
            "data_err": np.array([10.0, 12.2], dtype=float),
            "response": np.array([[0.8, 0.1, 0.6], [0.2, 0.9, 0.4]], dtype=float),
            "response_err": np.full((2, 3), 0.01, dtype=float),
            "efficiencies": np.array([0.4, 0.67, 0.8], dtype=float),
            "efficiencies_err": np.array([0.01, 0.01, 0.01], dtype=float),
            "prior": np.array([0.34, 0.21, 0.45], dtype=float),
        }
    )

    np.random.seed(2)
    samples = np.random.normal(loc=0.0, scale=1.0, size=int(1e5))
    bins = np.linspace(-1.0, 1.0, 10)
    counts, _ = np.histogram(samples, bins=bins)
    counts_err = np.sqrt(counts)
    efficiencies = np.ones_like(counts, dtype=float)
    efficiencies_err = np.full_like(efficiencies, 0.001)
    diag_response, diag_response_err = diagonal_response(len(counts))
    tri_response, tri_response_err = triangular_response(len(counts))
    cases.append(
        {
            "name": "diagonal_response",
            "data": counts.astype(float),
            "data_err": counts_err.astype(float),
            "response": np.asarray(diag_response, dtype=float),
            "response_err": np.asarray(diag_response_err, dtype=float),
            "efficiencies": efficiencies,
            "efficiencies_err": efficiencies_err,
            "prior": None,
        }
    )
    cases.append(
        {
            "name": "triangular_response",
            "data": counts.astype(float),
            "data_err": counts_err.astype(float),
            "response": np.asarray(tri_response, dtype=float),
            "response_err": np.asarray(tri_response_err, dtype=float),
            "efficiencies": efficiencies,
            "efficiencies_err": efficiencies_err,
            "prior": None,
        }
    )
    return iterative_unfold, cases


@pytest.mark.skipif(
    not NEUTRON_REFERENCE_ROOT.exists(),
    reason="Local Neutron-Unfolding reference repo is unavailable.",
)
@pytest.mark.parametrize(
    ("initial_key", "tolerance"),
    [("constant", 0.2), ("true", 0.8)],
)
def test_gravel_matches_neutron_reference_with_same_tolerance(
    initial_key: str,
    tolerance: float,
) -> None:
    response, measurements, guesses = _load_neutron_reference_inputs()
    initial_flux = guesses[initial_key]
    reference_gravel = _load_module("reference_gravel", NEUTRON_REFERENCE_ROOT / "gravel.py")

    with _silence_external_output():
        reference_flux, reference_history = reference_gravel.gravel(
            response.copy(),
            measurements.copy(),
            initial_flux.copy(),
            tolerance,
        )

    solution = gravel(
        response.tolist(),
        measurements.tolist(),
        initial_flux=initial_flux.tolist(),
        max_iters=1000,
        tolerance=tolerance,
        chi2_tolerance=-1.0,
        relaxation=1.0,
        convergence_mode="ddJ",
    )

    assert solution.converged
    assert solution.iterations == len(reference_history)
    np.testing.assert_allclose(
        _normalize_flux(solution.flux),
        _normalize_flux(reference_flux),
        rtol=1e-2,
        atol=1e-12,
    )


@pytest.mark.skipif(
    not NEUTRON_REFERENCE_ROOT.exists(),
    reason="Local Neutron-Unfolding reference repo is unavailable.",
)
@pytest.mark.parametrize(
    ("initial_key", "tolerance"),
    [("constant", 0.2), ("true", 0.8)],
)
def test_mlem_matches_neutron_reference_with_same_tolerance(
    initial_key: str,
    tolerance: float,
) -> None:
    response, measurements, guesses = _load_neutron_reference_inputs()
    initial_flux = guesses[initial_key]
    reference_mlem = _load_module("reference_mlem", NEUTRON_REFERENCE_ROOT / "mlem.py")

    with _silence_external_output():
        reference_flux, reference_history = reference_mlem.mlem(
            response.copy(),
            measurements.copy(),
            initial_flux.copy(),
            tolerance,
        )

    solution = mlem(
        response.tolist(),
        measurements.tolist(),
        initial_flux=initial_flux.tolist(),
        max_iters=1000,
        tolerance=tolerance,
        chi2_tolerance=-1.0,
        relaxation=1.0,
        convergence_mode="ddJ",
    )

    assert solution.converged
    assert solution.iterations == len(reference_history)
    np.testing.assert_allclose(
        _normalize_flux(solution.flux),
        _normalize_flux(reference_flux),
        rtol=1e-2,
        atol=1e-12,
    )


@pytest.mark.skipif(
    not NEUTRON_REFERENCE_ROOT.exists(),
    reason="Local Neutron-Unfolding reference repo is unavailable.",
)
def test_gravel_unfolder_matches_neutron_reference() -> None:
    response, measurements, guesses = _load_neutron_reference_inputs()
    tolerance = 0.2
    initial_flux = guesses["constant"]
    reference_gravel = _load_module("reference_gravel_adapter", NEUTRON_REFERENCE_ROOT / "gravel.py")

    with _silence_external_output():
        reference_flux, reference_history = reference_gravel.gravel(
            response.copy(),
            measurements.copy(),
            initial_flux.copy(),
            tolerance,
        )

    unfolder = GravelUnfolder(
        max_iterations=1000,
        tolerance=tolerance,
        chi2_tolerance=-1.0,
        relaxation=1.0,
        convergence_mode="ddJ",
    )
    solution = unfolder.unfold(
        measurements,
        response,
        initial_flux=initial_flux,
        convergence_mode="ddJ",
    )

    assert solution.converged
    assert solution.iterations == len(reference_history)
    np.testing.assert_allclose(
        _normalize_flux(solution.flux),
        _normalize_flux(reference_flux),
        rtol=1e-2,
        atol=1e-12,
    )


@pytest.mark.skipif(
    not PYUNFOLD_REFERENCE_ROOT.exists(),
    reason="Local pyunfold reference repo is unavailable.",
)
@pytest.mark.parametrize("case_index", range(5))
def test_neutron_ibu_matches_local_pyunfold_reference(
    case_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterative_unfold, cases = _pyunfold_reference_cases()
    case = cases[case_index]

    monkeypatch.setattr(neutron_ibu_module, "_HAS_PYUNFOLD", True)
    monkeypatch.setattr(neutron_ibu_module, "_pyunfold_unfold", iterative_unfold)

    with _silence_external_output():
        reference_result = iterative_unfold(
            data=case["data"],
            data_err=case["data_err"],
            response=case["response"],
            response_err=case["response_err"],
            efficiencies=case["efficiencies"],
            efficiencies_err=case["efficiencies_err"],
            prior=case["prior"],
            ts="ks",
            ts_stopping=0.01,
            max_iter=100,
            cov_type="multinomial",
            return_iterations=False,
        )

    with _silence_external_output():
        solver = neutron_ibu_module.NeutronUnfolderIBU(
            ts="ks",
            ts_stopping=0.01,
            max_iter=100,
            cov_type="multinomial",
        )
        result = solver.solve(
            ReactionRates(
                values=np.asarray(case["data"], dtype=float),
                uncertainties=np.asarray(case["data_err"], dtype=float),
            ),
            ResponseBundle(
                matrix=np.asarray(case["response"], dtype=float),
                energy_bins=np.arange(
                    np.asarray(case["response"], dtype=float).shape[1] + 1,
                    dtype=float,
                ),
            ),
            prior_flux=(
                None
                if case["prior"] is None
                else np.asarray(case["prior"], dtype=float)
            ),
            efficiencies=np.asarray(case["efficiencies"], dtype=float),
            efficiencies_err=np.asarray(case["efficiencies_err"], dtype=float),
            response_err=np.asarray(case["response_err"], dtype=float),
        )

    np.testing.assert_allclose(
        result.unfolded_flux,
        np.asarray(reference_result["unfolded"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.statistical_uncertainty,
        np.asarray(reference_result["stat_err"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.systematic_uncertainty,
        np.asarray(reference_result["sys_err"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert result.n_iterations == int(reference_result["num_iterations"])
    assert result.test_statistic == pytest.approx(
        float(reference_result["ts_iter"]),
        rel=1e-12,
        abs=1e-12,
    )
