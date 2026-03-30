import ast
from pathlib import Path

import numpy as np
import pytest

from fluxforge.plugins import PluginRegistries
from fluxforge.solvers.iterative import gravel as legacy_gravel
from fluxforge.unfolding import (
    GravelUnfolder,
    MLSeedUnfolder,
    MaxedUnfolder,
    RMLEUnfolder,
    UnfoldingResult,
    register_builtin_unfolders,
    unfolding_entries,
)


ROOT = Path(__file__).resolve().parents[1]


def test_register_builtin_unfolders_registers_rmle_as_default():
    registries = PluginRegistries()

    register_builtin_unfolders(registries)

    assert registries.unfolders.default_key == "rmle"
    gravel_entry = registries.unfolders.get_entry("gravel")
    assert isinstance(gravel_entry.implementation, GravelUnfolder)
    assert gravel_entry.metadata.tags == ("phase3", "unfolding", "gravel")

    maxed_entry = registries.unfolders.get_entry("maxed")
    assert isinstance(maxed_entry.implementation, MaxedUnfolder)
    assert maxed_entry.metadata.tags == ("phase3", "unfolding", "maxed")

    rmle_entry = registries.unfolders.get_entry("rmle")
    assert isinstance(rmle_entry.implementation, RMLEUnfolder)
    assert rmle_entry.metadata.recommended is True
    assert rmle_entry.metadata.tags == ("phase3", "unfolding", "rmle")

    ml_seed_entry = registries.unfolders.get_entry("ml_seed")
    assert isinstance(ml_seed_entry.implementation, MLSeedUnfolder)
    assert ml_seed_entry.metadata.tags == ("phase3", "unfolding", "ml_seed")
    assert [registered.key for registered in unfolding_entries(registries)] == [
        "gravel",
        "maxed",
        "rmle",
        "ml_seed",
    ]


def test_gravel_unfolder_matches_legacy_iterative_solver():
    response = np.array(
        [
            [0.80, 0.10, 0.05],
            [0.15, 0.75, 0.20],
            [0.05, 0.15, 0.75],
        ],
        dtype=float,
    )
    true_flux = np.array([120.0, 55.0, 18.0], dtype=float)
    measured = response @ true_flux
    initial = np.array([60.0, 60.0, 60.0], dtype=float)
    uncertainty = np.sqrt(np.maximum(measured, 1.0))

    unfolder = GravelUnfolder(max_iterations=400, tolerance=1e-8, chi2_tolerance=1e-10)
    result = unfolder.unfold(
        measured,
        response,
        initial_flux=initial,
        measurement_uncertainty=uncertainty,
    )
    legacy = legacy_gravel(
        response.tolist(),
        measured.tolist(),
        initial_flux=initial.tolist(),
        measurement_uncertainty=uncertainty.tolist(),
        max_iters=400,
        tolerance=1e-8,
        chi2_tolerance=1e-10,
    )

    assert isinstance(result, UnfoldingResult)
    assert result.method_used == "GRAVEL"
    assert result.parameters_used["used_initial_flux"] is True
    assert result.parameters_used["used_measurement_uncertainty"] is True
    assert result.flux.shape == true_flux.shape
    assert result.predicted_measurements.shape == measured.shape
    assert result.residuals.shape == measured.shape
    assert result.iterations == legacy.iterations
    assert result.convergence_history == pytest.approx(tuple(legacy.chi_squared_history))
    assert result.chi_squared == pytest.approx(legacy.chi_squared)
    assert np.allclose(result.flux, np.asarray(legacy.flux, dtype=float))
    assert np.allclose(result.predicted_measurements, response @ result.flux)
    assert result.negative_bin_count == 0


def test_gravel_unfolder_validates_input_shapes():
    unfolder = GravelUnfolder()

    with pytest.raises(ValueError):
        unfolder.unfold(np.array([1.0, 2.0]), np.ones((3, 2), dtype=float))

    with pytest.raises(ValueError):
        unfolder.unfold(
            np.array([1.0, 2.0, 3.0]),
            np.ones((3, 2), dtype=float),
            initial_flux=np.array([1.0, 2.0, 3.0]),
        )


def test_maxed_unfolder_returns_positive_flux_and_uncertainties():
    response = np.array(
        [
            [0.88, 0.14, 0.02],
            [0.10, 0.80, 0.15],
            [0.02, 0.06, 0.83],
        ],
        dtype=float,
    )
    true_flux = np.array([115.0, 64.0, 22.0], dtype=float)
    measured = response @ true_flux
    uncertainty = np.sqrt(np.maximum(measured, 1.0))
    unfolder = MaxedUnfolder(max_iterations=300, entropy_weight=0.02)

    result = unfolder.unfold(
        measured,
        response,
        initial_flux=np.array([70.0, 70.0, 70.0], dtype=float),
        measurement_uncertainty=uncertainty,
    )

    assert isinstance(result, UnfoldingResult)
    assert result.method_used == "MAXED"
    assert result.uncertainties is not None
    assert result.uncertainties.shape == result.flux.shape
    assert np.all(result.flux >= 0.0)
    assert np.all(result.uncertainties >= 0.0)
    assert np.allclose(response @ result.flux, measured, rtol=5e-2, atol=5e-2)


def test_rmle_unfolder_returns_uncertainties_and_nonnegative_flux():
    response = np.eye(3, dtype=float)
    measured = np.array([120.0, 60.0, 20.0], dtype=float)
    uncertainty = np.array([6.0, 4.0, 2.0], dtype=float)

    result = RMLEUnfolder(max_iterations=200, auto_regularization=True).unfold(
        measured,
        response,
        measurement_uncertainty=uncertainty,
    )

    assert isinstance(result, UnfoldingResult)
    assert result.method_used == "RMLE"
    assert result.parameters_used["parameter_selection"] == "automatic"
    assert result.uncertainties is not None
    assert np.all(result.flux >= 0.0)
    assert np.all(result.uncertainties >= 0.0)
    assert np.allclose(result.predicted_measurements, measured, rtol=1e-2, atol=1e-2)


def test_ml_seed_unfolder_returns_confidence_and_positive_flux():
    response = np.eye(3, dtype=float)
    measured = np.array([120.0, 60.0, 20.0], dtype=float)
    uncertainty = np.array([6.0, 4.0, 2.0], dtype=float)

    result = MLSeedUnfolder().unfold(
        measured,
        response,
        measurement_uncertainty=uncertainty,
        confidence_threshold=0.4,
    )

    assert isinstance(result, UnfoldingResult)
    assert result.method_used == "ML Seed"
    assert np.all(result.flux >= 0.0)
    assert result.uncertainties is not None
    assert result.parameters_used["confidence_score"] >= 0.4
    assert result.parameters_used["accepted"] is True
    assert np.allclose(result.predicted_measurements, measured, rtol=1e-2, atol=1e-2)


def test_unfolding_package_remains_gui_free():
    package_paths = (
        ROOT / "src" / "fluxforge" / "unfolding" / "__init__.py",
        ROOT / "src" / "fluxforge" / "unfolding" / "base.py",
        ROOT / "src" / "fluxforge" / "unfolding" / "gravel.py",
        ROOT / "src" / "fluxforge" / "unfolding" / "gpu_backend.py",
        ROOT / "src" / "fluxforge" / "unfolding" / "maxed.py",
        ROOT / "src" / "fluxforge" / "unfolding" / "ml_seed.py",
        ROOT / "src" / "fluxforge" / "unfolding" / "response_matrix.py",
        ROOT / "src" / "fluxforge" / "unfolding" / "rmle.py",
    )

    for path in package_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert all(not name.startswith("PySide6") for name in imports)
        assert all(not name.startswith("fluxforge.gui") for name in imports)
