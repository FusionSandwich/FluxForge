"""Integration test for 10-bin flux unfolding regression (raw vs processed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALARA_ROOT = REPO_ROOT.parent

PROC_DIR = ALARA_ROOT / "rafm_irradiation_ldrd" / "irradiation_QG_processed" / "flux_wires"
RAW_DIR = ALARA_ROOT / "rafm_irradiation_ldrd" / "raw_gamma_spec" / "flux_wires"
MODEL_PATH = RAW_DIR / "spectrum_vit_j.csv"

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "validation" / "flux_unfolding_10bin"


def load_regression_module():
    script_path = REPO_ROOT / "examples" / "validation" / "flux_unfolding_10bin_regression.py"
    spec = importlib.util.spec_from_file_location("flux_unfolding_10bin_regression", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flux_unfolding_10bin_regression():
    assert PROC_DIR.exists(), f"Processed directory missing: {PROC_DIR}"
    assert RAW_DIR.exists(), f"Raw directory missing: {RAW_DIR}"
    
    module = load_regression_module()
    results = module.run_unfolding_regression(PROC_DIR, RAW_DIR, MODEL_PATH)
    
    assert len(results.processed_discrete.flux) == 10
    assert len(results.raw_discrete.flux) == 10
    assert len(results.processed_gls.flux) == 50
    assert len(results.raw_gls.flux) == 50
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    module.save_spectrum_csv(
        ARTIFACT_DIR / "processed_discrete.csv",
        results.processed_discrete.energy_bounds_eV,
        results.processed_discrete.flux,
        results.processed_discrete.flux_unc,
    )
    module.save_spectrum_csv(
        ARTIFACT_DIR / "raw_discrete.csv",
        results.raw_discrete.energy_bounds_eV,
        results.raw_discrete.flux,
        results.raw_discrete.flux_unc,
    )
