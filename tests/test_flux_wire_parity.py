"""Integration tests for flux wire parity (processed vs raw)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np

from fluxforge.analysis.flux_wire_analysis import analyze_flux_wire_targeted
from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc, FluxWireData


REPO_ROOT = Path(__file__).resolve().parents[1]
ALARA_ROOT = REPO_ROOT.parent

PROC_DIR = ALARA_ROOT / "rafm_irradiation_ldrd" / "irradiation_QG_processed" / "flux_wires"
RAW_DIR = ALARA_ROOT / "rafm_irradiation_ldrd" / "raw_gamma_spec" / "flux_wires"

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "validation" / "flux_wire_parity"


def normalize_sample_id(sample_id: str) -> str:
    """Normalize sample IDs for matching raw/processed pairs."""
    text = sample_id.strip().replace(" ", "")
    text = text.replace("@", "_")
    text = re.sub(r"_?\d+cm$", "", text, flags=re.IGNORECASE)
    return text.lower()


def load_processed_files() -> Dict[str, FluxWireData]:
    """Load all processed flux wire files keyed by normalized sample ID."""
    processed: Dict[str, FluxWireData] = {}
    for path in sorted(PROC_DIR.glob("*.txt")):
        data = read_processed_txt(path)
        processed[normalize_sample_id(data.sample_id)] = data
    return processed


def load_raw_files() -> Dict[str, FluxWireData]:
    """Load all raw flux wire files keyed by normalized sample ID."""
    raw: Dict[str, FluxWireData] = {}
    for path in sorted(RAW_DIR.glob("*.ASC")):
        data = read_raw_asc(path)
        raw[normalize_sample_id(data.sample_id)] = data
    return raw


def apply_reference_calibration(raw: FluxWireData, reference: FluxWireData) -> None:
    """Apply reference calibration/efficiency to raw data in-place."""
    if reference.energy_calibration:
        raw.energy_calibration = reference.energy_calibration
        if raw.spectrum is not None:
            raw.spectrum.calibration["energy"] = reference.energy_calibration
            raw.spectrum.energies = raw.channel_to_energy(raw.spectrum.channels)
    if reference.efficiency is not None:
        raw.efficiency = reference.efficiency
    if reference.resolution:
        raw.resolution = reference.resolution


@dataclass
class ParityResult:
    sample_id: str
    isotope: str
    ref_bq: float
    calc_bq: float
    ratio: float
    diff_pct: float


def compare_raw_to_processed(tolerance: float = 0.02) -> Tuple[List[ParityResult], List[str]]:
    """Compare raw analysis to processed reference for all matched pairs."""
    processed = load_processed_files()
    raw = load_raw_files()
    
    comparisons: List[ParityResult] = []
    missing: List[str] = []
    
    for key, raw_data in raw.items():
        if key not in processed:
            continue
        ref = processed[key]
        apply_reference_calibration(raw_data, ref)
        
        analysis = analyze_flux_wire_targeted(
            data=raw_data,
            reference_data=ref,
            peak_threshold=0.0,
        )
        
        for nuclide in ref.nuclides:
            ref_bq = nuclide.activity_bq
            if nuclide.isotope not in analysis.nuclide_activities:
                missing.append(f"{raw_data.sample_id}:{nuclide.isotope}")
                continue
            calc_bq = analysis.nuclide_activities[nuclide.isotope]["activity_bq"]
            ratio = calc_bq / ref_bq if ref_bq > 0 else 0.0
            diff_pct = (ratio - 1.0) * 100.0
            comparisons.append(ParityResult(
                sample_id=raw_data.sample_id,
                isotope=nuclide.isotope,
                ref_bq=ref_bq,
                calc_bq=calc_bq,
                ratio=ratio,
                diff_pct=diff_pct,
            ))
    
    # Persist artifacts for inspection
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACT_DIR / "raw_vs_processed_parity.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_id,isotope,ref_bq,calc_bq,ratio,diff_pct\n")
        for row in comparisons:
            f.write(
                f"{row.sample_id},{row.isotope},{row.ref_bq:.6e},"
                f"{row.calc_bq:.6e},{row.ratio:.6f},{row.diff_pct:.3f}\n"
            )
    
    return comparisons, missing


def test_processed_flux_wire_parsing():
    """All processed files parse and expose nuclide results."""
    assert PROC_DIR.exists(), f"Processed directory missing: {PROC_DIR}"
    processed = load_processed_files()
    assert processed, "No processed flux wire files loaded"
    
    for key, data in processed.items():
        assert data.has_nuclides, f"No nuclides parsed for {data.sample_id}"
        assert data.energy_calibration, f"No energy calibration for {data.sample_id}"


def test_raw_flux_wire_parity():
    """Raw analysis reproduces processed activities within 2%."""
    assert RAW_DIR.exists(), f"Raw directory missing: {RAW_DIR}"
    
    comparisons, missing = compare_raw_to_processed(tolerance=0.02)
    assert comparisons, "No raw/processed comparisons generated"
    assert not missing, f"Missing isotopes in raw analysis: {missing}"
    
    out_of_tolerance = [
        c for c in comparisons if abs(c.diff_pct) > 2.0
    ]
    assert not out_of_tolerance, (
        "Raw/processed parity exceeded 2% for: "
        + ", ".join(f"{c.sample_id}:{c.isotope}({c.diff_pct:+.2f}%)" for c in out_of_tolerance)
    )
