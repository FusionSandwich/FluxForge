#!/usr/bin/env python3
import sys
import os
import math
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from _compare_common import ensure_fluxforge_src

ROOT = ensure_fluxforge_src(__file__)

from fluxforge.io.flux_wire import read_raw_asc, read_processed_txt
from fluxforge.data.k0_library import get_k0_library_record
from fluxforge.examples.rafm_workflow import qg_reference_peaks
from fluxforge.workflows.astm_inl_dosimetry import (
    INLDosimetryWorkflow,
    DetectorCalibrationRecord,
    DecayDataRecord,
)


def build_workflow_and_compare(target: str = "Ti"):
    print("=" * 80)
    print(f"Running ASTM INL Workflow for target: {target}")

    example_root = ROOT / "examples/RAFM_irradiation"
    raw_dir = example_root / "raw_gamma_spec/flux_wires"
    qg_dir = example_root / "QG_processed_gamma_data/flux_wires"

    raw_files = list(raw_dir.glob(f"{target}-*.ASC"))
    if not raw_files:
        print("No raw file found.")
        return

    raw_file = raw_files[0]
    raw_data = read_raw_asc(raw_file)
    print(f"Loaded raw file: {raw_file.name}")

    txt_stem = raw_file.stem
    qg_file = qg_dir / f"{txt_stem}.txt"
    if not qg_file.exists():
        qg_file = list(qg_dir.glob(f"*{target}*.txt"))[0]

    qg_data = read_processed_txt(qg_file)
    qg_peaks = qg_reference_peaks(qg_data)

    energy_cal = qg_data.energy_calibration
    fwhm_cal = [1.5, 0.02]

    cal = DetectorCalibrationRecord(
        detector_id="DET01",
        energy_poly=energy_cal,
        fwhm_poly=fwhm_cal,
        efficiency_curve=qg_data.efficiency,
    )

    workflow = INLDosimetryWorkflow()
    peaks = workflow.analyze_spectrum(raw_data.spectrum.counts, cal)

    print(f"Found {len(peaks)} candidate peaks.")

    def find_target_peak(energy_keV, tolerance=3.0):
        closest = None
        min_err = 1e9
        for p in peaks:
            err = abs(p.centroid_keV - energy_keV)
            if err < tolerance and err < min_err:
                closest = p
                min_err = err
        return closest

    print("\n[Comparison vs QG]")
    for qg_p in qg_peaks:
        isotope = qg_p["isotope"]
        qg_e = qg_p["energy_keV"]
        qg_net = qg_p.get("net_counts", 0.0)

        my_p = find_target_peak(qg_e)
        if not my_p:
            print(f"Missed {isotope} @ {qg_e:.1f} keV")
            continue

        print(f"Target: {isotope} @ {qg_e:.1f} keV")
        print(f"  QG Net Area: {qg_net:.1f}")
        print(
            f"  WF Net Area: {my_p.net_area:.1f}  (roi: {my_p.roi_left}-{my_p.roi_right})"
        )

        err_pct = abs(my_p.net_area - qg_net) / max(1.0, qg_net) * 100
        print(f"  Delta: {err_pct:.1f}%")


if __name__ == "__main__":
    for t in ["Co", "Ti", "Sc", "Ni"]:
        build_workflow_and_compare(t)
