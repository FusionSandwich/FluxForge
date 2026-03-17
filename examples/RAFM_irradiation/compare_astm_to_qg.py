#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path
from typing import Any, Dict

# Turn off tensorflow chatter if memory/disk constrained
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from _compare_common import ensure_fluxforge_src

ROOT = ensure_fluxforge_src(__file__)

from fluxforge.io.flux_wire import read_processed_txt, FluxWireData
from fluxforge.examples.rafm_workflow import (
    qg_reference_peaks,
    load_rafm_example_metadata,
)
from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan


def compare_astm_and_qg(max_spectra: int = None):
    # Load basic workflow context
    example_root = ROOT / "examples/RAFM_irradiation"
    meta = load_rafm_example_metadata(example_root)
    qg_dir = example_root / "QG_processed_gamma_data/flux_wires"

    txt_files = list(qg_dir.glob("*.txt"))
    if max_spectra is not None:
        txt_files = txt_files[:max_spectra]

    print(
        f"Comparing QG activity reporting vs strict ASTM E261 formulas for {len(txt_files)} samples."
    )
    print("=" * 80)

    total_lines = 0
    total_diff = 0.0

    for txt_path in txt_files:
        try:
            data = read_processed_txt(txt_path)
            qg_peaks = qg_reference_peaks(data)
        except Exception as err:
            continue

        sample_id = data.sample_id or txt_path.stem.split("_")[0]
        sample_printed = False

        for i, p in enumerate(qg_peaks):
            nuclide = next(
                (n for n in data.nuclides if n.isotope == p["isotope"]), None
            )
            hl_s = nuclide.half_life_seconds if nuclide else 0.0
            if hasattr(data, "efficiency_at_energy"):
                eff = data.efficiency_at_energy(p["energy_keV"])
            elif hasattr(data, "efficiency") and hasattr(data.efficiency, "efficiency"):
                eff = float(data.efficiency.efficiency(p["energy_keV"]))
            else:
                eff = 1.0

            # Use raw QG counts, live time, intensity, wait time=0
            # to verify pure base mathematics at measurement boundary
            plan = {
                "title": f"Compare {sample_id}",
                "irradiation": {"duration_s": 129600.0},  # Dummy valid duration
                "measurements": [
                    {
                        "measurement_id": p.get("isotope", "unk"),
                        "net_counts": p.get("net_counts", 0),
                        "live_time_s": data.live_time,
                        "efficiency": eff,
                        "gamma_intensity": p.get("rad_int_fraction", 0.0)
                        or (p.get("rad_int_percent", 0) / 100.0),
                        "half_life_s": hl_s,
                        "cooling_time_s": 0.0,
                        "sample_mass_g": 0.001,  # Dummy weight
                        "atomic_mass_g_mol": 50.0,  # Dummy molar mass
                        "effective_cross_section_barn": 5.0,  # Dummy cross-section
                    }
                ],
            }
            if hl_s > 1.0 and eff > 0 and p.get("net_counts", 0) > 0:
                astm_res = analyze_astm_e261_plan(plan)
                astm_meas = astm_res["measurements"][0]

                astm_act = astm_meas["activity_eoi_Bq"]
                qg_act = p["line_activity_bq"]
                if qg_act > 0:
                    rel_err = abs(astm_act - qg_act) / qg_act
                    if not sample_printed:
                        print(f"\n[{sample_id}] -> {txt_path.name}")
                        sample_printed = True
                    print(
                        f"  {p['isotope']} @ {p['energy_keV']:>6.1f} keV | QG Act: {qg_act:9.3e} Bq | ASTM Act: {astm_act:9.3e} Bq | Diff: {rel_err*100:5.2f}%"
                    )
                    total_diff += rel_err
                    total_lines += 1

    if total_lines > 0:
        print("=" * 80)
        print(
            f"Mean Absolute Relative Difference: {(total_diff/total_lines)*100:.3f}% across {total_lines} lines."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-spectra", type=int, default=None)
    args = parser.parse_args()
    compare_astm_and_qg(max_spectra=args.max_spectra)
