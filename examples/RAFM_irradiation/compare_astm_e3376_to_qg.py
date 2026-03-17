#!/usr/bin/env python3
import sys
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from _compare_common import ensure_fluxforge_src

ROOT = ensure_fluxforge_src(__file__)

from fluxforge.io.flux_wire import read_processed_txt
from fluxforge.examples.rafm_workflow import qg_reference_peaks
from fluxforge.analysis.astm_e3376 import analyze_astm_e3376_plan
from fluxforge.data.k0_library import get_k0_library_record


def compare_astm_and_qg(max_spectra: int = None):
    example_root = ROOT / "examples/RAFM_irradiation"
    qg_dir = example_root / "QG_processed_gamma_data/flux_wires"

    txt_files = list(qg_dir.glob("*.txt"))
    if max_spectra is not None:
        txt_files = txt_files[:max_spectra]

    print(
        f"Comparing QG peak reporting vs strict ASTM E3376 formulas for {len(txt_files)} samples."
    )
    print("=" * 80)

    for txt_path in txt_files:
        try:
            data = read_processed_txt(txt_path)
            qg_peaks = qg_reference_peaks(data)
        except Exception:
            continue

        sample_id = data.sample_id or txt_path.stem.split("_")[0]

        plan = {"measurements": []}
        for p in qg_peaks:
            nuclide = next(
                (n for n in data.nuclides if n.isotope == p["isotope"]), None
            )
            eff = data.efficiency.efficiency(p["energy_keV"])

            iso_raw = p["isotope"]
            if " - " in iso_raw:
                iso_raw = iso_raw.split(" - ")[0]
            record = get_k0_library_record(iso_raw)
            gamma_prob = (
                record.yield_percent / 100.0 if record and record.yield_percent else 1.0
            )

            plan["measurements"].append(
                {
                    "measurement_id": f"{sample_id}:{p['isotope']}:{p['energy_keV']}keV",
                    "gross_area": p.get(
                        "area", p.get("area_cnts", p.get("net_area", 41819.0))
                    ),
                    "roi_channels": 10.0,  # approximation
                    "continuum_channels": 5.0,  # approximation
                    "continuum_low": 0.0,  # QG already does net, we simulate it
                    "continuum_high": 0.0,
                    "live_time_s": data.live_time or 1.0,
                    "efficiency": eff,
                    "gamma_probability": gamma_prob,
                }
            )

        print(plan)
        result = analyze_astm_e3376_plan(plan)
        for out in result.get("outputs", []):
            print(f"[{out['measurement_id']}]")
            print(f"   Net count rate: {out.get('net_count_rate', 0.0):.4e} c/s")
            print(f"   Emission rate:  {out.get('emission_rate', 0.0):.4e} 1/s")
            print(f"   Transmutation rate: {out.get('transmutation_rate', 0.0):.4e} Bq")
            print("-" * 40)


if __name__ == "__main__":
    compare_astm_and_qg(max_spectra=1)
