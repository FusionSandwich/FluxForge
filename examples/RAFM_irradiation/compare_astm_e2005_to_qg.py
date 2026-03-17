#!/usr/bin/env python3
import os
import sys
import json
from itertools import combinations
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from _compare_common import ensure_fluxforge_src, pair_sample_files

ROOT = ensure_fluxforge_src(__file__)

from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc
from fluxforge.examples.rafm_workflow import (
    default_paths,
    qg_reference_peaks,
    workflow_profile_energy_calibration,
)
from fluxforge.analysis.flux_wire_analysis import (
    analyze_raw_spectrum_targeted,
    GammaLine,
)
from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan
from fluxforge.analysis.astm_e2005 import analyze_astm_e2005_plan


def _build_expected_lines(qg_peaks: list[dict]) -> list[GammaLine]:
    expected: list[GammaLine] = []
    for row in qg_peaks:
        intensity = float(row.get("rad_int_fraction") or 0.0)
        if intensity <= 0.0:
            intensity = max(float(row.get("rad_int_percent") or 0.0) / 100.0, 1.0e-6)
        expected.append(
            GammaLine(
                energy_keV=float(row.get("energy_keV") or row.get("energy_kev")),
                intensity=max(intensity, 1.0e-6),
                isotope=str(row.get("isotope", "")),
            )
        )
    return expected


def _find_raw_peak(raw_peaks, isotope: str, energy_kev: float, tol: float = 2.0):
    for peak in raw_peaks:
        if (
            peak.isotope == isotope
            and abs(float(peak.energy_keV) - float(energy_kev)) <= tol
        ):
            return peak
    return None


def _e261_rate_from_counts(
    *,
    reaction_id: str,
    isotope: str,
    energy_kev: float,
    net_counts: float,
    net_unc: float,
    live_time_s: float,
    efficiency: float,
    gamma_intensity: float,
    half_life_s: float,
    sample_mass_g: float = 1.0,
    atomic_mass_g_mol: float = 1.0,
    isotopic_abundance: float = 1.0,
    mass_fraction: float = 1.0,
    effective_cross_section_barn: float = 1.0,
):
    plan = {
        "title": f"E261 {reaction_id}",
        "irradiation": {"duration_s": max(float(live_time_s), 1.0)},
        "measurements": [
            {
                "measurement_id": reaction_id,
                "reaction_id": reaction_id,
                "target_isotope": isotope,
                "product_isotope": isotope,
                "line_energy_keV": float(energy_kev),
                "net_counts": max(float(net_counts), 1.0e-12),
                "live_time_s": max(float(live_time_s), 1.0),
                "efficiency": max(float(efficiency), 1.0e-12),
                "gamma_intensity": max(float(gamma_intensity), 1.0e-12),
                "half_life_s": max(float(half_life_s), 1.0),
                "sample_mass_g": max(float(sample_mass_g), 1.0e-12),
                "atomic_mass_g_mol": max(float(atomic_mass_g_mol), 1.0e-12),
                "isotopic_abundance": max(float(isotopic_abundance), 1.0e-12),
                "mass_fraction": max(float(mass_fraction), 1.0e-12),
                "effective_cross_section_barn": max(
                    float(effective_cross_section_barn), 1.0e-12
                ),
            }
        ],
    }
    result = analyze_astm_e261_plan(plan)
    row = result["measurements"][0]
    return float(row["reaction_rate_s"]), float(row["reaction_rate_unc_s"])


def compare_raw_astm_to_qg() -> int:
    paths = default_paths(Path(__file__).parent)
    config = json.loads(
        (paths.metadata_root / "workflow_config.json").read_text(encoding="utf-8")
    )

    energy_override = workflow_profile_energy_calibration(config)
    background_data = read_raw_asc(
        paths.background_path,
        energy_calibration_override=energy_override,
        profile_name=str(config["profile_name"]),
    )

    sample_files = pair_sample_files(
        paths.raw_root / "flux_wires", paths.qg_root / "flux_wires"
    )
    if not sample_files:
        print("No paired raw and QG samples found.")
        return 1

    print("\n" + "=" * 100)
    print("RAW DATA PIPELINE (ASTM E261) vs QG BENCHMARK (ASTM E2005 C/E)")
    print("=" * 100)

    printed_any = False
    for stem, raw_path, qg_path in sample_files:
        try:
            raw_data = read_raw_asc(
                raw_path,
                energy_calibration_override=energy_override,
                profile_name=str(config["profile_name"]),
            )
            qg_data = read_processed_txt(qg_path)
            qg_peaks = qg_reference_peaks(qg_data)
        except Exception as exc:
            print(f"[{stem}] load error: {exc}")
            continue

        expected_lines = _build_expected_lines(qg_peaks)
        raw_identified_peaks = analyze_raw_spectrum_targeted(
            data=raw_data,
            expected_lines=expected_lines,
            background_spectrum=background_data.spectrum,
            background_subtract=True,
            profile_name=str(config["profile_name"]),
        )

        rates_raw: dict[str, tuple[float, float]] = {}
        rates_qg: dict[str, tuple[float, float]] = {}

        for row in qg_peaks:
            isotope = str(row.get("isotope", ""))
            energy_kev = float(row.get("energy_keV") or row.get("energy_kev") or 0.0)
            if not isotope or energy_kev <= 0.0:
                continue

            qg_net = float(row.get("net_counts", 0.0) or 0.0)
            if qg_net <= 0.0:
                continue

            raw_peak = _find_raw_peak(
                raw_identified_peaks, isotope, energy_kev, tol=2.0
            )
            if raw_peak is None or float(raw_peak.net_counts) <= 0.0:
                continue

            reaction_id = f"{isotope}@{energy_kev:.1f}"
            gamma_intensity = float(
                row.get("rad_int_fraction")
                or max(float(row.get("rad_int_percent") or 0.0) / 100.0, 1.0e-6)
            )
            half_life_s = float(row.get("half_life_seconds") or 1.0)
            live_time_s = float(config.get("counting_time", 1.0) or 1.0)
            sample_mass_g = float(config.get("mass", 1.0) or 1.0)
            atomic_mass = float(config.get("atomic_weight", 1.0) or 1.0)
            iso_abund = float(row.get("isotopic_abundance", 1.0) or 1.0)
            mass_fraction = float(row.get("element_mass_fraction", 1.0) or 1.0)
            sigma = 1.0
            try:
                efficiency = float(raw_data.efficiency.efficiency(energy_kev))
            except Exception:
                efficiency = 1.0

            try:
                raw_rate, raw_unc = _e261_rate_from_counts(
                    reaction_id=f"RAW:{reaction_id}",
                    isotope=isotope,
                    energy_kev=energy_kev,
                    net_counts=float(raw_peak.net_counts),
                    net_unc=float(raw_peak.net_counts_unc),
                    live_time_s=live_time_s,
                    efficiency=efficiency,
                    gamma_intensity=gamma_intensity,
                    half_life_s=half_life_s,
                    sample_mass_g=sample_mass_g,
                    atomic_mass_g_mol=atomic_mass,
                    isotopic_abundance=iso_abund,
                    mass_fraction=mass_fraction,
                    effective_cross_section_barn=sigma,
                )
                qg_rate, qg_unc = _e261_rate_from_counts(
                    reaction_id=f"QG:{reaction_id}",
                    isotope=isotope,
                    energy_kev=energy_kev,
                    net_counts=qg_net,
                    net_unc=0.0,
                    live_time_s=live_time_s,
                    efficiency=efficiency,
                    gamma_intensity=gamma_intensity,
                    half_life_s=half_life_s,
                    sample_mass_g=sample_mass_g,
                    atomic_mass_g_mol=atomic_mass,
                    isotopic_abundance=iso_abund,
                    mass_fraction=mass_fraction,
                    effective_cross_section_barn=sigma,
                )
            except Exception:
                continue

            rates_raw[reaction_id] = (raw_rate, raw_unc)
            rates_qg[reaction_id] = (qg_rate, qg_unc)

        common = sorted(set(rates_raw) & set(rates_qg))
        if len(common) < 2:
            continue

        spectral_indices = []
        for a, b in combinations(common, 2):
            ra_raw, ua_raw = rates_raw[a]
            rb_raw, ub_raw = rates_raw[b]
            ra_qg, ua_qg = rates_qg[a]
            rb_qg, ub_qg = rates_qg[b]
            spectral_indices.append(
                {
                    "index_id": f"{a}/{b}",
                    "measured": {
                        "reaction_rate_a_s": ra_qg,
                        "reaction_rate_a_unc_s": ua_qg,
                        "reaction_rate_b_s": rb_qg,
                        "reaction_rate_b_unc_s": ub_qg,
                    },
                    "calculated": {
                        "reaction_rate_a_s": ra_raw,
                        "reaction_rate_a_unc_s": ua_raw,
                        "reaction_rate_b_s": rb_raw,
                        "reaction_rate_b_unc_s": ub_raw,
                    },
                }
            )

        e2005_plan = {
            "title": f"ASTM E2005 RAW-vs-QG {stem}",
            "spectral_indices": spectral_indices,
        }

        try:
            e2005 = analyze_astm_e2005_plan(e2005_plan)
        except Exception as exc:
            print(f"[{stem}] E2005 error: {exc}")
            continue

        printed_any = True
        print(f"\n>>> Sample: {stem}")
        print(
            f"  Lines used: {len(common)} | Spectral-index pairs: {len(e2005.get('spectral_indices', []))}"
        )
        for item in e2005.get("spectral_indices", []):
            print(
                f"    {item['index_id']}: "
                f"meas={item['measured_index']:.4e}, cal={item['calculated_index']:.4e}, "
                f"C/E={item['c_e_ratio']:.4f} ± {item['c_e_ratio_unc']:.4f}"
            )

    if not printed_any:
        print(
            "\nNo samples produced >=2 matched reactions for ASTM E2005 spectral-index comparison."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(compare_raw_astm_to_qg())
