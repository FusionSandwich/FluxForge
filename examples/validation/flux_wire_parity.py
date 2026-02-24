#!/usr/bin/env python3
"""
Flux wire parity validation (raw vs processed).

Generates CSV/JSON summaries under artifacts/validation/flux_wire_parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
from typing import Dict, List

from fluxforge.analysis.flux_wire_analysis import analyze_flux_wire_targeted
from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc, FluxWireData


def normalize_sample_id(sample_id: str) -> str:
    text = sample_id.strip().replace(" ", "")
    text = text.replace("@", "_")
    text = re.sub(r"_?\d+cm$", "", text, flags=re.IGNORECASE)
    return text.lower()


def load_processed_files(proc_dir: Path) -> Dict[str, FluxWireData]:
    processed: Dict[str, FluxWireData] = {}
    for path in sorted(proc_dir.glob("*.txt")):
        data = read_processed_txt(path)
        processed[normalize_sample_id(data.sample_id)] = data
    return processed


def load_raw_files(raw_dir: Path) -> Dict[str, FluxWireData]:
    raw: Dict[str, FluxWireData] = {}
    for path in sorted(raw_dir.glob("*.ASC")):
        data = read_raw_asc(path)
        raw[normalize_sample_id(data.sample_id)] = data
    return raw


def apply_reference_calibration(raw: FluxWireData, reference: FluxWireData) -> None:
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
class ParityRow:
    sample_id: str
    isotope: str
    ref_bq: float
    calc_bq: float
    ratio: float
    diff_pct: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "isotope": self.isotope,
            "ref_bq": self.ref_bq,
            "calc_bq": self.calc_bq,
            "ratio": self.ratio,
            "diff_pct": self.diff_pct,
        }


def run_parity(proc_dir: Path, raw_dir: Path) -> Dict[str, object]:
    processed = load_processed_files(proc_dir)
    raw = load_raw_files(raw_dir)
    
    rows: List[ParityRow] = []
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
            rows.append(ParityRow(
                sample_id=raw_data.sample_id,
                isotope=nuclide.isotope,
                ref_bq=ref_bq,
                calc_bq=calc_bq,
                ratio=ratio,
                diff_pct=diff_pct,
            ))
    
    ratios = [r.ratio for r in rows if r.ref_bq > 0]
    summary = {
        "comparisons": len(rows),
        "missing": missing,
        "ratio_mean": float(sum(ratios) / len(ratios)) if ratios else 0.0,
        "ratio_std": float((sum((x - (sum(ratios) / len(ratios))) ** 2 for x in ratios) / len(ratios)) ** 0.5) if ratios else 0.0,
        "max_abs_diff_pct": float(max(abs(r.diff_pct) for r in rows)) if rows else 0.0,
    }
    
    return {"rows": rows, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Flux wire parity validation")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[2]
    alara_root = repo_root.parent
    
    proc_dir = args.processed_dir or (alara_root / "rafm_irradiation_ldrd" / "irradiation_QG_processed" / "flux_wires")
    raw_dir = args.raw_dir or (alara_root / "rafm_irradiation_ldrd" / "raw_gamma_spec" / "flux_wires")
    output_dir = args.output_dir or (repo_root / "artifacts" / "validation" / "flux_wire_parity")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_parity(proc_dir, raw_dir)
    rows: List[ParityRow] = results["rows"]
    summary = results["summary"]
    
    csv_path = output_dir / "raw_vs_processed_parity.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_id,isotope,ref_bq,calc_bq,ratio,diff_pct\n")
        for row in rows:
            f.write(
                f"{row.sample_id},{row.isotope},{row.ref_bq:.6e},"
                f"{row.calc_bq:.6e},{row.ratio:.6f},{row.diff_pct:.3f}\n"
            )
    
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "rows": [r.to_dict() for r in rows],
        }, f, indent=2)
    
    print("Flux wire parity summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
