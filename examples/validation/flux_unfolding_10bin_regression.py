#!/usr/bin/env python3
"""
10-bin flux unfolding regression (processed vs raw).

Generates a combined plot and machine-readable outputs under
artifacts/validation/flux_unfolding_10bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from fluxforge.analysis.flux_unfold import (
    extract_reactions_from_processed,
    extract_reactions_from_raw,
    unfold_discrete_bins,
    unfold_gls,
)
from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc, FluxWireData


def normalize_sample_id(sample_id: str) -> str:
    text = sample_id.strip().replace(" ", "")
    text = text.replace("@", "_")
    text = re.sub(r"_?\d+cm$", "", text, flags=re.IGNORECASE)
    return text.lower()


def load_processed(proc_dir: Path) -> Dict[str, FluxWireData]:
    processed: Dict[str, FluxWireData] = {}
    for path in sorted(proc_dir.glob("*.txt")):
        data = read_processed_txt(path)
        processed[normalize_sample_id(data.sample_id)] = data
    return processed


def load_raw(raw_dir: Path) -> Dict[str, FluxWireData]:
    raw: Dict[str, FluxWireData] = {}
    for path in sorted(raw_dir.glob("*.ASC")):
        data = read_raw_asc(path)
        raw[normalize_sample_id(data.sample_id)] = data
    return raw


def load_model_spectrum(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return None
    
    def pick_column(candidates: List[str]) -> Optional[np.ndarray]:
        for name in data.dtype.names:
            lower = name.lower()
            for cand in candidates:
                if cand in lower:
                    return data[name]
        return None
    
    e_low = pick_column(["e_low"])
    e_high = pick_column(["e_high"])
    flux = pick_column(["flux_per_lethargy", "flux_per_energy", "flux"])
    
    if e_low is None or e_high is None or flux is None:
        return None
    
    return {"E_low": e_low, "E_high": e_high, "flux": flux}


def save_spectrum_csv(path: Path, energy_bounds_eV: np.ndarray, flux: np.ndarray, flux_unc: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["E_low[eV]", "E_high[eV]", "flux", "flux_unc"])
        for i in range(len(flux)):
            writer.writerow([energy_bounds_eV[i], energy_bounds_eV[i + 1], flux[i], flux_unc[i]])


def plot_comparison(
    processed_discrete,
    processed_gls,
    raw_discrete,
    raw_gls,
    model: Optional[Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    def plot_panel(ax, discrete, gls, title: str) -> None:
        centers = np.sqrt(discrete.energy_bounds_eV[:-1] * discrete.energy_bounds_eV[1:])
        ax.step(discrete.energy_bounds_eV[:-1], discrete.flux, where="post", label="Discrete 10-bin", linewidth=1.5)
        gls_centers = np.sqrt(gls.energy_bounds_eV[:-1] * gls.energy_bounds_eV[1:])
        ax.plot(gls_centers, gls.flux, label="GLS continuous", linewidth=1.5)
        ax.fill_between(gls_centers, gls.flux - gls.flux_unc, gls.flux + gls.flux_unc, alpha=0.2)
        
        if model is not None:
            model_centers = np.sqrt(model["E_low"] * model["E_high"])
            ax.plot(model_centers, model["flux"], label="Model spectrum", linewidth=1.2, alpha=0.8)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy (eV)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plot_panel(axes[0], processed_discrete, processed_gls, "Processed inputs")
    plot_panel(axes[1], raw_discrete, raw_gls, "Raw spectra inputs")
    axes[0].set_ylabel("Flux (arbitrary)")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


@dataclass
class UnfoldingResults:
    processed_discrete: object
    processed_gls: object
    raw_discrete: object
    raw_gls: object
    model: Optional[Dict[str, np.ndarray]]


def run_unfolding_regression(proc_dir: Path, raw_dir: Path, model_path: Path) -> UnfoldingResults:
    processed = load_processed(proc_dir)
    raw = load_raw(raw_dir)
    
    keys = sorted(set(processed.keys()) & set(raw.keys()))
    if not keys:
        raise RuntimeError("No matching raw/processed flux wire pairs found")
    
    processed_reactions = []
    raw_reactions = []
    
    for key in keys:
        proc = processed[key]
        raw_data = raw[key]
        
        processed_reactions.extend(
            extract_reactions_from_processed(
                proc,
                irradiation_time_s=8 * 3600,
                decay_time_s=4 * 3600,
            )
        )
        raw_reactions.extend(
            extract_reactions_from_raw(
                raw_data,
                reference_data=proc,
                irradiation_time_s=8 * 3600,
                decay_time_s=4 * 3600,
            )
        )
    
    processed_discrete = unfold_discrete_bins(processed_reactions, n_bins=10)
    processed_gls = unfold_gls(processed_reactions, n_groups=50)
    raw_discrete = unfold_discrete_bins(raw_reactions, n_bins=10)
    raw_gls = unfold_gls(raw_reactions, n_groups=50)
    
    model = load_model_spectrum(model_path)
    
    return UnfoldingResults(
        processed_discrete=processed_discrete,
        processed_gls=processed_gls,
        raw_discrete=raw_discrete,
        raw_gls=raw_gls,
        model=model,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="10-bin unfolding regression")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--model-spectrum", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[2]
    alara_root = repo_root.parent
    
    proc_dir = args.processed_dir or (alara_root / "rafm_irradiation_ldrd" / "irradiation_QG_processed" / "flux_wires")
    raw_dir = args.raw_dir or (alara_root / "rafm_irradiation_ldrd" / "raw_gamma_spec" / "flux_wires")
    model_path = args.model_spectrum or (alara_root / "rafm_irradiation_ldrd" / "raw_gamma_spec" / "flux_wires" / "spectrum_vit_j.csv")
    output_dir = args.output_dir or (repo_root / "artifacts" / "validation" / "flux_unfolding_10bin")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_unfolding_regression(proc_dir, raw_dir, model_path)
    
    # Save spectra
    save_spectrum_csv(
        output_dir / "processed_discrete.csv",
        results.processed_discrete.energy_bounds_eV,
        results.processed_discrete.flux,
        results.processed_discrete.flux_unc,
    )
    save_spectrum_csv(
        output_dir / "processed_gls.csv",
        results.processed_gls.energy_bounds_eV,
        results.processed_gls.flux,
        results.processed_gls.flux_unc,
    )
    save_spectrum_csv(
        output_dir / "raw_discrete.csv",
        results.raw_discrete.energy_bounds_eV,
        results.raw_discrete.flux,
        results.raw_discrete.flux_unc,
    )
    save_spectrum_csv(
        output_dir / "raw_gls.csv",
        results.raw_gls.energy_bounds_eV,
        results.raw_gls.flux,
        results.raw_gls.flux_unc,
    )
    
    # Save summary JSON
    summary = {
        "processed_reactions": len(results.processed_discrete.flux),
        "raw_reactions": len(results.raw_discrete.flux),
        "model_spectrum": str(model_path) if results.model else None,
        "processed_gls_chi2": results.processed_gls.chi2,
        "raw_gls_chi2": results.raw_gls.chi2,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Plot comparison
    plot_comparison(
        results.processed_discrete,
        results.processed_gls,
        results.raw_discrete,
        results.raw_gls,
        results.model,
        output_dir / "unfolding_comparison.png",
    )
    
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
