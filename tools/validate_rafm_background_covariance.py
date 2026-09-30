"""Replay committed UWNR Co monitors and save checked covariance evidence.

Run with an unused --output-root. Uses installed local dependencies only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fluxforge.analysis.flux_wire_analysis import (
    GammaLine,
    analyze_raw_spectrum_targeted,
    estimate_peak_area_local_background,
)
from fluxforge.analysis.hpge_processor import HPGeProcessor
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.examples.rafm_workflow import (
    analyze_flux_wire_sample,
    default_paths,
    ensure_results_tree,
    load_rafm_example_metadata,
    normalize_pairing_key,
)
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.io.genie import read_genie_spectrum


def independent_variance(sample, background, weights, scale):
    """Apply np.interp to independent source basis vectors, without sparse W."""
    active = np.flatnonzero(weights)
    basis = np.zeros(len(background.counts))
    variance = float(np.sum((weights * sample.counts_uncertainty) ** 2))
    for index, uncertainty in enumerate(background.counts_uncertainty):
        basis[index] = 1.0
        mapped = np.interp(
            sample.energies[active], background.energies, basis, left=0, right=0
        )
        variance += float((scale * uncertainty * (weights[active] @ mapped)) ** 2)
        basis[index] = 0.0
    return variance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    output = parser.parse_args().output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    example = ROOT / "examples" / "RAFM_irradiation"
    background_path = example / "background.ASC"
    metadata = load_rafm_example_metadata(example)
    # Independent raw counting: no QG reference report or activity substitution.
    metadata.config["flux_wire_counting_method"] = "covell"
    paths = default_paths(example, results_root=output / "raw_covell")
    tree = ensure_results_tree(paths.results_root)
    raw_background = read_raw_asc(background_path, profile_name="rafm_25cm").spectrum
    assert raw_background is not None
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            name: digest(ROOT / name)
            for name in [
                "src/fluxforge/io/spe.py",
                "src/fluxforge/analysis/spectrum_math.py",
                "src/fluxforge/analysis/flux_wire_analysis.py",
                "src/fluxforge/analysis/hpge_processor.py",
                "src/fluxforge/analysis/peakfit.py",
            ]
        },
        "background_sha256": digest(background_path),
        "samples": [],
        "limits": [
            "HPGe replay uses its default unity efficiency; its activity values are diagnostic, not calibrated UWNR activities.",
            "HPGe uses GLS counting covariance and a conservative ROI floor; model inadequacy still requires residual review.",
            "SNIP continuum-model uncertainty and full efficiency/emission budgets remain incomplete.",
            "Covell and other comparison count windows retain their existing method semantics.",
            "Count covariance across different samples sharing one background is not exported here.",
        ],
    }
    for stem in ["Co-RAFM-1_25cm", "Co-Cd-RAFM-1_25cm"]:
        raw_path = paths.raw_root / "flux_wires" / f"{stem}.ASC"
        sample = read_genie_spectrum(raw_path)
        background = read_genie_spectrum(background_path)
        corrected = subtract_measured_background(sample, background)
        scale = sample.live_time / background.live_time
        with warnings.catch_warnings(record=True):
            hpge = HPGeProcessor().analyze(
                sample, known_isotopes=["Co-60"], background_spectrum=background
            )
        assert len(hpge.gamma_lines) == 2
        row = {
            "sample": stem,
            "sample_sha256": digest(raw_path),
            "scale": scale,
            "negative_bins": int(np.count_nonzero(corrected.counts < 0)),
            "covariance_stored_entries": corrected.counts_covariance.nnz,
            "hpge": [],
            "targeted": {},
        }
        for line in hpge.gamma_lines:
            lo, hi = line.fit_result.fit_region
            weights = ((sample.channels >= lo) & (sample.channels <= hi)).astype(float)
            expected = independent_variance(sample, background, weights, scale)
            actual = corrected.weighted_counts_variance(weights)
            np.testing.assert_allclose(actual, expected, rtol=1e-10)
            assert line.net_counts_unc >= np.sqrt(expected) - 1e-8
            assert (
                line.activity_unc
                >= line.activity * np.sqrt(expected) / line.net_counts - 1e-8
            )
            row["hpge"].append(
                {
                    "energy_keV": line.energy,
                    "fit_region": [int(lo), int(hi)],
                    "net_counts": line.net_counts,
                    "net_counts_unc": line.net_counts_unc,
                    "centroid_channel": line.fit_result.peak.centroid,
                    "fwhm_channels": line.fit_result.peak.fwhm,
                    "reduced_chi_squared": line.fit_result.reduced_chi_squared,
                    "signed_window_counts": float(
                        np.sum(
                            sample.counts[lo : hi + 1]
                            - scale
                            * np.interp(
                                sample.energies[lo : hi + 1],
                                background.energies,
                                background.counts,
                                left=0,
                                right=0,
                            )
                        )
                    ),
                    "diagonal_roi_unc": float(
                        np.sqrt(np.sum((weights * corrected.counts_uncertainty) ** 2))
                    ),
                    "covariance_roi_unc": float(np.sqrt(actual)),
                    "oracle_roi_unc": float(np.sqrt(expected)),
                    "activity_bq": line.activity,
                    "activity_unc_bq": line.activity_unc,
                }
            )
        data = read_raw_asc(raw_path, profile_name="rafm_25cm")
        raw_corrected = subtract_measured_background(data.spectrum, raw_background)
        lines = [
            GammaLine(energy_keV=e, intensity=i, isotope="Co60")
            for e, i in [(1173.23, 0.9985), (1332.49, 0.9998)]
        ]
        for method in ["qg", "covell", "gilmore", "iec_tiered"]:
            peaks = analyze_raw_spectrum_targeted(
                data,
                lines,
                background_spectrum=raw_background,
                profile_name="rafm_25cm",
                counting_method=method,
            )
            assert len(peaks) == 2
            results = []
            for peak in peaks:
                slope = (
                    data.energy_calibration[1]
                    + 2 * data.energy_calibration[2] * peak.channel
                )
                _, uncertainty, _, _, (lo, hi) = estimate_peak_area_local_background(
                    raw_corrected.counts,
                    peak.channel,
                    peak.fwhm / slope,
                    spectrum_data=raw_corrected,
                )
                weights = np.zeros(len(sample.counts))
                weights[lo : hi + 1] = 1
                weights[[lo - 1, hi + 1]] = -(hi - lo + 1) / 2
                oracle = independent_variance(
                    data.spectrum, raw_background, weights, scale
                )
                np.testing.assert_allclose(uncertainty**2, oracle, rtol=1e-10)
                assert peak.net_counts_unc >= uncertainty - 1e-8
                results.append(
                    {
                        "energy_keV": peak.energy_keV,
                        "net_counts": peak.net_counts,
                        "net_counts_unc": peak.net_counts_unc,
                        "roi_unc": uncertainty,
                        "oracle_roi_unc": float(np.sqrt(oracle)),
                        "activity_bq": peak.activity_bq,
                        "activity_unc_bq": peak.activity_unc_bq,
                    }
                )
            row["targeted"][method] = results
        sample_key = normalize_pairing_key(stem, metadata.pairing_aliases)
        artifact = analyze_flux_wire_sample(
            raw_path, metadata, paths, tree, raw_background, None, sample_key
        )
        mass = artifact["reaction_rate_mass_metadata"]
        assert mass["alloy_co_mass_fraction"] == 0.0046
        assert mass["mass_basis"] == "element_mass"
        assert mass["element_mass_fraction"] == 1
        expected_atoms = mass["mass_mg"] / 1000 / 58.9332 * 6.02214076e23
        reaction = next(
            r for r in artifact["reactions"] if r["reaction_id"] == "Co-59(n,g)Co-60"
        )
        np.testing.assert_allclose(reaction["n_atoms"], expected_atoms, rtol=1e-6)
        row["raw_covell_workflow"] = {
            "mass_metadata": mass,
            "reaction": reaction,
            "artifact": str(tree["artifacts"] / f"{stem}.json"),
            "independent_of_qg_report": True,
        }
        receipt["samples"].append(row)
    receipt["status"] = "PASS"
    target = output / "covariance_review_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt": str(target),
                "samples": len(receipt["samples"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
