"""Bounded source screen and known-truth receipt; run with PYTHONPATH=src.

python examples/validation/multiplet_240/run_example.py --output FRESH.json
No activity, QuantumGold target tuning, or shared workflow changes occur.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from fluxforge.analysis import multiplet_validation, peakfit
from fluxforge.analysis.multiplet_validation import qualify_doublet
from fluxforge.io.genie import read_genie_spectrum
from fluxforge.validation.example_identity import source_identity


SOURCE_HASH = "4f2aadb52bcab511b6e8da042ebe689a93f4f6007a481b3f6bcac5c0238188d3"
ENGINE_BASE = "a7bcc680d1f5e06b1d9dae405241fc380087ca2b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).parent / "fixtures" / "RAFM-N-300s.ASC"
    if sha(source) != SOURCE_HASH:
        raise ValueError("source SHA-256 mismatch")
    spectrum = read_genie_spectrum(source)
    x = np.arange(len(spectrum.counts), dtype=float)
    y = spectrum.counts
    a, b, c = spectrum.calibration["energy"]
    energy = a + b * x + c * x * x
    profile_path = root / "src/fluxforge/data/rafm_profiles.json"
    profile = json.loads(profile_path.read_text())["rafm_25cm"]
    resolution = np.asarray(profile["resolution"])

    def sigma_at(channel):
        e = a + b * channel + c * channel * channel
        return np.polynomial.polynomial.polyval(e, resolution) / (
            (b + 2 * c * channel) * peakfit.FWHM_SIG_RATIO
        )

    # Fixed exploratory morphology, not a nuclide search. Smoothing is used
    # only for screening; every fitted count is original and un-subtracted.
    smooth = gaussian_filter1d(y.astype(float), 1.0)
    peaks, props = find_peaks(
        smooth, prominence=8 * np.sqrt(np.maximum(smooth, 1)), distance=2
    )
    in_range = (energy[peaks] >= 100) & (energy[peaks] <= 1800)
    peaks = peaks[in_range]
    prominences = props["prominences"][in_range]
    candidates = []
    for u, v in zip(peaks[:-1], peaks[1:]):
        separation = (v - u) / (peakfit.FWHM_SIG_RATIO * sigma_at((u + v) / 2))
        if 0.5 <= separation <= 2.0:
            candidates.append([int(u), int(v)])
    if candidates:
        centers = np.asarray(candidates[0], float)
        selection = "first_observed_morphological_pair_no_independent_line_identity"
    else:
        # Explicit unsupported-extra-component control around the strongest
        # observed isolated peak. These seeds are not evidence of two lines.
        center = float(peaks[np.argmax(prominences)])
        centers = np.array([center - sigma_at(center), center + sigma_at(center)])
        selection = "single_peak_control_no_observed_overlap_pair"
    mid = float(np.mean(centers))
    sigma = float(sigma_at(mid))
    margin = int(np.ceil(6 * 1.2 * sigma))
    lo = max(0, int(np.floor(centers[0])) - margin)
    hi = min(len(x) - 1, int(np.ceil(centers[1])) + margin)
    real = qualify_doublet(
        x[lo : hi + 1],
        y[lo : hi + 1],
        centers,
        sigma_bounds=(0.8 * sigma, 1.2 * sigma),
        shift_limit=1.0,
        resolution_evidence=f"conditional rafm_25cm profile sha256:{sha(profile_path)}",
        component_evidence=(None, None),
        continuum="linear",
    ).to_dict()
    # This example does not qualify the profile for this ASC acquisition.
    # The fit is a conditional diagnostic; no component may be admitted.
    assert real["component_admission"] == [False, False]
    synthetic = []
    sx = np.arange(60, 141, dtype=float)
    for name, centers in [
        ("separated", (94, 106)),
        ("partial", (98, 102)),
        ("unresolved", (99.8, 100.2)),
        ("single", (98, 102)),
    ]:
        sy = np.full_like(sx, 40.0)
        truth = [6000, 4000]
        for center, area in zip(centers, truth):
            if name != "single":
                sy += peakfit.gaussian(sx, area / (2 * np.sqrt(2 * np.pi)), center, 2)
        if name == "single":
            sy += peakfit.gaussian(sx, 10000 / (2 * np.sqrt(2 * np.pi)), 100.35, 2)
            truth = [10000]
        result = qualify_doublet(
            sx,
            sy,
            centers,
            sigma_bounds=(1.8, 2.2),
            resolution_evidence="known synthetic truth sigma=2 channels",
            component_evidence=("known truth A", "known truth B"),
            counts_uncertainty=np.sqrt(sy),
            count_basis="synthetic",
        ).to_dict()
        synthetic.append(dict(case=name, true_areas=truth, result=result))
    identity = source_identity(
        root,
        [
            "src/fluxforge/analysis/peakfit.py",
            "src/fluxforge/analysis/multiplet_validation.py",
            "examples/validation/multiplet_240/run_example.py",
            "src/fluxforge/io/genie.py",
            "tests/test_multiplet_validation.py",
        ],
    )
    return dict(
        engine_base=ENGINE_BASE,
        checked_out_head=identity["revision"],
        source_identity=identity,
        engine_identity={
            name: sha(path)
            for name, path in {
                "peakfit": peakfit.__file__,
                "multiplet_validation": multiplet_validation.__file__,
                "example": __file__,
                "genie_reader": root / "src/fluxforge/io/genie.py",
                "targeted_tests": root / "tests/test_multiplet_validation.py",
            }.items()
        },
        runtime=dict(
            python=platform.python_version(),
            executable=sys.executable,
            numpy=np.__version__,
            scipy=scipy.__version__,
            limitation="installed NumPy 2.5.1 is outside declared NumPy<2; no environment changes",
        ),
        source=dict(
            relative_path=str(source.relative_to(root)),
            sha256=SOURCE_HASH,
            origin_commit="46096eb",
            origin_path="examples/RAFM_irradiation/quantumgold_reference/originals/ASC/RAFM-N-300s.ASC",
            byte_identical_copy=True,
            energy_calibration=spectrum.calibration["energy"],
            live_time=spectrum.live_time,
            count_basis="raw_sample_no_ambient_subtraction",
            findings_sha256="6fa1a3e928010b5b03f7ba453e3b6a2212d2254c65ed85e3ea0bc6abb63dbb42",
        ),
        screen=dict(
            file_count=1,
            energy_range_keV=[100, 1800],
            smoothing_sigma_channels=1,
            prominence_rule="8*sqrt(smoothed_count)",
            separation_fwhm_range=[0.5, 2.0],
            observed_peak_channels=peaks.tolist(),
            close_pairs=candidates,
            selection=selection,
        ),
        resolution=dict(
            coefficients_keV=resolution.tolist(),
            profile_sha256=sha(profile_path),
            width_bounds_fraction=[0.8, 1.2],
            status="conditional_not_acquisition_qualified",
            limitation="profile uses saved QG settings; source ASC energy calibration differs; final report/acquisition resolution applicability unknown",
        ),
        real_roi=dict(
            channels=[lo, hi],
            energy_keV=[float(energy[lo]), float(energy[hi])],
            counts=y[lo : hi + 1].tolist(),
            result=real,
        ),
        conclusion="No qualified real overlap in this bounded one-file screen; retain synthetic validated method. No claim about the other study files.",
        synthetic=synthetic,
        integration="opt-in module only; physical-analysis and historical-comparison workflows unchanged; #232/#220 integration pending",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                real_status=receipt["real_roi"]["result"]["status"],
                close_pairs=receipt["screen"]["close_pairs"],
                synthetic_statuses=[
                    s["result"]["status"] for s in receipt["synthetic"]
                ],
            )
        )
    )
