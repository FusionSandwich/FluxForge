"""Fixed-energy Poisson ROI injection/blank study, not blind-search validation."""

import argparse
import json
from pathlib import Path

import numpy as np
from fluxforge.analysis.flux_wire_analysis import estimate_peak_area_local_background


VARIANTS = {
    "baseline": (4.0, 1, 0.0),
    "wider_sidebands": (4.0, 8, 0.5),
    "narrower_roi": (2.5, 1, 0.0),
    "narrower_roi_wider_sidebands": (2.5, 8, 0.5),
}


def simulate(trials=4000, seed=30261003):
    rng = np.random.default_rng(seed)
    center, fwhm, background = 64, 4.0, 100.0
    x = np.arange(129)
    gaussian = np.exp(-0.5 * ((x - center) / (fwhm / 2.354820045)) ** 2)
    gaussian /= gaussian.sum()
    result = {}
    for area in (0.0, 200.0, 10000.0):
        observed = rng.poisson(background + area * gaussian, size=(trials, len(x)))
        scores = []
        for name, (width, sidebands, gap) in VARIANTS.items():
            estimates = [
                estimate_peak_area_local_background(
                    row,
                    center,
                    fwhm_channels=fwhm,
                    roi_width_fwhm=width,
                    background_width_channels=sidebands,
                    background_gap_fwhm=gap,
                    spectrum_uncertainty=np.sqrt(row),
                )
                for row in observed
            ]
            net = np.asarray([r[0] for r in estimates])
            sigma = np.asarray([r[1] for r in estimates])
            z = net / sigma
            scores.append(z)
            lo, hi = estimates[0][4]
            # Expected estimator indication includes any true signal falling
            # in the sidebands. Compare uncertainty to this expectation.
            mean_estimate = estimate_peak_area_local_background(
                background + area * gaussian,
                center,
                fwhm_channels=fwhm,
                roi_width_fwhm=width,
                background_width_channels=sidebands,
                background_gap_fwhm=gap,
            )[0]
            result.setdefault(str(area), {})[name] = dict(
                detection_fraction=float(np.mean(z >= 2.0)),
                average_net=float(np.mean(net)),
                expected_net=float(mean_estimate),
                average_uncertainty=float(np.mean(sigma)),
                empirical_standard_deviation=float(np.std(net, ddof=1)),
                coverage_within_1_sigma=float(
                    np.mean(abs(net - mean_estimate) <= sigma)
                ),
                peak_capture_fraction=float(gaussian[lo : hi + 1].sum()),
            )
        result[str(area)]["choose_best_of_four"] = {
            "detection_fraction": float(np.mean(np.max(scores, axis=0) >= 2.0)),
            "warning": "Exploratory per-spectrum selection increases false positives; not a calibrated decision rule",
        }
    return dict(
        seed=seed,
        trials=trials,
        scientific_admission=False,
        scope="Fixed known energy and FWHM; flat independent Poisson continuum; excludes peak-search selection, tails, interferences, measured-background covariance and drift",
        results=result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=4000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.trials < 1000:
        raise ValueError("Use at least 1000 trials")
    payload = simulate(args.trials)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
