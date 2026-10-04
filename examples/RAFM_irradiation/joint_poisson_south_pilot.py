"""Source-bound South-background Co-Cd joint-Poisson model-check pilot (#249).

This is an opt-in diagnostic. It never loads QuantumGold target activities and it
never changes RAFM workflow defaults. The recovered South background remains a
later, temporally unqualified measurement. Numerical convergence is kept
separate from model adequacy and physical qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import csv
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import scipy

from fluxforge.analysis.flux_wire_analysis import estimate_peak_area_local_background
from fluxforge.analysis.joint_poisson import (
    BackgroundChoice,
    CountObservation,
    PeakResponse,
    fit_joint_poisson,
)
from fluxforge.analysis.peakfit import FWHM_SIG_RATIO
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.data.rafm_profile import load_rafm_profile
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.io.spe import GammaSpectrum

BASE_HEAD = "701288a30f418ed7dc331e6deda0044eb220e72e"
SAMPLE_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/originals/ASC/Co-Cd-RAFM-1.ASC"
)
SOUTH_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/"
    "South 4hr Background Terminal.ANS"
)
MANIFEST_REL = Path("examples/RAFM_irradiation/quantumgold_reference/manifest.json")
SUPPLEMENT_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/manifest.json"
)
CONFIG_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/runtime/metadata/workflow_config.json"
)
PROFILE_REL = Path("src/fluxforge/data/rafm_profiles.json")
IEC_RECEIPT_REL = Path(
    "artifacts/validation/quantumgold_integration_20261004/south_native/"
    "CURRENT_LINE_COMBINATIONS.json"
)
SOUTH_SHA256 = "96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344"
SAMPLE_SHA256 = "7482016df6c9d370b68e5d5646cd64c5fa74def9fe586e58d19d89dab0c87cba"
CO60_LINES = ((1173.228, 0.9985), (1332.492, 0.999826))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def midpoint_edges(energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    return np.r_[
        energies[0] - (energies[1] - energies[0]) / 2,
        (energies[:-1] + energies[1:]) / 2,
        energies[-1] + (energies[-1] - energies[-2]) / 2,
    ]


def native_observation(data, path: Path, edges, lo: int, hi: int) -> CountObservation:
    return CountObservation(
        data.spectrum.counts[lo : hi + 1],
        edges[lo : hi + 2],
        data.live_time,
        sha(path),
        data.start_time.isoformat() if data.start_time else None,
    )


def south_native_background() -> tuple[GammaSpectrum, dict]:
    supplement_path = ROOT / SUPPLEMENT_REL
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    pins = [
        item
        for item in supplement["resources"]
        if item["role"] == "recovered_South_native_background_not_ASC"
    ]
    if len(pins) != 1:
        raise ValueError("South source identity is ambiguous")
    pin = pins[0]
    if Path(pin["path"]) != SOUTH_REL or pin["sha256"] != SOUTH_SHA256:
        raise ValueError("South source manifest binding changed")
    path = ROOT / SOUTH_REL
    blob = path.read_bytes()
    if len(blob) != pin["bytes"] or sha(path) != pin["sha256"]:
        raise ValueError("South source identity mismatch")
    if len(blob) != 36616 or b"South 4 hr background terminal" not in blob[:1548]:
        raise ValueError("Unsupported South background layout")

    coefficients = struct.unpack_from("<3f", blob, 424)
    real_time = struct.unpack_from("<d", blob, 96)[0]
    live_time = struct.unpack_from("<d", blob, 104)[0]
    serial = struct.unpack_from("<d", blob, 80)[0]
    counts = np.asarray(struct.unpack_from("<8192I", blob, 1548), dtype=float)
    channels = np.arange(8192)
    energies = (
        coefficients[0]
        + coefficients[1] * channels
        + coefficients[2] * channels**2
    )
    if live_time != 14400 or not live_time <= real_time < 14500:
        raise ValueError("Unsupported South background timing")
    if int(counts.sum()) != 543427 or not np.all(np.diff(energies) > 0):
        raise ValueError("Unsupported South count/calibration anchors")
    start = datetime(1899, 12, 30) + timedelta(days=serial)
    details = {
        "detector": "South",
        "source_path": SOUTH_REL.as_posix(),
        "source_sha256": sha(path),
        "start_time_local_unzoned": start.isoformat(),
        "live_time_s": live_time,
        "real_time_s": real_time,
        "energy_polynomial_keV": list(coefficients),
        "channel_count": 8192,
        "count_sum": int(counts.sum()),
        "payload_offset": 1548,
        "temporal_applicability": "UNRESOLVED",
        "calibration_covariance": "UNAVAILABLE",
        "layout_status": "empirical_source-bound_layout_not_vendor_verified",
    }
    return (
        GammaSpectrum(
            counts=counts,
            channels=channels,
            energies=energies,
            live_time=live_time,
            real_time=real_time,
            start_time=start,
            spectrum_id="South measured source-bound background",
            detector_id="South",
            calibration={"energy": list(coefficients)},
            metadata=details,
        ),
        details,
    )


def _deviance_residuals(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    term = np.empty_like(observed)
    positive = observed > 0
    term[positive] = (
        observed[positive] * np.log(observed[positive] / expected[positive])
        - (observed[positive] - expected[positive])
    )
    term[~positive] = expected[~positive]
    return np.sign(observed - expected) * np.sqrt(np.maximum(2 * term, 0))


def _residual_summary(observed, expected, edges) -> dict:
    dev = _deviance_residuals(observed, expected)
    pearson = (np.asarray(observed) - np.asarray(expected)) / np.sqrt(
        np.maximum(expected, 1e-12)
    )
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    order = np.argsort(np.abs(dev))[::-1][:5]
    return {
        "max_abs_deviance_residual": float(np.max(np.abs(dev))),
        "n_abs_deviance_residual_gt3": int(np.count_nonzero(np.abs(dev) > 3)),
        "n_abs_deviance_residual_gt5": int(np.count_nonzero(np.abs(dev) > 5)),
        "deviance_residuals": dev,
        "pearson_residuals": pearson,
        "largest_bins": [
            {
                "energy_keV": float(centers[i]),
                "observed": float(observed[i]),
                "expected": float(expected[i]),
                "deviance_residual": float(dev[i]),
            }
            for i in order
        ],
    }


def _joint_payload(fit, observation, background_observation) -> dict:
    return {
        "success": fit.success,
        "status": fit.status,
        "message": fit.message,
        "area_full_response_counts": fit.area,
        "normalization": fit.normalization,
        "exposure_scale": fit.exposure_scale,
        "profile": asdict(fit.interval) if fit.interval else None,
        "deviance": fit.deviance,
        "identifiability_ratio": fit.identifiability_ratio,
        "model_diagnostics": fit.model_diagnostics,
        "parameter_names": list(fit.parameter_names),
        "parameters": fit.parameters,
        "sample_expected": fit.sample_expected,
        "background_expected": fit.background_expected,
        "sample_residuals": fit.sample_residuals,
        "background_residuals": fit.background_residuals,
        "sample_residual_diagnostics": _residual_summary(
            observation.counts, fit.sample_expected, observation.energy_edges_keV
        ),
        "background_residual_diagnostics": (
            _residual_summary(
                background_observation.counts,
