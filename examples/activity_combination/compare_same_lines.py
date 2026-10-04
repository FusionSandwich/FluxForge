"""Bounded equation controls; no spectrum campaigns or target fitting.

Run from the checkout: python examples/activity_combination/compare_same_lines.py
Optionally use --output NEW.json; an existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fluxforge.analysis import activity_combination as implementation
from fluxforge.analysis.activity_combination import (
    ActivityLine,
    CovarianceComponent,
    METHODS,
    combine_activity_lines,
)


def build_example() -> dict:
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "SOURCE_MANIFEST.json").read_text())
    for name, identity in manifest["files"].items():
        if (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            != identity["sha256"]
        ):
            raise ValueError(f"Source hash mismatch: {name}")
    controls = json.loads((directory / "combination_method_control.json").read_text())
    # Only the previously selected Co60 set is used. Ni57/Sc48 are not admitted.
    co = next(
        r
        for r in controls["rows"]
        if r["sample"] == "Co-Cd-RAFM-1_25cm" and r["isotope"] == "Co60"
    )
    report = json.loads((directory / "weighting_control.json").read_text())
    checkout_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            manifest["required_engine_ancestor"],
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )
    implementation_hash = hashlib.sha256(
        Path(implementation.__file__).read_bytes()
    ).hexdigest()
    engine = (
        f"checkout:{checkout_commit};activity_combination_sha256:{implementation_hash}"
    )

    def compare(activities, sigmas, source, count_basis, role):
        rows = [
            ActivityLine(
                f"Co60-line-{i}",
                "Co60",
                a,
                s,
                True,
                "frozen selected Co60 equation-control set; absolute accuracy unqualified",
                source,
                count_basis,
                "source measurement-time activity",
            )
            for i, (a, s) in enumerate(zip(activities, sigmas))
        ]
        components = [
            CovarianceComponent(
                "declared_sigma_diagonal_control",
                np.diag(np.array(sigmas) ** 2),
                source,
                "Independent diagonal control; line-error correlations unavailable",
            ),
            CovarianceComponent(
                "shared_efficiency_calibration",
                None,
                source,
                "not available from source receipt/report",
            ),
            CovarianceComponent(
                "shared_yield_calibration",
                None,
                source,
                "not available from source receipt/report",
            ),
        ]
        common = dict(
            uncertainty_definition="absolute 1-sigma Bq; conditional declared-error-only control",
            engine_identity=engine,
            analysis_role=role,
            covariance_components=components,
        )
        return {
            "source_identity": source,
            "complete_uncertainty_gls": combine_activity_lines(
                rows, method="gls", **common
            ),
            "conditional_same_input_methods": [
                combine_activity_lines(
                    rows, method=m, allow_incomplete_uncertainty=True, **common
                )
                for m in METHODS
            ],
        }

    source = f"46096eb:combination_method_control.json:sha256:{manifest['files']['combination_method_control.json']['sha256']};parent_input_sha256:{co['input_json_sha256']}"
    fixed = compare(
        co["fixed_FluxForge_line_activities_Bq"],
        co["fixed_FluxForge_line_sigmas_Bq"],
        source,
        "historical fixed ASC selected-line reduction; physical basis qualification unchanged",
        "method_control",
    )
    report_source = f"46096eb:weighting_control.json:sha256:{manifest['files']['weighting_control.json']['sha256']}"
    historical = compare(
        report["printed_line_activity_Bq"],
        report["approx_line_sigma_Bq_from_count_uncertainty_only"],
        report_source,
        "rounded vendor report comparison counts; count-error-only approximation",
        "historical_reproduction_control",
    )
    hist, inv, gls = fixed["conditional_same_input_methods"]
    np.testing.assert_allclose(
        hist["activity_bq"], co["manual_weighting_diagnostic_Bq"], rtol=1e-13
    )
    np.testing.assert_allclose(
        inv["activity_bq"], co["inverse_variance_Bq"], rtol=1e-13
    )
    np.testing.assert_allclose(gls["activity_bq"], inv["activity_bq"], rtol=1e-13)
    # Separate physical-method illustration with a declared synthetic shared calibration.
    synthetic = [
        ActivityLine(
            str(i),
            "synthetic",
            100.0,
            np.sqrt(29.0),
            True,
            "synthetic mathematical control",
            "analytic fixture",
            "synthetic",
            "common reference",
        )
        for i in range(4)
    ]
    physical = combine_activity_lines(
        synthetic,
        method="gls",
        uncertainty_definition="1-sigma Bq; 2 Bq independent + 5 Bq shared",
        engine_identity=engine,
        analysis_role="physical_analysis",
        covariance_components=[
            CovarianceComponent(
                "count",
                4 * np.eye(4),
                "analytic fixture",
                "independent 2 Bq standard uncertainty",
            ),
            CovarianceComponent(
                "shared_efficiency_yield",
                25 * np.ones((4, 4)),
                "analytic fixture",
                "fully shared 5 Bq standard uncertainty",
            ),
        ],
    )
    return {
        "scope": "gamma-line equation controls; no target fitting, no new isotope admission, no full vendor-error reproduction",
        "source_manifest": manifest,
        "execution": {
            "checkout_commit": checkout_commit,
            "implementation_sha256": implementation_hash,
            "python": platform.python_version(),
            "executable": sys.executable,
            "numpy": np.__version__,
        },
        "fixed_ASC_control": fixed,
        "historical_report_control": historical,
        "same_input_relative_method_change_percent": 100
        * (hist["activity_bq"] / inv["activity_bq"] - 1),
        "physical_synthetic_shared_covariance_control": physical,
        "limitations": [
            "Source reductions were not rerun with the current engine. Current engine descendant executes only this new aggregation module.",
            "Unavailable vendor/physical calibration errors stay unavailable; partial controls are conditional.",
            "Source count bases and calibration qualification remain separate. No QuantumGold summary is used as a fit target.",
            "Integration and complete physical uncertainty qualification remain under #232/#220.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_example(), indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(args.output.resolve())
    else:
        print(payload, end="")
