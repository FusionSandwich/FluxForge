#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from _compare_common import ensure_fluxforge_src

ROOT = ensure_fluxforge_src(__file__)

from fluxforge.examples.rafm_workflow import run_rafm_validation


COUNTING_METHOD_CHOICES = [
    "qg",
    "quantum_gold",
    "covell_local",
    "gilmore_minimum",
    "iec_tiered",
]

COUNTING_METHOD_ALIASES = {
    "qg_hybrid": "qg",
    "current_hybrid": "qg",
    "quantumgold": "qg",
    "quantum_gold": "qg",
}


def normalize_counting_method_name(value: str) -> str:
    method = str(value).strip().lower()
    return COUNTING_METHOD_ALIASES.get(method, method)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the RAFM irradiation validation workflow."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to examples/RAFM_irradiation/results/.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit non-zero when validation thresholds are violated.",
    )
    parser.add_argument(
        "--max-spectra",
        type=int,
        default=None,
        help="Optional cap for debugging or tests.",
    )
    parser.add_argument(
        "--flux-wire-counting-method",
        type=normalize_counting_method_name,
        choices=COUNTING_METHOD_CHOICES,
        default=None,
        help=(
            "Override the flux-wire counting method used for targeted QuantumGold parity analysis. "
            "Use qg or quantum_gold for the QuantumGold-focused workflow. "
            "Legacy aliases such as qg_hybrid, quantumgold, and current_hybrid are accepted and normalized to qg."
        ),
    )
    parser.add_argument(
        "--generic-counting-method",
        type=normalize_counting_method_name,
        choices=COUNTING_METHOD_CHOICES,
        default=None,
        help=(
            "Override the generic targeted counting method used for RAFM sample analysis. "
            "Use qg or quantum_gold for the QuantumGold-focused workflow. "
            "Legacy aliases such as qg_hybrid, quantumgold, and current_hybrid are accepted and normalized to qg."
        ),
    )
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parent
    run_rafm_validation(
        example_root=example_root,
        results_root=args.results_root,
        enforce_thresholds=not args.no_fail,
        max_spectra=args.max_spectra,
        flux_wire_counting_method=args.flux_wire_counting_method,
        generic_targeted_counting_method=args.generic_counting_method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
