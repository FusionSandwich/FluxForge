"""ASTM E3376-23 High-Purity Germanium Detector Calibration and Usage.

This module provides the core equations for computing net peak areas, 
efficiencies, and transmutation rates according to ASTM E3376-23.
"""

import math
from typing import Dict, Any


def analyze_astm_e3376_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze an ASTM E3376 calibration or usage plan."""
    results = dict(plan)
    outputs = []

    for row in plan.get("measurements", []):
        G_s = float(row.get("gross_area", 0.0))
        N = float(row.get("roi_channels", 1.0))
        n = float(row.get("continuum_channels", 1.0))
        B_1s = float(row.get("continuum_low", 0.0))
        B_2s = float(row.get("continuum_high", 0.0))

        # Calculate continuum B
        if n > 0:
            B = (N / (2.0 * n)) * (B_1s + B_2s)
        else:
            B = 0.0

        N_A = G_s - B

        T_s = float(row.get("live_time_s", 1.0))
        N_p = N_A / T_s if T_s > 0 else 0.0

        out = {
            "measurement_id": row.get("measurement_id", "unknown"),
            "net_peak_area": N_A,
            "net_count_rate": N_p,
            "continuum": B,
        }

        # Calibration mode
        if "emission_rate" in row:
            N_gamma = float(row["emission_rate"])
            E_f = N_p / N_gamma if N_gamma > 0 else 0.0
            out["efficiency"] = E_f

        # Measurement mode
        if "efficiency" in row and "gamma_probability" in row:
            E_f = float(row["efficiency"])
            P_gamma = float(row["gamma_probability"])

            N_R = N_p / E_f if E_f > 0 else 0.0
            A = N_R / P_gamma if P_gamma > 0 else 0.0
            out["transmutation_rate"] = A
            out["emission_rate"] = N_R

        outputs.append(out)

    results["outputs"] = outputs
    return results
