"""
Text calibration loaders for simple polynomial formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from fluxforge.core.spectrum_ops import Calibration


@dataclass
class PolynomialCalibrationFile:
    """Parsed calibration file contents."""

    channels: List[float]
    energies_keV: List[float]
    order: int
    coefficients_desc: List[float]


def read_pygammaspec_calibration(path: Union[str, Path]) -> PolynomialCalibrationFile:
    """
    Read PyGammaSpec-style calibration file.
    """
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [line for line in lines if line]

    if not lines or "Calibration points" not in lines[0]:
        raise ValueError("Calibration file missing header.")

    n_points = int(lines[0].split(":")[-1].strip())
    channels: List[float] = []
    energies: List[float] = []

    for i in range(1, 1 + n_points):
        parts = lines[i].split()
        if len(parts) < 2:
            raise ValueError("Calibration points line is malformed.")
        channels.append(float(parts[0]))
        energies.append(float(parts[1]))

    order_line = lines[1 + n_points]
    if "order" not in order_line:
        raise ValueError("Calibration file missing polynomial order line.")
    order = int(order_line.split(":")[-1].strip(") "))

    coeffs = [float(lines[2 + n_points + i]) for i in range(order + 1)]

    return PolynomialCalibrationFile(
        channels=channels,
        energies_keV=energies,
        order=order,
        coefficients_desc=coeffs,
    )


def pygammaspec_to_calibration(data: PolynomialCalibrationFile) -> Calibration:
    """
    Convert PyGammaSpec calibration file contents to FluxForge Calibration.
    """
    # PyGammaSpec stores coefficients in descending order.
    coeffs_asc = list(reversed(data.coefficients_desc))
    cal = Calibration(expression="polynomial", coefficients=np.array(coeffs_asc), degree=data.order)
    cal.add_points(data.channels, data.energies_keV)
    return cal
