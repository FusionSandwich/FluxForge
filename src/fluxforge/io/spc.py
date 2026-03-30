"""Minimal SPC reader support used by the Phase 1 reader factory.

This reader handles the common ASCII-export SPC interchange layout used for
portable handoff between spectroscopy tools:

```
SPECTRUM_ID=field_sample
LIVE_TIME=300
REAL_TIME=305
CALIBRATION=0.0,0.5,0.0
COUNTS=
1 2 3 4
```
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fluxforge.io.spe import GammaSpectrum


def read_spc_file(path: str | Path) -> GammaSpectrum:
    """Read a text SPC spectrum export."""

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    headers: dict[str, str] = {}
    counts_lines: list[str] = []
    reading_counts = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if reading_counts:
            counts_lines.append(line)
            continue
        if line.upper() == "COUNTS=":
            reading_counts = True
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            headers[key.strip().upper()] = value.strip()

    if not counts_lines:
        raise ValueError(
            f"{path} is not a supported ASCII SPC export or is missing counts data."
        )

    counts = np.asarray(
        [float(token) for line in counts_lines for token in line.replace(",", " ").split()],
        dtype=float,
    )
    calibration_text = headers.get("CALIBRATION", "")
    calibration = {}
    if calibration_text:
        calibration["energy"] = [
            float(token)
            for token in calibration_text.replace(",", " ").split()
            if token.strip()
        ]

    return GammaSpectrum(
        counts=counts,
        live_time=float(headers.get("LIVE_TIME", 0.0) or 0.0),
        real_time=float(headers.get("REAL_TIME", 0.0) or 0.0),
        spectrum_id=headers.get("SPECTRUM_ID", path.stem),
        detector_id=headers.get("DETECTOR_ID", ""),
        calibration=calibration,
        metadata={
            "source_file": str(path),
            "format": "spc_ascii",
        },
    )


__all__ = ["read_spc_file"]
