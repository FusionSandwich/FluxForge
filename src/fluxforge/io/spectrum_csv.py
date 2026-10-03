"""Generic CSV spectrum reader used by the reader factory."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from fluxforge.io.spe import GammaSpectrum


def _find_column(fieldnames: Sequence[str], *candidates: str) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def read_spectrum_csv(path: str | Path) -> GammaSpectrum:
    """Read a simple spectrum CSV with counts and optional channel/energy columns."""

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        count_key = _find_column(fieldnames, "counts", "count", "cts")
        if count_key is None:
            raise ValueError(f"{path} does not contain a counts column.")
        channel_key = _find_column(fieldnames, "channel", "channels", "bin")
        energy_key = _find_column(fieldnames, "energy_keV", "energy", "kev")
        uncertainty_key = _find_column(
            fieldnames, "uncertainty", "counts_uncertainty", "sigma"
        )

        counts: list[float] = []
        channels: list[float] = []
        energies: list[float] = []
        uncertainties: list[float] = []
        for index, row in enumerate(reader):
            count_text = str(row.get(count_key, "")).strip()
            if not count_text:
                if any(str(value or "").strip() for value in row.values()):
                    raise ValueError(f"Missing counts for measurement row {index + 2}.")
                continue
            if channel_key and not str(row.get(channel_key, "")).strip():
                raise ValueError(f"Missing channel for counted row {index + 2}.")
            counts.append(float(count_text))
            channels.append(
                float(str(row.get(channel_key, index)).strip() or index)
                if channel_key
                else float(index)
            )
            if energy_key:
                energy_text = str(row.get(energy_key, "")).strip()
                if not energy_text:
                    raise ValueError(f"Missing energy for counted row {index + 2}.")
                energies.append(float(energy_text))
            if uncertainty_key:
                uncertainty_text = str(row.get(uncertainty_key, "")).strip()
                if not uncertainty_text:
                    raise ValueError(
                        f"Missing uncertainty for counted row {index + 2}."
                    )
                uncertainty = float(uncertainty_text)
                if not np.isfinite(uncertainty) or uncertainty < 0:
                    raise ValueError(
                        "Count uncertainty must be finite and nonnegative."
                    )
                uncertainties.append(uncertainty)

    if any(not np.all(np.isfinite(values)) for values in (counts, channels, energies)):
        raise ValueError("Spectrum counts, channels and energies must be finite.")

    return GammaSpectrum(
        counts=np.asarray(counts, dtype=float),
        counts_uncertainty=(
            np.asarray(uncertainties, dtype=float) if uncertainty_key else None
        ),
        channels=np.asarray(channels, dtype=float),
        energies=(
            np.asarray(energies, dtype=float) if len(energies) == len(counts) else None
        ),
        spectrum_id=path.stem,
        metadata={
            "source_file": str(path),
            "format": "csv_spectrum",
        },
    )


__all__ = ["read_spectrum_csv"]
