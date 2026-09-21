"""Generic CSV spectrum reader used by the reader factory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from fluxforge.io.spe import GammaSpectrum


def _find_column(fieldnames: Sequence[str], *candidates: str) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_spectrum_csv(path: str | Path) -> GammaSpectrum:
    """Read a simple spectrum CSV with counts and optional channel/energy columns."""

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        metadata = {}
        lines = []
        for line in handle:
            if line.startswith("# fluxforge_spectrum="):
                metadata = json.loads(line.split("=", 1)[1])
            elif not line.startswith("#"):
                lines.append(line)
        reader = csv.DictReader(lines)
        fieldnames = list(reader.fieldnames or [])
        count_key = _find_column(fieldnames, "counts", "count", "cts", "net_counts")
        if count_key is None:
            raise ValueError(f"{path} does not contain a counts column.")
        channel_key = _find_column(fieldnames, "channel", "channels", "bin")
        energy_key = _find_column(fieldnames, "energy_kev", "energy", "kev")
        uncertainty_key = _find_column(fieldnames, "counts_uncertainty", "uncertainty")

        counts: list[float] = []
        channels: list[float] = []
        energies: list[float] = []
        uncertainties: list[float] = []
        for index, row in enumerate(reader):
            count_text = str(row.get(count_key, "")).strip()
            if not count_text:
                continue
            counts.append(float(count_text))
            if uncertainty_key:
                uncertainties.append(float(row[uncertainty_key]))
            channels.append(
                float(str(row.get(channel_key, index)).strip() or index)
                if channel_key
                else float(index)
            )
            if energy_key:
                energy_text = str(row.get(energy_key, "")).strip()
                if energy_text:
                    energies.append(float(energy_text))

    metadata.update(counts=counts, channels=channels,
                    energies=energies if len(energies) == len(counts) else None,
                    counts_uncertainty=uncertainties if uncertainty_key else None)
    metadata.setdefault("spectrum_id", path.stem)
    metadata.setdefault("metadata", {"source_file": str(path), "format": "csv_spectrum"})
    return GammaSpectrum.from_dict(metadata)


def write_spectrum_csv(path, spectrum, *, metadata_lines=()):
    """Write full precision counts and embedded, lossless covariance metadata."""
    payload = spectrum.to_dict()
    for key in ("counts", "counts_uncertainty", "channels", "energies"):
        payload.pop(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        for line in metadata_lines:
            handle.write(f"# {line}\n")
        handle.write("# fluxforge_spectrum=" + json.dumps(payload, allow_nan=False) + "\n")
        writer = csv.writer(handle)
        columns = ["channel", "net_counts", "counts_uncertainty"]
        arrays = [spectrum.channels, spectrum.counts, spectrum.counts_uncertainty]
        if spectrum.energies is not None:
            columns.append("energy_keV")
            arrays.append(spectrum.energies)
        writer.writerow(columns)
        writer.writerows(zip(*arrays))


__all__ = ["read_spectrum_csv"]
