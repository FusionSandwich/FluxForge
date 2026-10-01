"""Preserve vendor efficiency-table values for explicitly conditional QC.

This table is an example-data diagnostic, never an implicit replacement for a
qualified detector calibration. Invalid raw rows remain visible and unmodified.
"""

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


class QGEfficiencyTable:
    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Byte-bound efficiency snapshot is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, path, provenance_path):
        self._path = Path(path)
        payload = self.path.read_bytes()
        self._sha256 = hashlib.sha256(payload).hexdigest()
        self._provenance_json = Path(provenance_path).read_text(encoding="utf-8")
        if self.sha256 != self.provenance["sha256"]:
            raise ValueError("Efficiency table SHA256 disagrees with provenance")
        rows = []
        records = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
        coefficient_names = records[0][4:]
        self._vendor_coefficients_json = json.dumps(
            dict(zip(coefficient_names, map(float, records[1][4:])))
        )
        start = next(i for i, r in enumerate(records) if r and r[0].strip() == "Energy")
        for line, r in enumerate(records[start + 1 :], start + 2):
            if not r or not r[0].strip():
                continue
            energy, efficiency = float(r[0]), float(r[1])
            if not math.isfinite(energy) or not math.isfinite(efficiency):
                raise ValueError("Vendor table contains nonfinite values")
            rows.append(
                {
                    "source_line": line,
                    "energy_keV": energy,
                    "efficiency_reported": efficiency,
                    "status": (
                        "positive_raw_value"
                        if efficiency > 0
                        else "excluded_nonpositive_efficiency"
                    ),
                }
            )
        self._rows_json = json.dumps(rows)
        self._energies = tuple(r["energy_keV"] for r in rows)
        self._values = tuple(r["efficiency_reported"] for r in rows)
        if not len(rows) or np.any(np.diff(self._energies) <= 0):
            raise ValueError("Efficiency energies must be strictly increasing")
        self._sealed = True

    @property
    def path(self):
        return self._path

    @property
    def sha256(self):
        return self._sha256

    @property
    def provenance(self):
        return json.loads(self._provenance_json)

    @property
    def vendor_coefficients(self):
        return json.loads(self._vendor_coefficients_json)

    @property
    def rows(self):
        return json.loads(self._rows_json)

    @property
    def energies(self):
        result = np.array(self._energies)
        result.setflags(write=False)
        return result

    @property
    def values(self):
        result = np.array(self._values)
        result.setflags(write=False)
        return result

    def diagnostic_at(self, energy_keV, *, unit_assumption):
        """Interpolate original adjacent rows only; never bridge invalid rows."""
        if unit_assumption not in {"percent", "fraction"}:
            raise ValueError("Declare a percent or fraction efficiency-unit assumption")
        energies, values = np.asarray(self._energies), np.asarray(self._values)
        out = {
            "energy_keV": float(energy_keV),
            "curve_sha256": self.sha256,
            "unit_assumption": unit_assumption,
            "scientific_admission": False,
            "physical_qualified": False,
            "efficiency_fraction": None,
        }
        if (
            not math.isfinite(energy_keV)
            or not energies[0] <= energy_keV <= energies[-1]
        ):
            return dict(out, status="excluded_outside_table")
        right = int(np.searchsorted(energies, energy_keV))
        indices = [right] if energies[right] == energy_keV else [right - 1, right]
        out["source_lines"] = [self.rows[i]["source_line"] for i in indices]
        out["raw_bracket_values"] = [float(values[i]) for i in indices]
        if any(values[i] <= 0 for i in indices):
            return dict(out, status="excluded_nonpositive_efficiency")
        value = float(np.interp(energy_keV, energies, values))
        fraction = value / 100 if unit_assumption == "percent" else value
        if fraction > 1:
            return dict(
                out, status="excluded_efficiency_above_one_under_unit_assumption"
            )
        return dict(
            out, status="conditional_curve_comparison", efficiency_fraction=fraction
        )

    def require_physical(self):
        # No certificate, covariance, active identity or acquisition date is
        # supplied by this historical export. A metadata flag cannot supply it.
        raise ValueError(
            "Vendor table alone cannot qualify physical calibration: certificate, covariance, active applicability and acquisition date required"
        )
