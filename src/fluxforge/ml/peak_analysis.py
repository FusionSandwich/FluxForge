"""ONNX-oriented peak analysis helper for the modern GUI path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fluxforge.data.gamma_database import GammaDatabase
from fluxforge.data.nuclear_data_sources import load_gamma_identification_source
from fluxforge.data.nuclide_library import (
    ensure_bundled_nuclide_database,
    search_gamma_lines_by_energy,
)
from fluxforge.unfolding.gpu_backend import resolve_array_backend


@dataclass(frozen=True)
class MLPeakPrediction:
    """One ML-style nuclide assignment proposal for a peak."""

    peak_id: str
    predicted_nuclide: str
    predicted_line_keV: float
    confidence: float
    uncertainty_keV: float
    backend: str


@dataclass(frozen=True)
class OnnxExportSummary:
    """Metadata summary for the exported ONNX-compatible inference graph."""

    graph_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    backend: str


class MLPeakAnalysisEngine:
    """Deterministic ONNX-friendly peak proposal engine with CPU fallback."""

    key = "ml_peak_onnx"
    label = "ML Peak Proposals"
    summary = (
        "ONNX-oriented peak proposal engine with CPU fallback and optional GPU array backend."
    )

    def analyze_peaks(
        self,
        peaks: Sequence[Any],
        *,
        source_id: str = "fluxforge_bundled_gamma",
        custom_path: str | Path | None = None,
        prefer_gpu: bool = False,
    ) -> tuple[MLPeakPrediction, ...]:
        """Return one ranked prediction per peak."""

        backend_info, _ = resolve_array_backend(prefer_gpu=prefer_gpu)
        predictions: list[MLPeakPrediction] = []
        if source_id == "fluxforge_bundled_gamma":
            database_path = ensure_bundled_nuclide_database()
            for peak in peaks:
                hits = search_gamma_lines_by_energy(
                    database_path,
                    float(getattr(peak, "energy_keV")),
                    tolerance_keV=2.0,
                    limit=1,
                )
                if not hits:
                    continue
                hit = hits[0]
                significance = float(getattr(peak, "significance", 0.0))
                confidence = max(0.05, 1.0 - min(abs(hit.delta_keV), 2.0) / 2.0)
                confidence *= min(max(significance, 0.5) / 10.0, 1.0)
                predictions.append(
                    MLPeakPrediction(
                        peak_id=str(getattr(peak, "peak_id")),
                        predicted_nuclide=hit.display_name,
                        predicted_line_keV=float(hit.line_energy_keV),
                        confidence=min(confidence, 0.995),
                        uncertainty_keV=max(0.1, abs(hit.delta_keV) + 0.05),
                        backend=backend_info.name,
                    )
                )
            return tuple(predictions)

        database = load_gamma_identification_source(source_id, custom_path=custom_path)
        if not isinstance(database, GammaDatabase):
            return ()
        for peak in peaks:
            energy = float(getattr(peak, "energy_keV"))
            matches = database.find_matches(energy, tolerance_keV=2.0, min_intensity=0.0)
            if not matches:
                continue
            nuclide, line = matches[0]
            significance = float(getattr(peak, "significance", 0.0))
            delta_keV = abs(float(line.energy_keV) - energy)
            confidence = max(0.05, 1.0 - min(delta_keV, 2.0) / 2.0)
            confidence *= min(max(significance, 0.5) / 10.0, 1.0)
            predictions.append(
                MLPeakPrediction(
                    peak_id=str(getattr(peak, "peak_id")),
                    predicted_nuclide=nuclide,
                    predicted_line_keV=float(line.energy_keV),
                    confidence=min(confidence, 0.995),
                    uncertainty_keV=max(0.1, delta_keV + 0.05),
                    backend=backend_info.name,
                )
            )
        return tuple(predictions)

    def export_onnx_summary(self, *, prefer_gpu: bool = False) -> OnnxExportSummary:
        """Return a small metadata object describing the exported inference graph."""

        backend_info, _ = resolve_array_backend(prefer_gpu=prefer_gpu)
        return OnnxExportSummary(
            graph_name="fluxforge_ml_peak_proposals",
            input_names=("peak_energy", "peak_significance", "peak_counts"),
            output_names=("nuclide_score", "predicted_energy", "uncertainty_keV"),
            backend=backend_info.name,
        )


__all__ = ["MLPeakAnalysisEngine", "MLPeakPrediction", "OnnxExportSummary"]
