"""Batch-analysis queue helpers for the modern GUI."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from fluxforge.core.phase2_analysis import detect_peak_candidates
from fluxforge.io.spe import GammaSpectrum
from fluxforge.ml import MLPeakAnalysisEngine
from fluxforge.unfolding.gpu_backend import resolve_array_backend


@dataclass(frozen=True)
class BatchAnalysisJob:
    """One queued batch-analysis request."""

    job_id: str
    label: str
    spectrum: GammaSpectrum


@dataclass(frozen=True)
class BatchAnalysisResult:
    """Serialized result row for one processed spectrum."""

    job_id: str
    label: str
    total_counts: float
    total_uncertainty: float
    peak_count: int
    dominant_nuclide: str
    backend: str
    output_payload: dict[str, object]


def _analyze_batch_job(job: BatchAnalysisJob, prefer_gpu: bool) -> BatchAnalysisResult:
    backend_info, _ = resolve_array_backend(prefer_gpu=prefer_gpu)
    spectrum = job.spectrum
    peaks = detect_peak_candidates(spectrum)
    ml_engine = MLPeakAnalysisEngine()
    predictions = ml_engine.analyze_peaks(peaks, prefer_gpu=prefer_gpu)
    dominant = predictions[0].predicted_nuclide if predictions else (peaks[0].nuclide or "") if peaks else ""
    total_counts = float(np.sum(np.asarray(spectrum.counts, dtype=float)))
    total_uncertainty = float(
        np.sqrt(np.sum(np.asarray(spectrum.counts_uncertainty, dtype=float) ** 2))
    )
    payload = {
        "job_id": job.job_id,
        "label": job.label,
        "peak_count": len(peaks),
        "dominant_nuclide": dominant,
        "backend": backend_info.name,
        "peaks": [
            {
                "peak_id": peak.peak_id,
                "energy_keV": peak.energy_keV,
                "net_counts": peak.net_counts,
                "significance": peak.significance,
            }
            for peak in peaks
        ],
        "predictions": [asdict(prediction) for prediction in predictions],
    }
    return BatchAnalysisResult(
        job_id=job.job_id,
        label=job.label,
        total_counts=total_counts,
        total_uncertainty=total_uncertainty,
        peak_count=len(peaks),
        dominant_nuclide=dominant,
        backend=backend_info.name,
        output_payload=payload,
    )


def run_batch_analysis_queue(
    jobs: Sequence[BatchAnalysisJob],
    *,
    max_workers: int = 1,
    prefer_gpu: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[BatchAnalysisResult, ...]:
    """Run the queued jobs via a ProcessPoolExecutor."""

    if not jobs:
        if progress_callback is not None:
            progress_callback(0, 0)
        return ()
    if progress_callback is not None:
        progress_callback(0, len(jobs))
    with ProcessPoolExecutor(max_workers=max(int(max_workers), 1)) as executor:
        future_map = {
            executor.submit(_analyze_batch_job, job, bool(prefer_gpu)): index
            for index, job in enumerate(jobs)
        }
        ordered_results: list[BatchAnalysisResult | None] = [None] * len(jobs)
        completed = 0
        for future in as_completed(future_map):
            ordered_results[future_map[future]] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(jobs))
    return tuple(result for result in ordered_results if result is not None)


def aggregate_results_csv(results: Sequence[BatchAnalysisResult]) -> str:
    """Serialize aggregate batch results to CSV text."""

    header = [
        "job_id",
        "label",
        "total_counts",
        "total_uncertainty",
        "peak_count",
        "dominant_nuclide",
        "backend",
    ]
    rows = [header]
    for result in results:
        rows.append(
            [
                result.job_id,
                result.label,
                f"{result.total_counts:.6f}",
                f"{result.total_uncertainty:.6f}",
                str(result.peak_count),
                result.dominant_nuclide,
                result.backend,
            ]
        )
    lines = []
    for row in rows:
        lines.append(",".join(row))
    return "\n".join(lines)


def write_batch_outputs(
    results: Sequence[BatchAnalysisResult],
    output_dir: str | Path,
) -> tuple[Path, tuple[Path, ...]]:
    """Write per-spectrum JSON outputs and one aggregate CSV."""

    resolved = Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    json_paths: list[Path] = []
    for result in results:
        path = resolved / f"{result.job_id}.json"
        path.write_text(
            json.dumps(result.output_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        json_paths.append(path)
    aggregate_path = resolved / "aggregate.csv"
    aggregate_path.write_text(aggregate_results_csv(results), encoding="utf-8")
    return aggregate_path, tuple(json_paths)


__all__ = [
    "BatchAnalysisJob",
    "BatchAnalysisResult",
    "aggregate_results_csv",
    "run_batch_analysis_queue",
    "write_batch_outputs",
]
