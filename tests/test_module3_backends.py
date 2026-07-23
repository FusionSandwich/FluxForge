from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

from fluxforge.core.batch_analysis import (
    BatchAnalysisJob,
    aggregate_results_csv,
    run_batch_analysis_queue,
    write_batch_outputs,
)
from fluxforge.core.analysis_workspace import (
    PeakCandidate,
    register_builtin_nuclide_id_engines,
)
from fluxforge.io.spe import GammaSpectrum
from fluxforge.ml import MLPeakAnalysisEngine
from fluxforge.plugins import PluginRegistries
from fluxforge.reporting.engine import ReportingEngine
from fluxforge.standards import (
    QAMonitor,
    QARecord,
    StandardsEvaluationContext,
    compute_pu_isotopics,
    currie_mda,
    register_builtin_standards_modules,
)
from fluxforge.standards.c1030 import PuLineObservation
from fluxforge.unfolding.gpu_backend import resolve_array_backend
from fluxforge.unfolding.response_matrix import (
    build_analytical_hpge_response,
    load_response_matrix,
)


def _demo_spectrum() -> GammaSpectrum:
    counts = np.array([4.0, 8.0, 40.0, 120.0, 45.0, 9.0, 3.0], dtype=float)
    return GammaSpectrum(
        counts=counts,
        channels=np.arange(counts.size, dtype=float),
        calibration={"energy": [0.0, 100.0]},
        spectrum_id="batch-demo",
    )


def _demo_peaks() -> tuple[PeakCandidate, ...]:
    return (
        PeakCandidate(
            peak_id="p1",
            channel=3.0,
            energy_keV=300.0,
            significance=9.2,
            roi_bounds_keV=(280.0, 320.0),
            net_counts=120.0,
            fit_quality=0.98,
        ),
    )


def test_register_builtin_standards_modules_populates_registry():
    registries = PluginRegistries()
    register_builtin_standards_modules(registries)

    assert registries.standards_modules.default_key == "ASTM E181"
    assert "ASTM C1030" in registries.standards_modules
    assert "ASTM E261" in registries.standards_modules


def test_currie_mda_returns_positive_value():
    mda = currie_mda(
        background_counts=144.0,
        live_time_s=600.0,
        efficiency=0.12,
        emission_probability=0.85,
    )

    assert mda > 0.0
    assert mda < 1.0


def test_c1030_isotopics_classifies_plutonium_vector():
    result = compute_pu_isotopics(
        (
            PuLineObservation("Pu-239", 129.3, 4200.0, 0.12, 64.8),
            PuLineObservation("Pu-240", 160.3, 180.0, 0.11, 18.0),
            PuLineObservation("Pu-241", 148.6, 110.0, 0.10, 12.0),
            PuLineObservation("Am-241", 59.5, 60.0, 0.16, 8.0),
        ),
        source_age_years=6.0,
    )

    assert result.pu240_to_pu239 > 0.0
    assert result.age_years == 6.0
    assert result.classification in {"weapons-grade", "fuel-grade", "reactor-grade"}


def test_qa_monitor_records_history_and_computes_status(tmp_path):
    monitor = QAMonitor(tmp_path / "qa_history.db")
    baseline = QARecord(
        timestamp=np.datetime64("2026-03-15T14:22")
        .astype("datetime64[s]")
        .astype(object),
        nuclide="Cs-137",
        energy_keV=661.66,
        measured_centroid_keV=661.64,
        measured_fwhm_keV=1.80,
        measured_fwhm_channels=2.3,
        net_counts=12000,
        efficiency=0.081,
        spectrum_file="baseline.spe",
    )
    followup = QARecord(
        timestamp=np.datetime64("2026-03-22T14:22")
        .astype("datetime64[s]")
        .astype(object),
        nuclide="Cs-137",
        energy_keV=661.66,
        measured_centroid_keV=661.70,
        measured_fwhm_keV=1.88,
        measured_fwhm_channels=2.4,
        net_counts=11800,
        efficiency=0.078,
        spectrum_file="followup.spe",
    )

    monitor.record(baseline)
    monitor.record(followup)

    history = monitor.history()
    snapshot = monitor.status_snapshot()

    assert len(history) == 2
    assert len(snapshot) == 1
    assert snapshot[0].status in {"green", "amber", "red"}


def test_qa_demo_history_is_ephemeral_and_not_reopened(tmp_path):
    database = tmp_path / "qa_history.db"
    monitor = QAMonitor(database)
    monitor.seed_demo_history()

    assert len(monitor.history()) == 3
    assert database.exists()

    reopened = QAMonitor(database)
    assert reopened.history() == ()

    reopened.seed_demo_history()
    assert len(reopened.history()) == 3
    reopened.clear_demo_history()
    assert reopened.history() == ()


def test_reporting_engine_renders_bundled_templates(tmp_path):
    engine = ReportingEngine()
    context = {
        "title": "FluxForge Report",
        "spectrum_image": "canvas",
        "calibration_curve": "curve",
        "calibration_residuals": "residuals",
        "efficiency_curve": "eff curve",
        "efficiency_residuals": "eff residuals",
        "residuals_grid": "roi grid",
        "peak_table": "<table></table>",
        "activity_table": "<table></table>",
        "astm_status_table": "<table></table>",
        "qa_status_snapshot": "qa snapshot",
        "provenance": "mode=expert",
        "batch_rows": "none",
        "aggregate_csv": "header\nrow",
    }

    rendered = engine.render("standard_lab", context)
    astm_rendered = engine.render("astm_compliance", context)
    batch_rendered = engine.render("batch_summary", context)
    output = engine.export_html("standard_lab", context, tmp_path / "report.html")

    assert "FluxForge Report" in rendered.html
    assert "ASTM Status" in astm_rendered.html
    assert "Batch Rows" in batch_rendered.html
    assert output.exists()
    assert "Residuals Grid" in output.read_text(encoding="utf-8")


def test_reporting_engine_exports_pdf_with_weasyprint_backend(tmp_path, monkeypatch):
    class _FakeHtml:
        def __init__(self, string: str, base_url: str | None = None) -> None:
            self.string = string
            self.base_url = base_url

        def write_pdf(self, target: str) -> None:
            Path(target).write_bytes(b"%PDF-1.4\n% FluxForge test export\n")

    engine = ReportingEngine()
    context = {
        "title": "FluxForge PDF",
        "spectrum_image": "canvas",
        "calibration_curve": "curve",
        "calibration_residuals": "residuals",
        "efficiency_curve": "eff curve",
        "efficiency_residuals": "eff residuals",
        "residuals_grid": "roi grid",
        "peak_table": "<table></table>",
        "activity_table": "<table></table>",
        "astm_status_table": "<table></table>",
        "qa_status_snapshot": "qa snapshot",
        "provenance": "mode=expert",
        "batch_rows": "none",
        "aggregate_csv": "header\nrow",
    }
    fake_module = types.SimpleNamespace(HTML=_FakeHtml)
    monkeypatch.setattr(
        "fluxforge.reporting.engine.import_module",
        lambda name: fake_module if name == "weasyprint" else __import__(name),
    )

    output = engine.export_pdf("standard_lab", context, tmp_path / "report.pdf")

    assert engine.can_export_pdf() is True
    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF-1.4")


def test_response_matrix_loader_supports_csv_and_analytical_builder(tmp_path):
    matrix = np.array([[0.8, 0.1], [0.2, 0.9]], dtype=float)
    path = tmp_path / "response.csv"
    np.savetxt(path, matrix, delimiter=",")

    loaded = load_response_matrix(path)
    analytical = build_analytical_hpge_response(
        n_channels=4,
        energy_edges=np.array([0.0, 1.0, 2.0], dtype=float),
    )

    assert loaded.matrix.shape == (2, 2)
    assert loaded.source_format == "user_csv"
    assert analytical.matrix.shape == (4, 2)
    assert analytical.source_format == "analytical_hpge"


def test_array_backend_falls_back_to_numpy():
    backend, module = resolve_array_backend(prefer_gpu=True)

    assert backend.name in {"numpy", "cupy"}
    assert hasattr(module, "array")


def test_ml_peak_engine_registers_and_returns_predictions():
    registries = PluginRegistries()
    register_builtin_nuclide_id_engines(registries)
    engine = registries.nuclide_id_engines.get("ml_peak_onnx")

    predictions = engine.analyze_peaks(_demo_peaks())

    assert isinstance(engine, MLPeakAnalysisEngine)
    assert len(predictions) <= 1
    if predictions:
        assert predictions[0].confidence > 0.0
        assert predictions[0].backend in {"numpy", "cupy"}


def test_batch_analysis_queue_writes_json_and_aggregate_outputs(tmp_path):
    progress_events: list[tuple[int, int]] = []
    result_rows = run_batch_analysis_queue(
        (BatchAnalysisJob("job-1", "Demo", _demo_spectrum()),),
        max_workers=1,
        prefer_gpu=False,
        progress_callback=lambda completed, total: progress_events.append(
            (completed, total)
        ),
    )
    aggregate_path, json_paths = write_batch_outputs(result_rows, tmp_path / "batch")

    assert len(result_rows) == 1
    assert result_rows[0].peak_count >= 0
    assert aggregate_path.exists()
    assert len(json_paths) == 1
    assert "job_id,label" in aggregate_results_csv(result_rows)
    assert progress_events[0] == (0, 1)
    assert progress_events[-1] == (1, 1)


def test_reporting_engine_reports_pdf_unavailable_without_weasyprint(monkeypatch):
    monkeypatch.setattr(
        "fluxforge.reporting.engine.import_module",
        lambda name: (
            (_ for _ in ()).throw(ImportError("missing"))
            if name == "weasyprint"
            else __import__(name)
        ),
    )

    engine = ReportingEngine()

    assert engine.can_export_pdf() is False


def test_reporting_engine_reports_pdf_unavailable_when_native_library_is_missing(
    monkeypatch,
):
    engine = ReportingEngine()
    monkeypatch.setattr(
        "fluxforge.reporting.engine.import_module",
        lambda _name: (_ for _ in ()).throw(OSError("missing libgobject")),
    )

    assert engine.can_export_pdf() is False


def test_reporting_engine_imports_without_jinja2_backend(monkeypatch):
    monkeypatch.setattr(
        "fluxforge.reporting.engine._JINJA2_IMPORT_ERROR", ImportError("missing")
    )
    monkeypatch.setattr("fluxforge.reporting.engine.Environment", None)
    monkeypatch.setattr("fluxforge.reporting.engine.FileSystemLoader", None)
    monkeypatch.setattr("fluxforge.reporting.engine.select_autoescape", None)

    engine = ReportingEngine()

    assert engine.template_backend_available() is False
    with pytest.raises(RuntimeError, match="Jinja2"):
        engine.render(
            "standard_lab",
            {
                "title": "FluxForge Report",
                "spectrum_image": "canvas",
                "calibration_curve": "curve",
                "calibration_residuals": "residuals",
                "efficiency_curve": "eff curve",
                "efficiency_residuals": "eff residuals",
                "residuals_grid": "roi grid",
                "peak_table": "<table></table>",
                "activity_table": "<table></table>",
                "astm_status_table": "<table></table>",
                "qa_status_snapshot": "qa snapshot",
                "provenance": "mode=expert",
            },
        )
