from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pytest

from fluxforge.core.predictive import estimate_count_target_forecast
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.io.spe import GammaSpectrum


class _Settings:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def value(self, key: str, default=None):
        return self.data.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.data[key] = value

    def sync(self) -> None:
        return None


def _gamma_csv(path) -> None:
    path.write_text(
        "nuclide,energy_keV,intensity,half_life_s\n"
        "Co60,1332.5,0.99,166344192\n",
        encoding="utf-8",
    )


def test_equal_timestamp_history_uses_finite_constant_trend() -> None:
    acquired_at = datetime(2026, 9, 16)
    first = GammaSpectrum(
        counts=np.ones(16), live_time=4.0, start_time=acquired_at
    )
    second = GammaSpectrum(
        counts=np.full(16, 2.0), live_time=4.0, start_time=acquired_at
    )

    forecast = estimate_count_target_forecast(
        second,
        history_spectra=(first, second),
    )

    assert forecast.trend.slope == 0.0
    assert forecast.trend.intercept == pytest.approx(6.0)
    assert all(
        np.isfinite(value)
        for value in (
            forecast.trend.slope,
            forecast.trend.intercept,
            forecast.trend.slope_stderr,
            forecast.trend.r_squared,
        )
    )


def test_invalid_registration_does_not_mutate_registry_or_selection(
    tmp_path, monkeypatch
) -> None:
    registry = tmp_path / "registry.json"
    malformed = tmp_path / "gamma.csv"
    malformed.write_text(
        "nuclide,energy_keV,intensity\nCo60,not-a-number,.99\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry))
    manager = DataLibraryManager()
    original = manager.state

    with pytest.raises(ValueError):
        manager.register_user_gamma_source("Malformed Lab", str(malformed))

    assert manager.state == original
    assert not registry.exists()


def test_invalid_custom_file_does_not_mutate_selection(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry.json"
    malformed = tmp_path / "gamma.csv"
    malformed.write_text(
        "nuclide,energy_keV,intensity\nCo60,not-a-number,.99\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry))
    manager = DataLibraryManager()
    original = manager.state

    with pytest.raises(ValueError):
        manager.set_gamma_identification_source(
            "custom_gamma_file",
            custom_gamma_path=str(malformed),
        )

    assert manager.state == original


def test_deleted_selected_library_recovers_and_persists_default(
    tmp_path, monkeypatch
) -> None:
    registry = tmp_path / "registry.json"
    source = tmp_path / "gamma.csv"
    _gamma_csv(source)
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry))
    settings = _Settings()
    manager = DataLibraryManager(settings=settings)
    manager.register_user_gamma_source("Temporary Lab", str(source))
    source.unlink()

    reopened = DataLibraryManager(settings=settings)

    assert reopened.state.gamma_identification_source_id == "fluxforge_bundled_gamma"
    assert reopened.recovery_message is not None
    assert "user_gamma_temporary_lab" in reopened.recovery_message
    assert "could not be loaded" in reopened.recovery_message
    assert settings.data[DataLibraryManager.GAMMA_SOURCE_KEY] == (
        "fluxforge_bundled_gamma"
    )
    assert json.loads(registry.read_text(encoding="utf-8"))["user_gamma_sources"]


def test_report_escapes_metadata_and_keeps_explicit_markup() -> None:
    pytest.importorskip("jinja2")
    from fluxforge.reporting.engine import ReportingEngine

    rendered = ReportingEngine().render(
        "batch_summary",
        {
            "title": "Sample <A> & B",
            "batch_rows": "<strong>accepted row</strong>",
            "aggregate_csv": "detector,<HPGe>",
            "provenance": "operator <lab>",
        },
    )

    assert "Sample &lt;A&gt; &amp; B" in rendered.html
    assert "<strong>accepted row</strong>" in rendered.html
    assert "detector,&lt;HPGe&gt;" in rendered.html
    assert "operator &lt;lab&gt;" in rendered.html


def test_predictive_panel_failure_does_not_break_spectrum_open(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings

    import fluxforge.gui.panels.modern_shell_center as predictive_panel
    from fluxforge.gui.main_window import FluxForgeMainWindow
    from fluxforge.gui.qt_compat import QApplication
    from fluxforge.io.spe import write_spe_file
    from fluxforge.standards import QAMonitor

    monkeypatch.setattr(
        predictive_panel,
        "estimate_count_target_forecast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forecast")),
    )
    app = QApplication.instance() or QApplication([])
    spectrum_path = tmp_path / "valid.spe"
    write_spe_file(GammaSpectrum(counts=np.arange(32, dtype=float)), spectrum_path)
    window = FluxForgeMainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
        qa_monitor=QAMonitor(tmp_path / "qa.db"),
    )
    try:
        window.open_path(spectrum_path)
        app.processEvents()
        assert window.analysis_workspace.loaded_spectrum_records()
    finally:
        window.close()
        app.processEvents()


def test_background_display_rejects_invalid_inputs_and_recovers(
    tmp_path,
) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings

    from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
    from fluxforge.gui.main_window import FluxForgeMainWindow
    from fluxforge.gui.qt_compat import QApplication
    from fluxforge.standards import QAMonitor

    if not PYQTGRAPH_AVAILABLE:
        pytest.skip("PyQtGraph is unavailable")

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
        qa_monitor=QAMonitor(tmp_path / "qa.db"),
    )
    foreground = GammaSpectrum(
        counts=np.array([20.0, 30.0, 40.0]),
        calibration={"energy": [0.0, 1.0]},
        spectrum_id="foreground",
    )
    valid_background = GammaSpectrum(
        counts=np.array([1.0, 2.0, 3.0]),
        calibration={"energy": [0.0, 1.0]},
        spectrum_id="valid-background",
    )
    mismatched_background = GammaSpectrum(
        counts=np.array([4.0, 5.0, 6.0]),
        calibration={"energy": [0.5, 1.0]},
        spectrum_id="mismatched-background",
    )
    foreground_before = foreground.to_dict()
    valid_before = valid_background.to_dict()
    mismatched_before = mismatched_background.to_dict()

    try:
        controller = window.analysis_workspace
        foreground_key = controller.register_loaded_spectrum(
            foreground,
            label="sample",
            key="sample",
        )
        valid_key = controller.register_loaded_spectrum(
            valid_background,
            label="valid background",
            key="valid-background",
        )
        mismatch_key = controller.register_loaded_spectrum(
            mismatched_background,
            label="mismatched background",
            key="mismatched-background",
        )
        controller.assign_loaded_spectrum_to_slot(foreground_key, "foreground")
        controller.assign_loaded_spectrum_to_slot(valid_key, "background")
        app.processEvents()

        canvas = window.central_tabs.canvas
        np.testing.assert_allclose(
            np.asarray(canvas.buffer.full_resolution, dtype=float),
            foreground.counts - valid_background.counts,
        )

        controller.set_background_config(mode="scaled", scale=float("nan"))
        app.processEvents()
        np.testing.assert_array_equal(
            np.asarray(canvas.buffer.full_resolution, dtype=float),
            foreground.counts,
        )
        invalid_scale_status = canvas.status_label.text().lower()
        assert "background not applied" in invalid_scale_status
        assert "finite and non-negative" in invalid_scale_status

        controller.set_background_config(mode="scaled", scale=1.0)
        app.processEvents()
        np.testing.assert_allclose(
            np.asarray(canvas.buffer.full_resolution, dtype=float),
            foreground.counts - valid_background.counts,
        )
        assert "background not applied" not in canvas.status_label.text().lower()

        controller.assign_loaded_spectrum_to_slot(mismatch_key, "background")
        app.processEvents()
        np.testing.assert_array_equal(
            np.asarray(canvas.buffer.full_resolution, dtype=float),
            foreground.counts,
        )
        mismatch_status = canvas.status_label.text().lower()
        assert "background not applied" in mismatch_status
        assert "strict coverage" in mismatch_status

        controller.assign_loaded_spectrum_to_slot(valid_key, "background")
        app.processEvents()
        np.testing.assert_allclose(
            np.asarray(canvas.buffer.full_resolution, dtype=float),
            foreground.counts - valid_background.counts,
        )
        assert "background not applied" not in canvas.status_label.text().lower()
        assert foreground.to_dict() == foreground_before
        assert valid_background.to_dict() == valid_before
        assert mismatched_background.to_dict() == mismatched_before
    finally:
        window.close()
        app.processEvents()
