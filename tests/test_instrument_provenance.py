"""Acquisition provenance remains explicit and bound to each recorded spectrum."""

from copy import deepcopy
import json
from zipfile import ZipFile

import pytest

from fluxforge.reporting.instrument_provenance import (
    instrument_provenance,
    validate_instructional_snapshot,
)

SETTINGS = {
    "amplifier_gain": 12.5,
    "shaping_time_us": 6,
    "high_voltage_v": 2500,
    "count_geometry": "25 cm on source axis",
}


def _workspace():
    return {
        "spectra": [
            {
                "spectrum_id": "measured",
                "label": "foil",
                "source_hash": "recorded hash",
                "spectrum": {
                    "live_time": 60,
                    "real_time": 65,
                    "calibration": {"energy": [0.3, 0.5]},
                    "metadata": {"instrument_settings": deepcopy(SETTINGS)},
                },
            }
        ]
    }


def test_imported_and_user_entered_settings_preserve_recorded_timing_and_identity():
    workspace = _workspace()
    original = deepcopy(workspace)
    result = instrument_provenance(
        workspace, {"measured": {"amplifier_gain": "13"}}, require_complete=True
    )
    assert workspace == original
    record = result["records"][0]
    assert record["source_hash"] == "recorded hash"
    assert record["settings"]["amplifier_gain"] == {
        "value": 13,
        "source": "user_entered",
    }
    assert record["settings"]["shaping_time_us"]["source"] == "recorded_metadata"
    assert record["settings"]["live_time_s"]["value"] == 60
    assert record["settings"]["mca_calibration"]["value"]["energy"] == [0.3, 0.5]
    assert result["scientific_admission"] is False


def test_missing_settings_and_inconsistent_time_never_become_defaults():
    workspace = _workspace()
    spectrum = workspace["spectra"][0]["spectrum"]
    spectrum["metadata"] = {}
    spectrum["calibration"] = {}
    spectrum["real_time"] = 0
    result = instrument_provenance(workspace)
    record = result["records"][0]
    assert record["settings"]["amplifier_gain"]["value"] is None
    assert record["settings"]["real_time_s"]["value"] is None
    assert "counting_time_order" in record["missing_required"]
    assert "mca_calibration" in record["missing_required"]
    assert result["complete_required_settings"] is False
    with pytest.raises(ValueError, match="Instructional"):
        instrument_provenance(workspace, require_complete=True)


def test_instructional_export_rejects_a_stale_or_forged_settings_receipt():
    workspace = _workspace()
    snapshot = {
        "workspace": workspace,
        "instructional_report": True,
        "instrument_provenance": instrument_provenance(workspace),
    }
    validate_instructional_snapshot(snapshot)
    workspace["spectra"][0]["spectrum"]["live_time"] = 55
    with pytest.raises(ValueError, match="does not match"):
        validate_instructional_snapshot(snapshot)
    snapshot["instrument_provenance"] = {"complete_required_settings": True}
    with pytest.raises(ValueError):
        validate_instructional_snapshot(snapshot)


@pytest.mark.parametrize(
    "field,value",
    [
        ("amplifier_gain", True),
        ("amplifier_gain", 0),
        ("shaping_time_us", -1),
        ("high_voltage_v", "nan"),
        ("count_geometry", 25),
    ],
)
def test_invalid_recorded_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        instrument_provenance(_workspace(), {"measured": {field: value}})


@pytest.mark.parametrize("voltage", [-2500.0, 0.0, 2500.0])
def test_recorded_high_voltage_preserves_signed_bias_and_zero(voltage):
    result = instrument_provenance(
        _workspace(), {"measured": {"high_voltage_v": voltage}}
    )
    recorded = result["records"][0]["settings"]["high_voltage_v"]
    assert recorded == {"value": voltage, "source": "user_entered"}
    assert result["scientific_admission"] is False


def test_qt_large_campaign_keeps_missing_settings_readable_and_complete(tmp_path):
    pytest.importorskip("PySide6")
    from fluxforge.gui.qt_compat import QApplication
    from fluxforge.gui.dialogs.report_export_dialog import ReportExportDialog
    from fluxforge.reporting.engine import ReportingEngine

    app = QApplication.instance() or QApplication([])
    workspace = _workspace()
    workspace["spectra"][0]["spectrum"]["metadata"] = {}
    base = workspace["spectra"][0]
    workspace["spectra"] = [
        dict(deepcopy(base), spectrum_id=f"acquisition-{index}", label=f"Foil {index}")
        for index in range(30)
    ]
    workspace["active_spectrum_id"] = "acquisition-0"
    context = {
        key: ""
        for key in ReportingEngine().templates["standard_lab"].required_context_keys
    }
    context["run_snapshot"] = {"workspace": workspace}
    dialog = ReportExportDialog(context_factory=lambda _name: context)
    try:
        captured = dialog._capture_context()
        assert (
            len(captured["run_snapshot"]["instrument_provenance"]["missing_required"])
            == 120
        )
        message = dialog.instrument_status.text()
        assert "Foil 0" in message and "30 acquisitions" in message
        assert len(message) < 400
        assert "acquisition-29" not in message
        dialog.instructional_check.setChecked(True)
        dialog.path_input.setText(str(tmp_path / "blocked.html"))
        dialog.export_html()
        assert dialog.last_export_path is None
        assert len(dialog.export_status.text()) < 300
        assert not (tmp_path / "blocked.html").exists()
    finally:
        dialog.close()
        app.processEvents()


def test_qt_instructional_bundle_requires_all_spectra_and_preserves_settings_on_switch(
    tmp_path,
    wait_for_report_export,
):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from fluxforge.gui.qt_compat import QApplication
    from fluxforge.gui.main_window import FluxForgeMainWindow
    from fluxforge.gui.mode_manager import ModeManager
    from fluxforge.gui.selection_bus import SelectionBus

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(), selection_bus=SelectionBus(), load_example=True
    )
    window.show()
    app.processEvents()
    window._open_report_export()
    dialog = window._report_dialog
    try:
        dialog.path_input.setText(str(tmp_path / "instructional.html"))
        dialog.instructional_check.setChecked(True)
        dialog.generate_report()
        wait_for_report_export(dialog)
        assert dialog.last_bundle_path is None
        assert not (tmp_path / "instructional.zip").exists()
        assert "Instructional" in dialog.export_status.text()
        records = list(window.analysis_workspace.document.spectra)
        for index, record in enumerate(records):
            window.analysis_workspace.select_spectrum(record.spectrum_id)
            assert dialog._instrument_spectrum_id == record.spectrum_id
            assert not dialog.instrument_inputs["amplifier_gain"].text()
            for field, value in SETTINGS.items():
                dialog.instrument_inputs[field].setText(
                    str(value if field != "amplifier_gain" else index + 1)
                )
        window.analysis_workspace.select_spectrum(records[0].spectrum_id)
        assert dialog.instrument_inputs["amplifier_gain"].text() == "1"
        dialog.generate_report()
        wait_for_report_export(dialog)
        assert dialog.last_bundle_path is not None, dialog.export_status.text()
        with ZipFile(dialog.last_bundle_path) as archive:
            snapshot = json.loads(archive.read("snapshot.json"))
            provenance = snapshot["instrument_provenance"]
            assert snapshot["instructional_report"] is True
            assert provenance["complete_required_settings"] is True
            assert len(provenance["records"]) == len(records)
            assert [
                item["settings"]["amplifier_gain"]["value"]
                for item in provenance["records"]
            ] == list(range(1, len(records) + 1))
            assert (
                "Instrument Settings Provenance" in archive.read("report.html").decode()
            )
    finally:
        dialog.close()
        window.close()


def test_qt_recorded_settings_follow_acquisition_identity():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from fluxforge.core.workspace_document import CalibrationModel
    from fluxforge.gui import FluxForgeMainWindow, ModeManager, SelectionBus
    from fluxforge.gui.qt_compat import QApplication

    class MemorySettings:
        def __init__(self):
            self.values = {}

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

        def sync(self):
            pass

    app = QApplication.instance() or QApplication([])
    settings = MemorySettings()
    window = FluxForgeMainWindow(
        settings=settings,
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window._open_report_export()
    dialog = window._report_dialog
    try:
        spectrum_id = window.analysis_workspace.document.active_spectrum_id
        dialog.instrument_inputs["amplifier_gain"].setText("42")
        window.analysis_workspace.apply_calibration(
            spectrum_id,
            CalibrationModel(model_key="polynomial", coefficients=(0.5, 1.0)),
        )
        assert dialog.instrument_inputs["amplifier_gain"].text() == "42"
        window._load_example_workspace()
        assert window.analysis_workspace.document.active_spectrum_id == spectrum_id
        assert not dialog.instrument_inputs["amplifier_gain"].text()
        assert not dialog._instrument_overrides.get(spectrum_id)
        dialog.reject()
        assert dialog._workspace_controller is None
    finally:
        dialog.close()
        window.close()
        app.processEvents()
