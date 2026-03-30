import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.core.calibration import (  # noqa: E402
    ASTM_E181_LOCKED_ORDER,
    EnergyCalibrationPoint,
    FWHMCalibrationPoint,
    fit_energy_calibration,
    fit_fwhm_calibration,
)
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.gui.dialogs import CalibrationWorkspaceDialog  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import GUIMode, ModeManager, ModeState  # noqa: E402
from fluxforge.gui.panels.modern_shell import build_demo_spectrum  # noqa: E402
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402


def _qapp():
    return QApplication.instance() or QApplication([])


def test_fit_energy_calibration_locks_astm_order_and_flags_outliers():
    fit = fit_energy_calibration(
        [
            EnergyCalibrationPoint(channel=0.0, reference_energy_keV=0.0),
            EnergyCalibrationPoint(channel=500.0, reference_energy_keV=500.0),
            EnergyCalibrationPoint(channel=1000.0, reference_energy_keV=1000.0),
            EnergyCalibrationPoint(channel=1500.0, reference_energy_keV=1515.0),
        ],
        order=4,
        standard="ASTM E181",
    )

    assert fit.order == ASTM_E181_LOCKED_ORDER
    assert fit.locked_by is not None
    assert any(fit.out_of_tolerance)
    assert fit.chi_squared > 0.0


def test_fit_fwhm_calibration_returns_resolution_metrics():
    fit = fit_fwhm_calibration(
        [
            FWHMCalibrationPoint(energy_keV=121.78, fwhm_keV=0.95),
            FWHMCalibrationPoint(energy_keV=661.657, fwhm_keV=1.82),
            FWHMCalibrationPoint(energy_keV=1173.228, fwhm_keV=2.11),
        ]
    )

    assert fit.model == "sqrt_poly"
    assert len(fit.coefficients) == 3
    assert fit.fitted_fwhm_keV.shape == (3,)
    assert fit.rms_keV >= 0.0


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_calibration_dialog_locks_astm_order_and_applies_to_workspace_spectrum():
    _qapp()
    manager = ModeManager(
        initial_state=ModeState(
            mode=GUIMode.STANDARDS,
            standard="ASTM E181",
            theme="dark",
        )
    )
    spectrum = build_demo_spectrum()
    bus = SelectionBus()
    applied = []

    dialog = CalibrationWorkspaceDialog(
        spectrum=spectrum,
        mode_manager=manager,
        selection_bus=bus,
        on_apply=lambda spec, energy_fit, fwhm_fit: applied.append(
            (spec, energy_fit, fwhm_fit)
        ),
    )
    _qapp().processEvents()

    assert dialog.energy_order.value() == 2
    assert dialog.energy_order.isEnabled() is False
    assert dialog.energy_table.rowCount() == 3
    assert dialog.fwhm_table.rowCount() == 3
    assert dialog.apply_button.isEnabled() is True

    dialog._apply_workspace_results()

    assert len(applied) == 1
    applied_spectrum, energy_fit, fwhm_fit = applied[0]
    assert applied_spectrum.calibration["energy"] == pytest.approx(energy_fit.coefficients)
    assert isinstance(fwhm_fit.fitted_fwhm_keV, np.ndarray)
    assert bus.describe()["reference_lines_keV"]
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_main_window_opens_phase2_calibration_workspace():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window._open_energy_fwhm_workspace()
    _qapp().processEvents()

    assert window._calibration_dialog is not None
    assert (
        window._calibration_dialog.windowTitle()
        == "FluxForge Next - Unified Calibration Workspace"
    )
    assert window._calibration_dialog.energy_table.rowCount() >= 3

    window._calibration_dialog.close()
    window.close()
