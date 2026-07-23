from __future__ import annotations

import os
from dataclasses import replace

import pytest

from fluxforge.core.analysis_workspace import PeakCandidate
from fluxforge.gui import QT_AVAILABLE, SelectionBus
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, PyQtGraphSpectrumCanvas
from fluxforge.gui.qt_compat import QApplication
from fluxforge.gui.spectrum_canvas import ReferenceLine, SpectrumTrace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


pytestmark = pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_pyqtgraph_viewport_round_trip_restores_supported_display_state() -> None:
    app = _qapp()
    selection_bus = SelectionBus()
    canvas = PyQtGraphSpectrumCanvas(selection_bus=selection_bus)
    canvas.resize(900, 600)
    canvas.show()
    canvas.set_traces(
        (
            SpectrumTrace(
                label="Foreground",
                counts=tuple(float(index + 1) for index in range(1000)),
                channels=tuple(float(index) for index in range(1000)),
                x_axis_label="Energy (keV)",
            ),
        )
    )
    canvas.set_annotation_lines((ReferenceLine(energy_keV=661.657, label="Cs-137"),))
    canvas.set_peak_residuals(
        (
            PeakCandidate(
                peak_id="peak-1",
                channel=662.0,
                energy_keV=661.657,
                significance=12.0,
                roi_bounds_keV=(650.0, 675.0),
                net_counts=500.0,
                fit_quality=1.1,
                normalized_residuals=(-0.4, 0.2, 0.7),
                residual_channels=(660.0, 661.0, 662.0),
            ),
        ),
        visible=True,
    )
    canvas.set_residual_mode("full")
    canvas.set_log_scale(True)
    canvas.set_peak_labels_visible(False)
    selection_bus.publish_roi(650.0, 675.0)
    canvas.roi_button.setChecked(True)
    canvas.set_crosshair_enabled(True)
    canvas.plot_item.setXRange(600.0, 725.0, padding=0.0)
    canvas.plot_item.setYRange(0.0, 3.0, padding=0.0)
    app.processEvents()

    expected = canvas.viewport_state(
        viewport_id="primary-spectrum",
        spectrum_id="spectrum-1",
        selected_roi_id="roi-1",
    )
    assert expected.overlays == ("roi",)
    assert expected.residual_mode == "full"
    assert expected.crosshair_enabled is True
    assert expected.labels_visible is False

    canvas.set_log_scale(False)
    canvas.set_peak_labels_visible(True)
    canvas.roi_button.setChecked(False)
    canvas.set_residual_mode("off")
    canvas.set_crosshair_enabled(False)
    canvas.plot_item.setXRange(0.0, 100.0, padding=0.0)
    canvas.plot_item.setYRange(0.0, 1000.0, padding=0.0)
    app.processEvents()

    canvas.apply_viewport_state(expected)
    app.processEvents()
    restored = canvas.viewport_state(
        viewport_id="primary-spectrum",
        spectrum_id="spectrum-1",
        selected_roi_id="roi-1",
    )

    assert restored.viewport_id == expected.viewport_id
    assert restored.spectrum_id == expected.spectrum_id
    assert restored.x_range == pytest.approx(expected.x_range)
    assert restored.y_range == pytest.approx(expected.y_range)
    assert restored.x_unit == expected.x_unit
    assert restored.y_unit == expected.y_unit
    assert restored.log_x == expected.log_x
    assert restored.log_y == expected.log_y
    assert restored.overlays == expected.overlays
    assert restored.residual_mode == expected.residual_mode
    assert restored.selected_roi_id == expected.selected_roi_id
    assert restored.crosshair_enabled == expected.crosshair_enabled
    assert restored.labels_visible == expected.labels_visible
    assert canvas._roi_region.isVisible()
    assert canvas.residual_row.isVisible()
    assert canvas._crosshair_vertical.isVisible()
    assert canvas._crosshair_horizontal.isVisible()
    assert not canvas._annotation_label_items

    canvas.close()


def test_pyqtgraph_viewport_restore_hides_optional_layers() -> None:
    app = _qapp()
    canvas = PyQtGraphSpectrumCanvas(selection_bus=SelectionBus())
    canvas.show()
    canvas.set_spectrum((1.0, 3.0, 2.0))
    canvas.set_peak_residuals(
        (
            PeakCandidate(
                peak_id="peak-1",
                channel=1.0,
                energy_keV=1.0,
                significance=3.0,
                roi_bounds_keV=(0.5, 1.5),
                net_counts=2.0,
                fit_quality=1.0,
                normalized_residuals=(0.1,),
                residual_channels=(1.0,),
            ),
        ),
        visible=True,
    )
    canvas.roi_button.setChecked(True)
    canvas.set_crosshair_enabled(True)
    app.processEvents()

    hidden = replace(
        canvas.viewport_state(),
        overlays=(),
        residual_mode="off",
        crosshair_enabled=False,
        labels_visible=False,
    )
    canvas.apply_viewport_state(hidden)
    app.processEvents()

    assert not canvas._roi_region.isVisible()
    assert not canvas.residual_row.isVisible()
    assert not canvas._crosshair_vertical.isVisible()
    assert not canvas._crosshair_horizontal.isVisible()
    canvas.close()
