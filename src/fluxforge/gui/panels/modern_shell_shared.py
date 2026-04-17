"""Shared helpers for the split modern-shell panel modules."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionState
from fluxforge.io.spe import GammaSpectrum


MODERN_LOG_LINES = (
    "Qt shell initialized",
    "Renderer strategy: PyQtGraph first, Vispy additive",
    "Mode-aware workflow locking ready",
    "Tk GUI demoted to explicit legacy fallback",
)


def selection_summary(state: SelectionState) -> str:
    """Format the current cross-panel selection state."""

    fragments = []
    if state.peak_energy_keV is not None:
        fragments.append(f"Peak {state.peak_energy_keV:.3f} keV")
    if state.nuclide:
        fragments.append(state.nuclide)
    if state.roi_bounds_keV:
        fragments.append(
            f"ROI {state.roi_bounds_keV[0]:.1f}-{state.roi_bounds_keV[1]:.1f} keV"
        )
    if state.reference_lines_keV:
        fragments.append(f"{len(state.reference_lines_keV)} ref lines")
    if state.annotation_lines:
        fragments.append(f"{len(state.annotation_lines)} guides")
    return " | ".join(fragments) if fragments else "No active selection"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _demo_counts() -> tuple[float, ...]:
    counts = []
    for channel in range(2048):
        background = 18.0 + (channel / 96.0)
        peak_a = 1180.0 / (1.0 + ((channel - 662.0) / 10.5) ** 2)
        peak_b = 780.0 / (1.0 + ((channel - 1173.0) / 14.0) ** 2)
        peak_c = 640.0 / (1.0 + ((channel - 1332.0) / 16.0) ** 2)
        counts.append(background + peak_a + peak_b + peak_c)
    return tuple(counts)


def build_demo_spectrum() -> GammaSpectrum:
    counts = _demo_counts()
    channels = tuple(float(index) for index in range(len(counts)))
    start_time = datetime(2026, 3, 30, 11, 0)
    return GammaSpectrum(
        counts=np.asarray(counts, dtype=float),
        channels=np.asarray(channels, dtype=float),
        calibration={"energy": [0.0, 1.0]},
        live_time=300.0,
        real_time=321.0,
        start_time=start_time,
        spectrum_id="demo_hpge_workspace",
        detector_id="demo-hpge",
        gps={"latitude": 43.0731, "longitude": -89.4012},
        metadata={
            "source": "analysis-demo",
            "gps": {"latitude": 43.0731, "longitude": -89.4012},
            "input_count_rate_cps": 47500.0,
        },
    )


def build_demo_background_spectrum() -> GammaSpectrum:
    counts = np.asarray(_demo_counts(), dtype=float) * 0.16
    channels = np.arange(len(counts), dtype=float)
    start_time = datetime(2026, 3, 30, 10, 0)
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        calibration={"energy": [0.0, 1.0]},
        live_time=300.0,
        real_time=309.0,
        start_time=start_time,
        spectrum_id="demo_hpge_background",
        detector_id="demo-hpge",
        gps={"latitude": 43.0736, "longitude": -89.4019},
        metadata={
            "source": "analysis-demo-background",
            "gps": {"latitude": 43.0736, "longitude": -89.4019},
            "input_count_rate_cps": 13800.0,
        },
    )


def build_demo_overlay_spectrum() -> GammaSpectrum:
    counts = np.asarray(_demo_counts(), dtype=float) * 0.62
    counts[540:620] *= 1.18
    channels = np.arange(len(counts), dtype=float)
    start_time = datetime(2026, 3, 30, 10, 30)
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        calibration={"energy": [0.0, 1.0]},
        live_time=300.0,
        real_time=316.0,
        start_time=start_time,
        spectrum_id="demo_hpge_overlay",
        detector_id="demo-hpge",
        gps={"latitude": 43.0742, "longitude": -89.4024},
        metadata={
            "source": "analysis-demo-overlay",
            "gps": {"latitude": 43.0742, "longitude": -89.4024},
            "input_count_rate_cps": 29800.0,
        },
    )


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import QComboBox, QFrame, QLabel, QTabWidget, QVBoxLayout

    def card(title: str, body: str, accent: str | None = None) -> QFrame:
        frame = QFrame()
        frame.setObjectName("HeroCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title_label = QLabel(title, frame)
        title_label.setObjectName("HeroCardTitle")
        layout.addWidget(title_label)

        body_label = QLabel(body, frame)
        body_label.setWordWrap(True)
        body_label.setObjectName("HeroCardBody")
        layout.addWidget(body_label)

        if accent:
            accent_label = QLabel(accent, frame)
            accent_label.setObjectName("HeroCardAccent")
            accent_label.setWordWrap(True)
            layout.addWidget(accent_label)

        layout.addStretch(1)
        return frame

    def current_tab_label(widget: QTabWidget) -> str:
        index = widget.currentIndex()
        if index < 0:
            return ""
        return str(widget.tabText(index))

    def set_tab_label(widget: QTabWidget, label: str) -> None:
        target = str(label or "").strip()
        if not target:
            return
        for index in range(widget.count()):
            if widget.tabText(index) == target:
                widget.setCurrentIndex(index)
                return

    def set_combo_data(widget: QComboBox, value: object) -> None:
        if value is None:
            return
        index = widget.findData(value)
        if index < 0:
            index = widget.findText(str(value))
        if index >= 0:
            widget.setCurrentIndex(index)


__all__ = [
    "MODERN_LOG_LINES",
    "build_demo_background_spectrum",
    "build_demo_overlay_spectrum",
    "build_demo_spectrum",
    "format_duration",
    "format_percent",
    "selection_summary",
]

if QT_AVAILABLE:  # pragma: no cover - export Qt-only helper
    __all__ += [
        "card",
        "current_tab_label",
        "set_combo_data",
        "set_tab_label",
    ]
