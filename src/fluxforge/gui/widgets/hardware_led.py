"""Status widgets for the next-generation GUI shell."""

from __future__ import annotations

from fluxforge.gui.qt_compat import QT_AVAILABLE

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import QHBoxLayout, QLabel, QFrame, QWidget


LED_STYLES = {
    "offline": ("OFFLINE", "#64748b"),
    "idle": ("IDLE", "#2ecc71"),
    "busy": ("ACQUIRING", "#f59e0b"),
    "fault": ("FAULT", "#e74c3c"),
}


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class HardwareLedWidget(QWidget):
        """Compact hardware-state widget used in the status bar."""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)

            self.dot = QFrame(self)
            self.dot.setObjectName("HardwareLedDot")
            self.dot.setFixedSize(12, 12)

            self.label = QLabel(self)
            self.label.setObjectName("HardwareLedLabel")

            layout.addWidget(self.dot)
            layout.addWidget(self.label)

            self.set_status("offline")

        def set_status(self, status: str, message: str | None = None) -> None:
            tone, color = LED_STYLES.get(status, LED_STYLES["offline"])
            self.dot.setStyleSheet(
                "QFrame#HardwareLedDot {"
                f"background-color: {color}; border-radius: 6px;"
                "border: 1px solid rgba(255,255,255,0.12); }"
            )
            self.label.setText(message or tone)

else:

    class HardwareLedWidget:  # pragma: no cover - placeholder without Qt
        """Import-safe placeholder when Qt is unavailable."""

        def __init__(self, parent=None) -> None:
            self.parent = parent
            self.status = "offline"

        def set_status(self, status: str, message: str | None = None) -> None:
            self.status = status
