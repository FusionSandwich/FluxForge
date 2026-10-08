"""Single-line status text that can shrink without losing the full value."""

from fluxforge.gui.qt_compat import QT_AVAILABLE

if QT_AVAILABLE:  # pragma: no cover - optional GUI dependency
    from PySide6.QtWidgets import QStylePainter
    from fluxforge.gui.qt_compat import QLabel, Qt

    class ElidingLabel(QLabel):
        """Retain full text for tooltips/accessibility; paint an ellipsis to fit."""

        def __init__(self, text="", parent=None):
            super().__init__(parent)
            self.setText(text)

        def setText(self, text):
            super().setText(text)
            self.setToolTip(text)
            self.setAccessibleName(text)

        def sizeHint(self):
            size = super().sizeHint()
            size.setWidth(min(240, size.width()))
            return size

        def minimumSizeHint(self):
            size = super().minimumSizeHint()
            size.setWidth(min(32, size.width()))
            return size

        def paintEvent(self, event):
            painter = QStylePainter(self)
            rect = self.contentsRect().adjusted(
                self.margin(), self.margin(), -self.margin(), -self.margin()
            )
            text = self.fontMetrics().elidedText(
                self.text(), Qt.ElideMiddle, max(0, rect.width())
            )
            painter.drawItemText(
                rect,
                self.alignment().value,
                self.palette(),
                self.isEnabled(),
                text,
                self.foregroundRole(),
            )

else:

    class ElidingLabel:  # pragma: no cover - optional GUI fallback
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ElidingLabel requires PySide6")
