#!/usr/bin/env python3
"""
PySide6 tabanlı basit ROI seçici.

Kullanım:
    python3 common/roi_selector_qt.py

Kontroller:
    - Fareyle sürükleyerek ROI seç.
    - Ok tuşları ROI'yi taşır. Shift + Ok -> 10 piksel.
    - W/S yüksekliği, A/D genişliği değiştirir (Shift ile 10 piksel).
    - Enter veya Ctrl+S => Seçimi kaydedip koordinatları stdout'a yazar.
    - Esc veya Ctrl+C => Çıkış.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import mss
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("mss modülü bulunamadı. `pip install mss` komutunu çalıştırın.") from exc


@dataclass
class ROI:
    left: int
    top: int
    width: int
    height: int

    def to_qrect(self) -> QtCore.QRect:
        return QtCore.QRect(self.left, self.top, self.width, self.height)

    @classmethod
    def from_qrect(cls, rect: QtCore.QRect) -> "ROI":
        return cls(rect.x(), rect.y(), rect.width(), rect.height())


def capture_screenshot() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame = np.array(sct.grab(monitor))
    return frame


class GraphicsView(QtWidgets.QGraphicsView):
    roi_changed = QtCore.Signal(QtCore.QRect)

    def __init__(self, pixmap: QtGui.QPixmap, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._scene.addPixmap(pixmap)

        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.viewport())
        self._origin = QtCore.QPoint()
        self._current_rect = QtCore.QRect()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._origin = event.position().toPoint()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._rubber.isVisible():
            rect = QtCore.QRect(self._origin, event.position().toPoint()).normalized()
            self._rubber.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._rubber.isVisible():
            self._rubber.hide()
            rect = self._rubber.geometry()
            scene_rect = self.mapToScene(rect).boundingRect().toRect()
            self._current_rect = scene_rect.intersected(self.sceneRect().toRect())
            if self._current_rect.width() > 0 and self._current_rect.height() > 0:
                self.roi_changed.emit(self._current_rect)
        super().mouseReleaseEvent(event)

    def set_roi(self, rect: QtCore.QRect) -> None:
        self._current_rect = rect
        self.roi_changed.emit(rect)
        self.viewport().update()

    def current_roi(self) -> QtCore.QRect:
        return self._current_rect if not self._current_rect.isNull() else QtCore.QRect()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        delta = 10 if event.modifiers() & QtCore.Qt.ShiftModifier else 1
        rect = self._current_rect
        if rect.isNull():
            super().keyPressEvent(event)
            return
        updated = QtCore.QRect(rect)
        match event.key():
            case QtCore.Qt.Key_Left:
                updated.translate(-delta, 0)
            case QtCore.Qt.Key_Right:
                updated.translate(delta, 0)
            case QtCore.Qt.Key_Up:
                updated.translate(0, -delta)
            case QtCore.Qt.Key_Down:
                updated.translate(0, delta)
            case QtCore.Qt.Key_A:
                updated.setWidth(max(4, updated.width() - delta))
            case QtCore.Qt.Key_D:
                updated.setWidth(updated.width() + delta)
            case QtCore.Qt.Key_W:
                updated.setHeight(max(4, updated.height() - delta))
            case QtCore.Qt.Key_S:
                updated.setHeight(updated.height() + delta)
            case QtCore.Qt.Key_C:
                bounds = self.sceneRect().toRect()
                updated.moveCenter(bounds.center())
            case _:
                super().keyPressEvent(event)
                return
        bounds = self.sceneRect().toRect()
        updated = updated.intersected(bounds)
        self.set_roi(updated)

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        if self._current_rect.isNull():
            return
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 140, 0), 2))
        painter.drawRect(self._current_rect)


class ROISelector(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        frame = capture_screenshot()
        bgr = QtGui.QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format_BGR888,
        )
        pixmap = QtGui.QPixmap.fromImage(bgr)

        self.setWindowTitle("ROI Seçici")
        self.resize(1100, 700)

        self.view = GraphicsView(pixmap)
        self.setCentralWidget(self.view)

        self.status = self.statusBar()
        self.view.roi_changed.connect(self._update_status)

        self._current = QtCore.QRect()

        shortcuts = [
            (QtGui.QKeySequence(QtCore.Qt.Key_Return), self.accept),
            (QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_S), self.accept),
            (QtGui.QKeySequence(QtCore.Qt.Key_Escape), self.reject),
            (QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_C), self.reject),
        ]
        for keyseq, func in shortcuts:
            QtGui.QShortcut(keyseq, self, func)

        instructions = QtWidgets.QLabel(
            "Fare ile seç, ok/WASD ile ayarla, Shift ile 10 piksel, Enter: Kaydet, Esc: Çık"
        )
        instructions.setAlignment(QtCore.Qt.AlignCenter)
        instructions.setStyleSheet("padding: 6px; background: rgba(0,0,0,0.6); color: white;")
        dock = QtWidgets.QDockWidget()
        dock.setTitleBarWidget(QtWidgets.QWidget())
        dock.setWidget(instructions)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _update_status(self, rect: QtCore.QRect) -> None:
        self._current = rect
        if rect.isNull():
            self.status.showMessage("ROI seçilmedi.")
        else:
            self.status.showMessage(
                f"ROI -> left={rect.x()} top={rect.y()} width={rect.width()} height={rect.height()}"
            )

    def accept(self) -> None:
        if self._current.isNull():
            QtWidgets.QMessageBox.warning(self, "ROI Seçici", "Önce bir ROI seçmelisiniz.")
            return
        roi = ROI.from_qrect(self._current)
        print(
            f"left={roi.left} top={roi.top} width={roi.width} height={roi.height}",
            flush=True,
        )
        QtWidgets.QApplication.quit()

    def reject(self) -> None:
        QtWidgets.QApplication.quit()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    selector = ROISelector()
    selector.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
