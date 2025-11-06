#!/usr/bin/env python3
"""
PySide6 tabanlı ROI seçici.

Fareyle ROI belirledikten sonra seçilen bölgeyi ekran görüntüsünden kırpar
ve proje kökünde `y_eksini/` klasörüne kaydeder.

Kullanım:
    python3 common/roi_capture.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import mss
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("mss modülü bulunamadı. `pip install mss` komutunu çalıştırın.") from exc


OUTPUT_DIR = Path("y_eksini")


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


def capture_screen() -> np.ndarray:
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

        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        scene.addPixmap(pixmap)

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
            scene_rect = scene_rect.intersected(self.sceneRect().toRect())
            if scene_rect.width() > 0 and scene_rect.height() > 0:
                self._current_rect = scene_rect
                self.roi_changed.emit(scene_rect)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._current_rect.isNull():
            super().keyPressEvent(event)
            return

        delta = 10 if event.modifiers() & QtCore.Qt.ShiftModifier else 1
        updated = QtCore.QRect(self._current_rect)
        bounds = self.sceneRect().toRect()

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
                updated.setWidth(min(bounds.width(), updated.width() + delta))
            case QtCore.Qt.Key_W:
                updated.setHeight(max(4, updated.height() - delta))
            case QtCore.Qt.Key_S:
                updated.setHeight(min(bounds.height(), updated.height() + delta))
            case QtCore.Qt.Key_C:
                updated.moveCenter(bounds.center())
            case _:
                super().keyPressEvent(event)
                return

        updated = updated.intersected(bounds)
        self._current_rect = updated
        self.roi_changed.emit(updated)
        self.viewport().update()

    def set_roi(self, rect: QtCore.QRect) -> None:
        self._current_rect = rect
        self.roi_changed.emit(rect)
        self.viewport().update()

    def current_roi(self) -> QtCore.QRect:
        return self._current_rect

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        if self._current_rect.isNull():
            return
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
        painter.drawRect(self._current_rect)


class ROICaptureWindow(QtWidgets.QMainWindow):
    def __init__(self, default_roi: ROI | None = None) -> None:
        super().__init__()
        frame = capture_screen()
        image = QtGui.QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format_BGR888,
        )
        pixmap = QtGui.QPixmap.fromImage(image)

        self.setWindowTitle("ROI Seçici")
        self.resize(1100, 720)

        self.view = GraphicsView(pixmap)
        self.setCentralWidget(self.view)
        self.view.roi_changed.connect(self._show_status)
        if default_roi:
            self.view.set_roi(default_roi.to_qrect())

        self.status = self.statusBar()
        self._current: ROI | None = None

        shortcuts = [
            (QtGui.QKeySequence(QtCore.Qt.Key_Return), self.accept),
            (QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_S), self.accept),
            (QtGui.QKeySequence(QtCore.Qt.Key_Escape), self.reject),
            (QtGui.QKeySequence(QtCore.Qt.CTRL | QtCore.Qt.Key_C), self.reject),
        ]
        for seq, func in shortcuts:
            QtGui.QShortcut(seq, self, func)

        instructions = QtWidgets.QLabel(
            "Fare: ROI çiz | Ok/WASD: taşı-boyutlandır (Shift=10px) | "
            "Enter: kaydet | Esc: çık"
        )
        instructions.setAlignment(QtCore.Qt.AlignCenter)
        instructions.setStyleSheet("padding: 6px; background: rgba(0,0,0,0.6); color: white;")
        dock = QtWidgets.QDockWidget()
        dock.setTitleBarWidget(QtWidgets.QWidget())
        dock.setWidget(instructions)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _show_status(self, rect: QtCore.QRect) -> None:
        if rect.isNull():
            self._current = None
            self.status.showMessage("ROI seçilmedi.")
        else:
            self._current = ROI.from_qrect(rect)
            self.status.showMessage(
                f"ROI -> left={rect.x()} top={rect.y()} width={rect.width()} height={rect.height()}"
            )

    def accept(self) -> None:
        if not self._current:
            QtWidgets.QMessageBox.warning(self, "ROI Seçici", "Önce bir ROI seçmelisiniz.")
            return
        roi = self._current

        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        frame = capture_screen()
        crop = frame[
            roi.top : roi.top + roi.height,
            roi.left : roi.left + roi.width,
            :3,
        ]
        filename = OUTPUT_DIR / f"roi_{roi.left}_{roi.top}_{roi.width}_{roi.height}.png"
        cv2 = None
        try:
            import cv2  # type: ignore[import]

            cv2.imwrite(str(filename), crop)
        except ModuleNotFoundError:
            from PIL import Image

            Image.fromarray(crop[..., ::-1]).save(filename)

        print(
            f"ROI kaydedildi -> left={roi.left} top={roi.top} "
            f"width={roi.width} height={roi.height} (dosya: {filename})",
            flush=True,
        )
        QtWidgets.QApplication.quit()

    def reject(self) -> None:
        QtWidgets.QApplication.quit()


def main(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ROI seçici ve yakalama aracı")
    parser.add_argument("--left", type=int, help="Varsayılan ROI sol koordinatı")
    parser.add_argument("--top", type=int, help="Varsayılan ROI üst koordinatı")
    parser.add_argument("--width", type=int, help="Varsayılan ROI genişliği")
    parser.add_argument("--height", type=int, help="Varsayılan ROI yüksekliği")
    args = parser.parse_args(argv)

    default_roi = None
    if None not in (args.left, args.top, args.width, args.height):
        default_roi = ROI(args.left, args.top, args.width, args.height)

    app = QtWidgets.QApplication(sys.argv)
    window = ROICaptureWindow(default_roi=default_roi)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main(sys.argv[1:])
