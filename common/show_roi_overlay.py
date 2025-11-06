#!/usr/bin/env python3
"""
PySide6 tabanlı ROI overlay gösterici.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import mss
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("mss modülü bulunamadı. `pip install mss` komutunu çalıştırın.") from exc


def grab_screen() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return np.array(sct.grab(monitor))


@dataclass
class ROI:
    left: int
    top: int
    width: int
    height: int

    def to_qrect(self) -> QtCore.QRect:
        return QtCore.QRect(self.left, self.top, self.width, self.height)


class Overlay(QtWidgets.QWidget):
    def __init__(self, roi: ROI, interval: int) -> None:
        super().__init__()
        self.roi = roi
        self.setWindowTitle("ROI Önizleme")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(max(30, interval))
        self._refresh()

    def _refresh(self) -> None:
        frame = grab_screen()
        h, w, _ = frame.shape
        rect = self.roi.to_qrect().intersected(QtCore.QRect(0, 0, w, h))

        bgr = frame[:, :, :3].copy(order="C")
        image = QtGui.QImage(
            bgr.data, w, h, bgr.strides[0], QtGui.QImage.Format_BGR888
        )
        pixmap = QtGui.QPixmap.fromImage(image)
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(255, 140, 0), 3)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.end()

        self.label.setPixmap(pixmap)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            QtWidgets.QApplication.quit()
        else:
            super().keyPressEvent(event)


def main(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ROI noktası overlay")
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--interval", type=int, default=100)
    ns = parser.parse_args(args)

    app = QtWidgets.QApplication(sys.argv[:1])
    window = Overlay(ROI(ns.left, ns.top, ns.width, ns.height), ns.interval)
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main(sys.argv[1:])
