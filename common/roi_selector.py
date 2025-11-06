#!/usr/bin/env python3
"""
Canlı ekran üzerinden küçük bir ROI (region of interest) seçmek için yardımcı araç.

Kontroller:
    Ok tuşları  : ROI merkezini taşır. (Shift ile 10 piksel)
    W / S       : Yüksekliği azalt / artır.
    A / D       : Genişliği azalt / artır.
    Q / E       : ROI boyutunu 10 piksel azalt / artır.
    R           : Adım değerini 1 ↔ 5 arasında değiştir.
    C           : ROI'yi ekranın merkezine sıfırlar.
    Enter       : Koordinatları (left, top, width, height) stdout'a yazıp çıkar.
    Esc         : Kaydetmeden çıkar.

Gereken paketler:
    pip install mss opencv-python numpy
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass
class ROI:
    left: int
    top: int
    width: int
    height: int

    def clamp(self, max_w: int, max_h: int) -> None:
        self.width = max(4, min(self.width, max_w))
        self.height = max(4, min(self.height, max_h))
        self.left = max(0, min(self.left, max_w - self.width))
        self.top = max(0, min(self.top, max_h - self.height))


def draw_overlay(frame: np.ndarray, roi: ROI) -> None:
    x1, y1 = roi.left, roi.top
    x2, y2 = roi.left + roi.width, roi.top + roi.height
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    legend = [
        f"ROI: left={roi.left} top={roi.top} width={roi.width} height={roi.height}",
        "Arrows: move  (R ile adım 1/5 piksel)",
        "W/S: height -/+   A/D: width -/+   Q/E: size -/+10",
        "C: center  Enter: save  Esc: quit",
    ]
    for idx, text in enumerate(legend):
        cv2.putText(
            frame,
            text,
            (16, 32 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (16, 32 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def main() -> None:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screen_w = monitor["width"]
        screen_h = monitor["height"]

        roi = ROI(
            left=screen_w // 2 - 80,
            top=screen_h // 2 - 20,
            width=160,
            height=40,
        )
        step = 1

        window_name = "ROI Seçici"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        print("ROI seçiciyi kapatmak için Enter (kaydet) veya Esc (iptal) tuşuna basın.")
        print("Kontroller pencere içinde gösterilmektedir.")

        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            draw_overlay(frame, roi)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(10) & 0xFFFF

            if key == 27:  # Esc
                print("İptal edildi.")
                break
            if key in (13, 10):  # Enter
                print(
                    f"left={roi.left} top={roi.top} width={roi.width} height={roi.height}"
                )
                break

            if key in (2490368, 63232):  # Up
                roi.top -= step
            elif key in (2621440, 63233):  # Down
                roi.top += step
            elif key in (2424832, 63234):  # Left
                roi.left -= step
            elif key in (2555904, 63235):  # Right
                roi.left += step
            elif key in (ord("w"), ord("W")):
                roi.height = max(4, roi.height - step)
            elif key in (ord("s"), ord("S")):
                roi.height += step
            elif key in (ord("a"), ord("A")):
                roi.width = max(4, roi.width - step)
            elif key in (ord("d"), ord("D")):
                roi.width += step
            elif key in (ord("q"), ord("Q")):
                roi.width = max(4, roi.width - 10)
                roi.height = max(4, roi.height - 10)
            elif key in (ord("e"), ord("E")):
                roi.width += 10
                roi.height += 10
            elif key in (ord("r"), ord("R")):
                step = 5 if step == 1 else 1
            elif key in (ord("c"), ord("C")):
                roi.left = screen_w // 2 - roi.width // 2
                roi.top = screen_h // 2 - roi.height // 2

            roi.clamp(screen_w, screen_h)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
