#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
import mss

OUTPUT = Path("roi_snapshot.png")

with mss.mss() as sct:
    monitor = sct.monitors[1]
    frame = np.array(sct.grab(monitor))

cv2.imwrite(str(OUTPUT), frame)
print(f"Ekran görüntüsü kaydedildi: {OUTPUT.resolve()}")
