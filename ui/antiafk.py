import cv2
import numpy as np
from pathlib import Path

src = Path(r"C:\Users\OKAN ASLAN\Desktop\koordinat\ui\antiafk.png")
dst = Path(r"C:\Users\OKAN ASLAN\Desktop\koordinat\ui\antiafk2.png")

image = cv2.imread(str(src))
if image is None:
    raise SystemExit("antiafk.png okunamadı.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Sarı için sıkı bir aralık (gerekirse S/V değerlerini biraz daraltabilirsin)
mask = cv2.inRange(hsv, (18, 120, 140), (35, 255, 255))

# Harf boşluklarını doldurmak için küçük bir kapama işlemi
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Konturu yumuşatmak için hafif blur
mask = cv2.GaussianBlur(mask, (3, 3), 0)

# Tamamen siyah-beyaz hale getir
_, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

cv2.imwrite(str(dst), binary)
print("Sadeleştirilmiş şablon kaydedildi:", dst)
