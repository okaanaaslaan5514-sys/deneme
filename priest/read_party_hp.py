#!/usr/bin/env python3
"""
Parti HP çubuklarını oyun ekranından okuyup yüzdeleri raporlar.

Kalibrasyon adımları
--------------------
1. `python read_party_hp.py --calibrate-roi --slots 2` komutu ile
   - önce açılan pencerede ok tuşları + WASD ile X butonunu ortalayın (Enter ile onaylayın),
   - ardından ikinci pencerede ok tuşları + WASD ile 1. slot ROI'sini,
     `1/2` tuşları ile de slotlar arası dikey mesafeyi ayarlayın.
   Böylece `anchor.json` ve `party_hp_calibration.json` hazırlanır.
2. Canlı okumalar için `python read_party_hp.py --slots 2 --interval 0.5` yeterlidir.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract

# Tesseract OCR ayarları (yalnızca rakam ve '/' karakterlerini kabul et)
TESSERACT_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/"
DEFAULT_TESSERACT_PATHS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path("/mnt/c/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("/mnt/c/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    Path(__file__).resolve().parent / "tesseract" / "tesseract.exe",
)
BAR_PATTERN = re.compile(r"(?P<current>\d+)\s*/\s*(?P<maximum>\d+)")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANCHOR_FILE = BASE_DIR / "anchor.json"
DEFAULT_CALIBRATION_FILE = BASE_DIR / "party_hp_calibration.json"
MIN_SLOTS = 1
MAX_SLOTS = 8
DEFAULT_ROI_OFFSET_X = -110
DEFAULT_ROI_OFFSET_Y = 18
DEFAULT_ROI_WIDTH = 130
DEFAULT_ROI_HEIGHT = 28
DEFAULT_SLOT_SPACING = 71


@dataclass(frozen=True)
class ROI:
    left: int
    top: int
    width: int
    height: int


@dataclass
class AnchorSettings:
    center_x: int
    center_y: int
    width: int = 32
    height: int = 32

    def as_tuple(self) -> Tuple[int, int]:
        return self.center_x, self.center_y


@dataclass
class PartySlot:
    index: int
    roi: ROI


@dataclass
class Calibration:
    roi_offset_x: int
    roi_offset_y: int
    roi_width: int
    roi_height: int
    slot_vertical_spacing: int
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "roi_offset_x": int(self.roi_offset_x),
            "roi_offset_y": int(self.roi_offset_y),
            "roi_width": int(self.roi_width),
            "roi_height": int(self.roi_height),
            "slot_vertical_spacing": int(self.slot_vertical_spacing),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        try:
            return cls(
                roi_offset_x=int(data["roi_offset_x"]),
                roi_offset_y=int(data["roi_offset_y"]),
                roi_width=int(data["roi_width"]),
                roi_height=int(data["roi_height"]),
                slot_vertical_spacing=int(data["slot_vertical_spacing"]),
                timestamp=str(data.get("timestamp", "")),
            )
        except KeyError as exc:
            raise RuntimeError(f"Kalibrasyon verisi eksik: {exc}") from exc


def configure_tesseract(manual_path: Optional[Path]) -> None:
    """
    Tesseract ikili dosyasının yolunu belirle.
    """
    if manual_path:
        pytesseract.pytesseract.tesseract_cmd = str(manual_path)
        return
    if shutil.which("tesseract"):
        return
    for candidate in DEFAULT_TESSERACT_PATHS:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return
    raise SystemExit(
        "Tesseract yürütülebilir dosyası bulunamadı. "
        "Tesseract kurulumunu doğrulayın veya '--tesseract PATH' parametresi sağlayın."
    )


def compute_percentage(current: int, maximum: int) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(100.0, (current / maximum) * 100.0))


def parse_bar_text(text: str) -> Tuple[int, int]:
    cleaned = text.strip()
    match = BAR_PATTERN.search(cleaned)
    if not match:
        raise ValueError(f"Barkodu çözümlenemedi: {text!r}")
    return int(match.group("current")), int(match.group("maximum"))


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def capture_roi(roi: ROI) -> np.ndarray:
    import mss

    with mss.mss() as sct:
        region = {
            "left": int(roi.left),
            "top": int(roi.top),
            "width": int(roi.width),
            "height": int(roi.height),
        }
        frame = sct.grab(region)
    return np.array(frame)


def read_bar_from_frame(frame: np.ndarray) -> Tuple[int, int]:
    processed = preprocess(frame)
    text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
    return parse_bar_text(text)


def read_bar_from_screen(roi: ROI) -> Tuple[int, int]:
    frame = capture_roi(roi)
    return read_bar_from_frame(frame)


def grab_screen() -> np.ndarray:
    import mss

    with mss.mss() as sct:
        monitor = sct.monitors[0]
        frame = sct.grab(monitor)
    return np.array(frame)


def _prepare_display_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.copy()
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def _interactive_anchor_editor(initial: AnchorSettings) -> AnchorSettings:
    window_name = "Anchor Kalibrasyonu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    base_frame = grab_screen()
    frame_height, frame_width = base_frame.shape[:2]
    cv2.resizeWindow(window_name, min(frame_width, 1280), min(frame_height, 720))

    center_x = int(initial.center_x)
    center_y = int(initial.center_y)
    width = max(10, int(initial.width))
    height = max(10, int(initial.height))

    def clamp(frame_shape: Tuple[int, int, int]) -> None:
        nonlocal center_x, center_y, width, height
        frame_height, frame_width = frame_shape[:2]
        width = max(10, min(width, frame_width))
        height = max(10, min(height, frame_height))
        half_w = max(5, width // 2)
        half_h = max(5, height // 2)
        center_x = min(max(center_x, half_w), frame_width - half_w)
        center_y = min(max(center_y, half_h), frame_height - half_h)

    instructions = [
        "Ok tuşları: Anchor konumunu kaydır",
        "W/S: yükseklik azalt/artır",
        "A/D: genişlik azalt/artır",
        "R: ekran görüntüsünü yenile",
        "P: görünümü değiştir (gizli -> blur -> normal)",
        "Shift (büyük harf): 5 piksel adım",
        "Enter: kaydet | ESC: iptal",
    ]

    preview_mode = 0  # 0: hidden, 1: blurred, 2: normal
    blurred_frame: Optional[np.ndarray] = None

    try:
        while True:
            if preview_mode == 0:
                display = np.zeros_like(base_frame)
            elif preview_mode == 1:
                if blurred_frame is None:
                    blurred_frame = cv2.GaussianBlur(base_frame, (51, 51), 0)
                display = blurred_frame.copy()
            else:
                display = base_frame.copy()

            display = _prepare_display_frame(display)
            clamp(display.shape)

            half_w = width // 2
            half_h = height // 2
            x1 = int(center_x - half_w)
            y1 = int(center_y - half_h)
            x2 = int(center_x + half_w)
            y2 = int(center_y + half_h)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.drawMarker(display, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

            overlay_y = 24
            for line in instructions:
                cv2.putText(
                    display,
                    line,
                    (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                overlay_y += 24

            status = f"Anchor: x={center_x}, y={center_y}, w={width}, h={height}"
            cv2.putText(
                display,
                status,
                (10, overlay_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(50)
            if key == -1:
                continue

            if key in (27,):  # ESC
                raise RuntimeError("Anchor seçimi kullanıcı tarafından iptal edildi.")
            if key in (13, 10, 141, 65293):  # Enter
                return AnchorSettings(center_x=center_x, center_y=center_y, width=width, height=height)

            if key in (ord("r"), ord("R")):
                base_frame = grab_screen()
                blurred_frame = None
                frame_height, frame_width = base_frame.shape[:2]
                cv2.resizeWindow(window_name, min(frame_width, 1280), min(frame_height, 720))
                continue
            if key in (ord("p"), ord("P")):
                preview_mode = (preview_mode + 1) % 3
                continue

            step = 1
            if key in (ord("W"), ord("A"), ord("S"), ord("D")):
                step = 5

            if key in (2424832, 65361, 81):  # Left
                center_x -= step
            elif key in (2555904, 65363, 83):  # Right
                center_x += step
            elif key in (2490368, 65362, 82):  # Up
                center_y -= step
            elif key in (2621440, 65364, 84):  # Down
                center_y += step
            elif key in (ord("w"), ord("W")):
                height = max(10, height - step)
            elif key in (ord("s"), ord("S")):
                height += step
            elif key in (ord("a"), ord("A")):
                width = max(10, width - step)
            elif key in (ord("d"), ord("D")):
                width += step

            clamp(base_frame.shape)
    finally:
        cv2.destroyWindow(window_name)


def select_anchor(existing: Optional[AnchorSettings] = None) -> AnchorSettings:
    frame = grab_screen()
    height, width = frame.shape[:2]
    if existing:
        initial = AnchorSettings(
            center_x=int(existing.center_x),
            center_y=int(existing.center_y),
            width=max(10, int(existing.width)),
            height=max(10, int(existing.height)),
        )
    else:
        initial = AnchorSettings(
            center_x=width // 2,
            center_y=height // 2,
            width=32,
            height=32,
        )
    return _interactive_anchor_editor(initial)


def load_anchor_file(path: Path) -> AnchorSettings:
    data = json.loads(path.read_text())
    try:
        center_x = int(data["x"])
        center_y = int(data["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Anchor dosyası bozuk: {path}") from exc
    width = int(data.get("width", 32))
    height = int(data.get("height", 32))
    return AnchorSettings(center_x=center_x, center_y=center_y, width=max(10, width), height=max(10, height))


def save_anchor_file(path: Path, anchor: AnchorSettings) -> None:
    payload = {
        "x": int(anchor.center_x),
        "y": int(anchor.center_y),
        "width": int(anchor.width),
        "height": int(anchor.height),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> Calibration:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    return Calibration.from_dict(data)


def save_calibration(path: Path, calibration: Calibration) -> None:
    path.write_text(json.dumps(calibration.to_dict(), indent=2))


def _ensure_bounds(
    frame_shape: Tuple[int, int, int],
    slots: int,
    left: int,
    top: int,
    width: int,
    height: int,
    spacing: int,
) -> Tuple[int, int, int, int, int]:
    """Clamp ROI & spacing values so that all slots stay within the screen."""
    frame_height, frame_width = frame_shape[:2]

    width = max(10, min(width, frame_width))
    height = max(10, min(height, frame_height))

    if slots > 1:
        max_spacing = max(1, (frame_height - 10) // (slots - 1))
        spacing = max(1, min(spacing, max_spacing))
        available_height = frame_height - (slots - 1) * spacing
        if available_height < 10:
            available_height = max(5, frame_height // max(slots, 1))
        height = max(10, min(height, available_height))
        if height > available_height:
            height = max(10, available_height)
    else:
        spacing = 1

    total_height = height + (slots - 1) * spacing
    if total_height > frame_height:
        excess = total_height - frame_height
        reducible = min(excess, height - 10)
        height -= reducible
        total_height = height + (slots - 1) * spacing
        if total_height > frame_height and slots > 1:
            spacing = max(1, (frame_height - height) // (slots - 1))
            total_height = height + (slots - 1) * spacing

    max_left = max(0, frame_width - width)
    left = min(max(left, 0), max_left)

    max_top = max(0, frame_height - total_height)
    top = min(max(top, 0), max_top)

    return left, top, width, height, spacing


def _interactive_roi_editor(
    anchor: Tuple[int, int],
    initial_roi: ROI,
    spacing: int,
    slots: int,
) -> Tuple[ROI, int]:
    window_name = "ROI Kalibrasyonu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    base_frame = grab_screen()
    frame_height, frame_width = base_frame.shape[:2]
    cv2.resizeWindow(window_name, min(frame_width, 1600), min(frame_height, 900))

    base_left = int(initial_roi.left)
    base_top = int(initial_roi.top)
    width = int(initial_roi.width)
    height = int(initial_roi.height)
    spacing_val = int(max(1, spacing))

    def adjust_bounds(frame_shape: Tuple[int, int, int]) -> None:
        nonlocal base_left, base_top, width, height, spacing_val
        base_left, base_top, width, height, spacing_val = _ensure_bounds(
            frame_shape,
            slots,
            base_left,
            base_top,
            width,
            height,
            spacing_val,
        )

    def build_rois() -> List[ROI]:
        return [
            ROI(
                left=base_left,
                top=base_top + idx * spacing_val,
                width=width,
                height=height,
            )
            for idx in range(slots)
        ]

    instructions = [
        "Ok tuşları: ROI konumunu kaydır",
        "W/S: yükseklik artır/azalt (W küçültür, S büyütür)",
        "A/D: genişlik azalt/artır",
        "1/2: slotlar arası mesafeyi azalt/artır",
        "R: ekran görüntüsünü yenile",
        "P: görünümü değiştir (gizli -> blur -> normal)",
        "Shift (büyük harf/!/@): 5 piksel adım",
        "Enter: kaydet | ESC: iptal",
    ]

    blurred_frame: Optional[np.ndarray] = None
    preview_mode = 0

    try:
        while True:
            if preview_mode == 0:
                display = np.zeros_like(base_frame)
            elif preview_mode == 1:
                if blurred_frame is None:
                    blurred_frame = cv2.GaussianBlur(base_frame, (51, 51), 0)
                display = blurred_frame.copy()
            else:
                display = base_frame.copy()

            display = _prepare_display_frame(display)
            adjust_bounds(display.shape)
            rois = build_rois()
            for slot in rois:
                cv2.rectangle(
                    display,
                    (slot.left, slot.top),
                    (slot.left + slot.width, slot.top + slot.height),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    str(slot.top),
                    (slot.left, max(0, slot.top - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            overlay_y = 24
            for line in instructions:
                cv2.putText(
                    display,
                    line,
                    (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                overlay_y += 24

            status_line = (
                f"Base ROI: x={base_left}, y={base_top}, w={width}, h={height} | "
                f"Spacing={spacing_val}"
            )
            cv2.putText(
                display,
                status_line,
                (10, overlay_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(50)
            if key == -1:
                continue

            if key in (27,):  # ESC
                raise RuntimeError("Kalibrasyon kullanıcı tarafından iptal edildi.")
            if key in (13, 10, 141, 65293):  # Enter keys
                return ROI(base_left, base_top, width, height), spacing_val

            step = 1
            if key in (ord("W"), ord("A"), ord("S"), ord("D"), ord("!"), ord("@")):
                step = 5

            if key in (ord("r"), ord("R")):
                base_frame = grab_screen()
                blurred_frame = None
                frame_height, frame_width = base_frame.shape[:2]
                cv2.resizeWindow(window_name, min(frame_width, 1600), min(frame_height, 900))
                continue
            if key in (ord("p"), ord("P")):
                preview_mode = (preview_mode + 1) % 3
                continue

            # Arrow keys (Windows/Linux)
            if key in (2424832, 65361, 81):  # Left
                base_left -= step
            elif key in (2555904, 65363, 83):  # Right
                base_left += step
            elif key in (2490368, 65362, 82):  # Up
                base_top -= step
            elif key in (2621440, 65364, 84):  # Down
                base_top += step
            elif key in (ord("w"), ord("W")):  # height down
                height = max(10, height - step)
            elif key in (ord("s"), ord("S")):  # height up
                height += step
            elif key in (ord("a"), ord("A")):  # width down
                width = max(10, width - step)
            elif key in (ord("d"), ord("D")):  # width up
                width += step
            elif key in (ord("1"), ord("!")):
                spacing_val = max(1, spacing_val - step)
            elif key in (ord("2"), ord("@")):
                spacing_val += step

            adjust_bounds(base_frame.shape)
    finally:
        cv2.destroyWindow(window_name)


def calibrate_rois(anchor_file: Path, calibration_file: Path, slots: int) -> None:
    print("Anchor (X butonu) konumunu seçin...")
    existing_anchor: Optional[AnchorSettings] = None
    if anchor_file.exists():
        try:
            existing_anchor = load_anchor_file(anchor_file)
        except Exception:
            existing_anchor = None
    anchor_settings = select_anchor(existing_anchor)
    save_anchor_file(anchor_file, anchor_settings)
    anchor = anchor_settings.as_tuple()
    print(f"Anchor kaydedildi: X={anchor[0]}, Y={anchor[1]} -> {anchor_file}")

    time.sleep(0.3)

    if calibration_file.exists():
        try:
            prev_calibration = load_calibration(calibration_file)
            initial_roi = ROI(
                left=anchor[0] + prev_calibration.roi_offset_x,
                top=anchor[1] + prev_calibration.roi_offset_y,
                width=prev_calibration.roi_width,
                height=prev_calibration.roi_height,
            )
            spacing = prev_calibration.slot_vertical_spacing
        except Exception:
            initial_roi = ROI(
                left=anchor[0] + DEFAULT_ROI_OFFSET_X,
                top=anchor[1] + DEFAULT_ROI_OFFSET_Y,
                width=DEFAULT_ROI_WIDTH,
                height=DEFAULT_ROI_HEIGHT,
            )
            spacing = DEFAULT_SLOT_SPACING
    else:
        time.sleep(0.3)
        initial_roi = ROI(
            left=anchor[0] + DEFAULT_ROI_OFFSET_X,
            top=anchor[1] + DEFAULT_ROI_OFFSET_Y,
            width=DEFAULT_ROI_WIDTH,
            height=DEFAULT_ROI_HEIGHT,
        )
        spacing = DEFAULT_SLOT_SPACING

    print("ROI ayarlama penceresi açılıyor. Enter ile kaydedin, ESC ile iptal edin.")
    base_roi, spacing_val = _interactive_roi_editor(anchor, initial_roi, spacing, slots)

    calibration = Calibration(
        roi_offset_x=int(base_roi.left - anchor[0]),
        roi_offset_y=int(base_roi.top - anchor[1]),
        roi_width=int(base_roi.width),
        roi_height=int(base_roi.height),
        slot_vertical_spacing=int(spacing_val),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    save_calibration(calibration_file, calibration)
    print(f"Kalibrasyon kaydedildi: {calibration_file}")


def compute_slot_rois(anchor: Tuple[int, int], calibration: Calibration, slots: int) -> List[PartySlot]:
    cx, cy = anchor
    rois: List[PartySlot] = []
    for idx in range(slots):
        left = cx + calibration.roi_offset_x
        top = cy + calibration.roi_offset_y + idx * calibration.slot_vertical_spacing
        rois.append(
            PartySlot(
                index=idx + 1,
                roi=ROI(
                    left=int(left),
                    top=int(top),
                    width=int(calibration.roi_width),
                    height=int(calibration.roi_height),
                ),
            )
        )
    return rois


def draw_rois(frame: np.ndarray, rois: Sequence[PartySlot]) -> np.ndarray:
    annotated = frame.copy()
    display = _prepare_display_frame(annotated)
    for slot in rois:
        x1 = max(slot.roi.left, 0)
        y1 = max(slot.roi.top, 0)
        x2 = x1 + slot.roi.width
        y2 = y1 + slot.roi.height
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            str(slot.index),
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return display


def ensure_rois_within_frame(frame: np.ndarray, rois: Sequence[PartySlot]) -> None:
    height, width = frame.shape[:2]
    for slot in rois:
        x1 = slot.roi.left
        y1 = slot.roi.top
        x2 = x1 + slot.roi.width
        y2 = y1 + slot.roi.height
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise RuntimeError(f"Slot {slot.index} ROI ekran sınırını aşıyor: {slot.roi}")


def read_hp_from_frame(
    frame: np.ndarray,
    slots: int,
    anchor: Tuple[int, int],
    calibration: Calibration,
) -> Tuple[List[Optional[Tuple[int, int]]], List[PartySlot]]:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    rois = compute_slot_rois(anchor, calibration, slots)
    ensure_rois_within_frame(frame, rois)

    values: List[Optional[Tuple[int, int]]] = []
    for slot in rois:
        x1 = max(slot.roi.left, 0)
        y1 = max(slot.roi.top, 0)
        x2 = x1 + slot.roi.width
        y2 = y1 + slot.roi.height
        subframe = frame[y1:y2, x1:x2]
        if subframe.shape[2] == 3:
            subframe = cv2.cvtColor(subframe, cv2.COLOR_BGR2BGRA)
        try:
            current, maximum = read_bar_from_frame(subframe)
            values.append((current, maximum))
        except Exception as exc:
            values.append(None)
            print(f"Slot {slot.index} okunamadı: {exc}", file=sys.stderr)
    return values, rois


def read_hp_from_image(
    image_path: Path,
    slots: int,
    anchor: Tuple[int, int],
    calibration: Calibration,
) -> Tuple[List[Optional[Tuple[int, int]]], List[PartySlot], np.ndarray]:
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"Görsel açılamadı: {image_path}")
    values, rois = read_hp_from_frame(frame, slots, anchor, calibration)
    return values, rois, frame


def format_output(slot_index: int, values: Optional[Tuple[int, int]]) -> str:
    if values is None:
        return f"Slot {slot_index}: okunamadı"
    current, maximum = values
    pct = compute_percentage(current, maximum)
    return f"Slot {slot_index}: {current}/{maximum} ({pct:.1f}%)"


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--from-image",
        type=Path,
        help="Kaydedilmiş bir ekran görüntüsünden okuma yap.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.3,
        help="Sürekli okumada iki ölçüm arası bekleme süresi (saniye).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Belirtilirse bu sayıda ölçümden sonra çık.",
    )
    parser.add_argument(
        "--tesseract",
        type=Path,
        help="Tesseract yürütülebilir dosyasının yolu.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Tek seferlik okuma yapıp çık.",
    )
    return parser


def resolve_anchor(args: argparse.Namespace, anchor_file: Path) -> Tuple[int, int]:
    anchor_settings: Optional[AnchorSettings] = None
    if anchor_file.exists():
        try:
            anchor_settings = load_anchor_file(anchor_file)
        except Exception:
            anchor_settings = None

    if (args.anchor_x is not None) or (args.anchor_y is not None):
        if args.anchor_x is None or args.anchor_y is None:
            raise RuntimeError("--anchor-x ve --anchor-y birlikte belirtilmelidir.")
        anchor_settings = AnchorSettings(center_x=int(args.anchor_x), center_y=int(args.anchor_y))

    if args.calibrate_anchor:
        anchor_settings = select_anchor(anchor_settings)
        save_anchor_file(anchor_file, anchor_settings)
        print(f"Anchor seçildi: X={anchor_settings.center_x}, Y={anchor_settings.center_y}")

    if anchor_settings is None:
        raise RuntimeError(
            f"Anchor belirlenmedi. '--calibrate-anchor' parametresini kullanın "
            f"veya '{anchor_file}' dosyasını oluşturun."
        )
    return anchor_settings.as_tuple()


def main() -> None:
    parser = build_common_parser("Parti HP okuyucu (anchor ve ROI kalibrasyonlu)")
    parser.add_argument(
        "--slots",
        type=int,
        default=8,
        help=f"Okunacak parti üyesi sayısı (varsayılan {MAX_SLOTS}).",
    )
    parser.add_argument(
        "--full-screen",
        action="store_true",
        help="Anchor kalibrasyonu yapıldıysa etkisiz; uyumluluk için mevcut.",
    )
    parser.add_argument(
        "--debug-image",
        type=Path,
        help="Algılanan ROI'leri çizerek görüntüyü belirtilen dosyaya kaydet.",
    )
    parser.add_argument(
        "--calibrate-anchor",
        action="store_true",
        help="Sadece anchor'ı yeniden seçip kaydet.",
    )
    parser.add_argument(
        "--calibrate-roi",
        action="store_true",
        help="Anchor ve ilk iki slotu seçerek ROI kalibrasyonu yap.",
    )
    parser.add_argument(
        "--anchor-x",
        type=int,
        help="X butonu anchor X koordinatı (piksel).",
    )
    parser.add_argument(
        "--anchor-y",
        type=int,
        help="X butonu anchor Y koordinatı (piksel).",
    )
    parser.add_argument(
        "--anchor-file",
        type=Path,
        help=f"Anchor koordinatlarını saklamak için JSON dosyası (varsayılan: {DEFAULT_ANCHOR_FILE}).",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        help=f"ROI kalibrasyonunun saklanacağı JSON dosyası (varsayılan: {DEFAULT_CALIBRATION_FILE}).",
    )
    args = parser.parse_args()

    anchor_file = args.anchor_file or DEFAULT_ANCHOR_FILE
    calibration_file = args.calibration_file or DEFAULT_CALIBRATION_FILE
    slots = max(MIN_SLOTS, min(args.slots, MAX_SLOTS))

    if args.calibrate_roi:
        try:
            calibrate_rois(anchor_file, calibration_file, slots)
        except Exception as exc:  # pragma: no cover - kullanıcı etkileşim hataları
            print(f"Hata: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    try:
        calibration = load_calibration(calibration_file)
    except FileNotFoundError:
        print(
            f"Kalibrasyon dosyası bulunamadı: {calibration_file}. Önce '--calibrate-roi' çalıştırın.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Kalibrasyon dosyası okunamadı: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        configure_tesseract(args.tesseract)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Tesseract yapılandırılamadı: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        anchor = resolve_anchor(args, anchor_file)
    except Exception as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.from_image:
        try:
            values, rois, frame = read_hp_from_image(args.from_image, slots, anchor, calibration)
            for idx, result in enumerate(values, start=1):
                print(format_output(idx, result))
            if args.debug_image:
                annotated = draw_rois(frame, rois)
                cv2.imwrite(str(args.debug_image), annotated)
                print(f"Debug görseli kaydedildi: {args.debug_image}")
        except Exception as exc:
            print(f"Hata: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.once:
        try:
            frame = grab_screen()
            values, rois = read_hp_from_frame(frame, slots, anchor, calibration)
            for idx, result in enumerate(values, start=1):
                print(format_output(idx, result))
            if args.debug_image:
                annotated = draw_rois(frame, rois)
                cv2.imwrite(str(args.debug_image), annotated)
                print(f"Debug görseli kaydedildi: {args.debug_image}")
        except Exception as exc:
            print(f"Hata: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.full_screen:
        print("Uyarı: '--full-screen' anchor/ROI modunda gereksizdir ve yok sayılıyor.", file=sys.stderr)

    interval = max(args.interval, 0.2)
    limit = args.limit or sys.maxsize

    try:
        rois = compute_slot_rois(anchor, calibration, slots)
        frame = grab_screen()
        ensure_rois_within_frame(frame, rois)
        if args.debug_image:
            annotated = draw_rois(frame, rois)
            cv2.imwrite(str(args.debug_image), annotated)
            print(f"Debug görseli kaydedildi: {args.debug_image}")

        for _ in range(limit):
            outputs = []
            for slot in rois:
                try:
                    current, maximum = read_bar_from_screen(slot.roi)
                    outputs.append(format_output(slot.index, (current, maximum)))
                except Exception as exc:
                    outputs.append(format_output(slot.index, None))
                    print(f"Slot {slot.index} okunamadı: {exc}", file=sys.stderr)
            print(" | ".join(outputs), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
