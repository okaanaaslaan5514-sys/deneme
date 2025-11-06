#!/usr/bin/env python3
"""
Utility helpers to read HP/MP style bars from the screen or images using OCR.
"""
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract

try:
    import mss
except ImportError as exc:  # pragma: no cover - import guard for missing dependency
    raise SystemExit(
        "mss kütüphanesi bulunamadı. Önce `pip install mss` çalıştırın."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent

TESSERACT_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/"
TESSERACT_CANDIDATES = (
    Path(r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"),
    Path(r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"),
    Path("/mnt/c/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("/mnt/c/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    BASE_DIR / "tesseract" / "tesseract.exe",
)


@dataclass(frozen=True)
class ROI:
    left: int
    top: int
    width: int
    height: int


def configure_tesseract(manual_path: Optional[Path]) -> None:
    """
    Ensure pytesseract knows where the binary is.

    If `tesseract` is already on PATH we do nothing. Otherwise try the provided path.
    """
    if manual_path:
        pytesseract.pytesseract.tesseract_cmd = str(manual_path)
        return
    if shutil.which("tesseract"):
        return
    for candidate in TESSERACT_CANDIDATES:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return
    raise SystemExit(
        "Tesseract yürütülebilir dosyası bulunamadı. "
        "Tesseract kurulumunu doğrulayın veya '--tesseract PATH' parametresi sağlayın."
    )


def capture_roi(roi: ROI) -> np.ndarray:
    """Grab the ROI from the main monitor and return it as a BGRA numpy array."""
    with mss.mss() as sct:
        frame = sct.grab(asdict(roi))
    return np.array(frame)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame to grayscale and apply binary thresholding so the white digits stand out.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    # Slight blur to reduce aliasing artefacts.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresh


BAR_PATTERN = re.compile(r"(?P<current>\d+)\s*/\s*(?P<maximum>\d+)")


def parse_bar_text(text: str) -> Tuple[int, int]:
    """Extract current and maximum values from OCR output."""
    cleaned = text.strip()
    match = BAR_PATTERN.search(cleaned)
    if not match:
        raise ValueError(f"Barkodu çözümlenemedi: {text!r}")
    current = int(match.group("current"))
    maximum = int(match.group("maximum"))
    return current, maximum


def read_bar_from_frame(frame: np.ndarray) -> Tuple[int, int]:
    """Run OCR on a frame to obtain current and maximum values."""
    processed = preprocess(frame)
    text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
    return parse_bar_text(text)


def read_bar_from_screen(roi: ROI) -> Tuple[int, int]:
    """Capture and decode the bar from the screen."""
    frame = capture_roi(roi)
    return read_bar_from_frame(frame)


def read_bar_from_image(image_path: Path) -> Tuple[int, int]:
    """Read a saved ROI image from disk and decode it."""
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"Görsel açılamadı: {image_path}")
    return read_bar_from_frame(frame)


def compute_percentage(current: int, maximum: int) -> float:
    """Return current/max as a percentage (0-100)."""
    if maximum <= 0:
        return 0.0
    return max(0.0, min(100.0, (current / maximum) * 100.0))


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--from-image",
        type=Path,
        help="Kaydedilmiş ROI görüntüsünden okuma yap.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.3,
        help="Sürekli okumada iki ölçüm arası bekleme (saniye).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Belirli sayıda ölçümden sonra çık.",
    )
    parser.add_argument(
        "--tesseract",
        type=Path,
        help="Tesseract yürütülebilir dosyasının yolu.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Tek seferlik okuma yap ve çık.",
    )
    return parser


__all__ = [
    "ROI",
    "configure_tesseract",
    "read_bar_from_screen",
    "read_bar_from_image",
    "compute_percentage",
    "build_common_parser",
]
