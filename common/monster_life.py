#!/usr/bin/env python3
"""
Canavar HP çubuğunu belirtilen ROI üzerinden okuyup yüzdelik değerini raporlar.

Turkuaza çalan yazıları yakalayabilmek için HSV tabanlı maske + büyütme ile
özelleştirilmiş bir ön işleme uygular. ROI bilgisi JSON dosyasından veya komut
satırı argümanlarından sağlanabilir; notice çıktığında çubuğun aşağı kayması için
bire çok Y kayması denenir.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

try:
    import cv2  # type: ignore[import]
    import numpy as np  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Gerekli bağımlılıklar eksik. `pip install opencv-python numpy` komutunu çalıştırın."
    ) from exc

try:
    import mss  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("mss modülü bulunamadı. `pip install mss` komutunu çalıştırın.") from exc

import pytesseract  # type: ignore[import]

try:
    from .bar_reader import (  # type: ignore[import]
        ROI,
        build_common_parser,
        compute_percentage,
        configure_tesseract,
    )
except ImportError:  # pragma: no cover - module as script fallback
    from bar_reader import (  # type: ignore[import]  # noqa: F401
        ROI,
        build_common_parser,
        compute_percentage,
        configure_tesseract,
    )

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "monster_life_roi.json"
# Varsayılan ROI; gerçek değerleri kalibrasyon sonrası güncelleyin.
DEFAULT_MONSTER_ROI = ROI(left=0, top=0, width=150, height=30)
DEFAULT_Y_SHIFTS: Tuple[int, ...] = (0,)
TESSERACT_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/"
BAR_PATTERN = re.compile(r"(?P<current>\d+)\s*/\s*(?P<maximum>\d+)")


@dataclass(frozen=True)
class MonsterLifeConfig:
    roi: ROI
    y_shifts: Tuple[int, ...] = DEFAULT_Y_SHIFTS


def _normalize_shifts(values: Sequence[int]) -> Tuple[int, ...]:
    """Kayma değerlerini benzersiz ve sıralı olacak şekilde normalize et."""
    normalized = {int(v) for v in values}
    normalized.add(0)
    shifts = tuple(sorted(normalized))
    return shifts


def load_roi_from_config(path: Path) -> Optional[MonsterLifeConfig]:
    """JSON dosyasından ROI bilgisi yükle."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if all(key in data for key in ("left", "top", "width", "height")):
            base_data = data
            shift_raw = data.get("y_shifts")
        elif "base" in data and all(key in data["base"] for key in ("left", "top", "width", "height")):
            base_data = data["base"]
            shift_raw = data.get("y_shifts", base_data.get("y_shifts"))
        else:
            raise KeyError("Beklenen ROI anahtarları bulunamadı.")

        roi = ROI(
            left=int(base_data["left"]),
            top=int(base_data["top"]),
            width=int(base_data["width"]),
            height=int(base_data["height"]),
        )
        shifts: Tuple[int, ...]
        if shift_raw is None:
            shifts = DEFAULT_Y_SHIFTS
        elif isinstance(shift_raw, (list, tuple)):
            shifts = _normalize_shifts(shift_raw)
        else:
            shifts = _normalize_shifts([int(shift_raw)])
        return MonsterLifeConfig(roi=roi, y_shifts=shifts)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"ROI konfigürasyonu okunamadı: {path} ({exc})") from exc


def save_roi_to_config(path: Path, config: MonsterLifeConfig) -> None:
    """ROI + kayma bilgilerini JSON dosyasına kaydet."""
    payload = {
        "base": {
            "left": config.roi.left,
            "top": config.roi.top,
            "width": config.roi.width,
            "height": config.roi.height,
        }
    }
    if config.y_shifts != DEFAULT_Y_SHIFTS:
        payload["y_shifts"] = list(config.y_shifts)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_roi_values(values: Sequence[int]) -> ROI:
    """Komut satırından gelen dört değeri ROI nesnesine çevir."""
    if len(values) != 4:
        raise ValueError("ROI için dört değer gerekli: left top width height")
    left, top, width, height = (int(v) for v in values)
    if width <= 0 or height <= 0:
        raise ValueError("ROI genişlik ve yükseklik değerleri pozitif olmalıdır.")
    return ROI(left=left, top=top, width=width, height=height)


def format_output(current: int, maximum: int) -> str:
    percentage = compute_percentage(current, maximum)
    return f"Monster HP: {current}/{maximum} ({percentage:.1f}%)"


def capture_shifted_roi(config: MonsterLifeConfig, shift: int) -> np.ndarray:
    """Belirtilen kayma ile ROI'yi ekrandan yakala."""
    region = {
        "left": int(config.roi.left),
        "top": int(config.roi.top + shift),
        "width": int(config.roi.width),
        "height": int(config.roi.height),
    }
    with mss.mss() as sct:
        frame = sct.grab(region)
    return np.array(frame)


def preprocess_monster_frame(frame: np.ndarray) -> np.ndarray:
    """Turkuaz/beyaz tonları öne çıkarıp OCR için ikili hale getir."""
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        bgr = frame.copy()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    turquoise_mask = cv2.inRange(
        hsv,
        np.array([70, 40, 110], dtype=np.uint8),
        np.array([110, 255, 255], dtype=np.uint8),
    )
    bright_mask = cv2.inRange(hsv[:, :, 2], 200, 255)
    combined = cv2.bitwise_or(turquoise_mask, bright_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.dilate(combined, kernel, iterations=1)

    masked = cv2.bitwise_and(bgr, bgr, mask=combined)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    if not np.any(gray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresh


def parse_bar_text(text: str) -> Tuple[int, int]:
    cleaned = text.strip()
    match = BAR_PATTERN.search(cleaned)
    if not match:
        raise ValueError(f"Barkodu çözümlenemedi: {text!r}")
    return int(match.group("current")), int(match.group("maximum"))


def ocr_monster_frame(frame: np.ndarray) -> Tuple[int, int]:
    processed = preprocess_monster_frame(frame)
    text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
    return parse_bar_text(text)


def read_once(config: MonsterLifeConfig, image: Optional[Path]) -> str:
    if image:
        frame = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise FileNotFoundError(f"Görsel okunamadı: {image}")
        current, maximum = ocr_monster_frame(frame)
        return format_output(current, maximum)

    last_exc: Optional[Exception] = None
    for shift in config.y_shifts:
        try:
            frame = capture_shifted_roi(config, shift)
            current, maximum = ocr_monster_frame(frame)
            return format_output(current, maximum)
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            continue
    raise last_exc or RuntimeError("Canavar HP ROI'lerinden değer okunamadı.")
    return format_output(current, maximum)


def main() -> None:
    parser = build_common_parser("Canavar HP bar OCR okuyucusu")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="ROI değerlerini manuel olarak ver.",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="ROI değerlerinin okunacağı/kaydedileceği JSON dosyası.",
    )
    parser.add_argument(
        "--save-roi",
        action="store_true",
        help="--roi ile verilen değerleri yapılandırma dosyasına kaydet.",
    )
    parser.add_argument(
        "--y-shifts",
        type=int,
        nargs="+",
        metavar="SHIFT",
        help="Base ROI'ye eklenecek ek piksel kaymaları (ör. 0 28). 0 otomatik eklenir.",
    )
    args = parser.parse_args()

    configure_tesseract(args.tesseract)

    config = load_roi_from_config(args.roi_config) or MonsterLifeConfig(
        roi=DEFAULT_MONSTER_ROI,
        y_shifts=DEFAULT_Y_SHIFTS,
    )

    if args.roi:
        try:
            new_roi = parse_roi_values(args.roi)
        except ValueError as exc:
            print(f"ROI geçersiz: {exc}", file=sys.stderr)
            sys.exit(1)
        config = MonsterLifeConfig(roi=new_roi, y_shifts=config.y_shifts)

    if args.y_shifts:
        shifts = _normalize_shifts(args.y_shifts)
        config = MonsterLifeConfig(roi=config.roi, y_shifts=shifts)

    if args.save_roi:
        try:
            save_roi_to_config(args.roi_config, config)
        except OSError as exc:
            print(f"ROI kaydedilemedi: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.once or args.from_image:
        try:
            print(read_once(config, args.from_image))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Canavar HP okunamadı: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    interval = max(args.interval, 0.1)
    limit = args.limit or sys.maxsize

    try:
        for _ in range(limit):
            try:
                print(read_once(config, args.from_image), flush=True)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Canavar HP okunamadı: {exc}", file=sys.stderr, flush=True)
            time.sleep(interval)
            if args.from_image:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
