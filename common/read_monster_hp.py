#!/usr/bin/env python3
"""
Canavar HP çubuğunu ekrandan veya kaydedilmiş bir ROI görselinden okur.

`read_hp.py` ile aynı komut satırı arayüzünü kullanır; tek fark, rakamlar
turkuaz tonda olduğu için HSV tabanlı özel bir ön işleme uygulanmasıdır.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2  # type: ignore[import]
    import numpy as np  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "OpenCV ve NumPy modülleri bulunamadı. `pip install opencv-python numpy` çalıştırın."
    ) from exc

try:
    import mss  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("mss modülü eksik. `pip install mss` komutunu çalıştırın.") from exc

import pytesseract  # type: ignore[import]

from bar_reader import (  # type: ignore[import]
    ROI,
    build_common_parser,
    compute_percentage,
    configure_tesseract,
)

MONSTER_HP_ROI = ROI(left=430, top=66, width=124, height=13)
MONSTER_HP_SHIFT_SEQUENCE = (0, 0, 24, 24, 0, 0)  # notice sırasında bar aşağı kayıyor (yaklaşık 24 px)
LAST_VALUES: tuple[int, int] | None = None
TESSERACT_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/"
BAR_PATTERN = re.compile(r"(?P<current>\d+)\s*/\s*(?P<maximum>\d+)")
_OCR_FIXES = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "Z": "2",
        "z": "2",
    }
)


def format_output(current: int, maximum: int) -> str:
    percentage = compute_percentage(current, maximum)
    return f"Monster HP: {current}/{maximum} ({percentage:.1f}%)"


def capture_roi(roi: ROI, shift: int = 0) -> np.ndarray:
    """Belirlenen ROI'yi (isteğe bağlı y kaymasıyla) ekrandan yakala ve BGRA numpy dizisi döndür."""
    top = max(0, roi.top + int(shift))
    region = {
        "left": int(roi.left),
        "top": int(top),
        "width": int(roi.width),
        "height": int(roi.height),
    }
    with mss.mss() as sct:
        frame = sct.grab(region)
    return np.array(frame)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Turkuaz tonlu rakamları beyaza, geri kalanını siyaha yaklaştırarak OCR için hazırla.
    """
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        bgr = frame.copy()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    turquoise_mask = cv2.inRange(
        hsv,
        np.array([72, 40, 120], dtype=np.uint8),
        np.array([112, 255, 255], dtype=np.uint8),
    )
    value_mask = cv2.inRange(hsv[:, :, 2], 200, 255)
    mask = cv2.bitwise_or(turquoise_mask, value_mask)

    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    if not np.any(gray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresh


def parse_bar_text(text: str) -> Tuple[int, int]:
    """
    OCR çıktısını ayrıştırarak mevcut / maksimum HP değerlerini döndür.
    Slash eksildiğinde veya birleşik yazıldığında heuristik düzeltmeler uygular.
    """
    sanitized = text.replace("", " ").translate(_OCR_FIXES).strip()
    match = BAR_PATTERN.search(sanitized)
    if match:
        current = int(match.group("current"))
        maximum = int(match.group("maximum"))
        if current <= maximum:
            return current, maximum
    tokens = [token for token in re.findall(r"\d+", sanitized) if token]
    if len(tokens) >= 2:
        current = int(tokens[0])
        maximum = int(tokens[1])
        if current <= maximum:
            return current, maximum
    if len(tokens) == 1:
        token = tokens[0]
        best_idx = None
        best_score = float("inf")
        half = len(token) / 2.0
        for idx in range(1, len(token)):
            left = int(token[:idx])
            right = int(token[idx:])
            if right == 0 or left > right:
                continue
            score = abs(idx - half)
            if score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            left = int(token[:best_idx])
            right = int(token[best_idx:])
            if left <= right:
                return left, right
    raise ValueError(f"Barkodu çözümlenemedi: {text!r}")


def _save_debug_images(
    frame: np.ndarray,
    processed: np.ndarray,
    debug_dir: Optional[Path],
    label: str,
) -> None:
    if not debug_dir:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    raw_path = debug_dir / f"{label}_{timestamp}.png"
    proc_path = debug_dir / f"{label}_{timestamp}_processed.png"
    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(proc_path), processed)


def read_bar_from_frame(
    frame: np.ndarray,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    label: str = "frame",
) -> Tuple[int, int]:
    global LAST_VALUES  # pylint: disable=global-variable-undefined
    processed = preprocess(frame)
    _save_debug_images(frame, processed, debug_dir, label)
    text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
    if debug:
        print(f"OCR ham çıktı: {text!r}", file=sys.stderr, flush=True)
    try:
        current, maximum = parse_bar_text(text)
        LAST_VALUES = (current, maximum)
        return current, maximum
    except ValueError as exc:
        if LAST_VALUES:
            if debug:
                print(
                    f"Heuristik geri dönüş: {LAST_VALUES} (neden: {exc})",
                    file=sys.stderr,
                    flush=True,
                )
            return LAST_VALUES
        raise


def read_bar_from_screen(
    roi: ROI,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    shift: int = 0,
) -> Tuple[int, int]:
    label = "screen" if shift == 0 else f"screen_shift{shift}"
    frame = capture_roi(roi, shift=shift)
    return read_bar_from_frame(frame, debug=debug, debug_dir=debug_dir, label=label)


def read_bar_from_image(
    image_path: Path,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> Tuple[int, int]:
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"Görsel açılamadı: {image_path}")
    label = image_path.stem
    return read_bar_from_frame(frame, debug=debug, debug_dir=debug_dir, label=label)


def read_once(
    image: Optional[Path],
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> str:
    if image:
        current, maximum = read_bar_from_image(image, debug=debug, debug_dir=debug_dir)
    else:
        last_exc: Optional[Exception] = None
        for idx, shift in enumerate(MONSTER_HP_SHIFT_SEQUENCE):
            try:
                current, maximum = read_bar_from_screen(
                    MONSTER_HP_ROI,
                    debug=debug,
                    debug_dir=debug_dir,
                    shift=shift,
                )
                break
            except Exception as exc:
                last_exc = exc
                continue
        else:
            raise last_exc or ValueError("Y ekseninde değer bulunamadı.")
    return format_output(current, maximum)


def main() -> None:
    parser = build_common_parser("Canavar HP bar OCR okuyucusu")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="OCR ham çıktılarını stderr'e yazdır.",
    )
    parser.add_argument(
        "--debug-save",
        type=Path,
        help="Ön işlemeyi disk'e kaydetmek için klasör.",
    )
    args = parser.parse_args()

    configure_tesseract(args.tesseract)

    if args.once or args.from_image:
        try:
            print(
                read_once(
                    args.from_image,
                    debug=args.debug,
                    debug_dir=args.debug_save,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Monster HP okunamadı: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    interval = max(args.interval, 0.1)
    limit = args.limit or sys.maxsize

    try:
        for _ in range(limit):
            try:
                print(
                    read_once(
                        args.from_image,
                        debug=args.debug,
                        debug_dir=args.debug_save,
                    ),
                    flush=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Monster HP okunamadı: {exc}", file=sys.stderr, flush=True)
            time.sleep(interval)
            if args.from_image:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
