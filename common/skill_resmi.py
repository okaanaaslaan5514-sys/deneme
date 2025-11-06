#!/usr/bin/env python3
"""Ekrandan skill ikonuna ait ROI seçip boyutu kaydetme veya görüntü alma aracı."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mss kütüphanesi bulunamadı. Önce `pip install mss` çalıştırın."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DEFAULT = BASE_DIR / "skill_resmi.json"
OUTPUT_DIR_DEFAULT = BASE_DIR / "skills"


@dataclass
class ROI:
    left: int
    top: int
    width: int
    height: int

    def clamp(self, frame_width: int, frame_height: int) -> None:
        self.width = max(10, min(self.width, frame_width))
        self.height = max(10, min(self.height, frame_height))
        max_left = max(0, frame_width - self.width)
        max_top = max(0, frame_height - self.height)
        self.left = min(max(self.left, 0), max_left)
        self.top = min(max(self.top, 0), max_top)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.width, self.height


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def grab_screen() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        frame = sct.grab(monitor)
    return np.array(frame)


def prepare_display(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame.copy()


def load_frame(from_image: Optional[Path]) -> np.ndarray:
    if from_image:
        frame = cv2.imread(str(from_image), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise SystemExit(f"Görsel açılamadı: {from_image}")
        return frame
    return grab_screen()


def load_config(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Konfigürasyon dosyası bozuk: {path} ({exc})")
    else:
        data = {}
    data.setdefault("width", 120)
    data.setdefault("height", 120)
    return data


def save_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def compute_initial_roi(frame: np.ndarray, width: int, height: int) -> ROI:
    h, w = frame.shape[:2]
    roi = ROI(left=(w - width) // 2, top=(h - height) // 2, width=width, height=height)
    roi.clamp(w, h)
    return roi


# ---------------------------------------------------------------------------
# Etkileşimli seçim
# ---------------------------------------------------------------------------

def interactive_select(
    base_frame: np.ndarray,
    initial_roi: ROI,
    allow_resize: bool,
    from_image: Optional[Path],
) -> Optional[ROI]:
    current_frame = prepare_display(base_frame)
    roi = ROI(initial_roi.left, initial_roi.top, initial_roi.width, initial_roi.height)
    roi.clamp(current_frame.shape[1], current_frame.shape[0])

    window_name = "Skill ROI Seçici"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        min(current_frame.shape[1], 1600),
        min(current_frame.shape[0], 900),
    )

    preview_mode = 0  # 0 gizli, 1 blur, 2 normal
    blurred_frame: Optional[np.ndarray] = None

    instructions = ["Ok tuşları: ROI'yi taşı", "Mouse: sol tuşla sürükle"]
    if allow_resize:
        instructions.append("W/S: yükseklik azalt/artır (Shift = 5 px)")
        instructions.append("A/D: genişlik azalt/artır (Shift = 5 px)")
    instructions.extend(
        [
            "R: ekran görüntüsünü yenile",
            "P: görünümü değiştir (gizli/blur/normal)",
            "Enter: kaydet | ESC: iptal",
        ]
    )

    dragging = {"active": False, "dx": 0, "dy": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal dragging, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            if roi.left <= x <= roi.left + roi.width and roi.top <= y <= roi.top + roi.height:
                dragging["active"] = True
                dragging["dx"] = x - roi.left
                dragging["dy"] = y - roi.top
        elif event == cv2.EVENT_MOUSEMOVE and dragging["active"]:
            roi.left = x - dragging["dx"]
            roi.top = y - dragging["dy"]
            roi.clamp(current_frame.shape[1], current_frame.shape[0])
        elif event == cv2.EVENT_LBUTTONUP:
            dragging["active"] = False

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            if preview_mode == 0:
                overlay = np.zeros_like(current_frame)
            elif preview_mode == 1:
                if blurred_frame is None:
                    blurred_frame = cv2.GaussianBlur(current_frame, (51, 51), 0)
                overlay = blurred_frame.copy()
            else:
                overlay = current_frame.copy()

            x, y, w, h = roi.as_tuple()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"ROI: x={x}, y={y}, w={w}, h={h}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            y_offset = 52
            for line in instructions:
                cv2.putText(
                    overlay,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                y_offset += 24

            cv2.imshow(window_name, overlay)
            key = cv2.waitKeyEx(70)
            if key == -1:
                continue

            if key in (27,):  # ESC
                return None
            if key in (13, 10, 141, 65293):  # Enter
                return roi

            if key in (ord("r"), ord("R")):
                current_frame = prepare_display(load_frame(from_image))
                blurred_frame = None
                roi.clamp(current_frame.shape[1], current_frame.shape[0])
                cv2.resizeWindow(
                    window_name,
                    min(current_frame.shape[1], 1600),
                    min(current_frame.shape[0], 900),
                )
                continue

            if key in (ord("p"), ord("P")):
                preview_mode = (preview_mode + 1) % 3
                continue

            if key in (2424832, 65361, 81):  # Left
                roi.left -= 1
            elif key in (2555904, 65363, 83):  # Right
                roi.left += 1
            elif key in (2490368, 65362, 82):  # Up
                roi.top -= 1
            elif key in (2621440, 65364, 84):  # Down
                roi.top += 1
            elif allow_resize and key in (ord("w"), ord("W")):
                step = 5 if key == ord("W") else 1
                roi.height = max(10, roi.height - step)
            elif allow_resize and key in (ord("s"), ord("S")):
                step = 5 if key == ord("S") else 1
                roi.height += step
            elif allow_resize and key in (ord("a"), ord("A")):
                step = 5 if key == ord("A") else 1
                roi.width = max(10, roi.width - step)
            elif allow_resize and key in (ord("d"), ord("D")):
                step = 5 if key == ord("D") else 1
                roi.width += step
            else:
                continue

            roi.clamp(current_frame.shape[1], current_frame.shape[0])
    finally:
        cv2.destroyWindow(window_name)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skill ROI seçme ve kayıt aracı")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DEFAULT,
        help=f"ROI boyutu ayarlarının tutulacağı JSON (varsayılan: {CONFIG_DEFAULT}).",
    )
    parser.add_argument(
        "--from-image",
        type=Path,
        help="Ekran yerine belirtilen görüntü üzerinde çalış.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Kaydedilen skill görsellerinin saklanacağı klasör (varsayılan: {OUTPUT_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--set-size",
        action="store_true",
        help="ROI boyutunu W/A/S/D ile güncelle (konum kaydedilmez).",
    )
    parser.add_argument(
        "--initial-width",
        type=int,
        default=120,
        help="Konfigürasyon yoksa kullanılacak başlangıç genişliği.",
    )
    parser.add_argument(
        "--initial-height",
        type=int,
        default=120,
        help="Konfigürasyon yoksa kullanılacak başlangıç yüksekliği.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    if not args.config.exists():
        config["width"] = args.initial_width
        config["height"] = args.initial_height

    base_frame = load_frame(args.from_image)
    initial_roi = compute_initial_roi(base_frame, int(config["width"]), int(config["height"]))

    result = interactive_select(base_frame, initial_roi, args.set_size, args.from_image)
    if result is None:
        print("İptal edildi, değişiklik yapılmadı.")
        return

    left, top, width, height = result.as_tuple()

    if args.set_size:
        config["width"] = width
        config["height"] = height
        save_config(args.config, config)
        print(f"ROI boyutu kaydedildi: width={width}, height={height} (dosya: {args.config})")
        return

    # Normal mod: ROI'yi kaydır, Enter ile görsel kaydet.
    capture_frame = load_frame(args.from_image)
    h, w = capture_frame.shape[:2]
    result.clamp(w, h)
    left, top, width, height = result.as_tuple()

    crop = capture_frame[top : top + height, left : left + width]
    if crop.size == 0:
        print("Uyarı: ROI boş, çıktı alınmadı.", file=sys.stderr)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"skill_{timestamp}_{int(time.time() * 1000) % 1000:03d}.png"
    out_path = args.output_dir / filename
    cv2.imwrite(str(out_path), crop)

    print(
        f"Skill görüntüsü kaydedildi: {out_path}"
        f" | ROI (left={left}, top={top}, width={width}, height={height})"
    )


if __name__ == "__main__":
    main()
