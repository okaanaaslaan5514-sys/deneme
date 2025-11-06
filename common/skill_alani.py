#!/usr/bin/env python3
"""Skill ikonlarının bulunduğu alanı tanımlamak için ROI seçme aracı."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mss kütüphanesi bulunamadı. Önce `pip install mss` çalıştırın."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "skill_alani.json"
MIN_WIDTH = 40
DEFAULT_HEIGHT = 48
HANDLE_SIZE = 12


@dataclass
class SkillArea:
    right: int
    top: int
    width: int
    height: int

    @property
    def left(self) -> int:
        return self.right - self.width

    def clamp(self, frame_width: int, frame_height: int) -> None:
        self.width = max(MIN_WIDTH, min(self.width, frame_width))
        self.right = min(max(self.right, self.width), frame_width)
        if self.right - self.width < 0:
            self.width = self.right
        self.top = min(max(self.top, 0), max(0, frame_height - self.height))

    def as_dict(self) -> dict:
        return {
            "right": int(self.right),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


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


def load_config(path: Path) -> SkillArea:
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Konfigürasyon dosyası bozuk: {path} ({exc})")
        return SkillArea(
            right=int(data.get("right", 400)),
            top=int(data.get("top", 600)),
            width=int(data.get("width", 240)),
            height=int(data.get("height", DEFAULT_HEIGHT)),
        )
    return SkillArea(right=400, top=600, width=240, height=DEFAULT_HEIGHT)


def save_config(path: Path, area: SkillArea) -> None:
    path.write_text(json.dumps(area.as_dict(), indent=2))


# ---------------------------------------------------------------------------
# Etkileşimli seçim
# ---------------------------------------------------------------------------

def interactive_select(
    base_frame: np.ndarray,
    area: SkillArea,
    allow_resize: bool,
    from_image: Optional[Path],
) -> Optional[SkillArea]:
    current_frame = prepare_display(base_frame)
    area.clamp(current_frame.shape[1], current_frame.shape[0])

    window_name = "Skill Alanı Seçici"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        min(current_frame.shape[1], 1600),
        min(current_frame.shape[0], 900),
    )

    preview_mode = 0  # 0 gizli, 1 blur, 2 normal
    blurred_frame: Optional[np.ndarray] = None
    dragging = {"mode": None, "offset_x": 0, "offset_y": 0}

    instructions = ["Ok tuşları: alanı yukarı/aşağı kaydır"]
    if allow_resize:
        instructions.append("A: sol kenarı sola genişlet (Shift = 5 px)")
        instructions.append("D: sol kenarı sağa daralt (Shift = 5 px)")
        instructions.append("Sol kenarı mouse ile sürükleyebilirsin")
    else:
        instructions.append("Sol kenar sabit, sağdaki tutamaktan alanı taşı")
    instructions.extend(
        [
            "Sağ tutamak (mouse): alanı sürükle",
            "P: görünümü değiştir (gizli/blur/normal)",
            "R: ekran görüntüsünü yenile",
            "Enter: kaydet | ESC: iptal",
        ]
    )

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal area, current_frame
        left = area.left
        right = area.right
        top = area.top
        bottom = area.top + area.height

        if event == cv2.EVENT_LBUTTONDOWN:
            if allow_resize and left - HANDLE_SIZE <= x <= left + HANDLE_SIZE and top <= y <= bottom:
                dragging["mode"] = "resize_left"
            elif right - HANDLE_SIZE <= x <= right + HANDLE_SIZE and top <= y <= bottom:
                dragging["mode"] = "move"
                dragging["offset_x"] = x - right
                dragging["offset_y"] = y - top
            else:
                dragging["mode"] = None
        elif event == cv2.EVENT_MOUSEMOVE and dragging["mode"]:
            if dragging["mode"] == "resize_left" and allow_resize:
                new_left = min(x, right - MIN_WIDTH)
                new_left = max(0, new_left)
                area.width = right - new_left
                area.clamp(current_frame.shape[1], current_frame.shape[0])
            elif dragging["mode"] == "move":
                new_top = y - dragging["offset_y"]
                new_right = x - dragging["offset_x"]
                area.top = new_top
                area.right = new_right
                area.clamp(current_frame.shape[1], current_frame.shape[0])
        elif event == cv2.EVENT_LBUTTONUP:
            dragging["mode"] = None

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

            left = area.left
            right = area.right
            top = area.top
            bottom = area.top + area.height

            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 2)
            if allow_resize:
                cv2.rectangle(
                    overlay,
                    (left - HANDLE_SIZE, top + area.height // 2 - HANDLE_SIZE // 2),
                    (left + HANDLE_SIZE, top + area.height // 2 + HANDLE_SIZE // 2),
                    (0, 255, 255),
                    cv2.FILLED,
                )
            # Right move handle
            cv2.rectangle(
                overlay,
                (right - HANDLE_SIZE, top + area.height // 2 - HANDLE_SIZE // 2),
                (right + HANDLE_SIZE, top + area.height // 2 + HANDLE_SIZE // 2),
                (255, 128, 0),
                cv2.FILLED,
            )
            cv2.putText(
                overlay,
                f"Alan: left={left}, right={right}, top={top}, width={area.width}, height={area.height}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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
                    0.5,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 22

            cv2.imshow(window_name, overlay)
            key = cv2.waitKeyEx(70)
            if key == -1:
                continue

            if key in (27,):  # ESC
                return None
            if key in (13, 10, 141, 65293):
                return area

            if key in (ord("p"), ord("P")):
                preview_mode = (preview_mode + 1) % 3
                continue
            if key in (ord("r"), ord("R")):
                current_frame = prepare_display(load_frame(from_image))
                blurred_frame = None
                area.clamp(current_frame.shape[1], current_frame.shape[0])
                cv2.resizeWindow(
                    window_name,
                    min(current_frame.shape[1], 1600),
                    min(current_frame.shape[0], 900),
                )
                continue

            if key in (2490368, 65362, 82):  # Up
                area.top -= 1
                area.clamp(current_frame.shape[1], current_frame.shape[0])
            elif key in (2621440, 65364, 84):  # Down
                area.top += 1
                area.clamp(current_frame.shape[1], current_frame.shape[0])
            elif allow_resize and key in (ord("a"), ord("A")):
                delta = 5 if key == ord("A") else 1
                area.width = min(area.width + delta, area.right)
                area.clamp(current_frame.shape[1], current_frame.shape[0])
            elif allow_resize and key in (ord("d"), ord("D")):
                delta = 5 if key == ord("D") else 1
                area.width = max(MIN_WIDTH, area.width - delta)
                area.clamp(current_frame.shape[1], current_frame.shape[0])
    finally:
        cv2.destroyWindow(window_name)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skill alanı ROI seçme aracı")
    parser.add_argument(
        "--from-image",
        type=Path,
        help="Ekran yerine belirtilen görüntü üzerinden çalış.",
    )
    parser.add_argument(
        "--set-width",
        action="store_true",
        help="Sol kenarı klavye/drag ile yeniden ayarla (sağ kenar sabit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_frame = load_frame(args.from_image)
    area = load_config(CONFIG_PATH)
    area.clamp(base_frame.shape[1], base_frame.shape[0])

    result = interactive_select(base_frame, area, args.set_width, args.from_image)
    if result is None:
        print("İptal edildi, değişiklik yapılmadı.")
        return

    save_config(CONFIG_PATH, result)
    print(
        "Skill alanı kaydedildi: "
        f"left={result.left}, right={result.right}, top={result.top}, "
        f"width={result.width}, height={result.height}"
        f" | dosya: {CONFIG_PATH}"
    )


if __name__ == "__main__":
    main()
