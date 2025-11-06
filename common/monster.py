"""[ANTI AFK] başlığını şablon eşleştirme ile tespit eden araç."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

try:
    import cv2  # type: ignore[import]
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Gerekli bağımlılık eksik. `pip install opencv-python numpy` komutunu çalıştırın."
    ) from exc

# Merkezde incelenecek alan (ekranın %80'i)
ROI_FRACTION = 0.8  # global scale (x/y adjustments handled below)
# Şablon eşleşmesi için ölçekler
TEMPLATE_SCALES = (0.85, 0.9, 1.0, 1.1, 1.2)
# Kabul eşiği
MATCH_THRESHOLD = 0.55


def _resolve_template_path(explicit: Optional[Path]) -> Path:
    """Şablon dosyasını bul ve yolu döndür."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    module_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            module_dir / "antiafk.png",
            module_dir.parent / "ui" / "antiafk.png",
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "antiafk.png şablonu bulunamadı. `--template` argümanı ile yol belirtin."
    )


def _load_template(template_path: Path) -> np.ndarray:
    """Şablonu gri tonlamalı olarak yükle."""
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Şablon yüklenemedi: {template_path}")
    return template

def load_template(path: Optional[Path] = None) -> np.ndarray:
    """Varsayılan veya verilen yoldan şablon dön."""
    return _load_template(_resolve_template_path(path))

def detect_in_frame(frame_bgr: np.ndarray, template: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Bir karede `[ANTI AFK]` konumunu tespit et."""
    roi, offset = _center_roi(frame_bgr)
    bbox, _ = _match_template(roi, template)
    if not bbox:
        return None
    x_off, y_off = offset
    x, y, w, h = bbox
    return x + x_off, y + y_off, w, h


def _center_roi(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Görüntünün ortasında ROI kırp ve ofset döndür."""
    h, w = image.shape[:2]
    roi_w = int(w * ROI_FRACTION) - 10
    roi_h = int(h * ROI_FRACTION) + 20
    roi_w = max(min(roi_w, w), 1)
    roi_h = max(min(roi_h, h), 1)
    x1 = max((w - roi_w) // 2, 0)
    y1 = max((h - roi_h) // 2, 0)
    x2 = min(x1 + roi_w, w)
    y2 = min(y1 + roi_h, h)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def _match_template(
    roi: np.ndarray, template: np.ndarray
) -> Tuple[Optional[Tuple[int, int, int, int]], dict]:
    """ROI üzerinde şablon eşleştirme yapıp en iyi sonucu döndür."""
    debug: dict = {}
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    best_score = -1.0
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_scale = 1.0

    for scale in TEMPLATE_SCALES:
        scaled_w = int(template.shape[1] * scale)
        scaled_h = int(template.shape[0] * scale)
        if scaled_w < 5 or scaled_h < 5:
            continue
        resized_tpl = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        if resized_tpl.shape[0] >= roi_gray.shape[0] or resized_tpl.shape[1] >= roi_gray.shape[1]:
            continue
        result = cv2.matchTemplate(roi_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_scale = scale
            best_bbox = (max_loc[0], max_loc[1], resized_tpl.shape[1], resized_tpl.shape[0])

    debug["best_score"] = float(best_score)
    debug["best_scale"] = float(best_scale)
    debug["bbox"] = best_bbox

    if best_score >= MATCH_THRESHOLD and best_bbox is not None:
        return best_bbox, debug
    return None, debug


def analyze_image(
    image_path: Path,
    template: np.ndarray,
    debug_dir: Optional[Path] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Tek görselde `[ANTI AFK]` şablonunu arayıp koordinat döndür."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Görüntü yüklenemedi: {image_path}")

    roi, offset = _center_roi(image)
    bbox, debug_data = _match_template(roi, template)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "01_roi.png"), roi)
        if bbox:
            x, y, w, h = bbox
            annotated = roi.copy()
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / "02_bbox.png"), annotated)
        else:
            with (debug_dir / "info.txt").open("w", encoding="utf-8") as fh:
                fh.write(f"Şablon bulunamadı. Debug: {debug_data}\n")

    if not bbox:
        return None
    x_off, y_off = offset
    x, y, w, h = bbox
    return x + x_off, y + y_off, w, h


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="[ANTI AFK] şablon tespiti.")
    parser.add_argument("image", nargs="?", type=Path, help="Analiz edilecek görüntü")
    parser.add_argument("--template", type=Path, help="Şablon dosyası yolu (varsayılan otomatik bulunur).")
    parser.add_argument("--debug-dir", type=Path, help="Ara görüntüleri kaydetmek için klasör.")
    parser.add_argument("--live", action="store_true", help="Canlı ekran yakalama modu.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Canlı modda yakalamalar arası süre (saniye).",
    )
    return parser.parse_args(argv)


def _run_live_mode(template: np.ndarray, interval: float) -> int:
    try:
        import dxcam  # type: ignore[import]
    except ModuleNotFoundError:
        print("dxcam modülü bulunamadı. `pip install dxcam` komutu ile yükleyin.")
        return 1

    camera = dxcam.create(output_idx=0)
    if not camera:
        print("DX ekran yakalama başlatılamadı.")
        return 1

    print("Canlı mod başlatıldı. Çıkmak için Ctrl+C veya pencere içinden 'q'.")
    camera.start()
    roi_rect = None  # (x, y, w, h)
    move_step = 10
    resize_step = 10
    min_size = 40
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(interval)
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h_img, w_img = image.shape[:2]
            if roi_rect is None:
                roi_w = max(int(w_img * ROI_FRACTION) - 10, min_size)
                roi_h = max(int(h_img * ROI_FRACTION) + 20, min_size)
                roi_x = max((w_img - roi_w) // 2, 0)
                roi_y = max((h_img - roi_h) // 2, 0)
                roi_rect = [roi_x, roi_y, roi_w, roi_h]

            roi_x, roi_y, roi_w, roi_h = roi_rect
            roi_w = max(min(roi_w, w_img - 1), min_size)
            roi_h = max(min(roi_h, h_img - 1), min_size)
            roi_x = min(max(roi_x, 0), w_img - roi_w)
            roi_y = min(max(roi_y, 0), h_img - roi_h)
            roi_rect = [roi_x, roi_y, roi_w, roi_h]

            roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            bbox, debug_data = _match_template(roi, template)

            display = image.copy()
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(display, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 255, 0), 2)
                print(f"[ANTI AFK] -> x={roi_x + x}, y={roi_y + y}, w={w}, h={h}, skor={debug_data['best_score']:.2f}")
            else:
                cv2.rectangle(display, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 1)
                print(f"Tespit yok -> skor={debug_data['best_score']:.2f}")

            cv2.imshow("monster_live", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (81,):  # left arrow
                roi_x = max(roi_x - move_step, 0)
            elif key in (83,):  # right arrow
                roi_x = min(roi_x + move_step, w_img - roi_w)
            elif key in (82,):  # up arrow
                roi_y = max(roi_y - move_step, 0)
            elif key in (84,):  # down arrow
                roi_y = min(roi_y + move_step, h_img - roi_h)
            elif key == ord('a'):
                roi_w = max(roi_w - resize_step, min_size)
            elif key == ord('d'):
                roi_w = min(roi_w + resize_step, w_img)
            elif key == ord('w'):
                roi_h = min(roi_h + resize_step, h_img)
            elif key == ord('s'):
                roi_h = max(roi_h - resize_step, min_size)

            # Clamp again after size adjustments
            roi_w = max(min(roi_w, w_img), min_size)
            roi_h = max(min(roi_h, h_img), min_size)
            roi_x = min(max(roi_x, 0), w_img - roi_w)
            roi_y = min(max(roi_y, 0), h_img - roi_h)
            roi_rect = [roi_x, roi_y, roi_w, roi_h]

            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    template_path = _resolve_template_path(args.template)
    template = _load_template(template_path)

    if args.live:
        return _run_live_mode(template, args.interval)

    if args.image is None:
        print("Görüntü yolu belirtilmedi.")
        return 1

    bbox = analyze_image(args.image, template, debug_dir=args.debug_dir)
    if not bbox:
        print("[ANTI AFK] tespit edilemedi.")
        return 1

    x, y, w, h = bbox
    print(f"[ANTI AFK] bulundu -> x={x}, y={y}, w={w}, h={h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "load_template",
    "detect_in_frame",
    "analyze_image",
]
