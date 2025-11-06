#!/usr/bin/env python3
"""
Coordinate reader for the fixed ROI inside the game window.

Requirements
------------
- Python 3.8+
- pip install mss opencv-python pytesseract numpy pydirectinput
- System-wide Tesseract OCR binary (https://github.com/tesseract-ocr/tesseract)

Run `python read_coordinates.py --help` for usage details.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, Optional, Sequence, Tuple, Union
import threading

import cv2
import numpy as np
import pytesseract

try:
    import mss
except ImportError as exc:  # pragma: no cover - import guard for missing dependency
    raise SystemExit(
        "mss kütüphanesi bulunamadı. Önce `pip install mss` çalıştırın."
    ) from exc

try:
    import pydirectinput
except ImportError:
    pydirectinput = None

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ROI:
    left: int
    top: int
    width: int
    height: int


DEFAULT_ROI = ROI(left=143, top=108, width=103, height=16)
TESSERACT_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789,-"
TESSERACT_CANDIDATES = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path("/mnt/c/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("/mnt/c/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    BASE_DIR / "tesseract" / "tesseract.exe",
)


@dataclass(frozen=True)
class NavigationConfig:
    target: Tuple[int, int]
    tolerance: float = 5.0
    sample_duration: float = 0.18
    rotation_duration: float = 0.08
    min_rotation_duration: float = 0.05
    max_rotation_duration: float = 0.35
    align_degrees: float = 8.0
    max_move_duration: float = 0.5
    min_move_duration: float = 0.12
    move_gain: float = 0.6
    settle_delay: float = 0.12
    max_steps: int = 180
    invert_rotation: bool = False
    sample_interval: float = 0.08
    direction_refresh_steps: int = 5
    fine_tune_threshold: float = 3.5
    fine_tune_radius: float = 2.0
    fine_tune_duration: float = 0.12
    fine_tune_max_attempts: int = 4


KEY_RELEASE_DELAY = 0.05
MIN_FORWARD_DISTANCE = 1.0
DIAGONAL_COMBOS: Dict[str, Tuple[str, str]] = {
    "wa": ("w", "a"),
    "wd": ("w", "d"),
    "sa": ("s", "a"),
    "sd": ("s", "d"),
}
DIAGONAL_PAIRS = (("wa", "sd"), ("wd", "sa"))


@dataclass(frozen=True)
class CalibrationData:
    forward_speed: float
    backward_speed: float
    turn_left_deg_per_sec: float
    turn_right_deg_per_sec: float
    rotation_sign: int
    forward_sample_duration: float
    rotation_sample_duration: float
    settle_delay: float
    timestamp: str
    forward_vector: Tuple[float, float] = (0.0, 0.0)
    backward_vector: Tuple[float, float] = (0.0, 0.0)
    diagonal_vectors: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    diagonal_speeds: Dict[str, float] = field(default_factory=dict)
    diagonal_components: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    diagonal_sample_duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "forward_speed": self.forward_speed,
            "backward_speed": self.backward_speed,
            "turn_left_deg_per_sec": self.turn_left_deg_per_sec,
            "turn_right_deg_per_sec": self.turn_right_deg_per_sec,
            "rotation_sign": self.rotation_sign,
            "forward_sample_duration": self.forward_sample_duration,
            "rotation_sample_duration": self.rotation_sample_duration,
            "settle_delay": self.settle_delay,
            "timestamp": self.timestamp,
            "forward_vector": [float(self.forward_vector[0]), float(self.forward_vector[1])],
            "backward_vector": [float(self.backward_vector[0]), float(self.backward_vector[1])],
            "diagonal_vectors": {
                name: [float(vec[0]), float(vec[1])]
                for name, vec in self.diagonal_vectors.items()
            },
            "diagonal_speeds": {
                name: float(speed) for name, speed in self.diagonal_speeds.items()
            },
            "diagonal_components": {
                name: [float(comp[0]), float(comp[1])]
                for name, comp in self.diagonal_components.items()
            },
            "diagonal_sample_duration": self.diagonal_sample_duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationData":
        return cls(
            forward_speed=float(data["forward_speed"]),
            backward_speed=float(data["backward_speed"]),
            turn_left_deg_per_sec=float(data["turn_left_deg_per_sec"]),
            turn_right_deg_per_sec=float(data["turn_right_deg_per_sec"]),
            rotation_sign=int(data.get("rotation_sign", 1)),
            forward_sample_duration=float(data["forward_sample_duration"]),
            rotation_sample_duration=float(data["rotation_sample_duration"]),
            settle_delay=float(data.get("settle_delay", 0.12)),
            timestamp=str(data.get("timestamp", "")),
            forward_vector=(
                float(data.get("forward_vector", [0.0, 0.0])[0]),
                float(data.get("forward_vector", [0.0, 0.0])[1]),
            ),
            backward_vector=(
                float(data.get("backward_vector", [0.0, 0.0])[0]),
                float(data.get("backward_vector", [0.0, 0.0])[1]),
            ),
            diagonal_vectors={
                name: (float(vec[0]), float(vec[1]))
                for name, vec in data.get("diagonal_vectors", {}).items()
            },
            diagonal_speeds={
                name: float(speed)
                for name, speed in data.get("diagonal_speeds", {}).items()
            },
            diagonal_components={
                name: (float(comp[0]), float(comp[1]))
                for name, comp in data.get("diagonal_components", {}).items()
            },
            diagonal_sample_duration=float(data.get("diagonal_sample_duration", 0.0)),
        )


def configure_tesseract(manual_path: Optional[Path]) -> None:
    """
    Ensure pytesseract knows where the binary is.

    If `tesseract` is already on PATH we do nothing. Otherwise, we try manual
    input or common Windows install locations.
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
        "Kurulumu doğrulayın veya '--tesseract PATH' parametresi ile belirtin."
    )


def capture_roi(roi: ROI) -> np.ndarray:
    """Grab the ROI from the main monitor and return it as a BGRA numpy array."""
    with mss.mss() as sct:
        frame = sct.grab(asdict(roi))
    return np.array(frame)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame to grayscale and apply adaptive thresholding.

    The ROI only contains the white digits and a dark background, so global
    Otsu thresholding is enough to isolate the characters.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    # Light blur suppresses rendering noise without sacrificing sharp edges.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresh


def parse_coordinate(text: str) -> Tuple[int, int]:
    """Parse the OCR output into integer coordinates."""
    cleaned = text.strip().replace(" ", "")
    if not cleaned:
        raise ValueError("Boş OCR çıktısı alındı.")
    if "," not in cleaned:
        raise ValueError(f"Virgül bulunamadı: {cleaned!r}")
    x_str, y_str = cleaned.split(",", 1)
    try:
        return int(x_str), int(y_str)
    except ValueError as exc:
        raise ValueError(f"Sayılara dönüştürülemedi: {cleaned!r}") from exc


def read_coordinate_from_frame(frame: np.ndarray) -> Tuple[int, int]:
    """End-to-end OCR pipeline for a single ROI frame."""
    processed = preprocess(frame)
    text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
    return parse_coordinate(text)


def read_coordinate_from_screen(roi: ROI = DEFAULT_ROI) -> Tuple[int, int]:
    """Capture the ROI from the screen and decode the coordinate."""
    frame = capture_roi(roi)
    return read_coordinate_from_frame(frame)


def ensure_pydirectinput_ready() -> None:
    """Validate that pydirectinput is installed before sending key events."""
    if pydirectinput is None:
        raise SystemExit(
            "pydirectinput kütüphanesi bulunamadı. Otomatik hareket için "
            "`pip install pydirectinput` komutunu çalıştırın."
        )
    pydirectinput.FAILSAFE = False


KeyInput = Union[str, Sequence[str]]


def _normalize_keys(keys: KeyInput) -> Tuple[str, ...]:
    if isinstance(keys, str):
        return (keys,)
    return tuple(keys)


def press_keys(keys: KeyInput, duration: float, settle_delay: float) -> None:
    """Press and release one or more keys for a given duration."""
    if duration <= 0:
        return
    key_sequence = _normalize_keys(keys)
    for key in key_sequence:
        pydirectinput.keyDown(key)
    time.sleep(duration)
    for key in reversed(key_sequence):
        pydirectinput.keyUp(key)
    time.sleep(KEY_RELEASE_DELAY)
    if settle_delay > 0:
        time.sleep(settle_delay)


class KeyHold:
    """Context manager for holding keys down until released."""

    def __init__(self, keys: KeyInput):
        self.keys = _normalize_keys(keys)
        self._released = False

    def __enter__(self) -> "KeyHold":
        for key in self.keys:
            pydirectinput.keyDown(key)
        return self

    def release(self) -> None:
        if self._released:
            return
        for key in reversed(self.keys):
            pydirectinput.keyUp(key)
        self._released = True
        time.sleep(KEY_RELEASE_DELAY)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class CoordinateSampler:
    """Continuously read coordinates on a background thread."""

    def __init__(self, roi: ROI, interval: float):
        self.roi = roi
        self.interval = max(interval, 0.02)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: list[Tuple[float, Tuple[int, int]]] = []
        self._lock = threading.Lock()
        self._new_sample = threading.Event()

    def start(self, initial_coord: Optional[Tuple[int, int]] = None) -> None:
        if initial_coord is not None:
            now = time.time()
            with self._lock:
                self._samples.append((now, initial_coord))
                self._new_sample.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                coord = read_coordinate_from_screen(self.roi)
                now = time.time()
                with self._lock:
                    self._samples.append((now, coord))
                    self._new_sample.set()
            except Exception as exc:  # pragma: no cover - diagnostics
                print(f"Örnekleme hatası: {exc}", file=sys.stderr, flush=True)
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def wait_sample(self, timeout: float) -> Optional[Tuple[float, Tuple[int, int]]]:
        if not self._new_sample.wait(timeout):
            return None
        self._new_sample.clear()
        with self._lock:
            return self._samples[-1] if self._samples else None

    def latest(self) -> Optional[Tuple[float, Tuple[int, int]]]:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def snapshot(self) -> list[Tuple[float, Tuple[int, int]]]:
        with self._lock:
            return list(self._samples)


def vector_from_points(start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
    """Return the vector from start to end as a float numpy array."""
    return np.array([end[0] - start[0], end[1] - start[1]], dtype=float)


def vector_norm(vec: np.ndarray) -> float:
    """Return the Euclidean norm of a 2D vector."""
    return float(np.linalg.norm(vec))


def signed_angle_deg(source: np.ndarray, target: np.ndarray) -> float:
    """Compute signed angle in degrees from source to target vector."""
    cross = source[0] * target[1] - source[1] * target[0]
    dot = source[0] * target[0] + source[1] * target[1]
    return math.degrees(math.atan2(cross, dot))


def distance_between(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Return the Euclidean distance between two coordinate points."""
    return vector_norm(vector_from_points(a, b))


def average_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Return the element-wise average of displacement vectors."""
    if not vectors:
        raise ValueError("Boş vektör listesi ortalaması alınamaz.")
    stack = np.stack(vectors, axis=0)
    return np.mean(stack, axis=0)


def sample_move(
    current_position: Tuple[int, int],
    keys: KeyInput,
    duration: float,
    roi: ROI,
    settle_delay: float,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Press a key and return the resulting displacement vector."""
    press_keys(keys, duration, settle_delay)
    new_position = read_coordinate_from_screen(roi)
    move_vec = vector_from_points(current_position, new_position)
    return move_vec, new_position


def sample_forward_step(roi: ROI, config: NavigationConfig) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Tap W for a short duration to measure the current forward direction vector.

    Returns the measured displacement vector and the updated current coordinate.
    """
    start = read_coordinate_from_screen(roi)
    forward_vec, current = sample_move(
        current_position=start,
        keys="w",
        duration=config.sample_duration,
        roi=roi,
        settle_delay=config.settle_delay,
    )
    if vector_norm(forward_vec) < MIN_FORWARD_DISTANCE:
        raise RuntimeError(
            "İleri hareket algılanamadı. `--sample-duration` değerini yükseltmeyi deneyin."
        )
    return forward_vec, current


def execute_motion_with_feedback(
    keys: KeyInput,
    duration: float,
    current: Tuple[int, int],
    target: Tuple[int, int],
    roi: ROI,
    config: NavigationConfig,
    forward_unit: np.ndarray,
    left_unit: np.ndarray,
) -> Tuple[Tuple[int, int], np.ndarray, bool, float]:
    """
    Hold the specified keys while sampling coordinates on a background thread.

    Returns the new coordinate, displacement vector, and whether the target was
    reached during the key press.
    """
    sampler = CoordinateSampler(roi, config.sample_interval)
    sampler.start(initial_coord=current)
    reached = False
    start_vector = vector_from_points(current, target)
    initial_forward = float(np.dot(start_vector, forward_unit))
    initial_lateral = float(np.dot(start_vector, left_unit))

    elapsed_time = 0.0
    with KeyHold(keys) as hold:
        start_time = time.time()
        latest_coord = current
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            remaining = duration - elapsed
            sample = sampler.wait_sample(timeout=min(config.sample_interval, remaining))
            if sample is None:
                continue
            _, coord = sample
            latest_coord = coord
            residual_vec = vector_from_points(coord, target)
            residual_norm = vector_norm(residual_vec)
            if residual_norm <= config.tolerance:
                reached = True
                break
            residual_forward = float(np.dot(residual_vec, forward_unit))
            residual_lateral = float(np.dot(residual_vec, left_unit))
            if initial_forward != 0 and (residual_forward > 0) != (initial_forward > 0):
                break
            if initial_lateral != 0 and (residual_lateral > 0) != (initial_lateral > 0):
                break
        elapsed_time = time.time() - start_time

    sampler.stop()
    time.sleep(config.settle_delay)
    samples = sampler.snapshot()
    final_coord = samples[-1][1] if samples else read_coordinate_from_screen(roi)
    displacement = vector_from_points(current, final_coord)
    return final_coord, displacement, reached, elapsed_time


def fine_tune_adjustment(
    current: Tuple[int, int],
    target: Tuple[int, int],
    roi: ROI,
    config: NavigationConfig,
    forward_unit: np.ndarray,
    left_unit: np.ndarray,
    calibration: Optional[CalibrationData],
) -> Tuple[int, int]:
    """
    Apply small corrective motions if the residual distance is within fine-tune threshold.
    """
    correction_attempts = 0
    while correction_attempts < config.fine_tune_max_attempts:
        residual_vec = vector_from_points(current, target)
        residual_dist = vector_norm(residual_vec)
        if residual_dist <= config.tolerance:
            break
        if residual_dist > config.fine_tune_threshold:
            break

        forward_component = float(np.dot(residual_vec, forward_unit))
        lateral_component = float(np.dot(residual_vec, left_unit))

        keys: KeyInput = "w"
        if abs(forward_component) < config.tolerance and abs(lateral_component) >= config.tolerance:
            keys = ("w", "a") if lateral_component > 0 else ("w", "d")
        elif forward_component < 0:
            keys = "s"
        elif abs(lateral_component) >= config.tolerance:
            keys = ("w", "a") if lateral_component > 0 else ("w", "d")

        duration = config.fine_tune_duration
        if residual_dist <= config.fine_tune_radius:
            duration = max(duration * 0.5, 0.05)
        if calibration:
            if keys == "w" and forward_component > 0:
                duration = min(duration, abs(forward_component) / max(calibration.forward_speed, 1e-6))
            elif keys == "s" and forward_component < 0:
                duration = min(duration, abs(forward_component) / max(calibration.backward_speed, 1e-6))
        if calibration and isinstance(keys, tuple):
            name = "".join(keys)
            if name.lower() in calibration.diagonal_components:
                forward_speed, lateral_speed = calibration.diagonal_components[name.lower()]
                speed = max(abs(forward_speed), abs(lateral_speed), 1e-6)
                duration = min(duration * 1.5, residual_dist / speed)
        current, displacement, _, _ = execute_motion_with_feedback(
            keys=keys,
            duration=duration,
            current=current,
            target=target,
            roi=roi,
            config=config,
            forward_unit=forward_unit,
            left_unit=left_unit,
        )
        correction_attempts += 1
        if vector_norm(displacement) < config.tolerance * 0.5:
            # If movement is negligible, break to avoid oscillation.
            break
    return current


def run_calibration(
    roi: ROI,
    forward_duration: float,
    rotation_duration: float,
    settle_delay: float,
    diagonal_duration: float,
    samples: int,
    include_diagonals: bool,
) -> CalibrationData:
    """Interactively measure movement and turning characteristics."""
    ensure_pydirectinput_ready()
    print("Kalibrasyon başlıyor. Karakteri boş bir alanda sabit tutun.", flush=True)
    time.sleep(1.5)
    if samples < 1:
        raise SystemExit("Kalibrasyon için örnek sayısı en az 1 olmalıdır.")

    current = read_coordinate_from_screen(roi)
    start_position = current
    print(f"Başlangıç konumu: {current[0]}, {current[1]}", flush=True)

    forward_vectors: list[np.ndarray] = []
    backward_vectors: list[np.ndarray] = []
    diagonal_samples: Dict[str, list[np.ndarray]] = {
        name: [] for name in DIAGONAL_COMBOS
    } if include_diagonals else {}
    diag_duration = diagonal_duration if include_diagonals else 0.0
    if include_diagonals and diag_duration <= 0:
        diag_duration = forward_duration

    for idx in range(samples):
        print(f"[Kalibrasyon] İleri örnek {idx + 1}/{samples}", flush=True)
        forward_vec, current = sample_move(
            current_position=current,
            keys="w",
            duration=forward_duration,
            roi=roi,
            settle_delay=settle_delay,
        )
        forward_vectors.append(forward_vec)

        backward_vec, current = sample_move(
            current_position=current,
            keys="s",
            duration=forward_duration,
            roi=roi,
            settle_delay=settle_delay,
        )
        backward_vectors.append(backward_vec)

    forward_avg = average_vector(forward_vectors)
    backward_avg = average_vector(backward_vectors)
    forward_speed = vector_norm(forward_avg) / max(forward_duration, 1e-6)
    backward_speed = vector_norm(backward_avg) / max(forward_duration, 1e-6)
    forward_unit = forward_avg / max(vector_norm(forward_avg), 1e-6)
    left_unit = np.array([-forward_unit[1], forward_unit[0]])

    print(
        f"İleri ölçüm ortalaması: hız ≈ {forward_speed:.2f} birim/sn | "
        f"vektör = ({forward_avg[0]:.2f}, {forward_avg[1]:.2f})",
        flush=True,
    )
    print(
        f"Geri ölçüm ortalaması: hız ≈ {backward_speed:.2f} birim/sn | "
        f"vektör = ({backward_avg[0]:.2f}, {backward_avg[1]:.2f})",
        flush=True,
    )

    right_angles = []
    left_angles = []

    for idx in range(samples):
        print(f"[Kalibrasyon] Sağa dönüş örneği {idx + 1}/{samples}", flush=True)
        press_keys("d", rotation_duration, settle_delay)
        right_forward_vec, current = sample_move(
            current_position=current,
            keys="w",
            duration=forward_duration,
            roi=roi,
            settle_delay=settle_delay,
        )
        right_angles.append(signed_angle_deg(forward_avg, right_forward_vec))
        _, current = sample_move(
            current_position=current,
            keys="s",
            duration=forward_duration,
            roi=roi,
            settle_delay=settle_delay,
        )
        press_keys("a", rotation_duration, settle_delay)

    for idx in range(samples):
        print(f"[Kalibrasyon] Sola dönüş örneği {idx + 1}/{samples}", flush=True)
        press_keys("a", rotation_duration, settle_delay)
        left_forward_vec, current = sample_move(
            current_position=current,
            keys="w",
            duration=forward_duration,
            roi=roi,
            settle_delay=settle_delay,
        )
        left_angles.append(signed_angle_deg(forward_avg, left_forward_vec))
        _, current = sample_move(
            current_position=current,
            keys="s",
            duration=forward_duration,
            roi=roi,
            settle_delay=settle_delay,
        )
        press_keys("d", rotation_duration, settle_delay)

    right_angle_avg = float(np.mean(right_angles))
    left_angle_avg = float(np.mean(left_angles))
    turn_right_rate = abs(right_angle_avg) / max(rotation_duration, 1e-6)
    turn_left_rate = abs(left_angle_avg) / max(rotation_duration, 1e-6)

    print(
        f"Sağa dönüş ortalaması: açı ≈ {right_angle_avg:.2f}° | "
        f"hız ≈ {turn_right_rate:.2f}°/sn",
        flush=True,
    )
    print(
        f"Sola dönüş ortalaması: açı ≈ {left_angle_avg:.2f}° | "
        f"hız ≈ {turn_left_rate:.2f}°/sn",
        flush=True,
    )

    diagonal_vectors: Dict[str, Tuple[float, float]] = {}
    diagonal_speeds: Dict[str, float] = {}
    diagonal_components: Dict[str, Tuple[float, float]] = {}

    rotation_sign = 1 if right_angle_avg >= 0 else -1
    timestamp = datetime.now().isoformat(timespec="seconds")

    if include_diagonals:
        for primary, opposite in DIAGONAL_PAIRS:
            primary_keys = DIAGONAL_COMBOS[primary]
            opposite_keys = DIAGONAL_COMBOS[opposite]
            for idx in range(samples):
                print(
                    f"[Kalibrasyon] Çapraz {primary.upper()} örneği {idx + 1}/{samples}",
                    flush=True,
                )
                vec_primary, current = sample_move(
                    current_position=current,
                    keys=primary_keys,
                    duration=diag_duration,
                    roi=roi,
                    settle_delay=settle_delay,
                )
                diagonal_samples[primary].append(vec_primary)

                vec_opposite, current = sample_move(
                    current_position=current,
                    keys=opposite_keys,
                    duration=diag_duration,
                    roi=roi,
                    settle_delay=settle_delay,
                )
                diagonal_samples[opposite].append(vec_opposite)

        for name, vectors in diagonal_samples.items():
            if not vectors:
                continue
            avg_vec = average_vector(vectors)
            diagonal_vectors[name] = (float(avg_vec[0]), float(avg_vec[1]))
            per_sec_vec = avg_vec / max(diag_duration, 1e-6)
            diagonal_speeds[name] = float(vector_norm(per_sec_vec))
            forward_component = float(np.dot(per_sec_vec, forward_unit))
            lateral_component = float(np.dot(per_sec_vec, left_unit))
            diagonal_components[name] = (forward_component, lateral_component)
            print(
                f"Çapraz {name.upper()} ortalaması: ileri {forward_component:.2f} birim/sn | "
                f"yana {lateral_component:.2f} birim/sn",
                flush=True,
            )

    final_position = read_coordinate_from_screen(roi)
    print(
        f"Kalibrasyon tamamlandı. Başlangıca göre sapma = "
        f"{distance_between(start_position, final_position):.2f}",
        flush=True,
    )

    return CalibrationData(
        forward_speed=forward_speed,
        backward_speed=backward_speed,
        turn_left_deg_per_sec=turn_left_rate,
        turn_right_deg_per_sec=turn_right_rate,
        rotation_sign=rotation_sign,
        forward_sample_duration=forward_duration,
        rotation_sample_duration=rotation_duration,
        settle_delay=settle_delay,
        timestamp=timestamp,
        forward_vector=(float(forward_avg[0]), float(forward_avg[1])),
        backward_vector=(float(backward_avg[0]), float(backward_avg[1])),
        diagonal_vectors=diagonal_vectors,
        diagonal_speeds=diagonal_speeds,
        diagonal_components=diagonal_components,
        diagonal_sample_duration=diag_duration,
    )


def save_calibration(data: CalibrationData, path: Path) -> None:
    """Persist calibration data to disk."""
    path.write_text(json.dumps(data.to_dict(), indent=2))
    print(f"Kalibrasyon verisi kaydedildi: {path}", flush=True)


def load_calibration(path: Path) -> CalibrationData:
    """Load calibration data from disk."""
    try:
        raw = json.loads(path.read_text())
        data = CalibrationData.from_dict(raw)
        print(
            f"Kalibrasyon yüklendi ({path}): ileri {data.forward_speed:.2f} birim/sn, "
            f"sağ dönüş {data.turn_right_deg_per_sec:.2f}°/sn",
            flush=True,
        )
        return data
    except FileNotFoundError as exc:
        raise SystemExit(f"Kalibrasyon dosyası bulunamadı: {path}") from exc
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise SystemExit(f"Kalibrasyon dosyası okunamadı: {path}: {exc}") from exc


def navigate_to_target(
    config: NavigationConfig,
    roi: ROI,
    calibration: Optional[CalibrationData] = None,
) -> bool:
    """
    Drive the player towards the target coordinate using W/A/S/D controls.

    Returns True if the target is reached within tolerance, else False.
    """
    ensure_pydirectinput_ready()
    current = read_coordinate_from_screen(roi)
    last_forward_vec: Optional[np.ndarray] = None
    last_forward_duration = config.sample_duration
    refresh_counter = config.direction_refresh_steps

    for step in range(1, config.max_steps + 1):
        if last_forward_vec is None or refresh_counter >= config.direction_refresh_steps:
            forward_vec, current = sample_forward_step(roi, config)
            last_forward_vec = forward_vec
            last_forward_duration = config.sample_duration
            refresh_counter = 0

        forward_norm = vector_norm(last_forward_vec)
        if forward_norm < MIN_FORWARD_DISTANCE:
            forward_vec, current = sample_forward_step(roi, config)
            last_forward_vec = forward_vec
            forward_norm = vector_norm(forward_vec)
            last_forward_duration = config.sample_duration
            refresh_counter = 0

        forward_unit = last_forward_vec / max(forward_norm, 1e-6)
        left_unit = np.array([-forward_unit[1], forward_unit[0]])

        dist = distance_between(current, config.target)
        print(
            f"[Step {step}] Konum={current[0]}, {current[1]} | "
            f"Hedef={config.target[0]}, {config.target[1]} | Mesafe={dist:.2f}",
            flush=True,
        )

        if config.tolerance < dist <= config.fine_tune_threshold:
            current = fine_tune_adjustment(
                current=current,
                target=config.target,
                roi=roi,
                config=config,
                forward_unit=forward_unit,
                left_unit=left_unit,
                calibration=calibration,
            )
            dist = distance_between(current, config.target)
            last_forward_vec = None
            refresh_counter = config.direction_refresh_steps

        if dist <= config.tolerance:
            print("Hedef konuma ulaşıldı.", flush=True)
            return True

        target_vec = vector_from_points(current, config.target)
        base_distance = vector_norm(target_vec)
        if base_distance <= config.tolerance:
            continue

        angle = signed_angle_deg(last_forward_vec, target_vec)
        if config.invert_rotation:
            angle = -angle

        if abs(angle) > config.align_degrees:
            direction_key = "a" if angle > 0 else "d"
            rotation_duration = config.rotation_duration
            if calibration:
                rate = (
                    calibration.turn_left_deg_per_sec
                    if direction_key == "a"
                    else calibration.turn_right_deg_per_sec
                )
                if rate > 1e-3:
                    computed = abs(angle) / rate
                    rotation_duration = min(
                        config.max_rotation_duration,
                        max(config.min_rotation_duration, computed),
                    )
            print(
                f"  Hizalama gerekli: açı {angle:.1f}°. "
                f"{direction_key.upper()} {rotation_duration:.2f} sn basılıyor.",
                flush=True,
            )
            press_keys(direction_key, rotation_duration, config.settle_delay)
            current = read_coordinate_from_screen(roi)
            last_forward_vec = None
            refresh_counter = config.direction_refresh_steps
            continue

        measured_forward_speed = vector_norm(last_forward_vec) / max(
            last_forward_duration, 1e-6
        )
        forward_speed = measured_forward_speed
        if calibration:
            forward_speed = max(measured_forward_speed, calibration.forward_speed)

        target_forward = float(np.dot(target_vec, forward_unit))
        target_lateral = float(np.dot(target_vec, left_unit))
        if abs(target_forward) < config.tolerance:
            target_forward = 0.0
        if abs(target_lateral) < config.tolerance:
            target_lateral = 0.0

        candidate_actions = [
            {
                "label": "W",
                "keys": ("w",),
                "forward_speed": forward_speed,
                "lateral_speed": 0.0,
            }
        ]
        if calibration:
            candidate_actions.append(
                {
                    "label": "S",
                    "keys": ("s",),
                    "forward_speed": -calibration.backward_speed,
                    "lateral_speed": 0.0,
                }
            )
            for name, components in calibration.diagonal_components.items():
                combo_keys = DIAGONAL_COMBOS.get(name)
                if not combo_keys:
                    continue
                candidate_actions.append(
                    {
                        "label": name.upper(),
                        "keys": combo_keys,
                        "forward_speed": components[0],
                        "lateral_speed": components[1],
                    }
                )

        best_action: Optional[dict] = None
        base_distance = max(base_distance, 1e-6)
        for action in candidate_actions:
            f_speed = action["forward_speed"]
            l_speed = action["lateral_speed"]
            if abs(f_speed) < 1e-6 and abs(l_speed) < 1e-6:
                continue
            positive_times = []
            if abs(f_speed) > 1e-6:
                ratio = target_forward / f_speed
                if ratio > 0:
                    positive_times.append(ratio)
            if abs(l_speed) > 1e-6:
                ratio = target_lateral / l_speed
                if ratio > 0:
                    positive_times.append(ratio)
            if not positive_times:
                continue
            estimated_time = min(positive_times) * config.move_gain
            estimated_time = min(
                config.max_move_duration,
                max(config.min_move_duration, estimated_time),
            )
            if estimated_time <= 0:
                continue
            per_sec_vec = forward_unit * f_speed + left_unit * l_speed
            predicted = per_sec_vec * estimated_time
            new_target = target_vec - predicted
            score = vector_norm(new_target)
            if score >= base_distance:
                continue
            if best_action is None or score < best_action["score"]:
                best_action = {
                    "label": action["label"],
                    "keys": action["keys"],
                    "duration": estimated_time,
                    "score": score,
                }

        if best_action:
            key_label = "+".join(k.upper() for k in _normalize_keys(best_action["keys"]))
            print(
                f"  İlerleme: {key_label} {best_action['duration']:.2f} sn.",
                flush=True,
            )
            new_current, displacement, reached, elapsed = execute_motion_with_feedback(
                keys=best_action["keys"],
                duration=best_action["duration"],
                current=current,
                target=config.target,
                roi=roi,
                config=config,
                forward_unit=forward_unit,
                left_unit=left_unit,
            )
            current = new_current
            if vector_norm(displacement) >= config.tolerance * 0.5:
                last_forward_vec = displacement
                last_forward_duration = max(elapsed, 1e-6)
            refresh_counter += 1
            continue

        forward_speed = max(
            forward_speed, MIN_FORWARD_DISTANCE / max(last_forward_duration, 1e-6)
        )
        travel_time = (base_distance / forward_speed) * config.move_gain
        move_duration = min(
            config.max_move_duration,
            max(config.min_move_duration, travel_time),
        )
        print(f"  İlerleme (varsayılan): W {move_duration:.2f} sn.", flush=True)
        current, displacement, _, elapsed = execute_motion_with_feedback(
            keys="w",
            duration=move_duration,
            current=current,
            target=config.target,
            roi=roi,
            config=config,
            forward_unit=forward_unit,
            left_unit=left_unit,
        )
        if vector_norm(displacement) >= config.tolerance * 0.5:
            last_forward_vec = displacement
            last_forward_duration = max(elapsed, 1e-6)
        refresh_counter += 1

    print(
        "Uyarı: Maksimum adım sayısına ulaşıldı, hedef koordinata varılamadı.",
        file=sys.stderr,
        flush=True,
    )
    return False


def save_debug_images(
    frame: np.ndarray,
    processed: np.ndarray,
    directory: Optional[Path],
    iteration: int,
) -> None:
    """Optionally save raw and processed frames for manual inspection."""
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f"roi_{iteration:04d}.png"
    processed_path = directory / f"roi_{iteration:04d}_thresh.png"
    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(processed_path), processed)


def loop_read_coordinates(
    roi: ROI,
    interval: float,
    limit: Optional[int],
    debug_dir: Optional[Path],
) -> None:
    """Continuously read coordinates until interrupted or limit reached."""
    with mss.mss() as sct:
        for idx in range(limit or sys.maxsize):
            frame = np.array(sct.grab(asdict(roi)))
            processed = preprocess(frame)
            try:
                coord = parse_coordinate(
                    pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
                )
                print(f"{coord[0]}, {coord[1]}", flush=True)
            except ValueError as exc:
                print(f"OCR hatası: {exc}", file=sys.stderr, flush=True)
            save_debug_images(frame, processed, debug_dir, idx)
            if interval > 0:
                time.sleep(interval)


def read_from_image(image_path: Path) -> Tuple[int, int]:
    """Load a saved screenshot of the ROI and decode it."""
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"Görsel açılamadı: {image_path}")
    return read_coordinate_from_frame(frame)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Belirtilen ROI alanından oyun koordinatlarını okur."
    )
    parser.add_argument(
        "--from-image",
        type=Path,
        dest="image_path",
        help="Ekran görüntüsü dosyasından (ör. koordinat.png) koordinat oku.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Döngüde iki okuma arasındaki süre (saniye). 0 ile durmaksızın okur.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Belirli sayıda örnek okuduktan sonra çık.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Ham ve işlenmiş ROI karelerini belirtilen klasöre kaydet.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Tek seferlik okuma yap ve çık.",
    )
    parser.add_argument(
        "--navigate",
        nargs=2,
        type=int,
        metavar=("TARGET_X", "TARGET_Y"),
        help="Belirtilen hedef koordinata otomatik olarak W/A/S/D ile ilerle.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Hedefe ulaştı saymak için koordinat toleransı.",
    )
    parser.add_argument(
        "--sample-duration",
        type=float,
        default=0.18,
        help="Yön vektörünü ölçmek için W tuşuna basılacak süre (saniye).",
    )
    parser.add_argument(
        "--rotation-duration",
        type=float,
        default=0.08,
        help="Hizalama sırasında A/D tuşlarına basılacak süre (saniye).",
    )
    parser.add_argument(
        "--min-rotation-duration",
        type=float,
        default=0.05,
        help="Otomatik hizalama için minimum dönme süresi.",
    )
    parser.add_argument(
        "--max-rotation-duration",
        type=float,
        default=0.35,
        help="Otomatik hizalama için maksimum dönme süresi.",
    )
    parser.add_argument(
        "--align-deg",
        type=float,
        default=8.0,
        help="Hedef yön ile mevcut yön arasındaki izin verilen açı farkı (derece).",
    )
    parser.add_argument(
        "--min-move-duration",
        type=float,
        default=0.12,
        help="İleri hareket sırasında W tuşuna basılacak minimum süre.",
    )
    parser.add_argument(
        "--max-move-duration",
        type=float,
        default=0.5,
        help="İleri hareket sırasında W tuşuna basılacak maksimum süre.",
    )
    parser.add_argument(
        "--move-gain",
        type=float,
        default=0.6,
        help="Hedefe olan uzaklık ile W basma süresi arasındaki çarpan.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.08,
        help="Hareket sırasında koordinat okumaları için kullanılacak örnekleme aralığı.",
    )
    parser.add_argument(
        "--direction-refresh",
        type=int,
        default=5,
        help="Kaç adımda bir W yön örneklemesi yapılacağını belirtir.",
    )
    parser.add_argument(
        "--fine-threshold",
        type=float,
        default=3.5,
        help="İnce ayar denemesi yapılacak maksimum mesafe.",
    )
    parser.add_argument(
        "--fine-radius",
        type=float,
        default=2.0,
        help="İnce ayar hedef yarıçapı.",
    )
    parser.add_argument(
        "--fine-duration",
        type=float,
        default=0.12,
        help="İnce ayar sırasında tuşa basma süresi.",
    )
    parser.add_argument(
        "--fine-attempts",
        type=int,
        default=4,
        help="İnce ayar için maksimum deneme sayısı.",
    )
    parser.add_argument(
        "--settle-delay",
        type=float,
        default=0.12,
        help="Her tuş basışından sonra oyunun koordinat güncellemesi için beklenecek süre.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=180,
        help="Otomatik navigasyon denemesi sırasında izin verilen maksimum iterasyon sayısı.",
    )
    parser.add_argument(
        "--invert-rotation",
        action="store_true",
        help="Pozitif açı farklarını sağa dönüş (D) olarak yorumla (gerekirse).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Harita için otomatik kalibrasyon çalıştır ve çık.",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=BASE_DIR / "calibration.json",
        help="Kalibrasyon verisinin kaydedileceği veya yükleneceği dosya yolu.",
    )
    parser.add_argument(
        "--calibration-forward-duration",
        type=float,
        default=1.0,
        help="Kalibrasyon sırasında ileri/geri adımlar için W/S tuşlarına basılacak süre.",
    )
    parser.add_argument(
        "--calibration-rotation-duration",
        type=float,
        default=0.45,
        help="Kalibrasyon sırasında A/D tuşlarına basılacak süre.",
    )
    parser.add_argument(
        "--calibration-diagonal-duration",
        type=float,
        default=1.0,
        help="Kalibrasyon sırasında W/A, W/D, S/A, S/D kombinasyonlarına basılacak süre.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=3,
        help="Her yönde uygulanacak kalibrasyon tekrar sayısı.",
    )
    parser.add_argument(
        "--no-calibration-diagonals",
        action="store_false",
        dest="calibration_diagonals",
        help="Kalibrasyon sırasında çapraz kombinasyonları atla.",
    )
    parser.set_defaults(calibration_diagonals=True)
    parser.add_argument(
        "--tesseract",
        type=Path,
        help="Tesseract OCR yürütülebilir dosyasının yolu. "
        "PATH içinde değilse bu parametre ile verin.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    roi = DEFAULT_ROI

    configure_tesseract(args.tesseract)

    if args.calibrate:
        calibration = run_calibration(
            roi=roi,
            forward_duration=args.calibration_forward_duration,
            rotation_duration=args.calibration_rotation_duration,
            settle_delay=args.settle_delay,
            diagonal_duration=args.calibration_diagonal_duration,
            samples=args.calibration_samples,
            include_diagonals=args.calibration_diagonals,
        )
        save_calibration(calibration, args.calibration_file)
        return

    if args.image_path:
        coord = read_from_image(args.image_path)
        print(f"Dosya içindeki koordinat: {coord[0]}, {coord[1]}")
        return

    calibration: Optional[CalibrationData] = None
    if args.navigate and args.calibration_file and args.calibration_file.exists():
        calibration = load_calibration(args.calibration_file)
    elif args.navigate and args.calibration_file:
        print(
            f"Bilgi: {args.calibration_file} bulunamadı, varsayılan ayarlar kullanılacak.",
            flush=True,
        )

    if args.navigate:
        sample_duration = args.sample_duration
        rotation_duration = args.rotation_duration
        settle_delay = args.settle_delay
        min_rotation_duration = args.min_rotation_duration
        max_rotation_duration = args.max_rotation_duration
        invert_rotation_flag = args.invert_rotation

        if calibration:
            settle_delay = max(settle_delay, calibration.settle_delay)
            invert_rotation_flag = invert_rotation_flag or (calibration.rotation_sign > 0)
            print("Kalibrasyon verileri yüklendi; çapraz hareket ve hız katsayıları kullanılacak.", flush=True)

        navigation_config = NavigationConfig(
            target=(args.navigate[0], args.navigate[1]),
            tolerance=args.tolerance,
            sample_duration=sample_duration,
            rotation_duration=rotation_duration,
            min_rotation_duration=min_rotation_duration,
            max_rotation_duration=max_rotation_duration,
            align_degrees=args.align_deg,
            max_move_duration=args.max_move_duration,
            min_move_duration=args.min_move_duration,
            move_gain=args.move_gain,
            settle_delay=settle_delay,
            max_steps=args.max_steps,
            invert_rotation=invert_rotation_flag,
            sample_interval=args.sample_interval,
            direction_refresh_steps=args.direction_refresh,
            fine_tune_threshold=args.fine_threshold,
            fine_tune_radius=args.fine_radius,
            fine_tune_duration=args.fine_duration,
            fine_tune_max_attempts=args.fine_attempts,
        )
        success = navigate_to_target(navigation_config, roi, calibration)
        if not success:
            sys.exit(1)
        return

    if args.once:
        coord = read_coordinate_from_screen(roi)
        print(f"Ekran koordinatı: {coord[0]}, {coord[1]}")
        return

    try:
        loop_read_coordinates(
            roi=roi,
            interval=max(args.interval, 0.0),
            limit=args.limit,
            debug_dir=args.debug_dir,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
