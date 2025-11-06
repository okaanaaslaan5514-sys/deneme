"""Automation controller for Priest Restore skill triggered by party HP thresholds."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np  # type: ignore[import]

try:
    import mss  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mss = None  # type: ignore[assignment]

try:
    import pyautogui  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pyautogui = None  # type: ignore[assignment]

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pydirectinput = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_RESTORE
from priest import read_party_hp as partyhp
from priest.cure_micro import (
    MATCH_THRESHOLD,
    SkillArea,
    load_skill_area,
    load_skill_templates,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMON_DIR = PROJECT_ROOT / "common"
SKILL_AREA_PATH = COMMON_DIR / "skill_alani.json"
SKILL_IMAGES_DIR = COMMON_DIR / "skills"

ANCHOR_FILE = partyhp.DEFAULT_ANCHOR_FILE
CALIBRATION_FILE = partyhp.DEFAULT_CALIBRATION_FILE

HP_POLL_INTERVAL = 0.5
RESTORE_COOLDOWN_SECONDS = 3.0
FUNCTION_PRESS_DURATION = 0.025
FUNCTION_GAP_DURATION = 0.025
PRIMARY_PRESS_DURATION = 0.25
PRIMARY_GAP_DURATION = 0.25
RETURN_PRESS_DURATION = 0.025
RETURN_GAP_DURATION = 0.025
POST_SEQUENCE_DELAY = 1.5


@dataclass
class RestoreConfig:
    """Mutable configuration snapshot for Restore automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    party_size: int = 8
    threshold: Optional[int] = None
    primary_key: Optional[str] = None
    function_key: Optional[str] = None

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
            and self.threshold is not None
            and self.party_size > 0
        )


class RestoreMicroController:
    """Monitor party HP and trigger Restore when threshold is crossed."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = RestoreConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()

        self._anchor: Optional[Tuple[int, int]] = None
        self._calibration: Optional[partyhp.Calibration] = None

        self._skill_area: Optional[SkillArea] = None
        self._templates: List[np.ndarray] = []

        self._warned_missing_anchor = False
        self._warned_missing_calibration = False
        self._warned_missing_mss = False
        self._warned_missing_skill_area = False
        self._warned_missing_templates = False
        self._warned_missing_pydirectinput = False
        self._warned_missing_pyautogui = False

        self._last_action_ts = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_config(self, config: RestoreConfig) -> None:
        with self._state_lock:
            self._config = config
        self._refresh_monitor_thread()

    def shutdown(self) -> None:
        self._stop_monitor_thread()

    def calibrate(self, slots: int) -> None:
        partyhp.calibrate_rois(ANCHOR_FILE, CALIBRATION_FILE, max(1, min(slots, partyhp.MAX_SLOTS)))
        with self._state_lock:
            self._anchor = None
            self._calibration = None

    def invalidate_calibration_cache(self) -> None:
        with self._state_lock:
            self._anchor = None
            self._calibration = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_monitor_thread(self) -> None:
        with self._state_lock:
            config = self._config
        should_run = config.ready()
        if not should_run:
            self._stop_monitor_thread()
            return
        thread = self._monitor_thread
        if thread and thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="RestoreMonitor", daemon=True)
        self._monitor_thread.start()

    def _stop_monitor_thread(self) -> None:
        thread = self._monitor_thread
        if thread and thread.is_alive():
            self._monitor_stop.set()
            thread.join(timeout=1.0)
        self._monitor_thread = None
        self._monitor_stop.clear()

    def _monitor_loop(self) -> None:
        try:
            while not self._monitor_stop.wait(HP_POLL_INTERVAL):
                with self._state_lock:
                    config = self._config
                if not config.ready():
                    break
                if not self._ensure_hp_resources():
                    continue
                triggered = self._should_trigger_restore(config)
                if not triggered:
                    continue
                now = time.monotonic()
                if now - self._last_action_ts < RESTORE_COOLDOWN_SECONDS:
                    continue
                self._perform_restore(config)
                self._last_action_ts = time.monotonic()
        finally:
            self._monitor_stop.clear()
            with self._state_lock:
                if self._monitor_thread is threading.current_thread():
                    self._monitor_thread = None

    def _ensure_hp_resources(self) -> bool:
        if self._anchor is None:
            try:
                anchor_settings = partyhp.load_anchor_file(ANCHOR_FILE)
                self._anchor = anchor_settings.as_tuple()
                self._warned_missing_anchor = False
            except FileNotFoundError:
                if not self._warned_missing_anchor:
                    self._status_callback(
                        "Restore için anchor bulunamadı. 'python -m priest.read_party_hp --calibrate-roi' komutunu çalıştırın.",
                        6000,
                    )
                    self._warned_missing_anchor = True
                return False
            except Exception as exc:
                if not self._warned_missing_anchor:
                    self._status_callback(f"Anchor okunamadı: {exc}", 6000)
                    self._warned_missing_anchor = True
                return False
        if self._calibration is None:
            try:
                self._calibration = partyhp.load_calibration(CALIBRATION_FILE)
                self._warned_missing_calibration = False
            except FileNotFoundError:
                if not self._warned_missing_calibration:
                    self._status_callback(
                        "Restore kalibrasyonu bulunamadı. 'python -m priest.read_party_hp --calibrate-roi' komutunu çalıştırın.",
                        6000,
                    )
                    self._warned_missing_calibration = True
                return False
            except Exception as exc:
                if not self._warned_missing_calibration:
                    self._status_callback(f"Kalibrasyon yüklenemedi: {exc}", 6000)
                    self._warned_missing_calibration = True
                return False
        return True

    def _should_trigger_restore(self, config: RestoreConfig) -> bool:
        try:
            frame = partyhp.grab_screen()
        except Exception as exc:
            self._status_callback(f"Ekran görüntüsü alınamadı: {exc}", 6000)
            return False

        try:
            values, _ = partyhp.read_hp_from_frame(frame, config.party_size, self._anchor or (0, 0), self._calibration)  # type: ignore[arg-type]
        except Exception as exc:
            self._status_callback(f"HP okunamadı: {exc}", 6000)
            return False

        threshold = config.threshold if config.threshold is not None else 0
        for result in values:
            if result is None:
                continue
            current, maximum = result
            pct = partyhp.compute_percentage(current, maximum)
            if pct < threshold:
                return True
        return False

    def _perform_restore(self, config: RestoreConfig) -> None:
        if not self._prepare_skill_resources():
            self._execute_sequence(config, double_clicked=False)
            return
        double_clicked = self._attempt_double_click()
        self._execute_sequence(config, double_clicked=double_clicked)

    def _execute_sequence(self, config: RestoreConfig, double_clicked: bool) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Restore tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        self._warned_missing_pydirectinput = False

        with self._action_lock:
            try:
                with action_scheduler.claim(PRIORITY_RESTORE):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key("f1", RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
            except Exception as exc:
                self._status_callback(f"Restore komutu gönderilemedi: {exc}", 5000)

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)

    def _prepare_skill_resources(self) -> bool:
        if mss is None:
            if not self._warned_missing_mss:
                self._status_callback("mss modülü bulunamadı; skill alanı taranamadı.", 6000)
                self._warned_missing_mss = True
            return False
        if self._skill_area is None:
            try:
                self._skill_area = load_skill_area(SKILL_AREA_PATH)
                self._warned_missing_skill_area = False
            except FileNotFoundError:
                if not self._warned_missing_skill_area:
                    self._status_callback("Skill alanını işaretle.", 6000)
                    self._warned_missing_skill_area = True
                return False
            except ValueError as exc:
                if not self._warned_missing_skill_area:
                    self._status_callback(f"Skill alanı geçersiz: {exc}", 6000)
                    self._warned_missing_skill_area = True
                return False
        if not self._templates:
            templates = load_skill_templates(SKILL_IMAGES_DIR)
            if not templates:
                if not self._warned_missing_templates:
                    self._status_callback("Restore için skill görseli bulunamadı. skill_resmi.py ile kayıt yapın.", 6000)
                    self._warned_missing_templates = True
                return False
            self._templates = templates
            self._warned_missing_templates = False
        return True

    def _attempt_double_click(self) -> bool:
        location = self._locate_skill_icon()
        if location is None:
            return False
        if pyautogui is None:
            if not self._warned_missing_pyautogui:
                self._status_callback("pyautogui modülü bulunamadı; skill alanında çift tıklama yapılmadı.", 6000)
                self._warned_missing_pyautogui = True
            return False
        self._warned_missing_pyautogui = False
        try:
            pyautogui.moveTo(location[0], location[1])
            pyautogui.click()
            pyautogui.click()
            return True
        except Exception as exc:
            self._status_callback(f"Skill alanına çift tıklama başarısız: {exc}", 6000)
            return False

    def _locate_skill_icon(self) -> Optional[Tuple[int, int]]:
        if mss is None or not self._skill_area or not self._templates:
            return None
        try:
            with mss.mss() as sct:  # type: ignore[attr-defined]
                monitor = {
                    "top": int(self._skill_area.top),
                    "left": int(self._skill_area.left),
                    "width": int(self._skill_area.width),
                    "height": int(self._skill_area.height),
                }
                grab = sct.grab(monitor)
        except Exception:
            return None
        frame = np.array(grab)
        if frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_center: Optional[Tuple[int, int]] = None
        for template in self._templates:
            if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                center_x = self._skill_area.left + max_loc[0] + template.shape[1] // 2
                center_y = self._skill_area.top + max_loc[1] + template.shape[0] // 2
                best_center = (center_x, center_y)
        if best_score >= MATCH_THRESHOLD:
            return best_center
        return None


__all__ = ["RestoreConfig", "RestoreMicroController"]
