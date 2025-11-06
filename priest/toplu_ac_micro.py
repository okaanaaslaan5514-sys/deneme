"""Automation controller for Priest Toplu AC skill based on skill icon visibility."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2  # type: ignore[import]
import numpy as np  # type: ignore[import]

try:
    import mss  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mss = None  # type: ignore[assignment]

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pydirectinput = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_TOPLU_AC
from priest import cure_micro as cure_shared

FUNCTION_PRESS_DURATION = 0.025
FUNCTION_GAP_DURATION = 0.025
PRIMARY_PRESS_DURATION = 0.25
PRIMARY_GAP_DURATION = 0.25
RETURN_PRESS_DURATION = 0.025
RETURN_GAP_DURATION = 0.025
POST_SEQUENCE_DELAY = 1.5
MIN_ACTION_GAP = POST_SEQUENCE_DELAY
POLL_INTERVAL = 0.5


@dataclass
class TopluAcConfig:
    """Mutable configuration snapshot for Toplu AC automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    function_key: Optional[str] = None
    precure_enabled: bool = False

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
        )


class TopluAcMicroController:
    """Monitor skill icon and trigger Toplu AC with optional pre-cure step."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = TopluAcConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._skill_area: Optional[cure_shared.SkillArea] = None
        self._templates: List[np.ndarray] = []

        self._warned_missing_mss = False
        self._warned_missing_skill_area = False
        self._warned_missing_templates = False
        self._warned_missing_pydirectinput = False

        self._last_action_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_config(self, config: TopluAcConfig) -> None:
        with self._state_lock:
            self._config = config
        self._refresh_monitor_thread(config)

    def shutdown(self) -> None:
        self._stop_monitor_thread()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_monitor_thread(self, config: TopluAcConfig) -> None:
        should_run = config.ready()
        if not should_run:
            self._stop_monitor_thread()
            return
        if not self._prepare_skill_resources():
            self._stop_monitor_thread()
            return
        thread = self._monitor_thread
        if thread and thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="TopluAcMonitor", daemon=True)
        self._monitor_thread.start()

    def _stop_monitor_thread(self) -> None:
        thread = self._monitor_thread
        if thread and thread.is_alive():
            self._stop_event.set()
            thread.join(timeout=1.0)
        self._monitor_thread = None
        self._stop_event.clear()

    def _monitor_loop(self) -> None:
        try:
            while not self._stop_event.wait(POLL_INTERVAL):
                with self._state_lock:
                    config = self._config
                if not config.ready():
                    break
                if not self._prepare_skill_resources():
                    continue
                if self._is_skill_present():
                    continue
                self._execute_action(config)
        finally:
            self._stop_event.clear()
            with self._state_lock:
                if self._monitor_thread is threading.current_thread():
                    self._monitor_thread = None

    def _prepare_skill_resources(self) -> bool:
        if mss is None:
            if not self._warned_missing_mss:
                self._status_callback("mss modülü bulunamadı; skill alanı taranamadı.", 6000)
                self._warned_missing_mss = True
            return False
        if self._skill_area is None:
            try:
                self._skill_area = cure_shared.load_skill_area()
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
            templates = cure_shared.load_skill_templates()
            if not templates:
                if not self._warned_missing_templates:
                    self._status_callback("Toplu AC için skill görseli bulunamadı. skill_resmi.py ile kayıt yapın.", 6000)
                    self._warned_missing_templates = True
                return False
            self._templates = templates
            self._warned_missing_templates = False
        return True

    def _is_skill_present(self) -> bool:
        if mss is None or self._skill_area is None or not self._templates:
            return False
        try:
            with mss.mss() as screen:  # type: ignore[attr-defined]
                monitor = {
                    "top": int(self._skill_area.top),
                    "left": int(self._skill_area.left),
                    "width": int(self._skill_area.width),
                    "height": int(self._skill_area.height),
                }
                grab = screen.grab(monitor)
        except Exception:
            return False
        frame = np.array(grab)
        if frame.size == 0:
            return False
        if frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for template in self._templates:
            if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= cure_shared.MATCH_THRESHOLD:
                return True
        return False

    def _execute_action(self, config: TopluAcConfig) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Toplu AC tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        self._warned_missing_pydirectinput = False

        now = time.monotonic()
        with self._action_lock:
            if now - self._last_action_ts < MIN_ACTION_GAP:
                return
            try:
                with action_scheduler.claim(PRIORITY_TOPLU_AC):
                    if config.precure_enabled:
                        self._press_key("f1", FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                        self._press_key("0", PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                        time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key("f1", RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
            except Exception as exc:
                self._status_callback(f"Toplu AC komutu gönderilemedi: {exc}", 5000)
                return
            self._last_action_ts = time.monotonic()
        self._status_callback("Toplu AC tetiklendi.", 1500)

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)


__all__ = ["TopluAcConfig", "TopluAcMicroController"]
