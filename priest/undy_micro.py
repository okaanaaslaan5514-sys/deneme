"""Automation controller for Priest Undy skill."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import cv2  # type: ignore[import]
import numpy as np  # type: ignore[import]

try:
    import mss  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    mss = None  # type: ignore[assignment]

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    pydirectinput = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_UNDY
from priest import cure_micro as cure_shared

MATCH_THRESHOLD = 0.82
AUTO_COOLDOWN_SECONDS = 1.5
AUTO_POLL_SECONDS = 0.35
FUNCTION_PRESS_DURATION = 0.025
FUNCTION_GAP_DURATION = 0.025
PRIMARY_PRESS_DURATION = 0.25
PRIMARY_GAP_DURATION = 0.25
RETURN_PRESS_DURATION = 0.025
RETURN_GAP_DURATION = 0.025
POST_SEQUENCE_DELAY = 1.5
MIN_ACTION_GAP = POST_SEQUENCE_DELAY


@dataclass
class UndyConfig:
    """Mutable configuration snapshot for Undy automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    function_key: Optional[str] = None

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
        )


class UndyMicroController:
    """Coordinates Undy automation by tracking buff absence."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = UndyConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()

        self._templates: List[np.ndarray] = []
        self._skill_area: Optional[cure_shared.SkillArea] = None

        self._warned_missing_mss = False
        self._warned_missing_templates = False
        self._warned_missing_area = False
        self._warned_missing_pydirectinput = False

        self._last_action_ts: float = 0.0

    def update_config(self, config: UndyConfig) -> None:
        """Receive updated UI configuration."""
        with self._state_lock:
            self._config = config
        self._refresh_auto_thread(config)

    def shutdown(self) -> None:
        """Stop background processes."""
        self._stop_auto_thread()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_auto_thread(self, config: UndyConfig) -> None:
        should_run = config.ready()
        if not should_run:
            self._stop_auto_thread()
            return
        if not self._prepare_detection_resources():
            self._stop_auto_thread()
            return
        thread = self._auto_thread
        if thread and thread.is_alive():
            return
        self._auto_stop.clear()
        self._auto_thread = threading.Thread(target=self._auto_loop, name="UndyAuto", daemon=True)
        self._auto_thread.start()

    def _prepare_detection_resources(self) -> bool:
        if mss is None:
            if not self._warned_missing_mss:
                self._status_callback("mss modülü bulunamadı; Undy otomatik modu devre dışı.", 6000)
                self._warned_missing_mss = True
            return False
        try:
            self._skill_area = cure_shared.load_skill_area()
            self._warned_missing_area = False
        except FileNotFoundError:
            if not self._warned_missing_area:
                self._status_callback("Skill alanını işaretle.", 6000)
                self._warned_missing_area = True
            return False
        except ValueError as exc:
            self._status_callback(f"Skill alanı geçersiz: {exc}", 6000)
            self._warned_missing_area = True
            return False
        templates = cure_shared.load_skill_templates()
        if not templates:
            if not self._warned_missing_templates:
                self._status_callback("Skill görseli bulunamadı. skill_resmi.py ile ekleyin.", 6000)
                self._warned_missing_templates = True
            return False
        self._templates = templates
        self._warned_missing_templates = False
        return True

    def _auto_loop(self) -> None:
        if mss is None:
            return
        try:
            with mss.mss() as screen:  # type: ignore[attr-defined]
                while not self._auto_stop.is_set():
                    with self._state_lock:
                        config = self._config
                    if not config.ready():
                        break
                    detected = self._detect_any_skill(screen)
                    wait_time = AUTO_POLL_SECONDS
                    if not detected:
                        if self._execute_action(config):
                            wait_time = AUTO_COOLDOWN_SECONDS
                    self._auto_stop.wait(wait_time)
        finally:
            self._auto_stop.clear()
            with self._state_lock:
                self._auto_thread = None

    def _detect_any_skill(self, screen: Any) -> bool:
        if not self._templates or self._skill_area is None:
            return False
        monitor = {
            "top": self._skill_area.top,
            "left": self._skill_area.left,
            "width": self._skill_area.width,
            "height": self._skill_area.height,
        }
        try:
            grab = screen.grab(monitor)
        except Exception:  # pragma: no cover
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
            if max_val >= MATCH_THRESHOLD:
                return True
        return False

    def _execute_action(self, config: UndyConfig) -> bool:
        if not config.primary_key or not config.function_key:
            return False
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Undy tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return False
        now = time.monotonic()
        with self._action_lock:
            if now - self._last_action_ts < MIN_ACTION_GAP:
                return False
            try:
                with action_scheduler.claim(PRIORITY_UNDY):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    self._press_key("f1", RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
            except Exception as exc:  # pragma: no cover
                self._status_callback(f"Undy komutu gönderilemedi: {exc}", 5000)
                return False
            self._last_action_ts = time.monotonic()
        self._status_callback("Undy tetiklendi.", 1500)
        return True

    def _press_key(self, key: str, hold: float, gap: float) -> None:
        if pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)

    def _stop_auto_thread(self) -> None:
        thread = self._auto_thread
        if thread and thread.is_alive():
            self._auto_stop.set()
            thread.join(timeout=1.0)
        self._auto_thread = None
        self._auto_stop.clear()


__all__ = ["UndyConfig", "UndyMicroController"]
