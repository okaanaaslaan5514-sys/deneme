"""Automation controller for Priest Torment skill."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    pydirectinput = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_TORMENT

try:
    import pyautogui  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    pyautogui = None  # type: ignore[assignment]

F_RETURN_KEY = "f1"
FUNCTION_PRESS_DURATION = 0.025
FUNCTION_GAP_DURATION = 0.025
PRIMARY_PRESS_DURATION = 0.25
PRIMARY_GAP_DURATION = 0.25
POST_SEQUENCE_DELAY = 1.0  # wait after center click per spec (1s after click)
CLICK_DELAY_BEFORE = 0.5


def _screen_center() -> Optional[Tuple[int, int]]:
    if pyautogui is None:
        return None
    try:
        size = pyautogui.size()
    except Exception:
        return None
    return size.width // 2, size.height // 2


@dataclass
class TormentConfig:
    """Mutable configuration snapshot for Torment automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    function_key: Optional[str] = None
    interval_seconds: float = 15.0

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
        )


class TormentMicroController:
    """Trigger Torment sequence on the selected interval."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = TormentConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._timer_thread: Optional[threading.Thread] = None
        self._timer_stop = threading.Event()

        self._warned_missing_pydirectinput = False
        self._warned_missing_pyautogui = False

    def update_config(self, config: TormentConfig) -> None:
        with self._state_lock:
            self._config = config
        self._refresh_timer_thread()

    def shutdown(self) -> None:
        self._stop_timer_thread()

    # ------------------------------------------------------------------ #
    # Timer handling
    # ------------------------------------------------------------------ #

    def _refresh_timer_thread(self) -> None:
        with self._state_lock:
            config = self._config
        should_run = config.ready()
        if not should_run:
            self._stop_timer_thread()
            return
        thread = self._timer_thread
        if thread and thread.is_alive():
            return
        self._timer_stop.clear()
        self._timer_thread = threading.Thread(target=self._timer_loop, name="TormentTimer", daemon=True)
        self._timer_thread.start()

    def _stop_timer_thread(self) -> None:
        thread = self._timer_thread
        if thread and thread.is_alive():
            self._timer_stop.set()
            thread.join(timeout=1.0)
        self._timer_thread = None
        self._timer_stop.clear()

    def _timer_loop(self) -> None:
        try:
            while not self._timer_stop.is_set():
                with self._state_lock:
                    config = self._config
                if not config.ready():
                    break
                self._execute_sequence(config)
                wait_time = max(config.interval_seconds, 0.0)
                if self._timer_stop.wait(wait_time):
                    break
        finally:
            self._timer_stop.clear()
            with self._state_lock:
                if self._timer_thread is threading.current_thread():
                    self._timer_thread = None

    # ------------------------------------------------------------------ #
    # Core execution
    # ------------------------------------------------------------------ #

    def _execute_sequence(self, config: TormentConfig) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Torment tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        self._warned_missing_pydirectinput = False

        center = _screen_center()
        if center is None:
            if not self._warned_missing_pyautogui:
                self._status_callback("pyautogui modülü veya ekran bilgisi alınamadı; Torment tıklaması yapılamadı.", 6000)
                self._warned_missing_pyautogui = True
            return
        self._warned_missing_pyautogui = False

        with self._action_lock:
            try:
                with action_scheduler.claim(PRIORITY_TORMENT):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    time.sleep(CLICK_DELAY_BEFORE)
                    self._click_center(center)
                    time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key(F_RETURN_KEY, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._status_callback(f"Torment komutu gönderilemedi: {exc}", 5000)

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)

    def _click_center(self, center: Tuple[int, int]) -> None:
        if pyautogui is None:
            return
        x, y = center
        try:
            pyautogui.click(x, y)
        except Exception as exc:
            self._status_callback(f"Torment tıklaması başarısız: {exc}", 5000)


__all__ = ["TormentConfig", "TormentMicroController"]
