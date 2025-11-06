"""Automation controller for Priest Malice skill triggered by timer or Parazit completion."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    pydirectinput = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_MALICE
F_RETURN_KEY = "f1"
FUNCTION_PRESS_DURATION = 0.025
FUNCTION_GAP_DURATION = 0.025
PRIMARY_PRESS_DURATION = 0.25
PRIMARY_GAP_DURATION = 0.25
RETURN_PRESS_DURATION = 0.025
RETURN_GAP_DURATION = 0.025
POST_SEQUENCE_DELAY = 1.5


@dataclass
class MaliceConfig:
    """Mutable configuration snapshot for Malice automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    function_key: Optional[str] = None
    mode: Optional[str] = None  # "timer" or "parazit"
    interval_seconds: float = 10.0

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
        )

    def timer_ready(self) -> bool:
        return self.ready() and self.mode == "timer"

    def parazit_ready(self) -> bool:
        return self.ready() and self.mode == "parazit"


class MaliceMicroController:
    """Coordinate Malice automation for timer or Parazit completion triggers."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = MaliceConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._timer_thread: Optional[threading.Thread] = None
        self._timer_stop = threading.Event()

        self._warned_missing_pydirectinput = False
        self._completion_callbacks: List[Callable[[], None]] = []

    def update_config(self, config: MaliceConfig) -> None:
        """Receive updated UI configuration."""
        with self._state_lock:
            self._config = config
        self._refresh_timer_thread()

    def shutdown(self) -> None:
        """Stop background timers."""
        self._stop_timer_thread()
        self._completion_callbacks.clear()

    # ------------------------------------------------------------------ #
    # Timer handling
    # ------------------------------------------------------------------ #

    def _refresh_timer_thread(self) -> None:
        with self._state_lock:
            config = self._config
        should_run = config.timer_ready()
        if not should_run:
            self._stop_timer_thread()
            return
        thread = self._timer_thread
        if thread and thread.is_alive():
            return
        self._timer_stop.clear()
        self._timer_thread = threading.Thread(target=self._timer_loop, name="MaliceTimer", daemon=True)
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
                if not config.timer_ready():
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
    # Parazit completion trigger
    # ------------------------------------------------------------------ #

    def handle_parazit_completion(self) -> None:
        """Invoke Malice sequence when Parazit automation completes."""
        with self._state_lock:
            config = self._config
        if not config.parazit_ready():
            return
        self._execute_sequence(config)

    # ------------------------------------------------------------------ #
    # Core execution
    # ------------------------------------------------------------------ #

    def _execute_sequence(self, config: MaliceConfig) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Malice tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        self._warned_missing_pydirectinput = False

        with self._action_lock:
            try:
                with action_scheduler.claim(PRIORITY_MALICE):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key(F_RETURN_KEY, RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._status_callback(f"Malice komutu gönderilemedi: {exc}", 5000)
                return
        self._notify_completion()

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)

    def register_completion_callback(self, callback: Callable[[], None]) -> None:
        if callback not in self._completion_callbacks:
            self._completion_callbacks.append(callback)

    def unregister_completion_callback(self, callback: Callable[[], None]) -> None:
        if callback in self._completion_callbacks:
            self._completion_callbacks.remove(callback)

    def _notify_completion(self) -> None:
        for callback in list(self._completion_callbacks):
            try:
                callback()
            except Exception:  # pragma: no cover - defensive guard
                pass


__all__ = ["MaliceConfig", "MaliceMicroController"]
