"""Automation controller for Priest Heal skill based on HP percentage."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pydirectinput = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_HEAL
from common import bar_reader
from common import read_hp

FUNCTION_PRESS_DURATION = 0.025
FUNCTION_GAP_DURATION = 0.025
PRIMARY_PRESS_DURATION = 0.025
PRIMARY_GAP_DURATION = 0.025
CHAIN_PRESS_DURATION = 0.025
CHAIN_GAP_DURATION = 0.025
RETURN_PRESS_DURATION = 0.025
RETURN_GAP_DURATION = 0.025
POST_SEQUENCE_DELAY = 1.5
MIN_ACTION_GAP = POST_SEQUENCE_DELAY
POLL_INTERVAL = 0.5


@dataclass
class HealConfig:
    """Mutable configuration snapshot for Heal automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    function_key: Optional[str] = None
    threshold: Optional[int] = None

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
            and self.threshold is not None
        )


class HealMicroController:
    """Monitor HP percentage and trigger Heal skill below threshold."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = HealConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._warned_missing_pydirectinput = False
        self._warned_hp_error = False
        self._warned_invalid_primary = False

        self._last_action_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_config(self, config: HealConfig) -> None:
        with self._state_lock:
            self._config = config
        self._refresh_monitor_thread(config)

    def shutdown(self) -> None:
        self._stop_monitor_thread()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_monitor_thread(self, config: HealConfig) -> None:
        should_run = config.ready()
        if not should_run:
            self._stop_monitor_thread()
            return
        thread = self._monitor_thread
        if thread and thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="HealMonitor", daemon=True)
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
                percent = self._read_hp_percentage()
                if percent is None:
                    continue
                threshold = config.threshold if config.threshold is not None else 0
                if percent > threshold:
                    continue
                self._execute_action(config)
        finally:
            self._stop_event.clear()
            with self._state_lock:
                if self._monitor_thread is threading.current_thread():
                    self._monitor_thread = None

    def _read_hp_percentage(self) -> Optional[float]:
        try:
            current, maximum = bar_reader.read_bar_from_screen(read_hp.HP_ROI)
        except Exception as exc:
            if not self._warned_hp_error:
                self._status_callback(f"HP okunamadı: {exc}", 6000)
                self._warned_hp_error = True
            return None
        self._warned_hp_error = False
        return bar_reader.compute_percentage(current, maximum)

    def _execute_action(self, config: HealConfig) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Heal tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        self._warned_missing_pydirectinput = False

        next_key = self._compute_secondary_key(config.primary_key)
        if next_key is None:
            return

        now = time.monotonic()
        with self._action_lock:
            if now - self._last_action_ts < MIN_ACTION_GAP:
                return
            try:
                with action_scheduler.claim(PRIORITY_HEAL):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    self._press_key(next_key, CHAIN_PRESS_DURATION, CHAIN_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key("f1", RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._status_callback(f"Heal komutu gönderilemedi: {exc}", 5000)
                return
            self._last_action_ts = time.monotonic()
        self._status_callback("Heal tetiklendi.", 1500)

    def _compute_secondary_key(self, primary_key: Optional[str]) -> Optional[str]:
        if not primary_key:
            return None
        if not primary_key.isdigit():
            if not self._warned_invalid_primary:
                self._status_callback("Heal için numara seçiminde bir rakam seçin.", 5000)
                self._warned_invalid_primary = True
            return None
        self._warned_invalid_primary = False
        value = int(primary_key)
        secondary = (value + 1) % 10
        return str(secondary)

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)


__all__ = ["HealConfig", "HealMicroController"]
