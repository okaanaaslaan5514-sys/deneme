"""Shortcut-triggered automation for Priest Zehir Cure skill."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    pydirectinput = None  # type: ignore[assignment]

try:
    import keyboard  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    keyboard = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_POISON_CURE

FUNCTION_PRESS_DURATION = 0.025
FUNCTION_GAP_DURATION = 0.025
MOVE_PRESS_DURATION = 0.025
MOVE_GAP_DURATION = 0.025
PRIMARY_PRESS_DURATION = 0.25
PRIMARY_GAP_DURATION = 0.25
RETURN_PRESS_DURATION = 0.025
RETURN_GAP_DURATION = 0.025
POST_SEQUENCE_DELAY = 1.5
MIN_ACTION_GAP = POST_SEQUENCE_DELAY
MOVE_KEY = "w"


@dataclass
class PoisonCureConfig:
    """Mutable configuration snapshot for Zehir Cure automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    function_key: Optional[str] = None
    shortcut_key: Optional[str] = None

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
            and bool(self.shortcut_key)
        )


class PoisonCureMicroController:
    """Register shortcut and trigger Zehir Cure sequence on demand."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = PoisonCureConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._shortcut_handle: Optional[int] = None
        self._registered_hotkey: Optional[str] = None

        self._warned_missing_keyboard = False
        self._warned_missing_pydirectinput = False

        self._last_action_ts: float = 0.0

    def update_config(self, config: PoisonCureConfig) -> None:
        """Receive updated UI configuration."""
        with self._state_lock:
            self._config = config
        self._refresh_hotkey(config)

    def shutdown(self) -> None:
        """Unregister hotkey and stop background activity."""
        self._unregister_hotkey()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_hotkey(self, config: PoisonCureConfig) -> None:
        should_register = config.ready()
        if not should_register:
            self._unregister_hotkey()
            return
        if keyboard is None:
            if not self._warned_missing_keyboard:
                self._status_callback("keyboard modülü bulunamadı; Zehir Cure kısayolu devre dışı.", 6000)
                self._warned_missing_keyboard = True
            self._unregister_hotkey()
            return
        self._warned_missing_keyboard = False
        key = config.shortcut_key or ""
        if key == self._registered_hotkey and self._shortcut_handle is not None:
            return
        self._unregister_hotkey()
        try:
            handle = keyboard.add_hotkey(key, self._handle_manual_trigger, suppress=False)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._status_callback(f"Zehir Cure kısayolu kaydedilemedi: {exc}", 5000)
            return
        self._registered_hotkey = key
        self._shortcut_handle = handle

    def _handle_manual_trigger(self) -> None:
        threading.Thread(target=self._trigger_sequence, name="ZehirCureShortcut", daemon=True).start()

    def _trigger_sequence(self) -> None:
        with self._state_lock:
            config = self._config
        if not config.ready():
            return
        self._execute_action(config)

    def _execute_action(self, config: PoisonCureConfig) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Zehir Cure tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        now = time.monotonic()
        with self._action_lock:
            if now - self._last_action_ts < MIN_ACTION_GAP:
                return
            try:
                with action_scheduler.claim(PRIORITY_POISON_CURE):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(MOVE_KEY, MOVE_PRESS_DURATION, MOVE_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key("f1", RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._status_callback(f"Zehir Cure komutu gönderilemedi: {exc}", 5000)
                return
            self._last_action_ts = time.monotonic()
        self._status_callback("Zehir Cure tetiklendi.", 1500)

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)

    def _unregister_hotkey(self) -> None:
        if self._shortcut_handle is not None and keyboard is not None:
            try:
                keyboard.remove_hotkey(self._shortcut_handle)
            except KeyError:  # pragma: no cover
                pass
        self._shortcut_handle = None
        self._registered_hotkey = None


__all__ = ["PoisonCureConfig", "PoisonCureMicroController"]
