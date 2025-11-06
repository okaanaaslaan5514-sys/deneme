"""Shortcut-triggered automation for Priest R'siz Atak skill."""
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
PRIMARY_PRESS_DURATION = 0.025
PRIMARY_GAP_DURATION = 0.025
LOOP_DELAY = 0.12


@dataclass
class RsizAtakConfig:
    """Mutable configuration snapshot for R'siz Atak automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    shortcut_key: Optional[str] = None

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.shortcut_key)
        )


class RsizAtakMicroController:
    """Register shortcut and trigger the attack skill without switching F bars."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = RsizAtakConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._shortcut_handle: Optional[int] = None
        self._registered_hotkey: Optional[str] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_stop = threading.Event()

        self._warned_missing_keyboard = False
        self._warned_missing_pydirectinput = False

        self._loop_active = False

    def update_config(self, config: RsizAtakConfig) -> None:
        """Receive updated UI configuration."""
        with self._state_lock:
            self._config = config
        self._refresh_hotkey(config)
        if not config.ready():
            self._stop_loop()

    def shutdown(self) -> None:
        """Unregister hotkey and stop background activity."""
        self._stop_loop()
        self._unregister_hotkey()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_hotkey(self, config: RsizAtakConfig) -> None:
        should_register = config.ready()
        if not should_register:
            self._unregister_hotkey()
            return
        if keyboard is None:
            if not self._warned_missing_keyboard:
                self._status_callback("keyboard modülü bulunamadı; R'siz Atak kısayolu devre dışı.", 6000)
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
            self._status_callback(f"R'siz Atak kısayolu kaydedilemedi: {exc}", 5000)
            return
        self._registered_hotkey = key
        self._shortcut_handle = handle

    def _handle_manual_trigger(self) -> None:
        threading.Thread(target=self._toggle_loop, name="RsizAtakShortcut", daemon=True).start()

    def _toggle_loop(self) -> None:
        with self._state_lock:
            config = self._config
        if not config.ready():
            self._status_callback("R'siz Atak hazır değil; ayarları kontrol edin.", 4000)
            return
        if self._loop_active:
            self._stop_loop()
            self._status_callback("R'siz Atak döngüsü durduruldu.", 3000)
            return
        self._start_loop()
        self._status_callback("R'siz Atak döngüsü başladı.", 3000)

    def _execute_action(self, config: RsizAtakConfig) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; R'siz Atak tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        with self._action_lock:
            try:
                self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._status_callback(f"R'siz Atak komutu gönderilemedi: {exc}", 5000)
                return
        time.sleep(LOOP_DELAY)

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)

    def _start_loop(self) -> None:
        if self._loop_active:
            return
        self._loop_stop.clear()
        self._loop_thread = threading.Thread(target=self._loop_run, name="RsizAtakLoop", daemon=True)
        self._loop_active = True
        self._loop_thread.start()

    def _stop_loop(self) -> None:
        if not self._loop_active:
            return
        self._loop_stop.set()
        thread = self._loop_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._loop_active = False
        self._loop_thread = None
        self._loop_stop.clear()

    def _loop_run(self) -> None:
        try:
            while not self._loop_stop.is_set():
                with self._state_lock:
                    config = self._config
                if not config.ready():
                    break
                self._execute_action(config)
        finally:
            self._loop_active = False
            self._loop_thread = None
            self._loop_stop.clear()

    def _unregister_hotkey(self) -> None:
        if self._shortcut_handle is not None and keyboard is not None:
            try:
                keyboard.remove_hotkey(self._shortcut_handle)
            except KeyError:  # pragma: no cover
                pass
        self._shortcut_handle = None
        self._registered_hotkey = None


__all__ = ["RsizAtakConfig", "RsizAtakMicroController"]
