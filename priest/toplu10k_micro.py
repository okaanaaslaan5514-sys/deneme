"""Automated Toplu 10k micro triggered by party HP thresholds."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

try:
    import pydirectinput  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pydirectinput = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_TOPLU10K
from priest import read_party_hp as partyhp

ANCHOR_FILE = partyhp.DEFAULT_ANCHOR_FILE
CALIBRATION_FILE = partyhp.DEFAULT_CALIBRATION_FILE

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
class Toplu10kConfig:
    """Mutable configuration snapshot for Toplu 10k automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    primary_key: Optional[str] = None
    function_key: Optional[str] = None
    threshold: Optional[int] = None
    party_size: int = 8

    def ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
            and self.threshold is not None
            and self.party_size > 0
        )


class Toplu10kMicroController:
    """Monitor party HP and trigger Toplu 10k automatically below threshold."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = Toplu10kConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._anchor: Optional[Tuple[int, int]] = None
        self._calibration: Optional[partyhp.Calibration] = None

        self._warned_missing_pydirectinput = False
        self._warned_missing_anchor = False
        self._warned_missing_calibration = False
        self._warned_capture_failure = False

        self._last_action_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_config(self, config: Toplu10kConfig) -> None:
        """Receive updated UI configuration."""
        with self._state_lock:
            self._config = config
        self._refresh_monitor_thread(config)

    def shutdown(self) -> None:
        """Stop monitoring thread."""
        self._stop_monitor_thread()

    def calibrate(self, slots: int) -> None:
        """Run anchor + ROI calibration for party HP reading."""
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

    def _refresh_monitor_thread(self, config: Toplu10kConfig) -> None:
        should_run = config.ready()
        if not should_run:
            self._stop_monitor_thread()
            return
        thread = self._monitor_thread
        if thread and thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="Toplu10kMonitor", daemon=True)
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
                if not self._ensure_hp_resources():
                    continue
                triggered = self._should_trigger(config)
                if triggered:
                    self._execute_action(config)
        finally:
            self._stop_event.clear()
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
                        "Toplu 10k için anchor bulunamadı. 'python -m priest.read_party_hp --calibrate-roi' komutunu çalıştırın.",
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
                        "Toplu 10k kalibrasyonu bulunamadı. 'python -m priest.read_party_hp --calibrate-roi' komutunu çalıştırın.",
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

    def _should_trigger(self, config: Toplu10kConfig) -> bool:
        try:
            frame = partyhp.grab_screen()
        except Exception as exc:
            if not self._warned_capture_failure:
                self._status_callback(f"Ekran görüntüsü alınamadı: {exc}", 6000)
                self._warned_capture_failure = True
            return False
        self._warned_capture_failure = False

        try:
            values, _ = partyhp.read_hp_from_frame(
                frame,
                config.party_size,
                self._anchor or (0, 0),
                self._calibration,  # type: ignore[arg-type]
            )
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

    def _execute_action(self, config: Toplu10kConfig) -> None:
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Toplu 10k tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return
        now = time.monotonic()
        with self._action_lock:
            if now - self._last_action_ts < MIN_ACTION_GAP:
                return
            try:
                with action_scheduler.claim(PRIORITY_TOPLU10K):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
                    self._press_key("f1", RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._status_callback(f"Toplu 10k komutu gönderilemedi: {exc}", 5000)
                return
            self._last_action_ts = time.monotonic()
        self._status_callback("Toplu 10k tetiklendi.", 1500)

    def _press_key(self, key: Optional[str], hold: float, gap: float) -> None:
        if not key or pydirectinput is None:
            return
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)


__all__ = ["Toplu10kConfig", "Toplu10kMicroController"]
