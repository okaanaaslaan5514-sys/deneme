"""Automation controller for Priest Cure skill."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

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

try:
    import keyboard  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    keyboard = None  # type: ignore[assignment]

from priest.action_scheduler import action_scheduler, PRIORITY_CURE
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMON_DIR = PROJECT_ROOT / "common"
SKILL_AREA_PATH = COMMON_DIR / "skill_alani.json"
SKILL_IMAGES_DIR = COMMON_DIR / "skills"

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
class SkillArea:
    """Screen region where skill icons reside."""

    left: int
    top: int
    width: int
    height: int


@dataclass
class CureConfig:
    """Mutable configuration snapshot for Cure automation."""

    global_enabled: bool = False
    skill_enabled: bool = False
    auto_enabled: bool = False
    shortcut_enabled: bool = False
    shortcut_key: Optional[str] = None
    primary_key: Optional[str] = None
    function_key: Optional[str] = None

    def auto_ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and self.auto_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
        )

    def shortcut_ready(self) -> bool:
        return (
            self.global_enabled
            and self.skill_enabled
            and self.shortcut_enabled
            and bool(self.primary_key)
            and bool(self.function_key)
            and bool(self.shortcut_key)
        )


def load_skill_area(path: Path = SKILL_AREA_PATH) -> SkillArea:
    """Load skill area rectangle from JSON file."""
    if not path.exists():
        raise FileNotFoundError("Skill alanı JSON bulunamadı.")
    data = json.loads(path.read_text())
    try:
        right = int(data["right"])
        width = int(data["width"])
        top = int(data["top"])
        height = int(data["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Skill alanı JSON beklenen alanları içermiyor.") from exc
    left = right - width
    if width <= 0 or height <= 0:
        raise ValueError("Skill alanı genişlik/yükseklik değeri hatalı.")
    return SkillArea(left=left, top=top, width=width, height=height)


def load_skill_templates(directory: Path = SKILL_IMAGES_DIR) -> List[np.ndarray]:
    """Load grayscale skill templates from disk."""
    templates: List[np.ndarray] = []
    if not directory.exists():
        return templates
    for image_path in sorted(directory.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            continue
        templates.append(image)
    return templates


class CureMicroController:
    """Coordinate Cure automation for Priest support actions."""

    def __init__(self, status_callback: Callable[[str, int], None]) -> None:
        self._status_callback = status_callback
        self._config = CureConfig()
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()

        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()

        self._skill_area: Optional[SkillArea] = None
        self._templates: List[np.ndarray] = []

        self._shortcut_handle: Optional[int] = None
        self._registered_hotkey: Optional[str] = None

        self._last_action_ts: float = 0.0

        self._warned_missing_mss = False
        self._warned_missing_templates = False
        self._warned_missing_area = False
        self._warned_missing_keyboard = False
        self._warned_missing_pydirectinput = False

    def update_config(self, config: CureConfig) -> None:
        """Receive updated UI configuration."""
        with self._state_lock:
            previous = self._config
            self._config = config
        self._refresh_hotkey(previous, config)
        self._refresh_auto_thread(previous, config)

    def trigger_manual(self) -> None:
        """Invoke Cure action via shortcut (thread-safe)."""
        with self._state_lock:
            config = self._config
        if not config.shortcut_ready():
            return
        self._execute_action(config)

    def shutdown(self) -> None:
        """Stop background threads and unregister hotkeys."""
        self._stop_auto_thread()
        self._unregister_hotkey()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _refresh_auto_thread(self, previous: CureConfig, current: CureConfig) -> None:
        should_run = current.auto_ready()
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
        self._auto_thread = threading.Thread(target=self._auto_loop, name="CureAuto", daemon=True)
        self._auto_thread.start()

    def _refresh_hotkey(self, previous: CureConfig, current: CureConfig) -> None:
        should_register = current.shortcut_ready()
        if not should_register:
            self._unregister_hotkey()
            return
        if keyboard is None:
            if not self._warned_missing_keyboard:
                self._status_callback("keyboard modülü bulunamadı; Cure kısayolu devre dışı.", 6000)
                self._warned_missing_keyboard = True
            self._unregister_hotkey()
            return
        self._warned_missing_keyboard = False
        key = current.shortcut_key or ""
        if key == self._registered_hotkey and self._shortcut_handle is not None:
            return
        self._unregister_hotkey()
        try:
            handle = keyboard.add_hotkey(key, self._handle_manual_trigger, suppress=False)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._status_callback(f"Cure kısayolu kaydedilemedi: {exc}", 5000)
            return
        self._registered_hotkey = key
        self._shortcut_handle = handle

    def _prepare_detection_resources(self) -> bool:
        if mss is None:
            if not self._warned_missing_mss:
                self._status_callback("mss modülü bulunamadı; Cure otomatik modu devre dışı.", 6000)
                self._warned_missing_mss = True
            return False
        try:
            self._skill_area = load_skill_area()
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
        templates = load_skill_templates()
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
                    if not config.auto_ready():
                        break
                    detected = self._detect_any_skill(screen)
                    wait_time = AUTO_POLL_SECONDS
                    if detected:
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
        except Exception:  # pragma: no cover - screen capture guard
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

    def _execute_action(self, config: CureConfig) -> bool:
        if not config.primary_key or not config.function_key:
            return False
        if pydirectinput is None:
            if not self._warned_missing_pydirectinput:
                self._status_callback("pydirectinput modülü bulunamadı; Cure tetiklenemedi.", 6000)
                self._warned_missing_pydirectinput = True
            return False
        now = time.monotonic()
        with self._action_lock:
            if now - self._last_action_ts < MIN_ACTION_GAP:
                return False
            try:
                with action_scheduler.claim(PRIORITY_CURE):
                    self._press_key(config.function_key, FUNCTION_PRESS_DURATION, FUNCTION_GAP_DURATION)
                    self._press_key(config.primary_key, PRIMARY_PRESS_DURATION, PRIMARY_GAP_DURATION)
                    self._press_key("f1", RETURN_PRESS_DURATION, RETURN_GAP_DURATION)
                    time.sleep(POST_SEQUENCE_DELAY)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._status_callback(f"Cure komutu gönderilemedi: {exc}", 5000)
                return False
            self._last_action_ts = time.monotonic()
        self._status_callback("Cure tetiklendi.", 1500)
        return True

    def _press_key(self, key: str, hold: float, gap: float) -> None:
        """Simulate key press with given hold and gap durations."""
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(gap)

    def _handle_manual_trigger(self) -> None:
        threading.Thread(target=self.trigger_manual, name="CureManual", daemon=True).start()

    def _stop_auto_thread(self) -> None:
        thread = self._auto_thread
        if thread and thread.is_alive():
            self._auto_stop.set()
            thread.join(timeout=1.0)
        self._auto_thread = None
        self._auto_stop.clear()

    def _unregister_hotkey(self) -> None:
        if self._shortcut_handle is not None and keyboard is not None:
            try:
                keyboard.remove_hotkey(self._shortcut_handle)
            except KeyError:  # pragma: no cover - already removed
                pass
        self._shortcut_handle = None
        self._registered_hotkey = None


__all__ = ["CureConfig", "CureMicroController"]
