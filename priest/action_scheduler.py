"""Global priority scheduler for coordinating key press actions."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


class ActionScheduler:
    """Serialize keypress actions according to configured priorities."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_priority: Optional[int] = None
        self._waiting: Dict[int, int] = {}

    @contextmanager
    def claim(self, priority: int) -> Iterator[None]:
        """Acquire permission to perform a keypress action."""
        with self._condition:
            self._waiting[priority] = self._waiting.get(priority, 0) + 1
            try:
                while True:
                    if self._active_priority is not None:
                        self._condition.wait()
                        continue
                    highest_waiting = self._highest_waiting_priority()
                    if highest_waiting is not None and highest_waiting < priority:
                        self._condition.wait()
                        continue
                    break
                self._active_priority = priority
            finally:
                self._waiting[priority] -= 1
                if self._waiting[priority] <= 0:
                    del self._waiting[priority]
        try:
            yield
        finally:
            with self._condition:
                self._active_priority = None
                self._condition.notify_all()

    def _highest_waiting_priority(self) -> Optional[int]:
        if not self._waiting:
            return None
        return min(self._waiting.keys())


# Priority levels (lower value means higher priority)
PRIORITY_HEAL = 10
PRIORITY_RESTORE = 11
PRIORITY_TOPLU10K = 12

PRIORITY_CURE = 20
PRIORITY_POISON_CURE = 21
PRIORITY_UNDY = 22
PRIORITY_AC = 23
PRIORITY_STR30 = 24
PRIORITY_TOPLU_CURE = 25
PRIORITY_TOPLU_AC = 26
PRIORITY_TOPLU_BUFF = 27

PRIORITY_PARAZIT = 30
PRIORITY_MALICE = 31
PRIORITY_TEKLI = 32
PRIORITY_TORMENT = 33
PRIORITY_SUBSIDE = 34

PRIORITY_RSIZ = 40
PRIORITY_RR = 41


action_scheduler = ActionScheduler()


__all__ = [
    "action_scheduler",
    "PRIORITY_HEAL",
    "PRIORITY_RESTORE",
    "PRIORITY_TOPLU10K",
    "PRIORITY_CURE",
    "PRIORITY_POISON_CURE",
    "PRIORITY_UNDY",
    "PRIORITY_AC",
    "PRIORITY_STR30",
    "PRIORITY_TOPLU_CURE",
    "PRIORITY_TOPLU_AC",
    "PRIORITY_TOPLU_BUFF",
    "PRIORITY_PARAZIT",
    "PRIORITY_MALICE",
    "PRIORITY_TEKLI",
    "PRIORITY_TORMENT",
    "PRIORITY_SUBSIDE",
    "PRIORITY_RSIZ",
    "PRIORITY_RR",
]

