# utils/pipeline_log.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

@dataclass
class LogEntry:
    elapsed: float   # seconds since reset
    level: str       # "info" | "success" | "warning" | "error"
    message: str

_entries: List[LogEntry] = []
_t0: float = 0.0
_ui_callback: Optional[Callable[[List[LogEntry]], None]] = None


def set_ui_callback(fn: Optional[Callable[[List[LogEntry]], None]]) -> None:
    global _ui_callback
    _ui_callback = fn


def reset() -> None:
    """Clear the log buffer and reset the elapsed timer. Call once before each pipeline run."""
    global _entries, _t0, _ui_callback
    _entries = []
    _t0 = time.time()
    _ui_callback = None


def log(message: str, level: str = "info") -> None:
    """Append a timestamped log entry and push to live UI callback if set."""
    elapsed = round(time.time() - _t0, 2)
    _entries.append(LogEntry(elapsed=elapsed, level=level, message=message))
    if _ui_callback is not None:
        try:
            _ui_callback(list(_entries))
        except Exception:
            pass


def get_logs() -> List[LogEntry]:
    return list(_entries)
