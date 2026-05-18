# utils/pipeline_log.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List

@dataclass
class LogEntry:
    elapsed: float   # seconds since reset
    level: str       # "info" | "success" | "warning" | "error"
    message: str

_entries: List[LogEntry] = []
_t0: float = 0.0


def reset() -> None:
    """Clear the log buffer and reset the elapsed timer. Call once before each pipeline run."""
    global _entries, _t0
    _entries = []
    _t0 = time.time()


def log(message: str, level: str = "info") -> None:
    """Append a timestamped log entry."""
    elapsed = round(time.time() - _t0, 2)
    _entries.append(LogEntry(elapsed=elapsed, level=level, message=message))


def get_logs() -> List[LogEntry]:
    return list(_entries)
