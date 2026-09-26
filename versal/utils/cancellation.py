"""
Process-local access to a run's shared cooperative cancellation flag.

The parent is the only writer. A lock-free byte lets its signal handler request a
stop without acquiring a lock or calling logging code from an interrupted frame.
"""

from __future__ import annotations

from typing import Any

_flag: Any = None


def cancellation_flag() -> Any:
    """
    Return the installed flag, creating an idle one before worker startup if needed.
    """
    global _flag
    if _flag is None:
        import multiprocessing

        _flag = multiprocessing.get_context("spawn").RawValue("b", 0)
    return _flag


def install_cancellation_flag(flag: Any) -> Any:
    """
    Install a parent-owned flag in this process and return the previous binding.
    """
    global _flag
    previous, _flag = _flag, flag
    return previous


def cancelled() -> bool:
    """
    Check cancellation without allocating state or consuming randomness.
    """
    return _flag is not None and bool(_flag.value)
