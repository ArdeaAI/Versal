"""Monotonic deadline helpers shared by training, evaluation, and worker chunks."""

from __future__ import annotations

import time

from versal.utils.cancellation import cancelled


def expired(deadline: float | None) -> bool:
    return cancelled() or (deadline is not None and time.perf_counter() >= deadline)
