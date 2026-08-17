"""Monotonic deadline helpers shared by training, evaluation, and worker chunks."""

from __future__ import annotations

import time


def expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline
