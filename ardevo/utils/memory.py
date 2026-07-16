"""Best-effort release of free native heap pages at task boundaries."""

from __future__ import annotations

import ctypes
import sys


def release_unused_host_memory() -> bool:
    """Return glibc's free arenas to Linux without affecting live Python or Torch objects.

    ``gc.collect()`` destroys the old task tensors, but glibc can retain their now-free arenas in
    each spawned worker. That retained RSS accumulated across modalities on Lattice until the
    kernel killed the run. ``malloc_trim`` is advisory and glibc-specific, so unsupported platforms
    simply report that no trim occurred.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError):
        return False
