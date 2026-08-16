"""Terminal Escape handling for a graceful run stop.

The listener is deliberately inactive for redirected stdin and non-POSIX terminals.  Ctrl-C keeps
its ordinary interrupt semantics; Escape only sets a cooperative flag that search loops inspect at
their existing generation/optimizer-step boundaries.
"""

from __future__ import annotations

import os
import select
import sys
import threading
from collections.abc import Callable
from copy import deepcopy
from typing import IO, Any


class EscapeShutdown:
    """Turn a standalone Escape keypress into a thread-safe cooperative stop request."""

    def __init__(self, on_request: Callable[[], None] | None = None, *, stream: IO[str] | None = None) -> None:
        self._requested = threading.Event()
        self._closing = threading.Event()
        self._on_request = on_request
        self._stream = stream if stream is not None else sys.stdin
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._terminal_state: list[Any] | None = None

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def request(self) -> None:
        """Request shutdown once; safe to call from tests or another control surface."""

        if self._requested.is_set():
            return
        self._requested.set()
        if self._on_request is not None:
            self._on_request()

    def start(self) -> bool:
        """Begin listening when stdin is an interactive POSIX terminal."""

        if self._thread is not None or not self._stream.isatty():
            return False
        try:
            import termios
            import tty

            fd = self._stream.fileno()
            terminal_state = deepcopy(termios.tcgetattr(fd))
            tty.setcbreak(fd, termios.TCSANOW)
        except (AttributeError, ImportError, OSError, ValueError):
            return False
        self._fd = fd
        self._terminal_state = terminal_state
        self._closing.clear()
        self._thread = threading.Thread(target=self._listen, name="versal-escape-shutdown", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop listening and restore the exact terminal mode captured by :meth:`start`."""

        self._closing.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.25)
        if self._fd is not None and self._terminal_state is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSAFLUSH, self._terminal_state)
            except (ImportError, OSError, ValueError):
                pass
        self._fd = None
        self._terminal_state = None

    def _listen(self) -> None:
        fd = self._fd
        if fd is None:
            return
        while not self._closing.is_set() and not self._requested.is_set():
            try:
                readable, _writable, _errors = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                value = os.read(fd, 1)
                if value != b"\x1b":
                    continue
                # Arrow/function keys begin with Escape too.  A standalone Escape has no bytes
                # immediately following it; consume an escape sequence without stopping the run.
                continuation, _writable, _errors = select.select([fd], [], [], 0.04)
                if continuation:
                    os.read(fd, 32)
                    continue
                self.request()
            except OSError:
                return

    def __enter__(self) -> "EscapeShutdown":
        self.start()
        return self

    def __exit__(self, _error_type: object, _error: object, _traceback: object) -> None:
        self.stop()
