"""
Cooperative Escape and Ctrl-C handling, with a second Ctrl-C escape hatch.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import threading
from collections.abc import Callable
from copy import deepcopy
from typing import IO, Any

from versal.utils.cancellation import install_cancellation_flag

_ACTIVE: EscapeShutdown | None = None


class ForcedShutdown(BaseException):
    """
    Explicit second-interrupt exit; bypass expensive crash-report generation.
    """


def active_shutdown() -> EscapeShutdown | None:
    """
    Share the CLI controller with trials without putting runtime objects in config.
    """
    return _ACTIVE


class EscapeShutdown:
    """
    Own terminal controls and SIGINT until persistence and worker cleanup finish.
    """

    def __init__(self, on_request: Callable[[], None] | None = None, *, stream: IO[str] | None = None) -> None:
        import multiprocessing

        self._flag = multiprocessing.get_context("spawn").RawValue("b", 0)
        self._notified = False
        self._installed = False
        self._previous_handler: Any = None
        self._previous_flag: Any = None
        self._previous_controller: EscapeShutdown | None = None
        self._closing = threading.Event()
        self._on_request = on_request
        self._stream = stream if stream is not None else sys.stdin
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._terminal_state: list[Any] | None = None

    @property
    def requested(self) -> bool:
        requested = bool(self._flag.value)
        if requested and not self._notified:
            self._notified = True
            if self._on_request is not None:
                self._on_request()
        return requested

    def request(self) -> None:
        """
        Request shutdown once from an ordinary control surface.
        """
        self._flag.value = 1
        _ = self.requested

    def _interrupt(self, _number: int, _frame: Any) -> None:
        if self._flag.value:
            raise ForcedShutdown()
        self._flag.value = 1

    def start(self) -> bool:
        """
        Install SIGINT independently of stdin; listen for Escape on POSIX terminals.
        """
        global _ACTIVE
        if not self._installed and threading.current_thread() is threading.main_thread():
            self._previous_controller, _ACTIVE = _ACTIVE, self
            self._previous_flag = install_cancellation_flag(self._flag)
            self._previous_handler = signal.signal(signal.SIGINT, self._interrupt)
            self._installed = True

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

    def restore_terminal(self) -> None:
        """
        Restore input mode while keeping SIGINT ownership during final saving.
        """

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

    def stop(self) -> None:
        """
        Restore the terminal, original signal handler, and previous cancellation scope.
        """
        global _ACTIVE
        self.restore_terminal()
        if self._installed:
            signal.signal(signal.SIGINT, self._previous_handler)
            install_cancellation_flag(self._previous_flag)
            _ACTIVE = self._previous_controller
            self._installed = False

    def _listen(self) -> None:
        fd = self._fd
        if fd is None:
            return
        while not self._closing.is_set() and not self._flag.value:
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
