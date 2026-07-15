"""Escape is a cooperative terminal control, not an exception path."""

import os
import pty
import termios
import threading
import time

from ardevo.utils.shutdown import EscapeShutdown


def test_manual_shutdown_request_is_idempotent() -> None:
    calls: list[bool] = []
    shutdown = EscapeShutdown(lambda: calls.append(True))

    shutdown.request()
    shutdown.request()

    assert shutdown.requested
    assert calls == [True]


def test_terminal_escape_requests_shutdown_but_arrow_key_does_not() -> None:
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    observed = threading.Event()
    stream = os.fdopen(os.dup(slave), "r")
    shutdown = EscapeShutdown(observed.set, stream=stream)
    restored = None
    try:
        assert shutdown.start()
        os.write(master, b"\x1b[A")
        time.sleep(0.1)
        assert not shutdown.requested

        os.write(master, b"\x1b")
        assert observed.wait(1.0)
        assert shutdown.requested
    finally:
        shutdown.stop()
        restored = termios.tcgetattr(slave)
        stream.close()
        os.close(master)
        os.close(slave)
    assert restored == original
