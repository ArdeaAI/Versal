"""Escape is a cooperative terminal control, not an exception path."""

import os
import pty
import termios
import threading
import time

import pytest

from versal.utils.shutdown import EscapeShutdown


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


@pytest.mark.parametrize("mode", ["startup", "idle", "busy", "force", "cleanup"])
def test_process_group_interrupt_is_owned_by_parent(tmp_path, mode):
    """
    SIGINT reaches real spawned workers without killing them or printing tracebacks.
    """
    import select
    import signal
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root))
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "tests/fixtures/shutdown_probe.py"), str(tmp_path), mode],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )

    def line():
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 30)[0], "child did not reach interrupt boundary"
        return process.stdout.readline().decode().strip()

    try:
        assert line() == "READY"
        os.killpg(process.pid, signal.SIGINT)
        assert line() == "STOPPING"
        if mode == "cleanup":
            assert line() == "JOINING"
        if mode in {"force", "cleanup"}:
            os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == (130 if mode in {"force", "cleanup"} else 0), stderr.decode()
        assert b"Traceback" not in stderr and b"KeyboardInterrupt" not in stderr
        if mode == "busy":
            assert b'"worker_cancelled": [true, true]' in stdout
        for worker in tmp_path.glob("worker-*"):
            with pytest.raises(ProcessLookupError):
                os.kill(int(worker.name.removeprefix("worker-")), 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_signal_handler_restored_with_redirected_input(tmp_path):
    """
    Ctrl-C works without a terminal, and does not leak its handler to later callers.
    """
    import signal

    original = signal.getsignal(signal.SIGINT)
    with open(os.devnull) as stream:
        with EscapeShutdown(stream=stream) as shutdown:
            os.kill(os.getpid(), signal.SIGINT)
            assert shutdown.requested
        assert signal.getsignal(signal.SIGINT) == original


def test_interrupted_trial_saves_a_checkpoint_and_resumes(tmp_path):
    import json
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-u", str(root / "tests/fixtures/shutdown_trial_probe.py"), str(tmp_path)]
    environment = dict(os.environ, PYTHONPATH=str(root))
    with (tmp_path / "trial.log").open("w+") as log:
        process = subprocess.Popen(command + ["stop"], stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=environment, start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.02)
            assert (tmp_path / "ready").exists()
            os.killpg(process.pid, signal.SIGINT)
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stopped = json.loads((tmp_path / "stopped.json").read_text())
        assert stopped["status"] == "stopped" and stopped["tasks_attempted"] == 1
        assert (Path(stopped["run_dir"]) / "checkpoint.json").exists()
        resumed = subprocess.run(command + ["resume"], stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=environment, timeout=30)
        assert resumed.returncode == 0
        result = json.loads((tmp_path / "resumed.json").read_text())
        assert result["status"] == "done" and result["tasks_attempted"] == 2
        log.seek(0)
        output = log.read()
        assert "Traceback" not in output and "KeyboardInterrupt" not in output
