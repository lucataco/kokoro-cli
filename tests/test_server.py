"""Tests for kokoro_cli.server — daemon helpers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from kokoro_cli.server import (
    KOKORO_DIR,
    PID_PATH,
    SOCKET_PATH,
    _cleanup,
    get_pid_path,
    get_socket_path,
    is_daemon_running,
    stop_daemon,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPaths:
    def test_socket_path(self):
        path = get_socket_path()
        assert isinstance(path, Path)
        assert path.name == "kokoro.sock"
        assert path.parent.name == ".kokoro"

    def test_pid_path(self):
        path = get_pid_path()
        assert isinstance(path, Path)
        assert path.name == "kokoro.pid"
        assert path.parent.name == ".kokoro"

    def test_kokoro_dir_in_home(self):
        assert KOKORO_DIR == Path.home() / ".kokoro"


# ---------------------------------------------------------------------------
# _cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_removes_socket_and_pid(self, tmp_path, monkeypatch):
        sock = tmp_path / "kokoro.sock"
        pid = tmp_path / "kokoro.pid"
        sock.touch()
        pid.touch()

        monkeypatch.setattr("kokoro_cli.server.SOCKET_PATH", sock)
        monkeypatch.setattr("kokoro_cli.server.PID_PATH", pid)

        _cleanup()

        assert not sock.exists()
        assert not pid.exists()

    def test_cleanup_missing_files(self, tmp_path, monkeypatch):
        """Cleanup should not raise if files don't exist."""
        sock = tmp_path / "kokoro.sock"
        pid = tmp_path / "kokoro.pid"

        monkeypatch.setattr("kokoro_cli.server.SOCKET_PATH", sock)
        monkeypatch.setattr("kokoro_cli.server.PID_PATH", pid)

        # Should not raise
        _cleanup()


# ---------------------------------------------------------------------------
# is_daemon_running
# ---------------------------------------------------------------------------


class TestIsDaemonRunning:
    def test_false_when_no_socket(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "kokoro_cli.server.SOCKET_PATH", tmp_path / "nonexistent.sock"
        )
        assert is_daemon_running() is False

    def test_false_when_socket_exists_but_no_listener(self, tmp_path, monkeypatch):
        """A socket file without a listener should return False."""
        sock = tmp_path / "kokoro.sock"
        sock.touch()
        monkeypatch.setattr("kokoro_cli.server.SOCKET_PATH", sock)
        assert is_daemon_running() is False


# ---------------------------------------------------------------------------
# stop_daemon
# ---------------------------------------------------------------------------


class TestStopDaemon:
    def test_returns_false_when_nothing_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kokoro_cli.server.PID_PATH", tmp_path / "kokoro.pid")
        monkeypatch.setattr(
            "kokoro_cli.server.SOCKET_PATH", tmp_path / "nonexistent.sock"
        )
        assert stop_daemon() is False

    def test_cleans_up_stale_pid_file(self, tmp_path, monkeypatch):
        """If PID file exists but process is gone, should still clean up."""
        pid_file = tmp_path / "kokoro.pid"
        sock_file = tmp_path / "kokoro.sock"
        pid_file.write_text("999999999")  # Very unlikely to be a real PID
        sock_file.touch()

        monkeypatch.setattr("kokoro_cli.server.PID_PATH", pid_file)
        monkeypatch.setattr("kokoro_cli.server.SOCKET_PATH", sock_file)

        result = stop_daemon()
        assert result is True
        assert not pid_file.exists()
        assert not sock_file.exists()
