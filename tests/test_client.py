"""Tests for kokoro_cli.client — socket client helpers."""

import socket
import struct
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kokoro_cli.client import _recv_exact, wait_for_daemon


# ---------------------------------------------------------------------------
# _recv_exact
# ---------------------------------------------------------------------------


class TestRecvExact:
    def test_receives_exact_bytes(self):
        """Should receive exactly n bytes from a socket."""
        # Create a socket pair for testing
        server_sock, client_sock = socket.socketpair()
        try:
            server_sock.sendall(b"hello world")
            result = _recv_exact(client_sock, 5)
            assert result == b"hello"
            # Remaining data still in buffer
            result2 = _recv_exact(client_sock, 6)
            assert result2 == b" world"
        finally:
            server_sock.close()
            client_sock.close()

    def test_returns_none_on_closed_connection(self):
        """Should return None if connection closes before n bytes received."""
        server_sock, client_sock = socket.socketpair()
        try:
            server_sock.sendall(b"hi")
            server_sock.close()
            # Ask for more bytes than available
            result = _recv_exact(client_sock, 10)
            assert result is None
        finally:
            client_sock.close()

    def test_handles_partial_reads(self):
        """Should accumulate data from multiple recv() calls."""
        server_sock, client_sock = socket.socketpair()
        try:
            # Send data in small pieces with a slight delay
            def send_slowly():
                for byte in b"abcdef":
                    server_sock.sendall(bytes([byte]))
                    time.sleep(0.01)

            t = threading.Thread(target=send_slowly)
            t.start()
            result = _recv_exact(client_sock, 6)
            t.join()
            assert result == b"abcdef"
        finally:
            server_sock.close()
            client_sock.close()

    def test_empty_request(self):
        """Requesting 0 bytes should return empty bytes."""
        server_sock, client_sock = socket.socketpair()
        try:
            result = _recv_exact(client_sock, 0)
            assert result == b""
        finally:
            server_sock.close()
            client_sock.close()


# ---------------------------------------------------------------------------
# wait_for_daemon
# ---------------------------------------------------------------------------


class TestWaitForDaemon:
    @patch("kokoro_cli.client.daemon_available", return_value=False)
    def test_timeout_when_not_running(self, mock_avail):
        """Should return False when daemon never starts."""
        result = wait_for_daemon(timeout=0.3, poll_interval=0.1)
        assert result is False

    @patch("kokoro_cli.client.daemon_available", return_value=True)
    def test_immediate_return_when_running(self, mock_avail):
        """Should return True immediately if daemon is already running."""
        start = time.time()
        result = wait_for_daemon(timeout=5.0, poll_interval=0.1)
        elapsed = time.time() - start
        assert result is True
        assert elapsed < 1.0  # Should be near-instant

    @patch("kokoro_cli.client.daemon_available")
    def test_waits_then_succeeds(self, mock_avail):
        """Should poll until daemon becomes available."""
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            return call_count >= 3  # Becomes available on 3rd check

        mock_avail.side_effect = side_effect
        result = wait_for_daemon(timeout=5.0, poll_interval=0.05)
        assert result is True
        assert call_count >= 3
