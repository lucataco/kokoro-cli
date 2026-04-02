"""Socket client for communicating with the kokoro daemon.

Connects to the daemon's Unix domain socket, sends a TTS request
as JSON, and receives streamed audio chunks as binary frames.
"""

import json
import socket
import struct
import time
from collections.abc import Iterator

import numpy as np

from kokoro_cli.config import SAMPLE_RATE
from kokoro_cli.server import SOCKET_PATH, is_daemon_running


def daemon_available() -> bool:
    """Check if the kokoro daemon is running and accepting connections."""
    return is_daemon_running()


def generate_via_daemon(
    text: str,
    voice: str = "af_sky",
    speed: float = 1.0,
    lang: str = "a",
    timeout: float = 30.0,
) -> Iterator[np.ndarray]:
    """Send a TTS request to the daemon and yield audio chunks.

    Args:
        text: Text to synthesize.
        voice: Voice name or weighted spec.
        speed: Speech speed multiplier.
        lang: Language code.
        timeout: Socket timeout in seconds.

    Yields:
        numpy float32 arrays of audio samples at 24kHz.

    Raises:
        ConnectionError: If the daemon is not running or connection fails.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(str(SOCKET_PATH))

        # Send JSON request (newline-terminated)
        request = json.dumps(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "lang": lang,
            }
        )
        sock.sendall((request + "\n").encode("utf-8"))

        # Receive audio chunks
        while True:
            # Read 4-byte size header
            size_data = _recv_exact(sock, 4)
            if size_data is None:
                break

            chunk_size = struct.unpack(">I", size_data)[0]
            if chunk_size == 0:
                # End-of-stream sentinel
                break

            # Read audio data
            audio_data = _recv_exact(sock, chunk_size)
            if audio_data is None:
                break

            audio = np.frombuffer(audio_data, dtype=np.float32)
            yield audio

    except (ConnectionRefusedError, FileNotFoundError) as e:
        raise ConnectionError(f"Daemon not available: {e}") from e
    finally:
        sock.close()


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes from a socket.

    Returns None if the connection is closed before n bytes are received.
    """
    data = bytearray()
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except (ConnectionResetError, BrokenPipeError):
            return None
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def cancel_generation() -> bool:
    """Send a cancel request to the daemon to stop the current generation.

    Returns:
        True if the cancel was acknowledged, False on error.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect(str(SOCKET_PATH))
        request = json.dumps({"cancel": True}) + "\n"
        sock.sendall(request.encode("utf-8"))
        # Read the ack
        data = sock.recv(1024)
        if data:
            response = json.loads(data.decode("utf-8").strip())
            return response.get("status") == "cancelled"
        return False
    except (ConnectionRefusedError, FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    finally:
        sock.close()


def wait_for_daemon(timeout: float = 30.0, poll_interval: float = 0.2) -> bool:
    """Wait for the daemon to become ready.

    Args:
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between connection attempts.

    Returns:
        True if daemon became ready, False if timed out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if daemon_available():
            return True
        time.sleep(poll_interval)
    return False
