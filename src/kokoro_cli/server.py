"""Daemon server for kokoro-cli.

Keeps the MLX model and spaCy G2P pipeline loaded in memory,
accepting TTS requests over a Unix domain socket. This eliminates
the ~2.7s cold start on every invocation.

Protocol:
    Request:  JSON line (newline-terminated)
              {"text": "hello", "voice": "af_sky", "speed": 1.0, "lang": "a"}

    Response: Sequence of binary audio frames:
              [4 bytes: uint32 big-endian chunk size][N bytes: float32 PCM]
              ...
              [4 bytes: 0x00000000]  (end-of-stream sentinel)
"""

import asyncio
import json
import os
import re
import signal
import struct
import sys
import time
from pathlib import Path

import numpy as np
import setproctitle

from kokoro_cli.config import LANG_MAP, VOICES

# Speed limits for input validation
MIN_SPEED = 0.1
MAX_SPEED = 5.0

# Regex for voice spec: name or name:weight, comma-separated
_VOICE_SPEC_RE = re.compile(
    r"^[a-z][a-z]_[a-z0-9]+(:[0-9]*\.?[0-9]+)?"
    r"(,[a-z][a-z]_[a-z0-9]+(:[0-9]*\.?[0-9]+)?)*$"
)

KOKORO_DIR = Path.home() / ".kokoro"
SOCKET_PATH = KOKORO_DIR / "kokoro.sock"
PID_PATH = KOKORO_DIR / "kokoro.pid"

# Shared cancel event: when set, the current generation should stop ASAP.
# A cancel request from any client sets this; it is cleared at the start
# of each new generation request.
_cancel_event = asyncio.Event()


def get_socket_path() -> Path:
    return SOCKET_PATH


def get_pid_path() -> Path:
    return PID_PATH


def is_daemon_running() -> bool:
    """Check if the daemon is running by testing the socket."""
    import socket

    if not SOCKET_PATH.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(1.0)
        sock.connect(str(SOCKET_PATH))
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False
    finally:
        sock.close()


def _cleanup():
    """Remove socket and PID files."""
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    if PID_PATH.exists():
        PID_PATH.unlink()


def stop_daemon() -> bool:
    """Stop a running daemon. Returns True if stopped, False if not running."""
    if not PID_PATH.exists():
        # Try socket-based detection as fallback
        if not is_daemon_running():
            return False

    if PID_PATH.exists():
        try:
            pid = int(PID_PATH.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit
            for _ in range(20):  # 2 seconds max
                try:
                    os.kill(pid, 0)  # Check if still alive
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
        except (ProcessLookupError, ValueError):
            pass

    _cleanup()
    return True


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single TTS or cancel request from a client.

    Cancel protocol:
        A client can send ``{"cancel": true}`` to immediately stop the
        current generation. The server sets a shared cancel event which
        the generation loop checks between audio chunks.
    """
    try:
        # Read the JSON request line
        line = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not line:
            return

        request = json.loads(line.decode("utf-8").strip())

        # --- Handle cancel request ---
        if request.get("cancel"):
            _cancel_event.set()
            # Send a JSON ack so the caller knows it was received
            ack = json.dumps({"status": "cancelled"}).encode("utf-8") + b"\n"
            writer.write(ack)
            await writer.drain()
            return

        text = request.get("text", "")
        voice = request.get("voice", "af_sky")
        speed = request.get("speed", 1.0)
        lang = request.get("lang", "a")

        # --- Input validation ---
        # Validate voice spec format
        if not isinstance(voice, str) or not _VOICE_SPEC_RE.match(voice):
            # Check if it's a simple voice name in the catalog
            if voice not in VOICES:
                writer.write(struct.pack(">I", 0))
                await writer.drain()
                return

        # Clamp speed to safe range
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(MIN_SPEED, min(MAX_SPEED, speed))

        # Validate language code
        if not isinstance(lang, str) or lang not in LANG_MAP:
            lang = "a"  # default to American English

        if not text:
            # Send empty response
            writer.write(struct.pack(">I", 0))
            await writer.drain()
            return

        # Clear cancel event at the start of a new generation
        _cancel_event.clear()

        from kokoro_cli.engine import generate

        for audio_chunk in generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang,
        ):
            # Check if a cancel was requested between chunks
            if _cancel_event.is_set():
                break

            # Send audio chunk: [4-byte size][audio bytes]
            audio_bytes = audio_chunk.tobytes()
            writer.write(struct.pack(">I", len(audio_bytes)))
            writer.write(audio_bytes)
            await writer.drain()

        # End-of-stream sentinel
        writer.write(struct.pack(">I", 0))
        await writer.drain()

    except (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError):
        pass
    except Exception as e:
        print(f"Error handling request: {e}", file=sys.stderr)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError):
            pass


async def _run_server():
    """Main async server loop."""
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup()

    # Write PID file
    PID_PATH.write_text(str(os.getpid()))

    # Warm up the model and pipeline
    print("Loading model...", file=sys.stderr)
    t0 = time.time()

    from kokoro_cli.engine import warmup

    warmup()

    elapsed = time.time() - t0
    print(f"Model ready in {elapsed:.1f}s", file=sys.stderr)

    # Start the Unix socket server
    server = await asyncio.start_unix_server(_handle_client, path=str(SOCKET_PATH))
    # Make socket accessible
    SOCKET_PATH.chmod(0o600)

    print(f"Listening on {SOCKET_PATH}", file=sys.stderr)
    print("Ready.", file=sys.stderr)

    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        pass
    finally:
        server.close()
        await server.wait_closed()
        _cleanup()
        print("Server stopped.", file=sys.stderr)


def run_server():
    """Entry point for the foreground server."""
    setproctitle.setproctitle("kokoro-daemon")
    loop = asyncio.new_event_loop()

    def _shutdown(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...", file=sys.stderr)
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        loop.run_until_complete(_run_server())
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()
        loop.close()


def run_server_daemon():
    """Spawn the daemon as a detached subprocess.

    Uses subprocess.Popen instead of os.fork() because forking after
    Metal/MLX initialization breaks the MTLCompilerService connection.
    The subprocess starts fresh and loads the model independently.
    """
    import subprocess

    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    log_path = KOKORO_DIR / "kokoro.log"

    with open(log_path, "w") as log_fd, open(os.devnull, "r") as devnull:
        # Spawn `kokoro serve` as a detached subprocess
        subprocess.Popen(
            [sys.executable, "-m", "kokoro_cli.server"],
            stdin=devnull,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,  # detach from parent's process group
        )


# Entry point for the daemon subprocess
if __name__ == "__main__":
    run_server()
