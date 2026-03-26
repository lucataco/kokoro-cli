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
import signal
import struct
import sys
import time
from pathlib import Path

import numpy as np

KOKORO_DIR = Path.home() / ".kokoro"
SOCKET_PATH = KOKORO_DIR / "kokoro.sock"
PID_PATH = KOKORO_DIR / "kokoro.pid"


def get_socket_path() -> Path:
    return SOCKET_PATH


def get_pid_path() -> Path:
    return PID_PATH


def is_daemon_running() -> bool:
    """Check if the daemon is running by testing the socket."""
    import socket

    if not SOCKET_PATH.exists():
        return False
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(str(SOCKET_PATH))
        sock.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


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
    """Handle a single TTS request from a client."""
    try:
        # Read the JSON request line
        line = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not line:
            return

        request = json.loads(line.decode("utf-8").strip())
        text = request.get("text", "")
        voice = request.get("voice", "af_sky")
        speed = request.get("speed", 1.0)
        lang = request.get("lang", "a")

        if not text:
            # Send empty response
            writer.write(struct.pack(">I", 0))
            await writer.drain()
            return

        from kokoro_cli.engine import generate

        for audio_chunk in generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang,
        ):
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

    log_fd = open(log_path, "w")
    devnull = open(os.devnull, "r")

    # Spawn `kokoro serve` as a detached subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "kokoro_cli.server"],
        stdin=devnull,
        stdout=log_fd,
        stderr=log_fd,
        start_new_session=True,  # detach from parent's process group
    )

    devnull.close()
    log_fd.close()


# Entry point for the daemon subprocess
if __name__ == "__main__":
    run_server()
