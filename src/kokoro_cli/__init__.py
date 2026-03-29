"""kokoro-cli: Fast local TTS using Kokoro-82M on Apple Silicon via MLX."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kokoro-cli")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for development
