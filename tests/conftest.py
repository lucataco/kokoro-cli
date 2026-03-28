"""Shared fixtures for kokoro-cli tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Sample text fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def short_text():
    """A short sentence for quick tests."""
    return "Hello world, this is a test."


@pytest.fixture
def medium_text():
    """A paragraph-length text with multiple sentences."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
        "The five boxing wizards jump quickly."
    )


@pytest.fixture
def long_text():
    """A multi-paragraph text block (~3000 chars) for chunking/throughput tests."""
    paragraph = (
        "Artificial intelligence has transformed the way we interact with technology. "
        "From natural language processing to computer vision, AI systems are becoming "
        "increasingly capable of performing tasks that once required human intelligence. "
        "Machine learning algorithms can now recognize speech, translate languages, "
        "diagnose diseases, and even generate creative content like art and music. "
        "The rapid advancement of these technologies raises important questions about "
        "ethics, privacy, and the future of work. As we continue to develop more "
        "sophisticated AI systems, it is crucial that we consider the broader "
        "implications of these technologies on society and ensure that they are "
        "developed and deployed responsibly."
    )
    # Repeat to get ~3000 chars
    return " ".join([paragraph] * 5)


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audio():
    """One second of silence at 24kHz as float32."""
    return np.zeros(24000, dtype=np.float32)


@pytest.fixture
def sample_audio_2d():
    """One second of silence at 24kHz as float32, shape (24000, 1)."""
    return np.zeros((24000, 1), dtype=np.float32)


@pytest.fixture
def noisy_audio():
    """One second of random noise at 24kHz as float32."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.5, 0.5, size=24000).astype(np.float32)


# ---------------------------------------------------------------------------
# File fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def tmp_wav_path(tmp_path):
    """Provide a temporary WAV file path."""
    return str(tmp_path / "test_output.wav")


@pytest.fixture
def tmp_text_file(tmp_path):
    """Create a temporary text file with sample content."""
    path = tmp_path / "input.txt"
    path.write_text("This is text from a file.", encoding="utf-8")
    return str(path)


@pytest.fixture
def tmp_unicode_file(tmp_path):
    """Create a temporary text file with unicode content."""
    path = tmp_path / "unicode.txt"
    path.write_text(
        "Caf\u00e9 na\u00efve r\u00e9sum\u00e9. \u00dc\u00f6\u00e4\u00df. \u65e5\u672c\u8a9e\u30c6\u30b9\u30c8\u3002",
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sounddevice(monkeypatch):
    """Mock sounddevice to avoid needing audio hardware."""
    mock_stream = MagicMock()
    mock_stream.start = MagicMock()
    mock_stream.stop = MagicMock()
    mock_stream.close = MagicMock()
    mock_stream.write = MagicMock()

    mock_sd = MagicMock()
    mock_sd.OutputStream.return_value = mock_stream
    mock_sd.play = MagicMock()
    mock_sd.wait = MagicMock()

    monkeypatch.setattr("kokoro_cli.audio.sd", mock_sd)
    return mock_sd, mock_stream
