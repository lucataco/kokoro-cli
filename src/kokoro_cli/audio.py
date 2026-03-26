"""Streaming audio playback using sounddevice.

Provides low-latency audio output by streaming generated audio chunks
directly to the speakers as they are produced by the TTS engine.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf

from kokoro_cli.config import SAMPLE_RATE

# Sub-chunk size for writes: 0.2s at 24kHz.
# Keeps Ctrl+C responsive — Python can only handle KeyboardInterrupt
# between blocking calls, so smaller writes = faster interrupt response.
_WRITE_CHUNK_SAMPLES = 4800


def play_audio_blocking(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Play an audio array and block until finished."""
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


class StreamPlayer:
    """Streams audio chunks to speakers with minimal latency.

    Uses a sounddevice OutputStream that accepts numpy arrays and plays
    them sequentially without gaps between chunks. Writes are broken into
    small sub-chunks (~0.2s) so that Ctrl+C is handled promptly.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None

    def start(self) -> None:
        """Open the audio output stream."""
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        self._stream.start()

    def write(self, audio: np.ndarray) -> None:
        """Write an audio chunk to the stream for playback.

        The audio is written in small sub-chunks so that KeyboardInterrupt
        (Ctrl+C) can be caught between writes rather than blocking for the
        entire duration of a large chunk.

        Args:
            audio: 1D float32 numpy array of audio samples.
        """
        if self._stream is None:
            self.start()

        # Ensure correct shape: (n_samples, 1) for mono
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Write in small sub-chunks for responsive Ctrl+C handling
        offset = 0
        while offset < len(audio):
            end = min(offset + _WRITE_CHUNK_SAMPLES, len(audio))
            self._stream.write(audio[offset:end])
            offset = end

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def save_audio(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save audio to a WAV file.

    Args:
        audio: 1D or 2D numpy array of audio samples.
        path: Output file path (e.g., "output.wav").
        sample_rate: Sample rate in Hz.
    """
    sf.write(path, audio, sample_rate)
