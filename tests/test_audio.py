"""Tests for kokoro_cli.audio — playback and file saving."""

import numpy as np
import pytest
import soundfile as sf

from kokoro_cli.audio import StreamPlayer, play_audio_blocking, save_audio
from kokoro_cli.config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# save_audio
# ---------------------------------------------------------------------------


class TestSaveAudio:
    def test_saves_valid_wav(self, sample_audio, tmp_wav_path):
        save_audio(sample_audio, tmp_wav_path)
        data, sr = sf.read(tmp_wav_path)
        assert sr == SAMPLE_RATE
        assert len(data) == len(sample_audio)

    def test_saved_audio_dtype(self, sample_audio, tmp_wav_path):
        save_audio(sample_audio, tmp_wav_path)
        data, _ = sf.read(tmp_wav_path, dtype="float32")
        np.testing.assert_array_almost_equal(data, sample_audio, decimal=4)

    def test_saves_noisy_audio(self, noisy_audio, tmp_wav_path):
        save_audio(noisy_audio, tmp_wav_path)
        data, sr = sf.read(tmp_wav_path, dtype="float32")
        assert sr == SAMPLE_RATE
        assert len(data) == len(noisy_audio)
        # Values should be close (WAV may have tiny rounding)
        np.testing.assert_array_almost_equal(data, noisy_audio, decimal=4)

    def test_custom_sample_rate(self, sample_audio, tmp_wav_path):
        save_audio(sample_audio, tmp_wav_path, sample_rate=16000)
        _, sr = sf.read(tmp_wav_path)
        assert sr == 16000

    def test_2d_audio(self, sample_audio_2d, tmp_wav_path):
        save_audio(sample_audio_2d, tmp_wav_path)
        data, sr = sf.read(tmp_wav_path)
        assert sr == SAMPLE_RATE
        assert len(data) == 24000


# ---------------------------------------------------------------------------
# StreamPlayer
# ---------------------------------------------------------------------------


class TestStreamPlayer:
    def test_context_manager(self, mock_sounddevice):
        mock_sd, mock_stream = mock_sounddevice
        with StreamPlayer() as player:
            assert player._stream is not None
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_write_reshapes_1d(self, mock_sounddevice):
        mock_sd, mock_stream = mock_sounddevice
        audio = np.zeros(100, dtype=np.float32)
        with StreamPlayer() as player:
            player.write(audio)
        # Should have been called (may be multiple sub-chunks)
        assert mock_stream.write.called

    def test_write_casts_to_float32(self, mock_sounddevice):
        mock_sd, mock_stream = mock_sounddevice
        audio = np.zeros(100, dtype=np.float64)
        with StreamPlayer() as player:
            player.write(audio)
        # Check that the written data was float32
        call_args = mock_stream.write.call_args_list
        for call in call_args:
            written = call[0][0]
            assert written.dtype == np.float32

    def test_write_sub_chunking(self, mock_sounddevice):
        """Audio longer than _WRITE_CHUNK_SAMPLES should be split into sub-chunks."""
        mock_sd, mock_stream = mock_sounddevice
        # 10000 samples > _WRITE_CHUNK_SAMPLES (4800), so should be 3 writes
        audio = np.zeros(10000, dtype=np.float32)
        with StreamPlayer() as player:
            player.write(audio)
        # Should have been called multiple times (10000/4800 = 3 sub-chunks)
        assert mock_stream.write.call_count == 3

    def test_stop_without_start(self, mock_sounddevice):
        """Stopping without starting should not raise."""
        player = StreamPlayer()
        player.stop()  # Should not raise

    def test_auto_start_on_write(self, mock_sounddevice):
        """Writing without explicit start should auto-start the stream."""
        mock_sd, mock_stream = mock_sounddevice
        player = StreamPlayer()
        assert player._stream is None
        audio = np.zeros(100, dtype=np.float32)
        player.write(audio)
        assert player._stream is not None
        player.stop()


# ---------------------------------------------------------------------------
# play_audio_blocking
# ---------------------------------------------------------------------------


class TestPlayAudioBlocking:
    def test_calls_sounddevice(self, mock_sounddevice):
        mock_sd, _ = mock_sounddevice
        audio = np.zeros(100, dtype=np.float32)
        play_audio_blocking(audio)
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    def test_custom_sample_rate(self, mock_sounddevice):
        mock_sd, _ = mock_sounddevice
        audio = np.zeros(100, dtype=np.float32)
        play_audio_blocking(audio, sample_rate=16000)
        _, kwargs = mock_sd.play.call_args
        assert kwargs.get("samplerate") == 16000
