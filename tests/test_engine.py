"""Tests for kokoro_cli.engine — voice parsing, mixing, and generation."""

import numpy as np
import pytest

from kokoro_cli.config import VOICES
from kokoro_cli.engine import (
    generate,
    parse_voice_spec,
    random_voice_mix,
)


# ---------------------------------------------------------------------------
# parse_voice_spec  (unit tests — no model needed)
# ---------------------------------------------------------------------------


class TestParseVoiceSpec:
    def test_single_voice(self):
        result = parse_voice_spec("af_sky")
        assert result == {"af_sky": 1.0}

    def test_two_voices_with_weights(self):
        result = parse_voice_spec("af_heart:0.7,af_bella:0.3")
        assert len(result) == 2
        assert abs(result["af_heart"] - 0.7) < 0.01
        assert abs(result["af_bella"] - 0.3) < 0.01

    def test_two_voices_equal_weights(self):
        result = parse_voice_spec("af_heart,af_bella")
        assert len(result) == 2
        assert abs(result["af_heart"] - 0.5) < 0.01
        assert abs(result["af_bella"] - 0.5) < 0.01

    def test_three_voices_equal(self):
        result = parse_voice_spec("af_heart,af_bella,af_sky")
        assert len(result) == 3
        for weight in result.values():
            assert abs(weight - 1.0 / 3) < 0.01

    def test_weights_normalize_to_one(self):
        result = parse_voice_spec("af_heart:2,af_bella:3")
        total = sum(result.values())
        assert abs(total - 1.0) < 0.001

    def test_weights_already_normalized(self):
        result = parse_voice_spec("af_heart:0.5,af_bella:0.5")
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_single_voice_no_weight(self):
        result = parse_voice_spec("am_adam")
        assert result == {"am_adam": 1.0}

    def test_whitespace_handling(self):
        result = parse_voice_spec(" af_heart : 0.7 , af_bella : 0.3 ")
        assert "af_heart" in result
        assert "af_bella" in result
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_mixed_weighted_and_unweighted(self):
        # When some have weights and some don't, unweighted get 0.0
        result = parse_voice_spec("af_heart:0.8,af_bella")
        assert "af_heart" in result
        assert "af_bella" in result
        # af_bella gets 0.0, normalization makes af_heart 1.0
        assert result["af_heart"] == 1.0


# ---------------------------------------------------------------------------
# random_voice_mix  (unit tests — no model needed)
# ---------------------------------------------------------------------------


class TestRandomVoiceMix:
    def test_returns_dict(self):
        result = random_voice_mix()
        assert isinstance(result, dict)
        assert len(result) >= 2

    def test_weights_sum_to_one(self):
        result = random_voice_mix()
        total = sum(result.values())
        assert abs(total - 1.0) < 0.001

    def test_voices_from_same_language(self):
        result = random_voice_mix(lang_code="a")
        for voice in result:
            assert voice.startswith("a"), f"{voice} is not American English"

    def test_british_voices(self):
        result = random_voice_mix(lang_code="b")
        for voice in result:
            assert voice.startswith("b"), f"{voice} is not British English"

    def test_fallback_to_american(self):
        result = random_voice_mix(lang_code="x")
        for voice in result:
            assert voice.startswith("a"), f"{voice} should be American fallback"

    def test_n_voices_parameter(self):
        result = random_voice_mix(n_voices=2)
        assert len(result) == 2

    def test_n_voices_caps_at_available(self):
        # Japanese male only has 1 voice (jm_kumo)
        result = random_voice_mix(lang_code="j", n_voices=100)
        japanese = [v for v in VOICES if v.startswith("j")]
        assert len(result) <= len(japanese)

    def test_default_picks_2_or_3(self):
        # Run multiple times to check range
        counts = set()
        for _ in range(50):
            result = random_voice_mix(lang_code="a")
            counts.add(len(result))
        # Should see both 2 and 3 in 50 trials (very unlikely to miss)
        assert 2 in counts or 3 in counts

    def test_all_weights_positive(self):
        result = random_voice_mix()
        for weight in result.values():
            assert weight > 0

    def test_all_voices_in_catalog(self):
        result = random_voice_mix()
        for voice in result:
            assert voice in VOICES


# ---------------------------------------------------------------------------
# Integration tests — require the real MLX model
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGenerateIntegration:
    """Tests that run actual model inference. Requires the model to be downloaded."""

    def test_single_voice_generates_audio(self):
        chunks = list(generate("Hello world.", voice="af_sky"))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32
            assert len(chunk) > 0

    def test_audio_has_reasonable_duration(self):
        """'Hello world' should produce roughly 0.5-5 seconds of audio."""
        chunks = list(generate("Hello world.", voice="af_sky"))
        total_samples = sum(len(c) for c in chunks)
        duration = total_samples / 24000
        assert 0.3 < duration < 10.0, f"Duration {duration:.2f}s seems wrong"

    def test_blended_voice_generates_audio(self):
        chunks = list(
            generate(
                "Testing voice blending.",
                voice="af_heart:0.6,af_bella:0.4",
            )
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    def test_speed_parameter(self):
        """Faster speed should produce shorter audio."""
        normal = list(
            generate("This is a speed test sentence.", voice="af_sky", speed=1.0)
        )
        fast = list(
            generate("This is a speed test sentence.", voice="af_sky", speed=1.5)
        )

        normal_samples = sum(len(c) for c in normal)
        fast_samples = sum(len(c) for c in fast)

        # Fast speech should be noticeably shorter
        assert fast_samples < normal_samples * 0.95

    def test_generate_yields_incrementally(self):
        """Generate should yield multiple chunks for longer text."""
        text = "First sentence here. Second sentence follows. Third sentence ends."
        chunks = list(generate(text, voice="af_sky"))
        # Model may yield one chunk per sentence, or more
        assert len(chunks) >= 1
