"""Tests for kokoro_cli.config — defaults and voice catalog validation."""

import re

from kokoro_cli.config import (
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    GENDER_MAP,
    LANG_MAP,
    SAMPLE_RATE,
    VOICES,
    get_voice_info,
    get_voices_by_gender,
    get_voices_by_lang,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 24000

    def test_default_speed(self):
        assert DEFAULT_SPEED == 1.0

    def test_default_voice_in_catalog(self):
        assert DEFAULT_VOICE in VOICES

    def test_default_model_is_string(self):
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0


# ---------------------------------------------------------------------------
# Voice catalog
# ---------------------------------------------------------------------------


class TestVoiceCatalog:
    def test_voices_not_empty(self):
        assert len(VOICES) > 0

    def test_voices_count(self):
        """Expect 54 voices as documented."""
        assert len(VOICES) == 54

    def test_no_duplicates(self):
        assert len(VOICES) == len(set(VOICES))

    def test_naming_convention(self):
        """All voices should match [lang_code][gender_code]_name pattern."""
        pattern = re.compile(r"^[a-z][fm]_[a-z]+$")
        for voice in VOICES:
            assert pattern.match(voice), (
                f"Voice {voice!r} doesn't match naming convention"
            )

    def test_all_lang_codes_in_map(self):
        """Every language code used in VOICES should be in LANG_MAP."""
        lang_codes = {v[0] for v in VOICES}
        for code in lang_codes:
            assert code in LANG_MAP, f"Language code {code!r} not in LANG_MAP"

    def test_all_gender_codes_in_map(self):
        """Every gender code used in VOICES should be in GENDER_MAP."""
        gender_codes = {v[1] for v in VOICES if len(v) >= 2}
        for code in gender_codes:
            assert code in GENDER_MAP, f"Gender code {code!r} not in GENDER_MAP"

    def test_voices_are_sorted_by_language_group(self):
        """Voices within the same language prefix should be contiguous."""
        seen_langs = []
        current_lang = None
        for voice in VOICES:
            lang = voice[0]
            if lang != current_lang:
                assert lang not in seen_langs, (
                    f"Language {lang!r} appears non-contiguously in VOICES"
                )
                seen_langs.append(lang)
                current_lang = lang


# ---------------------------------------------------------------------------
# get_voice_info
# ---------------------------------------------------------------------------


class TestGetVoiceInfo:
    def test_known_voice(self):
        info = get_voice_info("af_sky")
        assert info["code"] == "af_sky"
        assert info["lang"] == "American English"
        assert info["gender"] == "Female"
        assert info["lang_code"] == "a"
        assert info["gender_code"] == "f"

    def test_male_voice(self):
        info = get_voice_info("am_adam")
        assert info["gender"] == "Male"
        assert info["lang"] == "American English"

    def test_british_voice(self):
        info = get_voice_info("bf_alice")
        assert info["lang"] == "British English"
        assert info["gender"] == "Female"

    def test_japanese_voice(self):
        info = get_voice_info("jf_alpha")
        assert info["lang"] == "Japanese"

    def test_short_code(self):
        info = get_voice_info("x")
        assert info["lang"] == "Unknown"
        assert info["gender"] == "Unknown"

    def test_empty_code(self):
        info = get_voice_info("")
        assert info["lang"] == "Unknown"

    def test_unknown_lang_code(self):
        info = get_voice_info("xf_test")
        assert info["lang"] == "Unknown"
        assert info["gender"] == "Female"


# ---------------------------------------------------------------------------
# get_voices_by_lang / get_voices_by_gender
# ---------------------------------------------------------------------------


class TestVoiceFiltering:
    def test_american_english_voices(self):
        voices = get_voices_by_lang("a")
        assert len(voices) > 0
        assert all(v.startswith("a") for v in voices)

    def test_british_english_voices(self):
        voices = get_voices_by_lang("b")
        assert len(voices) > 0
        assert all(v.startswith("b") for v in voices)

    def test_nonexistent_lang(self):
        voices = get_voices_by_lang("x")
        assert voices == []

    def test_female_voices(self):
        voices = get_voices_by_gender("f")
        assert len(voices) > 0
        assert all(v[1] == "f" for v in voices)

    def test_male_voices(self):
        voices = get_voices_by_gender("m")
        assert len(voices) > 0
        assert all(v[1] == "m" for v in voices)

    def test_all_voices_covered_by_gender(self):
        """Every voice should be either female or male."""
        female = set(get_voices_by_gender("f"))
        male = set(get_voices_by_gender("m"))
        all_voices = set(VOICES)
        assert female | male == all_voices

    def test_no_gender_overlap(self):
        """No voice should be both female and male."""
        female = set(get_voices_by_gender("f"))
        male = set(get_voices_by_gender("m"))
        assert female & male == set()
