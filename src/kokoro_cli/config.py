"""Default configuration and voice catalog for kokoro-cli."""

# Model defaults
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "af_sky"
DEFAULT_SPEED = 1.0
SAMPLE_RATE = 24000

# Voice catalog: (code, gender, name, language)
# First letter: language (a=American, b=British, j=Japanese, z=Chinese,
#               e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese)
# Second letter: gender (f=female, m=male)

LANG_MAP = {
    "a": "American English",
    "b": "British English",
    "j": "Japanese",
    "z": "Mandarin Chinese",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "p": "Brazilian Portuguese",
}

GENDER_MAP = {
    "f": "Female",
    "m": "Male",
}

VOICES = [
    # American English - Female
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    # American English - Male
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    # British English - Female
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    # British English - Male
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    # Japanese - Female
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    # Japanese - Male
    "jm_kumo",
    # Mandarin Chinese - Female
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    # Mandarin Chinese - Male
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
    # Spanish - Female
    "ef_dora",
    # Spanish - Male
    "em_alex",
    "em_santa",
    # French - Female
    "ff_siwis",
    # Hindi - Female
    "hf_alpha",
    "hf_beta",
    # Hindi - Male
    "hm_omega",
    "hm_psi",
    # Italian - Female
    "if_sara",
    # Italian - Male
    "im_nicola",
    # Brazilian Portuguese - Female
    "pf_dora",
    # Brazilian Portuguese - Male
    "pm_alex",
    "pm_santa",
]


def get_voice_info(voice_code: str) -> dict:
    """Get metadata for a voice code."""
    if len(voice_code) < 2:
        return {"lang": "Unknown", "gender": "Unknown", "code": voice_code}
    lang_code = voice_code[0]
    gender_code = voice_code[1]
    return {
        "code": voice_code,
        "lang": LANG_MAP.get(lang_code, "Unknown"),
        "gender": GENDER_MAP.get(gender_code, "Unknown"),
        "lang_code": lang_code,
        "gender_code": gender_code,
    }


def get_voices_by_lang(lang_code: str) -> list[str]:
    """Get all voices for a given language code."""
    return [v for v in VOICES if v.startswith(lang_code)]


def get_voices_by_gender(gender_code: str) -> list[str]:
    """Get all voices matching a gender code (f or m)."""
    return [v for v in VOICES if len(v) >= 2 and v[1] == gender_code]
