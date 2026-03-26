"""MLX-based Kokoro TTS engine.

Handles model loading, voice mixing, and audio generation using the
mlx-audio library for native Apple Silicon acceleration.
"""

import contextlib
import io
import logging
import os
import random
import time
from collections.abc import Iterator

import numpy as np

from kokoro_cli.config import (
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    VOICES,
)

# Suppress noisy output from huggingface_hub and mlx_audio before they're imported.
# HF_HUB_DISABLE_PROGRESS_BARS kills the tqdm "Fetching 63 files" bars.
# HF_HUB_DISABLE_TELEMETRY kills the "Download complete" line.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Lazy-loaded model singleton
_model = None
_model_path = None


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily redirect stdout to devnull.

    Used to silence print() calls inside mlx_audio internals
    (e.g., "Creating new KokoroPipeline for language: a").
    Redirects both the Python-level sys.stdout and the underlying
    file descriptor to catch all output.
    """
    import sys

    old_sys_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_sys_stdout


def load_model(model_path: str = DEFAULT_MODEL):
    """Load the Kokoro MLX model (cached singleton).

    The model is loaded once and reused for subsequent calls.
    First call downloads the model from HuggingFace if not cached locally.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        The loaded mlx-audio model instance.
    """
    global _model, _model_path

    if _model is not None and _model_path == model_path:
        return _model

    from mlx_audio.tts.utils import load_model as mlx_load_model

    with _suppress_stdout():
        _model = mlx_load_model(model_path)

    _model_path = model_path
    return _model


def parse_voice_spec(voice_spec: str) -> dict[str, float]:
    """Parse a voice specification string into a weighted voice dict.

    Supports:
        "af_sky"                     -> {"af_sky": 1.0}
        "af_heart:0.7,af_bella:0.3" -> {"af_heart": 0.7, "af_bella": 0.3}
        "af_heart,af_bella"          -> {"af_heart": 0.5, "af_bella": 0.5}

    Args:
        voice_spec: Voice specification string.

    Returns:
        Dict mapping voice names to their weights (0.0-1.0).
    """
    parts = [p.strip() for p in voice_spec.split(",")]

    if len(parts) == 1 and ":" not in parts[0]:
        return {parts[0]: 1.0}

    voices = {}
    has_weights = any(":" in p for p in parts)

    if has_weights:
        for part in parts:
            if ":" in part:
                name, weight = part.rsplit(":", 1)
                voices[name.strip()] = float(weight.strip())
            else:
                voices[part.strip()] = 0.0  # will be filled in
    else:
        # Equal weights
        weight = 1.0 / len(parts)
        for part in parts:
            voices[part.strip()] = weight

    # Normalize weights to sum to 1.0
    total = sum(voices.values())
    if total > 0:
        voices = {k: v / total for k, v in voices.items()}

    return voices


def random_voice_mix(lang_code: str = "a", n_voices: int = 0) -> dict[str, float]:
    """Generate a random voice mix from the same language group.

    Picks 2-3 voices from the specified language and assigns random weights.

    Args:
        lang_code: Language code prefix (e.g., "a" for American English).
        n_voices: Number of voices to mix (0 = random 2-3).

    Returns:
        Dict mapping voice names to their weights.
    """
    available = [v for v in VOICES if v.startswith(lang_code)]
    if not available:
        available = [v for v in VOICES if v.startswith("a")]

    if n_voices <= 0:
        n_voices = random.randint(2, min(3, len(available)))

    n_voices = min(n_voices, len(available))
    chosen = random.sample(available, n_voices)

    # Random weights that sum to 1.0
    raw_weights = [random.random() for _ in range(n_voices)]
    total = sum(raw_weights)
    weights = {v: w / total for v, w in zip(chosen, raw_weights)}

    return weights


def _blend_voices_on_pipeline(pipeline, voice_weights: dict[str, float]) -> str:
    """Load and blend multiple voice tensors with weighted averaging.

    Loads individual voice tensors via the pipeline, computes a weighted sum,
    and stores the result in the pipeline's voice cache.

    Args:
        pipeline: The KokoroPipeline instance.
        voice_weights: Dict mapping voice names to weights (sum to 1.0).

    Returns:
        Cache key string for the blended voice.
    """
    import mlx.core as mx

    packs = []
    weights = []

    for voice_name, weight in voice_weights.items():
        with _suppress_stdout():
            pack = pipeline.load_single_voice(voice_name)
        packs.append(pack)
        weights.append(weight)

    if len(packs) == 1:
        cache_key = next(iter(voice_weights.keys()))
        return cache_key

    # Weighted average: sum(weight_i * pack_i)
    blended = mx.zeros_like(packs[0])
    for pack, weight in zip(packs, weights):
        blended = blended + weight * pack

    # Store in pipeline cache with a unique key
    cache_key = "_blend_" + "_".join(f"{v}_{w:.2f}" for v, w in voice_weights.items())
    pipeline.voices[cache_key] = blended
    return cache_key


def generate(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    model_path: str = DEFAULT_MODEL,
    lang_code: str = "a",
) -> Iterator[np.ndarray]:
    """Generate audio from text using the Kokoro MLX model.

    Yields audio chunks as numpy arrays (float32, 24kHz mono).
    Each chunk corresponds to a segment of the input text.

    Args:
        text: Input text to synthesize.
        voice: Voice name or weighted spec (e.g., "af_heart:0.7,af_bella:0.3").
        speed: Speech speed multiplier (default 1.0).
        model_path: HuggingFace model ID or local path.
        lang_code: Language code for the model (e.g., "a" for American English).

    Yields:
        numpy float32 arrays of audio samples at 24kHz.
    """
    model = load_model(model_path)
    voice_weights = parse_voice_spec(voice)

    if len(voice_weights) > 1:
        # Multi-voice mix: we need to drive the pipeline directly
        # because model.generate() resets pipeline.voices = {} each call
        with _suppress_stdout():
            pipeline = model._get_pipeline(lang_code)
            pipeline.voices = {}  # match what model.generate does
            cache_key = _blend_voices_on_pipeline(pipeline, voice_weights)

        # Collect all audio chunks (generation is fast, ~30x realtime)
        audio_chunks = []
        with _suppress_stdout():
            for _graphemes, _phonemes, audio in pipeline(
                text, voice=cache_key, speed=speed
            ):
                audio = audio[0] if audio.ndim > 1 else audio
                audio = np.array(audio, dtype=np.float32)
                audio_chunks.append(audio)

        yield from audio_chunks
    else:
        # Single voice: use the standard model.generate path
        effective_voice = next(iter(voice_weights.keys()))

        # Collect results inside _suppress_stdout to catch all internal prints
        # (e.g., "Creating new KokoroPipeline for language: a").
        # Generation is ~30x realtime so this adds negligible latency.
        audio_chunks = []
        with _suppress_stdout():
            for result in model.generate(
                text,
                voice=effective_voice,
                speed=speed,
                lang_code=lang_code,
            ):
                audio = result.audio
                if hasattr(audio, "tolist"):
                    audio = np.array(audio, dtype=np.float32)
                elif not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                else:
                    audio = audio.astype(np.float32)
                audio_chunks.append(audio)

        yield from audio_chunks
