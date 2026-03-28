#!/usr/bin/env python3
"""Benchmarks for kokoro-cli TTS performance.

Measures:
  - Cold start time (model load + warmup)
  - Time to first audio (TTFA)
  - Throughput (chars/sec, realtime factor)
  - Memory usage (RSS delta during generation)

Usage:
    python -m benchmarks.bench_tts                     # Run all benchmarks
    python -m benchmarks.bench_tts --json              # JSON output
    python -m benchmarks.bench_tts --only cold_start   # Run one benchmark
    python -m benchmarks.bench_tts --only ttfa
    python -m benchmarks.bench_tts --only throughput
    python -m benchmarks.bench_tts --only memory
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

# Suppress noisy output during benchmarks
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from kokoro_cli.config import DEFAULT_VOICE, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Sample texts of varying lengths
# ---------------------------------------------------------------------------

TEXTS = {
    "tiny": "Hello world.",
    "short": (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs."
    ),
    "medium": (
        "Artificial intelligence has transformed the way we interact with "
        "technology. From natural language processing to computer vision, "
        "AI systems are becoming increasingly capable of performing tasks "
        "that once required human intelligence. Machine learning algorithms "
        "can now recognize speech, translate languages, diagnose diseases, "
        "and even generate creative content like art and music."
    ),
    "long": (
        "Artificial intelligence has transformed the way we interact with "
        "technology. From natural language processing to computer vision, "
        "AI systems are becoming increasingly capable of performing tasks "
        "that once required human intelligence. Machine learning algorithms "
        "can now recognize speech, translate languages, diagnose diseases, "
        "and even generate creative content like art and music. "
        "The rapid advancement of these technologies raises important questions "
        "about ethics, privacy, and the future of work. As we continue to "
        "develop more sophisticated AI systems, it is crucial that we consider "
        "the broader implications of these technologies on society and ensure "
        "that they are developed and deployed responsibly. "
        "Deep learning, a subset of machine learning, has been particularly "
        "transformative. Neural networks with many layers can learn hierarchical "
        "representations of data, enabling breakthroughs in image recognition, "
        "natural language understanding, and game playing. The success of models "
        "like GPT, BERT, and their successors has demonstrated the power of "
        "scaling both model size and training data. These foundation models "
        "can be fine-tuned for specific tasks, making AI more accessible and "
        "practical for a wide range of applications."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_memory_rss_mb() -> float:
    """Get current process RSS in megabytes using psutil."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: use resource module on macOS/Linux
        import resource

        # On macOS, ru_maxrss is in bytes; on Linux, it's in KB
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        else:
            return usage.ru_maxrss / 1024


def _reset_model():
    """Reset the engine's cached model singleton so we measure cold start."""
    import kokoro_cli.engine as engine

    engine._model = None
    engine._model_path = None
    engine._silenced = False
    gc.collect()


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    """Format rows as a simple ASCII table."""
    all_rows = [headers] + rows
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    lines = []
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows:
        lines.append("  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_cold_start() -> dict:
    """Measure model loading and warmup time."""
    print("\n=== Cold Start ===")

    _reset_model()
    gc.collect()

    # Measure model load
    t0 = time.perf_counter()
    from kokoro_cli.engine import load_model

    model = load_model()
    t_load = time.perf_counter() - t0

    # Measure warmup (first generation that triggers pipeline + G2P init)
    _reset_model()
    gc.collect()

    t0 = time.perf_counter()
    from kokoro_cli.engine import warmup

    warmup()
    t_total = time.perf_counter() - t0

    results = {
        "model_load_s": round(t_load, 3),
        "full_warmup_s": round(t_total, 3),
    }

    rows = [
        ["Model load", f"{t_load:.3f}s"],
        ["Full warmup (load + first gen)", f"{t_total:.3f}s"],
    ]
    print(_format_table(rows, ["Metric", "Time"]))

    return results


def bench_ttfa() -> dict:
    """Measure time to first audio chunk."""
    print("\n=== Time to First Audio (TTFA) ===")

    from kokoro_cli.engine import generate, load_model

    # Ensure model is loaded and warm
    load_model()

    results = {}
    rows = []

    for label, text in [("tiny", TEXTS["tiny"]), ("medium", TEXTS["medium"])]:
        times = []
        for _ in range(3):  # 3 runs for stability
            t0 = time.perf_counter()
            gen = generate(text, voice=DEFAULT_VOICE)
            first_chunk = next(gen)
            t_first = time.perf_counter() - t0
            times.append(t_first)
            # Drain remaining chunks
            for _ in gen:
                pass

        avg = sum(times) / len(times)
        best = min(times)
        results[f"ttfa_{label}_avg_ms"] = round(avg * 1000, 1)
        results[f"ttfa_{label}_best_ms"] = round(best * 1000, 1)
        rows.append(
            [
                f"{label} ({len(text)} chars)",
                f"{avg * 1000:.1f}ms",
                f"{best * 1000:.1f}ms",
            ]
        )

    print(_format_table(rows, ["Input", "Avg TTFA", "Best TTFA"]))

    return results


def bench_throughput() -> dict:
    """Measure generation throughput at different text lengths."""
    print("\n=== Throughput ===")

    from kokoro_cli.engine import generate, load_model

    # Ensure model is loaded
    load_model()

    results = {}
    rows = []

    for label, text in TEXTS.items():
        t0 = time.perf_counter()
        chunks = list(generate(text, voice=DEFAULT_VOICE))
        t_gen = time.perf_counter() - t0

        total_samples = sum(len(c) for c in chunks)
        audio_duration = total_samples / SAMPLE_RATE
        chars_per_sec = len(text) / t_gen if t_gen > 0 else 0
        rtf = audio_duration / t_gen if t_gen > 0 else 0

        results[f"throughput_{label}"] = {
            "chars": len(text),
            "gen_time_s": round(t_gen, 3),
            "audio_duration_s": round(audio_duration, 3),
            "chars_per_sec": round(chars_per_sec, 1),
            "realtime_factor": round(rtf, 1),
        }

        rows.append(
            [
                f"{label} ({len(text)}c)",
                f"{t_gen:.3f}s",
                f"{audio_duration:.2f}s",
                f"{chars_per_sec:.0f}",
                f"{rtf:.1f}x",
            ]
        )

    print(_format_table(rows, ["Input", "Gen Time", "Audio Dur", "Chars/s", "RTF"]))

    return results


def bench_memory() -> dict:
    """Measure memory usage during model load and generation."""
    print("\n=== Memory Usage ===")

    _reset_model()
    gc.collect()

    rss_baseline = _get_memory_rss_mb()

    # Load model
    from kokoro_cli.engine import load_model

    load_model()
    gc.collect()
    rss_model = _get_memory_rss_mb()

    # Generate audio to measure peak
    from kokoro_cli.engine import generate

    all_audio = []
    for chunk in generate(TEXTS["long"], voice=DEFAULT_VOICE):
        all_audio.append(chunk)
    gc.collect()
    rss_after_gen = _get_memory_rss_mb()

    results = {
        "rss_baseline_mb": round(rss_baseline, 1),
        "rss_model_loaded_mb": round(rss_model, 1),
        "rss_after_generation_mb": round(rss_after_gen, 1),
        "model_delta_mb": round(rss_model - rss_baseline, 1),
        "generation_delta_mb": round(rss_after_gen - rss_model, 1),
    }

    rows = [
        ["Baseline RSS", f"{rss_baseline:.1f} MB"],
        ["After model load", f"{rss_model:.1f} MB (+{rss_model - rss_baseline:.1f})"],
        [
            "After generation",
            f"{rss_after_gen:.1f} MB (+{rss_after_gen - rss_model:.1f})",
        ],
        ["Model memory delta", f"{rss_model - rss_baseline:.1f} MB"],
    ]
    print(_format_table(rows, ["Metric", "Value"]))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "cold_start": bench_cold_start,
    "ttfa": bench_ttfa,
    "throughput": bench_throughput,
    "memory": bench_memory,
}


def main():
    parser = argparse.ArgumentParser(description="kokoro-cli TTS benchmarks")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=list(BENCHMARKS.keys()),
        default=None,
        help="Run only a specific benchmark.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  kokoro-cli Benchmark Suite")
    print("=" * 60)
    print(f"  Voice: {DEFAULT_VOICE}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")

    # Show platform info
    import platform

    print(f"  Platform: {platform.platform()}")
    print(f"  Python: {platform.python_version()}")

    try:
        chip = platform.processor() or "unknown"
        print(f"  Processor: {chip}")
    except Exception:
        pass

    all_results = {}

    if args.only:
        bench_fn = BENCHMARKS[args.only]
        all_results[args.only] = bench_fn()
    else:
        for name, bench_fn in BENCHMARKS.items():
            all_results[name] = bench_fn()

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(all_results, indent=2))

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
