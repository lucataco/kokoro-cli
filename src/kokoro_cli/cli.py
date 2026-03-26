"""Command-line interface for kokoro-cli.

Usage:
    echo "Hello world" | kokoro
    kokoro --text "Hello world"
    kokoro --file document.txt
    kokoro --voice af_heart --speed 1.2 --text "Fast speech"
    kokoro --random-voice --text "Surprise me"
    kokoro --voice "af_heart:0.7,af_bella:0.3" --text "Blended voice"
    kokoro --list-voices
"""

import sys

import click
import numpy as np

from kokoro_cli.config import (
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    SAMPLE_RATE,
    VOICES,
    get_voice_info,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--text",
    "-t",
    type=str,
    default=None,
    help="Text to synthesize. If omitted, reads from stdin.",
)
@click.option(
    "--file",
    "-f",
    "filepath",
    type=click.Path(exists=True),
    default=None,
    help="Read text from a file.",
)
@click.option(
    "--voice",
    "-v",
    type=str,
    default=DEFAULT_VOICE,
    show_default=True,
    help='Voice name or weighted mix (e.g., "af_heart:0.7,af_bella:0.3").',
)
@click.option(
    "--random-voice",
    "-r",
    is_flag=True,
    default=False,
    help="Use a random mix of 2-3 voices.",
)
@click.option(
    "--speed",
    "-s",
    type=float,
    default=DEFAULT_SPEED,
    show_default=True,
    help="Speech speed multiplier.",
)
@click.option(
    "--lang",
    "-l",
    type=str,
    default="a",
    show_default=True,
    help="Language code (a=American, b=British, j=Japanese, z=Chinese, e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save audio to a WAV file instead of playing.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=DEFAULT_MODEL,
    show_default=True,
    help="HuggingFace model ID or local path.",
)
@click.option(
    "--list-voices",
    is_flag=True,
    default=False,
    help="List all available voices and exit.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress progress output.",
)
def main(
    text: str | None,
    filepath: str | None,
    voice: str,
    random_voice: bool,
    speed: float,
    lang: str,
    output: str | None,
    model: str,
    list_voices: bool,
    quiet: bool,
) -> None:
    """Kokoro TTS — fast local text-to-speech on Apple Silicon.

    Converts text to speech using the Kokoro-82M model via MLX.
    Reads from stdin by default, or use --text / --file.

    \b
    Examples:
        echo "Hello world" | kokoro
        kokoro --text "Hello world"
        kokoro --file document.txt
        kokoro --random-voice --text "Surprise me"
    """
    if list_voices:
        _print_voices()
        return

    # Resolve input text
    input_text = _resolve_input(text, filepath)
    if not input_text:
        click.echo(
            "Error: No input text. Use --text, --file, or pipe via stdin.", err=True
        )
        sys.exit(1)

    # Resolve voice
    if random_voice:
        from kokoro_cli.engine import random_voice_mix

        voice_mix = random_voice_mix(lang_code=lang)
        voice_spec = ",".join(f"{v}:{w:.2f}" for v, w in voice_mix.items())
        if not quiet:
            click.echo(f"Random voice mix: {voice_spec}", err=True)
    else:
        voice_spec = voice

    # Chunk the text
    from kokoro_cli.chunker import chunk_text

    chunks = chunk_text(input_text)

    if not quiet:
        n_chars = len(input_text)
        n_chunks = len(chunks)
        click.echo(
            f"Generating: {n_chars} chars, {n_chunks} chunk(s), "
            f"voice={voice_spec}, speed={speed}x",
            err=True,
        )

    # Generate and play/save
    if output:
        _generate_to_file(chunks, voice_spec, speed, model, lang, output, quiet)
    else:
        _generate_and_stream(chunks, voice_spec, speed, model, lang, quiet)


def _resolve_input(text: str | None, filepath: str | None) -> str:
    """Resolve input text from --text, --file, or stdin."""
    if text:
        return text

    if filepath:
        from kokoro_cli.chunker import read_file

        return read_file(filepath)

    # Try stdin
    from kokoro_cli.chunker import read_stdin

    return read_stdin()


def _generate_and_stream(
    chunks: list[str],
    voice: str,
    speed: float,
    model_path: str,
    lang: str,
    quiet: bool,
) -> None:
    """Generate audio from chunks and stream to speakers."""
    from kokoro_cli.audio import StreamPlayer
    from kokoro_cli.engine import generate

    try:
        with StreamPlayer() as player:
            for i, chunk in enumerate(chunks):
                if not quiet:
                    # Show chunk progress on stderr
                    preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
                    click.echo(f"  [{i + 1}/{len(chunks)}] {preview}", err=True)

                for audio_chunk in generate(
                    text=chunk,
                    voice=voice,
                    speed=speed,
                    model_path=model_path,
                    lang_code=lang,
                ):
                    player.write(audio_chunk)

        if not quiet:
            click.echo("Done.", err=True)

    except KeyboardInterrupt:
        if not quiet:
            click.echo("\nInterrupted.", err=True)
        sys.exit(130)


def _generate_to_file(
    chunks: list[str],
    voice: str,
    speed: float,
    model_path: str,
    lang: str,
    output_path: str,
    quiet: bool,
) -> None:
    """Generate audio from chunks and save to a file."""
    from kokoro_cli.audio import save_audio
    from kokoro_cli.engine import generate

    all_audio: list[np.ndarray] = []

    try:
        for i, chunk in enumerate(chunks):
            if not quiet:
                preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
                click.echo(f"  [{i + 1}/{len(chunks)}] {preview}", err=True)

            for audio_chunk in generate(
                text=chunk,
                voice=voice,
                speed=speed,
                model_path=model_path,
                lang_code=lang,
            ):
                all_audio.append(audio_chunk)

    except KeyboardInterrupt:
        if not quiet:
            click.echo("\nInterrupted. Saving partial audio...", err=True)

    if all_audio:
        combined = np.concatenate(all_audio)
        save_audio(combined, output_path, SAMPLE_RATE)
        if not quiet:
            duration = len(combined) / SAMPLE_RATE
            click.echo(
                f"Saved {duration:.1f}s of audio to {output_path}",
                err=True,
            )
    else:
        click.echo("Error: No audio generated.", err=True)
        sys.exit(1)


def _print_voices() -> None:
    """Print all available voices in a formatted table."""
    click.echo(f"{'Voice':<20} {'Language':<22} {'Gender':<8}")
    click.echo("-" * 50)

    current_lang = None
    for v in VOICES:
        info = get_voice_info(v)
        if info["lang"] != current_lang:
            if current_lang is not None:
                click.echo()  # blank line between language groups
            current_lang = info["lang"]
        click.echo(f"{v:<20} {info['lang']:<22} {info['gender']:<8}")

    click.echo(f"\nTotal: {len(VOICES)} voices")


if __name__ == "__main__":
    main()
