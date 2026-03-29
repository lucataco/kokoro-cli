"""Command-line interface for kokoro-cli.

Usage:
    echo "Hello world" | kokoro
    kokoro --text "Hello world"
    kokoro --file document.txt
    kokoro --voice af_heart --speed 1.2 --text "Fast speech"
    kokoro --random-voice --text "Surprise me"
    kokoro --voice "af_heart:0.7,af_bella:0.3" --text "Blended voice"
    kokoro --list-voices
    kokoro serve            # Start daemon in foreground
    kokoro serve --daemon   # Start daemon in background
    kokoro stop             # Stop the daemon
"""

import platform
import sys

import click
import numpy as np
import setproctitle

from kokoro_cli.config import (
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    LANG_MAP,
    SAMPLE_RATE,
    VOICES,
    get_voice_info,
)


def _check_platform():
    """Verify we're running on Apple Silicon. Exit with a clear message if not."""
    if platform.system() != "Darwin":
        click.echo(
            "Error: kokoro-cli requires macOS with Apple Silicon (M1+).\n"
            "This tool uses MLX for GPU-accelerated inference which is only "
            "available on Apple Silicon Macs.",
            err=True,
        )
        sys.exit(1)
    machine = platform.machine()
    if machine != "arm64":
        click.echo(
            f"Error: kokoro-cli requires Apple Silicon (arm64), "
            f"but this Mac is {machine}.\n"
            "MLX requires Apple Silicon (M1 or later) for GPU acceleration.",
            err=True,
        )
        sys.exit(1)


def _validate_voice_spec(voice_spec: str) -> None:
    """Validate that a voice spec string references known voices.

    Raises SystemExit with error message for unknown voices.
    """
    parts = [p.strip() for p in voice_spec.split(",")]
    for part in parts:
        name = part.split(":")[0].strip()
        if name not in VOICES:
            click.echo(
                f"Error: Unknown voice '{name}'. "
                f"Run 'kokoro --list-voices' to see available voices.",
                err=True,
            )
            sys.exit(1)


class KokoroGroup(click.Group):
    """Custom Click group that treats unknown subcommands as the default TTS command.

    This lets `kokoro --text "hello"` work without a subcommand, while also
    supporting `kokoro serve` and `kokoro stop` as explicit subcommands.
    """

    def parse_args(self, ctx, args):
        # If the first arg is a known subcommand, let Click handle it normally.
        # Otherwise, prepend "tts" so it routes to the default TTS command.
        if args and args[0] in self.commands:
            return super().parse_args(ctx, args)

        # For --help / -h: show the group help (which includes TTS + daemon info)
        if (not args and sys.stdin.isatty()) or "--help" in args or "-h" in args:
            click.echo(ctx.get_help())
            ctx.exit(0)

        # Route to the default "tts" command
        args = ["tts"] + list(args)
        return super().parse_args(ctx, args)

    def format_help(self, ctx, formatter):
        """Show the default TTS command help as the main help."""
        tts_cmd = self.commands.get("tts")
        if tts_cmd:
            with click.Context(
                tts_cmd, info_name=ctx.info_name, parent=ctx.parent
            ) as sub_ctx:
                tts_cmd.format_help(sub_ctx, formatter)
        formatter.write("\nDaemon:\n")
        formatter.write(
            "  kokoro serve   Start the TTS daemon (keeps model in memory)\n"
        )
        formatter.write("  kokoro stop    Stop the TTS daemon\n")
        formatter.write(
            "\nRun 'kokoro serve --help' or 'kokoro stop --help' for details.\n"
        )


@click.group(
    cls=KokoroGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.pass_context
def main(ctx):
    """Kokoro TTS — fast local text-to-speech on Apple Silicon."""
    _check_platform()
    setproctitle.setproctitle("kokoro")

    # If invoked with no args and no piped stdin, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command(
    name="tts", hidden=True, context_settings={"help_option_names": ["-h", "--help"]}
)
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
    type=click.FloatRange(min=0.1, max=5.0),
    default=DEFAULT_SPEED,
    show_default=True,
    help="Speech speed multiplier (0.1-5.0).",
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
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show progress output (voice mix, chunk info, etc.).",
)
@click.option(
    "--no-daemon",
    is_flag=True,
    default=False,
    help="Skip daemon, use direct mode.",
)
def tts_command(
    text: str | None,
    filepath: str | None,
    voice: str,
    random_voice: bool,
    speed: float,
    lang: str,
    output: str | None,
    model: str,
    list_voices: bool,
    verbose: bool,
    no_daemon: bool,
) -> None:
    """Generate speech from text.

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

    # Validate language code
    if lang not in LANG_MAP:
        click.echo(
            f"Error: Unknown language code '{lang}'. "
            f"Valid codes: {', '.join(sorted(LANG_MAP.keys()))}",
            err=True,
        )
        sys.exit(1)

    # Validate voice (unless using random)
    if not random_voice:
        _validate_voice_spec(voice)

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
        if verbose:
            click.echo(f"Random voice mix: {voice_spec}", err=True)
    else:
        voice_spec = voice

    # Chunk the text
    from kokoro_cli.chunker import chunk_text

    chunks = chunk_text(input_text)

    if verbose:
        n_chars = len(input_text)
        n_chunks = len(chunks)
        click.echo(
            f"Generating: {n_chars} chars, {n_chunks} chunk(s), "
            f"voice={voice_spec}, speed={speed}x",
            err=True,
        )

    # Decide: daemon or direct mode
    use_daemon = False
    if not no_daemon:
        use_daemon = _ensure_daemon(verbose)

    # Generate and play/save
    if output:
        _generate_to_file(
            chunks, voice_spec, speed, model, lang, output, verbose, use_daemon
        )
    else:
        _generate_and_stream(
            chunks, voice_spec, speed, model, lang, verbose, use_daemon
        )


@main.command()
@click.option(
    "--daemon",
    "-d",
    is_flag=True,
    default=False,
    help="Run in the background (detach from terminal).",
)
def serve(daemon: bool) -> None:
    """Start the kokoro TTS daemon.

    Loads the model into memory and listens for TTS requests on a Unix socket.
    Subsequent `kokoro` commands connect to this daemon for near-instant speech.

    \b
    Examples:
        kokoro serve            # Foreground (shows logs)
        kokoro serve --daemon   # Background (detach)
    """
    from kokoro_cli.server import is_daemon_running, run_server, run_server_daemon

    if is_daemon_running():
        click.echo("Daemon is already running.", err=True)
        sys.exit(0)

    if daemon:
        click.echo("Starting daemon in background...", err=True)
        run_server_daemon()
        # Wait for it to be ready
        from kokoro_cli.client import wait_for_daemon

        if wait_for_daemon(timeout=30.0):
            click.echo("Daemon started.", err=True)
        else:
            click.echo("Error: Daemon failed to start within 30s.", err=True)
            sys.exit(1)
    else:
        # Foreground — blocks until Ctrl+C
        run_server()


@main.command()
def stop() -> None:
    """Stop the kokoro TTS daemon."""
    from kokoro_cli.server import stop_daemon

    if stop_daemon():
        click.echo("Daemon stopped.", err=True)
    else:
        click.echo("No daemon running.", err=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_daemon(verbose: bool) -> bool:
    """Ensure the daemon is running. Auto-starts if needed.

    Returns True if daemon is available, False to fall back to direct mode.
    """
    from kokoro_cli.client import daemon_available, wait_for_daemon

    if daemon_available():
        if verbose:
            click.echo("Using daemon.", err=True)
        return True

    # Auto-start daemon in background
    if verbose:
        click.echo("Starting daemon...", err=True)

    from kokoro_cli.server import run_server_daemon

    run_server_daemon()

    if wait_for_daemon(timeout=30.0):
        if verbose:
            click.echo("Daemon ready.", err=True)
        return True
    else:
        if verbose:
            click.echo("Daemon startup timed out, using direct mode.", err=True)
        return False


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


def _generate_audio_chunks(
    chunk: str,
    voice: str,
    speed: float,
    model_path: str,
    lang: str,
    use_daemon: bool,
):
    """Yield audio chunks from either daemon or direct engine."""
    if use_daemon:
        from kokoro_cli.client import generate_via_daemon

        yield from generate_via_daemon(text=chunk, voice=voice, speed=speed, lang=lang)
    else:
        from kokoro_cli.engine import generate

        yield from generate(
            text=chunk,
            voice=voice,
            speed=speed,
            model_path=model_path,
            lang_code=lang,
        )


def _generate_and_stream(
    chunks: list[str],
    voice: str,
    speed: float,
    model_path: str,
    lang: str,
    verbose: bool,
    use_daemon: bool,
) -> None:
    """Generate audio from chunks and stream to speakers."""
    from kokoro_cli.audio import StreamPlayer

    try:
        with StreamPlayer() as player:
            for i, chunk in enumerate(chunks):
                if verbose:
                    preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
                    click.echo(f"  [{i + 1}/{len(chunks)}] {preview}", err=True)

                for audio_chunk in _generate_audio_chunks(
                    chunk, voice, speed, model_path, lang, use_daemon
                ):
                    player.write(audio_chunk)

        if verbose:
            click.echo("Done.", err=True)

    except KeyboardInterrupt:
        if verbose:
            click.echo("\nInterrupted.", err=True)
        sys.exit(130)


def _generate_to_file(
    chunks: list[str],
    voice: str,
    speed: float,
    model_path: str,
    lang: str,
    output_path: str,
    verbose: bool,
    use_daemon: bool,
) -> None:
    """Generate audio from chunks and save to a file."""
    from kokoro_cli.audio import save_audio

    all_audio: list[np.ndarray] = []

    try:
        for i, chunk in enumerate(chunks):
            if verbose:
                preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
                click.echo(f"  [{i + 1}/{len(chunks)}] {preview}", err=True)

            for audio_chunk in _generate_audio_chunks(
                chunk, voice, speed, model_path, lang, use_daemon
            ):
                all_audio.append(audio_chunk)

    except KeyboardInterrupt:
        if verbose:
            click.echo("\nInterrupted. Saving partial audio...", err=True)

    if all_audio:
        combined = np.concatenate(all_audio)
        save_audio(combined, output_path, SAMPLE_RATE)
        if verbose:
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
