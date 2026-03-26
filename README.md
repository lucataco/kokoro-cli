# kokoro-cli

Fast local text-to-speech CLI for Apple Silicon. Runs [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) natively on the Metal GPU via [MLX](https://github.com/Blaizzy/mlx-audio), with streaming audio playback and voice mixing.

**29x realtime** on M4 Pro — generates 13.7 seconds of audio in 0.47 seconds.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (phonemizer fallback)

```bash
brew install espeak-ng
```

## Install

**Development (editable):**

```bash
cd kokoro-cli
uv sync
uv run kokoro --text "Hello world"
```

**Global tool (available everywhere):**

```bash
cd kokoro-cli
uv tool install -e . --force --python 3.12
kokoro --text "Hello world"
```

First `uv sync` installs all dependencies including the spaCy English model (~13MB) used for phonemization. The Kokoro model itself (`mlx-community/Kokoro-82M-bf16`, ~164MB) is downloaded automatically on first run and cached in `~/.cache/huggingface/`.

## Usage

### Basic

```bash
# Stream to speakers (default)
kokoro --text "Hello world"

# Pipe from stdin
echo "Hello world" | kokoro

# Read from file
kokoro --file document.txt

# Save to WAV instead of playing
kokoro --text "Hello world" --output hello.wav
```

### Voice selection

```bash
# Use a specific voice (default: af_sky)
kokoro --voice af_heart --text "Hello"

# Male voice
kokoro --voice am_adam --text "Hello"

# List all 54 voices
kokoro --list-voices
```

### Voice mixing

Blend multiple voices with weighted averaging for unique tones.

```bash
# Weighted mix (70% af_heart, 30% af_bella)
kokoro --voice "af_heart:0.7,af_bella:0.3" --text "Blended voice"

# Equal mix
kokoro --voice "af_heart,af_bella" --text "Fifty-fifty blend"

# Random mix of 2-3 voices from the same language
kokoro --random-voice --text "Surprise me"
```

### Speed and language

```bash
# 1.2x speed (good for digesting long content)
kokoro --speed 1.2 --text "Faster speech"

# British English
kokoro --lang b --voice bf_emma --text "Cheerio"
```

### Verbose mode

Output is quiet by default. Use `-v` to see progress details.

```bash
# Show voice mix, chunk info, and progress
kokoro -v --random-voice --text "Hello"

# Default (quiet) — no output except errors
kokoro --text "Hello"
```

### All options

```
Usage: kokoro [OPTIONS]

Options:
  -t, --text TEXT     Text to synthesize. If omitted, reads from stdin.
  -f, --file PATH     Read text from a file.
  --voice TEXT        Voice name or weighted mix.  [default: af_sky]
  -r, --random-voice  Use a random mix of 2-3 voices.
  -s, --speed FLOAT   Speech speed multiplier.  [default: 1.0]
  -l, --lang TEXT     Language code.  [default: a]
  -o, --output PATH   Save audio to a WAV file instead of playing.
  -m, --model TEXT    HuggingFace model ID or local path.
  --list-voices       List all available voices and exit.
  -v, --verbose       Show progress output (voice mix, chunk info, etc.).
  --no-daemon         Skip daemon, use direct mode.
  -h, --help          Show this message and exit.

Daemon:
  kokoro serve   Start the TTS daemon
  kokoro stop    Stop the TTS daemon
```

## Daemon mode

The model and G2P pipeline take ~2.7s to load on first use. The daemon keeps them warm in memory so subsequent calls are near-instant.

### How it works

On first invocation, `kokoro` auto-starts a background daemon that loads the model and listens on a Unix socket (`~/.kokoro/kokoro.sock`). Subsequent calls connect to the daemon instead of loading the model from scratch.

```
First call:     auto-starts daemon (~2.7s), then generates via daemon (~0.5s)
Second call:    connects to warm daemon (~0.5s)
```

### Manual daemon management

```bash
# Start in foreground (useful for debugging, shows logs)
kokoro serve

# Start in background (detaches from terminal)
kokoro serve --daemon

# Stop the daemon
kokoro stop

# Skip daemon, use direct mode
kokoro --no-daemon --text "Hello"
```

### Performance comparison

| Mode | Time to audio | Notes |
|------|--------------|-------|
| Direct (cold) | ~3.5s | Loads model + G2P every time |
| Daemon (first call) | ~3.4s | Auto-starts daemon, then uses it |
| Daemon (warm) | ~0.5s | **7x faster** — model already in memory |

Daemon state is stored in `~/.kokoro/`:
- `kokoro.sock` — Unix domain socket
- `kokoro.pid` — daemon process ID
- `kokoro.log` — daemon log output

## Voices

54 voices across 9 languages. Voice names encode language and gender: first letter = language, second letter = gender (`f` = female, `m` = male).

| Code | Language             | Female | Male |
|------|----------------------|--------|------|
| `a`  | American English     | 11     | 9    |
| `b`  | British English      | 4      | 4    |
| `j`  | Japanese             | 4      | 1    |
| `z`  | Mandarin Chinese     | 4      | 4    |
| `e`  | Spanish              | 1      | 2    |
| `f`  | French               | 1      | --   |
| `h`  | Hindi                | 2      | 2    |
| `i`  | Italian              | 1      | 1    |
| `p`  | Brazilian Portuguese | 1      | 2    |

Run `kokoro --list-voices` for the full list.

## How it works

```
kokoro --text "Hello"
        |
        |  Is daemon running?
        |
   ┌────┴─── Yes ──────────────────┐
   |                                |
   |  client.py ──Unix socket──> server.py
   |    Send JSON request           Warm model generates audio
   |    Receive audio chunks        Streams chunks back
   |                                |
   └────┬─── No ───────────────────┘
        |
        |  Auto-start daemon, or --no-daemon for direct mode
        |
   engine.py     MLX model generates audio via Metal GPU
        |
   chunker.py    Text split at sentence boundaries (~1500 chars/chunk)
        |
   audio.py      sounddevice streams chunks to speakers as generated
        |
   Speaker output (or WAV file with --output)
```

- **Daemon**: Background process keeps model + G2P warm. Communicates via Unix socket with binary audio streaming.
- **True streaming**: Audio chunks are yielded incrementally and played as soon as generated — no waiting for full text processing.
- **Chunking**: Long text is split at sentence boundaries so each chunk fits within Kokoro's ~510 phoneme token window.
- **Voice mixing**: Individual voice tensors are loaded and combined via weighted averaging before generation.

## Project structure

```
src/kokoro_cli/
  __init__.py     Package metadata
  __main__.py     python -m kokoro_cli entry point
  cli.py          Click CLI group, TTS/serve/stop commands, auto-daemon logic
  config.py       Defaults, 54-voice catalog, language/gender metadata
  engine.py       MLX model loading, voice parsing/mixing, streaming generation
  chunker.py      Sentence-boundary text splitting, stdin/file readers
  audio.py        StreamPlayer for real-time playback, WAV file saving
  server.py       Asyncio Unix socket daemon, model warmup, binary audio protocol
  client.py       Socket client, daemon detection, audio chunk receiver
```

## Performance

Benchmarked on MacBook Pro M4 Pro (48GB):

| Metric | Value |
|--------|-------|
| Model load (cached) | ~1s |
| G2P pipeline init | ~1.5s |
| Generation speed | 29x realtime / 436 chars/sec |
| Daemon warm call | ~0.5s total |
| Direct cold call | ~3.5s total |
| Model size | ~164MB (bf16) |
| Audio format | 24kHz mono float32 |

## License

Apache 2.0 — matching the [Kokoro-82M model license](https://huggingface.co/hexgrad/Kokoro-82M).
