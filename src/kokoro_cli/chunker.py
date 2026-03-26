"""Text chunking for streaming TTS generation.

Splits long text into sentence-bounded chunks to enable streaming playback.
Each chunk is small enough for Kokoro to process quickly (~500 tokens max),
but large enough to maintain natural prosody.
"""

import re
import sys


# Approximate max characters per chunk. Kokoro handles ~510 phonemized tokens,
# which is roughly 1500-2000 characters of English text. We use a conservative
# limit to ensure we never exceed the model's context window.
MAX_CHUNK_CHARS = 1500

# Sentence-ending pattern: period, exclamation, question mark, or ellipsis
# followed by whitespace or end of string.
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

# Secondary split on semicolons, colons, or em-dashes if sentences are too long
CLAUSE_END = re.compile(r"(?<=[;:—])\s+")

# Tertiary split on commas if clauses are still too long
COMMA_END = re.compile(r"(?<=,)\s+")


def split_at_pattern(text: str, pattern: re.Pattern, max_chars: int) -> list[str]:
    """Split text at pattern boundaries, respecting max_chars."""
    parts = pattern.split(text)
    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if not current:
            current = part
        elif len(current) + len(part) + 1 <= max_chars:
            current = current + " " + part
        else:
            chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks suitable for TTS generation.

    Strategy:
    1. Split at sentence boundaries first
    2. If any chunk is still too long, split at clause boundaries
    3. If still too long, split at commas
    4. If still too long, hard-split at max_chars (last resort)
    """
    text = text.strip()
    if not text:
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    if len(text) <= max_chars:
        return [text]

    # Stage 1: Split at sentence boundaries
    chunks = split_at_pattern(text, SENTENCE_END, max_chars)

    # Stage 2: Split oversized chunks at clause boundaries
    refined = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            refined.append(chunk)
        else:
            refined.extend(split_at_pattern(chunk, CLAUSE_END, max_chars))
    chunks = refined

    # Stage 3: Split at commas
    refined = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            refined.append(chunk)
        else:
            refined.extend(split_at_pattern(chunk, COMMA_END, max_chars))
    chunks = refined

    # Stage 4: Hard split anything still oversized
    final = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            # Find last space before max_chars
            split_pos = chunk.rfind(" ", 0, max_chars)
            if split_pos == -1:
                split_pos = max_chars
            final.append(chunk[:split_pos].strip())
            chunk = chunk[split_pos:].strip()
        if chunk:
            final.append(chunk)

    return final


def read_stdin() -> str:
    """Read all text from stdin (non-interactive)."""
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def read_file(path: str) -> str:
    """Read text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
