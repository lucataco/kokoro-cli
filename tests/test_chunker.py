"""Tests for kokoro_cli.chunker — text splitting logic."""

import re

import pytest

from kokoro_cli.chunker import (
    CLAUSE_END,
    COMMA_END,
    MAX_CHUNK_CHARS,
    SENTENCE_END,
    chunk_text,
    read_file,
    split_at_pattern,
)


# ---------------------------------------------------------------------------
# split_at_pattern
# ---------------------------------------------------------------------------


class TestSplitAtPattern:
    """Tests for the low-level split_at_pattern helper."""

    def test_no_split_needed(self):
        text = "Hello world."
        result = split_at_pattern(text, SENTENCE_END, 100)
        assert result == ["Hello world."]

    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence."
        result = split_at_pattern(text, SENTENCE_END, 30)
        assert len(result) == 2
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."

    def test_merges_short_fragments(self):
        text = "A. B. C."
        result = split_at_pattern(text, SENTENCE_END, 100)
        # All fit in one chunk
        assert len(result) == 1
        assert result[0] == "A. B. C."

    def test_empty_string(self):
        result = split_at_pattern("", SENTENCE_END, 100)
        assert result == []

    def test_whitespace_only(self):
        result = split_at_pattern("   ", SENTENCE_END, 100)
        assert result == []

    def test_clause_pattern(self):
        text = "First clause; second clause: third part"
        result = split_at_pattern(text, CLAUSE_END, 25)
        assert len(result) >= 2

    def test_comma_pattern(self):
        text = "one, two, three, four, five"
        result = split_at_pattern(text, COMMA_END, 15)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    """Tests for the main chunk_text function."""

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   ") == []
        assert chunk_text("\n\t\n") == []

    def test_short_text_single_chunk(self):
        text = "Hello world."
        result = chunk_text(text)
        assert result == ["Hello world."]

    def test_short_text_at_limit(self):
        text = "x" * MAX_CHUNK_CHARS
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_sentence_boundary_splitting(self):
        # Create text that exceeds max_chars with sentence boundaries
        s1 = "A" * 800 + "."
        s2 = "B" * 800 + "."
        text = s1 + " " + s2
        result = chunk_text(text)
        assert len(result) == 2
        assert result[0] == s1
        assert result[1] == s2

    def test_exclamation_boundary(self):
        s1 = "Watch out! " * 50
        result = chunk_text(s1.strip(), max_chars=200)
        assert len(result) > 1
        # Each chunk should end with "out!" (possibly with more content)
        for chunk in result:
            assert len(chunk) <= 200

    def test_question_boundary(self):
        text = "Is this a test? Yes it is. Are we done? Not yet."
        result = chunk_text(text, max_chars=30)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 30

    def test_ellipsis_boundary(self):
        text = "First part\u2026 Second part\u2026 Third part."
        result = chunk_text(text, max_chars=20)
        assert len(result) >= 2

    def test_clause_fallback(self):
        # One very long "sentence" with clause boundaries
        clause = "word " * 40
        text = f"{clause}; {clause}; {clause}"
        result = chunk_text(text, max_chars=250)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 250

    def test_comma_fallback(self):
        # One long segment with only comma boundaries
        segment = "word " * 20
        text = f"{segment}, {segment}, {segment}"
        result = chunk_text(text, max_chars=120)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 120

    def test_hard_split_no_punctuation(self):
        # No punctuation at all, just a wall of text
        text = "word " * 500
        result = chunk_text(text.strip(), max_chars=100)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 100

    def test_hard_split_single_long_word(self):
        # A single word longer than max_chars (no spaces to split on)
        text = "a" * 2000
        result = chunk_text(text, max_chars=100)
        assert len(result) > 1
        # Hard split at max_chars since no space found
        for chunk in result:
            assert len(chunk) <= 100

    def test_whitespace_normalization(self):
        text = "Hello   world.\n\nThis   is\ta   test."
        result = chunk_text(text)
        assert len(result) == 1
        # Whitespace should be normalized to single spaces
        assert "  " not in result[0]
        assert "\n" not in result[0]
        assert "\t" not in result[0]

    def test_custom_max_chars(self):
        text = "Short. Text. Here."
        result = chunk_text(text, max_chars=10)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 10

    def test_unicode_text(self):
        text = "\u65e5\u672c\u8a9e\u306e\u30c6\u30b9\u30c8\u3002\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059\u3002"
        result = chunk_text(text)
        assert len(result) >= 1
        assert result[0]  # Non-empty

    def test_preserves_all_content(self):
        """All input text should appear in the output chunks (no data loss)."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, max_chars=30)
        combined = " ".join(result)
        # All words should be present
        for word in text.split():
            assert word.rstrip(".") in combined or word in combined

    def test_no_empty_chunks(self):
        text = "One.  Two.  Three.  Four.  Five."
        result = chunk_text(text, max_chars=15)
        for chunk in result:
            assert chunk.strip() != ""


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_utf8_file(self, tmp_text_file):
        content = read_file(tmp_text_file)
        assert content == "This is text from a file."

    def test_reads_unicode_file(self, tmp_unicode_file):
        content = read_file(tmp_unicode_file)
        assert "Caf\u00e9" in content
        assert "\u65e5\u672c\u8a9e" in content

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_file("/nonexistent/path/file.txt")
