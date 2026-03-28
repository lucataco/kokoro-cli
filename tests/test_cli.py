"""Tests for kokoro_cli.cli — Click CLI commands."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from kokoro_cli.cli import main
from kokoro_cli.config import VOICES


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Help and usage
# ---------------------------------------------------------------------------


class TestHelp:
    def test_help_flag(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "kokoro" in result.output.lower() or "tts" in result.output.lower()

    def test_h_flag(self, runner):
        result = runner.invoke(main, ["-h"])
        assert result.exit_code == 0

    def test_serve_help(self, runner):
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "daemon" in result.output.lower()

    def test_stop_help(self, runner):
        result = runner.invoke(main, ["stop", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --list-voices
# ---------------------------------------------------------------------------


class TestListVoices:
    def test_lists_all_voices(self, runner):
        result = runner.invoke(main, ["--list-voices"])
        assert result.exit_code == 0
        # Check that some known voices appear
        assert "af_sky" in result.output
        assert "am_adam" in result.output
        assert "bf_alice" in result.output

    def test_lists_correct_count(self, runner):
        result = runner.invoke(main, ["--list-voices"])
        assert result.exit_code == 0
        assert f"Total: {len(VOICES)} voices" in result.output

    def test_shows_language_and_gender(self, runner):
        result = runner.invoke(main, ["--list-voices"])
        assert "American English" in result.output
        assert "Female" in result.output
        assert "Male" in result.output


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_no_input_shows_error(self, runner):
        """No text, no file, no stdin should produce an error."""
        result = runner.invoke(main, ["tts"])
        assert result.exit_code != 0
        assert "no input" in result.output.lower() or "error" in result.output.lower()

    def test_nonexistent_file(self, runner):
        result = runner.invoke(main, ["--file", "/nonexistent/path.txt"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# stop command
# ---------------------------------------------------------------------------


class TestStopCommand:
    @patch("kokoro_cli.server.stop_daemon", return_value=False)
    def test_stop_no_daemon(self, mock_stop, runner):
        result = runner.invoke(main, ["stop"])
        assert result.exit_code == 0
        assert "no daemon" in result.output.lower()

    @patch("kokoro_cli.server.stop_daemon", return_value=True)
    def test_stop_running_daemon(self, mock_stop, runner):
        result = runner.invoke(main, ["stop"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()
