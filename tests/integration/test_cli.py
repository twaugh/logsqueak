"""Integration tests for CLI module."""

import pytest
from pathlib import Path
import os
import stat
from click.testing import CliRunner
from datetime import date

from logsqueak.cli import cli, load_config, load_journal_entries


class TestCLIExtractCommand:
    """Integration tests for CLI extract command."""

    def test_cli_with_valid_date_loads_journal(self, tmp_path):
        """Test CLI with valid date successfully loads journal."""
        # Setup graph structure
        graph_dir = tmp_path / "logseq-graph"
        journals_dir = graph_dir / "journals"
        journals_dir.mkdir(parents=True)

        # Create test journal file (2025-01-15 → 2025_01_15.md)
        journal_file = journals_dir / "2025_01_15.md"
        journal_file.write_text("- Test journal entry\n  - Nested bullet\n")

        # Create config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test-key
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        # Test load_journal_entries directly
        dates = [date(2025, 1, 15)]
        journals = load_journal_entries(graph_dir, dates)

        assert len(journals) == 1
        assert "2025-01-15" in journals
        assert len(journals["2025-01-15"].blocks) > 0

    def test_cli_with_date_range_loads_multiple_journals(self, tmp_path):
        """Test CLI with date range loads multiple journals."""
        # Setup graph structure
        graph_dir = tmp_path / "logseq-graph"
        journals_dir = graph_dir / "journals"
        journals_dir.mkdir(parents=True)

        # Create multiple journal files
        for day in range(10, 13):  # 2025-01-10, 11, 12
            journal_file = journals_dir / f"2025_01_{day}.md"
            journal_file.write_text(f"- Entry for Jan {day}\n")

        # Test load_journal_entries with range
        dates = [date(2025, 1, 10), date(2025, 1, 11), date(2025, 1, 12)]
        journals = load_journal_entries(graph_dir, dates)

        assert len(journals) == 3
        assert "2025-01-10" in journals
        assert "2025-01-11" in journals
        assert "2025-01-12" in journals

    def test_cli_with_missing_config_shows_error(self, tmp_path, monkeypatch):
        """Test CLI with missing config shows helpful error."""
        # Point to non-existent config
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "2025-01-15"])

        assert result.exit_code != 0
        assert "Configuration file not found" in result.output
        assert "Please create the file with the following format" in result.output
        assert "llm:" in result.output

    def test_cli_with_invalid_permissions_shows_error(self, tmp_path, monkeypatch):
        """Test CLI with invalid permissions shows error."""
        # Setup config with wrong permissions
        fake_home = tmp_path / "fake_home"
        config_dir = fake_home / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)

        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        config_file = config_dir / "config.yaml"
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test-key
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")

        # Set overly permissive permissions (644 - world readable)
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "2025-01-15"])

        assert result.exit_code != 0
        assert "overly permissive permissions" in result.output
        assert "chmod 600" in result.output

    def test_cli_with_missing_journal_shows_error(self, tmp_path, monkeypatch):
        """Test CLI with missing journal file shows helpful error."""
        # Setup config
        fake_home = tmp_path / "fake_home"
        config_dir = fake_home / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)

        graph_dir = tmp_path / "logseq-graph"
        journals_dir = graph_dir / "journals"
        journals_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test-key
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Try to extract from non-existent journal
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "2025-01-15"])

        assert result.exit_code != 0
        assert "Journal file not found" in result.output
        assert "2025-01-15" in result.output

    def test_cli_with_invalid_date_format_shows_error(self):
        """Test CLI with invalid date format shows helpful error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "2025/01/15"])

        assert result.exit_code != 0
        assert "Invalid date format" in result.output
        assert "Expected: YYYY-MM-DD" in result.output

    def test_load_config_function_success(self, tmp_path, monkeypatch):
        """Test load_config function with valid config."""
        fake_home = tmp_path / "fake_home"
        config_dir = fake_home / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)

        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        config_file = config_dir / "config.yaml"
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test-key
  model: gpt-4

logseq:
  graph_path: {graph_dir}

rag:
  top_k: 15
""")
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        config = load_config()

        assert config.llm.model == "gpt-4"
        assert config.llm.api_key == "sk-test-key"
        assert config.logseq.graph_path == str(graph_dir)
        assert config.rag.top_k == 15
