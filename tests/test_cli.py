"""Tests for CLI --tokenizer-json flag."""

from click.testing import CliRunner
from src.cli.main import cli


class TestCliTokenizerFlag:
    def test_ingest_dense_accepts_tokenizer_flag(self):
        """--tokenizer-json accepted without error on ingest-dense help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest-dense", "--help"])
        assert result.exit_code == 0
        assert "--tokenizer-json" in result.output

    def test_ingest_accepts_tokenizer_flag(self):
        """--tokenizer-json accepted on ingest help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--tokenizer-json" in result.output

    def test_group_accepts_tokenizer_flag(self):
        """--tokenizer-json accepted on CLI group help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--tokenizer-json" in result.output
