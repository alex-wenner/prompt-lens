import json

from typer.testing import CliRunner

from promptlens.cli.main import app


def test_explain_provider_echo_still_runs() -> None:
    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
        ],
    )

    assert result.exit_code == 0
    assert "promptlens Attribution" in result.output


def test_explain_accepts_scorer_and_sampler_flags(tmp_path) -> None:
    config = tmp_path / "scorer.json"
    config.write_text(json.dumps({"expected_tool": "search"}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--sampler",
            "leave-one-out",
            "--scorer",
            "tool-call",
            "--scorer-config",
            str(config),
        ],
    )

    assert result.exit_code == 0
    assert "promptlens Attribution" in result.output
