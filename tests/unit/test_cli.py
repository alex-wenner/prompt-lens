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


def test_explain_can_include_supplementary_rewrites(tmp_path) -> None:
    output = tmp_path / "result.json"

    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--supplementary-rewrites",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Supplementary prompt mutations" in result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["supplementary_evaluations"]) == 2
    assert data["supplementary_evaluations"][0]["kind"] == "prompt-mutation"


def test_optimize_command_runs_with_echo(tmp_path) -> None:
    output = tmp_path / "optimized.json"

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "promptlens Optimization" in result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["original_prompt"] == "Alpha sentence. Beta sentence."
    assert data["proposed_prompt"]
    assert "caveat" in data["metadata"]


def test_explain_dry_run_estimates_without_running() -> None:
    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "CostEstimate" in result.output
    assert "promptlens Attribution" not in result.output


def test_explain_confirm_aborts_on_no() -> None:
    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--confirm",
        ],
        input="n\n",
    )

    assert result.exit_code != 0
    assert "CostEstimate" in result.output
    assert "promptlens Attribution" not in result.output


def test_explain_confirm_runs_on_yes() -> None:
    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--confirm",
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "CostEstimate" in result.output
    assert "promptlens Attribution" in result.output


def test_auto_segmenter_picks_sections_for_markdown() -> None:
    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "# Role\nBe concise.\n\n# Task\nSummarize.",
            "--provider",
            "echo",
            "--segmenter",
            "auto",
        ],
    )

    assert result.exit_code == 0
    assert "section_1" in result.output


def test_explain_synopsis_attaches_llm_summary(tmp_path) -> None:
    output = tmp_path / "result.json"

    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--synopsis",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Synopsis (echo)" in result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["synopsis"]["model"] == "echo"
    # The echo adapter returns the brief itself, proving the evidence reached the LLM.
    assert "Ranked features" in data["synopsis"]["text"]
    assert data["drift_highlights"]


def test_explain_richer_output_shows_drift_highlights() -> None:
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
    assert "Largest output drifts" in result.output


def test_explain_accepts_tool_args_scorer() -> None:
    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--scorer",
            "tool-args",
        ],
    )

    assert result.exit_code == 0
    assert "promptlens Attribution" in result.output
