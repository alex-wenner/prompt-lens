from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


def _result():
    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    return harness.explain("A short one. A considerably longer sentence here.")


def test_attribution_to_html_is_self_contained() -> None:
    html = _result().to_html()
    assert html.startswith("<!DOCTYPE html>")
    assert "sentence_1" in html
    assert "Feature attribution" in html
    assert "<script" not in html  # no external deps, nothing executable


def test_attribution_html_escapes_content() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    html = harness.explain("Render <b>bold</b> & <script>alert(1)</script>.").to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_cli_writes_html_report(tmp_path) -> None:
    from typer.testing import CliRunner

    from promptlens.cli.main import app

    output = tmp_path / "report.html"
    result = CliRunner().invoke(
        app,
        [
            "explain",
            "--prompt",
            "Alpha sentence. Beta sentence.",
            "--provider",
            "echo",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    content = output.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")


def test_per_question_to_html_renders_heatmap() -> None:
    from promptlens.core.result import PerQuestionAttribution

    single = _result()
    report = PerQuestionAttribution(questions=["q1", "q2"], results=[single, single])
    html = report.to_html()
    assert "Share of attribution mass per question" in html
    assert "q1" in html and "q2" in html
