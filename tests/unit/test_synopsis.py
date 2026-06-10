from promptlens import AttributionHarness, AttributionResult, CompletionOutput, Synopsis
from promptlens.adapters import EchoAdapter
from promptlens.core.base import ToolDefinitions
from promptlens.reporters import LLMSynopsisWriter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


def _result(prompt: str) -> AttributionResult:
    harness = AttributionHarness(
        adapter=EchoAdapter(model="echo"),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    return harness.explain(prompt)


def test_brief_contains_full_attribution_evidence() -> None:
    prompt = "Always answer in JSON. Be concise."
    result = _result(prompt)
    writer = LLMSynopsisWriter(EchoAdapter(model="summarizer"))

    brief = writer.build_brief(prompt, result)

    assert "Prompt under attribution:" in brief
    assert "Baseline output:" in brief
    assert "Ranked features (most load-bearing first):" in brief
    assert "Largest output drifts" in brief


def test_brief_includes_baseline_tool_sequence() -> None:
    prompt = "Always answer in JSON. Be concise."
    result = _result(prompt)
    with_tools = result.model_copy(
        update={
            "baseline_output": CompletionOutput(
                text="done", tool_calls=[{"name": "search"}, {"name": "answer"}]
            )
        }
    )
    writer = LLMSynopsisWriter(EchoAdapter(model="summarizer"))

    brief = writer.build_brief(prompt, with_tools)

    assert "search -> answer" in brief


def test_summarize_returns_synopsis_from_adapter() -> None:
    prompt = "Always answer in JSON. Be concise."
    result = _result(prompt)

    class _CannedAdapter(EchoAdapter):
        def complete(
            self, prompt: str, tools: ToolDefinitions | None = None
        ) -> CompletionOutput:
            del prompt, tools
            return CompletionOutput(text="  The JSON instruction carries the output.  ")

    writer = LLMSynopsisWriter(_CannedAdapter(model="local-llm"))

    synopsis = writer.summarize(prompt, result)

    assert isinstance(synopsis, Synopsis)
    assert synopsis.text == "The JSON instruction carries the output."
    assert synopsis.model == "local-llm"
    assert synopsis.metadata["features"] == len(result.attributions)


def test_with_synopsis_round_trips_through_dict() -> None:
    prompt = "Always answer in JSON. Be concise."
    result = _result(prompt)
    synopsis = Synopsis(text="summary", model="local-llm")

    enriched = result.with_synopsis(synopsis)

    assert result.synopsis is None
    assert enriched.synopsis is synopsis
    data = enriched.to_dict()
    assert data["synopsis"] == {"text": "summary", "model": "local-llm", "metadata": {}}
    assert data["drift_highlights"]
    assert {"removed", "score", "output_text"} <= set(data["drift_highlights"][0])


def test_drift_highlights_name_removed_features() -> None:
    result = _result("Always answer in JSON. Be concise.")

    highlights = result.drift_highlights(limit=2)

    assert len(highlights) == 2
    scores = [highlight["score"] for highlight in highlights]
    assert scores == sorted(scores, reverse=True)
    assert all(highlight["removed"] for highlight in highlights)
