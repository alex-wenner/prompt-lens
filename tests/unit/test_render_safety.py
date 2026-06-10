"""Rendering safety: model/prompt-derived text must print literally.

Model outputs and prompts routinely contain bracketed tokens — ``[blocker]``
severity tags, ``[section_2]`` feature references in a synopsis, ``[citation]``
markers. Rendered as plain strings, rich parses those as console markup and
silently deletes them from tables. These tests pin that every print surface
renders dynamic text literally.
"""

from __future__ import annotations

import pytest

from promptlens import AttributionHarness, OptimizationResult, Synopsis
from promptlens.adapters import EchoAdapter
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.core.result import AttributionResult
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

# A short, valid-looking markup tag: exactly the kind rich would swallow.
TOKEN = "[tag]"


class _BracketAdapter(Adapter):
    """Return output containing a markup-like token."""

    def __init__(self) -> None:
        self.model = "bracket-model"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        return CompletionOutput(text=f"{TOKEN} reply about {prompt[:20]}")


def _result() -> AttributionResult:
    harness = AttributionHarness(
        adapter=_BracketAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    return harness.explain(f"Keep the {TOKEN} marker. Second sentence here.")


def test_attribution_print_keeps_bracketed_tokens(capsys: pytest.CaptureFixture[str]) -> None:
    _result().print()

    output = capsys.readouterr().out
    assert TOKEN in output


def test_synopsis_print_keeps_bracketed_tokens(capsys: pytest.CaptureFixture[str]) -> None:
    enriched = _result().with_synopsis(
        Synopsis(text=f"The {TOKEN} feature carries the output.", model="m")
    )

    enriched.print()

    output = capsys.readouterr().out
    assert f"The {TOKEN} feature" in output


def test_optimization_print_keeps_bracketed_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    OptimizationResult(
        original_prompt=f"Use {TOKEN} tags.",
        proposed_prompt=f"Always use {TOKEN} tags.",
        rationale=f"Kept the {TOKEN} instruction.",
    ).print()

    output = capsys.readouterr().out
    assert output.count(TOKEN) == 3


def test_echo_synopsis_brief_round_trips_feature_references(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The echoed brief names features as ``[sentence_1]``; they must survive print."""
    from promptlens.reporters import LLMSynopsisWriter

    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    prompt = "Alpha sentence. Beta sentence."
    result = harness.explain(prompt)
    enriched = result.with_synopsis(LLMSynopsisWriter(EchoAdapter()).summarize(prompt, result))

    enriched.print()

    output = capsys.readouterr().out
    assert "removed [sentence_1]" in output
