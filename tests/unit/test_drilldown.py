import pytest

from promptlens import AttributionHarness, DrilldownResult, explain_drilldown
from promptlens.adapters import EchoAdapter
from promptlens.core.base import CompletionOutput, ToolDefinitions
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import MarkdownSectionSegmenter, SentenceSegmenter

_PROMPT = """# Role
You are a support agent. Be helpful.

# Refund policy
Refunds over one hundred dollars escalate. Damaged items need an RMA. Never refund disputes.

# Tone
Be brief. Be kind.
"""


def _harness() -> AttributionHarness:
    return AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=MarkdownSectionSegmenter(),
        scorer=LengthDriftScorer(),
    )


def test_drilldown_refines_top_sections_into_sentences() -> None:
    result = explain_drilldown(_harness(), _PROMPT, top_k=2)

    assert isinstance(result, DrilldownResult)
    assert len(result.overview.attributions) == 3
    assert len(result.refinements) == 2
    for refinement in result.refinements:
        names = [a.feature.name for a in refinement.result.attributions]
        assert all(name.startswith(f"{refinement.feature.name}.") for name in names)
        assert len(names) >= 2


def test_refined_evaluations_keep_full_prompt_context() -> None:
    result = explain_drilldown(_harness(), _PROMPT, top_k=1)

    refinement = result.refinements[0]
    other_sections = [
        attribution.feature
        for attribution in result.overview.attributions
        if attribution.feature.name != refinement.feature.name
    ]
    for evaluation in refinement.result.evaluations:
        # Everything outside the refined section survives verbatim.
        for feature in other_sections:
            assert feature.text in evaluation.prompt


def test_drilldown_reports_call_accounting() -> None:
    result = explain_drilldown(_harness(), _PROMPT, top_k=2)

    overview_calls = len(result.overview.evaluations) + 1
    refinement_calls = sum(len(r.result.evaluations) + 1 for r in result.refinements)
    assert result.provider_calls_used == overview_calls + refinement_calls
    flat_sentences = len(SentenceSegmenter().segment(_PROMPT))
    assert result.flat_sweep_provider_calls == flat_sentences + 1


def test_top_k_zero_skips_refinement() -> None:
    result = explain_drilldown(_harness(), _PROMPT, top_k=0)

    assert result.refinements == []


def test_negative_top_k_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        explain_drilldown(_harness(), _PROMPT, top_k=-1)


def test_spanless_features_are_skipped() -> None:
    class _ToolBiasedAdapter(EchoAdapter):
        def complete(
            self, prompt: str, tools: ToolDefinitions | None = None
        ) -> CompletionOutput:
            del tools
            return CompletionOutput(text=prompt)

    tools: ToolDefinitions = [
        {"name": "lookup", "description": "Find things.", "input_schema": {}}
    ]
    harness = AttributionHarness(
        adapter=_ToolBiasedAdapter(),
        segmenter=MarkdownSectionSegmenter(),
        scorer=LengthDriftScorer(),
    )

    result = explain_drilldown(harness, _PROMPT, tools=tools, top_k=10)

    # The synthetic tools feature has no span; only real sections get refined.
    refined_names = {refinement.feature.name for refinement in result.refinements}
    assert "tools" not in refined_names


def test_drilldown_serializes_to_json() -> None:
    result = explain_drilldown(_harness(), _PROMPT, top_k=1)

    data = result.to_dict()
    assert data["overview"]["attributions"]
    assert data["refinements"][0]["feature"]["name"].startswith("section_")
    assert data["provider_calls_used"] > 0
    assert result.to_json()
