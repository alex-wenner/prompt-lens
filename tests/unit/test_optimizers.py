from promptlens import AttributionHarness, CompletionOutput, OptimizationResult
from promptlens.adapters import EchoAdapter
from promptlens.core.base import ToolDefinitions
from promptlens.optimizers import LLMPromptOptimizer
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


def _result(prompt: str) -> tuple[AttributionHarness, object]:
    harness = AttributionHarness(
        adapter=EchoAdapter(model="echo"),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    return harness, harness.explain(prompt)


def test_optimizer_builds_brief_from_attribution_evidence() -> None:
    prompt = "Always answer in JSON. Be concise."
    _, result = _result(prompt)
    optimizer = LLMPromptOptimizer(EchoAdapter(model="rewriter"))

    brief = optimizer.build_brief(prompt, result)

    assert "Original prompt:" in brief
    assert "Ranked features (most load-bearing first):" in brief
    assert "Masked-prompt permutations" in brief
    assert "REWRITTEN PROMPT:" in brief


def test_optimizer_parses_structured_response() -> None:
    prompt = "Always answer in JSON. Be concise."

    class _StructuredAdapter(EchoAdapter):
        def complete(
            self, prompt: str, tools: ToolDefinitions | None = None
        ) -> CompletionOutput:
            del prompt, tools
            return CompletionOutput(
                text="REWRITTEN PROMPT:\nRespond only with JSON.\n\nRATIONALE:\nTightened it."
            )

    _, result = _result(prompt)
    optimizer = LLMPromptOptimizer(_StructuredAdapter(model="rewriter"))

    optimized = optimizer.optimize(prompt, result)

    assert isinstance(optimized, OptimizationResult)
    assert optimized.original_prompt == prompt
    assert optimized.proposed_prompt == "Respond only with JSON."
    assert optimized.rationale == "Tightened it."
    assert optimized.metadata["rewrite_model"] == "rewriter"
    assert optimized.metadata["evaluations"] == len(result.evaluations)
    assert "caveat" in optimized.metadata


def test_optimizer_falls_back_to_whole_response() -> None:
    prompt = "Always answer in JSON. Be concise."

    class _PlainAdapter(EchoAdapter):
        def complete(
            self, prompt: str, tools: ToolDefinitions | None = None
        ) -> CompletionOutput:
            del prompt, tools
            return CompletionOutput(text="Respond only with compact JSON.")

    _, result = _result(prompt)
    optimizer = LLMPromptOptimizer(_PlainAdapter(model="rewriter"))

    optimized = optimizer.optimize(prompt, result)

    # No markers in the response, so the whole text becomes the proposal.
    assert optimized.proposed_prompt == "Respond only with compact JSON."
    assert optimized.rationale == ""


def test_harness_optimize_requires_optimizer() -> None:
    harness, result = _result("Always answer in JSON.")
    try:
        harness.optimize("Always answer in JSON.", result=result)
    except ValueError as exc:
        assert "optimizer" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("expected ValueError when optimizer is missing")


def test_harness_optimize_uses_supplied_result() -> None:
    prompt = "Always answer in JSON. Be concise."
    harness = AttributionHarness(
        adapter=EchoAdapter(model="echo"),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        optimizer=LLMPromptOptimizer(EchoAdapter(model="rewriter")),
    )
    result = harness.explain(prompt)

    optimized = harness.optimize(prompt, result=result)

    assert optimized.original_prompt == prompt
    assert optimized.metadata["features"] == len(result.attributions)
