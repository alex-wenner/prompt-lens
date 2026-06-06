"""Turn attribution evidence into a concrete before/after prompt rewrite.

Scenario: you have a verbose prompt and want a tighter version that keeps the
load-bearing instruction and prunes filler. ``AttributionHarness.optimize`` runs
a leave-one-out attribution sweep, then hands that evidence to an
``LLMPromptOptimizer`` which proposes a whole-prompt rewrite for review.

This runs entirely offline with a deterministic simulated adapter that plays two
roles, exactly like a real model would:

* During the attribution sweep it echoes the (masked) prompt so the length-drift
  scorer can rank features.
* When it receives the optimizer's rewrite brief (which ends with a
  ``REWRITTEN PROMPT:`` contract) it returns a scripted, tightened rewrite.

The proposed rewrite is returned for review and is **never** adopted
automatically. Swap in a real provider adapter to get a genuine model rewrite.

Attribution is a lens, not an oracle: a proposed rewrite is a candidate, not a
verified improvement. Re-run attribution and task checks before shipping it.
"""

from __future__ import annotations

from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.optimizers import LLMPromptOptimizer
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

ORIGINAL_PROMPT = (
    "You are an extremely helpful, friendly, and knowledgeable assistant. "
    "Summarize the input text in exactly three bullet points. "
    "Feel free to be as detailed and thorough as you possibly can. "
    "Thanks so much for your hard work on this important task."
)

# The rewrite the simulated model "proposes" once it has the attribution brief.
SCRIPTED_REWRITE = "Summarize the input text in exactly three bullet points."
SCRIPTED_RATIONALE = (
    "Kept the only load-bearing instruction (the three-bullet constraint) and "
    "pruned the inert pleasantries that carried no attribution."
)


class ScriptedOptimizerAdapter(Adapter):
    """Echo during attribution; return a scripted rewrite for the optimizer brief."""

    def __init__(self) -> None:
        self.model = "scripted-optimizer"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        if "REWRITTEN PROMPT:" in prompt:
            text = f"REWRITTEN PROMPT:\n{SCRIPTED_REWRITE}\n\nRATIONALE:\n{SCRIPTED_RATIONALE}"
            return CompletionOutput(text=text)
        return CompletionOutput(text=prompt)


def main() -> dict[str, Any]:
    """Run the demo and return the rewrite for inspection and tests."""
    harness = AttributionHarness(
        adapter=ScriptedOptimizerAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        optimizer=LLMPromptOptimizer(ScriptedOptimizerAdapter()),
    )
    result = harness.optimize(ORIGINAL_PROMPT)

    print("Attribution-informed prompt rewrite:\n")
    result.print()
    print(
        "\nLens, not oracle: the proposed rewrite is a candidate, not a verified "
        "improvement. Re-run attribution and a task metric before adopting it."
    )
    return {
        "original_prompt": result.original_prompt,
        "proposed_prompt": result.proposed_prompt,
        "rationale": result.rationale,
        "top_feature": result.metadata.get("top_feature"),
    }


if __name__ == "__main__":
    main()
