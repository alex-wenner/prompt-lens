"""Turn attribution evidence into a concrete before/after prompt rewrite.

Scenario: you have a verbose prompt and want a tighter version that keeps the
load-bearing instruction and prunes filler. ``AttributionHarness.optimize`` runs
a leave-one-out attribution sweep, then hands that evidence to an
``LLMPromptOptimizer`` which proposes a whole-prompt rewrite for review.

By default this runs against a **real provider** (set ``OPENAI_API_KEY`` or
``ANTHROPIC_API_KEY``; see ``examples/_shared.py`` for the env vars): the
same model both answers the attribution sweep and proposes the rewrite. When no
credential is available it falls back to a deterministic offline adapter that
echoes prompts during attribution and returns a scripted rewrite for the
optimizer brief, so the example still runs end-to-end and doubles as a CI smoke
test.

The proposed rewrite is returned for review and is **never** adopted
automatically.

Attribution is a lens, not an oracle: a proposed rewrite is a candidate, not a
verified improvement. Re-run attribution and task checks before shipping it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.optimizers import LLMPromptOptimizer
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import print_footer, select_adapter  # noqa: E402

ORIGINAL_PROMPT = (
    "You are an extremely helpful, friendly, and knowledgeable assistant. "
    "Summarize the input text in exactly three bullet points. "
    "Feel free to be as detailed and thorough as you possibly can. "
    "Thanks so much for your hard work on this important task."
)

# The rewrite the offline fallback model "proposes" once it has the attribution brief.
SCRIPTED_REWRITE = "Summarize the input text in exactly three bullet points."
SCRIPTED_RATIONALE = (
    "Kept the only load-bearing instruction (the three-bullet constraint) and "
    "pruned the inert pleasantries that carried no attribution."
)


class ScriptedOptimizerAdapter(Adapter):
    """Offline fallback: echo during attribution; return a scripted rewrite for the brief."""

    def __init__(self) -> None:
        self.model = "scripted-optimizer"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        if "REWRITTEN PROMPT:" in prompt:
            text = f"REWRITTEN PROMPT:\n{SCRIPTED_REWRITE}\n\nRATIONALE:\n{SCRIPTED_RATIONALE}"
            return CompletionOutput(text=text)
        return CompletionOutput(text=prompt)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the rewrite for inspection and tests."""
    adapter = adapter if adapter is not None else select_adapter(ScriptedOptimizerAdapter())
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        optimizer=LLMPromptOptimizer(adapter),
    )
    result = harness.optimize(ORIGINAL_PROMPT)

    print("Attribution-informed prompt rewrite:\n")
    result.print()
    print_footer(
        "the proposed rewrite is a candidate, not a verified improvement. Re-run "
        "attribution and a task metric before adopting it."
    )
    return {
        "original_prompt": result.original_prompt,
        "proposed_prompt": result.proposed_prompt,
        "rationale": result.rationale,
        "top_feature": result.metadata.get("top_feature"),
    }


if __name__ == "__main__":
    main()
