"""Run the whole attribution loop — and its synopsis — on a local model.

The "don't rely on token beasts" story end to end: attribution sweeps multiply
provider calls by feature count, so doing them on a hosted frontier model gets
expensive fast. Here both the attribution sweep *and* the natural-language
synopsis run on a local Ollama model, for $0 and with no data leaving the box.

Config permutation this example pins:

* provider ``ollama`` (local; alias ``local``) — opt in with
  ``PROMPTLENS_EXAMPLE_PROVIDER=ollama`` and a running Ollama server;
* ``ParagraphSegmenter`` — the prompt is a four-paragraph system prompt;
* ``DropMasker`` — masked paragraphs are removed outright rather than replaced
  with a placeholder;
* ``LLMSynopsisWriter`` — one extra local call turns the evidence into prose.

With no local server (as in CI) it falls back to a deterministic offline model
whose output length is shaped by the two formatting paragraphs, so the example
runs end-to-end and doubles as a smoke test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.maskers import DropMasker
from promptlens.reporters import LLMSynopsisWriter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import ParagraphSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import load_text, print_footer, select_adapter  # noqa: E402

PROMPT = load_text(__file__, "prompt.md").strip()

# The two formatting paragraphs the offline model keys on for output length.
SEVERITY_SIGNAL = "severity of blocker"
CHECKLIST_SIGNAL = "markdown checklist"
# Marker that tells the offline model it is being asked for a synopsis, not a review.
SYNOPSIS_MARKER = "attribution evidence"

_CANNED_SYNOPSIS = (
    "The severity-tagging and markdown-checklist paragraphs carry the output; "
    "together they account for nearly all measured drift. The role framing and "
    "the terse-tone line are inert under a length scorer for this diff. Next: "
    "re-check with an embedding scorer, since tone rarely shows up as length, "
    "and confirm the checklist format with a real review before trimming."
)


class SimulatedLocalModel(Adapter):
    """Offline fallback for the local-inference example.

    For an attribution call, returns review text whose length depends only on the
    severity-tagging and checklist-format paragraphs. For the synopsis call
    (detected by the attribution-evidence brief) it returns a canned narrative,
    so the offline path produces a clean summary instead of echoing the brief.
    """

    def __init__(self) -> None:
        self.model = "simulated-local-model"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        lowered = prompt.lower()
        if SYNOPSIS_MARKER in lowered:
            return CompletionOutput(text=_CANNED_SYNOPSIS)
        review = "Found a possible off-by-one in the pagination helper."
        if SEVERITY_SIGNAL in lowered:
            review += " [blocker] fails test_pagination_last_page; [nit] rename idx."
        if CHECKLIST_SIGNAL in lowered:
            review = "## pagination.py\n- " + review.replace("; ", "\n- ")
        return CompletionOutput(text=review)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run attribution plus a synopsis on a (preferably local) model."""
    adapter = adapter if adapter is not None else select_adapter(
        SimulatedLocalModel(), prefer="ollama"
    )
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=ParagraphSegmenter(),
        scorer=LengthDriftScorer(),
        masker=DropMasker(),
    )
    result = harness.explain(PROMPT)
    synopsis = LLMSynopsisWriter(adapter).summarize(PROMPT, result)
    result = result.with_synopsis(synopsis)

    print("Local attribution over a code-review system prompt (DropMasker, paragraphs):\n")
    result.print()

    shares = {
        attribution.feature.name: share for attribution, share in result.ranked()
    }
    print_footer(
        "the attribution sweep and the synopsis both ran on the local model — no "
        "data left the box and the run cost nothing. Tone may still matter for "
        "reasons a length scorer cannot see."
    )
    return {
        "shares": shares,
        "synopsis": synopsis.text,
        "synopsis_model": synopsis.model,
        "top_feature": result.ranked()[0][0].feature.name,
    }


if __name__ == "__main__":
    main()
