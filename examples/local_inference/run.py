"""Run the whole attribution loop — and its synopsis — on a local model.

The "don't rely on token beasts" story end to end: attribution sweeps multiply
provider calls by feature count, so doing them on a hosted frontier model gets
expensive fast. Here both the attribution sweep *and* the natural-language
synopsis run on a local Ollama model, for $0 and with no data leaving the box.

This example makes **real model calls against your local Ollama server**
(default ``http://localhost:11434``; pick the model with
``PROMPTLENS_EXAMPLE_MODEL``, e.g. ``llama3.2``). Point it at a hosted provider
instead with ``PROMPTLENS_EXAMPLE_PROVIDER=openai`` etc.

Config permutations this example pins:

* ``ParagraphSegmenter`` — the prompt is a four-paragraph system prompt;
* ``DropMasker`` — masked paragraphs are removed outright rather than replaced
  with a placeholder;
* ``LLMSynopsisWriter`` — one extra local call turns the evidence into prose.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter
from promptlens.maskers import DropMasker
from promptlens.reporters import LLMSynopsisWriter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import ParagraphSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    console,
    get_adapter,
    load_text,
    print_completion,
    print_footer,
)

PROMPT = load_text(__file__, "prompt.md").strip()


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run attribution plus a synopsis on a (preferably local) model."""
    adapter = adapter if adapter is not None else get_adapter(prefer="ollama")
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=ParagraphSegmenter(),
        scorer=LengthDriftScorer(),
        masker=DropMasker(),
    )

    baseline, estimate = harness.estimate(PROMPT)
    print_completion("Baseline review", baseline)
    estimate.print()

    console.print(
        "\n[bold]Local attribution over a code-review system prompt "
        "(DropMasker, paragraphs)[/bold]\n"
    )
    result = harness.explain(PROMPT, baseline=baseline)
    synopsis = LLMSynopsisWriter(adapter).summarize(PROMPT, result)
    result = result.with_synopsis(synopsis)
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
