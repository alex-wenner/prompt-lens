"""See redundancy that leave-one-out is blind to — and Banzhaf catches.

Leave-one-out measures each instruction's marginal effect against the *full*
prompt. When two instructions are **redundant** — either one alone produces the
same behavior — leave-one-out scores *both* as dead weight: masking either one
on its own changes nothing, because the other still covers for it. Delete the
"redundant" pair on that evidence and the behavior collapses.

The ``RandomCoalitionSampler`` masks several features per coalition, so each
instruction is judged across many contexts — including the ones where its twin
is also masked. At inclusion probability 0.5 that is a Monte-Carlo Banzhaf
value, and it recovers the real contribution of both redundant instructions.

Config permutation this example pins: ``RandomCoalitionSampler`` (seeded for
reproducibility) versus ``LeaveOneOutSampler``, scored with ``LengthDriftScorer``
and masked with ``FillerMasker`` (masked text replaced by same-length filler).
It runs fully offline — the lesson is about the estimator, not the model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.maskers import FillerMasker
from promptlens.samplers import LeaveOneOutSampler, RandomCoalitionSampler
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import print_footer  # noqa: E402

# Two redundant verbosity drivers (sentences 2 and 3): either one on its own
# produces a long, detailed reply, so neither has a marginal effect at the full
# prompt — yet together they decide whether replies are long or terse.
PROMPT = (
    "You are a support assistant for an email helpdesk. "
    "Explain your reasoning step by step in every reply. "
    "Always include a concrete worked example for the customer. "
    "Sign every reply with the team name."
)

REASONING_SIGNAL = "reasoning step by step"
EXAMPLE_SIGNAL = "worked example"

# Either driver yields the long reply; only masking both collapses it to terse.
_LONG = (
    "Here is what is happening: your order cleared payment, entered fulfillment, "
    "and shipped this morning. For example, order #4821 followed the same path "
    "yesterday and arrived within a day. Tracking is on its way to your inbox now."
)
_TERSE = "Your order shipped today."


class SimulatedHelpdeskAgent(Adapter):
    """Offline model where two redundant instructions both drive reply length.

    The reply is long when *either* the step-by-step or worked-example rule is
    visible, and only collapses to terse when both are masked — the redundancy
    leave-one-out cannot see from the full prompt alone.
    """

    def __init__(self) -> None:
        self.model = "simulated-helpdesk-agent"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        lowered = prompt.lower()
        verbose = REASONING_SIGNAL in lowered or EXAMPLE_SIGNAL in lowered
        return CompletionOutput(text=_LONG if verbose else _TERSE)


def _values_for(harness: AttributionHarness) -> dict[str, float]:
    result = harness.explain(PROMPT)
    by_signal: dict[str, float] = {}
    for attribution in result.attributions:
        if REASONING_SIGNAL in attribution.feature.text:
            by_signal["reasoning"] = attribution.value
        elif EXAMPLE_SIGNAL in attribution.feature.text:
            by_signal["example"] = attribution.value
    return by_signal


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Attribute the same prompt with leave-one-out and with Banzhaf sampling."""
    model = adapter if adapter is not None else SimulatedHelpdeskAgent()
    common = {
        "adapter": model,
        "segmenter": SentenceSegmenter(),
        "scorer": LengthDriftScorer(),
        "masker": FillerMasker(),
    }
    loo = _values_for(AttributionHarness(sampler=LeaveOneOutSampler(), **common))
    banzhaf = _values_for(
        AttributionHarness(sampler=RandomCoalitionSampler(n_coalitions=300, seed=7), **common)
    )

    print("Two redundant instructions, scored two ways (attribution value):\n")
    print(f"  {'instruction':<24}{'leave-one-out':>16}{'Banzhaf':>12}")
    for key, label in (("reasoning", "step-by-step rule"), ("example", "worked-example rule")):
        print(f"  {label:<24}{loo.get(key, 0.0):>16.4f}{banzhaf.get(key, 0.0):>12.4f}")
    print(
        "\nLeave-one-out scores both at ~0 — masking either alone changes nothing "
        "because its twin still triggers the long reply. Trusting that, you would "
        "delete both and watch replies go terse. Banzhaf judges each across "
        "coalitions where the twin is also masked, and recovers their real effect."
    )
    print_footer(
        "leave-one-out and Banzhaf answer different questions. Use leave-one-out "
        "for marginal effect against the live prompt; use random coalitions when "
        "redundancy or interactions may be hiding a feature."
    )
    return {"leave_one_out": loo, "banzhaf": banzhaf}


if __name__ == "__main__":
    main()
