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

This example makes **real provider calls** (export ``OPENAI_API_KEY``, or pick
another provider with ``PROMPTLENS_EXAMPLE_PROVIDER``). The Banzhaf sweep runs
dozens of coalitions, so it shows the measured cost estimate first.

Config permutation this example pins: ``RandomCoalitionSampler`` (seeded for
reproducibility) versus ``LeaveOneOutSampler``, scored with ``LengthDriftScorer``
and masked with ``FillerMasker`` (masked text replaced by same-length filler).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter
from promptlens.maskers import FillerMasker
from promptlens.samplers import LeaveOneOutSampler, RandomCoalitionSampler
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    confirm_spend,
    console,
    get_adapter,
    print_completion,
    print_footer,
)

# Two redundant verbosity drivers (sentences 2 and 3): either one on its own
# produces a long, detailed reply, so neither has a marginal effect at the full
# prompt — yet together they decide whether replies are long or terse.
PROMPT = (
    "You are a support assistant for an email helpdesk. "
    "Explain your reasoning step by step in every reply. "
    "Always include a concrete worked example for the customer. "
    "Sign every reply with the team name. "
    "Customer message: 'Where is my order?'"
)

REASONING_SIGNAL = "reasoning step by step"
EXAMPLE_SIGNAL = "worked example"

# Enough random coalitions for a stable Banzhaf estimate without a huge bill.
N_COALITIONS = 60


def _values_for(harness: AttributionHarness, baseline: Any = None) -> dict[str, float]:
    result = harness.explain(PROMPT, baseline=baseline)
    by_signal: dict[str, float] = {}
    for attribution in result.attributions:
        if REASONING_SIGNAL in attribution.feature.text:
            by_signal["reasoning"] = attribution.value
        elif EXAMPLE_SIGNAL in attribution.feature.text:
            by_signal["example"] = attribution.value
    return by_signal


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Attribute the same prompt with leave-one-out and with Banzhaf sampling."""
    model = adapter if adapter is not None else get_adapter()
    common: dict[str, Any] = {
        "adapter": model,
        "segmenter": SentenceSegmenter(),
        "scorer": LengthDriftScorer(),
        "masker": FillerMasker(),
    }
    banzhaf_harness = AttributionHarness(
        sampler=RandomCoalitionSampler(n_coalitions=N_COALITIONS, seed=7), **common
    )

    baseline, estimate = banzhaf_harness.estimate(PROMPT)
    print_completion("Baseline reply", baseline)
    estimate.print()
    confirm_spend(estimate.total_usd, estimate.evaluations)

    loo = _values_for(
        AttributionHarness(sampler=LeaveOneOutSampler(), **common), baseline=baseline
    )
    banzhaf = _values_for(banzhaf_harness, baseline=baseline)

    console.print(
        "\n[bold]Two redundant instructions, scored two ways (attribution value):[/bold]\n"
    )
    console.print(f"  {'instruction':<24}{'leave-one-out':>16}{'Banzhaf':>12}")
    for key, label in (("reasoning", "step-by-step rule"), ("example", "worked-example rule")):
        console.print(f"  {label:<24}{loo.get(key, 0.0):>16.4f}{banzhaf.get(key, 0.0):>12.4f}")
    console.print(
        "\nWhen the two rules are redundant for this model, leave-one-out scores "
        "both near 0 — masking either alone changes nothing because its twin "
        "still triggers the long reply. Banzhaf judges each across coalitions "
        "where the twin is also masked, and recovers their real effect."
    )
    print_footer(
        "leave-one-out and Banzhaf answer different questions. Use leave-one-out "
        "for marginal effect against the live prompt; use random coalitions when "
        "redundancy or interactions may be hiding a feature."
    )
    return {"leave_one_out": loo, "banzhaf": banzhaf}


if __name__ == "__main__":
    main()
