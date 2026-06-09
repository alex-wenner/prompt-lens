import math

from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.core.harness import _sampler_from_scale
from promptlens.samplers import LeaveOneOutSampler, RandomCoalitionSampler
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


def _harness(scale: str | int = "quick") -> AttributionHarness:
    return AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        perturbation_scale=scale,
    )


def test_perturbation_scale_controls_repeats() -> None:
    assert _sampler_from_scale("quick").repeats == 1
    assert _sampler_from_scale("standard").repeats == 3
    assert _sampler_from_scale("full").repeats == 5
    assert _sampler_from_scale(4).repeats == 4


def test_leave_one_out_repeats_each_feature() -> None:
    coalitions = list(LeaveOneOutSampler(repeats=2).sample(3))
    assert len(coalitions) == 6
    assert LeaveOneOutSampler(repeats=2).estimate_evaluations(3) == 6
    # Every coalition masks exactly one feature.
    assert all(coalition.count(False) == 1 for coalition in coalitions)


def test_ranked_shares_sum_to_one_and_sort_descending() -> None:
    result = _harness().explain("A short one. A considerably longer sentence here.")
    ranked = result.ranked()
    values = [attribution.value for attribution, _ in ranked]
    assert values == sorted(values, reverse=True)
    assert math.isclose(sum(share for _, share in ranked), 1.0)
    # The longer sentence carries more attribution mass than the short one.
    assert ranked[0][1] > ranked[1][1]


def test_to_dict_exposes_normalized_share() -> None:
    data = _harness().explain("Alpha sentence. Beta sentence.").to_dict()
    assert all("share" in attribution for attribution in data["attributions"])


def test_repeats_populate_stderr() -> None:
    result = _harness(scale="standard").explain("Alpha sentence. Beta sentence.")
    assert all(attribution.stderr is not None for attribution in result.attributions)


class _TriggerAdapter(Adapter):
    """Return a fixed long output only while the trigger sentence survives masking."""

    def __init__(self, trigger: str) -> None:
        self.model = "trigger"
        self.trigger = trigger

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        if self.trigger in prompt:
            return CompletionOutput(text="x" * 100)
        return CompletionOutput(text="")


def test_random_coalitions_use_masked_vs_kept_contrast() -> None:
    # Drift is 1.0 exactly when the trigger sentence is masked, 0.0 otherwise.
    # The naive mean-over-masked-coalitions estimator would hand the inert
    # sentences ~P(trigger co-masked) ~= 0.5; the contrast estimator cancels
    # that shared offset so only the trigger carries attribution mass.
    harness = AttributionHarness(
        adapter=_TriggerAdapter(trigger="Use the search tool."),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        sampler=RandomCoalitionSampler(n_coalitions=60, seed=11),
    )

    result = harness.explain("Use the search tool. Be concise. Answer in English.")
    by_name = {a.feature.name: a for a in result.attributions}

    # Every coalition masking the trigger scores 1.0 and every other scores 0.0,
    # so the driver's contrast is exactly 1.0 regardless of the random draw.
    assert by_name["sentence_1"].value == 1.0
    # Inert features lose the co-masking offset. Because the sampler skips the
    # all-masked/all-kept coalitions, their contrast lands at or below zero
    # rather than at ~0.5 — either way they carry no positive attribution mass.
    assert by_name["sentence_2"].value <= 0.0
    assert by_name["sentence_3"].value <= 0.0
    ranked = result.ranked()
    assert ranked[0][0].feature.name == "sentence_1"
    assert ranked[0][1] == 1.0  # the driver holds all of the positive share
