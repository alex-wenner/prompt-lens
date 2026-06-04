import math

from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.core.harness import _sampler_from_scale
from promptlens.samplers import LeaveOneOutSampler
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
