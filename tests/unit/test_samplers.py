import pytest

from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.samplers import LeaveOneOutSampler, RandomCoalitionSampler
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


def test_leave_one_out_estimate() -> None:
    assert LeaveOneOutSampler(repeats=3).estimate_evaluations(4) == 12


def test_random_sampler_emits_requested_count() -> None:
    sampler = RandomCoalitionSampler(n_coalitions=20, seed=7)

    coalitions = list(sampler.sample(4))

    assert len(coalitions) == 20
    assert sampler.estimate_evaluations(4) == 20


def test_random_sampler_skips_degenerate_coalitions() -> None:
    sampler = RandomCoalitionSampler(n_coalitions=50, seed=1)

    for coalition in sampler.sample(3):
        # Never the all-included (nothing masked) coalition; never all-excluded
        # when more than one feature exists.
        assert not all(coalition)
        assert any(coalition)


def test_random_sampler_is_reproducible_with_seed() -> None:
    first = list(RandomCoalitionSampler(n_coalitions=15, seed=42).sample(5))
    second = list(RandomCoalitionSampler(n_coalitions=15, seed=42).sample(5))

    assert first == second


def test_random_sampler_single_feature_allows_only_mask() -> None:
    coalitions = list(RandomCoalitionSampler(n_coalitions=5, seed=3).sample(1))

    assert coalitions == [(False,)] * 5


def test_random_sampler_validates_arguments() -> None:
    with pytest.raises(ValueError, match="n_coalitions"):
        RandomCoalitionSampler(n_coalitions=0)
    with pytest.raises(ValueError, match="inclusion_probability"):
        RandomCoalitionSampler(inclusion_probability=0.0)


def test_random_sampler_runs_in_harness() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        sampler=RandomCoalitionSampler(n_coalitions=12, seed=5),
    )

    result = harness.explain("Alpha sentence. Beta sentence. Gamma sentence.")

    assert len(result.attributions) == 3
    assert len(result.evaluations) == 12
