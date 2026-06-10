import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "planted_instructions",
    Path(__file__).resolve().parents[2] / "benchmarks" / "planted_instructions.py",
)
assert _SPEC is not None and _SPEC.loader is not None
planted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = planted
_SPEC.loader.exec_module(planted)


def test_clean_leave_one_out_recovers_all_planted_drivers() -> None:
    from promptlens.samplers import LeaveOneOutSampler

    result = planted.run_trial(
        sampler=LeaveOneOutSampler(),
        n_features=8,
        n_drivers=2,
        noise=0.0,
        samples_per_coalition=1,
        seed=3,
    )

    assert result.precision_at_k == 1.0
    assert result.pairwise_accuracy == 1.0


def test_clean_random_coalitions_recover_all_planted_drivers() -> None:
    from promptlens.samplers import RandomCoalitionSampler

    result = planted.run_trial(
        sampler=RandomCoalitionSampler(n_coalitions=64, seed=3),
        n_features=8,
        n_drivers=2,
        noise=0.0,
        samples_per_coalition=1,
        seed=3,
    )

    assert result.precision_at_k == 1.0
    assert result.pairwise_accuracy == 1.0
