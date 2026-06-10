"""Planted-instruction benchmark: validate rankings against known ground truth.

Constructs synthetic prompts where ground truth is known *by construction*:
``n_drivers`` planted "directive" sentences each provably shape the model
output (the simulated model emits a fixed-size marker per directive that
survives masking), while the remaining sentences are inert filler. Attribution
should rank every driver above every filler.

Reported metrics, averaged over randomized trials (driver positions shuffle
per trial):

* ``precision@k`` — fraction of the top ``n_drivers`` ranked features that are
  true drivers (1.0 means the top of the ranking is exactly the planted set).
* ``pairwise accuracy`` — fraction of (driver, filler) pairs where the driver
  outranks the filler (an AUC-style measure over the whole ranking).

Runs fully offline (no provider calls, no keys):

    python benchmarks/planted_instructions.py

A finding worth knowing: under heavy output noise, leave-one-out degrades
because the baseline is sampled *once* and its noise contaminates every
drift measurement, while the random-coalition masked-vs-kept contrast cancels
that shared offset and stays at ceiling. If your provider is noisy and repeats
don't stabilize leave-one-out rankings, switch to the random sampler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, Sampler, ToolDefinitions
from promptlens.samplers import LeaveOneOutSampler, RandomCoalitionSampler
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

_FILLER_SENTENCES = [
    "Please remember to be friendly at all times.",
    "The weather has no bearing on this task.",
    "Our company was founded a long time ago.",
    "Customers come from many different regions.",
    "This paragraph exists for historical reasons.",
    "Nothing in this sentence changes the output.",
    "Style guides are reviewed once per quarter.",
    "The mascot is a small cartoon telescope.",
]

_MARKER_WIDTH = 40


class PlantedAdapter(Adapter):
    """Simulated model: emits one fixed-size marker per surviving directive.

    ``noise`` appends a random-length suffix (up to ``noise * marker width``
    characters) to mimic a non-deterministic provider; pair it with repeats so
    the benchmark exercises the averaging/stderr machinery too.
    """

    def __init__(self, directives: list[str], noise: float = 0.0, seed: int = 0) -> None:
        self.model = "planted"
        self.directives = directives
        self.noise = noise
        self._rng = random.Random(seed)

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        parts = [
            f"<{index}:{'M' * _MARKER_WIDTH}>"
            for index, directive in enumerate(self.directives)
            if directive in prompt
        ]
        text = "".join(parts)
        if self.noise:
            text += "n" * self._rng.randint(0, int(_MARKER_WIDTH * self.noise))
        return CompletionOutput(text=text)


@dataclass
class TrialResult:
    precision_at_k: float
    pairwise_accuracy: float


def run_trial(
    *,
    sampler: Sampler,
    n_features: int,
    n_drivers: int,
    noise: float,
    samples_per_coalition: int,
    seed: int,
) -> TrialResult:
    rng = random.Random(seed)
    directives = [
        f"Directive {index}: include marker {index} in the answer." for index in range(n_drivers)
    ]
    fillers = [_FILLER_SENTENCES[i % len(_FILLER_SENTENCES)] for i in range(n_features - n_drivers)]
    sentences = directives + fillers
    rng.shuffle(sentences)
    prompt = " ".join(sentences)
    driver_positions = {
        f"sentence_{position + 1}"
        for position, sentence in enumerate(sentences)
        if sentence in directives
    }
    harness = AttributionHarness(
        adapter=PlantedAdapter(directives, noise=noise, seed=seed),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        sampler=sampler,
        samples_per_coalition=samples_per_coalition,
    )
    ranked = [attribution.feature.name for attribution, _ in harness.explain(prompt).ranked()]
    top_k = ranked[:n_drivers]
    precision = sum(1 for name in top_k if name in driver_positions) / n_drivers
    rank_of = {name: position for position, name in enumerate(ranked)}
    pairs = correct = 0
    for driver in driver_positions:
        for name in ranked:
            if name in driver_positions:
                continue
            pairs += 1
            correct += rank_of[driver] < rank_of[name]
    return TrialResult(precision, correct / pairs if pairs else 1.0)


def run_benchmark(
    *,
    n_features: int = 12,
    n_drivers: int = 3,
    trials: int = 20,
) -> dict[str, TrialResult]:
    """Run every configuration and return mean metrics keyed by config name."""
    configs: list[tuple[str, dict[str, object]]] = [
        (
            "leave-one-out (clean)",
            {"noise": 0.0, "samples_per_coalition": 1, "loo": True},
        ),
        (
            "leave-one-out (noisy, 1 sample)",
            {"noise": 2.0, "samples_per_coalition": 1, "loo": True},
        ),
        (
            "leave-one-out (noisy, 5 samples)",
            {"noise": 2.0, "samples_per_coalition": 5, "loo": True},
        ),
        ("random coalitions x64 (clean)", {"noise": 0.0, "samples_per_coalition": 1, "loo": False}),
        ("random coalitions x64 (noisy)", {"noise": 2.0, "samples_per_coalition": 1, "loo": False}),
    ]
    summary: dict[str, TrialResult] = {}
    for name, config in configs:
        results = []
        for trial in range(trials):
            sampler: Sampler = (
                LeaveOneOutSampler()
                if config["loo"]
                else RandomCoalitionSampler(n_coalitions=64, seed=trial)
            )
            results.append(
                run_trial(
                    sampler=sampler,
                    n_features=n_features,
                    n_drivers=n_drivers,
                    noise=float(config["noise"]),  # type: ignore[arg-type]
                    samples_per_coalition=int(config["samples_per_coalition"]),  # type: ignore[call-overload]
                    seed=trial,
                )
            )
        summary[name] = TrialResult(
            precision_at_k=sum(r.precision_at_k for r in results) / trials,
            pairwise_accuracy=sum(r.pairwise_accuracy for r in results) / trials,
        )
    return summary


def main() -> None:
    summary = run_benchmark()
    width = max(len(name) for name in summary)
    print(f"{'configuration'.ljust(width)}  precision@k  pairwise accuracy")
    for name, result in summary.items():
        accuracy = f"{result.pairwise_accuracy:>17.3f}"
        print(f"{name.ljust(width)}  {result.precision_at_k:>11.3f}  {accuracy}")
    clean = summary["leave-one-out (clean)"]
    if clean.precision_at_k < 1.0:
        msg = "sanity gate failed: clean leave-one-out should recover all planted drivers"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
