"""Attribution sweep over a code-review system prompt using GitHub Copilot.

Reads ``prompt.md`` from this directory, runs a leave-one-out attribution sweep
against the Copilot SDK, and prints each sentence ranked by how much the model
response changes when it is removed — alongside the actual model response for
that masked run.

Scoring uses local semantic embeddings (``sentence-transformers``) when
available, otherwise falls back to length drift.

Setup
-----
::

    pip install -e '.[copilot,hf]'
    export GITHUB_COPILOT_TOKEN="<your-token>"  # or rely on gh auth login
    python examples/real_world/run.py

Environment variables
---------------------
* ``COPILOT_MODEL``       — model name (default: ``claude-haiku-4.5``)
* ``ATTRIBUTION_SCALE``   — repeats per coalition, 1 = fastest (default: 1)
* ``MAX_PARALLEL``        — concurrent Copilot sessions (default: 8)
* ``HF_EMBEDDING_MODEL``  — sentence-transformers model (default: all-MiniLM-L6-v2)
* ``REAL_WORLD_USER_MESSAGE`` — code snippet sent as the user turn
"""

from __future__ import annotations

import os
import textwrap
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from promptlens import AttributionHarness
from promptlens.adapters import CopilotAdapter
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.scorers import EmbeddingScorer, LengthDriftScorer
from promptlens.scorers.embeddings import HuggingFaceEmbeddingClient
from promptlens.segmenters import SentenceSegmenter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

MODEL = os.environ.get("COPILOT_MODEL") or os.environ.get("GITHUB_COPILOT_MODEL") or "claude-haiku-4.5"
ATTRIBUTION_SCALE = int(os.environ.get("ATTRIBUTION_SCALE", "1"))
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "8"))
HF_EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

DEFAULT_USER_MESSAGE = """\
Please review the following Python function:

def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return db.execute(query)
"""

USER_MESSAGE = os.environ.get("REAL_WORLD_USER_MESSAGE") or DEFAULT_USER_MESSAGE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_scorer() -> tuple[object, str]:
    """Return (scorer, label). Uses local HF embeddings when sentence-transformers
    is installed, otherwise falls back to length drift."""
    try:
        client = HuggingFaceEmbeddingClient(HF_EMBEDDING_MODEL)
        client._get_encoder()  # probe — raises if not installed
        return EmbeddingScorer(client), f"embedding cosine distance ({HF_EMBEDDING_MODEL})"
    except RuntimeError:
        return LengthDriftScorer(), "length drift (install sentence-transformers for semantic scoring)"


def _masked_outputs(result: object) -> dict[str, list[str]]:
    """Map feature name -> model responses when that feature was masked out."""
    from collections import defaultdict
    mapping: dict[str, list[str]] = defaultdict(list)
    for ev in result.evaluations:  # type: ignore[attr-defined]
        masked = [i for i, present in enumerate(ev.coalition) if not present]
        for idx in masked:
            name = result.attributions[idx].feature.name  # type: ignore[attr-defined]
            mapping[name].append(ev.output.text)
    return dict(mapping)


def _wrap(text: str, width: int = 88) -> str:
    return textwrap.fill(text.replace("\n", " "), width=width, subsequent_indent="     ")


class _ProgressAdapter(Adapter):
    """Wraps an adapter to show a single overwriting progress bar during batches."""

    def __init__(self, inner: Adapter) -> None:
        self._inner = inner
        self.model = inner.model
        self._lock = threading.Lock()
        self._done = 0
        self._total: int | None = None

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        result = self._inner.complete(prompt, tools=tools)
        with self._lock:
            self._done += 1
            done, total = self._done, self._total
        if total:
            filled = int(done / total * 30)
            bar = "█" * filled + "░" * (30 - filled)
            end = "\n" if done == total else "\r"
            print(f"  [{bar}] {done}/{total}", end=end, flush=True)
        return result

    def complete_batch(
        self, prompts: Sequence[str], tools: ToolDefinitions | None = None
    ) -> list[CompletionOutput]:
        n = len(prompts)
        workers = min(getattr(self._inner, "max_parallel", MAX_PARALLEL), n)
        print(f"\nStep 2/2 — attribution sweep ({n} masked calls, {workers} parallel):")
        with self._lock:
            self._total = n
            self._done = 0
        outputs: list[CompletionOutput | None] = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.complete, p, tools): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                outputs[futures[fut]] = fut.result()
        self._total = None
        return outputs  # type: ignore[return-value]


def _build_scorer() -> tuple[object, str]:
    """Use local HF embeddings when sentence-transformers is installed, else length drift."""
    try:
        client = HuggingFaceEmbeddingClient(HF_EMBEDDING_MODEL)
        client._get_encoder()  # probe — raises RuntimeError if not installed
        return EmbeddingScorer(client), f"embedding cosine distance ({HF_EMBEDDING_MODEL})"
    except RuntimeError:
        return LengthDriftScorer(), "length drift (pip install sentence-transformers for semantic scoring)"


def _masked_outputs(result: object) -> dict[str, list[str]]:
    """Map feature name -> model responses when that feature was masked out."""
    from collections import defaultdict
    mapping: dict[str, list[str]] = defaultdict(list)
    for ev in result.evaluations:  # type: ignore[attr-defined]
        masked = [i for i, present in enumerate(ev.coalition) if not present]
        for idx in masked:
            name = result.attributions[idx].feature.name  # type: ignore[attr-defined]
            mapping[name].append(ev.output.text)
    return dict(mapping)


def _wrap(text: str, width: int = 88) -> str:
    return textwrap.fill(text.replace("\n", " "), width=width, subsequent_indent="     ")


def main() -> None:
    system_prompt = (_HERE / "prompt.md").read_text(encoding="utf-8").strip()
    scorer, scorer_label = _build_scorer()

    print(f"Loaded prompt ({len(system_prompt.split())} words) from prompt.md")
    print(f"Model   : {MODEL}")
    print(f"Scorer  : {scorer_label}")
    print(f"Scale   : {ATTRIBUTION_SCALE}x per sentence\n")

    adapter = CopilotAdapter(model=MODEL)
    tracked = _ProgressAdapter(adapter)

    harness = AttributionHarness(
        adapter=tracked,
        segmenter=SentenceSegmenter(),
        scorer=scorer,
        perturbation_scale=ATTRIBUTION_SCALE,
    )

    full_prompt = f"{system_prompt}\n\n{USER_MESSAGE}"

    print("Step 1/2 — baseline call:")
    result = harness.explain(full_prompt)

    # ── Baseline response ────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("BASELINE RESPONSE (all instructions present):")
    print("─" * 70)
    print(_wrap(result.baseline_output.text))

    # ── Ranked sentences with masked responses ───────────────────────────────
    masked = _masked_outputs(result)
    print(f"\n{'─' * 70}")
    print(f"SENTENCES ranked by attribution  [{scorer_label}]:")
    print("─" * 70)
    for rank, (feat, share) in enumerate(result.ranked(), 1):
        filled = max(1, round(share * 30))
        bar = "█" * filled + "░" * (30 - filled)
        print(f"\n  #{rank}  [{bar}] {share * 100:.1f}%  drift={feat.value:.3f}")
        print(f"  Prompt line : {feat.feature.text!r}")
        samples = masked.get(feat.feature.name, [])
        if samples:
            print(f"  Without it  : {_wrap(samples[0])}")

    print()
    adapter.close()


if __name__ == "__main__":
    main()
