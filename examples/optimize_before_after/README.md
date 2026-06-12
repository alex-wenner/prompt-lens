# Optimize before/after: attribution-driven prompt rewrite

**Problem:** you have a verbose prompt and want a tighter version that keeps what
matters and drops the filler.

`AttributionHarness.optimize` runs a leave-one-out attribution sweep, then hands
that evidence to an `LLMPromptOptimizer`, which proposes a whole-prompt rewrite
for review. The rewrite is **never** adopted automatically.

## Run it

```bash
OPENAI_API_KEY=sk-... python examples/optimize_before_after/run.py
```

This makes **real provider calls** — the same model answers the attribution
sweep and proposes the rewrite (one extra call). The default provider is
`openai`; pick another with `PROMPTLENS_EXAMPLE_PROVIDER` and override the
model with `PROMPTLENS_EXAMPLE_MODEL` (see [`_shared.py`](../_shared.py)).

## Example output

(output from a gpt-5.4-mini run; the rewrite and rationale are
model-generated and will vary)

```text
Provider: openai · model: gpt-5.4-mini
Attribution-informed prompt rewrite:

                       promptlens Optimization
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field           ┃ Value                                               ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ original prompt │ You are an extremely helpful, friendly, and         │
│                 │ knowledgeable assistant. Summarize the input text   │
│                 │ in exactly three bullet points. Feel free to be as  │
│                 │ detailed and thorough as you possibly can. Thanks   │
│                 │ so much for your hard work on this important task.  │
├─────────────────┼─────────────────────────────────────────────────────┤
│ proposed prompt │ Summarize the input text in exactly three bullet    │
│                 │ points.                                             │
├─────────────────┼─────────────────────────────────────────────────────┤
│ rationale       │ Attribution put 72% of the measured drift on the    │
│                 │ three-bullet constraint, so it is kept verbatim.    │
│                 │ The courtesy filler showed no measured effect, and  │
│                 │ "as detailed and thorough as you possibly can"      │
│                 │ conflicts with the three-bullet limit, so both      │
│                 │ were removed.                                       │
└─────────────────┴─────────────────────────────────────────────────────┘

Lens, not oracle: the proposed rewrite is a candidate, not a verified
improvement. Re-run attribution and a task metric before adopting it.
```

## Running it from the CLI

```bash
promptlens optimize \
  --prompt ./prompt.md \
  --provider openai --model gpt-5.4-mini
```

The CLI runs the baseline first, shows the projected sweep cost, and asks
before spending the remaining calls (`--yes` skips the question, `--dry-run`
stops after the baseline); the final rewrite adds one call on top of the
projection. From the SDK, pass an `LLMPromptOptimizer` to
`AttributionHarness(..., optimizer=...)` and call `harness.optimize(prompt)`.

## Lens, not oracle

A proposed rewrite is a candidate, not a verified improvement. Length and
embedding drift scores can hide precision-critical edits (flipped numbers,
negations, broken JSON). Re-run attribution and a task-level metric before
adopting any rewrite — the result metadata carries the same caveat.
