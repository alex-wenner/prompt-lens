# Optimize before/after: attribution-driven prompt rewrite

**Problem:** you have a verbose prompt and want a tighter version that keeps what
matters and drops the filler.

`AttributionHarness.optimize` runs a leave-one-out attribution sweep, then hands
that evidence to an `LLMPromptOptimizer`, which proposes a whole-prompt rewrite
for review. The rewrite is **never** adopted automatically.

## Run it

```bash
python examples/optimize_before_after/run.py
```

This calls a **real provider** — the same model answers the attribution sweep
and proposes the rewrite. Export `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` plus
`PROMPTLENS_EXAMPLE_PROVIDER=anthropic`) to choose the model;
`PROMPTLENS_EXAMPLE_MODEL` overrides the default model.

## Example output

The model proposes its own rewrite, so wording varies, but the shape is:

```text
                              promptlens Optimization
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field           ┃ Value                                                                   ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ original prompt │ You are an extremely helpful, friendly, and knowledgeable assistant.   │
│                 │ Summarize the input text in exactly three bullet points. Feel free to  │
│                 │ be as detailed and thorough as you possibly can. Thanks so much for    │
│                 │ your hard work on this important task.                                 │
├─────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ proposed prompt │ Summarize the input text in exactly three bullet points.               │
├─────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ rationale       │ Kept the only load-bearing instruction (the three-bullet constraint)   │
│                 │ and pruned the inert pleasantries that carried no attribution.         │
└─────────────────┴─────────────────────────────────────────────────────────────────────────┘
```

## Running it from the CLI

```bash
promptlens optimize \
  --prompt ./prompt.md \
  --provider openai --model gpt-4o-mini
```

From the SDK, pass an `LLMPromptOptimizer` to
`AttributionHarness(..., optimizer=...)` and call `harness.optimize(prompt)`.

## Lens, not oracle

A proposed rewrite is a candidate, not a verified improvement. Length and
embedding drift scores can hide precision-critical edits (flipped numbers,
negations, broken JSON). Re-run attribution and a task-level metric before
adopting any rewrite — the result metadata carries the same caveat.
