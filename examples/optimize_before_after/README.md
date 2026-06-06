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

No API keys required — it uses a deterministic simulated adapter that echoes
during the attribution sweep and returns a scripted rewrite when it receives the
optimizer's brief.

## What you should see

| Field           | Value                                                            |
| --------------- | ---------------------------------------------------------------- |
| original prompt | "You are an extremely helpful… Summarize… in exactly three bullet points. Feel free… Thanks so much…" |
| proposed prompt | "Summarize the input text in exactly three bullet points."        |
| rationale       | Kept the load-bearing three-bullet constraint; pruned filler.    |

## Running it against a real model

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
