# Cost compare: price the run before you spend

**Problem:** attribution multiplies provider calls by feature count, so the same
experiment costs very different amounts depending on the model — and you want to
know that *before* you spend, not after. This example estimates the spend for
one RAG-assistant prompt across a frontier model, a mid-tier model, a cheap
model, and a free local model, entirely offline.

It pins the `estimate` path with **`compare_models`** (count tokens once, apply
many price tables) and shows how the **perturbation scale** multiplies the bill.
No provider calls, no credentials.

## Run it

```bash
python examples/cost_compare/run.py
```

## What you should see

A `CostEstimate` table with one row per comparison model — the frontier model at
the top, the cheap and local models far below, `ollama/llama3.2` at `$0.00`.
Then a line showing how `full` scale (five sweeps) multiplies the `quick`
estimate.

This is the same machinery behind `promptlens estimate` and the `--dry-run` /
`--confirm` flags on `explain`, and behind the README's "don't accidentally
expense a tiny yacht" promise.

## Going exact

The built-in estimate uses a conservative character heuristic (and `tiktoken`
for OpenAI-family models). For an exact, still inference-free Anthropic count,
add `--exact-tokens` on the CLI:

```bash
promptlens estimate --prompt examples/cost_compare/prompt.md \
  --model anthropic/claude-opus-4-8
```
