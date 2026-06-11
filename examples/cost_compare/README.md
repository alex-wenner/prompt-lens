# Cost compare: price the run before you spend

**Problem:** attribution multiplies provider calls by feature count, so the same
experiment costs very different amounts depending on the model — and you want to
know that *before* you spend, not after.

promptlens never guesses token counts with a tokenizer or character heuristic.
This example runs the **baseline completion once for real**, reads the
provider-reported input/output token usage, and multiplies it by the number of
perturbation evaluations. It pins the `estimate` path with **`compare_models`**
(one measured baseline, many price tables) and shows how the **perturbation
scale** multiplies the bill.

## Run it

```bash
export OPENAI_API_KEY=sk-...
python examples/cost_compare/run.py
```

One real provider call is made (the baseline); the `full`-scale estimate reuses
the same measured baseline, so scale comparisons are free.

## Example output

```text
╭──────────────────── Baseline output (the one real call) ────────────────────╮
│ The brief proposes consolidating three regional fulfilment centres into …   │
╰──────────────────────── usage: 500 in / 120 out ─────────────────────────────╯

Projected attribution cost across providers (from the measured baseline):

 Estimated cost (projected from the measured baseline call)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric                             ┃          Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ model                              │ openai/gpt-5.4 │
│ features                           │             12 │
│ evaluations                        │             12 │
│ input tokens                       │          6,500 │
│ output tokens                      │          1,560 │
│ input cost                         │      $0.016250 │
│ output cost                        │      $0.023400 │
│ total                              │      $0.039650 │
│ compare anthropic/claude-opus-4-8  │      $0.071500 │
│ compare openai/gpt-5.4             │      $0.039650 │
│ compare openai/gpt-5.4-mini        │      $0.011895 │
│ compare anthropic/claude-haiku-4-5 │      $0.014300 │
│ compare ollama/llama3.2            │      $0.000000 │
└────────────────────────────────────┴────────────────┘

At 'quick' scale that is 12 evaluations; 'full' scale runs 60 and costs
$0.186050 on openai/gpt-5.4.
The same run on a local model (ollama/llama3.2) is $0.00 — the cost case for
local inference in one line.
```

This is the same machinery behind `promptlens estimate` and the cost gate that
every `promptlens explain` run shows before its sweep: run the baseline, show
the projected spend, ask before continuing (skip the prompt with `--yes`,
or stop after the baseline with `--dry-run`).

```bash
promptlens estimate --prompt examples/cost_compare/prompt.md \
  --provider openai --compare anthropic/claude-opus-4-8,ollama/llama3.2
```
