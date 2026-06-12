# Cost compare: price the run before you spend

**Problem:** attribution multiplies provider calls by feature count, so the same
experiment costs very different amounts depending on the model — and you want to
know that *before* you spend, not after.

This example runs the baseline completion **once** on a real provider, reads
the provider's own metered token usage off that response, and projects the
sweep cost as `baseline usage × (evaluations + 1)`. One call, real numbers, no
tokenizer guesswork — the same flow the CLI's cost gate uses before every
`explain` run. It pins `estimate_from_baseline` with **`compare_models`** (one
metered baseline, many price tables) and shows how the **perturbation scale**
multiplies the bill (`quick` = 1 sweep, `full` = 5).

## Run it

```bash
OPENAI_API_KEY=sk-... python examples/cost_compare/run.py
```

This makes exactly **one real provider call** (the baseline). The default
provider is `openai`; pick another with `PROMPTLENS_EXAMPLE_PROVIDER` and
override the model with `PROMPTLENS_EXAMPLE_MODEL` (see
[`_shared.py`](../_shared.py)).

## Example output

(output from a gpt-5.4-mini run; your token counts will vary)

```text
Provider: openai · model: gpt-5.4-mini
Projected attribution cost from one real baseline call:

╭───────────────────── Projected sweep cost ──────────────────────╮
│  Model                              gpt-5.4-mini                │
│  Features to attribute              11                          │
│  Provider calls remaining           11                          │
│  Baseline usage (metered)           312 in / 184 out            │
│  Projected tokens                   3,744 in / 2,208 out        │
│  Input cost                         $0.0028                     │
│  Output cost                        $0.0099                     │
│  Projected total                    $0.0127                     │
│  … on anthropic/claude-opus-4-8     $0.0739                     │
│  … on openai/gpt-5.4                $0.0425                     │
│  … on anthropic/claude-haiku-4-5    $0.0148                     │
│  … on ollama/llama3.2               $0.0000                     │
╰───────── baseline-derived · pricing updated 2026-06-09 ─────────╯

At 'quick' scale that is 11 evaluations; 'full' scale runs 55 and projects to
$0.0595 on gpt-5.4-mini.
The same run priced on a local model (ollama/llama3.2) is $0.00 — the cost
case for local inference in one line.

Lens, not oracle: projections use the provider's real metered usage for this
baseline and promptlens' built-in pricing table; check live provider pricing
before budgeting.
```

## The same gate on the CLI

`promptlens explain` always runs the baseline first, shows this panel, and asks
**"Run the remaining N provider calls?"** before spending the rest. `--yes/-y`
skips the question; `--dry-run` stops after the baseline. `promptlens estimate`
does the baseline-and-project step on its own (it costs one real call):

```bash
promptlens estimate --prompt examples/cost_compare/prompt.md \
  --provider openai --model gpt-5.4-mini \
  --compare anthropic/claude-opus-4-8,ollama/llama3.2
```

`promptlens models` lists the built-in pricing table.
