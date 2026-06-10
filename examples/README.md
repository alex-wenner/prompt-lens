# promptlens examples

Self-contained walkthroughs of the core promptlens workflow: **observe which
parts of a prompt move the model, then act on it.** By default each runs
against a **real provider**, and falls back to a small deterministic simulated
adapter when no API key is set — so they need no credentials to try, and double
as smoke tests in CI.

Start with the flagship: a production-sized instruction set tied to real
business objects and work process, attributed coarse-to-fine.

| Example | Question it answers | Demonstrates |
| ------- | ------------------- | ------------ |
| [`order_operations_agent/`](order_operations_agent/) ⭐ | Which sentences of my 8-section ops SOP drive the agent's trajectory? | `explain_drilldown`, `tool-args` scorer, `CompositeScorer`, section segmentation |
| [`tool_routing_bug/`](tool_routing_bug/) | Which instruction makes my agent call the wrong tool? | `tool-accuracy` (objective scorer), `@tool` definitions |
| [`system_prompt_cleanup/`](system_prompt_cleanup/) | Which lines of my system prompt actually matter? | `length` drift, sentence segmentation |
| [`optimize_before_after/`](optimize_before_after/) | Can attribution drive a concrete prompt rewrite? | `LLMPromptOptimizer` |

## Every attribution calculator, mapped

| Calculator | Try it |
| ---------- | ------ |
| `length` drift | `system_prompt_cleanup/`, `order_operations_agent/` (composite component) |
| `embedding` (semantic, provider-backed) | CLI snippet below |
| `embedding-local` (offline smoke only) | `promptlens explain --scorer embedding-local` |
| `logprob` | `promptlens explain --provider openai --model gpt-4o --scorer logprob` |
| `tool-accuracy` (objective) | `tool_routing_bug/` |
| `tool-sequence` drift | `promptlens explain --scorer tool-sequence` over any agent run |
| `tool-args` drift (weighted params) | `order_operations_agent/` |
| `CompositeScorer` | `order_operations_agent/` |
| Leave-one-out sampler | every example (the default) |
| Random-coalition / Banzhaf sampler | `promptlens explain --sampler random --scale standard` |
| Drill-down (coarse → fine) | `order_operations_agent/`, or `promptlens explain --drilldown` |

## Run them

```bash
python examples/order_operations_agent/run.py
python examples/tool_routing_bug/run.py
python examples/system_prompt_cleanup/run.py
python examples/optimize_before_after/run.py
```

Prefer an interactive tour? `promptlens wizard` walks through every choice with
explanations and prints the equivalent shell command when it finishes.

Each script exposes a `main(adapter=...)` that returns its headline numbers, so
the test suite can pin the offline-fallback behavior (see
`tests/unit/test_examples.py`).

## Choosing a provider

The examples call a real model whenever a credential is available, selected via
environment variables (see [`_realprovider.py`](_realprovider.py)):

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `PROMPTLENS_EXAMPLE_PROVIDER` | `openai` or `anthropic` | `openai` |
| `PROMPTLENS_EXAMPLE_MODEL` | Model id override | provider default |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Credential that enables the real call | — |

```bash
export OPENAI_API_KEY=sk-...
python examples/tool_routing_bug/run.py            # real OpenAI model

PROMPTLENS_EXAMPLE_PROVIDER=anthropic \
  ANTHROPIC_API_KEY=... python examples/tool_routing_bug/run.py
```

With no credential set the examples print a notice and use the offline simulated
adapter so they still run end-to-end.

## Semantic scoring

The text-drift examples score output-length drift, which works with any adapter.
To run the same workflow with a real semantic scorer, use the CLI with the
provider-backed `embedding` scorer:

```bash
promptlens explain \
  --prompt examples/system_prompt_cleanup/system_prompt.txt \
  --provider openai --model gpt-4o-mini \
  --scorer embedding \
  --scorer-config examples/system_prompt_cleanup/embedding.json
```

The `embedding` scorer is provider-backed and semantic; the offline
`embedding-local` scorer is a deterministic text-shape fallback for smoke tests
only.

## Lens, not oracle

Every example ends with the same reminder: attribution shows you *where* to look,
it does not certify truth. Confirm any attribution-driven edit with a task-level
metric before shipping it.
