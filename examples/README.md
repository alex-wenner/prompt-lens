# promptlens examples

Self-contained walkthroughs of the core promptlens workflow: **observe which
parts of a prompt move the model, then act on it.** Each runs against a **real
provider** when one is configured and falls back to a small deterministic
simulated adapter otherwise — so they need no credentials to try, and double as
smoke tests in CI.

Start with the flagship: a production-sized instruction set tied to real
business objects and work process, attributed coarse-to-fine.

| Example | Question it answers | Provider angle | Config it pins |
| ------- | ------------------- | -------------- | -------------- |
| [`order_operations_agent/`](order_operations_agent/) ⭐ | Which sentences of my 8-section ops SOP drive the agent's trajectory? | provider-neutral | drill-down, `tool-args` + `length` via `CompositeScorer`, section segmenter |
| [`local_inference/`](local_inference/) | Can I run the whole loop — and its synopsis — on a local model? | **ollama** (local, $0) | `DropMasker`, paragraph segmenter, `LLMSynopsisWriter` |
| [`interaction_effects/`](interaction_effects/) | Why does leave-one-out call two real instructions "dead weight"? | offline (estimator lesson) | `RandomCoalitionSampler` (Banzhaf) vs leave-one-out, `FillerMasker` |
| [`cost_compare/`](cost_compare/) | What will this run cost on each provider before I spend? | **many** (price tables) | `estimate` + `compare_models`, perturbation scale |
| [`tool_routing_bug/`](tool_routing_bug/) | Which instruction makes my agent call the wrong tool? | provider-neutral | `tool-accuracy` (objective), `@tool` defs |
| [`system_prompt_cleanup/`](system_prompt_cleanup/) | Which lines of my system prompt actually matter? | provider-neutral | `length` drift, sentence segmenter |
| [`optimize_before_after/`](optimize_before_after/) | Can attribution drive a concrete prompt rewrite? | provider-neutral | `LLMPromptOptimizer` |

## Every attribution calculator, mapped to an example

| Calculator | Where to see it |
| ---------- | --------------- |
| `length` drift | `system_prompt_cleanup/`, `interaction_effects/`, `cost_compare/` |
| `embedding` (semantic, provider-backed) | CLI snippet under "Semantic scoring" below |
| `embedding-local` (offline smoke only) | `promptlens explain --scorer embedding-local` |
| `logprob` | `promptlens explain --provider openai --model gpt-4o --scorer logprob` |
| `tool-accuracy` (objective) | `tool_routing_bug/` |
| `tool-sequence` drift | `promptlens explain --scorer tool-sequence` over any agent run |
| `tool-args` drift (weighted params) | `order_operations_agent/` |
| `CompositeScorer` | `order_operations_agent/` |
| Leave-one-out sampler | every example except where noted (the default) |
| Random-coalition / Banzhaf sampler | `interaction_effects/` |
| Drill-down (coarse → fine) | `order_operations_agent/`, or `promptlens explain --drilldown` |
| Synopsis (`LLMSynopsisWriter`) | `local_inference/`, or `promptlens explain --synopsis` |

## Run them

```bash
python examples/order_operations_agent/run.py
python examples/local_inference/run.py
python examples/interaction_effects/run.py
python examples/cost_compare/run.py
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

Every example shares the same provider selection (see [`_shared.py`](_shared.py)).
A provider only goes live when its **activation signal** is present, so nothing
makes a surprise network call; otherwise the example uses its offline simulated
adapter and prints a notice telling you how to run it for real.

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `PROMPTLENS_EXAMPLE_PROVIDER` | `openai`, `anthropic`, `gemini`, `grok`, `bedrock`, `copilot`, `ollama`, `openai-compatible` | each example's own preference (usually `openai`) |
| `PROMPTLENS_EXAMPLE_MODEL` | Model id override | provider default |

Activation signal per provider: the API-key / credential env var for keyed
providers (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`XAI_API_KEY`, AWS credentials, `GITHUB_COPILOT_TOKEN`), and an explicit
`PROMPTLENS_EXAMPLE_PROVIDER` opt-in for the keyless local providers (`ollama`,
`openai-compatible`).

```bash
# Hosted providers (keyed)
OPENAI_API_KEY=sk-... python examples/tool_routing_bug/run.py
PROMPTLENS_EXAMPLE_PROVIDER=anthropic ANTHROPIC_API_KEY=... \
  python examples/system_prompt_cleanup/run.py
PROMPTLENS_EXAMPLE_PROVIDER=gemini GEMINI_API_KEY=... \
  python examples/order_operations_agent/run.py

# Local, no key — needs a running Ollama server
PROMPTLENS_EXAMPLE_PROVIDER=ollama python examples/local_inference/run.py
```

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
