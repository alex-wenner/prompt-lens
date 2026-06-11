# promptlens examples

Self-contained walkthroughs of the core promptlens workflow: **observe which
parts of a prompt move the model, then act on it.** Every example makes **real
LLM calls** — there are no simulated models. Export an API key
(`OPENAI_API_KEY` is the default; see "Choosing a provider" below) before
running them. Each README includes example output so you can see the results of
every mode in detail without running anything.

Start with the flagship: a production-sized instruction set tied to real
business objects and work process, attributed coarse-to-fine.

| Example | Question it answers | Provider angle | Config it pins |
| ------- | ------------------- | -------------- | -------------- |
| [`order_operations_agent/`](order_operations_agent/) ⭐ | Which sentences of my 8-section ops SOP drive the agent's trajectory? | provider-neutral | drill-down, `tool-args` + `length` via `CompositeScorer`, section segmenter |
| [`local_inference/`](local_inference/) | Can I run the whole loop — and its synopsis — on a local model? | **ollama** (local, $0) | `DropMasker`, paragraph segmenter, `LLMSynopsisWriter` |
| [`interaction_effects/`](interaction_effects/) | Why does leave-one-out call two real instructions "dead weight"? | provider-neutral | `RandomCoalitionSampler` (Banzhaf) vs leave-one-out, `FillerMasker` |
| [`cost_compare/`](cost_compare/) | What will this run cost on each provider before I spend? | **many** (price tables) | measured-baseline `estimate` + `compare_models`, perturbation scale |
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
export OPENAI_API_KEY=sk-...

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

Each script exposes a `main(adapter=...)` that returns its headline numbers;
the test suite injects deterministic stub adapters through that parameter (see
`tests/unit/test_examples.py`), so CI never makes network calls — the examples
themselves always do.

## Choosing a provider

Every example shares the same provider selection (see [`_shared.py`](_shared.py)).
Examples default to `openai`; pick any other provider with
`PROMPTLENS_EXAMPLE_PROVIDER`. When the chosen provider's credential is missing
the example exits immediately with instructions — it never silently falls back
to a fake model.

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `PROMPTLENS_EXAMPLE_PROVIDER` | `openai`, `anthropic`, `gemini`, `grok`, `bedrock`, `copilot`, `ollama`, `openai-compatible` | each example's own preference (usually `openai`) |
| `PROMPTLENS_EXAMPLE_MODEL` | Model id override | provider default |

Credential per provider: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `XAI_API_KEY`, AWS credentials, `GITHUB_COPILOT_TOKEN`. The
keyless local providers (`ollama`, `openai-compatible`) only need their server
running.

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
